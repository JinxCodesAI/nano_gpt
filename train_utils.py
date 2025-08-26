"""
Training utilities for diffusion training script.
Contains all reusable functions for data loading, corruption strategies, and model utilities.
"""

import os
import time
import math
import pickle
import threading
from queue import Queue
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils import Timer, log_masking_stats, apply_sticky_masking
from dataclasses import dataclass

# Global variables for data caching and prefetching
_val_batch_cache = None
_progressive_val_cache = {}  # Cache for progressive validation batches
_progressive_val_full_cache = None  # Cache for full progressive validation set (all 320 samples)
_data_cache = {'train': None, 'val': None}
_valid_indices_cache = {'train': None, 'val': None}
_prefetch_enabled = True
_prefetch_queue = Queue(maxsize=2)
_prefetch_thread = None
_prefetch_active = False

# Global synthetic model for remasking
synthetic_model = None

@dataclass
class TrainingContext:
    """Configuration class for training parameters to avoid long parameter lists"""
    # Training configuration
    training_type: str = 'remasking'
    batch_size: int = 16
    block_size: int = 1024
    max_iters: int = 12000  # Maximum training iterations
    
    # Device configuration
    device: str = 'cuda'
    device_type: str = 'cuda'
    seed_offset: int = 0
    
    # Data configuration
    data_dir: str = 'data/shakespeare_char'
    meta_vocab_size: int = None
    
    # Token IDs
    mask_token_id: int = None
    wrong_token_id: int = None
    remask_good_id: int = None
    remask_wrong_id: int = None
    extended_vocab_size: int = None
    
    # Training iteration
    iter_num: int = 0
    
    # Masking parameters
    guaranteed_unmasked_max: float = 0.8  # Maximum guaranteed fraction (at start of training)
    guaranteed_unmasked_min: float = 0.0  # Minimum guaranteed fraction (at end of training)
    random_mask_warmup: int = 8000  # Iterations over which guaranteed_unmasked transitions from max to min
    noise_max_ratio: float = 0.05
    sticky_rounds: int = 10
    sticky_p1_p2_multiplier: float = 10.0
    sticky_p1_divisor: float = 2.0
    sticky_transition_start: int = 500
    sticky_transition_end: int = 12000
    
    # Remasking strategy
    remasking_corruption_strategy: str = 'mixed'
    remasking_strategy_weights: list = None
    
    # Evaluation parameters
    eval_iters: int = 20
    
    # Learning rate parameters
    warmup_iters: int = 2000
    lr_decay_iters: int = 8000
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    
    def __post_init__(self):
        if self.remasking_strategy_weights is None:
            self.remasking_strategy_weights = [0.25, 0.4, 0.25, 0.1]
    
    def get_guaranteed_unmasked(self, iter_num: int = None) -> float:
        """Calculate guaranteed_unmasked for given iteration using centralized logic"""
        if iter_num is None:
            iter_num = self.iter_num
        
        # Linear transition from max to min over random_mask_warmup iterations, then stay at min
        if iter_num >= self.random_mask_warmup:
            return self.guaranteed_unmasked_min
        
        progress = iter_num / self.random_mask_warmup if self.random_mask_warmup > 0 else 1.0
        return self.guaranteed_unmasked_max + progress * (self.guaranteed_unmasked_min - self.guaranteed_unmasked_max)

def find_double_newline_indices(data, meta_vocab_size, block_size):
    """Find all valid starting indices that begin with double newlines (\\n\\n)"""
    # Get the token IDs for newlines
    if meta_vocab_size is not None:
        # For Shakespeare character-level data, newline is token 0
        newline_id = 0
    else:
        # For GPT-2 style tokenization, this would be different
        newline_id = 198  # GPT-2 newline token
    
    # Find positions where we have \\n\\n (two consecutive newlines)
    valid_indices = []
    for i in range(len(data) - block_size - 1):  # -block_size-1 to ensure we can extract full block
        if i >= 1 and data[i] == newline_id and data[i+1] == newline_id:
            valid_indices.append(i)
    
    return np.array(valid_indices)

def _prepare_batch_data_only(split, ctx: TrainingContext):
    """Background function to prepare raw batch data (CPU only)"""
    global _data_cache, _valid_indices_cache
    
    # Ensure data is cached
    if _data_cache[split] is None:
        return None
        
    data = _data_cache[split]
    valid_indices = _valid_indices_cache[split]
    
    # Fast index sampling - all on CPU
    if len(valid_indices) == 0:
        ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
    else:
        ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
        ix_np = valid_indices[ix_indices]

    # VECTORIZED DATA LOADING - Load entire batch at once
    x_np = np.zeros((ctx.batch_size, ctx.block_size), dtype=np.int64)
    for i, start_idx in enumerate(ix_np):
        x_np[i] = data[start_idx:start_idx+ctx.block_size].astype(np.int64)
    
    return x_np

def _prefetch_worker(ctx: TrainingContext):
    """Background thread worker for data prefetching"""
    global _prefetch_active
    while _prefetch_active:
        try:
            # Prepare next batch in background
            x_np = _prepare_batch_data_only('train', ctx)
            if x_np is not None:
                _prefetch_queue.put(x_np, timeout=1.0)
        except:
            # Queue full or other error, just continue
            time.sleep(0.001)

def start_prefetch(ctx: TrainingContext):
    """Start background data prefetching"""
    global _prefetch_thread, _prefetch_active
    if _prefetch_thread is None and _prefetch_enabled:
        _prefetch_active = True
        _prefetch_thread = threading.Thread(target=lambda: _prefetch_worker(ctx), daemon=True)
        _prefetch_thread.start()

def stop_prefetch():
    """Stop background data prefetching"""
    global _prefetch_thread, _prefetch_active
    _prefetch_active = False
    if _prefetch_thread is not None:
        _prefetch_thread.join(timeout=1.0)
        _prefetch_thread = None

def clear_validation_cache():
    """Clear progressive validation cache - useful when training parameters change"""
    global _progressive_val_cache, _val_batch_cache, _progressive_val_full_cache
    _progressive_val_cache.clear()
    _val_batch_cache = None
    _progressive_val_full_cache = None

def create_progressive_validation_set(training_ctx: TrainingContext):
    """Create the full progressive validation set once - 320 samples representing training progression"""
    global _progressive_val_full_cache
    
    if _progressive_val_full_cache is not None:
        return _progressive_val_full_cache
    
    print(f"\n=== CREATING PROGRESSIVE VALIDATION SET ===")
    total_samples = training_ctx.eval_iters * training_ctx.batch_size
    print(f"Generating {total_samples} validation samples ({training_ctx.eval_iters} batches × {training_ctx.batch_size} batch_size)")
    print(f"Each sample will represent different training difficulty from 0 to {training_ctx.max_iters} iterations")
    
    # Calculate which training iterations each sample represents
    progressive_iterations = get_progressive_validation_iterations(training_ctx.eval_iters, training_ctx.max_iters)
    print(f"Progressive iterations: {progressive_iterations}")
    
    # Generate all samples at once
    all_X, all_Y, all_masks = [], [], []
    
    for sample_idx in range(total_samples):
        # Map sample index to training iteration
        iter_idx = sample_idx % len(progressive_iterations)
        represented_iter = progressive_iterations[iter_idx]
        
        if sample_idx < 10 or sample_idx >= total_samples - 5:  # Show first 10 and last 5
            print(f"  Sample {sample_idx:3d}: iteration {represented_iter:5d} (guaranteed_unmasked: {training_ctx.get_guaranteed_unmasked(represented_iter):.3f})")
        elif sample_idx == 10:
            print(f"  ... (showing first 10 and last 5 samples) ...")
        
        # Generate single validation sample for this iteration
        X_single, Y_single, mask_single = get_batch('val', training_ctx, validation_sample_idx=iter_idx)
        all_X.append(X_single[0:1])  # Take first sample only
        all_Y.append(Y_single[0:1])
        all_masks.append(mask_single[0:1])
    
    # Combine all samples
    full_X = torch.cat(all_X, dim=0)  # Shape: (total_samples, block_size)
    full_Y = torch.cat(all_Y, dim=0)  # Shape: (total_samples, block_size)
    full_masks = torch.cat(all_masks, dim=0)  # Shape: (total_samples, block_size)
    
    _progressive_val_full_cache = (full_X, full_Y, full_masks)
    
    print(f"Progressive validation set created: {full_X.shape[0]} samples, each with {full_X.shape[1]} tokens")
    print(f"Difficulty range: iteration 0 (easy) to {training_ctx.max_iters} (hard)")
    print(f"=== PROGRESSIVE VALIDATION SET COMPLETE ===\n")
    
    return _progressive_val_full_cache

def apply_random_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min,
                               sticky_transition_start, sticky_transition_end, meta_vocab_size, random_mask_warmup=None):
    """GPU-optimized Strategy 1: Random token corruption with dynamic guaranteed_unmasked"""
    device = x.device
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257

    # Calculate dynamic guaranteed_unmasked using new warmup logic
    # Note: This function should receive a TrainingContext, but for backward compatibility
    # we'll create a temporary one. TODO: Refactor callers to pass TrainingContext directly
    if random_mask_warmup is None:
        random_mask_warmup = sticky_transition_end  # Fallback to old behavior
    
    if iter_num >= random_mask_warmup:
        guaranteed_unmasked = guaranteed_unmasked_min
    else:
        progress = iter_num / random_mask_warmup if random_mask_warmup > 0 else 1.0
        guaranteed_unmasked = guaranteed_unmasked_max + progress * (guaranteed_unmasked_min - guaranteed_unmasked_max)

    # Generate mask on GPU - corruption probability is now (1.0 - guaranteed_unmasked)
    corruption_prob = 1.0 - guaranteed_unmasked
    mask = torch.rand(x.shape, device=device) < corruption_prob

    # Apply corruption in-place for efficiency
    corrupted_x = x.clone()
    if mask.any():
        # Generate random tokens directly on GPU
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=device)
        corrupted_x = torch.where(mask, random_tokens, corrupted_x)

    return corrupted_x, mask

def apply_random_corruption(x, corruption_prob, guaranteed_unmasked, meta_vocab_size):
    """Strategy 1: Random token corruption (original method)"""
    mask = torch.rand(x.shape) < (corruption_prob * (1.0 - guaranteed_unmasked))
    corrupted_x = x.clone()
    
    if mask.sum() > 0:
        vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
        random_tokens = torch.randint(0, vocab_size_to_use, (mask.sum().item(),))
        corrupted_x[mask] = random_tokens
    
    return corrupted_x, mask

def apply_sticky_corruption_gpu(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size, sticky_p1_divisor=2.0):
    """GPU-optimized Strategy 2: Sticky-style corruption without transitions"""
    device = x.device
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257

    # Use GPU-optimized sticky masking
    sticky_corrupted_x, mask = apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor)
    
    # Replace mask tokens with random tokens (vectorized on GPU)
    mask_positions = (sticky_corrupted_x == mask_token_id)
    if mask_positions.any():
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=device)
        sticky_corrupted_x = torch.where(mask_positions, random_tokens, sticky_corrupted_x)
    
    return sticky_corrupted_x, mask

def apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size, sticky_p1_divisor=2.0):
    """Strategy 2: Sticky-style corruption without transitions"""
    # Use sticky masking logic but replace mask tokens with random tokens
    sticky_corrupted_x, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor)
    
    # Replace mask tokens with random tokens
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    for batch_idx in range(sticky_corrupted_x.shape[0]):
        for pos_idx in range(sticky_corrupted_x.shape[1]):
            if sticky_corrupted_x[batch_idx, pos_idx] == mask_token_id:
                sticky_corrupted_x[batch_idx, pos_idx] = torch.randint(0, vocab_size_to_use, (1,)).item()
    
    return sticky_corrupted_x, mask

def apply_fragment_corruption_gpu(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, sticky_p1_divisor=2.0):
    """GPU-optimized Strategy 3: Fragment-based corruption using real text segments"""
    device = x.device
    batch_size = x.shape[0]

    # Use GPU-optimized sticky masking to get corruption patterns
    _, mask = apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor)
    
    corrupted_x = x.clone()
    
    # Pre-sample all fragment starts for the batch
    fragment_starts = torch.randint(0, len(data) - block_size, (batch_size,)).numpy()
    
    # Load all fragments at once (vectorized)
    fragments = np.zeros((batch_size, block_size), dtype=np.int64)
    for i, start in enumerate(fragment_starts):
        fragments[i] = data[start:start + block_size].astype(np.int64)
    
    # Convert to GPU tensor
    fragments_gpu = torch.from_numpy(fragments).to(device)
    
    # Apply fragment corruption using vectorized operations
    corrupted_x = torch.where(mask, fragments_gpu, corrupted_x)
    
    return corrupted_x, mask

def apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id):
    """Strategy 3: Fragment-based corruption using real text segments"""
    # Use sticky masking to get corruption patterns
    _, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)
    
    corrupted_x = x.clone()
    
    # For each sequence in the batch, get a different source fragment
    for batch_idx in range(x.shape[0]):
        batch_mask = mask[batch_idx]
        if batch_mask.sum() > 0:
            # Sample a random fragment from training data
            fragment_start = torch.randint(0, len(data) - block_size, (1,)).item()
            fragment = torch.from_numpy(data[fragment_start:fragment_start + block_size].astype(np.int64))
            
            # Replace corrupted positions with tokens from the fragment
            corrupted_x[batch_idx][batch_mask] = fragment[batch_mask]
    
    return corrupted_x, mask

def get_progressive_validation_iterations(eval_iters, max_iters):
    """Calculate iteration numbers that validation samples should represent
    
    For eval_iters=20, max_iters=13000: returns [0, 650, 1300, ..., 12350]
    This ensures validation samples represent the full training difficulty progression
    """
    if max_iters <= 0 or eval_iters <= 0:
        return [0] * eval_iters
    
    # Distribute iterations evenly across training duration
    step_size = max_iters / eval_iters
    iterations = [int(i * step_size) for i in range(eval_iters)]
    return iterations

def apply_gpu_masking_validation_progressive(x, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier, 
                                           guaranteed_unmasked, sticky_transition_start, sticky_transition_end, 
                                           sticky_p1_divisor, validation_iter):
    """GPU-optimized validation masking that represents a specific training iteration"""
    # Use the same logic as training, but for the specific validation iteration
    return apply_gpu_masking_training(x, validation_iter, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier,
                                    guaranteed_unmasked, sticky_transition_start, sticky_transition_end, sticky_p1_divisor)

def apply_gpu_masking_training(x, iter_num, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier,
                              guaranteed_unmasked, sticky_transition_start, sticky_transition_end, sticky_p1_divisor=2.0):
    """GPU-optimized training masking with dynamic sticky ratio"""
    # Calculate sticky masking ratio based on current training iteration
    if iter_num < sticky_transition_start:
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        sticky_ratio = 1.0
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress

    # Pre-allocate tensors on GPU
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)

    if sticky_ratio == 0.0:
        # Pure independent masking with per-sample probabilities (fully vectorized)
        batch_size = x.shape[0]
        # Generate different masking probabilities for each sample in the batch
        # Each sample gets a random probability in range [0, 1-guaranteed_unmasked]
        sample_masking_probs = torch.rand(batch_size, device=x.device) * (1.0 - guaranteed_unmasked)
        # Expand to match sequence length for broadcasting
        sample_masking_probs = sample_masking_probs.unsqueeze(1).expand(-1, x.shape[1])
        # Apply per-sample masking probabilities
        mask = torch.rand(x.shape, device=x.device) < sample_masking_probs
        masked_x[mask] = mask_token_id
    elif sticky_ratio == 1.0:
        # Pure sticky masking
        masked_x, mask = apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor)
    else:
        # Mixed strategy: some batches independent, some sticky
        current_batch_size = x.shape[0]
        num_sticky_batches = int(current_batch_size * sticky_ratio)

        # Apply independent masking to first part of batch (vectorized)
        if num_sticky_batches < current_batch_size:
            indep_batch_size = current_batch_size - num_sticky_batches
            # Generate different masking probabilities for each independent sample
            # Each sample gets a random probability in range [0, 1-guaranteed_unmasked]
            sample_masking_probs = torch.rand(indep_batch_size, device=x.device) * (1.0 - guaranteed_unmasked)
            # Expand to match sequence length for broadcasting
            sample_masking_probs = sample_masking_probs.unsqueeze(1).expand(-1, x.shape[1])
            # Apply per-sample masking probabilities
            indep_mask = torch.rand(x[:indep_batch_size].shape, device=x.device) < sample_masking_probs
            masked_x[:indep_batch_size][indep_mask] = mask_token_id
            mask[:indep_batch_size] = indep_mask

        # Apply GPU-accelerated sticky masking to remaining part of batch
        if num_sticky_batches > 0:
            sticky_masked_x, sticky_mask = apply_sticky_masking_gpu(
                x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor
            )
            masked_x[-num_sticky_batches:] = sticky_masked_x
            mask[-num_sticky_batches:] = sticky_mask
    
    return masked_x, mask

def apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor=2.0):
    """GPU-optimized sticky masking with parallel batch processing - logically equivalent to utils.py version"""
    batch_size, seq_len = x.shape
    device = x.device

    # Start with no masks (same as original)
    masked_x = x.clone()

    for round_idx in range(sticky_rounds):
        # Dynamically sample sticky probabilities each round (same logic as original)
        # Use CPU RNG for consistency between CPU and GPU implementations
        p1 = torch.rand(1).item() / (sticky_rounds * sticky_p1_divisor)  # Sample from (0, 1/(rounds*divisor))
        p2 = min(1.0, p1 * sticky_p1_p2_multiplier)  # p2 = p1 * multiplier, capped at 1
        
        # Current mask state
        current_mask = (masked_x == mask_token_id)
        
        # For each position, check if neighbors are masked (vectorized version of original logic)
        neighbor_masked = torch.zeros_like(current_mask, dtype=torch.bool, device=device)
        
        # Check left neighbor
        neighbor_masked[:, 1:] |= current_mask[:, :-1]
        # Check right neighbor  
        neighbor_masked[:, :-1] |= current_mask[:, 1:]
        
        # Generate random values for masking decision
        # Use CPU RNG for consistency, then move to target device
        rand_vals = torch.rand(batch_size, seq_len).to(device)
        
        # Apply p1 where neighbors not masked, p2 where neighbors masked
        mask_probs = torch.where(neighbor_masked, p2, p1)
        new_masks = rand_vals < mask_probs
        
        # Don't mask positions that are already masked (same as original)
        new_masks = new_masks & ~current_mask
        
        # Apply new masks
        masked_x[new_masks] = mask_token_id
    
    # Final mask state
    final_mask = (masked_x == mask_token_id)
    return masked_x, final_mask

def apply_random_noise_to_unmasked_gpu(x, mask, noise_max_ratio, meta_vocab_size, iter_num, 
                                       sticky_transition_start, sticky_transition_end):
    """GPU-optimized random noise application to unmasked positions"""
    # Calculate progressive noise ratio based on training iteration
    if iter_num < sticky_transition_start:
        progressive_noise_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        progressive_noise_ratio = noise_max_ratio
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        progressive_noise_ratio = progress * noise_max_ratio
    
    if progressive_noise_ratio <= 0.0:
        return x
    
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    
    # All operations on GPU, fully vectorized
    unmasked_positions = ~mask
    batch_size = x.shape[0]
    
    # Generate noise ratios for all batch elements at once
    noise_ratios = torch.rand(batch_size, device=x.device) * progressive_noise_ratio
    noise_ratios_expanded = noise_ratios.unsqueeze(1).expand(-1, x.shape[1])
    
    # Generate random probabilities for all positions at once
    random_probs = torch.rand_like(x, dtype=torch.float, device=x.device)
    
    # Determine which positions to noise (fully vectorized)
    should_noise = unmasked_positions & (random_probs < noise_ratios_expanded)
    
    # Apply noise in-place if any positions need noising
    if should_noise.any():
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=x.device)
        x = torch.where(should_noise, random_tokens, x)
    
    return x

def apply_random_noise_to_unmasked(x, mask, noise_max_ratio, meta_vocab_size, iter_num,
                                   sticky_transition_start, sticky_transition_end):
    """Apply random token noise to unmasked positions in input for unmasking training."""
    # Calculate progressive noise ratio based on training iteration
    if iter_num < sticky_transition_start:
        # No noise during early training
        progressive_noise_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        # Full noise ratio after transition
        progressive_noise_ratio = noise_max_ratio
    else:
        # Gradual increase from 0 to noise_max_ratio
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        progressive_noise_ratio = progress * noise_max_ratio
    
    if progressive_noise_ratio <= 0.0:
        return x
    
    noisy_x = x.clone()
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    
    # Get unmasked positions for all batch elements
    unmasked_positions = ~mask  # Shape: (batch_size, seq_len)
    
    # Sample noise ratios for each batch element using progressive ratio
    batch_size = x.shape[0]
    noise_ratios = torch.rand(batch_size, device=x.device) * progressive_noise_ratio  # Shape: (batch_size,)
    
    # For each position, determine if it should be noised
    # First, generate random values for all unmasked positions
    random_probs = torch.rand_like(x, dtype=torch.float, device=x.device)  # Shape: (batch_size, seq_len)
    
    # Create noise mask: position gets noised if it's unmasked AND random_prob < noise_ratio
    noise_ratios_expanded = noise_ratios.unsqueeze(1).expand(-1, x.shape[1])  # Shape: (batch_size, seq_len)
    should_noise = unmasked_positions & (random_probs < noise_ratios_expanded)
    
    # Generate random tokens for all positions that should be noised
    if should_noise.any():
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=x.device)
        noisy_x = torch.where(should_noise, random_tokens, noisy_x)
    
    return noisy_x

def load_synthetic_model(checkpoint_path, device, extended_vocab_size):
    """Load the synthetic model for generating fake data in remasking training"""
    global synthetic_model
    
    if not checkpoint_path or synthetic_model is not None:
        return
    
    try:
        print(f"Loading synthetic model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model arguments from checkpoint
        checkpoint_model_args = checkpoint['model_args']
        
        # Create synthetic model with same architecture as checkpoint
        synthetic_gptconf = GPTConfig(**checkpoint_model_args)
        synthetic_model = GPT(synthetic_gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        # Fix keys if needed (same as main model loading)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        synthetic_model.load_state_dict(state_dict)
        synthetic_model.to(device)
        synthetic_model.eval()  # Always in eval mode
        
        print(f"Synthetic model loaded successfully (vocab_size: {synthetic_model.config.vocab_size})")
        
    except Exception as e:
        print(f"Warning: Could not load synthetic model from {checkpoint_path}: {e}")
        synthetic_model = None

def apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size, sticky_p1_divisor=2.0):
    """Strategy 4: Synthetic corruption using loaded unmasking model"""
    global synthetic_model

    if synthetic_model is None:
        print("Warning: Synthetic model not loaded, falling back to sticky corruption")
        return apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size, sticky_p1_divisor)

    # Use sticky masking to get corruption patterns
    _, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier, sticky_p1_divisor)
    
    # Create input for synthetic model: replace masked positions with mask tokens
    synthetic_input = x.clone()
    synthetic_input[mask] = mask_token_id
    
    # Generate synthetic data using the loaded model
    with torch.no_grad():
        # Move to device if needed
        if synthetic_input.device != next(synthetic_model.parameters()).device:
            synthetic_input = synthetic_input.to(next(synthetic_model.parameters()).device)
        
        # Get logits from synthetic model
        logits, _ = synthetic_model(synthetic_input, None)
        
        # Sample from the model's distribution
        # Use temperature sampling for more realistic synthetic data
        temperature = 0.8
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        
        # Sample tokens for masked positions - vectorized approach
        corrupted_x = x.clone()
        
        # Create a sampling mask and sample all at once
        if mask.any():
            # Get flattened indices where mask is True
            mask_flat = mask.view(-1)
            probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
            
            # Sample for all masked positions at once
            masked_probs = probs_flat[mask_flat]  # (num_masked_total, vocab_size)
            sampled_tokens = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
            
            # Move sampled tokens to same device as corrupted_x
            sampled_tokens = sampled_tokens.to(corrupted_x.device)
            
            # Place sampled tokens back
            corrupted_x_flat = corrupted_x.view(-1)
            corrupted_x_flat[mask_flat] = sampled_tokens
            corrupted_x = corrupted_x_flat.view_as(x)
    
    return corrupted_x, mask

def get_batch(split, ctx: TrainingContext, validation_sample_idx=None):
    """Main batch generation function that delegates to specific training type functions"""
    if ctx.training_type == 'remasking':
        return get_batch_remasking(split, ctx, validation_sample_idx)
    elif ctx.training_type == 'remasking_binary':
        return get_batch_remasking_binary(split, ctx, validation_sample_idx)
    else:
        return get_batch_unmasking(split, ctx, validation_sample_idx)

def get_batch_unmasking(split, ctx: TrainingContext, validation_sample_idx=None):
    """Ultra-fast unmasking implementation with aggressive caching + prefetching"""
    global _val_batch_cache, _progressive_val_cache, _data_cache, _valid_indices_cache

    # For validation with progressive sampling
    if split == 'val' and validation_sample_idx is not None:
        cache_key = f"unmasking_{validation_sample_idx}"
        if cache_key in _progressive_val_cache:
            return _progressive_val_cache[cache_key]
    # For validation, use legacy cached batch for backward compatibility
    elif split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Cache memory-mapped data and valid indices - MAJOR SPEEDUP
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Cache the expensive valid indices computation
        print(f"Computing valid indices for {split}... (one-time cost)")
        _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        print(f"Found {len(_valid_indices_cache[split])} valid indices for {split}")
        
        # Start prefetching for training data
        if split == 'train':
            start_prefetch(ctx)

    # Try to get prefetched data for training
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = _prefetch_queue.get_nowait()
        except:
            pass  # Queue empty, generate normally
    
    # Generate data if not prefetched
    if x_np is None:
        data = _data_cache[split]
        valid_indices = _valid_indices_cache[split]
        
        # Fast index sampling - all on CPU to avoid GPU-CPU sync
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        # VECTORIZED DATA LOADING - Load entire batch at once
        x_np = np.zeros((ctx.batch_size, ctx.block_size), dtype=np.int64)
        for i, start_idx in enumerate(ix_np):
            x_np[i] = data[start_idx:start_idx+ctx.block_size].astype(np.int64)
    
    # Single GPU transfer with pinned memory
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # For progressive validation, calculate parameters for the specific validation iteration
    if split == 'val' and validation_sample_idx is not None:
        # Calculate which training iteration this validation sample represents
        progressive_iterations = get_progressive_validation_iterations(ctx.eval_iters, ctx.max_iters)
        validation_iter = progressive_iterations[validation_sample_idx % len(progressive_iterations)]
        
        # Calculate guaranteed_unmasked for this specific iteration
        current_guaranteed_unmasked = ctx.get_guaranteed_unmasked(validation_iter)
        
        # Use deterministic seed for validation consistency
        torch.manual_seed(42 + validation_sample_idx)
        masked_x, mask = apply_gpu_masking_validation_progressive(x, ctx.mask_token_id, ctx.sticky_rounds, 
                                                               ctx.sticky_p1_p2_multiplier, current_guaranteed_unmasked,
                                                               ctx.sticky_transition_start, ctx.sticky_transition_end, 
                                                               ctx.sticky_p1_divisor, validation_iter)
        torch.manual_seed(1337 + ctx.seed_offset)
        
        # Apply progressive noise for this validation iteration
        masked_x = apply_random_noise_to_unmasked_gpu(masked_x, mask, ctx.noise_max_ratio, ctx.meta_vocab_size, validation_iter,
                                                      ctx.sticky_transition_start, ctx.sticky_transition_end)
    else:
        # Calculate dynamic guaranteed_unmasked using centralized logic
        current_guaranteed_unmasked = ctx.get_guaranteed_unmasked()
        
        # GPU-accelerated masking operations (already on GPU)
        if split == 'val':
            torch.manual_seed(42)
            # Legacy validation masking - use training logic with current iter_num for consistency
            masked_x, mask = apply_gpu_masking_training(x, ctx.iter_num, ctx.mask_token_id, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier,
                                                      current_guaranteed_unmasked, ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.sticky_p1_divisor)
            torch.manual_seed(1337 + ctx.seed_offset)
        else:
            masked_x, mask = apply_gpu_masking_training(x, ctx.iter_num, ctx.mask_token_id, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier,
                                                      current_guaranteed_unmasked, ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.sticky_p1_divisor)

        # Apply random noise to unmasked positions (already on GPU)
        if split == 'val' and validation_sample_idx is None:
            # Legacy path - use current iteration for noise
            masked_x = apply_random_noise_to_unmasked_gpu(masked_x, mask, ctx.noise_max_ratio, ctx.meta_vocab_size, ctx.iter_num,
                                                          ctx.sticky_transition_start, ctx.sticky_transition_end)
        elif split == 'train':
            masked_x = apply_random_noise_to_unmasked_gpu(masked_x, mask, ctx.noise_max_ratio, ctx.meta_vocab_size, ctx.iter_num,
                                                          ctx.sticky_transition_start, ctx.sticky_transition_end)

    # Target is original x
    y = x.clone()

    # Cache validation batch for consistency
    if split == 'val':
        if validation_sample_idx is not None:
            cache_key = f"unmasking_{validation_sample_idx}"
            _progressive_val_cache[cache_key] = (masked_x, y, mask)
        else:
            _val_batch_cache = (masked_x, y, mask)

    return masked_x, y, mask

def get_batch_remasking(split, ctx: TrainingContext):
    """Ultra-fast remasking implementation with aggressive caching + prefetching"""
    global _val_batch_cache, _data_cache, _valid_indices_cache

    # For validation, use cached batch to ensure consistency
    if split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Cache memory-mapped data and valid indices - MAJOR SPEEDUP (same as unmasking)
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Cache the expensive valid indices computation
        print(f"Computing valid indices for {split}... (one-time cost)")
        _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        print(f"Found {len(_valid_indices_cache[split])} valid indices for {split}")
        
        # Start prefetching for training data
        if split == 'train':
            start_prefetch(ctx)

    # Try to get prefetched data for training (reuse existing prefetch system)
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = _prefetch_queue.get_nowait()
        except:
            pass  # Queue empty, generate normally
    
    # Generate data if not prefetched
    if x_np is None:
        data = _data_cache[split]
        valid_indices = _valid_indices_cache[split]
        
        # Fast index sampling - all on CPU to avoid GPU-CPU sync
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        # VECTORIZED DATA LOADING - Load entire batch at once
        x_np = np.zeros((ctx.batch_size, ctx.block_size), dtype=np.int64)
        for i, start_idx in enumerate(ix_np):
            x_np[i] = data[start_idx:start_idx+ctx.block_size].astype(np.int64)
    
    # Single GPU transfer with pinned memory
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # GPU-accelerated corruption strategy selection and application
    if ctx.remasking_corruption_strategy == 'random':
        corrupted_x, mask = apply_random_corruption_gpu(x, ctx.iter_num, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                       ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)
    elif ctx.remasking_corruption_strategy == 'sticky':
        corrupted_x, mask = apply_sticky_corruption_gpu(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'fragment':
        corrupted_x, mask = apply_fragment_corruption_gpu(x, _data_cache[split], ctx.block_size, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'synthetic':
        corrupted_x, mask = apply_synthetic_corruption(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'mixed':
        # Select strategy based on weights (keep CPU choice for simplicity)
        strategy_choice = np.random.choice(['random', 'sticky', 'fragment', 'synthetic'], p=ctx.remasking_strategy_weights)
        
        if strategy_choice == 'random':
            corrupted_x, mask = apply_random_corruption_gpu(x, ctx.iter_num, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                           ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)
        elif strategy_choice == 'sticky':
            corrupted_x, mask = apply_sticky_corruption_gpu(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
        elif strategy_choice == 'fragment':
            corrupted_x, mask = apply_fragment_corruption_gpu(x, _data_cache[split], ctx.block_size, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.sticky_p1_divisor)
        else:  # synthetic
            corrupted_x, mask = apply_synthetic_corruption(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    else:
        # Default fallback to random
        corrupted_x, mask = apply_random_corruption_gpu(x, ctx.iter_num, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                       ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)

    # Target: original tokens at correct positions, wrong_token_id at corrupted positions (already on GPU)
    y = x.clone()
    y[mask] = ctx.wrong_token_id

    # Cache validation batch for consistency
    if split == 'val':
        if validation_sample_idx is not None:
            cache_key = f"remasking_{validation_sample_idx}"
            _progressive_val_cache[cache_key] = (corrupted_x, y, mask)
        else:
            _val_batch_cache = (corrupted_x, y, mask)

    return corrupted_x, y, mask

def get_batch_remasking_binary(split, ctx: TrainingContext, validation_sample_idx=None):
    """GPU-optimized remasking binary training: symmetric task with remask_good_id and remask_wrong_id targets"""
    # Ultra-fast remasking binary implementation - reuse all optimizations from remasking
    global _val_batch_cache, _progressive_val_cache, _data_cache, _valid_indices_cache

    # For validation with progressive sampling
    if split == 'val' and validation_sample_idx is not None:
        cache_key = f"remasking_binary_{validation_sample_idx}"
        if cache_key in _progressive_val_cache:
            return _progressive_val_cache[cache_key]
    # For validation, use legacy cached batch for backward compatibility
    elif split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Use same data caching and prefetching as remasking
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        print(f"Computing valid indices for {split}... (one-time cost)")
        _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        print(f"Found {len(_valid_indices_cache[split])} valid indices for {split}")
        
        if split == 'train':
            start_prefetch(ctx)

    # Try to get prefetched data for training (reuse existing prefetch system)
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = _prefetch_queue.get_nowait()
        except:
            pass

    # Generate data if not prefetched (same as remasking)
    if x_np is None:
        data = _data_cache[split]
        valid_indices = _valid_indices_cache[split]
        
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + ctx.seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        x_np = np.zeros((ctx.batch_size, ctx.block_size), dtype=np.int64)
        for i, start_idx in enumerate(ix_np):
            x_np[i] = data[start_idx:start_idx+ctx.block_size].astype(np.int64)
    
    # Single GPU transfer
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # Determine which iteration to use for corruption strategy
    if split == 'val' and validation_sample_idx is not None:
        # For progressive validation, use the specific validation iteration
        progressive_iterations = get_progressive_validation_iterations(ctx.eval_iters, ctx.max_iters)
        corruption_iter = progressive_iterations[validation_sample_idx % len(progressive_iterations)]
        # Use deterministic seed for validation consistency
        np.random.seed(42 + validation_sample_idx)
    else:
        corruption_iter = ctx.iter_num

    # GPU-accelerated corruption (same strategies as remasking)
    if ctx.remasking_corruption_strategy == 'random':
        corrupted_x, mask = apply_random_corruption_gpu(x, corruption_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                       ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)
    elif ctx.remasking_corruption_strategy == 'sticky':
        corrupted_x, mask = apply_sticky_corruption_gpu(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'fragment':
        corrupted_x, mask = apply_fragment_corruption_gpu(x, _data_cache[split], ctx.block_size, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'synthetic':
        corrupted_x, mask = apply_synthetic_corruption(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    elif ctx.remasking_corruption_strategy == 'mixed':
        strategy_choice = np.random.choice(['random', 'sticky', 'fragment', 'synthetic'], p=ctx.remasking_strategy_weights)
        
        if strategy_choice == 'random':
            corrupted_x, mask = apply_random_corruption_gpu(x, corruption_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                           ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)
        elif strategy_choice == 'sticky':
            corrupted_x, mask = apply_sticky_corruption_gpu(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
        elif strategy_choice == 'fragment':
            corrupted_x, mask = apply_fragment_corruption_gpu(x, _data_cache[split], ctx.block_size, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.sticky_p1_divisor)
        else:  # synthetic
            corrupted_x, mask = apply_synthetic_corruption(x, ctx.sticky_rounds, ctx.sticky_p1_p2_multiplier, ctx.mask_token_id, ctx.meta_vocab_size, ctx.sticky_p1_divisor)
    else:
        corrupted_x, mask = apply_random_corruption_gpu(x, corruption_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                       ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, ctx.random_mask_warmup)
    
    # Restore random seed for training
    if split == 'val' and validation_sample_idx is not None:
        np.random.seed()

    # Binary targets: remask_good_id for uncorrupted, remask_wrong_id for corrupted (already on GPU)
    y = torch.full_like(x, ctx.remask_good_id)
    y[mask] = ctx.remask_wrong_id

    # Cache validation batch for consistency
    if split == 'val':
        if validation_sample_idx is not None:
            cache_key = f"remasking_binary_{validation_sample_idx}"
            _progressive_val_cache[cache_key] = (corrupted_x, y, mask)
        else:
            _val_batch_cache = (corrupted_x, y, mask)

    return corrupted_x, y, mask

def estimate_loss(model, torch_ctx, timer, training_ctx: TrainingContext):
    """Estimate loss over either split using many batches"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(training_ctx.eval_iters)
        # Track masked token ratios for all splits
        masked_token_ratios = []

        if split == 'val':
            # For validation, also track model vs random performance
            model_probs = []
            # For binary classification and remasking, track corruption statistics
            if training_ctx.training_type in ['remasking_binary', 'remasking']:
                total_positions = 0
                corrupted_positions = 0
            else:
                random_prob = 1.0 / training_ctx.extended_vocab_size  # Random chance probability

        # For validation, create/use the progressive validation set
        if split == 'val':
            # Create the progressive validation set once (cached)
            full_val_X, full_val_Y, full_val_masks = create_progressive_validation_set(training_ctx)
            print(f"Using cached progressive validation set ({full_val_X.shape[0]} total samples)")
        
        for k in range(training_ctx.eval_iters):
            with timer.time_function('validation_data_generation'):
                if split == 'val':
                    # Extract the k-th batch from pre-generated validation set
                    start_idx = k * training_ctx.batch_size
                    end_idx = start_idx + training_ctx.batch_size
                    X = full_val_X[start_idx:end_idx]
                    Y = full_val_Y[start_idx:end_idx]
                    mask = full_val_masks[start_idx:end_idx]
                    
                    if k == 0:  # Show batch composition for first batch only
                        progressive_iterations = get_progressive_validation_iterations(training_ctx.eval_iters, training_ctx.max_iters)
                        print(f"  Batch {k+1}/{training_ctx.eval_iters}: samples {start_idx}-{end_idx-1} (iterations {progressive_iterations[0]} to {progressive_iterations[-1]})")
                else:
                    X, Y, mask = get_batch(split, training_ctx)

            # Calculate masked token ratio for this batch
            mask_ratio = mask.float().mean().item()
            masked_token_ratios.append(mask_ratio)

            with torch_ctx:
                with timer.time_function('validation_forward_pass'):
                    # This is handled in validation_loss_computation section
                    pass
                with timer.time_function('validation_loss_computation'):
                    # Optimized single forward pass for validation
                    logits, loss = model(X, Y)

                    # Apply masking for unmasking training only
                    if training_ctx.training_type == 'unmasking' and mask.any():
                        # Fast validation path - single reshape and boolean indexing
                        logits_reshaped = logits.view(-1, logits.size(-1))
                        targets_reshaped = Y.view(-1)
                        mask_reshaped = mask.view(-1)

                        loss = torch.nn.functional.cross_entropy(
                            logits_reshaped[mask_reshaped],
                            targets_reshaped[mask_reshaped],
                            reduction='mean'
                        )
                    # For remasking variants, model's internal loss is correct

                # For validation, compute model vs random statistics
                if split == 'val':
                    # Get probabilities from logits and flatten for statistics
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                    probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
                    targets_flat = Y.view(-1)  # (batch_size * seq_len,) - needed for validation stats

                    if training_ctx.training_type == 'remasking_binary':
                        # For binary classification, compute accuracy on all positions
                        # Track corruption statistics for proper baseline
                        total_positions += targets_flat.numel()
                        corrupted_positions += (targets_flat == training_ctx.remask_wrong_id).sum().item()

                        # Get probabilities for correct binary classification
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                    elif training_ctx.training_type == 'remasking':
                        # For remasking, compute accuracy on ALL positions (corrupted + uncorrupted)
                        # Track corruption statistics for proper baseline
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        total_positions += targets_flat.numel()
                        corrupted_positions += mask_flat.sum().item()  # mask indicates corrupted positions

                        # Get probabilities for correct predictions at ALL positions
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                    else:
                        # For unmasking, compute on masked positions only
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        masked_positions = mask_flat.bool()
                        if masked_positions.sum() > 0:  # Only if there are masked tokens
                            correct_token_probs = probs_flat[masked_positions, targets_flat[masked_positions]]
                            model_probs.extend(correct_token_probs.cpu().tolist())

            losses[k] = loss.item()

        out[split] = losses.mean()
        
        if split == 'val':
            total_samples = training_ctx.eval_iters * training_ctx.batch_size
            print(f"  Validation complete: {training_ctx.eval_iters} batches processed ({total_samples} cached samples), avg loss = {out[split]:.4f}")

        # Add masked token ratio statistics
        if masked_token_ratios:
            avg_masked_ratio = sum(masked_token_ratios) / len(masked_token_ratios)
            out[f'{split}_masked_token_ratio'] = avg_masked_ratio
            out[f'{split}_min_masked_token_ratio'] = min(masked_token_ratios)
            out[f'{split}_max_masked_token_ratio'] = max(masked_token_ratios)

        # Add model vs random comparison for validation
        if split == 'val' and model_probs:
            # VALIDATION METRICS STABILITY CHECK
            finite_probs = [p for p in model_probs if math.isfinite(p)]
            if len(finite_probs) == 0:
                print(f"\n*** VALIDATION METRICS INSTABILITY ***")
                print(f"All model probabilities are NaN/Inf (total: {len(model_probs)})")
                print(f"Sample of problematic values: {model_probs[:5]}")
                out[f'{split}_model_vs_random'] = float('nan')
                out[f'{split}_avg_correct_prob'] = float('nan')
                if training_ctx.training_type in ['remasking_binary', 'remasking']:
                    out[f'{split}_corruption_ratio'] = corrupted_positions / total_positions if total_positions > 0 else 0.0
                    out[f'{split}_random_baseline'] = 0.5  # Fallback value
            elif len(finite_probs) < len(model_probs):
                print(f"WARNING: {len(model_probs) - len(finite_probs)}/{len(model_probs)} model probabilities are NaN/Inf")
                avg_model_prob = sum(finite_probs) / len(finite_probs)
            else:
                avg_model_prob = sum(model_probs) / len(model_probs)
            
            # Only proceed if we have valid probabilities
            if len(finite_probs) > 0:
                if training_ctx.training_type == 'remasking_binary':
                    # For binary classification, compare against distribution-aware random baseline
                    corruption_ratio = corrupted_positions / total_positions if total_positions > 0 else 0.0
                    # Random classifier matching the distribution would get:
                    # P(correct) = P(guess_good) * P(actual_good) + P(guess_wrong) * P(actual_wrong)
                    # With optimal random strategy: P(guess_good) = P(actual_good), P(guess_wrong) = P(actual_wrong)
                    random_accuracy = (1 - corruption_ratio) ** 2 + corruption_ratio ** 2
                    prob_ratio = avg_model_prob / random_accuracy if random_accuracy > 0 else float('inf')
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob
                    out[f'{split}_corruption_ratio'] = corruption_ratio
                    out[f'{split}_random_baseline'] = random_accuracy
                elif training_ctx.training_type == 'remasking':
                    # For remasking, the task is corruption detection + appropriate response
                    corruption_ratio = corrupted_positions / total_positions if total_positions > 0 else 0.0
                    # Optimal random baseline: always guess the majority class
                    # With corruption_ratio=0.2: always guess "uncorrupted" → 80% accuracy
                    # General: max(corruption_ratio, 1-corruption_ratio)
                    random_accuracy = max(corruption_ratio, 1 - corruption_ratio)
                    prob_ratio = avg_model_prob / random_accuracy if random_accuracy > 0 else float('inf')
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob
                    out[f'{split}_corruption_ratio'] = corruption_ratio
                    out[f'{split}_random_baseline'] = random_accuracy
                else:
                    # For unmasking, use uniform random baseline
                    prob_ratio = avg_model_prob / random_prob
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob

    model.train()
    return out

def get_lr(it, ctx: TrainingContext):
    """Learning rate decay scheduler (cosine with warmup)"""
    # 1) linear warmup for warmup_iters steps
    if it < ctx.warmup_iters:
        return ctx.learning_rate * (it + 1) / (ctx.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > ctx.lr_decay_iters:
        return ctx.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - ctx.warmup_iters) / (ctx.lr_decay_iters - ctx.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return ctx.min_lr + coeff * (ctx.learning_rate - ctx.min_lr)