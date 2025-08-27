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
from enum import Enum
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils import Timer, log_masking_stats
from dataclasses import dataclass

# Global variables for data caching and prefetching
_val_batch_cache = None
_progressive_val_cache = {}  # Cache for progressive validation batches
_progressive_val_full_cache = None  # Cache for full progressive validation set (all 320 samples)
_unmasking_val_set = None  # Complete validation set for unmasking training (eval_iters * batch_size samples)
_data_cache = {'train': None, 'val': None}
_valid_indices_cache = {'train': None, 'val': None}
_prefetch_enabled = True
_prefetch_queue = Queue(maxsize=2)
_prefetch_thread = None
_prefetch_active = False

# Global synthetic model for remasking
synthetic_model = None

class UnmaskingStageType(Enum):
    """Enumeration of available unmasking stage types"""
    STICKY = "sticky"
    RANDOM = "random"

@dataclass
class BaseStageConfig(ABC):
    """Base class for stage-specific configuration"""
    val_loss_stale_count: int
    
    @abstractmethod
    def get_stage_type(self) -> UnmaskingStageType:
        """Return the stage type"""
        pass

@dataclass
class StickyStageConfig(BaseStageConfig):
    """Configuration for sticky masking stages"""
    target_masked_ratio: float
    p1_probability: float
    p2_probability: float
    
    def get_stage_type(self) -> UnmaskingStageType:
        return UnmaskingStageType.STICKY

@dataclass
class RandomStageConfig(BaseStageConfig):
    """Configuration for random masking stages"""
    max_masked_ratio: float
    
    def get_stage_type(self) -> UnmaskingStageType:
        return UnmaskingStageType.RANDOM

@dataclass
class UnmaskingStage:
    """Configuration for a single stage of unmasking training"""
    config: Union[StickyStageConfig, RandomStageConfig]
    
    def get_stage_type(self) -> UnmaskingStageType:
        return self.config.get_stage_type()
    
    def get_val_loss_stale_count(self) -> int:
        return self.config.val_loss_stale_count

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
    
    # Training iteration and stage tracking
    iter_num: int = 0
    current_stage: int = 0
    val_loss_stale_count: int = 0
    best_val_loss_this_stage: float = float('inf')
    
    # Unmasking stages (only for unmasking training)
    unmasking_stages: list = None
    
    # Remasking strategy (only for remasking/remasking_binary)
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
        
        # Default unmasking stages if not provided
        if self.training_type == 'unmasking' and self.unmasking_stages is None:
            self.unmasking_stages = [
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.2, p1_probability=0.3, p2_probability=0.0, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.4, p1_probability=0.2, p2_probability=0.8, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.6, p1_probability=0.1, p2_probability=0.9, val_loss_stale_count=10)),
            ]
    
    def get_current_stage_config(self) -> UnmaskingStage:
        """Get configuration for current unmasking stage"""
        if self.training_type != 'unmasking' or not self.unmasking_stages:
            return None
        
        if self.current_stage >= len(self.unmasking_stages):
            # Return last stage if we've exceeded all stages
            return self.unmasking_stages[-1]
        
        return self.unmasking_stages[self.current_stage]
    
    def advance_stage(self):
        """Advance to next unmasking stage and reset stale count"""
        if self.current_stage < len(self.unmasking_stages) - 1:
            self.current_stage += 1
            self.val_loss_stale_count = 0
            self.best_val_loss_this_stage = float('inf')
            return True
        return False

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

    # VECTORIZED DATA LOADING - Use advanced indexing for parallel loading
    ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
    x_np = data[ix_expanded].astype(np.int64)
    
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
    global _progressive_val_cache, _val_batch_cache, _progressive_val_full_cache, _unmasking_val_set
    _progressive_val_cache.clear()
    _val_batch_cache = None
    _progressive_val_full_cache = None
    _unmasking_val_set = None

def create_unmasking_validation_set(ctx: TrainingContext):
    """Create complete validation set with samples evenly distributed across all stages"""
    global _unmasking_val_set, _data_cache, _valid_indices_cache
    
    if _unmasking_val_set is not None:
        print("Using existing validation set from cache")
        return  # Already created
    
    print("Creating validation set with samples from all stages...")
    
    # Cache validation data if not already cached
    if _data_cache['val'] is None:
        _data_cache['val'] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        _valid_indices_cache['val'] = find_double_newline_indices(_data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
    
    data = _data_cache['val']
    valid_indices = _valid_indices_cache['val']
    
    total_samples = ctx.eval_iters * ctx.batch_size
    num_stages = len(ctx.unmasking_stages)
    samples_per_stage = total_samples // num_stages
    
    validation_batches = []
    
    # Generate samples for each stage
    torch.manual_seed(42)  # Fixed seed for reproducible validation set
    
    for stage_idx, stage_config in enumerate(ctx.unmasking_stages):
        stage_samples = samples_per_stage
        # Handle remainder samples
        if stage_idx < (total_samples % num_stages):
            stage_samples += 1
            
        stage_type = stage_config.get_stage_type()
        stage_info = f"  Stage {stage_idx} ({stage_type.value}): {stage_samples} samples"
        if stage_type == UnmaskingStageType.STICKY:
            config = stage_config.config
            stage_info += f" (target_ratio={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f})"
        elif stage_type == UnmaskingStageType.RANDOM:
            config = stage_config.config
            stage_info += f" (max_ratio={config.max_masked_ratio:.1f})"
        print(stage_info)
        
        # Generate batches for this stage
        stage_batches = []
        samples_generated = 0
        
        while samples_generated < stage_samples:
            batch_size = min(ctx.batch_size, stage_samples - samples_generated)
            
            # Sample data indices
            if len(valid_indices) == 0:
                ix_np = torch.randint(len(data) - ctx.block_size, (batch_size,)).numpy()
            else:
                ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
            
            # Load data with vectorized indexing
            ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
            x_np = data[ix_expanded].astype(np.int64)
            
            # Convert to tensor
            x = torch.from_numpy(x_np)
            if ctx.device_type == 'cuda':
                x = x.pin_memory().to(ctx.device, non_blocking=True)
            else:
                x = x.to(ctx.device)
            
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id)
            
            stage_batches.append((masked_x.clone(), x.clone(), mask.clone()))
            samples_generated += batch_size
        
        validation_batches.extend(stage_batches)
    
    torch.manual_seed(1337 + ctx.seed_offset)  # Reset seed
    _unmasking_val_set = validation_batches
    print(f"Validation set created: {len(validation_batches)} batches, {sum(b[0].size(0) for b in validation_batches)} total samples")

def get_unmasking_validation_batch(ctx: TrainingContext, batch_idx=None):
    """Get a specific batch from the pre-created validation set"""
    global _unmasking_val_set
    
    if _unmasking_val_set is None:
        create_unmasking_validation_set(ctx)
    
    if batch_idx is None:
        batch_idx = 0
    
    # Handle batch index wrapping
    batch_idx = batch_idx % len(_unmasking_val_set)
    return _unmasking_val_set[batch_idx]


def apply_random_masking_gpu(x, max_masked_ratio, mask_token_id):
    """
    GPU-optimized random masking for unmasking training.
    Each sample in the batch gets a different random masking probability.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        max_masked_ratio: Maximum fraction of tokens to mask (0.0 to 1.0)
        mask_token_id: Token ID to use for masking
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Generate different masking probability for each sample in the batch
    mask_probs = torch.rand(batch_size, device=device) * max_masked_ratio  # Shape: (batch_size,)
    
    # Generate random values for each token position
    rand_vals = torch.rand_like(x, dtype=torch.float, device=device)  # Shape: (batch_size, seq_len)
    
    # Expand mask probabilities to match token dimensions
    mask_probs_expanded = mask_probs.unsqueeze(1).expand(-1, seq_len)  # Shape: (batch_size, seq_len)
    
    # Create mask: True where random value is less than the sample's masking probability
    mask = rand_vals < mask_probs_expanded
    
    # Apply masking
    masked_x = x.clone()
    masked_x[mask] = mask_token_id
    
    return masked_x, mask

def apply_stage_masking(x, stage_config: UnmaskingStage, mask_token_id):
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: UnmaskingStage configuration
        mask_token_id: Token ID to use for masking
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config.get_stage_type()
    
    if stage_type == UnmaskingStageType.RANDOM:
        config = stage_config.config
        return apply_random_masking_gpu(x, config.max_masked_ratio, mask_token_id)
    elif stage_type == UnmaskingStageType.STICKY:
        config = stage_config.config
        return apply_target_driven_sticky_masking_gpu(
            x, config.target_masked_ratio, config.p1_probability, 
            config.p2_probability, mask_token_id
        )
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")

def apply_target_driven_sticky_masking_gpu(x, target_masked_ratio, p1_probability, p2_probability, mask_token_id):
    """
    GPU-optimized target-driven sticky masking for unmasking training.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        target_masked_ratio: Target fraction of tokens to mask (0.0 to 1.0)
        p1_probability: Probability of masking when no neighbors are masked
        p2_probability: Probability of masking when neighbors are masked
        mask_token_id: Token ID to use for masking
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate target number of masked tokens per sequence
    target_masked_count = int(target_masked_ratio * seq_len)
    
    if target_masked_count == 0:
        # No masking needed - return early
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Start with no masks
    masked_x = x.clone()
    
    # Pre-allocate tensors to avoid repeated allocations
    current_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    neighbor_masked = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Continue masking until we reach the target for each sequence
    max_rounds = min(1000, target_masked_count * 10)  # Adaptive safety limit
    target_tensor = torch.tensor(target_masked_count, device=device, dtype=torch.long)
    
    for round_idx in range(max_rounds):
        # Update current mask state
        current_mask = (masked_x == mask_token_id)
        
        # Check if we've reached target for all sequences (GPU-only operation)
        current_counts = current_mask.sum(dim=1)  # (batch_size,)
        sequences_need_more = current_counts < target_tensor
        
        if not sequences_need_more.any():
            break  # All sequences reached target
        
        # Find neighbor positions for sticky masking (reuse buffer)
        neighbor_masked.zero_()
        
        # Check left and right neighbors (vectorized)
        neighbor_masked[:, 1:] |= current_mask[:, :-1]  # Left neighbor
        neighbor_masked[:, :-1] |= current_mask[:, 1:]  # Right neighbor
        
        # Generate random values for masking decision (single GPU call)
        rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
        
        # Apply different probabilities based on neighbor status (vectorized)
        mask_probs = torch.where(neighbor_masked, p2_probability, p1_probability)
        new_masks = (rand_vals < mask_probs) & ~current_mask
        
        # Only mask sequences that haven't reached target yet (vectorized)
        sequences_need_more_expanded = sequences_need_more.unsqueeze(1).expand(-1, seq_len)
        new_masks &= sequences_need_more_expanded
        
        # Apply new masks (vectorized)
        masked_x[new_masks] = mask_token_id
    
    # Final adjustment: remove excess masks with fully vectorized approach
    final_mask = (masked_x == mask_token_id)
    final_counts = final_mask.sum(dim=1)  # (batch_size,)
    
    # Only process sequences that exceeded target (minimize CPU-GPU sync)
    exceeded_sequences = torch.where(final_counts > target_tensor)[0]
    
    if exceeded_sequences.numel() > 0:
        # Process exceeded sequences with minimal loops
        for batch_idx in exceeded_sequences:
            excess = (final_counts[batch_idx] - target_tensor).item()
            if excess > 0:
                # Find masked positions (keep on GPU)
                seq_mask = final_mask[batch_idx]
                masked_positions = torch.where(seq_mask)[0]
                
                # Randomly select positions to unmask (single GPU operation)
                perm_indices = torch.randperm(masked_positions.size(0), device=device)[:excess]
                positions_to_unmask = masked_positions[perm_indices]
                
                # Restore original tokens (vectorized)
                masked_x[batch_idx, positions_to_unmask] = x[batch_idx, positions_to_unmask]
    
    # Return final mask state
    final_mask = (masked_x == mask_token_id)
    return masked_x, final_mask





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

    # Use target-driven sticky masking to get corruption patterns
    # Convert old sticky parameters to target-driven approach
    target_ratio = min(0.7, sticky_rounds * 0.15)  # Rough conversion from rounds to ratio
    p1_prob = min(0.8, sticky_p1_p2_multiplier / sticky_p1_divisor)
    p2_prob = min(0.3, sticky_p1_p2_multiplier * 0.1)
    _, mask = apply_target_driven_sticky_masking_gpu(x, target_ratio, p1_prob, p2_prob, mask_token_id)
    
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
    """Stage-based unmasking with target-driven sticky masking"""
    global _val_batch_cache, _progressive_val_cache, _data_cache, _valid_indices_cache, _unmasking_val_set

    # For validation, use the pre-created validation set distributed across all stages
    if split == 'val':
        return get_unmasking_validation_batch(ctx, validation_sample_idx)

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

        # VECTORIZED DATA LOADING - Use advanced indexing for parallel loading
        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
        x_np = data[ix_expanded].astype(np.int64)
    
    # Single GPU transfer with pinned memory
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # For validation: create samples from all stages for comprehensive evaluation
    if split == 'val':
        torch.manual_seed(42 + (validation_sample_idx or 0))
        
        # Distribute validation samples across all stages
        if ctx.unmasking_stages and len(ctx.unmasking_stages) > 1:
            stage_idx = (validation_sample_idx or 0) % len(ctx.unmasking_stages)
            stage_config = ctx.unmasking_stages[stage_idx]
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id)
        else:
            # Fallback to current stage if no stages defined
            stage_config = ctx.get_current_stage_config()
            if stage_config is None:
                raise ValueError(f"No stage configuration available for {ctx.training_type} training")
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id)
    else:
        # For training: distribute batch samples across all stages up to current stage (inclusive)
        if ctx.unmasking_stages and ctx.current_stage >= 0:
            # Get all available stages up to current stage
            available_stages = ctx.unmasking_stages[:ctx.current_stage + 1]
            num_stages = len(available_stages)
            
            # Distribute batch_size samples evenly across all available stages
            samples_per_stage = ctx.batch_size // num_stages
            remainder = ctx.batch_size % num_stages
            
            # Create mixed batch with samples from all stages
            all_masked_x = []
            all_masks = []
            
            start_idx = 0
            for stage_idx, stage_config in enumerate(available_stages):
                # Determine number of samples for this stage
                stage_samples = samples_per_stage + (1 if stage_idx < remainder else 0)
                
                if stage_samples > 0:
                    # Get DIFFERENT subset of data for this stage (FIX: was x[:stage_samples])
                    stage_x = x[start_idx:start_idx + stage_samples]
                    start_idx += stage_samples
                    
                    # Apply masking for this stage
                    stage_masked_x, stage_mask = apply_stage_masking(stage_x, stage_config, ctx.mask_token_id)
                    
                    all_masked_x.append(stage_masked_x)
                    all_masks.append(stage_mask)
            
            # Concatenate all stages back into batch
            masked_x = torch.cat(all_masked_x, dim=0)
            mask = torch.cat(all_masks, dim=0)
            
            # Shuffle the batch to mix stages randomly
            perm = torch.randperm(masked_x.size(0))
            masked_x = masked_x[perm]
            mask = mask[perm]
            x = x[perm]  # Also permute original x to match
            
            # Log stage distribution occasionally for training only
            if split == 'train' and ctx.iter_num % 1500 == 0 and ctx.iter_num > 0:
                stage_counts = [samples_per_stage + (1 if i < remainder else 0) for i in range(num_stages)]
                print(f"Training iter {ctx.iter_num}: Mixed batch from {num_stages} stages: {stage_counts}")
        else:
            # Fallback to current stage configuration
            stage_config = ctx.get_current_stage_config()
            if stage_config is None:
                raise ValueError(f"No stage configuration available for {ctx.training_type} training")
            
            # Apply single stage masking (fallback case)
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id)
    
    if split == 'val':
        torch.manual_seed(1337 + ctx.seed_offset)

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

def get_batch_remasking(split, ctx: TrainingContext, validation_sample_idx=None):
    """Ultra-fast remasking implementation with aggressive caching + prefetching"""
    global _val_batch_cache, _progressive_val_cache, _data_cache, _valid_indices_cache

    # For validation with progressive sampling
    if split == 'val' and validation_sample_idx is not None:
        cache_key = f"remasking_{validation_sample_idx}"
        if cache_key in _progressive_val_cache:
            return _progressive_val_cache[cache_key]
    # For validation, use legacy cached batch for backward compatibility
    elif split == 'val' and _val_batch_cache is not None:
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

        # VECTORIZED DATA LOADING - Use advanced indexing for parallel loading
        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
        x_np = data[ix_expanded].astype(np.int64)
    
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

        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (batch_size, block_size)
        x_np = data[ix_expanded].astype(np.int64)
    
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
    
    # Add current stage information for unmasking training
    if training_ctx.training_type == 'unmasking':
        stage_config = training_ctx.get_current_stage_config()
        if stage_config:
            out['current_stage'] = training_ctx.current_stage
            stage_type = stage_config.get_stage_type()
            out['stage_type'] = stage_type.value
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                out['target_masked_ratio'] = config.target_masked_ratio
                out['p1_probability'] = config.p1_probability
                out['p2_probability'] = config.p2_probability
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                out['max_masked_ratio'] = config.max_masked_ratio
            out['val_loss_stale_count'] = training_ctx.val_loss_stale_count
    
    for split in ['train', 'val']:
        losses = torch.zeros(training_ctx.eval_iters)
        # Track masked token ratios for all splits
        masked_token_ratios = []

        if split == 'val':
            # For validation, also track model vs random performance
            model_probs = []
            # Track most likely predictions for accuracy calculation
            most_likely_correct = []
            # For binary classification and remasking, track corruption statistics
            if training_ctx.training_type in ['remasking_binary', 'remasking']:
                total_positions = 0
                corrupted_positions = 0
            else:
                random_prob = 1.0 / training_ctx.extended_vocab_size  # Random chance probability

        # For unmasking, use pre-created validation set with samples from all stages
        # Track per-stage losses for detailed analysis
        stage_losses = {}
        stage_sample_counts = {}
        if split == 'val' and training_ctx.training_type == 'unmasking':
            print(f"Using validation set with samples from all {len(training_ctx.unmasking_stages)} stages")
            # Initialize per-stage tracking
            for stage_idx in range(len(training_ctx.unmasking_stages)):
                stage_losses[stage_idx] = []
                stage_sample_counts[stage_idx] = 0
        
        for k in range(training_ctx.eval_iters):
            with timer.time_function('validation_data_generation'):
                if split == 'val' and training_ctx.training_type == 'unmasking':
                    # Use pre-created validation set with batch index
                    X, Y, mask = get_batch(split, training_ctx, validation_sample_idx=k)
                    # Determine which stage this batch belongs to based on validation set structure
                    # The validation set distributes samples evenly across stages
                    total_samples = training_ctx.eval_iters * training_ctx.batch_size
                    num_stages = len(training_ctx.unmasking_stages)
                    samples_per_stage = total_samples // num_stages
                    current_sample_idx = k * training_ctx.batch_size
                    current_stage_idx = min(current_sample_idx // samples_per_stage, num_stages - 1)
                else:
                    X, Y, mask = get_batch(split, training_ctx)
                    current_stage_idx = None

            # Calculate masked token ratio for this batch
            # Get per-sample ratios to capture the full range of masking rates
            sample_ratios = mask.float().mean(dim=1)  # Shape: (batch_size,) - ratio per sample
            masked_token_ratios.extend(sample_ratios.cpu().tolist())  # Add all individual sample ratios

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
                    
                    # Calculate most likely predictions (argmax)
                    predictions = torch.argmax(probs, dim=-1)  # (batch_size, seq_len)
                    predictions_flat = predictions.view(-1)  # (batch_size * seq_len,)

                    if training_ctx.training_type == 'remasking_binary':
                        # For binary classification, compute accuracy on all positions
                        # Track corruption statistics for proper baseline
                        total_positions += targets_flat.numel()
                        corrupted_positions += (targets_flat == training_ctx.remask_wrong_id).sum().item()

                        # Get probabilities for correct binary classification
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                        
                        # Track most likely prediction accuracy for all positions
                        correct_predictions = (predictions_flat == targets_flat).cpu().tolist()
                        most_likely_correct.extend(correct_predictions)
                    elif training_ctx.training_type == 'remasking':
                        # For remasking, compute accuracy on ALL positions (corrupted + uncorrupted)
                        # Track corruption statistics for proper baseline
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        total_positions += targets_flat.numel()
                        corrupted_positions += mask_flat.sum().item()  # mask indicates corrupted positions

                        # Get probabilities for correct predictions at ALL positions
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                        
                        # Track most likely prediction accuracy for all positions
                        correct_predictions = (predictions_flat == targets_flat).cpu().tolist()
                        most_likely_correct.extend(correct_predictions)
                    else:
                        # For unmasking, compute on masked positions only
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        masked_positions = mask_flat.bool()
                        if masked_positions.sum() > 0:  # Only if there are masked tokens
                            correct_token_probs = probs_flat[masked_positions, targets_flat[masked_positions]]
                            model_probs.extend(correct_token_probs.cpu().tolist())
                            
                            # Track most likely prediction accuracy for masked positions only
                            correct_predictions = (predictions_flat[masked_positions] == targets_flat[masked_positions]).cpu().tolist()
                            most_likely_correct.extend(correct_predictions)

            losses[k] = loss.item()
            
            # Track per-stage losses for unmasking validation
            if split == 'val' and training_ctx.training_type == 'unmasking' and current_stage_idx is not None:
                stage_losses[current_stage_idx].append(loss.item())
                stage_sample_counts[current_stage_idx] += X.size(0)  # Add batch size

        out[split] = losses.mean()
        
        # Add per-stage validation losses for unmasking
        if split == 'val' and training_ctx.training_type == 'unmasking' and stage_losses:
            for stage_idx, stage_loss_list in stage_losses.items():
                if stage_loss_list:  # Only if we have samples for this stage
                    avg_stage_loss = sum(stage_loss_list) / len(stage_loss_list)
                    out[f'val_stage_{stage_idx}_loss'] = avg_stage_loss
                    out[f'val_stage_{stage_idx}_samples'] = stage_sample_counts[stage_idx]
        
        if split == 'val':
            total_samples = training_ctx.eval_iters * training_ctx.batch_size
            print(f"  Validation complete: {training_ctx.eval_iters} batches processed ({total_samples} samples), avg loss = {out[split]:.4f}")
            
            # Print per-stage validation losses for unmasking
            if training_ctx.training_type == 'unmasking' and stage_losses:
                print("  Per-stage validation losses:")
                for stage_idx in range(len(training_ctx.unmasking_stages)):
                    if stage_idx in stage_losses and stage_losses[stage_idx]:
                        stage_config = training_ctx.unmasking_stages[stage_idx]
                        stage_type = stage_config.get_stage_type()
                        avg_loss = sum(stage_losses[stage_idx]) / len(stage_losses[stage_idx])
                        sample_count = stage_sample_counts[stage_idx]
                        
                        stage_info = f"    Stage {stage_idx} ({stage_type.value}): {avg_loss:.4f} ({sample_count} samples)"
                        if stage_type == UnmaskingStageType.STICKY:
                            config = stage_config.config
                            stage_info += f" - ratio={config.target_masked_ratio:.1f}"
                        elif stage_type == UnmaskingStageType.RANDOM:
                            config = stage_config.config
                            stage_info += f" - max_ratio={config.max_masked_ratio:.1f}"
                        print(stage_info)

        # Add masked token ratio statistics
        if masked_token_ratios:
            avg_masked_ratio = sum(masked_token_ratios) / len(masked_token_ratios)
            out[f'{split}_masked_token_ratio'] = avg_masked_ratio
            out[f'{split}_min_masked_token_ratio'] = min(masked_token_ratios)
            out[f'{split}_max_masked_token_ratio'] = max(masked_token_ratios)

        # Add model vs random comparison for validation
        if split == 'val' and model_probs:
            # Calculate most likely prediction accuracy percentage
            if most_likely_correct:
                most_likely_accuracy = (sum(most_likely_correct) / len(most_likely_correct)) * 100.0
                out[f'{split}_most_likely_accuracy'] = most_likely_accuracy
            
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

def update_stage_progress(training_ctx: TrainingContext, val_loss: float):
    """
    Update stage progress for unmasking training based on validation loss.
    Returns True if stage was advanced, False otherwise.
    """
    if training_ctx.training_type != 'unmasking':
        return False
    
    stage_config = training_ctx.get_current_stage_config()
    if stage_config is None:
        return False
    
    # Check if validation loss improved
    if val_loss < training_ctx.best_val_loss_this_stage:
        training_ctx.best_val_loss_this_stage = val_loss
        training_ctx.val_loss_stale_count = 0
        print(f"  Stage {training_ctx.current_stage}: New best val loss {val_loss:.4f}, reset stale count to 0")
        return False
    else:
        training_ctx.val_loss_stale_count += 1
        print(f"  Stage {training_ctx.current_stage}: Val loss stale count {training_ctx.val_loss_stale_count}/{stage_config.get_val_loss_stale_count()}")
        
        # Check if we should advance to next stage
        if training_ctx.val_loss_stale_count >= stage_config.get_val_loss_stale_count():
            advanced = training_ctx.advance_stage()
            if advanced:
                new_stage_config = training_ctx.get_current_stage_config()
                stage_type = new_stage_config.get_stage_type()
                print(f"\n*** ADVANCING TO STAGE {training_ctx.current_stage} ({stage_type.value}) ***")
                if stage_type == UnmaskingStageType.STICKY:
                    config = new_stage_config.config
                    print(f"  Target masked ratio: {config.target_masked_ratio}")
                    print(f"  P1 probability: {config.p1_probability}")
                    print(f"  P2 probability: {config.p2_probability}")
                elif stage_type == UnmaskingStageType.RANDOM:
                    config = new_stage_config.config
                    print(f"  Max masked ratio: {config.max_masked_ratio}")
                print(f"  Val loss stale count limit: {new_stage_config.get_val_loss_stale_count()}")
                print("*** STAGE ADVANCEMENT COMPLETE ***\n")
                return True
            else:
                print(f"  Stage {training_ctx.current_stage}: Reached final stage, continuing training")
                return False
        
        return False

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