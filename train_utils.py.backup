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
_remasking_val_set = None  # Complete validation set for remasking training (eval_iters * batch_size samples)
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
    vocab_size: int = None  # extended_vocab_size (meta_vocab_size + 15 reserved special tokens)
    use_paragraph_boundaries: bool = True  # If True, start samples at paragraph boundaries (double newlines)
    use_all_stages_for_training: bool = False  # If True, generate training batches from all stages like validation
    weight_loss_by_mask_ratio: bool = False  # If True, weight loss by sqrt(1.0 / mask_ratio) to balance gradient magnitude
    
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
    validation_stages: list = None  # Separate stages for validation set creation
    
    # Remasking strategy - only random supported
    
    # Remasking corruption parameters
    # Random corruption parameters
    guaranteed_unmasked_max: float = 0.95
    guaranteed_unmasked_min: float = 0.1
    sticky_transition_start: int = 1000
    sticky_transition_end: int = 6000
    random_mask_warmup: int = 1000
    
    # Sticky corruption parameters
    sticky_rounds: int = 4
    sticky_p1_p2_multiplier: float = 1.2
    sticky_p1_divisor: float = 2.0
    p1_p2_ratio: float = 1.0  # For remasking sticky masking: if 1.0 use random, else use sticky
    
    # Evaluation parameters
    eval_iters: int = 20
    
    # Learning rate parameters
    warmup_iters: int = 2000
    lr_decay_iters: int = 8000
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    
    # Entropy penalty parameters (incentivize uniform wrong answer distributions)
    enable_entropy_penalty: bool = False
    max_entropy_penalty: float = 0.5  # Penalty for concentrated wrong answers
    entropy_penalty_start_iter: int = 6000
    
    # Entropy multiplier tracking
    entropy_multiplier_ema: float = 1.0  # Exponential moving average of entropy multiplier
    entropy_multiplier_ema_factor: float = 0.99  # EMA decay factor
    
    # Label smoothing parameters
    uncertainty_factor: float = 0.0  # Label smoothing factor: 0 = no smoothing, >0 = apply smoothing
    
    def __post_init__(self):
        # Default unmasking stages if not provided
        if self.training_type == 'unmasking' and self.unmasking_stages is None:
            self.unmasking_stages = [
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.2, p1_probability=0.3, p2_probability=0.0, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.4, p1_probability=0.2, p2_probability=0.8, val_loss_stale_count=5)),
                UnmaskingStage(StickyStageConfig(target_masked_ratio=0.6, p1_probability=0.1, p2_probability=0.9, val_loss_stale_count=10)),
            ]
        
        # Default validation stages if not provided (same as training stages by default)
        if self.training_type == 'unmasking' and self.validation_stages is None:
            self.validation_stages = self.unmasking_stages
        
        # Keep both meta_vocab_size (for random token generation) and extended_vocab_size (for model)
        # vocab_size is kept for backward compatibility but should equal extended_vocab_size
        self.vocab_size = self.extended_vocab_size
    
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
    global _progressive_val_cache, _val_batch_cache, _progressive_val_full_cache, _unmasking_val_set, _remasking_val_set
    _progressive_val_cache.clear()
    _val_batch_cache = None
    _progressive_val_full_cache = None
    _unmasking_val_set = None
    _remasking_val_set = None

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
        if ctx.use_paragraph_boundaries:
            _valid_indices_cache['val'] = find_double_newline_indices(_data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
        else:
            _valid_indices_cache['val'] = np.array([])
    
    data = _data_cache['val']
    valid_indices = _valid_indices_cache['val']
    
    total_samples = ctx.eval_iters * ctx.batch_size
    num_stages = len(ctx.validation_stages)
    samples_per_stage = total_samples // num_stages
    
    validation_batches = []
    
    # Generate samples for each stage
    torch.manual_seed(42)  # Fixed seed for reproducible validation set
    
    for stage_idx, stage_config in enumerate(ctx.validation_stages):
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
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
            
            # Apply label smoothing to targets if enabled
            y = x.clone()
            if ctx.uncertainty_factor > 0.0:
                # Determine special token IDs to exclude from smoothing
                special_token_ids = []
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
            
            stage_batches.append((masked_x.clone(), y.clone(), mask.clone()))
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

def get_unmasking_training_batch_all_stages(ctx: TrainingContext):
    """Generate fresh training batch with samples distributed across all stages"""
    global _data_cache, _valid_indices_cache
    
    # Cache training data if not already cached
    if _data_cache['train'] is None:
        _data_cache['train'] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            _valid_indices_cache['train'] = find_double_newline_indices(_data_cache['train'], ctx.meta_vocab_size, ctx.block_size)
        else:
            _valid_indices_cache['train'] = np.array([])
    
    data = _data_cache['train']
    valid_indices = _valid_indices_cache['train']
    
    num_stages = len(ctx.unmasking_stages)
    samples_per_stage = ctx.batch_size // num_stages
    remainder = ctx.batch_size % num_stages
    
    # Create mixed batch with samples from all stages
    all_masked_x = []
    all_x = []
    all_masks = []
    
    for stage_idx, stage_config in enumerate(ctx.unmasking_stages):
        # Determine number of samples for this stage
        stage_samples = samples_per_stage + (1 if stage_idx < remainder else 0)
        
        if stage_samples > 0:
            # Sample data indices for this stage
            if len(valid_indices) == 0:
                ix_np = torch.randint(len(data) - ctx.block_size, (stage_samples,)).numpy()
            else:
                ix_indices = torch.randint(len(valid_indices), (stage_samples,)).numpy()
                ix_np = valid_indices[ix_indices]
            
            # Load data with vectorized indexing
            ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]  # (stage_samples, block_size)
            x_np = data[ix_expanded].astype(np.int64)
            
            # Convert to tensor
            x = torch.from_numpy(x_np)
            if ctx.device_type == 'cuda':
                x = x.pin_memory().to(ctx.device, non_blocking=True)
            else:
                x = x.to(ctx.device)
            
            # Apply masking for this stage
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
            
            # Apply label smoothing to targets if enabled
            y = x.clone()
            if ctx.uncertainty_factor > 0.0:
                # Determine special token IDs to exclude from smoothing
                special_token_ids = []
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
            
            all_masked_x.append(masked_x)
            all_x.append(y)  # Use smoothed targets
            all_masks.append(mask)
    
    # Concatenate all stages back into batch
    final_masked_x = torch.cat(all_masked_x, dim=0)
    final_x = torch.cat(all_x, dim=0)
    final_mask = torch.cat(all_masks, dim=0)
    
    # Shuffle the batch to mix stages randomly
    perm = torch.randperm(final_masked_x.size(0))
    final_masked_x = final_masked_x[perm]
    final_x = final_x[perm]
    final_mask = final_mask[perm]
    
    return final_masked_x, final_x, final_mask

def create_remasking_validation_set(ctx: TrainingContext, force_recreate=False):
    """Create complete validation set for remasking training with progressive corruption intensities"""
    global _remasking_val_set, _data_cache, _valid_indices_cache
    
    if _remasking_val_set is not None and not force_recreate:
        print("Using existing remasking validation set from cache")
        return  # Already created
    
    if force_recreate:
        print("Force recreating remasking validation set...")
        _remasking_val_set = None
    
    print("Creating remasking validation set with progressive corruption intensities...")
    
    # Cache validation data if not already cached
    if _data_cache['val'] is None:
        _data_cache['val'] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        if ctx.use_paragraph_boundaries:
            _valid_indices_cache['val'] = find_double_newline_indices(_data_cache['val'], ctx.meta_vocab_size, ctx.block_size)
        else:
            _valid_indices_cache['val'] = np.array([])
    
    data = _data_cache['val']
    valid_indices = _valid_indices_cache['val']
    
    total_samples = ctx.eval_iters * ctx.batch_size
    validation_batches = []
    
    # Use fixed seed for reproducible validation set
    torch.manual_seed(42)  
    
    # Use a fixed mid-training corruption level for consistent validation
    # This represents a reasonable difficulty level for evaluation
    fixed_validation_iter = ctx.max_iters // 2  # Mid-training corruption level
    
    # Calculate what the corruption rate will be for validation
    corruption_min = 1.0 - ctx.guaranteed_unmasked_max
    corruption_max = 1.0 - ctx.guaranteed_unmasked_min
    if fixed_validation_iter < ctx.random_mask_warmup:
        progress = fixed_validation_iter / ctx.random_mask_warmup
        val_corruption_rate = corruption_min + progress * (corruption_max - corruption_min)
    else:
        val_corruption_rate = corruption_max
    
    print(f"  Using fixed validation corruption level: iter {fixed_validation_iter} = {val_corruption_rate:.1%} corruption")
    
    for k in range(ctx.eval_iters):
        # Sample data indices
        if len(valid_indices) == 0:
            ix_np = torch.randint(len(data) - ctx.block_size, (ctx.batch_size,)).numpy()
        else:
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]
        
        # Load data with vectorized indexing
        ix_expanded = ix_np[:, None] + np.arange(ctx.block_size)[None, :]
        x_np = data[ix_expanded].astype(np.int64)
        
        # Convert to tensor
        x = torch.from_numpy(x_np)
        if ctx.device_type == 'cuda':
            x = x.pin_memory().to(ctx.device, non_blocking=True)
        else:
            x = x.to(ctx.device)
        
        # Apply corruption using the fixed validation iteration (disable debug to avoid spam)
        corrupted_x, mask = apply_corruption_gpu(x, fixed_validation_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                                ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, 
                                                ctx.random_mask_warmup, ctx.p1_p2_ratio, debug=False)
        
        if ctx.training_type == 'remasking_binary':
            # Binary targets: 0=keep, 1=remask
            y = torch.full_like(x, ctx.remask_good_id)
            y[mask] = ctx.remask_wrong_id
        else:  # remasking
            # Target: original tokens at correct positions, wrong_token_id at corrupted positions
            y = x.clone()
            y[mask] = ctx.wrong_token_id
        
        # Apply label smoothing if enabled
        if ctx.uncertainty_factor > 0.0:
            if ctx.training_type == 'remasking_binary':
                # For binary classification, smooth over the 2 classes
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          device=ctx.device)
            else:  # remasking
                # For remasking, exclude special tokens from smoothing
                special_token_ids = []
                if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.wrong_token_id)
                if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.mask_token_id)
                if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_good_id)
                if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
                    special_token_ids.append(ctx.remask_wrong_id)
                
                y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                          special_token_ids=special_token_ids, device=ctx.device)
        
        validation_batches.append((corrupted_x.clone(), y.clone(), mask.clone()))
    
    torch.manual_seed(1337 + ctx.seed_offset)  # Reset seed
    _remasking_val_set = validation_batches
    print(f"Remasking validation set created: {len(validation_batches)} batches, {sum(b[0].size(0) for b in validation_batches)} total samples")

def get_remasking_validation_batch(ctx: TrainingContext, batch_idx=None):
    """Get a specific batch from the pre-created remasking validation set"""
    global _remasking_val_set
    
    if _remasking_val_set is None:
        create_remasking_validation_set(ctx, force_recreate=False)
    
    if batch_idx is None:
        batch_idx = 0
    
    # Handle batch index wrapping
    batch_idx = batch_idx % len(_remasking_val_set)
    return _remasking_val_set[batch_idx]


# In train_utils.py, modify this function
def apply_random_masking_gpu(x, max_masked_ratio, mask_token_id, meta_vocab_size):
    """
    GPU-optimized random masking for unmasking training.
    Each sample in the batch gets a different random masking probability.
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    mask_probs = torch.rand(batch_size, device=device) * max_masked_ratio
    rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
    mask_probs_expanded = mask_probs.unsqueeze(1).expand(-1, seq_len)
    
    # Step 1: Generate the boolean mask of positions to predict (this logic is unchanged)
    mask = rand_vals < mask_probs_expanded
    
    # Step 2: Apply the 80/10/10 corruption using the new function
    # NOTE: We use meta_vocab_size to avoid generating special tokens randomly
    corrupted_x = apply_bert_style_corruption_gpu(x, mask, mask_token_id, meta_vocab_size)
    
    return corrupted_x, mask

def apply_stage_masking(x, stage_config: UnmaskingStage, mask_token_id, meta_vocab_size):
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: UnmaskingStage configuration
        mask_token_id: Token ID to use for masking
        meta_vocab_size: Size of original vocabulary (for random token generation, excluding special tokens)
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config.get_stage_type()
    
    if stage_type == UnmaskingStageType.RANDOM:
        config = stage_config.config
        return apply_random_masking_gpu(x, config.max_masked_ratio, mask_token_id, meta_vocab_size)
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

# Deprecated synthetic corruption removed - only random corruption supported

def apply_sticky_corruption_gpu(x, target_masked_ratio, p1_probability, p2_probability, meta_vocab_size, debug=True):
    """Sticky corruption for remasking training with target-driven masking"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate target number of masked tokens per sequence
    target_masked_count = int(target_masked_ratio * seq_len)
    
    if target_masked_count == 0:
        # No masking needed - return original unchanged
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Start with original text
    corrupted_x = x.clone()
    
    # Pre-allocate tensors to avoid repeated allocations
    current_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    neighbor_masked = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Continue masking until we reach the target for each sequence
    max_rounds = min(1000, target_masked_count * 10)  # Adaptive safety limit
    target_tensor = torch.tensor(target_masked_count, device=device, dtype=torch.long)
    
    for round_idx in range(max_rounds):
        # Update current mask state (positions that are corrupted)
        current_mask = (corrupted_x != x)  # True where tokens have been corrupted
        
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
        
        # Apply corruption to newly masked positions (vectorized)
        if new_masks.any():
            # Replace with random tokens from vocabulary
            random_tokens = torch.randint(0, meta_vocab_size, new_masks.sum().shape, device=device)
            corrupted_x[new_masks] = random_tokens
    
    # Final adjustment: remove excess corruptions with fully vectorized approach
    final_mask = (corrupted_x != x)
    final_counts = final_mask.sum(dim=1)  # (batch_size,)
    
    # Only process sequences that exceeded target (minimize CPU-GPU sync)
    exceeded_sequences = torch.where(final_counts > target_tensor)[0]
    
    if exceeded_sequences.numel() > 0:
        # Process exceeded sequences with minimal loops
        for batch_idx in exceeded_sequences:
            excess = (final_counts[batch_idx] - target_tensor).item()
            if excess > 0:
                # Find corrupted positions (keep on GPU)
                seq_mask = final_mask[batch_idx]
                corrupted_positions = torch.where(seq_mask)[0]
                
                # Randomly select positions to restore (single GPU operation)
                perm_indices = torch.randperm(corrupted_positions.size(0), device=device)[:excess]
                positions_to_restore = corrupted_positions[perm_indices]
                
                # Restore original tokens (vectorized)
                corrupted_x[batch_idx, positions_to_restore] = x[batch_idx, positions_to_restore]
    
    # Return final mask state
    final_mask = (corrupted_x != x)
    return corrupted_x, final_mask

def apply_random_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, sticky_transition_start, sticky_transition_end, meta_vocab_size, random_mask_warmup, debug=True):
    """Random corruption for remasking training with iteration-based masking probability"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # FIXED: Convert "unmasked" parameters to actual corruption probabilities
    # guaranteed_unmasked_max=0.9 means 90% unmasked -> 10% corrupted
    # guaranteed_unmasked_min=0.6 means 60% unmasked -> 40% corrupted
    corruption_min = 1.0 - guaranteed_unmasked_max  # Start: 10% corruption
    corruption_max = 1.0 - guaranteed_unmasked_min  # End: 40% corruption
    
    # Calculate masking probability based on iteration
    if iter_num < random_mask_warmup:
        # During warmup, gradually increase corruption from min to max
        progress = iter_num / random_mask_warmup
        mask_prob = corruption_min + progress * (corruption_max - corruption_min)
    elif iter_num < sticky_transition_start:
        mask_prob = corruption_max  # Maximum corruption (40%)
    elif iter_num < sticky_transition_end:
        # Keep maximum corruption during transition
        mask_prob = corruption_max
    else:
        mask_prob = corruption_max  # Stay at maximum corruption
    
    # Apply random masking
    rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
    mask = rand_vals < mask_prob
    
    # Debug: Print corruption rate occasionally during training only (not validation set creation)
    if debug and iter_num % 1000 == 0 and iter_num > 0:
        actual_mask_ratio = mask.float().mean().item()
        print(f"DEBUG: iter {iter_num}, target_corruption={mask_prob:.3f} ({mask_prob*100:.1f}%), actual={actual_mask_ratio:.3f} ({actual_mask_ratio*100:.1f}%)")
        print(f"  corruption_min={corruption_min:.3f}, corruption_max={corruption_max:.3f}, warmup={random_mask_warmup}")
    
    # Create corrupted version by randomly replacing masked tokens
    corrupted_x = x.clone()
    if mask.any():
        # Replace with random tokens from vocabulary
        random_tokens = torch.randint(0, meta_vocab_size, mask.sum().shape, device=device)
        corrupted_x[mask] = random_tokens
    
    return corrupted_x, mask

def apply_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, sticky_transition_start, sticky_transition_end, meta_vocab_size, random_mask_warmup, p1_p2_ratio=1.0, debug=True):
    """Unified corruption function that chooses between random and sticky masking based on p1_p2_ratio"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate current corruption probability based on iteration
    corruption_min = 1.0 - guaranteed_unmasked_max  # Start: 10% corruption
    corruption_max = 1.0 - guaranteed_unmasked_min  # End: 40% corruption
    
    if iter_num < random_mask_warmup:
        # During warmup, gradually increase corruption from min to max
        progress = iter_num / random_mask_warmup
        target_corruption_rate = corruption_min + progress * (corruption_max - corruption_min)
    elif iter_num < sticky_transition_start:
        target_corruption_rate = corruption_max  # Maximum corruption
    elif iter_num < sticky_transition_end:
        target_corruption_rate = corruption_max  # Keep maximum corruption during transition
    else:
        target_corruption_rate = corruption_max  # Stay at maximum corruption
    
    # Choose masking strategy based on p1_p2_ratio
    if p1_p2_ratio == 1.0:
        # Use random corruption
        return apply_random_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, 
                                         sticky_transition_start, sticky_transition_end, meta_vocab_size, 
                                         random_mask_warmup, debug=debug)
    else:
        # Use sticky corruption
        # Calculate p1 and p2 based on ratio, with max(p1, p2) = target_corruption_rate / 4
        max_prob = target_corruption_rate / 4.0
        
        if p1_p2_ratio > 1.0:
            # p1 is larger
            p1_probability = max_prob
            p2_probability = max_prob / p1_p2_ratio
        else:
            # p2 is larger
            p2_probability = max_prob
            p1_probability = max_prob * p1_p2_ratio
        
        if debug and iter_num % 1000 == 0 and iter_num > 0:
            print(f"DEBUG: iter {iter_num}, sticky masking: target_corruption={target_corruption_rate:.3f} ({target_corruption_rate*100:.1f}%)")
            print(f"  p1_p2_ratio={p1_p2_ratio:.3f}, p1={p1_probability:.3f}, p2={p2_probability:.3f}")
        
        return apply_sticky_corruption_gpu(x, target_corruption_rate, p1_probability, p2_probability, meta_vocab_size, debug=debug)

# Deprecated sticky corruption removed - random and new sticky corruption now supported

# Deprecated fragment corruption removed - only random corruption supported

def get_progressive_validation_iterations(eval_iters, max_iters):
    """Generate validation iterations for progressive validation"""
    # Create a range of iterations from early to late training
    iterations = []
    for i in range(eval_iters):
        progress = i / (eval_iters - 1) if eval_iters > 1 else 0
        iter_val = int(progress * max_iters)
        iterations.append(iter_val)
    return iterations

def get_batch(split, ctx: TrainingContext, validation_sample_idx=None):
    """Main batch generation function that delegates to specific training type functions"""
    if ctx.training_type == 'unmasking':
        return get_batch_unmasking(split, ctx, validation_sample_idx)
    elif ctx.training_type == 'remasking_binary':
        return get_batch_remasking_binary(split, ctx, validation_sample_idx)
    else:
        raise ValueError(f"Unsupported training type: {ctx.training_type}")

def get_batch_unmasking(split, ctx: TrainingContext, validation_sample_idx=None):
    """Stage-based unmasking with target-driven sticky masking"""
    global _val_batch_cache, _progressive_val_cache, _data_cache, _valid_indices_cache, _unmasking_val_set

    # For validation, use the pre-created validation set distributed across all stages
    if split == 'val':
        return get_unmasking_validation_batch(ctx, validation_sample_idx)
    
    # For training, check if we should generate batch from all stages
    if split == 'train' and ctx.use_all_stages_for_training:
        return get_unmasking_training_batch_all_stages(ctx)

    # Cache memory-mapped data and valid indices - MAJOR SPEEDUP
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Cache the expensive valid indices computation
        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            _valid_indices_cache[split] = np.array([])  # Empty array indicates random sampling
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
        
        # Distribute validation samples across all validation stages
        if ctx.validation_stages and len(ctx.validation_stages) > 1:
            stage_idx = (validation_sample_idx or 0) % len(ctx.validation_stages)
            stage_config = ctx.validation_stages[stage_idx]
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
        else:
            # Fallback to current stage if no stages defined
            stage_config = ctx.get_current_stage_config()
            if stage_config is None:
                raise ValueError(f"No stage configuration available for {ctx.training_type} training")
            # Apply stage-specific masking
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
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
                    stage_masked_x, stage_mask = apply_stage_masking(stage_x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
                    
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
            masked_x, mask = apply_stage_masking(x, stage_config, ctx.mask_token_id, ctx.meta_vocab_size)
    
    if split == 'val':
        torch.manual_seed(1337 + ctx.seed_offset)

    # Target is original x
    y = x.clone()
    
    # Apply label smoothing if enabled
    if ctx.uncertainty_factor > 0.0:
        # Determine special token IDs to exclude from smoothing
        special_token_ids = []
        if ctx.mask_token_id is not None and ctx.mask_token_id < ctx.extended_vocab_size:
            special_token_ids.append(ctx.mask_token_id)
        if ctx.wrong_token_id is not None and ctx.wrong_token_id < ctx.extended_vocab_size:
            special_token_ids.append(ctx.wrong_token_id)
        if ctx.remask_good_id is not None and ctx.remask_good_id < ctx.extended_vocab_size:
            special_token_ids.append(ctx.remask_good_id)
        if ctx.remask_wrong_id is not None and ctx.remask_wrong_id < ctx.extended_vocab_size:
            special_token_ids.append(ctx.remask_wrong_id)
        
        y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                  special_token_ids=special_token_ids, device=ctx.device)

    # Cache validation batch for consistency
    if split == 'val':
        if validation_sample_idx is not None:
            cache_key = f"unmasking_{validation_sample_idx}"
            _progressive_val_cache[cache_key] = (masked_x, y, mask)
        else:
            _val_batch_cache = (masked_x, y, mask)

    return masked_x, y, mask

def get_batch_remasking_binary(split, ctx: TrainingContext, validation_sample_idx=None):
    """GPU-optimized remasking binary training: symmetric task with remask_good_id and remask_wrong_id targets"""
    # Use pre-created validation set for validation
    if split == 'val':
        return get_remasking_validation_batch(ctx, validation_sample_idx)
    
    # Training data generation - fast implementation 
    global _data_cache, _valid_indices_cache

    # Use same data caching and prefetching as remasking
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], ctx.meta_vocab_size, ctx.block_size)
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            _valid_indices_cache[split] = np.array([])  # Empty array indicates random sampling
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
    else:
        corruption_iter = ctx.iter_num

    # Use unified corruption function (random or sticky based on p1_p2_ratio)
    corrupted_x, mask = apply_corruption_gpu(x, corruption_iter, ctx.guaranteed_unmasked_max, ctx.guaranteed_unmasked_min,
                                           ctx.sticky_transition_start, ctx.sticky_transition_end, ctx.meta_vocab_size, 
                                           ctx.random_mask_warmup, ctx.p1_p2_ratio, debug=True)

    # Binary targets: remask_good_id for uncorrupted, remask_wrong_id for corrupted (already on GPU)
    y = torch.full_like(x, ctx.remask_good_id)
    y[mask] = ctx.remask_wrong_id
    
    # Apply label smoothing if enabled
    if ctx.uncertainty_factor > 0.0:
        # For binary classification, only smooth over the 2 classes (remask_good_id, remask_wrong_id)
        # Don't include other special tokens as they shouldn't appear in binary classification
        y = apply_label_smoothing(y, ctx.uncertainty_factor, ctx.extended_vocab_size, 
                                  device=ctx.device)

    # No caching needed for training - validation uses pre-created set
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
            # Track signal to noise ratio (correct prob vs most probable incorrect prob)
            signal_to_noise_ratios = []
            # Track detailed probability breakdown for binary classification by class
            right_probs_p0 = []  # Probabilities for correct predictions where target=0
            right_probs_p1 = []  # Probabilities for correct predictions where target=1
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
            print(f"Using validation set with samples from all {len(training_ctx.validation_stages)} stages")
            # Initialize per-stage tracking
            for stage_idx in range(len(training_ctx.validation_stages)):
                stage_losses[stage_idx] = []
                stage_sample_counts[stage_idx] = 0
        
        for k in range(training_ctx.eval_iters):
            with timer.time_function('validation_data_generation'):
                if split == 'val' and training_ctx.training_type == 'unmasking':
                    # Use pre-created validation set with batch index
                    X, Y, mask = get_batch(split, training_ctx, validation_sample_idx=k)
                    # Determine which stage this batch belongs to based on validation set structure
                    total_samples = training_ctx.eval_iters * training_ctx.batch_size
                    num_stages = len(training_ctx.validation_stages)
                    samples_per_stage = total_samples // num_stages
                    current_sample_idx = k * training_ctx.batch_size
                    current_stage_idx = min(current_sample_idx // samples_per_stage, num_stages - 1)
                elif split == 'val' and training_ctx.training_type in ['remasking_binary', 'remasking']:
                    # Fix: pass validation_sample_idx to get different validation batches
                    X, Y, mask = get_batch(split, training_ctx, validation_sample_idx=k)
                    current_stage_idx = None
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
                        # Cross-entropy handles both hard targets (indices) and soft targets (probabilities)
                        logits_reshaped = logits.view(-1, logits.size(-1))
                        mask_reshaped = mask.view(-1)
                        
                        if Y.dim() == 3:
                            # Soft targets (probability distributions)
                            targets_reshaped = Y.view(-1, Y.size(-1))
                            loss = torch.nn.functional.cross_entropy(
                                logits_reshaped[mask_reshaped],
                                targets_reshaped[mask_reshaped],
                                reduction='mean'
                            )
                        else:
                            # Hard targets (token indices)
                            targets_reshaped = Y.view(-1)
                            loss = torch.nn.functional.cross_entropy(
                                logits_reshaped[mask_reshaped],
                                targets_reshaped[mask_reshaped],
                                reduction='mean'
                            )
                        
                        # Apply mask ratio weighting if enabled (same as training)
                        if training_ctx.weight_loss_by_mask_ratio:
                            mask_ratio = mask.float().mean().item()
                            if mask_ratio > 0:
                                weight = (1.0 / mask_ratio) ** 0.5  # sqrt(1.0 / mask_ratio)
                                loss = loss * weight
                    # For remasking variants, model's internal loss is correct

                # For validation, compute model vs random statistics
                if split == 'val':
                    # Get probabilities from logits and flatten for statistics
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                    probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
                    
                    # Handle both hard and soft targets
                    if Y.dim() == 3:
                        # Soft targets - get the most likely class from probability distribution
                        targets_flat = torch.argmax(Y.view(-1, Y.size(-1)), dim=-1)  # (batch_size * seq_len,)
                    else:
                        # Hard targets
                        targets_flat = Y.view(-1)  # (batch_size * seq_len,)
                    
                    # Calculate most likely predictions (argmax)
                    predictions = torch.argmax(probs, dim=-1)  # (batch_size, seq_len)
                    predictions_flat = predictions.view(-1)  # (batch_size * seq_len,)

                    if training_ctx.training_type == 'remasking_binary':
                        # For binary classification, compute accuracy on all positions
                        # Track corruption statistics for proper baseline
                        total_positions += targets_flat.numel()
                        corrupted_positions += (targets_flat == training_ctx.remask_wrong_id).sum().item()
                        
                        # Track validation statistics for summary
                        if split == 'val':
                            # Initialize counters on first batch
                            if k == 0:
                                val_total_class_0, val_total_class_1 = 0, 0
                                val_pred_class_0, val_pred_class_1 = 0, 0
                                val_correct_pred_0, val_correct_pred_1 = 0, 0
                            
                            # Count actual class distribution
                            class_0_count = (targets_flat == 0).sum().item()
                            class_1_count = (targets_flat == 1).sum().item()
                            val_total_class_0 += class_0_count
                            val_total_class_1 += class_1_count
                            
                            # Count predictions
                            pred_0_count = (predictions_flat == 0).sum().item()
                            pred_1_count = (predictions_flat == 1).sum().item()
                            val_pred_class_0 += pred_0_count
                            val_pred_class_1 += pred_1_count
                            
                            # Count correct predictions by class
                            correct_0 = ((predictions_flat == 0) & (targets_flat == 0)).sum().item()
                            correct_1 = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
                            val_correct_pred_0 += correct_0
                            val_correct_pred_1 += correct_1

                        # Get probabilities for correct binary classification
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                        
                        # Calculate signal to noise ratio for binary classification
                        # For each position, find the most probable incorrect class
                        incorrect_probs = torch.where(targets_flat == 0, probs_flat[:, 1], probs_flat[:, 0])
                        # Calculate signal to noise ratio, capped at 100
                        sn_ratios = torch.clamp(correct_token_probs / (incorrect_probs + 1e-10), max=100.0)
                        signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                        
                        # Track detailed probability breakdown by class
                        class_0_mask = (targets_flat == 0)
                        class_1_mask = (targets_flat == 1)
                        
                        # Get probabilities for correct predictions by class
                        if class_0_mask.sum() > 0:
                            class_0_correct_probs = probs_flat[class_0_mask, 0]  # P(class=0) where target=0
                            right_probs_p0.extend(class_0_correct_probs.cpu().tolist())
                        
                        if class_1_mask.sum() > 0:
                            class_1_correct_probs = probs_flat[class_1_mask, 1]  # P(class=1) where target=1
                            right_probs_p1.extend(class_1_correct_probs.cpu().tolist())
                        
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
                        
                        # Calculate signal to noise ratio for remasking
                        # Create a copy of probabilities and zero out the correct class to find max incorrect
                        probs_masked = probs_flat.clone()
                        probs_masked[range(len(targets_flat)), targets_flat] = 0.0
                        max_incorrect_probs = torch.max(probs_masked, dim=1)[0]
                        # Calculate signal to noise ratio, capped at 100
                        sn_ratios = torch.clamp(correct_token_probs / (max_incorrect_probs + 1e-10), max=100.0)
                        signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                        
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
                            
                            # Calculate signal to noise ratio for unmasking (masked positions only)
                            masked_probs = probs_flat[masked_positions]
                            masked_targets = targets_flat[masked_positions]
                            # Create a copy and zero out correct probabilities to find max incorrect
                            probs_masked = masked_probs.clone()
                            probs_masked[range(len(masked_targets)), masked_targets] = 0.0
                            max_incorrect_probs = torch.max(probs_masked, dim=1)[0]
                            # Calculate signal to noise ratio, capped at 100
                            sn_ratios = torch.clamp(correct_token_probs / (max_incorrect_probs + 1e-10), max=100.0)
                            signal_to_noise_ratios.extend(sn_ratios.cpu().tolist())
                            
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
            
            # Print class distribution summary for binary classification
            if training_ctx.training_type == 'remasking_binary' and 'val_total_class_0' in locals():
                total_targets = val_total_class_0 + val_total_class_1
                total_preds = val_pred_class_0 + val_pred_class_1
                if total_targets > 0 and total_preds > 0:
                    # Class distribution
                    class_0_pct = (val_total_class_0 / total_targets) * 100
                    class_1_pct = (val_total_class_1 / total_targets) * 100
                    
                    # Prediction distribution (for display only)
                    pred_0_pct = (val_pred_class_0 / total_preds) * 100
                    pred_1_pct = (val_pred_class_1 / total_preds) * 100
                    
                    # Accuracy by class
                    acc_0 = (val_correct_pred_0 / val_total_class_0 * 100) if val_total_class_0 > 0 else 0
                    acc_1 = (val_correct_pred_1 / val_total_class_1 * 100) if val_total_class_1 > 0 else 0
                    
                    print(f"  Class distribution: no-mask {val_total_class_0} ({class_0_pct:.1f}%), mask {val_total_class_1} ({class_1_pct:.1f}%)")
                    print(f"  Model predictions: no-mask {val_pred_class_0} ({pred_0_pct:.1f}%), mask {val_pred_class_1} ({pred_1_pct:.1f}%)")
                    print(f"  Accuracy by class: no-mask {acc_0:.1f}%, mask {acc_1:.1f}%")
                    
                    # Print detailed probability breakdown if available
                    if f'{split}_avg_prob_right_p0' in out and f'{split}_avg_prob_right_p1' in out:
                        avg_p_right_p0 = out[f'{split}_avg_prob_right_p0']
                        avg_p_right_p1 = out[f'{split}_avg_prob_right_p1']
                        print(f"  Validation probabilities: avg_p_right_p0={avg_p_right_p0:.3f}, avg_p_right_p1={avg_p_right_p1:.3f}")
                    
                    # Add per-class accuracies and distributions to output for wandb logging
                    out[f'{split}_accuracy_no_mask'] = acc_0
                    out[f'{split}_accuracy_mask'] = acc_1
                    out[f'{split}_class_dist_no_mask'] = class_0_pct
                    out[f'{split}_class_dist_mask'] = class_1_pct
            
            # Print per-stage validation losses for unmasking
            if training_ctx.training_type == 'unmasking' and stage_losses:
                print("  Per-stage validation losses:")
                for stage_idx in range(len(training_ctx.validation_stages)):
                    if stage_idx in stage_losses and stage_losses[stage_idx]:
                        stage_config = training_ctx.validation_stages[stage_idx]
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
            # Add signal to noise ratio
            if signal_to_noise_ratios:
                finite_sn_ratios = [r for r in signal_to_noise_ratios if math.isfinite(r)]
                if finite_sn_ratios:
                    avg_signal_to_noise = sum(finite_sn_ratios) / len(finite_sn_ratios)
                    out[f'{split}_signal_to_noise'] = avg_signal_to_noise
                    # Add median signal to noise ratio
                    finite_sn_ratios_sorted = sorted(finite_sn_ratios)
                    n = len(finite_sn_ratios_sorted)
                    if n % 2 == 0:
                        median_sn = (finite_sn_ratios_sorted[n//2 - 1] + finite_sn_ratios_sorted[n//2]) / 2.0
                    else:
                        median_sn = finite_sn_ratios_sorted[n//2]
                    out[f'{split}_signal_to_noise_median'] = median_sn
                else:
                    out[f'{split}_signal_to_noise'] = float('nan')
                    out[f'{split}_signal_to_noise_median'] = float('nan')
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
                    
                    # Calculate detailed probability breakdown by class
                    if right_probs_p0 or right_probs_p1:
                        finite_right_p0 = [p for p in right_probs_p0 if math.isfinite(p)]
                        finite_right_p1 = [p for p in right_probs_p1 if math.isfinite(p)]
                        
                        avg_p_right_p0 = sum(finite_right_p0) / len(finite_right_p0) if finite_right_p0 else 0.0
                        avg_p_right_p1 = sum(finite_right_p1) / len(finite_right_p1) if finite_right_p1 else 0.0
                        
                        out[f'{split}_avg_prob_right_p0'] = avg_p_right_p0
                        out[f'{split}_avg_prob_right_p1'] = avg_p_right_p1
                elif training_ctx.training_type == 'unmasking':
                    # For unmasking, use uniform random baseline
                    prob_ratio = avg_model_prob / random_prob
                    out[f'{split}_model_vs_random'] = prob_ratio
                    out[f'{split}_avg_correct_prob'] = avg_model_prob
                else:
                    raise ValueError(f"Unsupported training type: {training_ctx.training_type}")

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

# In train_utils.py

# Add this new function to train_utils.py
def apply_bert_style_corruption_gpu(x, mask, mask_token_id, meta_vocab_size):
    """
    Applies the 80/10/10 corruption strategy from BERT to the selected positions.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions selected for prediction (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token.
        meta_vocab_size: The size of the original vocabulary for generating random tokens (excluding special tokens).
        
    Returns:
        corrupted_x: The input tokens after applying the 80/10/10 rule.
    """
    corrupted_x = x.clone()
    
    # Generate random numbers to decide on the corruption type for each masked position
    rand = torch.rand(x.shape, device=x.device)
    
    # Determine the positions for each case based on the main mask
    # 80% of the time, we replace with [MASK]
    mask_token_positions = mask & (rand < 0.8)
    
    # 10% of the time, we replace with a random token (0.8 <= rand < 0.9)
    random_token_positions = mask & (rand >= 0.8) & (rand < 0.9)
    
    # 10% of the time, we keep the original token (rand >= 0.9) - no action needed for these
    
    # Apply the [MASK] tokens
    corrupted_x[mask_token_positions] = mask_token_id
    
    # Apply the random tokens
    num_random = random_token_positions.sum()
    if num_random > 0:
        random_tokens = torch.randint(0, meta_vocab_size, (num_random,), device=x.device)
        corrupted_x[random_token_positions] = random_tokens
        
    return corrupted_x

def calculate_wrong_answer_entropy(logits, targets, vocab_size):
    """
    Calculate entropy of wrong answer distributions for entropy penalty.
    
    HIGH entropy (uniform wrong answers) = GOOD (high signal-to-noise ratio)
    LOW entropy (concentrated wrong answers) = BAD (low signal-to-noise ratio)
    
    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        targets: Target tokens (batch_size, seq_len)
        vocab_size: Size of vocabulary
        
    Returns:
        avg_entropy: Average entropy of wrong answer distributions across all positions
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    epsilon = 1e-9 # Use a slightly larger epsilon for stability
    
    # Get probabilities from logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Flatten for easier processing
    probs_flat = probs.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # --- START FIX ---
    
    # Create a mask to zero out the correct answer probabilities
    wrong_probs = probs_flat.clone()
    wrong_probs[range(len(targets_flat)), targets_flat] = 0.0
    
    # Calculate the sum of the remaining "wrong" probabilities
    # This sum is (1.0 - p_correct)
    sum_wrong_probs = wrong_probs.sum(dim=1, keepdim=True)
    
    # Avoid division by zero for positions where p_correct was close to 1.0
    # If sum_wrong_probs is near zero, entropy is also zero, so we can ignore these.
    # We create a mask for safe normalization.
    safe_mask = sum_wrong_probs.squeeze() > epsilon
    
    if not safe_mask.any():
        # Handle the edge case where no positions have significant wrong probabilities
        return torch.tensor(0.0, device=device)
        
    # Re-normalize the wrong probabilities so they sum to 1
    # This creates a true probability distribution over the incorrect tokens
    normalized_wrong_probs = wrong_probs[safe_mask] / sum_wrong_probs[safe_mask]
    
    # Calculate entropy on the properly normalized distribution
    log_probs = torch.log(normalized_wrong_probs + epsilon)
    entropies = -(normalized_wrong_probs * log_probs).sum(dim=1)
    
    # --- END FIX ---
    
    # Return average entropy across all valid positions
    return entropies.mean()

def get_current_entropy_penalty(iter_num, ctx: TrainingContext):
    """
    Calculate current entropy penalty based on iteration number.
    
    Args:
        iter_num: Current iteration number
        ctx: Training context with penalty parameters
        
    Returns:
        current_penalty: Current entropy penalty multiplier (0 to max_entropy_penalty)
    """
    if not ctx.enable_entropy_penalty:
        return 0.0
    
    if iter_num < ctx.entropy_penalty_start_iter:
        return 0.0
    
    if iter_num >= ctx.max_iters:
        return ctx.max_entropy_penalty
    
    # Linear increase from start_iter to max_iters
    progress = (iter_num - ctx.entropy_penalty_start_iter) / (ctx.max_iters - ctx.entropy_penalty_start_iter)
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
    
    return progress * ctx.max_entropy_penalty

def update_entropy_multiplier_ema(ctx: TrainingContext, current_multiplier: float):
    """
    Update the exponential moving average of entropy multiplier.
    
    Args:
        ctx: Training context
        current_multiplier: Current entropy multiplier value
    """
    if ctx.enable_entropy_penalty:
        # EMA update: ema = alpha * ema + (1-alpha) * current_value
        alpha = ctx.entropy_multiplier_ema_factor
        ctx.entropy_multiplier_ema = alpha * ctx.entropy_multiplier_ema + (1 - alpha) * current_multiplier

def apply_label_smoothing(targets, uncertainty_factor, vocab_size, special_token_ids=None, device=None):
    """
    Apply label smoothing to target tokens.
    
    Args:
        targets: Target token IDs (batch_size, seq_len)
        uncertainty_factor: Label smoothing factor (0.0 = no smoothing, >0 = apply smoothing)
        vocab_size: Size of vocabulary
        special_token_ids: List of special token IDs to exclude from smoothing (optional)
        device: Device to create tensors on
        
    Returns:
        smoothed_targets: Probability distribution targets (batch_size, seq_len, vocab_size)
    """
    if uncertainty_factor <= 0.0:
        # No smoothing, return one-hot encoded targets
        return torch.nn.functional.one_hot(targets, num_classes=vocab_size).float()
    
    if device is None:
        device = targets.device
    
    batch_size, seq_len = targets.shape
    
    # Create smoothed probability distribution
    smoothed_targets = torch.zeros(batch_size, seq_len, vocab_size, device=device)
    
    # Set correct answer probability to (1 - uncertainty_factor)
    correct_prob = 1.0 - uncertainty_factor
    smoothed_targets.scatter_(2, targets.unsqueeze(-1), correct_prob)
    
    # Calculate incorrect answer probability: u / (vocab_size - 1)
    # But we need to exclude special tokens from getting smoothed probability
    incorrect_prob = uncertainty_factor / (vocab_size - len(special_token_ids))
    
    # Add smoothing probability to all positions except the correct answer
    smoothed_targets += incorrect_prob
    
    # Remove the extra probability that was added to correct answers
    smoothed_targets.scatter_(2, targets.unsqueeze(-1), correct_prob)
    
    # Handle special tokens - set their probability to 0 (except when they are the correct answer)
    if special_token_ids is not None:
        for special_id in special_token_ids:
            if special_id < vocab_size:
                # Create mask for positions where special_id is NOT the correct answer
                not_correct_mask = (targets != special_id).unsqueeze(-1)
                # Zero out probability for this special token where it's not correct
                special_mask = torch.zeros(batch_size, seq_len, vocab_size, device=device)
                special_mask[:, :, special_id] = 1.0
                smoothed_targets = smoothed_targets * (1 - special_mask * not_correct_mask.float())
    
    sum_probs = smoothed_targets.sum(dim=-1, keepdim=True)
    # Renormalize to ensure probabilities sum to 1
    smoothed_targets = smoothed_targets / sum_probs
    
    return smoothed_targets