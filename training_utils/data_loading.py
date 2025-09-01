"""
Data loading, prefetching, and caching utilities for diffusion training.
Handles efficient batch data preparation and background prefetching.
"""

import time
import threading
from queue import Queue

import numpy as np
import torch

from .training_config import TrainingContext


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


# Export global variables for use by other modules
def get_data_cache():
    """Get reference to data cache"""
    global _data_cache
    return _data_cache


def get_valid_indices_cache():
    """Get reference to valid indices cache"""
    global _valid_indices_cache
    return _valid_indices_cache


def get_prefetch_queue():
    """Get reference to prefetch queue"""
    global _prefetch_queue
    return _prefetch_queue


def get_validation_caches():
    """Get references to all validation caches"""
    global _val_batch_cache, _progressive_val_cache, _progressive_val_full_cache, _unmasking_val_set, _remasking_val_set
    return _val_batch_cache, _progressive_val_cache, _progressive_val_full_cache, _unmasking_val_set, _remasking_val_set


def set_validation_caches(val_batch_cache, progressive_val_cache, progressive_val_full_cache, unmasking_val_set, remasking_val_set):
    """Set references to all validation caches"""
    global _val_batch_cache, _progressive_val_cache, _progressive_val_full_cache, _unmasking_val_set, _remasking_val_set
    _val_batch_cache = val_batch_cache
    _progressive_val_cache = progressive_val_cache
    _progressive_val_full_cache = progressive_val_full_cache
    _unmasking_val_set = unmasking_val_set
    _remasking_val_set = remasking_val_set