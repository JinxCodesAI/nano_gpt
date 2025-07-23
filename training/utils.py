"""
Training utilities for nanoGPT including timing, profiling, and helper functions.
"""
import os
import time
import math
import torch
import numpy as np
import psutil
import signal
import sys
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Dict, Any, Optional, Callable


class TimingProfiler:
    """
    A utility class for measuring training loop component timing.
    It supports a hierarchical EMA summary and provides a backward-compatible
    interface that also uses the smoothed EMA data.
    """

    def __init__(self, alpha=0.1):
        # --- Attributes for EMA calculation ---
        self.alpha = alpha
        self.ema_timings = {}
        self._context_stack = []
        self._current_raw_timings = {}

    def _record_timing(self, section_name, duration):
        """Record a timing measurement and update EMA."""
        if section_name not in self.ema_timings:
            self.ema_timings[section_name] = duration
        else:
            self.ema_timings[section_name] = (
                self.alpha * duration + (1 - self.alpha) * self.ema_timings[section_name]
            )
        
        # Also store raw timing for immediate access
        self._current_raw_timings[section_name] = duration

    def time_section(self, section_name):
        """Context manager for timing a section."""
        return TimingContext(self, section_name)

    def get_summary(self):
        """Get timing summary using EMA values."""
        if not self.ema_timings:
            return "No timing data available."
        
        lines = ["Timing Summary (EMA):"]
        total_time = sum(self.ema_timings.values())
        
        for section, duration in sorted(self.ema_timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            lines.append(f"  {section}: {duration:.4f}s ({percentage:.1f}%)")
        
        lines.append(f"  Total: {total_time:.4f}s")
        return "\n".join(lines)

    def get_ema_timings(self):
        """Get EMA timings dictionary."""
        return self.ema_timings.copy()

    def reset(self):
        """Reset all timing data."""
        self.ema_timings.clear()
        self._current_raw_timings.clear()
        self._context_stack.clear()


class TimingContext:
    """Context manager for timing sections."""
    
    def __init__(self, profiler, section_name):
        self.profiler = profiler
        self.section_name = section_name
        self.start_time = None

    def __enter__(self):
        self.profiler._context_stack.append(self.section_name)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.profiler._record_timing(self.section_name, duration)
        self.profiler._context_stack.pop()


class BatchManager:
    """Manages batch loading and token remapping for training."""
    
    def __init__(self, train_shard_filenames, num_train_shards, data_dir, block_size, batch_size,
                 device, shrunken_vocab_size=None, vocab_remapping=None, rare_token_id=None):
        self.train_shard_filenames = train_shard_filenames
        self.num_train_shards = num_train_shards
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.shrunken_vocab_size = shrunken_vocab_size
        self.vocab_remapping = vocab_remapping
        self.RARE_TOKEN_ID = rare_token_id
        
        # Initialize shard state
        self.current_shard_index = 0
        self.current_position = 0
        self.current_shard_data = None
        self.shard_sizes = {}
        
        # Load initial shard
        self._load_shard(self.current_shard_index)
    
    def _load_shard(self, shard_index):
        """Load a specific training shard."""
        if shard_index >= len(self.train_shard_filenames):
            shard_index = 0  # Wrap around
        
        filename = self.train_shard_filenames[shard_index]
        shard_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Training shard not found: {shard_path}")
        
        self.current_shard_data = np.memmap(shard_path, dtype=np.uint16, mode='r')
        self.shard_sizes[shard_index] = len(self.current_shard_data)
        self.current_shard_index = shard_index
        self.current_position = 0
        
        print(f"Loaded shard {shard_index}: {filename} ({len(self.current_shard_data):,} tokens)")
    
    def get_batch(self):
        """Get the next training batch."""
        # Check if we need to move to the next shard
        if self.current_position + self.batch_size * self.block_size >= len(self.current_shard_data):
            # Move to next shard
            next_shard = (self.current_shard_index + 1) % len(self.train_shard_filenames)
            self._load_shard(next_shard)
        
        # Extract batch data
        start_pos = self.current_position
        end_pos = start_pos + self.batch_size * self.block_size
        
        batch_data = self.current_shard_data[start_pos:end_pos]
        self.current_position = end_pos
        
        # Reshape and convert to tensors
        batch_data = batch_data.astype(np.int64)
        data = torch.from_numpy(batch_data)
        
        # Apply vocabulary remapping if using shrunken vocabulary
        if self.shrunken_vocab_size is not None and self.vocab_remapping is not None:
            data = self._apply_vocab_remapping(data)
        
        # Split into input and target sequences
        data = data.view(self.batch_size, self.block_size)
        x = data[:, :-1].contiguous()  # Input sequences
        y = data[:, 1:].contiguous()   # Target sequences (shifted by 1)
        
        # Move to device
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        
        return x, y
    
    def _apply_vocab_remapping(self, data):
        """Apply vocabulary remapping for shrunken vocabulary training."""
        # Create a mask for tokens that are in the remapping
        valid_mask = data < len(self.vocab_remapping)
        
        # Apply remapping to valid tokens
        remapped_data = torch.where(
            valid_mask,
            self.vocab_remapping[data.clamp(0, len(self.vocab_remapping) - 1)],
            self.RARE_TOKEN_ID
        )
        
        return remapped_data
    
    def get_shard_info(self):
        """Get information about current shard state."""
        return {
            'current_shard': self.current_shard_index,
            'total_shards': len(self.train_shard_filenames),
            'current_position': self.current_position,
            'shard_size': len(self.current_shard_data) if self.current_shard_data is not None else 0,
            'progress': self.current_position / len(self.current_shard_data) if self.current_shard_data is not None else 0
        }


def get_lr(it: int, learning_rate: float, warmup_iters: int, lr_decay_iters: int, 
          min_lr: float, decay_lr: bool = True, lr_schedule_offset: int = 0) -> float:
    """Calculate learning rate with warmup and cosine decay."""
    # Adjust iteration based on schedule offset (for resets)
    adjusted_it = it - lr_schedule_offset
    
    if not decay_lr:
        return learning_rate
    
    # 1) Linear warmup for warmup_iters steps
    if adjusted_it < warmup_iters:
        return learning_rate * adjusted_it / warmup_iters
    
    # 2) If adjusted_it > lr_decay_iters, return min learning rate
    if adjusted_it > lr_decay_iters:
        return min_lr
    
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (adjusted_it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def estimate_loss(model, get_val_batch_fn: Callable, eval_iters: int, 
                 device_type: str, dtype: str) -> float:
    """Estimate validation loss by averaging over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    # Determine context and autocast settings
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16 if dtype == 'bfloat16' else torch.float16
    )
    
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_val_batch_fn()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
    
    model.train()
    return losses.mean().item()


def print_timings(timing_profiler: TimingProfiler, training_logger, master_process: bool = True):
    """Print and log timing information."""
    if master_process:
        timing_summary = timing_profiler.get_summary()
        print(f"\n{timing_summary}")
        training_logger.log(timing_summary)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f'\nReceived signal {signum}. Gracefully shutting down...')
        # Set a global flag that the training loop can check
        import __main__
        __main__.should_terminate = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    return {
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
    }


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f}PB"


def format_time(seconds: float) -> str:
    """Format time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"