"""
Training utilities for nanoGPT including timing, profiling, and helper functions.
"""
import os
import time
import math
import pickle
import torch
import numpy as np
import psutil
import signal
import sys
import threading
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
    
    def get_breakdown_percentages(self):
        """Get timing breakdown as percentages for display."""
        if not self.ema_timings:
            return {}
        
        total_time = sum(self.ema_timings.values())
        if total_time == 0:
            return {}
        
        breakdown = {}
        for section, duration in self.ema_timings.items():
            percentage = (duration / total_time) * 100
            breakdown[section] = percentage
        
        return breakdown
    
    def get_average_percentages(self, last_n=None):
        """Get average timing percentages (backward compatibility with original)."""
        return self.get_breakdown_percentages()

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
    """High-performance batch manager with background threading and curriculum learning."""
    
    def __init__(self, train_shard_filenames, num_train_shards, data_dir, block_size, batch_size,
                 device, shrunken_vocab_size=None, vocab_remapping=None, rare_token_id=None,
                 starting_estimation_token_count=100_000_000, buffer_size=2000):
        print("Initializing High-Performance BatchManager (V2)...")
        self.train_shard_filenames = train_shard_filenames
        self.num_train_shards = num_train_shards
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.shrunken_vocab_size = shrunken_vocab_size
        self.vocab_remapping = vocab_remapping
        self.RARE_TOKEN_ID = rare_token_id
        self.buffer_size = buffer_size
        
        # Determine vocab size
        if shrunken_vocab_size is not None:
            self.vocab_size = shrunken_vocab_size
        else:
            # Try to get vocab size from meta.pkl
            meta_path = os.path.join(data_dir, 'meta.pkl')
            if os.path.exists(meta_path):
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                self.vocab_size = meta['vocab_size']
            else:
                self.vocab_size = 50304  # Default
        
        # 1. Approximate or load the corpus token distribution
        self.corpus_distribution = self._get_corpus_distribution(starting_estimation_token_count)
        self.uniform_distribution = torch.ones(self.vocab_size, dtype=torch.float32) / self.vocab_size
        
        # 2. Initialize state for tracking served tokens
        self.served_token_counts = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.total_tokens_served = 0
        
        # 3. Thread-safe candidate buffer and control variables for the worker
        self.candidate_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.rescore_event = threading.Event()
        self.shutdown_event = threading.Event()
        
        # 4. Initialize the curriculum and target distribution
        self.alpha = 1.0
        self.target_distribution = self.uniform_distribution.clone()
        
        # 5. Start the background worker thread
        self.worker_thread = threading.Thread(target=self._buffer_management_worker, daemon=True)
        self.worker_thread.start()
        print("BatchManager initialized and background worker started.")
        
        # Wait for buffer to fill initially
        print("Main thread is waiting for the batch buffer to fill...")
        while len(self.candidate_buffer) < min(10, self.buffer_size // 4):
            time.sleep(0.1)
            if self.shutdown_event.is_set():
                break
    
    def _get_corpus_distribution(self, estimation_tokens):
        """Calculates an approximate token distribution from a sample of the dataset and caches it."""
        cache_path = os.path.join(self.data_dir, f'corpus_dist_approx_{estimation_tokens}.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached approximate corpus distribution from {cache_path}")
            return torch.load(cache_path)
        
        # Calculate distribution from sample
        print(f"Calculating approximate corpus distribution from {estimation_tokens:,} tokens...")
        total_counts = torch.zeros(self.vocab_size, dtype=torch.int64)
        tokens_per_shard = estimation_tokens // len(self.train_shard_filenames)
        unknown_token_id = self.vocab_size - 1  # Use last token as unknown
        
        for shard_name in self.train_shard_filenames:
            shard_path = os.path.join(self.data_dir, shard_name)
            if os.path.exists(shard_path):
                data = np.memmap(shard_path, dtype=np.uint16, mode='r')
                sample = data[:min(tokens_per_shard, len(data))]
                sample = np.clip(sample, 0, unknown_token_id)
                shard_counts = torch.from_numpy(np.bincount(sample, minlength=self.vocab_size))
                total_counts += shard_counts
        
        distribution = total_counts.float() / total_counts.sum()
        print(f"Saving approximate corpus distribution to {cache_path}")
        torch.save(distribution, cache_path)
        return distribution
    
    def _buffer_management_worker(self):
        """Runs in a background thread to continuously read, score, and manage the candidate buffer."""
        shard_cycle = iter(self.train_shard_filenames * 1000)  # Loop over the dataset many times
        
        while not self.shutdown_event.is_set():
            # --- Phase 1: Refill the buffer if it has space ---
            if len(self.candidate_buffer) < self.buffer_size:
                try:
                    shard_name = next(shard_cycle)
                    shard_path = os.path.join(self.data_dir, shard_name)
                    
                    if os.path.exists(shard_path):
                        data = np.memmap(shard_path, dtype=np.uint16, mode='r')
                        
                        # Generate random batches from this shard
                        for _ in range(5):  # Generate a few batches per shard
                            if len(self.candidate_buffer) >= self.buffer_size:
                                break
                                
                            # Random starting position
                            max_start = len(data) - self.batch_size * self.block_size
                            if max_start <= 0:
                                continue
                            start_pos = np.random.randint(0, max_start)
                            
                            # Extract batch
                            batch_data = data[start_pos:start_pos + self.batch_size * self.block_size]
                            batch_data = batch_data.astype(np.int64)
                            batch_tensor = torch.from_numpy(batch_data)
                            
                            # Apply vocabulary remapping if needed
                            if self.shrunken_vocab_size is not None and self.vocab_remapping is not None:
                                batch_tensor = self._apply_vocab_remapping(batch_tensor)
                            
                            # Split into x, y
                            batch_tensor = batch_tensor.view(self.batch_size, self.block_size)
                            x = batch_tensor[:, :-1].contiguous()
                            y = batch_tensor[:, 1:].contiguous()
                            
                            # Calculate initial score
                            tokens, counts = torch.unique(x, return_counts=True)
                            score = counts.sum().item()  # Simple initial score
                            
                            batch_data_dict = {
                                'x': x,
                                'y': y,
                                'score': score
                            }
                            
                            with self.buffer_lock:
                                self.candidate_buffer.append(batch_data_dict)
                                
                except StopIteration:
                    print("Worker has finished all shard cycles.")
                    break
                except Exception as e:
                    print(f"Error in buffer refill worker: {e}")
                    time.sleep(1)
            
            # --- Phase 2: Re-score and sort the entire buffer if signaled ---
            if self.rescore_event.is_set():
                with self.buffer_lock:
                    print("(Async Worker) Re-scoring candidate buffer...")
                    served_dist = (self.served_token_counts / (self.total_tokens_served + 1e-9)).to(torch.float32)
                    temp_list = list(self.candidate_buffer)
                    
                    for batch_data in temp_list:
                        tokens, counts = torch.unique(batch_data['x'], return_counts=True)
                        neglect_score = self.target_distribution[tokens] / (served_dist[tokens] + 1e-9)
                        batch_data['score'] = (neglect_score * counts).sum().item()
                    
                    # Sort the buffer by score (highest first) and trim excess
                    temp_list.sort(key=lambda b: b['score'], reverse=True)
                    self.candidate_buffer = deque(temp_list[:self.buffer_size])
                    
                self.rescore_event.clear()
                print("(Async Worker) Buffer re-scoring completed.")
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def get_batch(self):
        """Get the next training batch from the high-performance buffer."""
        # Wait for at least one batch to be available
        while len(self.candidate_buffer) == 0:
            if self.shutdown_event.is_set():
                return None, None
            time.sleep(0.01)
        
        with self.buffer_lock:
            # The buffer is kept sorted by the worker, so the best is always at the front
            best_batch_data = self.candidate_buffer.popleft()
        
        best_x, best_y = best_batch_data['x'], best_batch_data['y']
        
        # Update the state of served tokens
        unique_tokens, counts = torch.unique(best_x, return_counts=True)
        self.served_token_counts[unique_tokens] += counts.to(self.served_token_counts.dtype)
        self.total_tokens_served += best_x.numel()
        
        # Move the chosen batch to the correct device
        if self.device_type == 'cuda':
            best_x = best_x.pin_memory().to(self.device, non_blocking=True)
            best_y = best_y.pin_memory().to(self.device, non_blocking=True)
        else:
            best_x, best_y = best_x.to(self.device), best_y.to(self.device)
        
        return best_x, best_y
    
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
        """Get information about current buffer state."""
        with self.buffer_lock:
            buffer_size = len(self.candidate_buffer)
        
        return {
            'buffer_size': buffer_size,
            'max_buffer_size': self.buffer_size,
            'total_tokens_served': self.total_tokens_served,
            'worker_active': self.worker_thread.is_alive() if hasattr(self, 'worker_thread') else False
        }
    
    def shutdown(self):
        """Shutdown the background worker thread."""
        print("Shutting down BatchManager background worker...")
        self.shutdown_event.set()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        print("BatchManager shut down.")
    
    def get_non_outlier_tokens(self, ignored_outlayers_sum=0.01):
        """Extract token IDs that sum up to (1 - ignored_outlayers_sum) of total_tokens_served.
        Returns a list of token IDs, excluding the most and least frequent outliers.
        """
        if self.total_tokens_served == 0:
            return list(range(self.vocab_size))  # Return all tokens if no tokens served yet
        
        # Calculate the served distribution
        served_distribution = self.served_token_counts / self.total_tokens_served
        
        # Sort tokens by their served frequency
        sorted_indices = torch.argsort(served_distribution, descending=True)
        sorted_counts = served_distribution[sorted_indices]
        
        # Calculate cumulative sum
        cumulative_sum = torch.cumsum(sorted_counts, dim=0)
        
        # Find tokens that sum up to (1 - ignored_outlayers_sum) of total
        target_sum = 1.0 - ignored_outlayers_sum
        cutoff_index = torch.searchsorted(cumulative_sum, target_sum).item()
        
        # Return the non-outlier token IDs
        selected_tokens = sorted_indices[:cutoff_index + 1]
        return selected_tokens.tolist()


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


def get_vram_usage():
    """
    Get current VRAM usage statistics.
    
    Returns:
        Tuple of (reserved_gb, total_gb, used_percent)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Use reserved memory as it's more accurate for actual GPU usage
        used_percent = (reserved / total) * 100
        
        return reserved, total, used_percent  # Return reserved instead of allocated
    return 0, 0, 0


def calculate_relative_batch_size(current_batch_size: int, scale_factor: float, 
                                master_process: bool = True) -> int:
    """
    Calculate new batch size by scaling current batch size by a factor.
    
    Args:
        current_batch_size: Current batch size
        scale_factor: Factor to scale by (0 < scale_factor <= 1)
        master_process: Whether this is the master process
    
    Returns:
        New batch size rounded down to nearest multiple of 8
    """
    if scale_factor <= 0 or scale_factor > 1:
        raise ValueError(f"scale_factor must be in range (0, 1], got {scale_factor}")
    
    # Calculate new batch size
    new_batch_size_float = current_batch_size * scale_factor
    
    # Round down to nearest multiple of 8, minimum of 8
    new_batch_size = max(8, int(new_batch_size_float // 8) * 8)
    
    if master_process:
        print(f"Scaling batch size: {current_batch_size} × {scale_factor:.3f} = {new_batch_size_float:.1f}")
        print(f"Rounded down to nearest multiple of 8: {new_batch_size}")
    
    return new_batch_size


def calculate_optimal_batch_size(model, current_batch_size: int, max_batch_size: int = 1024, 
                               target_vram_percent: float = 82.0, device_type: str = 'cuda',
                               master_process: bool = True) -> int:
    """
    Calculate optimal batch size based on current VRAM usage.
    
    Args:
        model: The model to test
        current_batch_size: Current batch size
        max_batch_size: Maximum batch size to consider
        target_vram_percent: Target VRAM utilization percentage
        device_type: Device type ('cuda' or 'cpu')
        master_process: Whether this is the master process
    
    Returns:
        Optimal batch size (multiple of 8, preferably power of 2)
    """
    if device_type != 'cuda' or not torch.cuda.is_available():
        return current_batch_size
    
    # Get current VRAM state
    current_vram_used, total_vram, current_percent = get_vram_usage()
    
    if master_process:
        print(f"Current VRAM usage: {current_vram_used:.1f}/{total_vram:.1f}GB ({current_percent:.1f}%)")
        print(f"Target VRAM usage: {target_vram_percent:.1f}%")
    
    # If current usage is already optimal, keep current batch size
    if abs(current_percent - target_vram_percent) < 5.0:
        if master_process:
            print(f"Current VRAM usage is optimal, keeping batch size: {current_batch_size}")
        return current_batch_size
    
    # Estimate memory per sample (rough approximation)
    memory_per_sample = current_vram_used / current_batch_size if current_batch_size > 0 else 0.1
    target_vram_used = total_vram * (target_vram_percent / 100.0)
    estimated_optimal_batch = int(target_vram_used / memory_per_sample) if memory_per_sample > 0 else current_batch_size
    
    # Generate candidate batch sizes (multiples of 8, preferably powers of 2)
    candidates = []
    
    # Powers of 2
    power = 3  # Start from 8
    while 2**power <= max_batch_size:
        candidates.append(2**power)
        power += 1
    
    # Multiples of 8 that aren't powers of 2
    for mult in range(3, max_batch_size // 8 + 1):
        candidate = mult * 8
        if candidate <= max_batch_size and candidate not in candidates:
            candidates.append(candidate)
    
    candidates.sort()
    
    # Find the best candidate closest to our estimate
    best_batch_size = current_batch_size
    best_diff = float('inf')
    
    for candidate in candidates:
        # Prefer candidates close to our estimate
        diff = abs(candidate - estimated_optimal_batch)
        if diff < best_diff:
            best_diff = diff
            best_batch_size = candidate
    
    # Safety bounds - don't change too drastically
    min_batch_size = max(8, current_batch_size // 4)
    max_safe_batch_size = min(max_batch_size, current_batch_size * 4)
    best_batch_size = max(min_batch_size, min(best_batch_size, max_safe_batch_size))
    
    if master_process:
        estimated_vram_with_new_batch = memory_per_sample * best_batch_size
        estimated_percent = (estimated_vram_with_new_batch / total_vram) * 100
        print(f"Estimated optimal batch size: {estimated_optimal_batch}")
        print(f"Selected batch size: {best_batch_size} (estimated VRAM: {estimated_percent:.1f}%)")
    
    return best_batch_size