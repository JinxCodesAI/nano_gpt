"""
Utility functions for diffusion training including timing and logging
"""
import time
from collections import defaultdict


class Timer:
    """Timer class for performance monitoring with context manager support"""
    
    def __init__(self):
        self.times = defaultdict(list)
    
    def time_function(self, name):
        """Context manager for timing function calls"""
        class TimerContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.timer.times[self.name].append(elapsed)
        
        return TimerContext(self, name)
    
    def get_average(self, name, last_n=100):
        """Get average time for last N calls"""
        if name not in self.times or not self.times[name]:
            return 0.0
        return sum(self.times[name][-last_n:]) / min(len(self.times[name]), last_n)


def log_masking_stats(mask, iter_num, log_interval):
    """Log statistics about masking patterns"""
    mask_ratio = mask.float().mean().item()
    batch_size, seq_len = mask.shape
    
    # Count consecutive masked regions
    mask_np = mask.cpu().numpy()
    consecutive_regions = []
    for batch_idx in range(batch_size):
        regions = []
        current_length = 0
        for pos in range(seq_len):
            if mask_np[batch_idx, pos]:
                current_length += 1
            else:
                if current_length > 0:
                    regions.append(current_length)
                    current_length = 0
        if current_length > 0:
            regions.append(current_length)
        consecutive_regions.extend(regions)
    
    avg_region_length = sum(consecutive_regions) / len(consecutive_regions) if consecutive_regions else 0
    
    if iter_num % (log_interval * 10) == 0:  # Less frequent detailed stats
        print(f"Masking stats: {mask_ratio:.3f} ratio, {avg_region_length:.1f} avg region length")
