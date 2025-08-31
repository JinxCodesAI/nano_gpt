"""
Utility functions for diffusion training including timing and logging
"""
import time
import torch
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
    
    def get_recent_average(self, name, last_n=10):
        """Get average time for very recent calls only"""
        if name not in self.times or not self.times[name]:
            return 0.0
        return sum(self.times[name][-last_n:]) / min(len(self.times[name]), last_n)
    
    def get_last_time(self, name):
        """Get the most recent time measurement"""
        if name not in self.times or not self.times[name]:
            return 0.0
        return self.times[name][-1]




def analyze_clustering(mask):
    """Analyze clustering properties of mask patterns"""
    batch_size, seq_len = mask.shape
    cluster_sizes = []

    for batch_idx in range(batch_size):
        mask_seq = mask[batch_idx].cpu().numpy()

        # Find connected components (clusters)
        in_cluster = False
        current_cluster_size = 0

        for pos in range(seq_len):
            if mask_seq[pos]:  # Masked position
                if not in_cluster:
                    in_cluster = True
                    current_cluster_size = 1
                else:
                    current_cluster_size += 1
            else:  # Unmasked position
                if in_cluster:
                    cluster_sizes.append(current_cluster_size)
                    in_cluster = False
                    current_cluster_size = 0

        # Handle cluster at end of sequence
        if in_cluster:
            cluster_sizes.append(current_cluster_size)

    if cluster_sizes:
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        num_clusters = len(cluster_sizes)
        return {
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size,
            'num_clusters_per_batch': num_clusters / batch_size
        }
    else:
        return {
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'num_clusters_per_batch': 0
        }


def analyze_masking_patterns_with_transition(mask, iter_num, sticky_transition_start, sticky_transition_end):
    """Analyze masking patterns during independent->sticky transition"""
    mask_ratio = mask.float().mean().item()

    # Calculate current transition state
    if iter_num < sticky_transition_start:
        transition_state = "independent"
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        transition_state = "sticky"
        sticky_ratio = 1.0
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress
        transition_state = f"transition ({sticky_ratio:.2f})"

    # Analyze clustering (more relevant during sticky phase)
    if sticky_ratio > 0.1:  # Only analyze clusters when some sticky masking present
        cluster_stats = analyze_clustering(mask)
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio,
            **cluster_stats
        }
    else:
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio
        }


def log_masking_stats(mask, iter_num, log_interval, sticky_transition_start=None, sticky_transition_end=None):
    """Log statistics about masking patterns"""
    if sticky_transition_start is not None and sticky_transition_end is not None:
        # Enhanced logging with transition tracking
        masking_stats = analyze_masking_patterns_with_transition(mask, iter_num, sticky_transition_start, sticky_transition_end)
        if iter_num % (log_interval * 10) == 0:  # Less frequent detailed stats
            print(f"Masking: {masking_stats}")
    else:
        # Original simple logging
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
