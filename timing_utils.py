"""
Timing utilities for profiling generation performance.

Provides context managers and accumulators for measuring time spent in different
phases of the generation process (forward pass, sampling, remasking, etc.).
"""
import time
from contextlib import contextmanager
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    """Statistics for a single timing category."""
    total_time: float = 0.0
    count: int = 0
    
    @property
    def avg_time(self) -> float:
        """Average time per call."""
        return self.total_time / max(1, self.count)
    
    def add(self, duration: float):
        """Add a timing measurement."""
        self.total_time += duration
        self.count += 1
    
    def reset(self):
        """Reset statistics."""
        self.total_time = 0.0
        self.count = 0


class TimingAccumulator:
    """
    Accumulates timing measurements across multiple operations.
    
    Usage:
        timer = TimingAccumulator()
        
        with timer.measure('forward'):
            # ... forward pass code ...
        
        with timer.measure('sampling'):
            # ... sampling code ...
        
        stats = timer.get_stats()
        timer.print_summary()
    """
    
    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
        self._enabled = True
    
    def enable(self):
        """Enable timing measurements."""
        self._enabled = True
    
    def disable(self):
        """Disable timing measurements (for minimal overhead)."""
        self._enabled = False
    
    @contextmanager
    def measure(self, category: str):
        """
        Context manager for measuring time spent in a code block.
        
        Args:
            category: Name of the timing category (e.g., 'forward', 'sampling')
        """
        if not self._enabled:
            yield
            return
        
        if category not in self.stats:
            self.stats[category] = TimingStats()
        
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.stats[category].add(duration)
    
    def get_stats(self) -> Dict[str, TimingStats]:
        """Get all timing statistics."""
        return self.stats
    
    def get_total_time(self) -> float:
        """Get total time across all categories."""
        return sum(stat.total_time for stat in self.stats.values())
    
    def get_percentages(self) -> Dict[str, float]:
        """Get percentage of total time for each category."""
        total = self.get_total_time()
        if total == 0:
            return {cat: 0.0 for cat in self.stats}
        return {cat: (stat.total_time / total) * 100 for cat, stat in self.stats.items()}
    
    def reset(self):
        """Reset all statistics."""
        for stat in self.stats.values():
            stat.reset()
    
    def print_summary(self, title: str = "Timing Summary", show_counts: bool = False):
        """
        Print a formatted summary of timing statistics.
        
        Args:
            title: Title for the summary
            show_counts: Whether to show call counts
        """
        if not self.stats:
            print(f"\n{title}: No timing data")
            return
        
        total_time = self.get_total_time()
        percentages = self.get_percentages()
        
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
        print(f"Total time: {total_time:.3f}s")
        print(f"{'-'*60}")
        
        # Sort by total time (descending)
        sorted_cats = sorted(self.stats.items(), key=lambda x: x[1].total_time, reverse=True)
        
        for category, stat in sorted_cats:
            pct = percentages[category]
            avg = stat.avg_time * 1000  # Convert to ms
            total_ms = stat.total_time * 1000
            
            if show_counts:
                print(f"{category:20s}: {total_ms:8.1f}ms ({pct:5.1f}%) | "
                      f"avg: {avg:6.2f}ms | calls: {stat.count}")
            else:
                print(f"{category:20s}: {total_ms:8.1f}ms ({pct:5.1f}%) | avg: {avg:6.2f}ms")
        
        print(f"{'='*60}")
    
    def get_summary_dict(self) -> Dict[str, any]:
        """
        Get timing summary as a dictionary (for JSON export).
        
        Returns:
            Dictionary with timing statistics
        """
        total_time = self.get_total_time()
        percentages = self.get_percentages()
        
        return {
            'total_time_s': total_time,
            'categories': {
                cat: {
                    'total_time_s': stat.total_time,
                    'avg_time_ms': stat.avg_time * 1000,
                    'count': stat.count,
                    'percentage': percentages[cat]
                }
                for cat, stat in self.stats.items()
            }
        }


class IterationTimer:
    """
    Timer for tracking per-iteration timing with automatic averaging.
    
    Usage:
        iter_timer = IterationTimer()
        
        for iteration in range(num_iterations):
            with iter_timer.iteration():
                with iter_timer.measure('forward'):
                    # ... forward pass ...
                with iter_timer.measure('sampling'):
                    # ... sampling ...
            
            # Print per-iteration stats
            iter_timer.print_iteration_summary(iteration)
        
        # Print overall summary
        iter_timer.print_overall_summary()
    """
    
    def __init__(self):
        self.overall = TimingAccumulator()
        self.current_iter = TimingAccumulator()
        self._in_iteration = False
        self._iter_start = None
    
    @contextmanager
    def iteration(self):
        """Context manager for a single iteration."""
        self._in_iteration = True
        self._iter_start = time.perf_counter()
        self.current_iter.reset()
        try:
            yield
        finally:
            self._in_iteration = False
    
    @contextmanager
    def measure(self, category: str):
        """Measure time for a category (accumulates in both current and overall)."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.current_iter.stats.setdefault(category, TimingStats()).add(duration)
            self.overall.stats.setdefault(category, TimingStats()).add(duration)
    
    def get_iteration_time(self) -> float:
        """Get total time for current iteration."""
        if self._iter_start is None:
            return 0.0
        return time.perf_counter() - self._iter_start
    
    def print_iteration_summary(self, iteration: int, show_percentages: bool = True):
        """Print summary for current iteration."""
        if not self.current_iter.stats:
            return
        
        total = self.current_iter.get_total_time()
        percentages = self.current_iter.get_percentages()
        
        parts = []
        for cat in sorted(self.current_iter.stats.keys()):
            stat = self.current_iter.stats[cat]
            ms = stat.total_time * 1000
            if show_percentages:
                pct = percentages[cat]
                parts.append(f"{cat} {ms:.1f}ms ({pct:.0f}%)")
            else:
                parts.append(f"{cat} {ms:.1f}ms")
        
        timing_str = " | ".join(parts)
        print(f"  Iter {iteration} timing: {timing_str} | total {total*1000:.1f}ms")
    
    def print_overall_summary(self):
        """Print overall summary across all iterations."""
        self.overall.print_summary(title="Overall Timing Summary", show_counts=True)
    
    def get_overall_summary_dict(self) -> Dict[str, any]:
        """Get overall summary as dictionary."""
        return self.overall.get_summary_dict()

