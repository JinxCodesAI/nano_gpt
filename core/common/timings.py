"""
Timing utilities for profiling generation performance.

Provides context managers and accumulators for measuring time spent in different
phases of the generation process (forward pass, sampling, remasking, etc.).
"""
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, Optional, List
import torch

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


class TimingNode:
    """A node in the hierarchical timing tree."""
    def __init__(self, name: str):
        self.name = name
        self.stats = TimingStats()
        self.children: Dict[str, 'TimingNode'] = {}

    def child(self, name: str) -> 'TimingNode':
        if name not in self.children:
            self.children[name] = TimingNode(name)
        return self.children[name]


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
        # Flat stats for backward compatibility and quick aggregates per category
        self.stats: Dict[str, TimingStats] = {}
        # Hierarchical tree and context stack
        self._root = TimingNode('__root__')
        self._stack: List[TimingNode] = [self._root]
        # Global wall-clock window across all measurements
        self._overall_first_start: Optional[float] = None
        self._overall_last_end: Optional[float] = None
        # Controls
        self._enabled = True
        self._cuda_sync = False  # when True, synchronize CUDA before/after to get accurate timings

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

        # Flat category bucket
        if category not in self.stats:
            self.stats[category] = TimingStats()
        # Hierarchical node under current parent
        parent = self._stack[-1]
        node = parent.child(category)

        # Optional CUDA synchronization for accurate GPU timings
        if self._cuda_sync and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        start = time.perf_counter()
        # Update global window
        if self._overall_first_start is None or start < self._overall_first_start:
            self._overall_first_start = start
        # Push context
        self._stack.append(node)
        try:
            yield
        finally:
            # Pop context first to ensure well-nested behavior even on exceptions
            self._stack.pop()
            if self._cuda_sync and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            end = time.perf_counter()
            duration = end - start
            # Flat aggregate
            self.stats[category].add(duration)
            # Hierarchical aggregate
            node.stats.add(duration)
            # Update global window
            if self._overall_last_end is None or end > self._overall_last_end:
                self._overall_last_end = end

    def get_stats(self) -> Dict[str, TimingStats]:
        """Get all timing statistics."""
        return self.stats

    def get_total_time(self) -> float:
        """Get total wall-clock time from first start to last end across all measurements."""
        if self._overall_first_start is None or self._overall_last_end is None:
            return 0.0
        return max(0.0, self._overall_last_end - self._overall_first_start)

    def set_cuda_sync(self, enabled: bool = True):
        """Enable/disable CUDA synchronization around timing blocks."""
        self._cuda_sync = bool(enabled)

    def record(self, category: str, duration: float):
        """Record an externally measured duration into a category (attaches under current parent)."""
        if not self._enabled:
            return
        # Flat bucket
        if category not in self.stats:
            self.stats[category] = TimingStats()
        self.stats[category].add(float(duration))
        # Hierarchical bucket under current context
        parent = self._stack[-1]
        node = parent.child(category)
        node.stats.add(float(duration))
        # Update global window assuming this record ends "now"
        try:
            end = time.perf_counter()
            start = end - float(duration)
        except Exception:
            end = None
            start = None
        if start is not None:
            if self._overall_first_start is None or start < self._overall_first_start:
                self._overall_first_start = start
        if end is not None:
            if self._overall_last_end is None or end > self._overall_last_end:
                self._overall_last_end = end

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

    def print_hierarchical_summary(self, title: str = "OPERATION TIMING (hierarchical)", show_counts: bool = True):
        """Print a hierarchical timing tree with percentages relative to parent."""
        # Nothing to print
        if not self._root.children:
            print(f"\n{title}: No timing data")
            return

        total = self.get_total_time()
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
        print(f"Total wall time: {total:.3f}s")
        print(f"{'-'*60}")

        def _print_node(node: 'TimingNode', parent_total: float, indent: int = 0):
            name = node.name
            ms_total = node.stats.total_time * 1000.0
            avg_ms = node.stats.avg_time * 1000.0
            pct = (node.stats.total_time / parent_total * 100.0) if parent_total > 0 else 0.0
            pad = '  ' * indent
            if show_counts:
                print(f"{pad}{name}: ({pct:5.1f}%) | total: {ms_total:8.1f}ms | avg: {avg_ms:6.2f}ms | calls: {node.stats.count}")
            else:
                print(f"{pad}{name}: ({pct:5.1f}%) | total: {ms_total:8.1f}ms | avg: {avg_ms:6.2f}ms")
            # Children (sorted by total time desc)
            if node.children:
                children_sorted = sorted(node.children.values(), key=lambda n: n.stats.total_time, reverse=True)
                for child in children_sorted:
                    _print_node(child, parent_total=node.stats.total_time, indent=indent+1)

        # Top-level nodes under root
        top_sorted = sorted(self._root.children.values(), key=lambda n: n.stats.total_time, reverse=True)
        for top in top_sorted:
            _print_node(top, parent_total=total, indent=0)
        print(f"{'='*60}")


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


# ------------------------------
# Global timer (singleton) utils
# ------------------------------
# Delegate global timer storage to a top-level singleton module to avoid
# circular imports involving the core package.
from timings_singleton import get_global_timer as _get_global_timer, set_global_timer as _set_global_timer

def set_global_timer(timer: TimingAccumulator):
    """Register a global timer instance to be used across modules."""
    _set_global_timer(timer)

def get_global_timer() -> Optional[TimingAccumulator]:
    """Retrieve the registered global timer (or None if not set)."""
    return _get_global_timer()

@contextmanager
def global_measure(category: str):
    """Context manager that measures using the global timer when available."""
    timer = get_global_timer()
    if timer is None:
        with nullcontext():
            yield
    else:
        with timer.measure(category):
            yield

def print_global_summary(title: str = "Timing Summary", show_counts: bool = True):
    timer = get_global_timer()
    if timer is not None:
        timer.print_summary(title=title, show_counts=show_counts)



def print_global_hierarchical_summary(title: str = "OPERATION TIMING (hierarchical)", show_counts: bool = True):
    timer = get_global_timer()
    if timer is not None:
        timer.print_hierarchical_summary(title=title, show_counts=show_counts)
