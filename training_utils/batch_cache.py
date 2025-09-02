"""Runtime batch caching utilities - separate from dataset preparation"""
import torch
from queue import Queue
from typing import Tuple, Optional
import threading


class BatchCache:
    """Runtime cache for efficient batch serving"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = Queue(maxsize=cache_size)
        self.cache_size = cache_size
        self._lock = threading.Lock()
    
    def get_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get batch from cache if available"""
        try:
            return self.cache.get_nowait()
        except:
            return None
    
    def put_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Add batch to cache"""
        try:
            self.cache.put_nowait(batch)
        except:
            pass  # Cache full, skip
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            while not self.cache.empty():
                try:
                    self.cache.get_nowait()
                except:
                    break
    
    def size(self) -> int:
        """Get current cache size"""
        return self.cache.qsize()
    
    def is_full(self) -> bool:
        """Check if cache is full"""
        return self.cache.full()
    
    def is_empty(self) -> bool:
        """Check if cache is empty"""
        return self.cache.empty()