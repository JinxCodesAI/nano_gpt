"""
Learning rate schedulers for training.

This module provides abstractions for learning rate scheduling, extracted from
the original get_lr() function in train.py to improve modularity and testability.
"""

import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Abstract base class for learning rate schedulers."""
    
    @abstractmethod
    def get_lr(self, iter_num: int) -> float:
        """
        Get learning rate for the given iteration number.
        
        Args:
            iter_num: Current training iteration number
            
        Returns:
            Learning rate value for this iteration
        """
        pass


class CosineLRScheduler(LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup.
    
    Implements the exact same logic as the original get_lr() function:
    1. Linear warmup for warmup_iters steps
    2. Cosine decay down to min_lr
    3. Constant min_lr after lr_decay_iters
    """
    
    def __init__(
        self, 
        learning_rate: float, 
        warmup_iters: int, 
        lr_decay_iters: int, 
        min_lr: float,
        decay_lr: bool = True
    ):
        """
        Initialize cosine scheduler with warmup.
        
        Args:
            learning_rate: Maximum learning rate (reached after warmup)
            warmup_iters: Number of iterations for linear warmup
            lr_decay_iters: Number of iterations for decay (should be ~= max_iters)
            min_lr: Minimum learning rate (should be ~= learning_rate/10)
            decay_lr: Whether to decay learning rate (if False, constant learning_rate)
        """
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.decay_lr = decay_lr
    
    def get_lr(self, iter_num: int) -> float:
        """
        Get learning rate for the given iteration - exact replica of original get_lr().
        
        Args:
            iter_num: Current training iteration number
            
        Returns:
            Learning rate value for this iteration
        """
        if not self.decay_lr:
            return self.learning_rate
            
        # 1) linear warmup for warmup_iters steps
        if iter_num < self.warmup_iters:
            return self.learning_rate * (iter_num + 1) / (self.warmup_iters + 1)
        
        # 2) if it > lr_decay_iters, return min learning rate
        if iter_num > self.lr_decay_iters:
            return self.min_lr
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter_num - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)