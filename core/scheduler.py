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


class TransferLearningScheduler(LRScheduler):
    """
    Specialized scheduler for transfer learning with feature extraction and fine-tuning phases.
    
    Features:
    - Lower learning rates optimized for fine-tuning
    - Different schedules for frozen vs unfrozen phases
    - Smooth transitions when unfreezing
    """
    
    def __init__(
        self,
        base_lr: float = 5e-5,           # Lower LR for fine-tuning
        head_lr_multiplier: float = 10.0,  # Higher LR for new heads
        warmup_iters: int = 500,         # Shorter warmup
        feature_extraction_iters: int = 2000,  # Frozen phase duration
        unfreeze_lr_drop: float = 0.1,   # LR reduction when unfreezing
        decay_lr: bool = True,
        min_lr_ratio: float = 0.1
    ):
        self.base_lr = base_lr
        self.head_lr_multiplier = head_lr_multiplier
        self.warmup_iters = warmup_iters
        self.feature_extraction_iters = feature_extraction_iters
        self.unfreeze_lr_drop = unfreeze_lr_drop
        self.decay_lr = decay_lr
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr(self, iter_num: int, is_frozen: bool = True) -> dict:
        """
        Get learning rates for different parameter groups.
        
        Returns:
            dict with 'transformer' and 'head' learning rates
        """
        # Phase 1: Feature extraction (transformer frozen)
        if iter_num < self.feature_extraction_iters and is_frozen:
            if iter_num < self.warmup_iters:
                # Warmup for head only
                progress = (iter_num + 1) / (self.warmup_iters + 1)
                head_lr = self.base_lr * self.head_lr_multiplier * progress
            else:
                # Constant LR for head, frozen transformer
                head_lr = self.base_lr * self.head_lr_multiplier
            
            return {
                'transformer': 0.0,  # Frozen
                'head': head_lr
            }
        
        # Phase 2: Full fine-tuning (transformer unfrozen)
        else:
            if not is_frozen:
                # Apply LR drop when first unfreezing
                effective_base_lr = self.base_lr * self.unfreeze_lr_drop
            else:
                effective_base_lr = self.base_lr
            
            # Cosine decay from unfreeze point
            if self.decay_lr and iter_num > self.feature_extraction_iters:
                decay_progress = (iter_num - self.feature_extraction_iters) / max(1, iter_num - self.feature_extraction_iters + 1000)
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * min(decay_progress, 1.0)))
                transformer_lr = self.min_lr_ratio * effective_base_lr + decay_factor * (effective_base_lr - self.min_lr_ratio * effective_base_lr)
            else:
                transformer_lr = effective_base_lr
            
            return {
                'transformer': transformer_lr,
                'head': transformer_lr * 2.0  # Slightly higher for head
            }


class WarmupOnlyScheduler(LRScheduler):
    """
    Simple warmup-only scheduler for short fine-tuning runs.
    Good for classification tasks with small datasets.
    """
    
    def __init__(
        self,
        learning_rate: float,
        warmup_iters: int,
        hold_iters: int = None,  # Hold at peak LR, then decay
        min_lr_ratio: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.hold_iters = hold_iters or warmup_iters
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr(self, iter_num: int) -> float:
        """Simple warmup then hold/decay schedule"""
        if iter_num < self.warmup_iters:
            # Linear warmup
            return self.learning_rate * (iter_num + 1) / (self.warmup_iters + 1)
        elif iter_num < self.warmup_iters + self.hold_iters:
            # Hold at peak
            return self.learning_rate
        else:
            # Linear decay to minimum
            decay_progress = (iter_num - self.warmup_iters - self.hold_iters) / max(1, iter_num - self.warmup_iters - self.hold_iters + 100)
            return self.learning_rate * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - min(decay_progress, 1.0)))


class AdaptiveScheduler(LRScheduler):
    """
    Adaptive scheduler that adjusts based on validation performance.
    Useful for fine-tuning when optimal schedule length is unknown.
    """
    
    def __init__(
        self,
        initial_lr: float,
        patience: int = 3,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        warmup_iters: int = 100
    ):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.best_loss = float('inf')
        self.wait_count = 0
        self.reductions = 0
    
    def get_lr(self, iter_num: int) -> float:
        """Get current learning rate"""
        if iter_num < self.warmup_iters:
            # Linear warmup to current LR
            return self.current_lr * (iter_num + 1) / (self.warmup_iters + 1)
        return self.current_lr
    
    def step(self, val_loss: float) -> bool:
        """
        Update scheduler based on validation loss.
        Returns True if LR was reduced.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait_count = 0
                self.reductions += 1
                return self.current_lr < old_lr
        return False