#!/usr/bin/env python3
"""
Quick validation test for Milestone 1: LR Scheduler.
This script verifies that the CosineLRScheduler produces identical results to the original get_lr function.
"""

import math
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Direct import to avoid torch dependency from evaluator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))
from scheduler import CosineLRScheduler


def original_get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr, decay_lr=True):
    """Original get_lr function from train.py for comparison."""
    if not decay_lr:
        return learning_rate
        
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def test_scheduler():
    """Test that CosineLRScheduler produces identical results to original get_lr."""
    # Test parameters (from default config)
    learning_rate = 6e-4
    warmup_iters = 2000
    lr_decay_iters = 600000
    min_lr = 6e-5
    decay_lr = True
    
    scheduler = CosineLRScheduler(
        learning_rate=learning_rate,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        decay_lr=decay_lr
    )
    
    # Test different iteration points
    test_iterations = [
        0, 1, 10, 100, 1000,           # warmup phase
        1999, 2000, 2001,              # warmup transition
        10000, 50000, 100000, 300000,  # decay phase
        599999, 600000, 600001, 700000 # min lr phase
    ]
    
    print("Testing LR Scheduler consistency...")
    all_passed = True
    
    for it in test_iterations:
        original_lr = original_get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr, decay_lr)
        scheduler_lr = scheduler.get_lr(it)
        
        if abs(original_lr - scheduler_lr) > 1e-10:
            print(f"FAIL: iter {it}: original={original_lr:.10f}, scheduler={scheduler_lr:.10f}")
            all_passed = False
        else:
            print(f"PASS: iter {it}: lr={original_lr:.8f}")
    
    # Test with decay_lr=False
    scheduler_no_decay = CosineLRScheduler(
        learning_rate=learning_rate,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        decay_lr=False
    )
    
    for it in [0, 1000, 10000, 100000]:
        original_lr = original_get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr, decay_lr=False)
        scheduler_lr = scheduler_no_decay.get_lr(it)
        
        if abs(original_lr - scheduler_lr) > 1e-10:
            print(f"FAIL (no decay): iter {it}: original={original_lr:.10f}, scheduler={scheduler_lr:.10f}")
            all_passed = False
        else:
            print(f"PASS (no decay): iter {it}: lr={original_lr:.8f}")
    
    return all_passed


if __name__ == "__main__":
    print("=" * 50)
    print("MILESTONE 1 VALIDATION TEST: LR SCHEDULER")
    print("=" * 50)
    
    # Test Milestone 1: LR Scheduler
    scheduler_passed = test_scheduler()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    print(f"Milestone 1 (LR Scheduler): {'PASS' if scheduler_passed else 'FAIL'}")
    
    if scheduler_passed:
        print("\n✅ LR Scheduler refactoring is functionally identical!")
        print("The CosineLRScheduler produces exact same results as original get_lr().")
    else:
        print("\n❌ LR Scheduler refactoring has issues - please review!")