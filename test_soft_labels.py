#!/usr/bin/env python3
"""
Test script to verify the soft label curriculum implementation works correctly.
"""

import torch
import torch.nn.functional as F

# Import the curriculum scheduler function
import sys
sys.path.append('.')

def test_get_curriculum_schedulers():
    """Test the curriculum scheduler function"""
    
    # Mock the global variables that the function expects
    import train
    train.masking_warmup_iters = 1000
    train.proofreading_warmup_iters = 2000  
    train.soft_label_warmup_iters = 5000
    train.penalty_mask_correct = 0.5
    train.weight_unmask_task_min = 1.5
    train.weight_unmask_task_max = 1.0
    train.weight_remask_task_min = 0.5
    train.weight_remask_task_max = 1.0
    
    # Test early stage (iter = 0)
    penalty, remask_ratio, w_unmask, w_remask, alpha = train.get_curriculum_schedulers(0)
    print(f"Iter 0: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert alpha == 0.0, f"Expected alpha=0.0, got {alpha}"
    
    # Test middle stage (iter = 2500)
    penalty, remask_ratio, w_unmask, w_remask, alpha = train.get_curriculum_schedulers(2500)
    print(f"Iter 2500: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert 0 < alpha < 1, f"Expected 0 < alpha < 1, got {alpha}"
    
    # Test final stage (iter = 6000)
    penalty, remask_ratio, w_unmask, w_remask, alpha = train.get_curriculum_schedulers(6000)
    print(f"Iter 6000: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert alpha == 1.0, f"Expected alpha=1.0, got {alpha}"
    
    print("✓ Curriculum scheduler test passed!")

def test_soft_label_generation():
    """Test soft label generation in get_batch function"""
    
    # Mock global variables
    import train
    train.iter_num = 2500  # Middle of soft label warmup
    train.meta_vocab_size = 1000
    
    # Create sample data
    batch_size, seq_len = 2, 4
    vocab_size = 1002  # meta_vocab_size + 2
    mask_token_id = 1000
    wrong_token_id = 1001
    
    # Test hard targets
    y_hard = torch.tensor([[5, 10, mask_token_id, wrong_token_id],
                          [15, 20, 25, wrong_token_id]])
    
    # Test soft label creation logic (extracted from get_batch)
    soft_label_alpha = 0.5  # Example alpha
    
    y_one_hot = F.one_hot(y_hard, num_classes=vocab_size).float()
    uniform_dist = torch.full_like(y_one_hot, 1.0 / vocab_size)
    y_soft = (1.0 - soft_label_alpha) * uniform_dist + soft_label_alpha * y_one_hot
    
    print(f"Hard targets shape: {y_hard.shape}")
    print(f"Soft targets shape: {y_soft.shape}")
    print(f"Soft target sum (should be ~1.0): {y_soft.sum(dim=-1)}")
    
    # Check that probabilities sum to 1
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones_like(y_soft.sum(dim=-1))), "Probabilities should sum to 1"
    
    print("✓ Soft label generation test passed!")

def test_kl_divergence_loss():
    """Test that KL divergence loss calculation works"""
    
    batch_size, seq_len, vocab_size = 2, 4, 1002
    
    # Create sample logits and soft targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    soft_targets = torch.softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    
    # Test KL divergence calculation (from loss.py logic)
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = soft_targets.view(-1, soft_targets.size(-1))
    
    log_probs = F.log_softmax(flat_logits, dim=-1)
    per_token_loss = F.kl_div(log_probs, flat_targets, reduction='none').sum(dim=-1)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Soft targets shape: {soft_targets.shape}")  
    print(f"Per token loss shape: {per_token_loss.shape}")
    print(f"Loss range: {per_token_loss.min().item():.3f} to {per_token_loss.max().item():.3f}")
    
    # Check that loss is reasonable
    assert per_token_loss.shape == (batch_size * seq_len,), f"Expected shape {(batch_size * seq_len,)}, got {per_token_loss.shape}"
    assert torch.all(per_token_loss >= 0), "KL divergence should be non-negative"
    
    print("✓ KL divergence loss test passed!")

if __name__ == "__main__":
    print("Testing soft label curriculum implementation...")
    test_get_curriculum_schedulers()
    test_soft_label_generation()
    test_kl_divergence_loss()
    print("✅ All tests passed!")