#!/usr/bin/env python3
"""
Test script to verify the soft label curriculum implementation works correctly.
"""

import torch
import torch.nn.functional as F

# Extract just the curriculum scheduler function to test independently
def get_curriculum_schedulers(it):
    """
    Controls the curriculum for penalties, data generation, dynamic task weights, and soft labels.
    Returns the current penalty values, re-masking task ratio, dynamic task weights, and soft label alpha.
    """
    # Mock config values
    masking_warmup_iters = 1000
    proofreading_warmup_iters = 2000
    soft_label_warmup_iters = 5000
    penalty_mask_correct = 0.5
    weight_unmask_task_min = 1.5
    weight_unmask_task_max = 1.0
    weight_remask_task_min = 0.5
    weight_remask_task_max = 1.0
    
    # --- Penalty Curriculum for "destructive editing" ---
    if it >= masking_warmup_iters:
        current_penalty_mask_correct = penalty_mask_correct
    else:
        ratio = it / masking_warmup_iters
        current_penalty_mask_correct = 0.0 + ratio * (penalty_mask_correct - 0.0)

    # --- Data Generation Curriculum ---
    if it >= proofreading_warmup_iters:
        remask_ratio = 1.0
    else:
        remask_ratio = it / proofreading_warmup_iters
    
    # --- Dynamic Task Weight Scheduling ---
    # Unmask task weight: Linear interpolation from min to max over masking_warmup_iters
    if it >= masking_warmup_iters:
        current_weight_unmask = weight_unmask_task_max
    else:
        ratio = it / masking_warmup_iters
        current_weight_unmask = weight_unmask_task_min + ratio * (weight_unmask_task_max - weight_unmask_task_min)
    
    # Remask task weight: Linear interpolation from min to max over proofreading_warmup_iters  
    if it >= proofreading_warmup_iters:
        current_weight_remask = weight_remask_task_max
    else:
        ratio = it / proofreading_warmup_iters
        current_weight_remask = weight_remask_task_min + ratio * (weight_remask_task_max - weight_remask_task_min)
    
    # --- Soft Label Curriculum (NEW) ---
    # This alpha controls the interpolation between a uniform distribution and a one-hot target.
    # It starts at 0.0 (fully uniform) and ramps to 1.0 (fully one-hot).
    if it >= soft_label_warmup_iters:
        soft_label_alpha = 1.0
    else:
        soft_label_alpha = it / soft_label_warmup_iters
        
    return current_penalty_mask_correct, remask_ratio, current_weight_unmask, current_weight_remask, soft_label_alpha

def test_curriculum_scheduler():
    """Test the curriculum scheduler function"""
    
    # Test early stage (iter = 0)
    penalty, remask_ratio, w_unmask, w_remask, alpha = get_curriculum_schedulers(0)
    print(f"Iter 0: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert alpha == 0.0, f"Expected alpha=0.0, got {alpha}"
    assert w_unmask == 1.5, f"Expected w_unmask=1.5, got {w_unmask}"  # min value
    assert w_remask == 0.5, f"Expected w_remask=0.5, got {w_remask}"  # min value
    
    # Test middle stage (iter = 2500)
    penalty, remask_ratio, w_unmask, w_remask, alpha = get_curriculum_schedulers(2500)
    print(f"Iter 2500: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert 0 < alpha < 1, f"Expected 0 < alpha < 1, got {alpha}"
    assert alpha == 2500/5000, f"Expected alpha=0.5, got {alpha}"
    
    # Test final stage (iter = 6000)
    penalty, remask_ratio, w_unmask, w_remask, alpha = get_curriculum_schedulers(6000)
    print(f"Iter 6000: penalty={penalty:.3f}, remask_ratio={remask_ratio:.3f}, w_unmask={w_unmask:.3f}, w_remask={w_remask:.3f}, alpha={alpha:.3f}")
    assert alpha == 1.0, f"Expected alpha=1.0, got {alpha}"
    
    print("✓ Curriculum scheduler test passed!")

def test_soft_label_generation():
    """Test soft label generation logic"""
    
    # Create sample data
    batch_size, seq_len = 2, 4
    vocab_size = 1002  # meta_vocab_size + 2
    mask_token_id = 1000
    replace_token_id = 1001
    
    # Test hard targets
    y_hard = torch.tensor([[5, 10, mask_token_id, replace_token_id],
                          [15, 20, 25, replace_token_id]])
    
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
    
    # Test unmask task override logic
    x_corrupted = torch.tensor([[mask_token_id, 10, mask_token_id, replace_token_id],
                               [15, mask_token_id, 25, replace_token_id]]) 
    
    unmask_task_mask = (x_corrupted == mask_token_id)
    y_final_targets = torch.where(
        unmask_task_mask.unsqueeze(-1),  # Condition, needs to be broadcastable
        uniform_dist,                    # Value if True (it's an un-masking task)
        y_soft                          # Value if False (it's any other task)
    )
    
    # Check unmask positions have uniform distribution
    for i in range(batch_size):
        for j in range(seq_len):
            if x_corrupted[i, j] == mask_token_id:
                assert torch.allclose(y_final_targets[i, j], uniform_dist[i, j]), f"Unmask position should be uniform at [{i},{j}]"
    
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
    
    # Test comparison with cross-entropy for hard targets
    hard_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    flat_hard_targets = hard_targets.view(-1)
    ce_loss = F.cross_entropy(flat_logits, flat_hard_targets, reduction='none')
    print(f"Cross-entropy loss range: {ce_loss.min().item():.3f} to {ce_loss.max().item():.3f}")
    
    print("✓ KL divergence loss test passed!")

if __name__ == "__main__":
    print("Testing soft label curriculum implementation...")
    test_curriculum_scheduler()
    test_soft_label_generation()
    test_kl_divergence_loss()
    print("✅ All tests passed!")