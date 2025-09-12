#!/usr/bin/env python3
"""
Test script to verify out-of-bounds target fix in EntropyModifier
"""
import os
import torch
import torch.nn.functional as F

# Set CUDA_LAUNCH_BLOCKING for better error reporting
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from loss_modifiers.entropy_modifier import EntropyModifier

def test_bounds_fix():
    """Test entropy modifier with out-of-bounds target scenarios"""
    print("Testing EntropyModifier out-of-bounds target fix...")
    
    config = {
        'enabled': True,
        'weight': 0.3,
        'entropy_threshold': 0.1,
        'eps': 1e-8,
        'verbose': True  # Enable to see bounds warnings
    }
    
    modifier = EntropyModifier(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size, seq_len, vocab_size = 2, 4, 1000
    
    # Normal case first
    print("\n=== Test 1: Normal case ===")
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    
    try:
        modified_loss = modifier.modify_loss(logits, targets, loss, mask=mask)
        print(f"✓ Normal case: loss={loss.item():.6f} -> modified_loss={modified_loss.item():.6f}")
    except Exception as e:
        print(f"✗ Normal case failed: {e}")
        return False
    
    # Out-of-bounds case
    print("\n=== Test 2: Out-of-bounds targets ===")
    oob_targets = targets.clone()
    oob_targets[0, 0] = vocab_size + 100  # Way too high
    oob_targets[0, 1] = -50  # Negative (not ignore_index)
    oob_targets[1, 0] = -100  # Standard ignore_index (should be handled)
    oob_targets[1, 1] = vocab_size - 1  # Valid edge case
    
    print(f"Target range: [{oob_targets.min().item()}, {oob_targets.max().item()}]")
    print(f"Vocab size: {vocab_size}")
    
    try:
        modified_loss = modifier.modify_loss(logits, oob_targets, loss, mask=mask)
        print(f"✓ Out-of-bounds case handled: loss={loss.item():.6f} -> modified_loss={modified_loss.item():.6f}")
        print("✓ No CUDA scatter/gather error!")
        return True
    except Exception as e:
        print(f"✗ Out-of-bounds case failed: {e}")
        return False

if __name__ == "__main__":
    success = test_bounds_fix()
    if success:
        print("\n🎉 Out-of-bounds target fix verified!")
        print("The scatter/gather CUDA error should now be resolved.")
    else:
        print("\n❌ Tests failed - issue not fully resolved")