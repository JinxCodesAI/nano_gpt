#!/usr/bin/env python3
"""
Comprehensive test for head switching with weight tying verification.
"""

import torch
import torch.nn as nn
from model import GPT, GPTConfig

def test_comprehensive_switching():
    """Comprehensive test of head switching behavior"""
    print("=" * 70)
    print("Comprehensive Head Switching Test with Weight Tying Verification")
    print("=" * 70)
    
    # Create test model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=64,  # Smaller for easier verification
        dropout=0.0,
        bias=False,
        binary_classification=False
    )
    
    model = GPT(config)
    
    # Expected parameter counts
    expected_wte_params = config.vocab_size * config.n_embd  # 100 * 64 = 6400
    expected_binary_head_params = 2 * config.n_embd  # 2 * 64 = 128
    
    print(f"Expected token embedding params: {expected_wte_params}")
    print(f"Expected binary head params: {expected_binary_head_params}")
    
    # Phase 1: Initial Language Model State
    print("\n" + "=" * 50)
    print("PHASE 1: Initial Language Model State")
    print("=" * 50)
    
    initial_total_params = sum(p.numel() for p in model.parameters())
    initial_trainable_params = model.get_trainable_param_count()
    
    print(f"Total parameters: {initial_total_params:,}")
    print(f"Trainable parameters: {initial_trainable_params:,}")
    print(f"get_num_params(): {model.get_num_params():,}")
    
    # Check weight tying
    wte_tied = model.transformer.wte.weight is model.lm_head.weight
    print(f"Weight tying active: {wte_tied}")
    print(f"wte shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head shape: {model.lm_head.weight.shape}")
    
    assert wte_tied, "Initial model should have weight tying"
    assert initial_total_params == initial_trainable_params, "All params should be trainable initially"
    
    # Phase 2: Switch to Binary Classification
    print("\n" + "=" * 50)
    print("PHASE 2: Switch to Binary Classification")
    print("=" * 50)
    
    model.switch_to_binary_classification()
    
    binary_total_params = sum(p.numel() for p in model.parameters())
    binary_trainable_params = model.get_trainable_param_count()
    
    print(f"Total parameters: {binary_total_params:,}")
    print(f"Trainable parameters: {binary_trainable_params:,}")
    print(f"get_num_params(): {model.get_num_params():,}")
    
    # Check weight tying is broken
    wte_tied = model.transformer.wte.weight is model.lm_head.weight
    print(f"Weight tying active: {wte_tied}")
    print(f"wte shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head shape: {model.lm_head.weight.shape}")
    
    assert not wte_tied, "Binary classification should break weight tying"
    
    # Parameter count should increase by binary head size
    param_increase = binary_trainable_params - initial_trainable_params
    print(f"Parameter increase: {param_increase} (expected: {expected_binary_head_params})")
    assert param_increase == expected_binary_head_params, f"Expected increase of {expected_binary_head_params}, got {param_increase}"
    
    # Phase 3: Switch Back to Language Model
    print("\n" + "=" * 50)
    print("PHASE 3: Switch Back to Language Model")
    print("=" * 50)
    
    model.switch_to_language_modeling(config.vocab_size)
    
    final_total_params = sum(p.numel() for p in model.parameters())
    final_trainable_params = model.get_trainable_param_count()
    
    print(f"Total parameters: {final_total_params:,}")
    print(f"Trainable parameters: {final_trainable_params:,}")
    print(f"get_num_params(): {model.get_num_params():,}")
    
    # Check weight tying is restored
    wte_tied = model.transformer.wte.weight is model.lm_head.weight
    print(f"Weight tying active: {wte_tied}")
    print(f"wte shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head shape: {model.lm_head.weight.shape}")
    
    assert wte_tied, "Weight tying should be restored"
    
    # Parameter count should return to original
    print(f"Final trainable params: {final_trainable_params:,}")
    print(f"Initial trainable params: {initial_trainable_params:,}")
    print(f"Difference: {final_trainable_params - initial_trainable_params}")
    
    # This is the key test: trainable params should return to original count
    assert final_trainable_params == initial_trainable_params, f"Final params {final_trainable_params} should equal initial params {initial_trainable_params}"
    
    # Phase 4: Verify Forward Pass Works
    print("\n" + "=" * 50)
    print("PHASE 4: Forward Pass Verification")
    print("=" * 50)
    
    batch_size = 4
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test language model forward pass
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    
    expected_shape = (batch_size, 1, config.vocab_size)  # Causal attention in inference mode
    print(f"Forward pass output shape: {logits.shape}")
    print(f"Expected shape: {expected_shape}")
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    # Test training mode
    model.train()
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(x, targets)
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    print(f"Training mode output shape: {logits.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Loss: {loss.item():.4f}")
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    assert loss is not None and torch.isfinite(loss), "Loss should be finite"
    
    print("\n" + "=" * 70)
    print("🎉 ALL COMPREHENSIVE SWITCHING TESTS PASSED!")
    print("✅ Weight tying is correctly restored")
    print("✅ Parameter counts are correct")
    print("✅ Forward passes work correctly")
    print("=" * 70)

if __name__ == "__main__":
    torch.manual_seed(42)
    test_comprehensive_switching()