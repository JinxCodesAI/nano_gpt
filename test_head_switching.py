#!/usr/bin/env python3
"""
Test script for model head switching functionality (Milestone 1).
Tests the ability to switch between language modeling and binary classification heads.
"""

import torch
import torch.nn as nn
from model import GPT, GPTConfig

def test_head_switching():
    """Test head switching between language modeling and binary classification"""
    print("=" * 60)
    print("Testing Model Head Switching (Milestone 1)")
    print("=" * 60)
    
    # Create a small test model configuration
    vocab_size = 1000
    config = GPTConfig(
        block_size=64,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False,
        binary_classification=False  # Start as language model
    )
    
    print(f"Creating model with vocab_size={vocab_size}, binary_classification={config.binary_classification}")
    model = GPT(config)
    
    # Test 1: Initial state should be language modeling
    print("\n--- Test 1: Initial Language Modeling State ---")
    assert model.lm_head.out_features == vocab_size, f"Expected {vocab_size} outputs, got {model.lm_head.out_features}"
    assert model.config.binary_classification == False, "Config should indicate language modeling mode"
    print("✓ Model correctly initialized in language modeling mode")
    
    # Test 2: Switch to binary classification
    print("\n--- Test 2: Switch to Binary Classification ---")
    initial_param_count = model.get_trainable_param_count()
    print(f"Initial trainable parameters: {initial_param_count:,}")
    
    model.switch_to_binary_classification()
    
    assert model.lm_head.out_features == 2, f"Expected 2 outputs after switch, got {model.lm_head.out_features}"
    assert model.config.binary_classification == True, "Config should indicate binary classification mode"
    
    binary_param_count = model.get_trainable_param_count()
    print(f"Parameters after switch to binary: {binary_param_count:,}")
    
    # Parameter count should increase due to breaking weight tying:
    # Before: Token embeddings (vocab_size * n_embd) + LM head (SHARED via weight tying) = vocab_size * n_embd total
    # After:  Token embeddings (vocab_size * n_embd) + Binary head (2 * n_embd, independent) = vocab_size * n_embd + 2 * n_embd total  
    # Net increase: 2 * n_embd parameters (the new independent binary head)
    binary_head_params = 2 * config.n_embd
    expected_param_count = initial_param_count + binary_head_params
    assert binary_param_count == expected_param_count, f"Expected {expected_param_count} params, got {binary_param_count}"
    print("✓ Successfully switched to binary classification mode")
    
    # Test 3: Switch back to language modeling
    print("\n--- Test 3: Switch Back to Language Modeling ---")
    model.switch_to_language_modeling(vocab_size)
    
    assert model.lm_head.out_features == vocab_size, f"Expected {vocab_size} outputs after switch back, got {model.lm_head.out_features}"
    assert model.config.binary_classification == False, "Config should indicate language modeling mode"
    assert model.config.vocab_size == vocab_size, f"Config vocab_size should be {vocab_size}, got {model.config.vocab_size}"
    
    final_param_count = model.get_trainable_param_count()
    print(f"Parameters after switch back to LM: {final_param_count:,}")
    # When switching back to LM, weight tying should be restored, so params should match initial count
    assert final_param_count == initial_param_count, f"Expected {initial_param_count} params, got {final_param_count}"
    print("✓ Successfully switched back to language modeling mode")
    
    # Test 4: Test idempotency (switching when already in target mode)
    print("\n--- Test 4: Idempotency Tests ---")
    print("Testing switch to binary when already binary...")
    model.switch_to_binary_classification()
    model.switch_to_binary_classification()  # Should be no-op
    assert model.lm_head.out_features == 2, "Should remain binary after redundant switch"
    
    print("Testing switch to LM when already LM...")
    model.switch_to_language_modeling(vocab_size)
    model.switch_to_language_modeling(vocab_size)  # Should be no-op  
    assert model.lm_head.out_features == vocab_size, "Should remain LM after redundant switch"
    print("✓ Idempotency tests passed")
    
    # Test 5: Test forward pass works after switching
    print("\n--- Test 5: Forward Pass Tests ---")
    batch_size = 4
    seq_len = 32
    
    # Test in language modeling mode
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # For causal attention in inference mode, only last position is computed
    # So we expect shape (batch_size, 1, vocab_size)
    logits, _ = model(x)
    expected_seq_len = 1 if model.config.attention_type == 'causal' else seq_len
    assert logits.shape == (batch_size, expected_seq_len, vocab_size), f"LM forward pass shape mismatch: expected {(batch_size, expected_seq_len, vocab_size)}, got {logits.shape}"
    print("✓ Language modeling forward pass works")
    
    # Test in binary classification mode - always computes all positions
    model.switch_to_binary_classification()
    logits, _ = model(x)
    assert logits.shape == (batch_size, seq_len, 2), f"Binary forward pass shape mismatch: {logits.shape}"
    print("✓ Binary classification forward pass works")
    
    # Test with training mode (should compute all positions for both modes)
    print("Testing in training mode...")
    model.train()
    targets = torch.randint(0, 2, (batch_size, seq_len))  # Binary targets
    logits, loss = model(x, targets)
    assert logits.shape == (batch_size, seq_len, 2), f"Training mode binary forward pass shape mismatch: {logits.shape}"
    assert loss is not None, "Loss should be computed in training mode"
    
    # Switch back to LM and test training mode
    model.switch_to_language_modeling(vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))  # LM targets
    logits, loss = model(x, targets)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Training mode LM forward pass shape mismatch: {logits.shape}"
    assert loss is not None, "Loss should be computed in training mode"
    
    model.eval()  # Return to eval mode
    print("✓ Training mode forward passes work")
    
    # Test 6: Test device handling
    print("\n--- Test 6: Device Handling ---")
    if torch.cuda.is_available():
        device = 'cuda'
        model = model.to(device)
        x = x.to(device)
        
        # Test switching while on GPU
        model.switch_to_language_modeling(vocab_size)
        assert model.lm_head.weight.device.type == device, "Head should be on correct device after switch"
        
        logits, _ = model(x)
        assert logits.device.type == device, "Output should be on correct device"
        print("✓ Device handling works correctly")
    else:
        print("⚠ CUDA not available, skipping device test")
    
    print("\n" + "=" * 60)
    print("🎉 ALL HEAD SWITCHING TESTS PASSED!")
    print("✅ Milestone 1: Model Head Switching - COMPLETE")
    print("=" * 60)

def test_trainable_param_counting():
    """Test the trainable parameter counting functionality"""
    print("\n--- Bonus Test: Trainable Parameter Counting ---")
    
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False,
        binary_classification=False
    )
    
    model = GPT(config)
    
    # All parameters should be trainable initially
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_param_count()
    assert total_params == trainable_params, "All params should be trainable initially"
    
    # Freeze some parameters manually
    for name, param in model.named_parameters():
        if 'wte' in name:  # Freeze token embeddings
            param.requires_grad = False
    
    new_trainable = model.get_trainable_param_count()
    assert new_trainable < trainable_params, "Trainable count should decrease after freezing"
    
    print("✓ Trainable parameter counting works correctly")

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    test_head_switching()
    test_trainable_param_counting()