#!/usr/bin/env python3
"""
Test script to verify weight tying behavior in head switching.
"""

import torch
import torch.nn as nn
from model import GPT, GPTConfig

def test_weight_tying():
    """Test that weight tying is correctly managed during head switching"""
    print("=" * 60)
    print("Testing Weight Tying Behavior")
    print("=" * 60)
    
    # Create a small test model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False,
        binary_classification=False  # Start as language model
    )
    
    model = GPT(config)
    
    # Test 1: Initial state should have weight tying
    print("\n--- Test 1: Initial Weight Tying State ---")
    initial_wte_id = id(model.transformer.wte.weight)
    initial_lm_head_id = id(model.lm_head.weight)
    initial_tied = model.transformer.wte.weight is model.lm_head.weight
    
    print(f"Token embedding tensor ID: {initial_wte_id}")
    print(f"LM head tensor ID: {initial_lm_head_id}")
    print(f"Are weights tied? {initial_tied}")
    print(f"Initial parameter count: {model.get_trainable_param_count():,}")
    
    assert initial_tied, "Initial model should have weight tying between wte and lm_head"
    
    # Test 2: Switch to binary classification should break weight tying
    print("\n--- Test 2: Switch to Binary Classification ---")
    model.switch_to_binary_classification()
    
    binary_wte_id = id(model.transformer.wte.weight)
    binary_lm_head_id = id(model.lm_head.weight)
    binary_tied = model.transformer.wte.weight is model.lm_head.weight
    binary_param_count = model.get_trainable_param_count()
    
    print(f"Token embedding tensor ID: {binary_wte_id}")
    print(f"Binary head tensor ID: {binary_lm_head_id}")
    print(f"Are weights tied? {binary_tied}")
    print(f"Parameter count after switch: {binary_param_count:,}")
    
    assert not binary_tied, "Binary classification should not have weight tying"
    assert binary_wte_id != initial_wte_id, "Token embeddings should be independent after breaking tie"
    assert binary_param_count > model.get_num_params(), "Should have more trainable params than total model params initially"
    
    # Expected increase: 2 * n_embd (new binary head)
    expected_increase = 2 * config.n_embd
    actual_increase = binary_param_count - model.get_num_params()  # Compare against original model size
    print(f"Expected parameter increase: {expected_increase}")
    print(f"Actual parameter increase: {actual_increase}")
    
    # Test 3: Switch back to language modeling should restore weight tying
    print("\n--- Test 3: Switch Back to Language Modeling ---")
    model.switch_to_language_modeling(config.vocab_size)
    
    final_wte_id = id(model.transformer.wte.weight)
    final_lm_head_id = id(model.lm_head.weight)
    final_tied = model.transformer.wte.weight is model.lm_head.weight
    final_param_count = model.get_trainable_param_count()
    
    print(f"Token embedding tensor ID: {final_wte_id}")
    print(f"LM head tensor ID: {final_lm_head_id}")
    print(f"Are weights tied? {final_tied}")
    print(f"Final parameter count: {final_param_count:,}")
    
    assert final_tied, "Weight tying should be restored"
    assert final_wte_id == final_lm_head_id, "wte and lm_head should share the same tensor"
    
    # Parameter count should return to original (due to weight tying restoration)
    original_param_count = model.get_num_params()
    print(f"Original model param count: {original_param_count:,}")
    print(f"Final trainable param count: {final_param_count:,}")
    
    # With weight tying, trainable params should equal total model params
    assert final_param_count == original_param_count, f"Trainable params {final_param_count} should equal model params {original_param_count}"
    
    # Test 4: Verify the weights are actually the same values
    print("\n--- Test 4: Weight Value Verification ---")
    
    # The token embeddings and LM head should have identical values
    wte_weights = model.transformer.wte.weight
    lm_head_weights = model.lm_head.weight
    
    print(f"Token embedding shape: {wte_weights.shape}")
    print(f"LM head shape: {lm_head_weights.shape}")
    
    if wte_weights.shape == lm_head_weights.shape:
        weights_equal = torch.allclose(wte_weights, lm_head_weights)
        print(f"Weights are numerically equal: {weights_equal}")
        assert weights_equal, "Tied weights should be numerically identical"
    
    print("\n" + "=" * 60)
    print("🎉 ALL WEIGHT TYING TESTS PASSED!")
    print("✅ Weight Tying Behavior is Correct")
    print("=" * 60)

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    test_weight_tying()