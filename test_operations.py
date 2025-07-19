#!/usr/bin/env python3
"""
Test script for the new morphing operations in the training orchestrator.
"""

import torch
import torch.nn as nn
from model import GPTConfig, GPT

def test_stack_layers():
    """Test the stack_layers operation."""
    print("Testing stack_layers operation...")
    
    config = GPTConfig(n_layer=4, n_head=4, n_embd=64, vocab_size=1000, block_size=32)
    model = GPT(config)
    
    original_layers = model.config.n_layer
    print(f"Original layers: {original_layers}")
    
    # Test stacking layers
    model.stack_layers(2)
    new_layers = model.config.n_layer
    print(f"New layers after stacking: {new_layers}")
    
    assert new_layers == original_layers * 2, f"Expected {original_layers * 2}, got {new_layers}"
    print("✓ stack_layers test passed\n")

def test_widen_mlp():
    """Test the widen_mlp operation."""
    print("Testing widen_mlp operation...")
    
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=1000, block_size=32, n_hidden=128)
    model = GPT(config)
    
    original_hidden = model.config.n_hidden or 4 * model.config.n_embd
    print(f"Original hidden dim: {original_hidden}")
    
    # Test widening MLP
    model.widen_mlp(1.5)
    new_hidden = model.config.n_hidden
    print(f"New hidden dim after widening: {new_hidden}")
    
    expected_hidden = int(original_hidden * 1.5)
    assert new_hidden == expected_hidden, f"Expected {expected_hidden}, got {new_hidden}"
    print("✓ widen_mlp test passed\n")

def test_lora_operations():
    """Test LoRA-related operations."""
    print("Testing LoRA operations...")
    
    config = GPTConfig(
        n_layer=2, n_head=4, n_embd=64, vocab_size=1000, block_size=32,
        embedding_mode='lora', embedding_rank=16, attn_lora_rank=8, lora_alpha=2.0
    )
    model = GPT(config)
    
    print(f"Original LoRA alpha: {model.config.lora_alpha}")
    print(f"Original embedding rank: {model.config.embedding_rank}")
    print(f"Original attention LoRA rank: {model.config.attn_lora_rank}")
    
    # Test resize operations with proper integer ranks
    model.resize_lora_rank(4)  # resize to rank 4
    model.resize_embedding_rank(8)  # resize to rank 8
    
    # Test merge operation
    model.merge_lora_weights()
    
    print("✓ LoRA operations test passed\n")

def test_forward_pass():
    """Test that the model can still do forward passes after operations."""
    print("Testing forward pass after operations...")
    
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=1000, block_size=32)
    model = GPT(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test forward pass before operations
    with torch.no_grad():
        logits1, _ = model(x)
    
    print(f"Output shape before operations: {logits1.shape}")
    
    # Apply operations
    model.stack_layers(2)
    model.widen_mlp(1.5)
    
    # Test forward pass after operations
    with torch.no_grad():
        logits2, _ = model(x)
    
    print(f"Output shape after operations: {logits2.shape}")
    
    # Output should have same shape (vocab_size dimension)
    assert logits1.shape[-1] == logits2.shape[-1], f"Vocab dimension changed: {logits1.shape[-1]} -> {logits2.shape[-1]}"
    assert logits1.shape[:-1] == logits2.shape[:-1], f"Sequence shape changed: {logits1.shape[:-1]} -> {logits2.shape[:-1]}"
    
    print("✓ Forward pass test passed\n")

def test_parameter_count():
    """Test that parameter counts increase as expected."""
    print("Testing parameter count changes...")
    
    config = GPTConfig(n_layer=2, n_head=4, n_embd=64, vocab_size=1000, block_size=32, n_hidden=128)
    model = GPT(config)
    
    original_params = model.get_num_params()
    print(f"Original parameter count: {original_params:,}")
    
    # Stack layers should roughly double parameters (minus embeddings)
    model.stack_layers(2)
    stacked_params = model.get_num_params()
    print(f"Parameter count after stacking: {stacked_params:,}")
    
    # Widen MLP should increase parameters
    model.widen_mlp(1.5)
    widened_params = model.get_num_params()
    print(f"Parameter count after widening: {widened_params:,}")
    
    assert stacked_params > original_params, "Parameter count should increase after stacking"
    assert widened_params > stacked_params, "Parameter count should increase after widening"
    
    print("✓ Parameter count test passed\n")

def main():
    """Run all tests."""
    print("Running morphing operations tests...\n")
    
    try:
        test_stack_layers()
        test_widen_mlp()
        test_lora_operations()
        test_forward_pass()
        test_parameter_count()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main()