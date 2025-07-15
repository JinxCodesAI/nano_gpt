#!/usr/bin/env python3
"""
Test script for validating rotary embeddings functionality in nanoGPT.
This script tests the basic functionality of rotary embeddings.
"""

import torch
from model import GPT, GPTConfig

def test_rotary_embeddings():
    """Test rotary embeddings functionality."""
    print("Testing rotary embeddings...")
    
    # Test configuration with rotary embeddings enabled
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_rotary_embeddings=True,
        rotary_base=10000.0,
        rotary_max_position_embeddings=128
    )
    
    # Create model
    model = GPT(config)
    print(f"Model created with rotary embeddings: {config.use_rotary_embeddings}")
    print(f"Total parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"Non-embedding parameters: {model.get_num_params(non_embedding=True)/1e6:.2f}M")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(x)
    
    print(f"Forward pass successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss}")
    
    # Test backward compatibility - create model without rotary embeddings
    config_no_rotary = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        use_rotary_embeddings=False
    )
    
    model_no_rotary = GPT(config_no_rotary)
    print(f"\nModel created without rotary embeddings: {config_no_rotary.use_rotary_embeddings}")
    print(f"Total parameters: {model_no_rotary.get_num_params()/1e6:.2f}M")
    print(f"Non-embedding parameters: {model_no_rotary.get_num_params(non_embedding=True)/1e6:.2f}M")
    
    with torch.no_grad():
        logits_no_rotary, loss_no_rotary = model_no_rotary(x)
    
    print(f"Forward pass successful!")
    print(f"Output shape: {logits_no_rotary.shape}")
    print(f"Loss: {loss_no_rotary}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_rotary_embeddings()