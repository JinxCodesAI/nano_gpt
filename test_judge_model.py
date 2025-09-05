#!/usr/bin/env python3
"""
Test script for judge model integration.

This script tests:
1. Judge model loading
2. Predicted masking ratio calculation
3. Wrongness factor calculation
4. Integration with training loop components
"""

import os
import sys
import torch
import numpy as np
from contextlib import nullcontext

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT, ModelMode
from training_utils import TrainingContext
from training_utils.model_initializer import ModelInitializer
from training_utils.loss_processing import (
    calculate_predicted_masking_ratio, 
    calculate_wrongness_factor,
    calculate_per_sample_losses,
    apply_per_sample_modifications
)

def create_mock_sequence_scorer(vocab_size, block_size, device):
    """Create a mock sequence scorer model for testing."""
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=2,  # Small model for testing
        n_head=2,
        n_embd=64,
        dropout=0.0,
        mode=ModelMode.SEQUENCE_SCORER,
        attention_type='bidirectional',
        cls_token_id=vocab_size - 1
    )
    
    model = GPT(config)
    model.to(device)
    model.eval()
    
    return model

def create_mock_unmasking_model(vocab_size, block_size, device):
    """Create a mock unmasking model for testing."""
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=2,  # Small model for testing
        n_head=2,
        n_embd=64,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional'
    )
    
    model = GPT(config)
    model.to(device)
    model.eval()
    
    return model

def create_test_context(device='cpu'):
    """Create a test training context."""
    vocab_size = 100
    block_size = 64
    batch_size = 4
    
    ctx = TrainingContext(
        training_type='unmasking',
        batch_size=batch_size,
        block_size=block_size,
        device=device,
        device_type=device,
        extended_vocab_size=vocab_size,
        mask_token_id=vocab_size - 10,
        cls_token_id=vocab_size - 1,
        enable_entropy_penalty=False
    )
    
    # Create mock models
    ctx.unmasking_model = create_mock_unmasking_model(vocab_size, block_size, device)
    ctx.judge_model = create_mock_sequence_scorer(vocab_size, block_size, device)
    
    return ctx

def test_wrongness_factor_calculation():
    """Test the wrongness factor calculation formula."""
    print("Testing wrongness factor calculation...")
    
    # Test cases
    predicted = torch.tensor([0.5, 0.3, 0.8, 0.1])
    real = torch.tensor([0.4, 0.2, 0.9, 0.05])
    
    wrongness = calculate_wrongness_factor(predicted, real)
    
    print(f"Predicted: {predicted.tolist()}")
    print(f"Real: {real.tolist()}")
    print(f"Wrongness: {wrongness.tolist()}")
    
    # Check that results are in expected range [0, 10]
    assert torch.all(wrongness >= 0.0), "Wrongness factor should be >= 0"
    assert torch.all(wrongness <= 10.0), "Wrongness factor should be <= 10"
    
    # Check specific cases
    # Case 1: predicted=0.5, real=0.4 -> wrongness = 0.5/(0.4+0.1) = 0.5/0.5 = 1.0
    assert abs(wrongness[0].item() - 1.0) < 0.001, f"Expected ~1.0, got {wrongness[0].item()}"
    
    print("✓ Wrongness factor calculation test passed")

def test_predicted_masking_ratio():
    """Test predicted masking ratio calculation."""
    print("Testing predicted masking ratio calculation...")
    
    device = 'cpu'
    ctx_training = create_test_context(device)
    ctx_autocast = nullcontext()
    
    # Create test data
    batch_size = ctx_training.batch_size
    seq_len = ctx_training.block_size
    
    # Create original sequences (random tokens)
    original_sequences = torch.randint(0, ctx_training.extended_vocab_size - 10, 
                                     (batch_size, seq_len), device=device)
    
    # Create random mask (30% masking)
    mask = torch.rand(batch_size, seq_len, device=device) < 0.3
    
    try:
        # Test the function
        predicted_ratios = calculate_predicted_masking_ratio(
            original_sequences, mask, ctx_training, ctx_autocast
        )
        
        print(f"Predicted ratios shape: {predicted_ratios.shape}")
        print(f"Predicted ratios: {predicted_ratios.tolist()}")
        
        # Check output shape and range
        assert predicted_ratios.shape == (batch_size,), f"Expected shape ({batch_size},), got {predicted_ratios.shape}"
        assert torch.all(predicted_ratios >= 0.0), "Predicted ratios should be >= 0"
        assert torch.all(predicted_ratios <= 1.0), "Predicted ratios should be <= 1"
        
        print("✓ Predicted masking ratio test passed")
        
    except Exception as e:
        print(f"✗ Predicted masking ratio test failed: {e}")
        raise

def test_integration():
    """Test integration with per-sample loss processing."""
    print("Testing integration with loss processing...")
    
    device = 'cpu'
    ctx_training = create_test_context(device)
    
    # Create test data
    batch_size = ctx_training.batch_size
    seq_len = ctx_training.block_size
    vocab_size = ctx_training.extended_vocab_size
    
    # Create mock logits, targets, and mask
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size - 10, (batch_size, seq_len), device=device)
    mask = torch.rand(batch_size, seq_len, device=device) < 0.3
    
    try:
        # Calculate per-sample losses
        per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, targets, mask)
        
        # Calculate wrongness factor
        ctx_autocast = nullcontext()
        predicted_ratios = calculate_predicted_masking_ratio(targets, mask, ctx_training, ctx_autocast)
        real_ratios = mask.float().mean(dim=1)
        wrongness_factor = calculate_wrongness_factor(predicted_ratios, real_ratios)
        
        # Apply per-sample modifications
        modified_losses = apply_per_sample_modifications(
            per_sample_losses, logits, targets, mask, ctx_training, 0, wrongness_factor
        )
        
        print(f"Original losses: {per_sample_losses.tolist()}")
        print(f"Wrongness factors: {wrongness_factor.tolist()}")
        print(f"Modified losses: {modified_losses.tolist()}")
        
        # Check that modification was applied
        assert not torch.allclose(per_sample_losses, modified_losses), "Losses should be modified"
        
        print("✓ Integration test passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise

def main():
    """Run all tests."""
    print("Running judge model integration tests...\n")
    
    try:
        test_wrongness_factor_calculation()
        print()
        
        test_predicted_masking_ratio()
        print()
        
        test_integration()
        print()
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
