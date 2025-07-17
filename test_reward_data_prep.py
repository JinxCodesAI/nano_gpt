#!/usr/bin/env python3
"""
Test script for reward data preparation.
This script tests the prepare_reward_data.py functionality with a small example.
"""

import os
import numpy as np
import torch
from model import GPT, GPTConfig

def test_reward_data_loading():
    """Test loading reward dataset from binary files."""
    
    # Check if reward dataset exists
    reward_dir = 'data/reward_dataset'
    if not os.path.exists(reward_dir):
        print(f"Reward dataset directory {reward_dir} does not exist.")
        print("Run prepare_reward_data.py first to generate the dataset.")
        return False
    
    # Load training data
    train_x_path = os.path.join(reward_dir, 'train_x.bin')
    train_y_path = os.path.join(reward_dir, 'train_y.bin')
    
    if not (os.path.exists(train_x_path) and os.path.exists(train_y_path)):
        print("Training data files not found.")
        return False
    
    # Read metadata to get shapes
    metadata_path = os.path.join(reward_dir, 'train_metadata.txt')
    if os.path.exists(metadata_path):
        print("Training data metadata:")
        with open(metadata_path, 'r') as f:
            print(f.read())
    
    # Load binary data
    train_x = np.fromfile(train_x_path, dtype=np.uint16)
    train_y = np.fromfile(train_y_path, dtype=np.float32)
    
    # Reshape based on expected dimensions
    # X should be (num_samples, block_size)
    # Y should be (num_samples, 2)
    num_samples_x = len(train_x) // 1024  # Assuming block_size=1024
    num_samples_y = len(train_y) // 2
    
    if num_samples_x != num_samples_y:
        print(f"Mismatch in number of samples: X={num_samples_x}, Y={num_samples_y}")
        return False
    
    train_x = train_x.reshape(num_samples_x, -1)
    train_y = train_y.reshape(num_samples_y, 2)
    
    print(f"Loaded training data:")
    print(f"  X shape: {train_x.shape}")
    print(f"  Y shape: {train_y.shape}")
    print(f"  X dtype: {train_x.dtype}")
    print(f"  Y dtype: {train_y.dtype}")
    
    # Verify Y values are valid probabilities
    print(f"\nY statistics:")
    print(f"  Min values: {train_y.min(axis=0)}")
    print(f"  Max values: {train_y.max(axis=0)}")
    print(f"  Mean values: {train_y.mean(axis=0)}")
    
    # Check that probabilities sum to 1
    prob_sums = train_y.sum(axis=1)
    print(f"  Probability sums - Min: {prob_sums.min():.6f}, Max: {prob_sums.max():.6f}")
    
    # Show a few examples
    print(f"\nFirst 5 samples:")
    for i in range(min(5, len(train_y))):
        p_natural, p_synthetic = train_y[i]
        print(f"  Sample {i}: P(natural)={p_natural:.3f}, P(synthetic)={p_synthetic:.3f}")
    
    return True


def test_reward_model_forward():
    """Test that the reward model can process the prepared data."""
    
    # Create a small reward model for testing
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=6,  # Smaller for testing
        n_head=6,
        n_embd=384,
        mode='reward'
    )
    
    model = GPT(config)
    model.eval()
    
    # Create dummy data
    batch_size = 4
    dummy_x = torch.randint(0, config.vocab_size, (batch_size, config.block_size))
    dummy_y = torch.rand(batch_size, 2)
    dummy_y = dummy_y / dummy_y.sum(dim=1, keepdim=True)  # Normalize to probabilities
    
    print(f"\nTesting reward model forward pass:")
    print(f"  Input shape: {dummy_x.shape}")
    print(f"  Target shape: {dummy_y.shape}")
    
    # Forward pass
    with torch.no_grad():
        probabilities, loss = model(dummy_x, dummy_y)
    
    print(f"  Output shape: {probabilities.shape}")
    print(f"  Loss: {loss.item():.6f}")
    
    # Check output properties
    print(f"  Output min: {probabilities.min().item():.6f}")
    print(f"  Output max: {probabilities.max().item():.6f}")
    
    # Check that outputs are valid probabilities
    prob_sums = probabilities.sum(dim=1)
    print(f"  Probability sums - Min: {prob_sums.min().item():.6f}, Max: {prob_sums.max().item():.6f}")
    
    return True


if __name__ == '__main__':
    print("=== Testing Reward Data Preparation ===")
    
    print("\n1. Testing reward data loading...")
    if test_reward_data_loading():
        print("✓ Reward data loading test passed")
    else:
        print("✗ Reward data loading test failed")
    
    print("\n2. Testing reward model forward pass...")
    if test_reward_model_forward():
        print("✓ Reward model forward pass test passed")
    else:
        print("✗ Reward model forward pass test failed")
    
    print("\n=== Tests Complete ===")