#!/usr/bin/env python3
"""
Test suite for enhanced data augmentation feature
"""

import os
import sys
import tempfile
import numpy as np
import torch
import threading
import time
from unittest.mock import MagicMock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the classes and functions from train.py
from train import (
    EnhancedSampleBuffer, 
    EnhancedSampleGenerator,
    determine_batch_composition,
    generate_enhanced_samples_batch,
    sample_random_fragments
)

def test_enhanced_sample_buffer():
    """Test EnhancedSampleBuffer functionality"""
    print("Testing EnhancedSampleBuffer...")
    
    # Test basic operations
    buffer = EnhancedSampleBuffer(max_size=5)
    
    # Test empty buffer
    assert buffer.size() == 0
    assert not buffer.is_full()
    assert buffer.get_samples(3) == []
    
    # Test adding samples
    sample1 = (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4]))
    sample2 = (torch.tensor([4, 5, 6]), torch.tensor([5, 6, 7]))
    sample3 = (torch.tensor([7, 8, 9]), torch.tensor([8, 9, 10]))
    
    buffer.add_samples([sample1, sample2, sample3])
    assert buffer.size() == 3
    
    # Test getting samples
    samples = buffer.get_samples(2)
    assert len(samples) == 2
    assert buffer.size() == 1  # Should have 1 remaining
    
    # Test buffer overflow (should replace oldest)
    for i in range(6):
        new_sample = (torch.tensor([i, i+1, i+2]), torch.tensor([i+1, i+2, i+3]))
        buffer.add_samples([new_sample])
    
    assert buffer.size() == 5  # Should be at max capacity
    assert buffer.is_full()
    
    # Test clearing
    buffer.clear()
    assert buffer.size() == 0
    
    print("✓ EnhancedSampleBuffer tests passed")

def test_determine_batch_composition():
    """Test batch composition determination"""
    print("Testing determine_batch_composition...")
    
    # Test probability 0.0 (all natural)
    mask = determine_batch_composition(10, 0.0)
    assert mask.sum() == 0
    assert mask.dtype == torch.bool
    
    # Test probability 1.0 (all enhanced)
    mask = determine_batch_composition(10, 1.0)
    assert mask.sum() == 10
    
    # Test intermediate probability (should be stochastic)
    torch.manual_seed(42)
    mask = determine_batch_composition(100, 0.5)
    enhanced_count = mask.sum().item()
    assert 30 <= enhanced_count <= 70  # Should be roughly 50%
    
    print("✓ determine_batch_composition tests passed")

def test_sample_random_fragments():
    """Test random fragment sampling"""
    print("Testing sample_random_fragments...")
    
    block_size = 10
    
    # Test with sequences longer than block_size
    seq1 = torch.arange(20)  # Length 20
    seq2 = torch.arange(15)  # Length 15
    sequences = [seq1, seq2]
    
    fragments = sample_random_fragments(sequences, block_size)
    assert len(fragments) == 2
    assert all(len(frag) == block_size for frag in fragments)
    
    # Test with sequence shorter than block_size
    short_seq = torch.arange(5)  # Length 5 < block_size
    fragments = sample_random_fragments([short_seq], block_size)
    assert len(fragments) == 0  # Should be excluded
    
    # Test with sequence exactly block_size
    exact_seq = torch.arange(block_size)
    fragments = sample_random_fragments([exact_seq], block_size)
    assert len(fragments) == 1
    assert torch.equal(fragments[0], exact_seq)
    
    print("✓ sample_random_fragments tests passed")

def test_generate_enhanced_samples_batch():
    """Test enhanced sample generation (with mock model)"""
    print("Testing generate_enhanced_samples_batch...")
    
    # Create mock inference model
    mock_model = MagicMock()
    
    # Mock the generate method to return a simple continuation
    def mock_generate(prefix, max_tokens, temperature=1.0, top_k=None):
        # Return a tensor that's longer than block_size
        batch_size = prefix.shape[0]
        seq_length = prefix.shape[1] + max_tokens
        return torch.randint(0, 1000, (batch_size, seq_length))
    
    mock_model.generate = mock_generate
    
    # Create mock training data
    data = np.random.randint(0, 1000, size=10000, dtype=np.uint16)
    
    # Test parameters
    batch_size = 4
    min_prefix_length = 10
    max_prefix_length = 20
    block_size = 32
    temperature = 0.8
    top_k = 200
    device = 'cpu'
    ctx = torch.no_grad()
    
    # Generate samples
    samples = generate_enhanced_samples_batch(
        mock_model, data, batch_size, min_prefix_length, max_prefix_length,
        block_size, temperature, top_k, device, ctx
    )
    
    # Verify results
    assert len(samples) == batch_size
    for x, y in samples:
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)
        assert x.device.type == device
        assert y.device.type == device
    
    print("✓ generate_enhanced_samples_batch tests passed")

def test_enhanced_sample_generator():
    """Test EnhancedSampleGenerator functionality"""
    print("Testing EnhancedSampleGenerator...")
    
    # Create components
    buffer = EnhancedSampleBuffer(max_size=10)
    
    # Mock inference model
    mock_model = MagicMock()
    def mock_generate(prefix, max_tokens, temperature=1.0, top_k=None):
        batch_size = prefix.shape[0]
        seq_length = prefix.shape[1] + max_tokens
        return torch.randint(0, 1000, (batch_size, seq_length))
    mock_model.generate = mock_generate
    
    # Mock training data
    data = np.random.randint(0, 1000, size=1000, dtype=np.uint16)
    
    # Configuration
    config = {
        'enhanced_generation_batch_size': 2,
        'min_prefix_length': 5,
        'max_prefix_length': 10,
        'block_size': 16,
        'enhanced_generation_temperature': 0.8,
        'enhanced_generation_top_k': 200,
    }
    
    # Create generator
    generator = EnhancedSampleGenerator(
        buffer, mock_model, data, config, 'cpu', torch.no_grad()
    )
    
    # Test initial state
    assert not generator.is_running()
    
    # Start generator
    generator.start()
    assert generator.is_running()
    
    # Wait for some samples to be generated
    time.sleep(0.5)
    
    # Check if samples were generated
    assert buffer.size() > 0
    
    # Test model update
    new_mock_model = MagicMock()
    new_mock_model.generate = mock_generate
    old_size = buffer.size()
    generator.update_model(new_mock_model)
    assert buffer.size() == 0  # Buffer should be cleared
    
    # Stop generator
    generator.stop()
    assert not generator.is_running()
    
    print("✓ EnhancedSampleGenerator tests passed")

def test_integration():
    """Test integration with mock get_batch function"""
    print("Testing integration...")
    
    # This test would require importing and patching the actual get_batch function
    # For now, we'll just verify that our components work together
    
    buffer = EnhancedSampleBuffer(max_size=20)
    
    # Add some sample data
    for i in range(10):
        x = torch.randint(0, 1000, (32,))
        y = torch.randint(0, 1000, (32,))
        buffer.add_samples([(x, y)])
    
    # Test batch composition
    batch_size = 8
    probability = 0.5
    mask = determine_batch_composition(batch_size, probability)
    n_enhanced = mask.sum().item()
    
    # Get enhanced samples
    enhanced_samples = buffer.get_samples(n_enhanced)
    
    # Verify we can build a mixed batch
    assert len(enhanced_samples) <= n_enhanced
    
    print("✓ Integration tests passed")

def run_all_tests():
    """Run all tests"""
    print("Running enhanced data augmentation tests...\n")
    
    try:
        test_enhanced_sample_buffer()
        test_determine_batch_composition()
        test_sample_random_fragments()
        test_generate_enhanced_samples_batch()
        test_enhanced_sample_generator()
        test_integration()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)