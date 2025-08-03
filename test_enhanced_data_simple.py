#!/usr/bin/env python3
"""
Simple test for enhanced data augmentation components
"""

import torch
import numpy as np
import threading
import time
import random
from collections import deque

# Copy the classes and functions we need to test (extracted from train.py)

class EnhancedSampleBuffer:
    """Thread-safe rotating buffer for pre-generated enhanced samples"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_consumed = 0
        self.total_generated = 0
    
    def get_samples(self, n):
        """Get n random samples from buffer, returns list of (x, y) tensor pairs"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            # Sample with replacement to reuse samples if generation is slow
            available = len(self.buffer)
            samples = []
            
            for _ in range(n):
                idx = random.randint(0, available - 1)
                samples.append(self.buffer[idx])
            
            self.total_consumed += n
            return samples
    
    def add_samples(self, samples):
        """Add samples to buffer, oldest samples are automatically evicted if full"""
        with self.lock:
            for sample in samples:
                self.buffer.append(sample)
            self.total_generated += len(samples)
    
    def clear(self):
        """Clear all samples from buffer"""
        with self.lock:
            self.buffer.clear()
            # Reset counters when buffer is cleared (e.g., model update)
            self.total_consumed = 0
            self.total_generated = 0
    
    def is_full(self):
        """Check if buffer is at maximum capacity"""
        with self.lock:
            return len(self.buffer) >= self.max_size
    
    def size(self):
        """Get current number of samples in buffer"""
        with self.lock:
            return len(self.buffer)
    
    def get_consumption_stats(self):
        """Get generation and consumption statistics"""
        with self.lock:
            return self.total_generated, self.total_consumed

def determine_batch_composition(batch_size, probability):
    """Returns boolean mask indicating which batch elements should be enhanced"""
    if probability <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool)
    elif probability >= 1.0:
        return torch.ones(batch_size, dtype=torch.bool)
    else:
        return torch.rand(batch_size) < probability

def sample_random_fragments(sequences, block_size):
    """Extract random fragments of exact block_size from generated sequences"""
    fragments = []
    
    for seq in sequences:
        if len(seq) < block_size:
            continue
        
        # Random starting position
        max_start = len(seq) - block_size
        start_idx = random.randint(0, max_start)
        fragment = seq[start_idx:start_idx + block_size]
        fragments.append(fragment)
    
    return fragments

# Test functions

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
    
    # Test getting samples (with replacement - samples should still be in buffer)
    initial_size = buffer.size()
    samples = buffer.get_samples(2)
    final_size = buffer.size()
    assert len(samples) == 2
    assert final_size == initial_size  # Should still have all samples (reused)
    
    # Test buffer overflow (should replace oldest)
    for i in range(6):
        new_sample = (torch.tensor([i, i+1, i+2]), torch.tensor([i+1, i+2, i+3]))
        buffer.add_samples([new_sample])
    
    assert buffer.size() == 5  # Should be at max capacity
    assert buffer.is_full()
    
    # Test clearing
    buffer.clear()
    assert buffer.size() == 0
    
    # Test consumption tracking
    sample4 = (torch.tensor([10, 11, 12]), torch.tensor([11, 12, 13]))
    sample5 = (torch.tensor([13, 14, 15]), torch.tensor([14, 15, 16]))
    buffer.add_samples([sample4, sample5])
    
    generated, consumed = buffer.get_consumption_stats()
    assert generated == 2  # Added 2 samples
    assert consumed == 0   # Haven't consumed any yet
    
    samples = buffer.get_samples(3)  # Request 3, but only 2 available (with reuse)
    assert len(samples) == 3  # Should get 3 (with reuse)
    
    generated, consumed = buffer.get_consumption_stats()
    assert generated == 2  # Still 2 generated
    assert consumed == 3   # Consumed 3 (with reuse)
    
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

def test_thread_safety():
    """Test thread safety of buffer"""
    print("Testing thread safety...")
    
    buffer = EnhancedSampleBuffer(max_size=100)
    results = {'retrieved': 0}
    
    def add_samples_worker():
        for i in range(50):
            sample = (torch.tensor([i]), torch.tensor([i+1]))
            buffer.add_samples([sample])
            time.sleep(0.001)
    
    def get_samples_worker():
        for _ in range(20):
            samples = buffer.get_samples(5)
            results['retrieved'] += len(samples)
            time.sleep(0.002)
    
    # Start threads
    threads = []
    for _ in range(2):  # Reduce number of add threads
        t = threading.Thread(target=add_samples_worker)
        threads.append(t)
        t.start()
    
    time.sleep(0.05)  # Let some samples accumulate
    
    # Start retrieval thread
    get_thread = threading.Thread(target=get_samples_worker)
    get_thread.start()
    
    # Wait for all threads
    for t in threads:
        t.join()
    get_thread.join()
    
    # Should have retrieved samples successfully (samples are reused)
    assert results['retrieved'] > 0
    print(f"✓ Thread safety tests passed (retrieved {results['retrieved']} samples)")

def run_all_tests():
    """Run all tests"""
    print("Running enhanced data augmentation tests...\n")
    
    try:
        test_enhanced_sample_buffer()
        test_determine_batch_composition()
        test_sample_random_fragments()
        test_thread_safety()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)