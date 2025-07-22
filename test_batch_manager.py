#!/usr/bin/env python3
"""
Comprehensive tests for the new BatchManager V2 with curriculum learning.
Tests both synchronous and asynchronous functionality.
"""

import os
import sys
import time
import tempfile
import shutil
import numpy as np
import torch
import threading
from collections import deque
import unittest

# Add the current directory to the path so we can import from train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the BatchManager class from train.py
# We need to import it in a way that doesn't execute the training script
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", "train.py")
train_module = importlib.util.module_from_spec(spec)

# Mock the global variables that train.py expects
train_module.batch_manager = None
train_module.master_process = True

# Execute only the BatchManager class definition
with open('train.py', 'r') as f:
    content = f.read()
    
# Extract just the BatchManager class definition
start_marker = "class BatchManager:"
end_marker = "# Simple validation batch function"
start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    raise RuntimeError("Could not find BatchManager class in train.py")

batch_manager_code = content[start_idx:end_idx]

# Add necessary imports for the BatchManager class
imports_code = """
import random
import time
import os
import torch
import numpy as np
import threading
from collections import deque
"""

# Execute the imports and BatchManager class definition
exec(imports_code + batch_manager_code)


class TestBatchManager(unittest.TestCase):
    """Test suite for BatchManager V2"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_size = 1000
        self.batch_size = 4
        self.block_size = 8
        self.device = torch.device('cpu')
        self.device_type = 'cpu'
        
        # Create test data shards
        self.shard_filenames = ['test_shard1.bin', 'test_shard2.bin']
        self.create_test_shards()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Give some time for any background threads to finish
        time.sleep(0.5)
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # On Windows, sometimes files are still locked by background threads
            time.sleep(1)
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                print(f"Warning: Could not clean up temp directory {self.temp_dir}")
        
    def create_test_shards(self):
        """Create test data shards with known token distributions"""
        # Create shard 1 with uniform-ish distribution
        shard1_data = np.random.randint(0, self.vocab_size, size=10000, dtype=np.uint16)
        shard1_path = os.path.join(self.temp_dir, self.shard_filenames[0])
        shard1_data.tofile(shard1_path)
        
        # Create shard 2 with skewed distribution (more low-value tokens)
        shard2_data = np.concatenate([
            np.random.randint(0, 100, size=7000, dtype=np.uint16),  # Many low tokens
            np.random.randint(100, self.vocab_size, size=3000, dtype=np.uint16)  # Fewer high tokens
        ])
        np.random.shuffle(shard2_data)
        shard2_path = os.path.join(self.temp_dir, self.shard_filenames[1])
        shard2_data.tofile(shard2_path)
        
    def test_batch_manager_initialization(self):
        """Test BatchManager initialization"""
        batch_manager = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,  # Small for testing
            buffer_size=10  # Small buffer for testing
        )
        
        # Check initialization
        self.assertEqual(batch_manager.vocab_size, self.vocab_size)
        self.assertEqual(batch_manager.batch_size, self.batch_size)
        self.assertEqual(batch_manager.block_size, self.block_size)
        self.assertEqual(batch_manager.buffer_size, 10)
        
        # Check that distributions are initialized
        self.assertIsNotNone(batch_manager.corpus_distribution)
        self.assertIsNotNone(batch_manager.uniform_distribution)
        self.assertEqual(batch_manager.corpus_distribution.shape[0], self.vocab_size)
        self.assertEqual(batch_manager.uniform_distribution.shape[0], self.vocab_size)
        
        # Check that worker thread is running
        self.assertTrue(batch_manager.worker_thread.is_alive())
        
        # Clean shutdown
        batch_manager.shutdown()
        
    def test_corpus_distribution_caching(self):
        """Test that corpus distribution is cached correctly"""
        # First initialization should create cache
        batch_manager1 = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,
            buffer_size=10
        )
        
        # Check cache file exists
        cache_path = os.path.join(self.temp_dir, 'corpus_dist_approx_1000.pt')
        self.assertTrue(os.path.exists(cache_path))
        
        batch_manager1.shutdown()
        
        # Second initialization should load from cache
        batch_manager2 = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,
            buffer_size=10
        )
        
        # Distributions should be the same
        torch.testing.assert_close(
            batch_manager1.corpus_distribution,
            batch_manager2.corpus_distribution,
            rtol=1e-5, atol=1e-6
        )
        
        batch_manager2.shutdown()
        
    def test_batch_retrieval(self):
        """Test basic batch retrieval functionality"""
        batch_manager = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,
            buffer_size=20
        )
        
        # Wait for buffer to fill
        time.sleep(2)
        
        # Get a batch
        X, Y = batch_manager.get_next_batch()
        
        # Check batch shape
        self.assertEqual(X.shape, (self.batch_size, self.block_size))
        self.assertEqual(Y.shape, (self.batch_size, self.block_size))
        
        # Check that Y is X shifted by 1
        self.assertTrue(torch.all(Y[:, :-1] == X[:, 1:]))
        
        # Check that tokens are in valid range
        self.assertTrue(torch.all(X >= 0))
        self.assertTrue(torch.all(X < self.vocab_size))
        self.assertTrue(torch.all(Y >= 0))
        self.assertTrue(torch.all(Y < self.vocab_size))
        
        batch_manager.shutdown()
        
    def test_curriculum_control(self):
        """Test curriculum learning alpha control"""
        batch_manager = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,
            buffer_size=20
        )
        
        # Wait for buffer to fill
        time.sleep(2)
        
        # Test alpha = 1.0 (uniform)
        batch_manager.update_target_distribution(1.0)
        time.sleep(1)  # Wait for re-scoring
        target_uniform = batch_manager.target_distribution
        expected_uniform = batch_manager.uniform_distribution
        torch.testing.assert_close(target_uniform, expected_uniform, rtol=1e-5, atol=1e-6)

        # Test alpha = 0.0 (corpus)
        batch_manager.update_target_distribution(0.0)
        time.sleep(1)  # Wait for re-scoring
        target_corpus = batch_manager.target_distribution
        expected_corpus = batch_manager.corpus_distribution
        torch.testing.assert_close(target_corpus, expected_corpus, rtol=1e-5, atol=1e-6)

        # Test alpha = 0.5 (blend)
        batch_manager.update_target_distribution(0.5)
        time.sleep(1)  # Wait for re-scoring
        target_blend = batch_manager.target_distribution
        expected_blend = 0.5 * batch_manager.corpus_distribution + 0.5 * batch_manager.uniform_distribution
        torch.testing.assert_close(target_blend, expected_blend, rtol=1e-5, atol=1e-6)
        
        batch_manager.shutdown()
        
    def test_served_token_tracking(self):
        """Test that served tokens are tracked correctly"""
        batch_manager = BatchManager(
            data_dir=self.temp_dir,
            shard_filenames=self.shard_filenames,
            vocab_size=self.vocab_size,
            batch_size=self.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            starting_estimation_token_count=1000,
            buffer_size=20
        )
        
        # Wait for buffer to fill
        time.sleep(2)
        
        # Initial state
        initial_total = batch_manager.total_tokens_served
        initial_counts = batch_manager.served_token_counts.clone()
        
        # Get a batch
        X, Y = batch_manager.get_next_batch()
        
        # Check that counts were updated
        self.assertGreater(batch_manager.total_tokens_served, initial_total)
        self.assertEqual(
            batch_manager.total_tokens_served - initial_total,
            X.numel()
        )
        
        # Check that token counts were updated
        unique_tokens, counts = torch.unique(X, return_counts=True)
        for token, count in zip(unique_tokens, counts):
            expected_count = initial_counts[token] + count
            self.assertEqual(batch_manager.served_token_counts[token], expected_count)
        
        batch_manager.shutdown()


if __name__ == '__main__':
    unittest.main()
