#!/usr/bin/env python3
"""
Unit tests for enhanced reward_dataset_loader with tokenization compatibility.
"""

import os
import tempfile
import unittest
import numpy as np
import shutil
from unittest.mock import patch

from reward_dataset_loader import RewardDataset, create_reward_dataloaders, print_dataset_info
from reward_data_config import TokenizationInfo


class TestRewardDatasetLoader(unittest.TestCase):
    """Test cases for enhanced RewardDataset with tokenization compatibility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample dataset files
        self.num_samples = 10
        self.block_size = 8
        
        # Create sample data
        np.random.seed(42)
        x_data = np.random.randint(0, 100, size=(self.num_samples, self.block_size), dtype=np.uint16)
        y_data = np.random.random((self.num_samples, 2)).astype(np.float32)
        
        # Normalize y_data to sum to 1 (probability distributions)
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        
        # Save binary files
        x_data.tofile(os.path.join(self.temp_dir, 'train_x.bin'))
        y_data.tofile(os.path.join(self.temp_dir, 'train_y.bin'))
        x_data.tofile(os.path.join(self.temp_dir, 'val_x.bin'))
        y_data.tofile(os.path.join(self.temp_dir, 'val_y.bin'))
        
        # Create metadata files with tokenization info
        self.create_metadata_file('train', with_tokenization=True)
        self.create_metadata_file('val', with_tokenization=True)
        
        # Store expected tokenization info
        self.expected_tokenization = TokenizationInfo(
            method='char',
            vocab_size=65,
            meta_path='data/shakespeare_char/meta.pkl'
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_metadata_file(self, split, with_tokenization=False):
        """Create a metadata file for testing."""
        metadata_path = os.path.join(self.temp_dir, f'{split}_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"num_samples: {self.num_samples}\n")
            f.write(f"block_size: {self.block_size}\n")
            f.write(f"x_shape: ({self.num_samples}, {self.block_size})\n")
            f.write(f"y_shape: ({self.num_samples}, 2)\n")
            f.write("x_dtype: uint16\n")
            f.write("y_dtype: float32\n")
            
            if with_tokenization:
                f.write("tokenization_method: char\n")
                f.write("vocab_size: 65\n")
                f.write("meta_path: data/shakespeare_char/meta.pkl\n")
    
    def test_load_dataset_with_tokenization_info(self):
        """Test loading dataset with tokenization information."""
        dataset = RewardDataset(self.temp_dir, 'train')
        
        # Check basic properties
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.block_size, self.block_size)
        
        # Check tokenization info
        tokenization_info = dataset.get_tokenization_info()
        self.assertIsNotNone(tokenization_info)
        self.assertEqual(tokenization_info.method, 'char')
        self.assertEqual(tokenization_info.vocab_size, 65)
        self.assertEqual(tokenization_info.meta_path, 'data/shakespeare_char/meta.pkl')
    
    def test_load_dataset_without_tokenization_info(self):
        """Test loading dataset without tokenization information."""
        # Create metadata without tokenization info
        self.create_metadata_file('train', with_tokenization=False)
        
        dataset = RewardDataset(self.temp_dir, 'train')
        
        # Check that tokenization info is None
        tokenization_info = dataset.get_tokenization_info()
        self.assertIsNone(tokenization_info)
    
    def test_tokenization_compatibility_validation_success(self):
        """Test successful tokenization compatibility validation."""
        # Should not raise any exception
        dataset = RewardDataset(self.temp_dir, 'train', self.expected_tokenization)
        
        tokenization_info = dataset.get_tokenization_info()
        self.assertEqual(tokenization_info.method, self.expected_tokenization.method)
        self.assertEqual(tokenization_info.vocab_size, self.expected_tokenization.vocab_size)
    
    def test_tokenization_compatibility_validation_method_mismatch(self):
        """Test tokenization compatibility validation with method mismatch."""
        incompatible_tokenization = TokenizationInfo(
            method='bpe',  # Different method
            vocab_size=65,
            meta_path=None
        )
        
        with self.assertRaises(ValueError) as context:
            RewardDataset(self.temp_dir, 'train', incompatible_tokenization)
        
        self.assertIn("Tokenization method mismatch", str(context.exception))
        self.assertIn("char", str(context.exception))
        self.assertIn("bpe", str(context.exception))
    
    def test_tokenization_compatibility_validation_vocab_size_mismatch(self):
        """Test tokenization compatibility validation with vocab size mismatch."""
        incompatible_tokenization = TokenizationInfo(
            method='char',
            vocab_size=100,  # Different vocab size
            meta_path='data/shakespeare_char/meta.pkl'
        )
        
        with self.assertRaises(ValueError) as context:
            RewardDataset(self.temp_dir, 'train', incompatible_tokenization)
        
        self.assertIn("Vocabulary size mismatch", str(context.exception))
        self.assertIn("65", str(context.exception))
        self.assertIn("100", str(context.exception))
    
    def test_get_stats_with_tokenization_info(self):
        """Test get_stats includes tokenization information."""
        dataset = RewardDataset(self.temp_dir, 'train')
        stats = dataset.get_stats()
        
        # Check basic stats
        self.assertEqual(stats['num_samples'], self.num_samples)
        self.assertEqual(stats['block_size'], self.block_size)
        
        # Check tokenization info in stats
        self.assertEqual(stats['tokenization_method'], 'char')
        self.assertEqual(stats['vocab_size'], 65)
        self.assertEqual(stats['meta_path'], 'data/shakespeare_char/meta.pkl')
    
    def test_create_dataloaders_with_tokenization_validation(self):
        """Test creating dataloaders with tokenization validation."""
        train_loader, val_loader = create_reward_dataloaders(
            self.temp_dir, 
            batch_size=4, 
            expected_tokenization_info=self.expected_tokenization
        )
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        
        # Test loading a batch
        for batch_x, batch_y in train_loader:
            self.assertEqual(batch_x.shape[1], self.block_size)
            self.assertEqual(batch_y.shape[1], 2)
            break
    
    def test_create_dataloaders_tokenization_mismatch_between_splits(self):
        """Test error when train and val have different tokenization."""
        # Create val metadata with different tokenization
        val_metadata_path = os.path.join(self.temp_dir, 'val_metadata.txt')
        with open(val_metadata_path, 'w') as f:
            f.write(f"num_samples: {self.num_samples}\n")
            f.write(f"block_size: {self.block_size}\n")
            f.write(f"x_shape: ({self.num_samples}, {self.block_size})\n")
            f.write(f"y_shape: ({self.num_samples}, 2)\n")
            f.write("x_dtype: uint16\n")
            f.write("y_dtype: float32\n")
            f.write("tokenization_method: bpe\n")  # Different method
            f.write("vocab_size: 50257\n")  # Different vocab size
            f.write("meta_path: None\n")
        
        with self.assertRaises(ValueError) as context:
            create_reward_dataloaders(self.temp_dir, batch_size=4)
        
        self.assertIn("Tokenization mismatch between train and val", str(context.exception))
    
    def test_print_dataset_info_with_tokenization(self):
        """Test print_dataset_info shows tokenization information."""
        # Capture print output
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_dataset_info(self.temp_dir)
            output = captured_output.getvalue()
            
            # Check that tokenization info is displayed
            self.assertIn("Tokenization: char", output)
            self.assertIn("vocab_size=65", output)
            self.assertIn("Meta path:", output)
            self.assertIn("Tokenization consistency", output)
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_dataset_info_tokenization_mismatch(self):
        """Test print_dataset_info detects tokenization mismatch."""
        # Create val metadata with different tokenization
        val_metadata_path = os.path.join(self.temp_dir, 'val_metadata.txt')
        with open(val_metadata_path, 'w') as f:
            f.write(f"num_samples: {self.num_samples}\n")
            f.write(f"block_size: {self.block_size}\n")
            f.write(f"x_shape: ({self.num_samples}, {self.block_size})\n")
            f.write(f"y_shape: ({self.num_samples}, 2)\n")
            f.write("x_dtype: uint16\n")
            f.write("y_dtype: float32\n")
            f.write("tokenization_method: bpe\n")
            f.write("vocab_size: 50257\n")
            f.write("meta_path: None\n")
        
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_dataset_info(self.temp_dir)
            output = captured_output.getvalue()
            
            # Check that mismatch is detected
            self.assertIn("Tokenization mismatch detected", output)
            self.assertIn("Train: char", output)
            self.assertIn("Val: bpe", output)
            
        finally:
            sys.stdout = sys.__stdout__


if __name__ == '__main__':
    unittest.main()
