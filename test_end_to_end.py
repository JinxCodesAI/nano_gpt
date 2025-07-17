#!/usr/bin/env python3
"""
End-to-end tests for the configurable reward data preparation system.
Tests with real data and validates complete workflows.
"""

import os
import tempfile
import unittest
import shutil
import subprocess
import sys
import time
from unittest.mock import patch, MagicMock

from tokenization_manager import TokenizationManager
from data_loader import DataLoader
from reward_data_config import RewardDataConfig, ConfigurationValidator
from reward_dataset_loader import RewardDataset, print_dataset_info


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests with real data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Check data availability
        self.shakespeare_char_available = (
            os.path.exists('data/shakespeare_char/input.txt') and
            os.path.exists('data/shakespeare_char/meta.pkl') and
            os.path.exists('data/shakespeare_char/train.bin') and
            os.path.exists('data/shakespeare_char/val.bin')
        )
        
        self.shakespeare_bpe_available = (
            os.path.exists('data/shakespeare/input.txt') and
            os.path.exists('data/shakespeare/train.bin') and
            os.path.exists('data/shakespeare/val.bin')
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_character_tokenization_workflow(self):
        """Test complete workflow with character tokenization."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Test tokenization manager
        manager = TokenizationManager(data_path='data/shakespeare_char')
        self.assertEqual(manager.tokenization_type, 'char')
        self.assertEqual(manager.vocab_size, 65)
        
        # Test data loading from text
        loader = DataLoader(manager)
        train_tokens, val_tokens = loader.load_from_text(
            'data/shakespeare_char/input.txt', 
            train_split=0.9
        )
        
        # Verify data quality
        self.assertGreater(len(train_tokens), 900000)  # Should be ~1M tokens
        self.assertGreater(len(val_tokens), 100000)   # Should be ~100K tokens
        self.assertTrue(manager.validate_tokens(train_tokens[:1000]))
        self.assertTrue(manager.validate_tokens(val_tokens[:1000]))
        
        # Test data loading from binary
        train_tokens_bin, val_tokens_bin = loader.load_from_binary(
            'data/shakespeare_char/train.bin',
            'data/shakespeare_char/val.bin'
        )
        
        # Binary and text should have same total size (approximately)
        text_total = len(train_tokens) + len(val_tokens)
        bin_total = len(train_tokens_bin) + len(val_tokens_bin)
        self.assertAlmostEqual(text_total, bin_total, delta=1000)
    
    def test_bpe_tokenization_workflow(self):
        """Test complete workflow with BPE tokenization."""
        if not self.shakespeare_bpe_available:
            self.skipTest("Shakespeare BPE data not available")
        
        # Test tokenization manager
        manager = TokenizationManager()
        manager.load_bpe_tokenization()
        self.assertEqual(manager.tokenization_type, 'bpe')
        self.assertGreater(manager.vocab_size, 50000)
        
        # Test data loading from text
        loader = DataLoader(manager)
        train_tokens, val_tokens = loader.load_from_text(
            'data/shakespeare/input.txt',
            train_split=0.9
        )
        
        # Verify data quality
        self.assertGreater(len(train_tokens), 250000)  # Should be ~300K tokens
        self.assertGreater(len(val_tokens), 30000)     # Should be ~36K tokens
        self.assertTrue(manager.validate_tokens(train_tokens[:1000]))
        self.assertTrue(manager.validate_tokens(val_tokens[:1000]))
        
        # Test data loading from binary
        train_tokens_bin, val_tokens_bin = loader.load_from_binary(
            'data/shakespeare/train.bin',
            'data/shakespeare/val.bin'
        )
        
        # Binary should match expected sizes from prepare.py
        self.assertEqual(len(train_tokens_bin), 301966)
        self.assertEqual(len(val_tokens_bin), 36059)
    
    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation scenarios."""
        validator = ConfigurationValidator()
        
        # Test valid character tokenization config
        if self.shakespeare_char_available:
            config = RewardDataConfig(
                model_path='data/shakespeare_char/input.txt',  # Use existing file as dummy
                input_mode='text',
                data_path='data/shakespeare_char/input.txt',
                tokenization='char',
                meta_path='data/shakespeare_char/meta.pkl',
                output_dir=os.path.join(self.temp_dir, 'test_output')
            )
            
            self.assertTrue(validator.validate_config(config))
            self.assertEqual(len(config.get_errors()), 0)
        
        # Test valid binary mode config
        if self.shakespeare_bpe_available:
            config = RewardDataConfig(
                model_path='data/shakespeare/input.txt',  # Use existing file as dummy
                input_mode='binary',
                train_bin='data/shakespeare/train.bin',
                val_bin='data/shakespeare/val.bin',
                output_dir=os.path.join(self.temp_dir, 'test_output')
            )
            
            self.assertTrue(validator.validate_config(config))
            self.assertEqual(len(config.get_errors()), 0)
        
        # Test invalid config (missing files)
        config = RewardDataConfig(
            model_path='nonexistent_model.pt',
            input_mode='text',
            data_path='nonexistent_data.txt',
            output_dir=os.path.join(self.temp_dir, 'test_output')
        )
        
        self.assertFalse(validator.validate_config(config))
        self.assertGreater(len(config.get_errors()), 0)
    
    def test_command_line_interface_validation(self):
        """Test command-line interface validation."""
        # Test help output
        result = subprocess.run([
            sys.executable, 'prepare_reward_data.py', '--help'
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn('input_mode', result.stdout)
        self.assertIn('tokenization', result.stdout)
        self.assertIn('meta_path', result.stdout)
        
        # Test validation error for missing model
        result = subprocess.run([
            sys.executable, 'prepare_reward_data.py',
            '--model_path', 'nonexistent.pt',
            '--data_path', 'data/shakespeare_char/input.txt' if self.shakespeare_char_available else 'dummy.txt'
        ], capture_output=True, text=True)

        # Should show validation errors
        self.assertIn('Validation Errors', result.stdout)
        self.assertIn('ERROR:', result.stdout)
        self.assertIn('nonexistent.pt', result.stdout)
    
    def test_tokenization_auto_detection(self):
        """Test tokenization auto-detection with real data."""
        if self.shakespeare_char_available:
            # Test character detection
            manager = TokenizationManager()
            detected = manager.detect_tokenization_method('data/shakespeare_char')
            self.assertEqual(detected, 'char')
            
            # Test with file path
            detected = manager.detect_tokenization_method('data/shakespeare_char/input.txt')
            self.assertEqual(detected, 'char')
        
        if self.shakespeare_bpe_available:
            # Test BPE detection (default)
            manager = TokenizationManager()
            detected = manager.detect_tokenization_method('data/shakespeare')
            self.assertEqual(detected, 'bpe')
    
    def test_performance_comparison(self):
        """Test and compare performance between text and binary modes."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        manager = TokenizationManager(data_path='data/shakespeare_char')
        loader = DataLoader(manager)
        
        # Time text loading
        start_time = time.time()
        train_text, val_text = loader.load_from_text(
            'data/shakespeare_char/input.txt',
            train_split=0.9
        )
        text_time = time.time() - start_time
        
        # Time binary loading
        start_time = time.time()
        train_bin, val_bin = loader.load_from_binary(
            'data/shakespeare_char/train.bin',
            'data/shakespeare_char/val.bin'
        )
        binary_time = time.time() - start_time
        
        print(f"\nPerformance comparison:")
        print(f"  Text mode: {text_time:.3f}s")
        print(f"  Binary mode: {binary_time:.3f}s")

        # Note: Binary might not always be faster for small files due to overhead
        # but should be faster for large files. For testing, just verify both work.
        if text_time > binary_time:
            print(f"  Speedup: {text_time/binary_time:.1f}x")
        else:
            print(f"  Text mode was faster (small file overhead)")

        # Both modes should work and produce consistent results
        self.assertGreater(text_time, 0)
        self.assertGreater(binary_time, 0)
        
        # Data should be consistent
        self.assertEqual(len(train_text) + len(val_text), len(train_bin) + len(val_bin))
    
    def test_dataset_compatibility_validation(self):
        """Test dataset compatibility validation with real data."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Create a test dataset with tokenization metadata
        output_dir = os.path.join(self.temp_dir, 'test_dataset')
        os.makedirs(output_dir)
        
        # Create sample data files
        import numpy as np
        x_data = np.random.randint(0, 65, size=(10, 8), dtype=np.uint16)
        y_data = np.random.random((10, 2)).astype(np.float32)
        y_data = y_data / y_data.sum(axis=1, keepdims=True)
        
        x_data.tofile(os.path.join(output_dir, 'train_x.bin'))
        y_data.tofile(os.path.join(output_dir, 'train_y.bin'))
        
        # Create metadata with tokenization info
        with open(os.path.join(output_dir, 'train_metadata.txt'), 'w') as f:
            f.write("num_samples: 10\n")
            f.write("block_size: 8\n")
            f.write("x_shape: (10, 8)\n")
            f.write("y_shape: (10, 2)\n")
            f.write("tokenization_method: char\n")
            f.write("vocab_size: 65\n")
            f.write("meta_path: data/shakespeare_char/meta.pkl\n")
        
        # Test loading with compatible tokenization
        from reward_data_config import TokenizationInfo
        expected_tokenization = TokenizationInfo(
            method='char',
            vocab_size=65,
            meta_path='data/shakespeare_char/meta.pkl'
        )
        
        dataset = RewardDataset(output_dir, 'train', expected_tokenization)
        self.assertEqual(len(dataset), 10)
        
        # Test loading with incompatible tokenization
        incompatible_tokenization = TokenizationInfo(
            method='bpe',
            vocab_size=50257,
            meta_path=None
        )
        
        with self.assertRaises(ValueError):
            RewardDataset(output_dir, 'train', incompatible_tokenization)
    
    def test_backward_compatibility(self):
        """Test that existing workflows still work."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Test legacy load_and_split_data function
        from prepare_reward_data import load_and_split_data
        
        train_tokens, val_tokens, tokenizer = load_and_split_data(
            'data/shakespeare_char/input.txt',
            train_split=0.9
        )
        
        # Should work with BPE tokenization (legacy behavior)
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        
        # Should be BPE tokens (different from char tokens)
        self.assertNotEqual(len(train_tokens), 1003854)  # Not char token count


if __name__ == '__main__':
    # Set up logging for better test output
    import logging
    logging.basicConfig(level=logging.INFO)
    
    unittest.main(verbosity=2)
