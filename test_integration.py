#!/usr/bin/env python3
"""
Integration tests for the configurable reward data preparation system.
"""

import os
import tempfile
import unittest
import subprocess
import sys
import shutil
from unittest.mock import patch

from tokenization_manager import TokenizationManager
from data_loader import DataLoader
from reward_data_config import RewardDataConfig, ConfigurationValidator


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.temp_dir, 'test_output')
        
        # Check if test data exists
        self.shakespeare_char_dir = 'data/shakespeare_char'
        self.shakespeare_dir = 'data/shakespeare'
        
        self.shakespeare_char_available = (
            os.path.exists(os.path.join(self.shakespeare_char_dir, 'input.txt')) and
            os.path.exists(os.path.join(self.shakespeare_char_dir, 'meta.pkl'))
        )
        
        self.shakespeare_bpe_available = (
            os.path.exists(os.path.join(self.shakespeare_dir, 'input.txt')) and
            os.path.exists(os.path.join(self.shakespeare_dir, 'train.bin')) and
            os.path.exists(os.path.join(self.shakespeare_dir, 'val.bin'))
        )
        
        self.shakespeare_char_bins_available = (
            os.path.exists(os.path.join(self.shakespeare_char_dir, 'train.bin')) and
            os.path.exists(os.path.join(self.shakespeare_char_dir, 'val.bin'))
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_tokenization_manager_char_integration(self):
        """Test TokenizationManager with real shakespeare_char data."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Test auto-detection
        manager = TokenizationManager(data_path=self.shakespeare_char_dir)
        self.assertEqual(manager.tokenization_type, 'char')
        self.assertEqual(manager.vocab_size, 65)
        
        # Test encoding/decoding with real data
        test_text = "To be or not to be, that is the question."
        tokens = manager.encode(test_text)
        decoded_text = manager.decode(tokens)
        self.assertEqual(decoded_text, test_text)
        
        # Validate tokens are in range
        self.assertTrue(manager.validate_tokens(tokens))
    
    def test_tokenization_manager_bpe_integration(self):
        """Test TokenizationManager with BPE tokenization."""
        manager = TokenizationManager()
        manager.load_bpe_tokenization()
        
        self.assertEqual(manager.tokenization_type, 'bpe')
        self.assertGreater(manager.vocab_size, 50000)  # GPT-2 vocab size
        
        # Test encoding/decoding
        test_text = "To be or not to be, that is the question."
        tokens = manager.encode(test_text)
        decoded_text = manager.decode(tokens)
        self.assertEqual(decoded_text, test_text)
        
        # Validate tokens are in range
        self.assertTrue(manager.validate_tokens(tokens))
    
    def test_data_loader_text_mode_char(self):
        """Test DataLoader with text mode and character tokenization."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Initialize tokenization manager
        manager = TokenizationManager(data_path=self.shakespeare_char_dir)
        loader = DataLoader(manager)
        
        # Load data
        input_path = os.path.join(self.shakespeare_char_dir, 'input.txt')
        train_tokens, val_tokens = loader.load_from_text(input_path, train_split=0.9)
        
        # Verify data was loaded
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        
        # Verify split ratio
        total_tokens = len(train_tokens) + len(val_tokens)
        actual_split = len(train_tokens) / total_tokens
        self.assertAlmostEqual(actual_split, 0.9, delta=0.01)
        
        # Verify tokens are valid
        self.assertTrue(manager.validate_tokens(train_tokens[:100]))  # Check first 100
        self.assertTrue(manager.validate_tokens(val_tokens[:100]))
    
    def test_data_loader_binary_mode_char(self):
        """Test DataLoader with binary mode and character tokenization."""
        if not self.shakespeare_char_bins_available:
            self.skipTest("Shakespeare character binary files not available")
        
        # Initialize tokenization manager
        manager = TokenizationManager(data_path=self.shakespeare_char_dir)
        loader = DataLoader(manager)
        
        # Load binary data
        train_bin = os.path.join(self.shakespeare_char_dir, 'train.bin')
        val_bin = os.path.join(self.shakespeare_char_dir, 'val.bin')
        
        train_tokens, val_tokens = loader.load_from_binary(train_bin, val_bin)
        
        # Verify data was loaded
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        
        # Verify tokens are valid
        self.assertTrue(manager.validate_tokens(train_tokens[:100]))
        self.assertTrue(manager.validate_tokens(val_tokens[:100]))
    
    def test_data_loader_binary_mode_bpe(self):
        """Test DataLoader with binary mode and BPE tokenization."""
        if not self.shakespeare_bpe_available:
            self.skipTest("Shakespeare BPE binary files not available")
        
        # Initialize tokenization manager
        manager = TokenizationManager()
        manager.load_bpe_tokenization()
        loader = DataLoader(manager)
        
        # Load binary data
        train_bin = os.path.join(self.shakespeare_dir, 'train.bin')
        val_bin = os.path.join(self.shakespeare_dir, 'val.bin')
        
        train_tokens, val_tokens = loader.load_from_binary(train_bin, val_bin)
        
        # Verify data was loaded
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        
        # Verify tokens are valid
        self.assertTrue(manager.validate_tokens(train_tokens[:100]))
        self.assertTrue(manager.validate_tokens(val_tokens[:100]))
    
    def test_configuration_validation_integration(self):
        """Test configuration validation with real scenarios."""
        validator = ConfigurationValidator()

        # Create a dummy model file for testing
        dummy_model_path = os.path.join(self.temp_dir, 'dummy_model.pt')
        with open(dummy_model_path, 'w') as f:
            f.write('dummy model content')

        # Test valid text mode configuration
        if self.shakespeare_char_available:
            config = RewardDataConfig(
                model_path=dummy_model_path,
                input_mode='text',
                data_path=os.path.join(self.shakespeare_char_dir, 'input.txt'),
                tokenization='char',
                meta_path=os.path.join(self.shakespeare_char_dir, 'meta.pkl'),
                output_dir=self.test_output_dir
            )

            is_valid = validator.validate_config(config)
            self.assertTrue(is_valid)

        # Test valid binary mode configuration
        if self.shakespeare_bpe_available:
            config = RewardDataConfig(
                model_path=dummy_model_path,
                input_mode='binary',
                train_bin=os.path.join(self.shakespeare_dir, 'train.bin'),
                val_bin=os.path.join(self.shakespeare_dir, 'val.bin'),
                output_dir=self.test_output_dir
            )

            is_valid = validator.validate_config(config)
            self.assertTrue(is_valid)

        # Test invalid configuration (missing model)
        config = RewardDataConfig(
            model_path='nonexistent_model.pt',
            input_mode='text',
            data_path=os.path.join(self.shakespeare_char_dir, 'input.txt') if self.shakespeare_char_available else 'dummy.txt',
            output_dir=self.test_output_dir
        )

        is_valid = validator.validate_config(config)
        self.assertFalse(is_valid)
        self.assertTrue(len(config.get_errors()) > 0)
    
    def test_backward_compatibility_text_mode(self):
        """Test that existing text mode workflows still work."""
        if not self.shakespeare_char_available:
            self.skipTest("Shakespeare character data not available")
        
        # Test the legacy load_and_split_data function
        from prepare_reward_data import load_and_split_data
        
        input_path = os.path.join(self.shakespeare_char_dir, 'input.txt')
        train_tokens, val_tokens, tokenizer = load_and_split_data(input_path, train_split=0.9)
        
        # Verify data was loaded
        self.assertGreater(len(train_tokens), 0)
        self.assertGreater(len(val_tokens), 0)
        
        # Verify split ratio
        total_tokens = len(train_tokens) + len(val_tokens)
        actual_split = len(train_tokens) / total_tokens
        self.assertAlmostEqual(actual_split, 0.9, delta=0.01)
    
    def test_auto_detection_integration(self):
        """Test auto-detection with real data directories."""
        # Test character detection
        if self.shakespeare_char_available:
            manager = TokenizationManager()
            detected = manager.detect_tokenization_method(self.shakespeare_char_dir)
            self.assertEqual(detected, 'char')
        
        # Test BPE detection (default)
        if self.shakespeare_bpe_available:
            manager = TokenizationManager()
            detected = manager.detect_tokenization_method(self.shakespeare_dir)
            self.assertEqual(detected, 'bpe')
    
    def test_tokenization_info_serialization(self):
        """Test TokenizationInfo serialization and deserialization."""
        from reward_data_config import TokenizationInfo
        
        # Test character tokenization info
        if self.shakespeare_char_available:
            info = TokenizationInfo(
                method='char',
                vocab_size=65,
                meta_path=os.path.join(self.shakespeare_char_dir, 'meta.pkl')
            )
            
            # Serialize and deserialize
            data = info.to_dict()
            restored_info = TokenizationInfo.from_dict(data)
            
            self.assertEqual(restored_info.method, info.method)
            self.assertEqual(restored_info.vocab_size, info.vocab_size)
            self.assertEqual(restored_info.meta_path, info.meta_path)
        
        # Test BPE tokenization info
        info = TokenizationInfo(
            method='bpe',
            vocab_size=50257,
            meta_path=None
        )
        
        data = info.to_dict()
        restored_info = TokenizationInfo.from_dict(data)
        
        self.assertEqual(restored_info.method, info.method)
        self.assertEqual(restored_info.vocab_size, info.vocab_size)
        self.assertIsNone(restored_info.meta_path)
    
    def test_data_consistency_between_modes(self):
        """Test that text and binary modes produce consistent results."""
        if not (self.shakespeare_char_available and self.shakespeare_char_bins_available):
            self.skipTest("Both text and binary shakespeare_char data needed")
        
        # Load via text mode
        manager_text = TokenizationManager(data_path=self.shakespeare_char_dir)
        loader_text = DataLoader(manager_text)
        input_path = os.path.join(self.shakespeare_char_dir, 'input.txt')
        train_text, val_text = loader_text.load_from_text(input_path, train_split=0.9)
        
        # Load via binary mode
        manager_binary = TokenizationManager(data_path=self.shakespeare_char_dir)
        loader_binary = DataLoader(manager_binary)
        train_bin_path = os.path.join(self.shakespeare_char_dir, 'train.bin')
        val_bin_path = os.path.join(self.shakespeare_char_dir, 'val.bin')
        train_binary, val_binary = loader_binary.load_from_binary(train_bin_path, val_bin_path)
        
        # The binary files should match the expected split from text mode
        # (allowing for small differences due to rounding in split calculation)
        text_total = len(train_text) + len(val_text)
        binary_total = len(train_binary) + len(val_binary)
        
        # Total should be the same
        self.assertEqual(text_total, binary_total)
        
        # Split ratios should be similar
        text_split = len(train_text) / text_total
        binary_split = len(train_binary) / binary_total
        self.assertAlmostEqual(text_split, binary_split, delta=0.01)


if __name__ == '__main__':
    unittest.main()
