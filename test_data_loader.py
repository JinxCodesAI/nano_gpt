#!/usr/bin/env python3
"""
Unit tests for DataLoader class.
"""

import os
import tempfile
import pickle
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from tokenization_manager import TokenizationManager
from data_loader import DataLoader, DataLoadError, BinaryFileError, TextFileError


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample meta.pkl for character tokenization
        self.sample_meta = {
            'vocab_size': 30,
            'itos': {i: chr(ord('a') + i) for i in range(26)},
            'stoi': {chr(ord('a') + i): i for i in range(26)}
        }
        # Add special characters
        self.sample_meta['itos'].update({26: ' ', 27: '.', 28: '!', 29: '?'})
        self.sample_meta['stoi'].update({' ': 26, '.': 27, '!': 28, '?': 29})
        
        self.meta_path = os.path.join(self.temp_dir, 'meta.pkl')
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.sample_meta, f)
        
        # Create sample text file
        self.text_content = "hello world. this is a test!"
        self.text_path = os.path.join(self.temp_dir, 'input.txt')
        with open(self.text_path, 'w', encoding='utf-8') as f:
            f.write(self.text_content)
        
        # Create sample binary files
        self.train_tokens = [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, 27]  # "hello world."
        self.val_tokens = [19, 7, 8, 18, 26, 8, 18, 26, 0, 26, 19, 4, 18, 19, 28]  # "this is a test!"
        
        self.train_bin_path = os.path.join(self.temp_dir, 'train.bin')
        self.val_bin_path = os.path.join(self.temp_dir, 'val.bin')
        
        np.array(self.train_tokens, dtype=np.uint16).tofile(self.train_bin_path)
        np.array(self.val_tokens, dtype=np.uint16).tofile(self.val_bin_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_with_valid_tokenizer(self):
        """Test DataLoader initialization with valid tokenizer."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        
        loader = DataLoader(tokenizer)
        self.assertEqual(loader.tokenizer, tokenizer)
    
    def test_initialization_with_uninitialized_tokenizer(self):
        """Test DataLoader initialization with uninitialized tokenizer."""
        tokenizer = TokenizationManager()
        
        with self.assertRaises(DataLoadError):
            DataLoader(tokenizer)
    
    def test_load_from_text_with_char_tokenization(self):
        """Test loading text file with character tokenization."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        train_tokens, val_tokens = loader.load_from_text(self.text_path, train_split=0.6)
        
        # Check that tokens were generated
        self.assertIsInstance(train_tokens, list)
        self.assertIsInstance(val_tokens, list)
        self.assertTrue(len(train_tokens) > 0)
        self.assertTrue(len(val_tokens) > 0)
        
        # Check split ratio (approximately)
        total_tokens = len(train_tokens) + len(val_tokens)
        actual_split = len(train_tokens) / total_tokens
        self.assertAlmostEqual(actual_split, 0.6, delta=0.1)
    
    def test_load_from_text_with_bpe_tokenization(self):
        """Test loading text file with BPE tokenization."""
        tokenizer = TokenizationManager()
        tokenizer.load_bpe_tokenization()
        loader = DataLoader(tokenizer)
        
        train_tokens, val_tokens = loader.load_from_text(self.text_path)
        
        self.assertIsInstance(train_tokens, list)
        self.assertIsInstance(val_tokens, list)
        self.assertTrue(len(train_tokens) > 0)
        self.assertTrue(len(val_tokens) > 0)
        
        # Default split should be 0.9
        total_tokens = len(train_tokens) + len(val_tokens)
        actual_split = len(train_tokens) / total_tokens
        self.assertAlmostEqual(actual_split, 0.9, delta=0.05)
    
    def test_load_from_text_missing_file(self):
        """Test error handling for missing text file."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.txt')
        
        with self.assertRaises(TextFileError):
            loader.load_from_text(nonexistent_path)
    
    def test_load_from_text_empty_file(self):
        """Test error handling for empty text file."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        empty_file = os.path.join(self.temp_dir, 'empty.txt')
        with open(empty_file, 'w') as f:
            f.write("")
        
        with self.assertRaises(TextFileError):
            loader.load_from_text(empty_file)
    
    def test_load_from_text_invalid_split(self):
        """Test error handling for invalid train_split values."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        # Test split values outside valid range
        with self.assertRaises(TextFileError):
            loader.load_from_text(self.text_path, train_split=0.0)
        
        with self.assertRaises(TextFileError):
            loader.load_from_text(self.text_path, train_split=1.0)
        
        with self.assertRaises(TextFileError):
            loader.load_from_text(self.text_path, train_split=-0.1)
        
        with self.assertRaises(TextFileError):
            loader.load_from_text(self.text_path, train_split=1.1)
    
    def test_load_from_binary_valid_files(self):
        """Test loading from valid binary files."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        train_tokens, val_tokens = loader.load_from_binary(self.train_bin_path, self.val_bin_path)
        
        self.assertEqual(train_tokens, self.train_tokens)
        self.assertEqual(val_tokens, self.val_tokens)
    
    def test_load_from_binary_missing_train_file(self):
        """Test error handling for missing train binary file."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        nonexistent_train = os.path.join(self.temp_dir, 'nonexistent_train.bin')
        
        with self.assertRaises(BinaryFileError):
            loader.load_from_binary(nonexistent_train, self.val_bin_path)
    
    def test_load_from_binary_missing_val_file(self):
        """Test error handling for missing validation binary file."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        nonexistent_val = os.path.join(self.temp_dir, 'nonexistent_val.bin')
        
        with self.assertRaises(BinaryFileError):
            loader.load_from_binary(self.train_bin_path, nonexistent_val)
    
    def test_load_from_binary_empty_file(self):
        """Test error handling for empty binary file."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        empty_bin = os.path.join(self.temp_dir, 'empty.bin')
        with open(empty_bin, 'wb') as f:
            pass  # Create empty file
        
        with self.assertRaises(BinaryFileError):
            loader.load_from_binary(empty_bin, self.val_bin_path)
    
    def test_load_from_binary_incompatible_tokens(self):
        """Test error handling for tokens outside vocab range."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        # Create binary file with tokens outside vocab range
        invalid_tokens = [0, 10, 50, 100]  # 50 and 100 are outside vocab_size=30
        invalid_bin = os.path.join(self.temp_dir, 'invalid.bin')
        np.array(invalid_tokens, dtype=np.uint16).tofile(invalid_bin)
        
        with self.assertRaises(BinaryFileError):
            loader.load_from_binary(invalid_bin, self.val_bin_path)
    
    def test_validate_binary_files_valid(self):
        """Test binary file validation with valid files."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        result = loader.validate_binary_files(self.train_bin_path, self.val_bin_path)
        self.assertTrue(result)
    
    def test_validate_binary_files_missing(self):
        """Test binary file validation with missing files."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        nonexistent = os.path.join(self.temp_dir, 'nonexistent.bin')
        
        with self.assertRaises(BinaryFileError):
            loader.validate_binary_files(nonexistent, self.val_bin_path)
    
    def test_validate_binary_files_odd_size(self):
        """Test binary file validation with odd file size."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        # Create file with odd number of bytes (not multiple of 2)
        odd_size_bin = os.path.join(self.temp_dir, 'odd_size.bin')
        with open(odd_size_bin, 'wb') as f:
            f.write(b'abc')  # 3 bytes, not multiple of 2
        
        with self.assertRaises(BinaryFileError):
            loader.validate_binary_files(odd_size_bin, self.val_bin_path)
    
    def test_get_data_info(self):
        """Test getting data information."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        train_tokens = [1, 2, 3, 4, 5]
        val_tokens = [6, 7, 8]
        
        info = loader.get_data_info(train_tokens, val_tokens)
        
        self.assertEqual(info['train_size'], 5)
        self.assertEqual(info['val_size'], 3)
        self.assertEqual(info['total_size'], 8)
        self.assertEqual(info['train_split_ratio'], 5/8)
        self.assertEqual(info['tokenization_method'], 'char')
        self.assertEqual(info['vocab_size'], 30)
        self.assertEqual(info['train_token_range'], (1, 5))
        self.assertEqual(info['val_token_range'], (6, 8))
    
    def test_get_data_info_empty_tokens(self):
        """Test getting data info with empty token lists."""
        tokenizer = TokenizationManager()
        tokenizer.load_char_tokenization(self.meta_path)
        loader = DataLoader(tokenizer)
        
        info = loader.get_data_info([], [])
        
        self.assertEqual(info['train_size'], 0)
        self.assertEqual(info['val_size'], 0)
        self.assertEqual(info['total_size'], 0)
        self.assertNotIn('train_token_range', info)
        self.assertNotIn('val_token_range', info)


if __name__ == '__main__':
    unittest.main()
