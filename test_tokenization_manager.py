#!/usr/bin/env python3
"""
Unit tests for TokenizationManager class.
"""

import os
import tempfile
import pickle
import unittest
from unittest.mock import patch, MagicMock
import tiktoken

from tokenization_manager import (
    TokenizationManager, 
    TokenizationError, 
    TokenizationDetectionError, 
    MetaFileError
)


class TestTokenizationManager(unittest.TestCase):
    """Test cases for TokenizationManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample meta.pkl file for character tokenization
        self.sample_meta = {
            'vocab_size': 65,
            'itos': {i: chr(ord('a') + i) for i in range(26)},  # Simple a-z mapping
            'stoi': {chr(ord('a') + i): i for i in range(26)}
        }
        # Add some special characters
        self.sample_meta['itos'].update({26: ' ', 27: '.', 28: '!', 29: '?'})
        self.sample_meta['stoi'].update({' ': 26, '.': 27, '!': 28, '?': 29})
        
        self.meta_path = os.path.join(self.temp_dir, 'meta.pkl')
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.sample_meta, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_bpe_tokenization_initialization(self):
        """Test BPE tokenization initialization."""
        manager = TokenizationManager()
        manager.load_bpe_tokenization()
        
        self.assertEqual(manager.tokenization_type, 'bpe')
        self.assertIsNotNone(manager.vocab_size)
        self.assertIsNotNone(manager.encoder)
        self.assertIsNotNone(manager.decoder)
        self.assertIsInstance(manager.tiktoken_encoder, tiktoken.Encoding)
    
    def test_bpe_encoding_decoding(self):
        """Test BPE encoding and decoding."""
        manager = TokenizationManager()
        manager.load_bpe_tokenization()
        
        test_text = "Hello, world!"
        tokens = manager.encode(test_text)
        decoded_text = manager.decode(tokens)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens))
        self.assertEqual(decoded_text, test_text)
    
    def test_char_tokenization_initialization(self):
        """Test character tokenization initialization."""
        manager = TokenizationManager()
        manager.load_char_tokenization(self.meta_path)
        
        self.assertEqual(manager.tokenization_type, 'char')
        self.assertEqual(manager.vocab_size, 65)
        self.assertIsNotNone(manager.encoder)
        self.assertIsNotNone(manager.decoder)
        self.assertEqual(manager.meta_path, self.meta_path)
    
    def test_char_encoding_decoding(self):
        """Test character encoding and decoding."""
        manager = TokenizationManager()
        manager.load_char_tokenization(self.meta_path)
        
        test_text = "hello world."
        tokens = manager.encode(test_text)
        decoded_text = manager.decode(tokens)
        
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens))
        self.assertEqual(decoded_text, test_text)
        
        # Test specific character mappings
        expected_tokens = [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3, 27]  # hello world.
        self.assertEqual(tokens, expected_tokens)
    
    def test_auto_detection_with_meta_file(self):
        """Test auto-detection when meta.pkl exists."""
        # Create a directory with meta.pkl
        data_dir = os.path.join(self.temp_dir, 'shakespeare_char')
        os.makedirs(data_dir)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.sample_meta, f)
        
        manager = TokenizationManager()
        detected_method = manager.detect_tokenization_method(data_dir)
        
        self.assertEqual(detected_method, 'char')
    
    def test_auto_detection_with_char_in_name(self):
        """Test auto-detection based on directory name containing 'char'."""
        char_dir = os.path.join(self.temp_dir, 'shakespeare_char')
        os.makedirs(char_dir)
        
        manager = TokenizationManager()
        detected_method = manager.detect_tokenization_method(char_dir)
        
        self.assertEqual(detected_method, 'char')
    
    def test_auto_detection_defaults_to_bpe(self):
        """Test auto-detection defaults to BPE when no indicators found."""
        regular_dir = os.path.join(self.temp_dir, 'shakespeare')
        os.makedirs(regular_dir)
        
        manager = TokenizationManager()
        detected_method = manager.detect_tokenization_method(regular_dir)
        
        self.assertEqual(detected_method, 'bpe')
    
    def test_auto_detection_with_file_path(self):
        """Test auto-detection when given a file path instead of directory."""
        # Create a file in a directory with meta.pkl
        data_dir = os.path.join(self.temp_dir, 'shakespeare_char')
        os.makedirs(data_dir)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.sample_meta, f)
        
        file_path = os.path.join(data_dir, 'input.txt')
        with open(file_path, 'w') as f:
            f.write("test content")
        
        manager = TokenizationManager()
        detected_method = manager.detect_tokenization_method(file_path)
        
        self.assertEqual(detected_method, 'char')
    
    def test_initialization_with_data_path(self):
        """Test initialization with data_path for auto-detection."""
        # Create directory with meta.pkl
        data_dir = os.path.join(self.temp_dir, 'shakespeare_char')
        os.makedirs(data_dir)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.sample_meta, f)
        
        manager = TokenizationManager(data_path=data_dir)
        
        self.assertEqual(manager.tokenization_type, 'char')
        self.assertEqual(manager.vocab_size, 65)
    
    def test_initialization_with_explicit_meta_path(self):
        """Test initialization with explicit meta_path."""
        manager = TokenizationManager(meta_path=self.meta_path)
        
        self.assertEqual(manager.tokenization_type, 'char')
        self.assertEqual(manager.vocab_size, 65)
    
    def test_token_validation(self):
        """Test token validation functionality."""
        manager = TokenizationManager()
        manager.load_char_tokenization(self.meta_path)
        
        # Valid tokens
        valid_tokens = [0, 10, 25, 64]
        self.assertTrue(manager.validate_tokens(valid_tokens))
        
        # Invalid tokens (out of range)
        invalid_tokens = [0, 10, 65, 100]  # 65 and 100 are out of range
        self.assertFalse(manager.validate_tokens(invalid_tokens))
        
        # Negative tokens
        negative_tokens = [-1, 0, 10]
        self.assertFalse(manager.validate_tokens(negative_tokens))
    
    def test_get_tokenization_info(self):
        """Test getting tokenization information."""
        manager = TokenizationManager()
        manager.load_char_tokenization(self.meta_path)
        
        info = manager.get_tokenization_info()
        
        self.assertEqual(info['method'], 'char')
        self.assertEqual(info['vocab_size'], 65)
        self.assertEqual(info['meta_path'], self.meta_path)
        self.assertTrue(info['is_initialized'])
    
    def test_compatibility_check(self):
        """Test compatibility checking between configurations."""
        manager1 = TokenizationManager()
        manager1.load_char_tokenization(self.meta_path)
        
        manager2 = TokenizationManager()
        manager2.load_bpe_tokenization()
        
        # Same configuration should be compatible
        info1 = manager1.get_tokenization_info()
        self.assertTrue(manager1.is_compatible_with(info1))
        
        # Different configurations should not be compatible
        info2 = manager2.get_tokenization_info()
        self.assertFalse(manager1.is_compatible_with(info2))
    
    def test_missing_meta_file_error(self):
        """Test error handling for missing meta.pkl file."""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.pkl')
        
        manager = TokenizationManager()
        with self.assertRaises(MetaFileError):
            manager.load_char_tokenization(nonexistent_path)
    
    def test_invalid_meta_file_error(self):
        """Test error handling for invalid meta.pkl file."""
        # Create invalid meta file
        invalid_meta_path = os.path.join(self.temp_dir, 'invalid_meta.pkl')
        with open(invalid_meta_path, 'wb') as f:
            pickle.dump({'invalid': 'data'}, f)
        
        manager = TokenizationManager()
        with self.assertRaises(MetaFileError):
            manager.load_char_tokenization(invalid_meta_path)
    
    def test_detection_error_for_nonexistent_path(self):
        """Test error handling for nonexistent data path."""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent')
        
        manager = TokenizationManager()
        with self.assertRaises(TokenizationDetectionError):
            manager.detect_tokenization_method(nonexistent_path)
    
    def test_encoding_without_initialization(self):
        """Test error handling when encoding without initialization."""
        manager = TokenizationManager()
        
        with self.assertRaises(TokenizationError):
            manager.encode("test text")
    
    def test_decoding_without_initialization(self):
        """Test error handling when decoding without initialization."""
        manager = TokenizationManager()
        
        with self.assertRaises(TokenizationError):
            manager.decode([1, 2, 3])


if __name__ == '__main__':
    unittest.main()
