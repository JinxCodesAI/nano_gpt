#!/usr/bin/env python3
"""
Unit tests for completion_variance_analysis.py

Tests the major components of the completion variance analysis script
using mock models and data to avoid requiring actual trained models.
"""

import unittest
import tempfile
import os
import pickle
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from completion_variance_analysis import CompletionVarianceAnalyzer, parse_arguments
from model import ModelMode, GPTConfig


class MockArgs:
    """Mock arguments for testing"""
    def __init__(self):
        self.language_model = 'test_language_model.pt'
        self.scoring_model = 'test_scoring_model.pt'
        self.dataset = 'test_dataset'
        self.out_dir = 'test_out'
        self.batch_size = 4
        self.num_samples = 2
        self.sequence_length = 64
        self.mask_ratio = 0.3
        self.temperature = 1.0
        self.top_p = 1.0
        self.diffusion_iterations = 3
        self.seed = 42
        self.verbose = False
        self.debug = False
        self.device = 'cpu'
        self.dtype = 'float32'
        self.compile = False


class MockModel:
    """Mock model for testing"""
    def __init__(self, mode):
        self.config = Mock()
        self.config.mode = mode
        self.config.attention_type = 'bidirectional'

    def eval(self):
        return self

    def to(self, device):
        return self

    def get_num_params(self):
        return 1000000

    def load_state_dict(self, state_dict):
        """Mock load_state_dict method"""
        pass

    def __call__(self, tokens, targets):
        batch_size, seq_len = tokens.shape
        vocab_size = 100

        if self.config.mode == ModelMode.LANGUAGE_MODEL:
            # Return logits for language model
            logits = torch.randn(batch_size, seq_len, vocab_size)
            return logits, None
        elif self.config.mode == ModelMode.SEQUENCE_SCORER:
            # Return sequence scores
            scores = torch.rand(batch_size, 1)
            return scores, None


class TestCompletionVarianceAnalyzer(unittest.TestCase):
    """Test cases for CompletionVarianceAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.args = MockArgs()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock vocabulary
        self.vocab_info = {
            'stoi': {chr(i): i for i in range(65, 91)},  # A-Z
            'itos': {i: chr(i) for i in range(65, 91)},
            'vocab_size': 26,
            'mask_token_id': 26,
            'cls_token_id': 31,
            'extended_vocab_size': 41,
            'decode': lambda tokens: ''.join([chr(t) if t < 91 else '[MASK]' if t == 26 else '[CLS]' if t == 31 else '[UNK]' for t in tokens]),
            'dataset_name': 'test_dataset'
        }
        
        # Create mock training data
        self.training_data = np.random.randint(65, 91, size=1000, dtype=np.uint16)
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_meta_file(self):
        """Create a mock meta.pkl file"""
        data_dir = os.path.join(self.temp_dir, 'data', self.args.dataset)
        os.makedirs(data_dir, exist_ok=True)
        
        meta = {
            'vocab_size': 26,
            'stoi': self.vocab_info['stoi'],
            'itos': self.vocab_info['itos']
        }
        
        meta_path = os.path.join(data_dir, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        
        return meta_path
    
    def create_mock_training_data(self):
        """Create mock training data file"""
        data_dir = os.path.join(self.temp_dir, 'data', self.args.dataset)
        os.makedirs(data_dir, exist_ok=True)
        
        train_path = os.path.join(data_dir, 'train.bin')
        self.training_data.tofile(train_path)
        
        return train_path
    
    def create_mock_checkpoint(self, model_mode):
        """Create a mock model checkpoint"""
        checkpoint = {
            'model_args': {
                'n_layer': 2,
                'n_head': 2,
                'n_embd': 64,
                'block_size': 64,
                'vocab_size': 41,
                'bias': False,
                'dropout': 0.0,
                'attention_type': 'bidirectional',
                'mode': model_mode,
                'num_token_classes': 2,
                'cls_token_id': 31
            },
            'model': {}  # Empty state dict for testing
        }
        return checkpoint
    
    @patch('completion_variance_analysis.os.path.exists')
    @patch('completion_variance_analysis.torch.load')
    @patch('completion_variance_analysis.GPT')
    def test_load_model(self, mock_gpt_class, mock_torch_load, mock_exists):
        """Test model loading functionality"""
        # Setup mocks
        mock_exists.return_value = True
        mock_checkpoint = self.create_mock_checkpoint(ModelMode.LANGUAGE_MODEL)
        mock_torch_load.return_value = mock_checkpoint

        mock_model = MockModel(ModelMode.LANGUAGE_MODEL)
        mock_gpt_class.return_value = mock_model

        # Create analyzer
        analyzer = CompletionVarianceAnalyzer(self.args)

        # Test loading language model
        model = analyzer.load_model('test_model.pt', ModelMode.LANGUAGE_MODEL)

        self.assertIsNotNone(model)
        self.assertEqual(model.config.mode, ModelMode.LANGUAGE_MODEL)
        mock_torch_load.assert_called_once()
        mock_gpt_class.assert_called_once()
    
    def test_load_vocabulary(self):
        """Test vocabulary loading"""
        # Create mock meta file
        self.create_mock_meta_file()
        
        # Change working directory temporarily
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            
            analyzer = CompletionVarianceAnalyzer(self.args)
            vocab_info = analyzer.load_vocabulary()
            
            self.assertEqual(vocab_info['vocab_size'], 26)
            self.assertEqual(vocab_info['mask_token_id'], 26)
            self.assertEqual(vocab_info['cls_token_id'], 31)
            self.assertIn('decode', vocab_info)
            
        finally:
            os.chdir(original_cwd)
    
    @patch('completion_variance_analysis.os.path.exists')
    @patch('completion_variance_analysis.load_memmap_data')
    def test_load_training_data(self, mock_load_memmap, mock_exists):
        """Test training data loading"""
        # Setup mocks
        mock_exists.return_value = True
        mock_load_memmap.return_value = self.training_data

        analyzer = CompletionVarianceAnalyzer(self.args)
        analyzer.load_training_data()

        self.assertIsNotNone(analyzer.training_data)
        self.assertEqual(len(analyzer.training_data), 1000)
        mock_load_memmap.assert_called_once()
    
    @patch('completion_variance_analysis.apply_random_masking_gpu')
    def test_sample_and_mask_data(self, mock_masking):
        """Test data sampling and masking"""
        # Setup mocks
        mock_tokens = torch.randint(65, 91, (1, 64))
        mock_mask = torch.rand(1, 64) < 0.3
        mock_masking.return_value = (mock_tokens, mock_mask)
        
        analyzer = CompletionVarianceAnalyzer(self.args)
        analyzer.vocab_info = self.vocab_info
        analyzer.training_data = self.training_data
        
        masked_tokens, mask = analyzer.sample_and_mask_data(0)
        
        self.assertEqual(masked_tokens.shape, (1, 64))
        self.assertEqual(mask.shape, (1, 64))
        mock_masking.assert_called_once()
    
    def test_rate_completions(self):
        """Test completion rating functionality"""
        analyzer = CompletionVarianceAnalyzer(self.args)
        analyzer.vocab_info = self.vocab_info
        analyzer.scoring_model = MockModel(ModelMode.SEQUENCE_SCORER)
        analyzer.device = 'cpu'
        analyzer.ctx = torch.no_grad()
        
        # Create mock completions
        completions = [torch.randint(65, 91, (1, 64)) for _ in range(3)]
        
        with patch('completion_variance_analysis.calculate_sequence_scores') as mock_calc_scores:
            mock_calc_scores.return_value = [0.5]
            
            ratings = analyzer.rate_completions(completions)
            
            self.assertEqual(len(ratings), 3)
            self.assertEqual(mock_calc_scores.call_count, 3)
    
    def test_check_completion_diversity(self):
        """Test completion diversity checking"""
        analyzer = CompletionVarianceAnalyzer(self.args)
        analyzer.vocab_info = self.vocab_info
        
        # Create completions with known diversity
        completions = [
            torch.tensor([[65, 66, 67, 68]]),  # ABCD
            torch.tensor([[65, 66, 67, 69]]),  # ABCE (different)
            torch.tensor([[65, 66, 67, 68]]),  # ABCD (duplicate)
        ]
        
        diversity_stats = analyzer.check_completion_diversity(completions)
        
        self.assertEqual(diversity_stats['total_completions'], 3)
        self.assertEqual(diversity_stats['unique_completions'], 2)
        self.assertAlmostEqual(diversity_stats['diversity_ratio'], 2/3)
        self.assertEqual(len(diversity_stats['token_differences']), 3)  # 3 pairwise comparisons
    
    def test_analyze_rating_variance(self):
        """Test rating variance analysis"""
        analyzer = CompletionVarianceAnalyzer(self.args)
        
        ratings = [0.1, 0.3, 0.5, 0.7, 0.9]
        diversity_stats = {
            'diversity_ratio': 1.0,
            'unique_completions': 5,
            'total_completions': 5,
            'token_differences': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        
        variance_stats = analyzer.analyze_rating_variance(ratings, diversity_stats)
        
        self.assertEqual(variance_stats['num_ratings'], 5)
        self.assertAlmostEqual(variance_stats['mean_rating'], 0.5)
        self.assertAlmostEqual(variance_stats['min_rating'], 0.1)
        self.assertAlmostEqual(variance_stats['max_rating'], 0.9)
        self.assertAlmostEqual(variance_stats['range_rating'], 0.8)
        self.assertEqual(variance_stats['diversity_ratio'], 1.0)
        self.assertIn('mean_token_differences', variance_stats)
    
    def test_parse_arguments(self):
        """Test argument parsing"""
        test_args = [
            '--language_model', 'lang_model.pt',
            '--scoring_model', 'score_model.pt',
            '--batch_size', '8',
            '--num_samples', '5',
            '--mask_ratio', '0.4',
            '--verbose'
        ]
        
        with patch('sys.argv', ['test'] + test_args):
            args = parse_arguments()
            
            self.assertEqual(args.language_model, 'lang_model.pt')
            self.assertEqual(args.scoring_model, 'score_model.pt')
            self.assertEqual(args.batch_size, 8)
            self.assertEqual(args.num_samples, 5)
            self.assertEqual(args.mask_ratio, 0.4)
            self.assertTrue(args.verbose)


if __name__ == '__main__':
    unittest.main()
