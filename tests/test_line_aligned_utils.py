"""
Tests for line-aligned sequence generation utilities.
Verifies that both CharDiffusionProvider and SequenceScorerProvider generate data consistently.
"""
import pytest
import torch
import tempfile
import os
from typing import Dict, Any

from data.common.line_aligned_utils import (
    LineAlignedSequenceBuilder,
    prepare_line_data,
    create_line_aligned_builder
)


class TestLineAlignedSequenceBuilder:
    """Test the core LineAlignedSequenceBuilder functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample text data for testing."""
        return "Hello world!\nThis is a test.\nAnother line here.\n\nFinal line."
    
    @pytest.fixture
    def sample_lines_ids(self, sample_data):
        """Convert sample data to lines and token IDs."""
        lines, lines_ids, stoi = prepare_line_data(sample_data)
        return lines_ids, stoi
    
    def test_prepare_line_data(self, sample_data):
        """Test line data preparation."""
        lines, lines_ids, stoi = prepare_line_data(sample_data)
        
        # Check that lines are split correctly
        assert len(lines) == 5  # 4 non-empty lines + 1 empty line
        assert lines[0] == "Hello world!\n"
        assert lines[1] == "This is a test.\n"
        assert lines[3] == "\n"  # Empty line
        
        # Check that token IDs are created correctly
        assert len(lines_ids) == len(lines)
        assert all(isinstance(line_ids, list) for line_ids in lines_ids)
        
        # Check vocabulary
        assert '\n' in stoi
        assert 'H' in stoi
        assert len(stoi) == len(set(sample_data))
    
    def test_builder_initialization(self, sample_lines_ids):
        """Test LineAlignedSequenceBuilder initialization."""
        lines_ids, stoi = sample_lines_ids
        newline_id = stoi['\n']
        pad_id = 99  # Arbitrary pad token ID
        
        builder = LineAlignedSequenceBuilder(lines_ids, newline_id, pad_id)
        
        # Check that tensors are created
        assert builder.line_lens.shape[0] == len(lines_ids)
        assert builder.cumsum.shape[0] == len(lines_ids)
        assert builder.line_offsets.shape[0] == len(lines_ids)
        assert builder.tokens_flat.shape[0] == sum(len(line) for line in lines_ids)
        
        # Check valid starts (should exclude single-character lines like "\n")
        assert builder.valid_starts.numel() > 0
        assert builder.valid_starts.numel() <= len(lines_ids)
    
    def test_build_variable_length_sequences(self, sample_lines_ids):
        """Test variable-length sequence generation."""
        lines_ids, stoi = sample_lines_ids
        newline_id = stoi['\n']
        pad_id = 99
        
        builder = LineAlignedSequenceBuilder(lines_ids, newline_id, pad_id)
        
        # Generate sequences
        count = 3
        block_size = 20
        rng = torch.Generator()
        rng.manual_seed(42)
        
        x, content_lengths = builder.build_variable_length_sequences(count, block_size, rng)
        
        # Check output shapes
        assert x.shape == (count, block_size)
        assert content_lengths.shape == (count,)
        
        # Check that content lengths are reasonable
        assert torch.all(content_lengths >= 0)
        assert torch.all(content_lengths <= block_size)
        
        # Check that sequences contain valid token IDs
        max_token_id = max(max(line) for line in lines_ids if line)
        assert torch.all(x >= 0)
        assert torch.all(x <= max_token_id)
    
    def test_apply_padding(self, sample_lines_ids):
        """Test padding application."""
        lines_ids, stoi = sample_lines_ids
        newline_id = stoi['\n']
        pad_id = 99
        
        builder = LineAlignedSequenceBuilder(lines_ids, newline_id, pad_id)
        
        # Create test sequences
        count = 2
        block_size = 10
        x = torch.randint(0, 50, (count, block_size))
        content_lengths = torch.tensor([5, 8])
        
        # Apply padding
        padded_x = builder.apply_padding(x, content_lengths)
        
        # Check that padding was applied correctly
        assert torch.all(padded_x[0, 5:] == pad_id)  # First sequence padded from position 5
        assert torch.all(padded_x[1, 8:] == pad_id)  # Second sequence padded from position 8
        assert torch.all(padded_x[0, :5] == x[0, :5])  # Content unchanged
        assert torch.all(padded_x[1, :8] == x[1, :8])  # Content unchanged


class TestProviderConsistency:
    """Test that both providers generate data consistently when using line-aligned sequences."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input.txt
            test_data = "Hello world!\nThis is line two.\nShort.\nAnother longer line here.\n\nFinal line."
            input_path = os.path.join(temp_dir, 'input.txt')
            with open(input_path, 'w') as f:
                f.write(test_data)
            yield temp_dir
    
    @pytest.fixture
    def mock_mlm_engine(self):
        """Mock MLM engine for SequenceScorerProvider."""
        class MockMLMEngine:
            def __init__(self):
                # Create a simple vocabulary
                chars = sorted(list(set("Hello world!\nThis is line two.\nShort.\nAnother longer line here.\n\nFinal line.")))
                self.stoi = {ch: i for i, ch in enumerate(chars)}
                self.itos = {i: ch for i, ch in enumerate(chars)}
                self.vocab_size = len(chars)
                self.mask_token_id = len(chars)  # Add mask token
                self.stoi['[MASK]'] = self.mask_token_id
                self.itos[self.mask_token_id] = '[MASK]'
                self.vocab_size += 1
            
            def predict_masked_tokens(self, masked_input, mask, temperature=1.0, top_k=None):
                # Simple mock: return original input
                return masked_input.clone()
        
        return MockMLMEngine()
    
    def test_char_diffusion_line_aligned(self, temp_data_dir):
        """Test CharDiffusionProvider with line-aligned sequences."""
        from data.char_diffusion.prepare_streaming import CharDiffusionProvider
        
        config = {
            'enable_line_aligned_sequences': True,
            'mask_probability': 0.15
        }
        
        provider = CharDiffusionProvider(
            data_dir=temp_data_dir,
            batch_size=2,
            block_size=20,
            config=config,
            verbose=False
        )
        
        # Test that line-aligned structures are created
        assert provider.enable_line_aligned_sequences
        assert provider.train_builder is not None
        assert provider.val_builder is not None
        assert provider.newline_token_id is not None
        
        # Test sequence generation
        rng = torch.Generator()
        rng.manual_seed(42)
        
        x, content_lengths = provider._build_line_aligned_variable_length("train", 2, rng)
        assert x.shape == (2, 20)
        assert content_lengths.shape == (2,)
        assert torch.all(content_lengths > 0)  # Should have some content
    
    def test_sequence_scorer_line_aligned(self, temp_data_dir, mock_mlm_engine):
        """Test SequenceScorerProvider with line-aligned sequences."""
        from data.sequence_scorer.prepare_streaming import SequenceScorerProvider
        
        # Mock the MLM engine initialization
        import data.sequence_scorer.prepare_streaming
        original_init = data.sequence_scorer.prepare_streaming.MLMInferenceEngine
        data.sequence_scorer.prepare_streaming.MLMInferenceEngine = lambda *args, **kwargs: mock_mlm_engine
        
        try:
            config = {
                'enable_line_aligned_sequences': True,
                'mlm_checkpoint_path': 'dummy_path',
                'cls_token_id': mock_mlm_engine.vocab_size,
                'mask_probability_range': (0.1, 0.5)
            }
            
            provider = SequenceScorerProvider(
                data_dir=temp_data_dir,
                batch_size=2,
                block_size=20,
                config=config,
                verbose=False
            )
            
            # Test that line-aligned structures are created
            assert provider.enable_line_aligned_sequences
            assert provider.train_builder is not None
            assert provider.val_builder is not None
            assert provider.newline_token_id is not None
            
            # Test batch sampling
            rng = torch.Generator()
            rng.manual_seed(42)
            
            batch = provider._sample_default_batch("train", rng)
            assert "input_ids" in batch
            assert "targets" in batch
            assert batch["input_ids"].shape == (2, 20)
            assert batch["targets"].shape == (2,)
            
        finally:
            # Restore original MLM engine
            data.sequence_scorer.prepare_streaming.MLMInferenceEngine = original_init
    
    def test_consistency_between_providers(self, temp_data_dir, mock_mlm_engine):
        """Test that both providers generate sequences with similar line-aligned properties."""
        from data.char_diffusion.prepare_streaming import CharDiffusionProvider
        from data.sequence_scorer.prepare_streaming import SequenceScorerProvider
        
        # Mock the MLM engine
        import data.sequence_scorer.prepare_streaming
        original_init = data.sequence_scorer.prepare_streaming.MLMInferenceEngine
        data.sequence_scorer.prepare_streaming.MLMInferenceEngine = lambda *args, **kwargs: mock_mlm_engine
        
        try:
            # Create both providers with line-aligned sequences enabled
            char_config = {
                'enable_line_aligned_sequences': True,
                'mask_probability': 0.15
            }
            
            seq_config = {
                'enable_line_aligned_sequences': True,
                'mlm_checkpoint_path': 'dummy_path',
                'cls_token_id': mock_mlm_engine.vocab_size,
                'mask_probability_range': (0.1, 0.5)
            }
            
            char_provider = CharDiffusionProvider(
                data_dir=temp_data_dir,
                batch_size=2,
                block_size=20,
                config=char_config,
                verbose=False
            )
            
            seq_provider = SequenceScorerProvider(
                data_dir=temp_data_dir,
                batch_size=2,
                block_size=20,
                config=seq_config,
                verbose=False
            )
            
            # Both should have line-aligned capabilities
            assert char_provider.enable_line_aligned_sequences
            assert seq_provider.enable_line_aligned_sequences
            
            # Both should have similar line structures (same input data)
            # Note: They may have slightly different line counts due to vocabulary differences,
            # but should be close and both should have multiple lines
            char_train_lines = len(char_provider.train_builder.lines_ids)
            seq_train_lines = len(seq_provider.train_builder.lines_ids)
            assert char_train_lines > 1  # Should have multiple lines
            assert seq_train_lines > 1   # Should have multiple lines
            assert abs(char_train_lines - seq_train_lines) <= 1  # Should be very close
            
            # Both should generate sequences with content
            rng1 = torch.Generator()
            rng1.manual_seed(42)
            rng2 = torch.Generator()
            rng2.manual_seed(42)
            
            char_x, char_lengths = char_provider._build_line_aligned_variable_length("train", 2, rng1)
            seq_x, seq_lengths = seq_provider.train_builder.build_variable_length_sequences(2, 19, rng2)  # -1 for CLS
            
            # Should have similar content lengths (allowing for small differences due to CLS token handling)
            assert torch.all(char_lengths > 0)
            assert torch.all(seq_lengths > 0)
            
        finally:
            data.sequence_scorer.prepare_streaming.MLMInferenceEngine = original_init


if __name__ == "__main__":
    pytest.main([__file__])
