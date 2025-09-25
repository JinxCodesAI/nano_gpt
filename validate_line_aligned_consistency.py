#!/usr/bin/env python3
"""
Validation script to test that both CharDiffusionProvider and SequenceScorerProvider
generate data consistently when using line-aligned sequences.
"""
import os
import tempfile
import torch
from typing import Dict, Any

def create_test_data(data_dir: str) -> str:
    """Create test input.txt file."""
    test_data = """Hello world!
This is a test line.
Short line.
Another longer line with more content here.

Final line at the end."""
    
    input_path = os.path.join(data_dir, 'input.txt')
    with open(input_path, 'w') as f:
        f.write(test_data)
    return test_data

def test_char_diffusion_provider(data_dir: str) -> Dict[str, Any]:
    """Test CharDiffusionProvider with line-aligned sequences."""
    from data.char_diffusion.prepare_streaming import CharDiffusionProvider
    
    config = {
        'enable_line_aligned_sequences': True,
        'mask_probability': 0.15
    }
    
    provider = CharDiffusionProvider(
        data_dir=data_dir,
        batch_size=4,
        block_size=50,
        config=config,
        verbose=True
    )
    
    print(f"\n=== CharDiffusionProvider Results ===")
    print(f"Line-aligned enabled: {provider.enable_line_aligned_sequences}")
    print(f"Vocab size: {provider.vocab_size}")
    print(f"Mask token ID: {provider.mask_token_id}")
    print(f"Pad token ID: {provider.pad_token_id}")
    print(f"Newline token ID: {provider.newline_token_id}")
    
    if provider.train_builder:
        print(f"Train lines: {len(provider.train_builder.lines_ids)}")
        print(f"Val lines: {len(provider.val_builder.lines_ids)}")
        print(f"Train valid starts: {provider.train_builder.valid_starts.numel()}")
        print(f"Val valid starts: {provider.val_builder.valid_starts.numel()}")
    
    # Test sequence generation
    rng = torch.Generator()
    rng.manual_seed(42)
    
    x, content_lengths = provider._build_line_aligned_variable_length("train", 4, rng)
    print(f"Generated sequences shape: {x.shape}")
    print(f"Content lengths: {content_lengths.tolist()}")
    
    # Test batch sampling
    rng.manual_seed(42)
    batch = provider._sample_default_batch("train", rng)
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch x shape: {batch['x'].shape}")
    print(f"Batch y shape: {batch['y'].shape}")
    
    return {
        'provider_type': 'CharDiffusion',
        'line_aligned': provider.enable_line_aligned_sequences,
        'vocab_size': provider.vocab_size,
        'train_lines': len(provider.train_builder.lines_ids) if provider.train_builder else 0,
        'val_lines': len(provider.val_builder.lines_ids) if provider.val_builder else 0,
        'content_lengths': content_lengths.tolist(),
        'batch_shape': batch['x'].shape
    }

def test_sequence_scorer_provider(data_dir: str) -> Dict[str, Any]:
    """Test SequenceScorerProvider with line-aligned sequences."""
    from data.sequence_scorer.prepare_streaming import SequenceScorerProvider
    
    # Create a mock MLM checkpoint for testing
    class MockMLMEngine:
        def __init__(self, *args, **kwargs):
            # Read the same data to create compatible vocabulary
            input_path = os.path.join(data_dir, 'input.txt')
            with open(input_path, 'r') as f:
                data = f.read()
            
            chars = sorted(list(set(data)))
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
    
    # Mock the MLM engine
    import data.sequence_scorer.prepare_streaming
    original_init = data.sequence_scorer.prepare_streaming.MLMInferenceEngine
    data.sequence_scorer.prepare_streaming.MLMInferenceEngine = MockMLMEngine
    
    try:
        config = {
            'enable_line_aligned_sequences': True,
            'mlm_checkpoint_path': 'dummy_path',
            'cls_token_id': 100,  # Will be adjusted based on vocab
            'mask_probability_range': (0.1, 0.5)
        }
        
        provider = SequenceScorerProvider(
            data_dir=data_dir,
            batch_size=4,
            block_size=50,
            config=config,
            verbose=True
        )
        
        print(f"\n=== SequenceScorerProvider Results ===")
        print(f"Line-aligned enabled: {provider.enable_line_aligned_sequences}")
        print(f"Vocab size: {provider.vocab_size}")
        print(f"Mask token ID: {provider.mask_token_id}")
        print(f"CLS token ID: {provider.cls_token_id}")
        print(f"Newline token ID: {provider.newline_token_id}")
        
        if provider.train_builder:
            print(f"Train lines: {len(provider.train_builder.lines_ids)}")
            print(f"Val lines: {len(provider.val_builder.lines_ids)}")
            print(f"Train valid starts: {provider.train_builder.valid_starts.numel()}")
            print(f"Val valid starts: {provider.val_builder.valid_starts.numel()}")
        
        # Test sequence generation
        rng = torch.Generator()
        rng.manual_seed(42)
        
        x, content_lengths = provider.train_builder.build_variable_length_sequences(4, 49, rng)  # -1 for CLS
        print(f"Generated sequences shape: {x.shape}")
        print(f"Content lengths: {content_lengths.tolist()}")
        
        # Test batch sampling
        rng.manual_seed(42)
        batch = provider._sample_line_aligned_batch("train", rng)
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch targets shape: {batch['targets'].shape}")
        
        return {
            'provider_type': 'SequenceScorer',
            'line_aligned': provider.enable_line_aligned_sequences,
            'vocab_size': provider.vocab_size,
            'train_lines': len(provider.train_builder.lines_ids) if provider.train_builder else 0,
            'val_lines': len(provider.val_builder.lines_ids) if provider.val_builder else 0,
            'content_lengths': content_lengths.tolist(),
            'batch_shape': batch['input_ids'].shape
        }
    
    finally:
        # Restore original MLM engine
        data.sequence_scorer.prepare_streaming.MLMInferenceEngine = original_init

def main():
    """Main validation function."""
    print("=== Line-Aligned Sequence Generation Validation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test data
        test_data = create_test_data(temp_dir)
        print(f"Created test data with {len(test_data)} characters")
        print(f"Test data lines: {len(test_data.splitlines())}")
        
        # Test both providers
        char_results = test_char_diffusion_provider(temp_dir)
        seq_results = test_sequence_scorer_provider(temp_dir)
        
        # Compare results
        print(f"\n=== Comparison ===")
        print(f"Both use line-aligned: {char_results['line_aligned'] and seq_results['line_aligned']}")
        print(f"CharDiffusion train lines: {char_results['train_lines']}")
        print(f"SequenceScorer train lines: {seq_results['train_lines']}")
        print(f"Line count difference: {abs(char_results['train_lines'] - seq_results['train_lines'])}")
        
        print(f"CharDiffusion content lengths: {char_results['content_lengths']}")
        print(f"SequenceScorer content lengths: {seq_results['content_lengths']}")
        
        print(f"CharDiffusion batch shape: {char_results['batch_shape']}")
        print(f"SequenceScorer batch shape: {seq_results['batch_shape']}")
        
        # Validation checks
        success = True
        
        if not (char_results['line_aligned'] and seq_results['line_aligned']):
            print("❌ ERROR: Both providers should have line-aligned sequences enabled")
            success = False
        
        if abs(char_results['train_lines'] - seq_results['train_lines']) > 1:
            print("❌ ERROR: Providers have significantly different line counts")
            success = False
        
        if char_results['batch_shape'] != seq_results['batch_shape']:
            print("❌ ERROR: Providers generate different batch shapes")
            success = False
        
        # Check that content lengths are reasonable
        char_lengths = char_results['content_lengths']
        seq_lengths = seq_results['content_lengths']
        
        if not all(length > 0 for length in char_lengths):
            print("❌ ERROR: CharDiffusion generated sequences with zero content")
            success = False
        
        if not all(length > 0 for length in seq_lengths):
            print("❌ ERROR: SequenceScorer generated sequences with zero content")
            success = False
        
        if success:
            print("✅ SUCCESS: Both providers generate line-aligned sequences consistently!")
        else:
            print("❌ FAILURE: Providers are not consistent")
        
        return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
