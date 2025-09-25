#!/usr/bin/env python3
"""
Test script for line replacement masking functionality.
"""
import os
import sys
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.char_diffusion.prepare_streaming import CharDiffusionProvider


def test_line_masking():
    """Test the line replacement masking functionality."""
    print("Testing line replacement masking...")
    
    # Create a minimal config for testing
    config = {
        'mask_probability': 0.15,
        'use_all_stages_for_training': True,
        'unmasking_stages': [
            {'type': 'line', 'min_ratio': 0.2, 'max_ratio': 0.4, 'val_loss_stale_count': 10}
        ],
        'validation_stages': [
            {'type': 'line', 'min_ratio': 0.1, 'max_ratio': 0.3, 'val_loss_stale_count': 5}
        ],
        'enable_line_aligned_sequences': True
    }
    
    # Initialize provider
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'char_diffusion')
    
    try:
        provider = CharDiffusionProvider(
            data_dir=data_dir,
            batch_size=4,
            block_size=128,
            batches_per_file=2,
            max_backlog_files=1,
            sleep_seconds=1.0,
            seed=1337,
            verbose=True,
            config=config
        )
        
        print(f"Provider initialized successfully!")
        print(f"Vocab size: {provider.vocab_size}")
        print(f"Mask token ID: {provider.mask_token_id}")
        print(f"Newline token ID: {provider.newline_token_id}")
        print(f"Number of train lines: {len(provider.train_lines_ids)}")
        print(f"Number of val lines: {len(provider.val_lines_ids)}")
        
        # Test stage-based file generation
        print("\nTesting stage-based file generation...")
        provider._produce_stage_based_file('train', 0)
        print("Stage-based file generation completed!")
        
        # Test individual line masking function
        print("\nTesting line replacement masking function...")
        from data.char_diffusion.masking_utils import apply_line_replacement_masking_cpu
        
        # Create a simple test case
        rng = torch.Generator()
        rng.manual_seed(42)
        
        # Create test input with multiple lines
        test_lines = [
            "Hello world\n",
            "This is a test\n", 
            "Another line here\n"
        ]
        
        # Convert to token IDs
        test_tokens = []
        for line in test_lines:
            line_tokens = [provider.stoi[c] for c in line]
            test_tokens.extend(line_tokens)
        
        # Create batch
        batch_size = 2
        seq_len = 64
        x = torch.zeros((batch_size, seq_len), dtype=torch.long)
        content_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill first sample
        content_len = min(len(test_tokens), seq_len)
        x[0, :content_len] = torch.tensor(test_tokens[:content_len], dtype=torch.long)
        content_lengths[0] = content_len
        
        # Fill second sample (same content for simplicity)
        x[1, :content_len] = torch.tensor(test_tokens[:content_len], dtype=torch.long)
        content_lengths[1] = content_len
        
        print(f"Original content length: {content_len}")
        print(f"Original first few tokens: {x[0, :20].tolist()}")
        
        # Apply line replacement masking
        replaced_x, mask = apply_line_replacement_masking_cpu(
            x, content_lengths, 
            min_ratio=0.3, max_ratio=0.6,
            replacement_lines_ids=provider.train_lines_ids,
            newline_token_id=provider.newline_token_id,
            mask_token_id=provider.mask_token_id,
            vocab_size=provider.base_vocab_size,
            rng=rng
        )
        
        print(f"Replaced first few tokens: {replaced_x[0, :20].tolist()}")
        print(f"Mask first few positions: {mask[0, :20].tolist()}")
        print(f"Number of replaced positions in sample 0: {mask[0].sum().item()}")
        print(f"Number of replaced positions in sample 1: {mask[1].sum().item()}")
        
        # Convert back to text for visualization
        def tokens_to_text(tokens, length):
            return ''.join([provider.itos.get(t.item(), f'[{t.item()}]') for t in tokens[:length]])
        
        print(f"\nOriginal sample 0: {repr(tokens_to_text(x[0], content_lengths[0]))}")
        print(f"Replaced sample 0: {repr(tokens_to_text(replaced_x[0], content_lengths[0]))}")
        
        print("\nLine replacement masking test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_line_masking()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
