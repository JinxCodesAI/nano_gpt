"""
Test script for sampler inference (wavefront filling)
"""
import torch
from model import GPTConfig, GPT, ModelMode
from sample_utils import sampler_wavefront_fill, _bootstrap_fill_by_confidence

def test_bootstrap_fill():
    """Test bootstrap filling when no neighbors available"""
    print("Testing bootstrap fill...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.eval()
    
    # Create fully masked sequence
    batch_size = 2
    seq_len = 32
    mask_token_id = 99
    filled = torch.full((batch_size, seq_len), mask_token_id)
    is_masked = (filled == mask_token_id)
    
    # Get hidden states
    with torch.no_grad():
        hidden_states = model._encode_tokens(filled)
    
    # Test bootstrap
    eligible = _bootstrap_fill_by_confidence(
        filled, is_masked, hidden_states, model,
        min_ratio=0.1,  # 10% for testing
        mask_token_id=mask_token_id
    )
    
    num_eligible = eligible.sum().item()
    expected = max(1, int(is_masked.sum().item() * 0.1))
    
    if num_eligible != expected:
        print(f"  FAIL: Expected {expected} eligible tokens, got {num_eligible}")
        return False
    
    print(f"  PASS: Bootstrap selected {num_eligible} tokens (10% of {is_masked.sum().item()})")
    return True


def test_wavefront_fill_simple():
    """Test wavefront filling with simple case"""
    print("\nTesting wavefront fill (simple case)...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.eval()
    
    # Create sequence with some masks
    batch_size = 2
    seq_len = 16
    mask_token_id = 99
    
    # Pattern: [token, mask, token, mask, ...]
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i in range(seq_len):
        if i % 2 == 0:
            tokens[:, i] = torch.randint(0, 98, (batch_size,))
        else:
            tokens[:, i] = mask_token_id
    
    print(f"  Input pattern: {tokens[0, :8].tolist()}")
    
    # Get hidden states
    with torch.no_grad():
        hidden_states = model._encode_tokens(tokens)
        
        # Fill masks
        filled = sampler_wavefront_fill(
            model=model,
            tokens=tokens,
            hidden_states=hidden_states,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            base_vocab_size=None,
            min_neighbors_ratio=0.01
        )
    
    # Check that all masks are filled
    remaining_masks = (filled == mask_token_id).sum().item()
    if remaining_masks > 0:
        print(f"  FAIL: {remaining_masks} masks remaining")
        return False
    
    print(f"  Output pattern: {filled[0, :8].tolist()}")
    print(f"  PASS: All masks filled")
    return True


def test_wavefront_fill_fully_masked():
    """Test wavefront filling with fully masked sequence"""
    print("\nTesting wavefront fill (fully masked)...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0,
        sampler_min_neighbors_ratio=0.05  # 5% bootstrap
    )
    model = GPT(config)
    model.eval()
    
    # Create fully masked sequence
    batch_size = 2
    seq_len = 32
    mask_token_id = 99
    tokens = torch.full((batch_size, seq_len), mask_token_id)
    
    print(f"  Input: all {mask_token_id} (mask token)")
    
    # Get hidden states
    with torch.no_grad():
        hidden_states = model._encode_tokens(tokens)
        
        # Fill masks
        filled = sampler_wavefront_fill(
            model=model,
            tokens=tokens,
            hidden_states=hidden_states,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            base_vocab_size=None,
            min_neighbors_ratio=config.sampler_min_neighbors_ratio
        )
    
    # Check that all masks are filled
    remaining_masks = (filled == mask_token_id).sum().item()
    if remaining_masks > 0:
        print(f"  FAIL: {remaining_masks} masks remaining")
        return False
    
    # Check that filled tokens are valid (not mask token)
    if (filled == mask_token_id).any():
        print(f"  FAIL: Some tokens still masked")
        return False
    
    print(f"  Output sample: {filled[0, :8].tolist()}")
    print(f"  PASS: All masks filled via bootstrap + wavefront")
    return True


def test_wavefront_fill_partial():
    """Test wavefront filling with partial masking"""
    print("\nTesting wavefront fill (partial masking)...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.eval()
    
    # Create sequence with 50% masking
    batch_size = 2
    seq_len = 32
    mask_token_id = 99
    tokens = torch.randint(0, 98, (batch_size, seq_len))
    
    # Randomly mask 50% of tokens
    mask_prob = torch.rand(batch_size, seq_len) < 0.5
    tokens[mask_prob] = mask_token_id
    
    num_masked = (tokens == mask_token_id).sum().item()
    print(f"  Input: {num_masked}/{batch_size*seq_len} tokens masked ({100*num_masked/(batch_size*seq_len):.1f}%)")
    
    # Get hidden states
    with torch.no_grad():
        hidden_states = model._encode_tokens(tokens)
        
        # Fill masks
        filled = sampler_wavefront_fill(
            model=model,
            tokens=tokens,
            hidden_states=hidden_states,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            base_vocab_size=None,
            min_neighbors_ratio=0.01
        )
    
    # Check that all masks are filled
    remaining_masks = (filled == mask_token_id).sum().item()
    if remaining_masks > 0:
        print(f"  FAIL: {remaining_masks} masks remaining")
        return False
    
    # Check that non-masked tokens are unchanged
    non_masked = tokens != mask_token_id
    if not torch.all(filled[non_masked] == tokens[non_masked]):
        print(f"  FAIL: Non-masked tokens were changed")
        return False
    
    print(f"  PASS: All {num_masked} masks filled, non-masked tokens preserved")
    return True


def main():
    print("="*60)
    print("SAMPLER INFERENCE TESTS (WAVEFRONT FILLING)")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_bootstrap_fill()
    all_passed &= test_wavefront_fill_simple()
    all_passed &= test_wavefront_fill_fully_masked()
    all_passed &= test_wavefront_fill_partial()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

