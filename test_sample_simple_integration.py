"""
Test that sample_simple.py integration with sampler head works correctly
"""
import torch
from model import GPTConfig, GPT, ModelMode
from sample_utils import predict_and_sample_tokens

def test_predict_and_sample_without_sampler():
    """Test predict_and_sample_tokens without sampler head (backward compatibility)"""
    print("Testing predict_and_sample_tokens WITHOUT sampler head...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=False,  # No sampler
        mask_token_id=99
    )
    model = GPT(config)
    model.eval()
    
    # Create input with some masks
    batch_size = 2
    seq_len = 32
    mask_token_id = 99
    tokens = torch.randint(0, 98, (batch_size, seq_len))
    
    # Mask 50% of tokens
    mask_prob = torch.rand(batch_size, seq_len) < 0.5
    tokens[mask_prob] = mask_token_id
    
    num_masked = (tokens == mask_token_id).sum().item()
    print(f"  Input: {num_masked} masked tokens")
    
    # Run prediction
    with torch.no_grad():
        result, _ = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            device='cpu'
        )
    
    # Check that all masks are filled
    remaining_masks = (result == mask_token_id).sum().item()
    if remaining_masks > 0:
        print(f"  FAIL: {remaining_masks} masks remaining")
        return False
    
    print(f"  PASS: All {num_masked} masks filled (naive sampling)")
    return True


def test_predict_and_sample_with_sampler():
    """Test predict_and_sample_tokens with sampler head"""
    print("\nTesting predict_and_sample_tokens WITH sampler head...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,  # With sampler
        mask_token_id=99,
        start_sampler_iteration=0
    )
    model = GPT(config)
    model.eval()
    
    # Create input with some masks
    batch_size = 2
    seq_len = 32
    mask_token_id = 99
    tokens = torch.randint(0, 98, (batch_size, seq_len))
    
    # Mask 50% of tokens
    mask_prob = torch.rand(batch_size, seq_len) < 0.5
    tokens[mask_prob] = mask_token_id
    
    num_masked = (tokens == mask_token_id).sum().item()
    print(f"  Input: {num_masked} masked tokens")
    
    # Run prediction
    with torch.no_grad():
        result, _ = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            device='cpu'
        )
    
    # Check that all masks are filled
    remaining_masks = (result == mask_token_id).sum().item()
    if remaining_masks > 0:
        print(f"  FAIL: {remaining_masks} masks remaining")
        return False
    
    print(f"  PASS: All {num_masked} masks filled (sampler wavefront)")
    return True


def test_return_logits_flag():
    """Test that return_logits flag works correctly"""
    print("\nTesting return_logits flag...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=True,
        mask_token_id=99
    )
    model = GPT(config)
    model.eval()
    
    # Create input with masks
    batch_size = 2
    seq_len = 16
    mask_token_id = 99
    tokens = torch.full((batch_size, seq_len), mask_token_id)
    
    # Test with return_logits=True
    with torch.no_grad():
        result, logits = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            device='cpu',
            return_logits=True
        )
    
    if logits is None:
        print("  FAIL: logits should not be None when return_logits=True")
        return False
    
    if logits.shape != (batch_size, seq_len, config.vocab_size):
        print(f"  FAIL: logits shape {logits.shape} != expected {(batch_size, seq_len, config.vocab_size)}")
        return False
    
    print(f"  PASS: return_logits=True returns logits with correct shape")
    
    # Test with return_logits=False (default)
    with torch.no_grad():
        result1, result2 = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=1.0,
            top_p=1.0,
            vocab_size=config.vocab_size,
            device='cpu',
            return_logits=False
        )
    
    # Both should be token tensors
    if result1.shape != (batch_size, seq_len):
        print(f"  FAIL: result shape incorrect")
        return False
    
    print(f"  PASS: return_logits=False returns tokens")
    return True


def main():
    print("="*60)
    print("SAMPLE_SIMPLE.PY INTEGRATION TESTS")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_predict_and_sample_without_sampler()
    all_passed &= test_predict_and_sample_with_sampler()
    all_passed &= test_return_logits_flag()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nsample_simple.py will automatically use sampler head when available")
    else:
        print("SOME TESTS FAILED ✗")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

