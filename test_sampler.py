"""
Test script for sampler head implementation
"""
import torch
from model import GPTConfig, GPT, ModelMode

def test_sampler_config():
    """Test sampler configuration validation"""
    print("Testing sampler configuration...")
    
    # Test 1: Sampler requires LANGUAGE_MODEL mode
    try:
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=100,
            mode=ModelMode.TOKEN_CLASSIFIER,
            add_sampler_head=True,
            mask_token_id=99
        )
        print("  FAIL: Should have raised ValueError for non-LANGUAGE_MODEL mode")
        return False
    except ValueError as e:
        if "LANGUAGE_MODEL mode" in str(e):
            print("  PASS: Correctly rejects non-LANGUAGE_MODEL mode")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False
    
    # Test 2: Sampler requires bidirectional attention
    try:
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=100,
            mode=ModelMode.LANGUAGE_MODEL,
            attention_type='causal',
            add_sampler_head=True,
            mask_token_id=99
        )
        print("  FAIL: Should have raised ValueError for causal attention")
        return False
    except ValueError as e:
        if "bidirectional attention" in str(e):
            print("  PASS: Correctly rejects causal attention")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False
    
    # Test 3: Sampler requires mask_token_id
    try:
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            block_size=128,
            vocab_size=100,
            mode=ModelMode.LANGUAGE_MODEL,
            attention_type='bidirectional',
            add_sampler_head=True,
            mask_token_id=None
        )
        print("  FAIL: Should have raised ValueError for missing mask_token_id")
        return False
    except ValueError as e:
        if "mask_token_id" in str(e):
            print("  PASS: Correctly rejects missing mask_token_id")
        else:
            print(f"  FAIL: Wrong error message: {e}")
            return False
    
    # Test 4: Valid sampler configuration
    try:
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
            sampler_min_neighbors_ratio=0.01
        )
        print("  PASS: Valid sampler configuration accepted")
    except Exception as e:
        print(f"  FAIL: Valid config rejected: {e}")
        return False
    
    return True


def test_sampler_model_creation():
    """Test model creation with sampler head"""
    print("\nTesting model creation with sampler...")
    
    # Test without sampler
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='bidirectional',
        add_sampler_head=False,
        mask_token_id=99
    )
    model = GPT(config)
    
    if hasattr(model, 'sampler_head') and model.sampler_head is not None:
        print("  FAIL: Model should not have sampler_head when disabled")
        return False
    print("  PASS: Model without sampler created correctly")
    
    # Test with sampler
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
    
    if not hasattr(model, 'sampler_head') or model.sampler_head is None:
        print("  FAIL: Model should have sampler_head when enabled")
        return False
    print("  PASS: Model with sampler created correctly")
    
    # Check sampler head structure
    if not hasattr(model.sampler_head, 'mlp'):
        print("  FAIL: Sampler head missing mlp")
        return False
    print("  PASS: Sampler head has correct structure")
    
    return True


def test_sampler_forward():
    """Test forward pass with sampler"""
    print("\nTesting forward pass with sampler...")
    
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
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, 98, (batch_size, seq_len))  # Avoid mask token
    targets = torch.randint(0, 98, (batch_size, seq_len))
    
    # Set current iteration to enable sampler
    model._current_iter = 0
    
    # Forward pass
    try:
        with torch.no_grad():
            logits, loss = model(idx, targets)
        
        if logits is None:
            print("  FAIL: Forward pass returned None logits")
            return False
        
        if loss is None:
            print("  FAIL: Forward pass returned None loss")
            return False
        
        print(f"  PASS: Forward pass successful (loss={loss.item():.4f})")
        
        # Check loss components
        if not hasattr(model, '_last_lm_loss'):
            print("  FAIL: Model missing _last_lm_loss")
            return False
        
        if not hasattr(model, '_last_sampler_loss'):
            print("  FAIL: Model missing _last_sampler_loss")
            return False
        
        print(f"  PASS: Loss components tracked (lm={model._last_lm_loss:.4f}, sampler={model._last_sampler_loss:.4f})")
        
    except Exception as e:
        print(f"  FAIL: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_backward_compatibility():
    """Test that models without sampler still work"""
    print("\nTesting backward compatibility...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        vocab_size=100,
        mode=ModelMode.LANGUAGE_MODEL,
        attention_type='causal',  # Old-style causal attention
        add_sampler_head=False
    )
    model = GPT(config)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))
    
    # Forward pass
    try:
        with torch.no_grad():
            logits, loss = model(idx, targets)
        
        if logits is None or loss is None:
            print("  FAIL: Forward pass failed for backward compatible model")
            return False
        
        print(f"  PASS: Backward compatible model works (loss={loss.item():.4f})")
        
    except Exception as e:
        print(f"  FAIL: Backward compatible model failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("="*60)
    print("SAMPLER HEAD IMPLEMENTATION TESTS")
    print("="*60)
    
    all_passed = True
    
    all_passed &= test_sampler_config()
    all_passed &= test_sampler_model_creation()
    all_passed &= test_sampler_forward()
    all_passed &= test_backward_compatibility()
    
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

