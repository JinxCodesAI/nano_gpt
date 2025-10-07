#!/usr/bin/env python
"""
Test script to verify dual-mode model functionality.
Tests that the model can switch between LANGUAGE_MODEL and SEQUENCE_SCORER modes at runtime.
"""

import sys
import torch
from model import GPT, GPTConfig, ModelMode

def test_dual_mode():
    """Test that model can switch modes and use both heads"""
    print("Testing dual-mode model functionality...")
    
    # Create a small model for testing
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        dropout=0.0,
        bias=False,
        cls_token_id=0,  # Enable CLS token for sequence scorer
    )
    
    model = GPT(config)
    model.eval()
    
    # Test 1: Default mode should be LANGUAGE_MODEL
    print("\n1. Testing default mode...")
    assert model.get_mode() == ModelMode.LANGUAGE_MODEL, "Default mode should be LANGUAGE_MODEL"
    print("   ✓ Default mode is LANGUAGE_MODEL")
    
    # Test 2: Model should have both heads
    print("\n2. Testing dual heads...")
    assert hasattr(model, 'lm_head'), "Model should have lm_head"
    assert hasattr(model, 'sequence_head'), "Model should have sequence_head"
    print("   ✓ Model has both lm_head and sequence_head")
    
    # Test 3: Forward pass in LANGUAGE_MODEL mode
    print("\n3. Testing LANGUAGE_MODEL mode forward pass...")
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(x, y)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape ({batch_size}, {seq_len}, {config.vocab_size}), got {logits.shape}"
    assert loss is not None, "Loss should not be None when targets provided"
    print(f"   ✓ LANGUAGE_MODEL output shape: {logits.shape}")
    print(f"   ✓ Loss computed: {loss.item():.4f}")
    
    # Test 4: Switch to SEQUENCE_SCORER mode
    print("\n4. Testing mode switching to SEQUENCE_SCORER...")
    model.set_mode(ModelMode.SEQUENCE_SCORER)
    assert model.get_mode() == ModelMode.SEQUENCE_SCORER, "Mode should be SEQUENCE_SCORER"
    print("   ✓ Successfully switched to SEQUENCE_SCORER mode")
    
    # Test 5: Forward pass in SEQUENCE_SCORER mode
    print("\n5. Testing SEQUENCE_SCORER mode forward pass...")
    # For sequence scorer, input should have CLS token at position 0
    x_seq = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    x_seq[:, 0] = config.cls_token_id  # Set CLS token
    y_seq = torch.rand(batch_size)  # Targets are scalars in [0, 1]
    
    with torch.no_grad():
        scores, loss_seq = model(x_seq, y_seq)
    
    assert scores.shape == (batch_size,), \
        f"Expected shape ({batch_size},), got {scores.shape}"
    assert loss_seq is not None, "Loss should not be None when targets provided"
    print(f"   ✓ SEQUENCE_SCORER output shape: {scores.shape}")
    print(f"   ✓ Loss computed: {loss_seq.item():.4f}")
    print(f"   ✓ Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    # Test 6: Switch back to LANGUAGE_MODEL
    print("\n6. Testing mode switching back to LANGUAGE_MODEL...")
    model.set_mode(ModelMode.LANGUAGE_MODEL)
    assert model.get_mode() == ModelMode.LANGUAGE_MODEL, "Mode should be LANGUAGE_MODEL"
    
    with torch.no_grad():
        logits2, loss2 = model(x, y)
    
    assert logits2.shape == (batch_size, seq_len, config.vocab_size), \
        "Should work after switching back"
    print("   ✓ Successfully switched back to LANGUAGE_MODEL mode")
    print(f"   ✓ Output shape: {logits2.shape}")
    
    # Test 7: Test inference mode (no targets)
    print("\n7. Testing inference mode (no targets)...")
    model.set_mode(ModelMode.LANGUAGE_MODEL)
    with torch.no_grad():
        logits_inf, loss_inf = model(x[:, :10])  # Shorter sequence
    assert loss_inf is None, "Loss should be None in inference mode"
    print("   ✓ LANGUAGE_MODEL inference works (loss=None)")
    
    model.set_mode(ModelMode.SEQUENCE_SCORER)
    with torch.no_grad():
        scores_inf, loss_inf2 = model(x_seq)
    assert loss_inf2 is None, "Loss should be None in inference mode"
    print("   ✓ SEQUENCE_SCORER inference works (loss=None)")
    
    # Test 8: Test invalid mode
    print("\n8. Testing invalid mode handling...")
    try:
        model.set_mode("invalid_mode")
        print("   ✗ Should have raised TypeError")
        return False
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError: {e}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = test_dual_mode()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

