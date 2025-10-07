#!/usr/bin/env python
"""
Test backward compatibility with existing checkpoints and configs.
Verifies that old checkpoints can still be loaded and used.
"""

import sys
import torch
from model import GPT, GPTConfig, ModelMode

def test_config_without_mode():
    """Test that GPTConfig works without mode field"""
    print("\n1. Testing GPTConfig without mode field...")
    
    # Old-style config (without mode field)
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        dropout=0.0,
        bias=False,
    )
    
    model = GPT(config)
    
    # Should default to LANGUAGE_MODEL
    assert model.get_mode() == ModelMode.LANGUAGE_MODEL
    print("   ✓ Config without mode field works, defaults to LANGUAGE_MODEL")
    
    return True

def test_config_with_cls_token():
    """Test that GPTConfig works with cls_token_id for sequence scorer"""
    print("\n2. Testing GPTConfig with cls_token_id...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        dropout=0.0,
        bias=False,
        cls_token_id=0,
    )
    
    model = GPT(config)
    
    # Should have CLS embedding
    assert hasattr(model, 'cls_embedding')
    print("   ✓ Config with cls_token_id creates CLS embedding")
    
    # Can switch to SEQUENCE_SCORER mode
    model.set_mode(ModelMode.SEQUENCE_SCORER)
    assert model.get_mode() == ModelMode.SEQUENCE_SCORER
    print("   ✓ Can switch to SEQUENCE_SCORER mode")
    
    return True

def test_checkpoint_loading_simulation():
    """Simulate loading a checkpoint with old model_args"""
    print("\n3. Testing checkpoint loading with old model_args...")
    
    # Simulate old checkpoint model_args that might have 'mode' field
    old_model_args = {
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'vocab_size': 256,
        'block_size': 64,
        'dropout': 0.0,
        'bias': False,
        # Old checkpoints might have these fields
        'mode': ModelMode.LANGUAGE_MODEL,  # This will be ignored
        'num_token_classes': 2,  # This will be ignored
    }
    
    # Filter out fields that are no longer in GPTConfig
    valid_fields = {
        'n_layer', 'n_head', 'n_embd', 'vocab_size', 'block_size',
        'dropout', 'bias', 'ignore_index', 'attention_type', 'position_encoding',
        'cls_token_id', 'freeze_transformer', 'init_from_checkpoint',
        'unfreeze_at_iteration', 'unfreeze_lr_multiplier',
        'add_critic_head', 'critic_alpha', 'critic_target_scope',
        'mask_token_id', 'pad_token_id',
        'start_critic_iteration', 'end_critic_iteration'
    }
    
    filtered_args = {k: v for k, v in old_model_args.items() if k in valid_fields}
    
    # Create model with filtered args
    config = GPTConfig(**filtered_args)
    model = GPT(config)
    
    # Model should work fine
    assert model.get_mode() == ModelMode.LANGUAGE_MODEL
    print("   ✓ Old checkpoint args can be filtered and loaded")
    
    return True

def test_both_heads_exist():
    """Test that both heads are created regardless of mode"""
    print("\n4. Testing that both heads exist...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        cls_token_id=0,
    )
    
    model = GPT(config)
    
    # Both heads should exist
    assert hasattr(model, 'lm_head'), "lm_head should exist"
    assert hasattr(model, 'sequence_head'), "sequence_head should exist"
    
    # lm_head should have vocab_size output
    assert model.lm_head.out_features == 256
    
    # sequence_head should exist
    assert hasattr(model.sequence_head, 'base_predictor')
    
    print("   ✓ Both lm_head and sequence_head exist")
    print(f"   ✓ lm_head output: {model.lm_head.out_features}")
    print("   ✓ sequence_head exists")
    
    return True

def test_forward_pass_both_modes():
    """Test forward pass in both modes"""
    print("\n5. Testing forward pass in both modes...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        cls_token_id=0,
    )
    
    model = GPT(config)
    model.eval()
    
    batch_size = 2
    seq_len = 32
    
    # Test LANGUAGE_MODEL mode
    model.set_mode(ModelMode.LANGUAGE_MODEL)
    x = torch.randint(0, 256, (batch_size, seq_len))
    y = torch.randint(0, 256, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, loss = model(x, y)
    
    assert logits.shape == (batch_size, seq_len, 256)
    assert loss is not None
    print("   ✓ LANGUAGE_MODEL mode forward pass works")
    
    # Test SEQUENCE_SCORER mode
    model.set_mode(ModelMode.SEQUENCE_SCORER)
    x_seq = torch.randint(0, 256, (batch_size, seq_len))
    x_seq[:, 0] = 0  # CLS token
    y_seq = torch.rand(batch_size)
    
    with torch.no_grad():
        scores, loss_seq = model(x_seq, y_seq)
    
    assert scores.shape == (batch_size,)
    assert loss_seq is not None
    print("   ✓ SEQUENCE_SCORER mode forward pass works")
    
    return True

def test_critic_head_compatibility():
    """Test that critic head still works"""
    print("\n6. Testing critic head compatibility...")
    
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        vocab_size=256,
        block_size=64,
        add_critic_head=True,
        critic_alpha=0.5,
    )
    
    model = GPT(config)
    model.eval()
    
    # Critic head should exist
    assert hasattr(model, 'critic_head')
    print("   ✓ Critic head created when add_critic_head=True")
    
    # Test critic_scores method
    x = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        scores = model.critic_scores(x)
    
    assert scores.shape == (2, 32)
    print("   ✓ critic_scores() method works")
    
    return True

def main():
    print("="*60)
    print("BACKWARD COMPATIBILITY TESTS")
    print("="*60)
    
    tests = [
        test_config_without_mode,
        test_config_with_cls_token,
        test_checkpoint_loading_simulation,
        test_both_heads_exist,
        test_forward_pass_both_modes,
        test_critic_head_compatibility,
    ]
    
    failed = []
    for test in tests:
        try:
            if not test():
                failed.append(test.__name__)
        except Exception as e:
            print(f"   ✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    if not failed:
        print("All backward compatibility tests passed! ✓")
        print("="*60)
        return True
    else:
        print(f"Failed tests: {', '.join(failed)}")
        print("="*60)
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

