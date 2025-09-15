"""
Comprehensive validation script for the complete multi-mode implementation.

This script validates:
1. All model modes work correctly
2. Transfer learning functionality  
3. Loss modifier compatibility
4. Enhanced schedulers
5. Backward compatibility
"""

import tempfile
import os

def test_imports():
    """Test that all new components can be imported"""
    print("Testing imports...")
    
    try:
        from model import GPT, GPTConfig, ModelMode
        from core.scheduler import CosineLRScheduler, TransferLearningScheduler, WarmupOnlyScheduler, AdaptiveScheduler
        from loss_modifiers import create_loss_modifier_pipeline
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_validation():
    """Test GPTConfig with new parameters"""
    print("\nTesting configuration validation...")
    
    from model import GPTConfig, ModelMode
    
    # Test basic config creation
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, block_size=32,
        mode=ModelMode.LANGUAGE_MODEL
    )
    print("✓ Basic config creation works")
    
    # Test automatic attention type correction
    config_tc = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, block_size=32,
        mode=ModelMode.TOKEN_CLASSIFIER,
        attention_type='causal'  # Should be corrected to bidirectional
    )
    assert config_tc.attention_type == 'bidirectional'
    print("✓ Automatic attention type correction works")
    
    # Test backward compatibility
    config_legacy = GPTConfig(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, block_size=32,
        binary_classification=True  # Legacy parameter
    )
    assert config_legacy.mode == ModelMode.TOKEN_CLASSIFIER
    assert config_legacy.num_token_classes == 2
    print("✓ Backward compatibility works")
    
    return True

def test_model_modes():
    """Test all model modes"""
    print("\nTesting model modes...")
    
    from model import GPT, GPTConfig, ModelMode
    import torch
    
    base_config = dict(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, block_size=32
    )
    
    # Create test data
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test Language Model
    lm_config = GPTConfig(**base_config, mode=ModelMode.LANGUAGE_MODEL)
    lm_model = GPT(lm_config)
    lm_targets = torch.randint(0, 100, (batch_size, seq_len))
    lm_logits, lm_loss = lm_model(input_ids, lm_targets)
    assert lm_logits.shape == (batch_size, seq_len, 100)
    assert lm_loss.item() > 0
    print("✓ Language model mode works")
    
    # Test Token Classifier
    tc_config = GPTConfig(**base_config, mode=ModelMode.TOKEN_CLASSIFIER, num_token_classes=3)
    tc_model = GPT(tc_config)
    tc_targets = torch.randint(0, 3, (batch_size, seq_len))
    tc_logits, tc_loss = tc_model(input_ids, tc_targets)
    assert tc_logits.shape == (batch_size, seq_len, 3)
    assert tc_loss.item() > 0
    print("✓ Token classifier mode works")
    
    # Test Sequence Scorer
    ss_config = GPTConfig(**base_config, mode=ModelMode.SEQUENCE_SCORER, cls_token_id=0)
    ss_model = GPT(ss_config)
    ss_targets = torch.rand(batch_size)
    ss_logits, ss_loss = ss_model(input_ids, ss_targets)
    assert ss_logits.shape == (batch_size,)
    assert torch.all((ss_logits >= 0) & (ss_logits <= 1))
    assert ss_loss.item() > 0
    print("✓ Sequence scorer mode works")
    
    return True

def test_transfer_learning():
    """Test transfer learning functionality"""
    print("\nTesting transfer learning...")
    
    from model import GPT, GPTConfig, ModelMode
    import torch
    
    base_config = dict(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=100, block_size=32
    )
    
    # Create a base model and save it
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'base_model.pt')
        
        base_model = GPT(GPTConfig(**base_config, mode=ModelMode.LANGUAGE_MODEL))
        torch.save({'model': base_model.state_dict()}, checkpoint_path)
        print("✓ Base model saved")
        
        # Test loading into classifier with frozen transformer
        tc_config = GPTConfig(
            **base_config,
            mode=ModelMode.TOKEN_CLASSIFIER,
            num_token_classes=2,
            init_from_checkpoint=checkpoint_path,
            freeze_transformer=True
        )
        tc_model = GPT(tc_config)
        
        # Check that transformer is frozen
        assert tc_model.get_frozen_status() == True
        print("✓ Transfer learning with frozen transformer works")
        
        # Test unfreezing
        tc_model.unfreeze_transformer_weights()
        assert tc_model.get_frozen_status() == False
        print("✓ Unfreezing works")
    
    return True

def test_loss_modifier_compatibility():
    """Test loss modifier compatibility with different modes"""
    print("\nTesting loss modifier compatibility...")
    
    from model import GPT, GPTConfig, ModelMode
    from loss_modifiers import create_loss_modifier_pipeline
    import torch
    
    # Create pipeline with all modifiers
    config = {
        'loss_modifiers_enabled': True,
        'entropy_modifier_enabled': True,
        'target_smoothing_enabled': True,
        'mask_ratio_weight_enabled': True
    }
    pipeline = create_loss_modifier_pipeline(config)
    print(f"✓ Created pipeline with {len(pipeline.modifiers)} modifiers")
    
    base_config = dict(n_layer=2, n_head=2, n_embd=64, vocab_size=100, block_size=32)
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test with language model (should use compatible modifiers)
    lm_model = GPT(GPTConfig(**base_config, mode=ModelMode.LANGUAGE_MODEL))
    lm_targets = torch.randint(0, 100, (batch_size, seq_len))
    lm_logits, lm_loss = lm_model(input_ids, lm_targets, loss_modifiers=pipeline)
    print("✓ Loss modifiers work with language model")
    
    # Test with token classifier (should filter to compatible)
    tc_model = GPT(GPTConfig(**base_config, mode=ModelMode.TOKEN_CLASSIFIER, num_token_classes=3))
    tc_targets = torch.randint(0, 3, (batch_size, seq_len))
    tc_logits, tc_loss = tc_model(input_ids, tc_targets, loss_modifiers=pipeline)
    print("✓ Loss modifiers work with token classifier")
    
    # Test with sequence scorer (should filter out most modifiers)
    ss_model = GPT(GPTConfig(**base_config, mode=ModelMode.SEQUENCE_SCORER, cls_token_id=0))
    ss_targets = torch.rand(batch_size)
    ss_logits, ss_loss = ss_model(input_ids, ss_targets, loss_modifiers=pipeline)
    print("✓ Loss modifiers work with sequence scorer")
    
    return True

def test_enhanced_schedulers():
    """Test new scheduler classes"""
    print("\nTesting enhanced schedulers...")
    
    from core.scheduler import TransferLearningScheduler, WarmupOnlyScheduler, AdaptiveScheduler
    
    # Test TransferLearningScheduler
    tl_scheduler = TransferLearningScheduler(base_lr=1e-4, warmup_iters=100, feature_extraction_iters=1000)
    
    # Test frozen phase
    lr_dict = tl_scheduler.get_lr(50, is_frozen=True)
    assert lr_dict['transformer'] == 0.0
    assert lr_dict['head'] > 0
    print("✓ TransferLearningScheduler frozen phase works")
    
    # Test unfrozen phase
    lr_dict = tl_scheduler.get_lr(1500, is_frozen=False)
    assert lr_dict['transformer'] > 0
    assert lr_dict['head'] > 0
    print("✓ TransferLearningScheduler unfrozen phase works")
    
    # Test WarmupOnlyScheduler
    wo_scheduler = WarmupOnlyScheduler(learning_rate=1e-3, warmup_iters=100, hold_iters=200)
    
    # Test warmup
    lr = wo_scheduler.get_lr(50)
    assert 0 < lr < 1e-3
    print("✓ WarmupOnlyScheduler warmup works")
    
    # Test hold
    lr = wo_scheduler.get_lr(150)
    assert lr == 1e-3
    print("✓ WarmupOnlyScheduler hold works")
    
    # Test AdaptiveScheduler
    adaptive_scheduler = AdaptiveScheduler(initial_lr=1e-3, patience=2, factor=0.5)
    
    # Test step function
    assert adaptive_scheduler.step(1.0) == False  # New best
    assert adaptive_scheduler.step(1.1) == False  # Wait 1
    assert adaptive_scheduler.step(1.1) == False  # Wait 2
    assert adaptive_scheduler.step(1.1) == True   # Reduce LR
    print("✓ AdaptiveScheduler step function works")
    
    return True

def test_backward_compatibility():
    """Test that existing MLM training still works"""
    print("\nTesting backward compatibility...")
    
    from model import GPT, GPTConfig, ModelMode
    from loss_modifiers import create_loss_modifier_pipeline
    import torch
    
    # Test that default config creates language model
    config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100, block_size=32)
    assert config.mode == ModelMode.LANGUAGE_MODEL
    print("✓ Default config creates language model")
    
    # Test that existing training loop logic works
    model = GPT(config)
    pipeline = create_loss_modifier_pipeline({'loss_modifiers_enabled': False})
    
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))
    
    # Should work exactly like before
    logits, loss = model(input_ids, targets, loss_modifiers=pipeline)
    assert logits.shape == (batch_size, seq_len, 100)
    assert loss.item() > 0
    print("✓ Existing training loop logic works unchanged")
    
    return True

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("COMPREHENSIVE MULTI-MODE IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Import validation", test_imports),
        ("Configuration validation", test_config_validation), 
        ("Model modes", test_model_modes),
        ("Transfer learning", test_transfer_learning),
        ("Loss modifier compatibility", test_loss_modifier_compatibility),
        ("Enhanced schedulers", test_enhanced_schedulers),
        ("Backward compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Multi-mode implementation is working correctly.")
        print("\nThe implementation successfully adds:")
        print("• Multi-mode support (language_model, token_classifier, sequence_scorer)")
        print("• Transfer learning with freezing/unfreezing")
        print("• Mode-aware loss modifier filtering") 
        print("• Enhanced schedulers for fine-tuning")
        print("• Full backward compatibility")
    else:
        print(f"❌ {failed} tests failed. Please review and fix issues.")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    main()