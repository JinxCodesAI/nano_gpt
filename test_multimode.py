"""Quick test script to validate multi-mode functionality"""

import torch
from model import GPT, GPTConfig, ModelMode

def test_mode_creation():
    """Test that models can be created in different modes"""
    print("Testing multi-mode model creation...")
    
    base_config = dict(
        n_layer=2, n_head=2, n_embd=64, 
        vocab_size=100, block_size=32
    )
    
    # Test Language Model mode
    print("\n1. Testing LANGUAGE_MODEL mode...")
    lm_config = GPTConfig(**base_config, mode=ModelMode.LANGUAGE_MODEL)
    lm_model = GPT(lm_config)
    print(f"✓ Language model created with {lm_model.get_num_params():,} parameters")
    assert hasattr(lm_model, 'lm_head')
    assert lm_model.lm_head.out_features == 100  # vocab_size
    
    # Test Token Classifier mode
    print("\n2. Testing TOKEN_CLASSIFIER mode...")
    tc_config = GPTConfig(**base_config, 
                         mode=ModelMode.TOKEN_CLASSIFIER,
                         num_token_classes=3,
                         attention_type='bidirectional')
    tc_model = GPT(tc_config)
    print(f"✓ Token classifier created with {tc_model.get_num_params():,} parameters")
    assert hasattr(tc_model, 'lm_head')
    assert tc_model.lm_head.out_features == 3  # num_token_classes
    
    # Test Sequence Scorer mode
    print("\n3. Testing SEQUENCE_SCORER mode...")
    ss_config = GPTConfig(**base_config,
                         mode=ModelMode.SEQUENCE_SCORER,
                         cls_token_id=0,
                         attention_type='bidirectional')
    ss_model = GPT(ss_config)
    print(f"✓ Sequence scorer created with {ss_model.get_num_params():,} parameters")
    assert hasattr(ss_model, 'sequence_head')
    
    return lm_model, tc_model, ss_model

def test_forward_passes():
    """Test forward passes in different modes"""
    print("\n\nTesting forward passes...")
    
    lm_model, tc_model, ss_model = test_mode_creation()
    
    # Create test data
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test Language Model forward
    print("\n1. Testing language model forward...")
    lm_targets = torch.randint(0, 100, (batch_size, seq_len))
    lm_logits, lm_loss = lm_model(input_ids, lm_targets)
    print(f"✓ LM logits shape: {lm_logits.shape}, loss: {lm_loss.item():.4f}")
    assert lm_logits.shape == (batch_size, seq_len, 100)
    
    # Test Token Classifier forward
    print("\n2. Testing token classifier forward...")
    tc_targets = torch.randint(0, 3, (batch_size, seq_len))
    tc_logits, tc_loss = tc_model(input_ids, tc_targets)
    print(f"✓ TC logits shape: {tc_logits.shape}, loss: {tc_loss.item():.4f}")
    assert tc_logits.shape == (batch_size, seq_len, 3)
    
    # Test Sequence Scorer forward
    print("\n3. Testing sequence scorer forward...")
    ss_targets = torch.rand(batch_size)  # 0-1 regression targets
    ss_logits, ss_loss = ss_model(input_ids, ss_targets)
    print(f"✓ SS logits shape: {ss_logits.shape}, loss: {ss_loss.item():.4f}")
    assert ss_logits.shape == (batch_size,)
    assert torch.all((ss_logits >= 0) & (ss_logits <= 1))  # Sigmoid output

def test_loss_modifiers():
    """Test loss modifiers with different modes"""
    print("\n\nTesting loss modifiers...")
    
    from loss_modifiers import create_loss_modifier_pipeline
    
    # Create a config with all modifiers enabled
    config = {
        'loss_modifiers_enabled': True,
        'entropy_modifier_enabled': True,
        'target_smoothing_enabled': True, 
        'mask_ratio_weight_enabled': True
    }
    
    pipeline = create_loss_modifier_pipeline(config)
    print(f"Created pipeline with {len(pipeline.modifiers)} modifiers")
    
    lm_model, tc_model, ss_model = test_mode_creation()
    
    # Test data
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    
    # Test with language model (should use all compatible modifiers)
    print("\n1. Testing loss modifiers with LANGUAGE_MODEL...")
    lm_targets = torch.randint(0, 100, (batch_size, seq_len))
    lm_logits, lm_loss = lm_model(input_ids, lm_targets, loss_modifiers=pipeline)
    print(f"✓ LM with modifiers - loss: {lm_loss.item():.4f}")
    
    # Test with token classifier (should filter compatible modifiers)
    print("\n2. Testing loss modifiers with TOKEN_CLASSIFIER...")
    tc_targets = torch.randint(0, 3, (batch_size, seq_len))
    tc_logits, tc_loss = tc_model(input_ids, tc_targets, loss_modifiers=pipeline)
    print(f"✓ TC with modifiers - loss: {tc_loss.item():.4f}")
    
    # Test with sequence scorer (should filter out most modifiers)
    print("\n3. Testing loss modifiers with SEQUENCE_SCORER...")
    ss_targets = torch.rand(batch_size)
    ss_logits, ss_loss = ss_model(input_ids, ss_targets, loss_modifiers=pipeline)
    print(f"✓ SS with modifiers - loss: {ss_loss.item():.4f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Mode GPT Implementation Test")
    print("=" * 60)
    
    try:
        test_mode_creation()
        test_forward_passes()
        test_loss_modifiers()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Multi-mode implementation works correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()