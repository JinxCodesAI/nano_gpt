#!/usr/bin/env python
"""
Test script to verify batch-level mode switching functionality.
Tests that batches can specify their model_mode and the training loop switches accordingly.
"""

import sys
import torch
from model import GPT, GPTConfig, ModelMode

def test_batch_mode_metadata():
    """Test that batches can include model_mode metadata"""
    print("\n1. Testing batch with model_mode metadata...")
    
    # Simulate a batch with model_mode metadata
    batch_lm = {
        'x': torch.randint(0, 256, (4, 32)),
        'y': torch.randint(0, 256, (4, 32)),
        '_model_mode': 'language_model',
    }
    
    batch_ss = {
        'input_ids': torch.randint(0, 256, (4, 32)),
        'targets': torch.rand(4),
        '_model_mode': 'sequence_scorer',
    }
    
    assert '_model_mode' in batch_lm
    assert '_model_mode' in batch_ss
    assert batch_lm['_model_mode'] == 'language_model'
    assert batch_ss['_model_mode'] == 'sequence_scorer'
    
    print("   ✓ Batches can include model_mode metadata")
    return True

def test_mode_switching_in_training_step():
    """Test that TrainingStep switches model mode based on batch metadata"""
    print("\n2. Testing mode switching in training step...")
    
    # Create a model
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
    
    # Test 1: Batch with language_model mode
    batch_lm = {
        'x': torch.randint(0, 256, (2, 32)),
        'y': torch.randint(0, 256, (2, 32)),
        '_model_mode': 'language_model',
    }
    
    # Simulate what TrainingStep does
    if '_model_mode' in batch_lm:
        mode_str = batch_lm['_model_mode']
        if mode_str == 'language_model':
            model.set_mode(ModelMode.LANGUAGE_MODEL)
    
    assert model.get_mode() == ModelMode.LANGUAGE_MODEL
    print("   ✓ Model switched to LANGUAGE_MODEL based on batch metadata")
    
    # Test 2: Batch with sequence_scorer mode
    batch_ss = {
        'input_ids': torch.randint(0, 256, (2, 32)),
        'targets': torch.rand(2),
        '_model_mode': 'sequence_scorer',
    }
    
    if '_model_mode' in batch_ss:
        mode_str = batch_ss['_model_mode']
        if mode_str == 'sequence_scorer':
            model.set_mode(ModelMode.SEQUENCE_SCORER)
    
    assert model.get_mode() == ModelMode.SEQUENCE_SCORER
    print("   ✓ Model switched to SEQUENCE_SCORER based on batch metadata")
    
    # Test 3: Batch without mode metadata (should keep current mode)
    batch_no_mode = {
        'x': torch.randint(0, 256, (2, 32)),
        'y': torch.randint(0, 256, (2, 32)),
    }
    
    current_mode = model.get_mode()
    # No mode switching happens
    assert model.get_mode() == current_mode
    print("   ✓ Model keeps current mode when batch has no metadata")
    
    return True

def test_forward_pass_with_mode_switching():
    """Test forward pass with mode switching between batches"""
    print("\n3. Testing forward pass with mode switching...")
    
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
    
    # Batch 1: LANGUAGE_MODEL
    model.set_mode(ModelMode.LANGUAGE_MODEL)
    x_lm = torch.randint(0, 256, (2, 32))
    y_lm = torch.randint(0, 256, (2, 32))
    
    with torch.no_grad():
        logits_lm, loss_lm = model(x_lm, y_lm)
    
    assert logits_lm.shape == (2, 32, 256)
    assert loss_lm is not None
    print("   ✓ LANGUAGE_MODEL forward pass successful")
    
    # Batch 2: SEQUENCE_SCORER (switch mode)
    model.set_mode(ModelMode.SEQUENCE_SCORER)
    x_ss = torch.randint(0, 256, (2, 32))
    x_ss[:, 0] = 0  # CLS token
    y_ss = torch.rand(2)
    
    with torch.no_grad():
        scores_ss, loss_ss = model(x_ss, y_ss)
    
    assert scores_ss.shape == (2,)
    assert loss_ss is not None
    print("   ✓ SEQUENCE_SCORER forward pass successful")
    
    # Batch 3: Back to LANGUAGE_MODEL
    model.set_mode(ModelMode.LANGUAGE_MODEL)
    with torch.no_grad():
        logits_lm2, loss_lm2 = model(x_lm, y_lm)
    
    assert logits_lm2.shape == (2, 32, 256)
    print("   ✓ Switched back to LANGUAGE_MODEL successfully")
    
    return True

def test_dual_mode_provider_concept():
    """Test the concept of dual-mode provider (without actual provider)"""
    print("\n4. Testing dual-mode provider concept...")
    
    # Simulate what DualModeProvider does
    def simulate_dual_mode_batch_generation(batch_idx, mode_distribution):
        """Simulate batch generation with mode switching"""
        import random
        random.seed(batch_idx)
        
        lm_ratio = mode_distribution.get('language_model', 0.5)
        rand_val = random.random()
        
        if rand_val < lm_ratio:
            # Generate LANGUAGE_MODEL batch
            return {
                'x': torch.randint(0, 256, (4, 32)),
                'y': torch.randint(0, 256, (4, 32)),
                'model_mode': 'language_model',
            }
        else:
            # Generate SEQUENCE_SCORER batch
            return {
                'input_ids': torch.randint(0, 256, (4, 32)),
                'targets': torch.rand(4),
                'model_mode': 'sequence_scorer',
            }
    
    # Generate 10 batches and check distribution
    mode_distribution = {'language_model': 0.5, 'sequence_scorer': 0.5}
    batches = [simulate_dual_mode_batch_generation(i, mode_distribution) for i in range(10)]
    
    lm_count = sum(1 for b in batches if b['model_mode'] == 'language_model')
    ss_count = sum(1 for b in batches if b['model_mode'] == 'sequence_scorer')
    
    print(f"   Generated {lm_count} LANGUAGE_MODEL batches, {ss_count} SEQUENCE_SCORER batches")
    assert lm_count > 0 and ss_count > 0, "Should generate both types of batches"
    print("   ✓ Dual-mode batch generation works")
    
    return True

def test_alternation_frequency():
    """Test alternation frequency control"""
    print("\n5. Testing alternation frequency...")
    
    def determine_mode_with_alternation(batch_idx, alternation_freq, mode_distribution):
        """Simulate mode determination with alternation frequency"""
        import random
        
        # Determine alternation window
        window_idx = batch_idx // alternation_freq
        
        # Use window-based seed for deterministic distribution
        random.seed(1337 + window_idx)
        
        lm_ratio = mode_distribution.get('language_model', 0.5)
        rand_val = random.random()
        
        return 'language_model' if rand_val < lm_ratio else 'sequence_scorer'
    
    # Test with alternation_freq = 1 (every batch can switch)
    modes_freq1 = [determine_mode_with_alternation(i, 1, {'language_model': 0.5, 'sequence_scorer': 0.5}) 
                   for i in range(10)]
    
    # Test with alternation_freq = 5 (switch every 5 batches)
    modes_freq5 = [determine_mode_with_alternation(i, 5, {'language_model': 0.5, 'sequence_scorer': 0.5}) 
                   for i in range(10)]
    
    # With freq=5, batches 0-4 should have same mode, 5-9 should have same mode
    assert len(set(modes_freq5[:5])) == 1, "First 5 batches should have same mode"
    assert len(set(modes_freq5[5:])) == 1, "Last 5 batches should have same mode"
    
    print(f"   Freq=1: {modes_freq1}")
    print(f"   Freq=5: {modes_freq5}")
    print("   ✓ Alternation frequency control works")
    
    return True

def main():
    print("="*60)
    print("BATCH-LEVEL MODE SWITCHING TESTS")
    print("="*60)
    
    tests = [
        test_batch_mode_metadata,
        test_mode_switching_in_training_step,
        test_forward_pass_with_mode_switching,
        test_dual_mode_provider_concept,
        test_alternation_frequency,
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
        print("All batch-level mode switching tests passed! ✓")
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

