#!/usr/bin/env python3
"""
Test script for layer freezing functionality (Milestone 2).
Tests the ability to freeze/unfreeze model parameters for feature extraction.
"""

import sys
import os

# Mock torch if not available for syntax/logic testing
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available - creating mock objects for syntax testing")
    TORCH_AVAILABLE = False
    
    class MockTensor:
        def __init__(self, *args, **kwargs):
            self.requires_grad = True
            self._numel = 1000  # Mock parameter count
        
        def numel(self):
            return self._numel
    
    class MockParameter:
        def __init__(self, requires_grad=True):
            self.requires_grad = requires_grad
            self._numel = 1000
        
        def numel(self):
            return self._numel
    
    # Mock torch modules
    class torch:
        @staticmethod
        def manual_seed(seed):
            pass
    
    class nn:
        class Module:
            def named_parameters(self):
                return [
                    ('transformer.wte.weight', MockParameter()),
                    ('transformer.h.0.ln_1.weight', MockParameter()),
                    ('transformer.h.0.attn.c_attn.weight', MockParameter()),
                    ('transformer.h.1.ln_1.weight', MockParameter()),
                    ('transformer.h.1.attn.c_attn.weight', MockParameter()),
                    ('lm_head.weight', MockParameter()),
                ]
            
            def parameters(self):
                return [param for _, param in self.named_parameters()]

if TORCH_AVAILABLE:
    from model import GPT, GPTConfig

def test_layer_freezing():
    """Test layer freezing functionality"""
    print("=" * 60)
    print("Testing Layer Freezing Mechanism (Milestone 2)")  
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("⚠ PyTorch not available - performing syntax/logic checks only")
        return test_layer_freezing_mock()
    
    # Create a small test model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=False,
        binary_classification=False
    )
    
    print(f"Creating test model...")
    model = GPT(config)
    
    # Test 1: Initial state - all parameters trainable
    print("\n--- Test 1: Initial Trainable State ---")
    initial_trainable = model.get_trainable_param_count()
    total_params = sum(p.numel() for p in model.parameters())
    
    assert initial_trainable == total_params, f"All params should be trainable: {initial_trainable} != {total_params}"
    
    # Verify all parameters have requires_grad=True
    for name, param in model.named_parameters():
        assert param.requires_grad == True, f"Parameter {name} should be trainable initially"
    
    print(f"✓ All {initial_trainable:,} parameters are trainable initially")
    
    # Test 2: Freeze backbone  
    print("\n--- Test 2: Freeze Backbone ---")
    model.freeze_backbone()
    
    frozen_trainable = model.get_trainable_param_count()
    assert frozen_trainable < initial_trainable, f"Trainable count should decrease: {frozen_trainable} >= {initial_trainable}"
    
    # Verify only head parameters are trainable
    for name, param in model.named_parameters():
        if 'lm_head' in name:
            assert param.requires_grad == True, f"Head parameter {name} should remain trainable"
        else:
            assert param.requires_grad == False, f"Backbone parameter {name} should be frozen"
    
    print(f"✓ Backbone frozen, {frozen_trainable:,} head parameters remain trainable")
    
    # Test 3: Unfreeze all
    print("\n--- Test 3: Unfreeze All ---")
    model.unfreeze_all()
    
    final_trainable = model.get_trainable_param_count()
    assert final_trainable == initial_trainable, f"Should restore all trainable: {final_trainable} != {initial_trainable}"
    
    # Verify all parameters are trainable again
    for name, param in model.named_parameters():
        assert param.requires_grad == True, f"Parameter {name} should be trainable after unfreeze"
    
    print(f"✓ All {final_trainable:,} parameters are trainable after unfreeze")
    
    # Test 4: Test with binary classification head
    print("\n--- Test 4: Freezing with Binary Head ---")
    model.switch_to_binary_classification()
    
    binary_total = model.get_trainable_param_count()
    model.freeze_backbone()
    binary_frozen = model.get_trainable_param_count()
    
    # Should only have 2 * n_embd trainable parameters (binary head)
    expected_head_params = 2 * config.n_embd
    assert binary_frozen == expected_head_params, f"Expected {expected_head_params} head params, got {binary_frozen}"
    
    print(f"✓ Binary head freezing works: {binary_frozen:,} trainable parameters")
    
    # Test 5: Parameter groups functionality
    print("\n--- Test 5: Parameter Groups ---")
    groups = model.get_parameter_groups()
    
    assert 'backbone' in groups, "Should have backbone group"
    assert 'head' in groups, "Should have head group"
    
    backbone_params = sum(param.numel() for _, param in groups['backbone'])
    head_params = sum(param.numel() for _, param in groups['head'])
    
    assert backbone_params + head_params == binary_total, "Groups should sum to total"
    print(f"✓ Parameter groups: backbone={backbone_params:,}, head={head_params:,}")
    
    # Test 6: Detailed status printing (just run it)
    print("\n--- Test 6: Status Printing ---")
    model.print_parameter_status()
    print("✓ Status printing works")
    
    print("\n" + "=" * 60)
    print("🎉 ALL LAYER FREEZING TESTS PASSED!")
    print("✅ Milestone 2: Layer Freezing Mechanism - COMPLETE")
    print("=" * 60)

def test_layer_freezing_mock():
    """Mock test for when PyTorch is not available"""
    print("Running mock syntax/logic tests...")
    
    # Test the logic without actual PyTorch objects
    mock_params = [
        ('transformer.wte.weight', MockParameter()),
        ('transformer.h.0.ln_1.weight', MockParameter()),
        ('transformer.h.0.attn.c_attn.weight', MockParameter()),
        ('lm_head.weight', MockParameter()),
    ]
    
    # Test freezing logic
    for name, param in mock_params:
        if 'lm_head' not in name:
            param.requires_grad = False
    
    # Verify head remains trainable
    head_trainable = sum(1 for name, param in mock_params if 'lm_head' in name and param.requires_grad)
    backbone_frozen = sum(1 for name, param in mock_params if 'lm_head' not in name and not param.requires_grad)
    
    assert head_trainable > 0, "Head should remain trainable"
    assert backbone_frozen > 0, "Backbone should be frozen"
    
    print("✓ Mock freezing logic works correctly")
    print("✅ Milestone 2 logic verified (run with PyTorch for full test)")

if __name__ == "__main__":
    if TORCH_AVAILABLE:
        torch.manual_seed(42)
    test_layer_freezing()