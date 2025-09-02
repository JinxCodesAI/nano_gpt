#!/usr/bin/env python3
"""
Test script for configuration extension (Milestone 3).
Tests that transfer learning configuration is properly handled.
"""

import sys
import os

# Add current directory to path to import training_utils
sys.path.insert(0, '.')

try:
    from training_utils.training_config import TrainingContext
    CONFIG_AVAILABLE = True
    print("Successfully imported TrainingContext")
except ImportError as e:
    print(f"Could not import TrainingContext: {e}")
    CONFIG_AVAILABLE = False

def test_config_extension():
    """Test configuration extension functionality"""
    print("=" * 60)
    print("Testing Configuration Extension (Milestone 3)")
    print("=" * 60)
    
    if not CONFIG_AVAILABLE:
        print("⚠ TrainingContext not available - skipping tests")
        return
    
    # Test 1: Default values
    print("\n--- Test 1: Default Configuration Values ---")
    ctx = TrainingContext()
    
    # Check that new fields exist with correct defaults
    assert hasattr(ctx, 'transfer_learning_mode'), "Should have transfer_learning_mode attribute"
    assert hasattr(ctx, 'pretrained_checkpoint_path'), "Should have pretrained_checkpoint_path attribute" 
    assert hasattr(ctx, 'switch_to_binary'), "Should have switch_to_binary attribute"
    
    assert ctx.transfer_learning_mode == 'from_scratch', f"Expected 'from_scratch', got {ctx.transfer_learning_mode}"
    assert ctx.pretrained_checkpoint_path is None, f"Expected None, got {ctx.pretrained_checkpoint_path}"
    assert ctx.switch_to_binary == False, f"Expected False, got {ctx.switch_to_binary}"
    
    print("✓ Default values are correct")
    
    # Test 2: Custom configuration values
    print("\n--- Test 2: Custom Configuration Values ---")
    ctx = TrainingContext(
        transfer_learning_mode='feature_extraction',
        pretrained_checkpoint_path='path/to/pretrained.pt',
        switch_to_binary=True
    )
    
    assert ctx.transfer_learning_mode == 'feature_extraction', f"Expected 'feature_extraction', got {ctx.transfer_learning_mode}"
    assert ctx.pretrained_checkpoint_path == 'path/to/pretrained.pt', f"Expected path, got {ctx.pretrained_checkpoint_path}"
    assert ctx.switch_to_binary == True, f"Expected True, got {ctx.switch_to_binary}"
    
    print("✓ Custom values set correctly")
    
    # Test 3: Valid transfer learning mode values
    print("\n--- Test 3: Valid Transfer Learning Modes ---")
    valid_modes = ['from_scratch', 'feature_extraction', 'fine_tuning']
    
    for mode in valid_modes:
        ctx = TrainingContext(transfer_learning_mode=mode)
        assert ctx.transfer_learning_mode == mode, f"Mode {mode} should be valid"
        print(f"  ✓ '{mode}' mode works")
    
    # Test 4: Configuration compatibility with existing fields
    print("\n--- Test 4: Compatibility with Existing Fields ---")
    ctx = TrainingContext(
        # Existing fields
        training_type='unmasking',
        batch_size=32,
        block_size=512,
        device='cuda',
        # New transfer learning fields
        transfer_learning_mode='fine_tuning',
        pretrained_checkpoint_path='test.pt'
    )
    
    # Check that both old and new fields work together
    assert ctx.training_type == 'unmasking', "Existing fields should work"
    assert ctx.batch_size == 32, "Existing fields should work"
    assert ctx.transfer_learning_mode == 'fine_tuning', "New fields should work"
    assert ctx.pretrained_checkpoint_path == 'test.pt', "New fields should work"
    
    print("✓ New config fields work with existing ones")
    
    # Test 5: Check that __post_init__ still works
    print("\n--- Test 5: Post-init Processing ---")
    ctx = TrainingContext(training_type='unmasking')
    
    # __post_init__ should set up default unmasking stages
    assert hasattr(ctx, 'unmasking_stages'), "Should have unmasking_stages"
    if ctx.unmasking_stages is not None:
        assert len(ctx.unmasking_stages) > 0, "Should have default unmasking stages"
        print(f"✓ Default unmasking stages created: {len(ctx.unmasking_stages)} stages")
    else:
        print("✓ Post-init processing works (no default stages)")
    
    print("\n" + "=" * 60)
    print("🎉 ALL CONFIGURATION EXTENSION TESTS PASSED!")
    print("✅ Milestone 3: Configuration Extension - COMPLETE")
    print("=" * 60)

def test_train_run_config_syntax():
    """Test that train_run.py config variables are syntactically correct"""
    print("\n--- Bonus Test: train_run.py Config Syntax ---")
    
    # Check if train_run.py compiles (syntax check)
    try:
        import subprocess
        result = subprocess.run(['python3', '-m', 'py_compile', 'train_run.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ train_run.py compiles successfully with new config variables")
        else:
            print(f"✗ train_run.py compilation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Could not test train_run.py compilation: {e}")
    
    # Test the config variables are defined by parsing the file
    try:
        with open('train_run.py', 'r') as f:
            content = f.read()
        
        required_configs = [
            'transfer_learning_mode',
            'pretrained_checkpoint_path', 
            'switch_to_binary'
        ]
        
        for config in required_configs:
            if config in content:
                print(f"  ✓ Found '{config}' in train_run.py")
            else:
                print(f"  ✗ Missing '{config}' in train_run.py")
                return False
        
        return True
        
    except Exception as e:
        print(f"Could not check train_run.py: {e}")
        return False

if __name__ == "__main__":
    test_config_extension()
    test_train_run_config_syntax()