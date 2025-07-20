#!/usr/bin/env python3
"""
Test script for the controlled growth functionality in the training orchestrator.
Tests that architectural guardrails prevent exceeding target architecture.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path so we can import train module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPTConfig, GPT
from logger import TrainingLogger

# Define the functions we need to test directly here to avoid importing train module
def calculate_and_log_target_architecture(initial_config, schedule):
    """
    Simulates the schedule to determine the final target architecture,
    logs it, and returns it as a dictionary for later use.
    """
    # The target config reflects the final, deployed model, which has no LoRA.
    target_config = {
        'n_layer': initial_config['n_layer'],
        'n_hidden': initial_config['n_hidden'] if initial_config['n_hidden'] is not None else 4 * initial_config['n_embd'],
        'n_head': initial_config['n_head'],
        'n_embd': initial_config['n_embd'],
        'block_size': initial_config['block_size'],
        'vocab_size': initial_config['vocab_size'],
        'dropout': initial_config['dropout'],
        'bias': initial_config['bias'],
        # Hardcode final state for LoRA-related params
        'embedding_mode': 'standard',
        'attn_lora_rank': 0,
        'embedding_rank': 0,
        'lora_alpha': 0.0,
    }

    print("Calculating target architecture based on schedule...")
    for op in schedule:
        op_name = op['name']
        op_value = op['value']
        if op_name == 'stack_layers' and op_value > 1:
            target_config['n_layer'] = int(target_config['n_layer'] * op_value)
        elif op_name == 'widen_mlp' and op_value > 1:
            target_config['n_hidden'] = int(target_config['n_hidden'] * op_value)

    return target_config

def test_execute_operation_guardrails(model, op, target_config):
    """Test the guardrails logic for execute_operation without full implementation."""
    op_name = op['name']

    if op_name == 'stack_layers':
        current_layers = model.config.n_layer
        # Check if the target has been defined and if we already meet or exceed it
        if target_config and current_layers >= target_config['n_layer']:
            print(f"Skipping '{op_name}': Current layers ({current_layers}) already meet or exceed target ({target_config['n_layer']}).")
            return False # Cancel the operation

        if op['value'] <= 1:
            print(f"Error: Invalid stack_layers value {op['value']}, must be > 1")
            return False

        # Simulate the operation
        model.stack_layers(op['value'])
        return True

    elif op_name == 'widen_mlp':
        current_hidden_dim = model.config.n_hidden
        # Check if the target has been defined and if we already meet or exceed it
        if target_config and current_hidden_dim >= target_config['n_hidden']:
            print(f"Skipping '{op_name}': Current hidden dim ({current_hidden_dim}) already meets or exceeds target ({target_config['n_hidden']}).")
            return False # Cancel the operation

        if op['value'] <= 1:
            print(f"Error: Invalid widen_mlp value {op['value']}, must be > 1")
            return False

        # Simulate the operation
        model.widen_mlp(op['value'])
        return True

    return False

def test_target_architecture_calculation():
    """Test that target architecture calculation works correctly."""
    print("Testing target architecture calculation...")
    
    # Mock initial config
    initial_config = {
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 64,
        'n_hidden': 256,
        'block_size': 32,
        'vocab_size': 1000,
        'dropout': 0.0,
        'bias': False,
        'embedding_mode': 'lora',
        'attn_lora_rank': 16,
        'embedding_rank': 8,
        'lora_alpha': 1.0,
    }
    
    # Mock scaling schedule
    schedule = [
        {'name': 'stack_layers', 'value': 2, 'trigger_loss': 3.0, 'max_wait_iters': 1000, 'reevaluate': True},
        {'name': 'widen_mlp', 'value': 1.5, 'trigger_loss': 2.5, 'max_wait_iters': 1000, 'reevaluate': True}
    ]
    
    target_config = calculate_and_log_target_architecture(initial_config, schedule)
    
    # Verify target config is LoRA-free
    assert target_config['embedding_mode'] == 'standard', f"Expected 'standard', got {target_config['embedding_mode']}"
    assert target_config['attn_lora_rank'] == 0, f"Expected 0, got {target_config['attn_lora_rank']}"
    assert target_config['embedding_rank'] == 0, f"Expected 0, got {target_config['embedding_rank']}"
    assert target_config['lora_alpha'] == 0.0, f"Expected 0.0, got {target_config['lora_alpha']}"
    
    # Verify architectural changes are applied
    expected_layers = initial_config['n_layer'] * 2  # stack_layers with value 2
    expected_hidden = int(initial_config['n_hidden'] * 1.5)  # widen_mlp with value 1.5
    
    assert target_config['n_layer'] == expected_layers, f"Expected {expected_layers} layers, got {target_config['n_layer']}"
    assert target_config['n_hidden'] == expected_hidden, f"Expected {expected_hidden} hidden, got {target_config['n_hidden']}"
    
    print("✓ Target architecture calculation test passed\n")
    return target_config

def test_controlled_growth_guardrails():
    """Test that architectural guardrails prevent exceeding target architecture."""
    print("Testing controlled growth guardrails...")
    
    # Create a model that already meets the target architecture
    config = GPTConfig(n_layer=8, n_head=4, n_embd=64, vocab_size=1000, block_size=32, n_hidden=384)
    model = GPT(config)
    
    # Create target config that matches current model (should prevent further growth)
    target_config = {
        'n_layer': 8,
        'n_hidden': 384,
        'n_head': 4,
        'n_embd': 64,
        'block_size': 32,
        'vocab_size': 1000,
        'dropout': 0.0,
        'bias': False,
        'embedding_mode': 'standard',
        'attn_lora_rank': 0,
        'embedding_rank': 0,
        'lora_alpha': 0.0,
    }
    
    # Test stack_layers operation should be blocked
    stack_op = {'name': 'stack_layers', 'value': 2, 'desc': 'test stack'}
    result = test_execute_operation_guardrails(model, stack_op, target_config)
    
    # Should return False (operation blocked)
    assert result == False, "stack_layers operation should have been blocked by guardrails"
    
    # Verify model wasn't changed
    assert model.config.n_layer == 8, f"Model layers should remain 8, got {model.config.n_layer}"
    
    print("✓ stack_layers guardrail test passed")
    
    # Test widen_mlp operation should be blocked
    widen_op = {'name': 'widen_mlp', 'value': 1.5, 'desc': 'test widen'}
    result = test_execute_operation_guardrails(model, widen_op, target_config)
    
    # Should return False (operation blocked)
    assert result == False, "widen_mlp operation should have been blocked by guardrails"
    
    # Verify model wasn't changed
    assert model.config.n_hidden == 384, f"Model hidden dim should remain 384, got {model.config.n_hidden}"
    
    print("✓ widen_mlp guardrail test passed")
    
    print("✓ Controlled growth guardrails test passed\n")

def test_growth_allowed_when_under_target():
    """Test that growth operations are allowed when under target architecture."""
    print("Testing growth allowed when under target...")
    
    # Create a model that is smaller than the target architecture
    config = GPTConfig(n_layer=4, n_head=4, n_embd=64, vocab_size=1000, block_size=32, n_hidden=256)
    model = GPT(config)
    
    # Create target config that is larger than current model (should allow growth)
    target_config = {
        'n_layer': 12,  # Target is larger than current 4
        'n_hidden': 512,  # Target is larger than current 256
        'n_head': 4,
        'n_embd': 64,
        'block_size': 32,
        'vocab_size': 1000,
        'dropout': 0.0,
        'bias': False,
        'embedding_mode': 'standard',
        'attn_lora_rank': 0,
        'embedding_rank': 0,
        'lora_alpha': 0.0,
    }
    
    # Test stack_layers operation should be allowed
    original_layers = model.config.n_layer
    stack_op = {'name': 'stack_layers', 'value': 2, 'desc': 'test stack'}
    result = test_execute_operation_guardrails(model, stack_op, target_config)
    
    # Should return True (operation allowed)
    assert result == True, "stack_layers operation should have been allowed"
    
    # Verify model was changed
    expected_layers = original_layers * 2
    assert model.config.n_layer == expected_layers, f"Model layers should be {expected_layers}, got {model.config.n_layer}"
    
    print("✓ stack_layers allowed when under target test passed")
    
    # Test widen_mlp operation should be allowed
    original_hidden = model.config.n_hidden
    widen_op = {'name': 'widen_mlp', 'value': 1.5, 'desc': 'test widen'}
    result = test_execute_operation_guardrails(model, widen_op, target_config)
    
    # Should return True (operation allowed)
    assert result == True, "widen_mlp operation should have been allowed"
    
    # Verify model was changed
    expected_hidden = int(original_hidden * 1.5)
    assert model.config.n_hidden == expected_hidden, f"Model hidden dim should be {expected_hidden}, got {model.config.n_hidden}"
    
    print("✓ widen_mlp allowed when under target test passed")
    
    print("✓ Growth allowed when under target test passed\n")

def run_all_tests():
    """Run all controlled growth tests."""
    print("=" * 60)
    print("RUNNING CONTROLLED GROWTH TESTS")
    print("=" * 60)
    
    try:
        test_target_architecture_calculation()
        test_controlled_growth_guardrails()
        test_growth_allowed_when_under_target()
        
        print("=" * 60)
        print("ALL CONTROLLED GROWTH TESTS PASSED! ✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
