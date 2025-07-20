#!/usr/bin/env python3
"""
Test script for enhanced logging functionality
"""

import os
import sys
import time
from datetime import datetime

# Add the current directory to the path so we can import train modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock wandb for testing
class MockWandB:
    def __init__(self):
        self.logged_data = []
    
    def log(self, data):
        self.logged_data.append(data)
        print(f"[MOCK W&B] Logged: {data}")

# Set up test environment
os.environ['RANK'] = '-1'  # Disable DDP
master_process = True
wandb_log = True
wandb = MockWandB()

# Define the functions we want to test (copied from train.py to avoid import issues)
def log_model_architecture(model, iter_num, is_initial=False, is_target=False):
    """Logs the model's current or target architecture to the console and W&B."""
    if not master_process:
        return

    # Determine the header based on the context
    if is_target:
        header = "TARGET MODEL ARCHITECTURE (at end of schedule)"
    elif is_initial:
        header = f"INITIAL MODEL ARCHITECTURE (at Iter {iter_num})"
    else:
        header = f"ARCHITECTURE CHANGE (at Iter {iter_num})"

    # Get the raw model config
    config = model.config if hasattr(model, 'config') else model

    print("\n" + "="*60)
    print(f"{header:^60}")
    print("="*60)

    arch_info = {
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd,
        'n_hidden': config.n_hidden if config.n_hidden is not None else 4 * config.n_embd,
        'block_size': config.block_size,
        'vocab_size': config.vocab_size,
        'dropout': config.dropout,
        'bias': config.bias,
        'embedding_mode': config.embedding_mode,
        'attn_lora_rank': config.attn_lora_rank,
        'embedding_rank': config.embedding_rank,
        'lora_alpha': config.lora_alpha
    }

    # Print to console
    for key, value in arch_info.items():
        print(f"  {key:<22} | {value}")
    print("="*60 + "\n")

    # Log to Weights & Biases
    if wandb_log:
        # We prefix with 'arch/' to group these parameters in the W&B UI
        wandb_log_data = {f"arch/{k}": v for k, v in arch_info.items()}
        wandb_log_data['iter'] = iter_num
        wandb.log(wandb_log_data)

def calculate_and_log_target_architecture(initial_config, schedule):
    """Simulates the scaling schedule to determine and log the final architecture."""
    if not master_process:
        return

    # Use a dictionary to avoid modifying the actual config object
    target_config = {
        'n_layer': initial_config['n_layer'],
        'n_hidden': initial_config['n_hidden'] if initial_config['n_hidden'] is not None else 4 * initial_config['n_embd'],
        # Add other relevant initial parameters
        'n_head': initial_config['n_head'],
        'n_embd': initial_config['n_embd'],
        'block_size': initial_config['block_size'],
        'vocab_size': initial_config['vocab_size'],
        'dropout': initial_config['dropout'],
        'bias': initial_config['bias'],
        'embedding_mode': initial_config.get('embedding_mode', 'standard'), # Assume standard at start
        'attn_lora_rank': initial_config.get('attn_lora_rank', 0), # Assume 0 at start
        'embedding_rank': initial_config.get('embedding_rank', 0),
        'lora_alpha': initial_config.get('lora_alpha', 1.0),
    }

    print("Calculating target architecture based on schedule...")
    for op in schedule:
        op_name = op['name']
        op_value = op['value']
        if op_name == 'stack_layers' and op_value > 1:
            target_config['n_layer'] = int(target_config['n_layer'] * op_value)
        elif op_name == 'widen_mlp' and op_value > 1:
            target_config['n_hidden'] = int(target_config['n_hidden'] * op_value)

    # Log this calculated target architecture using our new function
    # We pass the dictionary directly since we don't have a full model object for the target
    log_model_architecture(
        type('FakeConfig', (), target_config)(), # Create a temporary object with a .config attribute
        iter_num=0,
        is_target=True
    )

# Create a mock model config for testing
class MockConfig:
    def __init__(self):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.n_hidden = 3072
        self.block_size = 1024
        self.vocab_size = 50304
        self.dropout = 0.0
        self.bias = False
        self.embedding_mode = 'standard'
        self.attn_lora_rank = 0
        self.embedding_rank = 0
        self.lora_alpha = 1.0

class MockModel:
    def __init__(self):
        self.config = MockConfig()

def test_log_model_architecture():
    """Test the log_model_architecture function"""
    print("\n" + "="*60)
    print("TESTING log_model_architecture function")
    print("="*60)
    
    model = MockModel()
    
    # Test initial architecture logging
    print("\n1. Testing initial architecture logging:")
    log_model_architecture(model, iter_num=0, is_initial=True)
    
    # Test architecture change logging
    print("\n2. Testing architecture change logging:")
    log_model_architecture(model, iter_num=1000)
    
    # Test target architecture logging
    print("\n3. Testing target architecture logging:")
    log_model_architecture(model, iter_num=0, is_target=True)
    
    print(f"\nW&B logged {len(wandb.logged_data)} entries")
    return True

def test_calculate_and_log_target_architecture():
    """Test the calculate_and_log_target_architecture function"""
    print("\n" + "="*60)
    print("TESTING calculate_and_log_target_architecture function")
    print("="*60)
    
    # Create mock initial config
    initial_config = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'n_hidden': None,  # Should default to 4 * n_embd
        'block_size': 1024,
        'vocab_size': 50304,
        'dropout': 0.0,
        'bias': False,
        'embedding_mode': 'standard',
        'attn_lora_rank': 0,
        'embedding_rank': 0,
        'lora_alpha': 1.0,
    }
    
    # Create mock scaling schedule
    scaling_schedule = [
        {
            'name': 'stack_layers',
            'value': 2,
            'trigger_loss': 3.0,
            'max_wait_iters': 1000,
            'reevaluate': True
        },
        {
            'name': 'widen_mlp',
            'value': 1.5,
            'trigger_loss': 2.5,
            'max_wait_iters': 2000,
            'reevaluate': True
        }
    ]
    
    print("\n1. Testing target architecture calculation:")
    print(f"Initial config: n_layer={initial_config['n_layer']}, n_hidden={initial_config['n_hidden']}")
    print(f"Schedule operations: {[op['name'] + ' x' + str(op['value']) for op in scaling_schedule]}")
    
    calculate_and_log_target_architecture(initial_config, scaling_schedule)
    
    return True

def test_timestamp_generation():
    """Test timestamp generation for W&B run names"""
    print("\n" + "="*60)
    print("TESTING timestamp generation")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = "test_run"
    final_wandb_run_name = f"{wandb_run_name}_{timestamp}"
    
    print(f"Original run name: {wandb_run_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Final run name: {final_wandb_run_name}")
    
    # Verify format
    assert len(timestamp) == 15, f"Timestamp should be 15 chars, got {len(timestamp)}"
    assert "_" in final_wandb_run_name, "Final run name should contain underscore"
    
    return True

def main():
    """Run all tests"""
    print("Starting Enhanced Logging Tests")
    print("="*60)
    
    tests = [
        test_timestamp_generation,
        test_log_model_architecture,
        test_calculate_and_log_target_architecture,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                print(f"✓ {test.__name__} PASSED")
                passed += 1
            else:
                print(f"✗ {test.__name__} FAILED")
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
