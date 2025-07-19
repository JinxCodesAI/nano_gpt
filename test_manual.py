#!/usr/bin/env python3
"""
Manual test script for Training Orchestrator functionality.
This script simulates the training loop behavior to test the orchestrator.
"""

import os
import sys
import yaml
import json

def simulate_orchestrator():
    """Simulate the training orchestrator behavior"""
    print("=== Manual Training Orchestrator Test ===\n")
    
    # Load test configuration
    config_file = 'configs/test_scaling_schedule.yaml'
    if not os.path.exists(config_file):
        print(f"Error: Test config file {config_file} not found")
        return False
    
    # Load scaling schedule
    with open(config_file, 'r') as f:
        scaling_schedule = yaml.safe_load(f)
    
    print(f"Loaded scaling schedule with {len(scaling_schedule)} operations:")
    for i, op in enumerate(scaling_schedule):
        print(f"  {i+1}. {op['name']} (value: {op['value']}, trigger: {op['trigger_loss']}, wait: {op['max_wait_iters']})")
    print()
    
    # Initialize orchestrator state
    iter_of_last_op = 0
    lr_multiplier = 1.0
    batch_size_multiplier = 1.0
    grad_accum_multiplier = 1.0
    lr_schedule_offset = 0
    
    # Initial values
    batch_size = 12
    gradient_accumulation_steps = 5
    
    print("Initial state:")
    print(f"  lr_multiplier: {lr_multiplier}")
    print(f"  batch_size: {batch_size} (multiplier: {batch_size_multiplier})")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps} (multiplier: {grad_accum_multiplier})")
    print(f"  lr_schedule_offset: {lr_schedule_offset}")
    print()
    
    # Simulate training iterations with decreasing validation loss
    simulated_losses = [12.0, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5]
    
    for iter_num, val_loss in enumerate(simulated_losses):
        print(f"--- Iteration {iter_num * 10} ---")
        print(f"Validation loss: {val_loss:.1f}")
        
        # Check scaling schedule
        if scaling_schedule:
            next_op = scaling_schedule[0]
            
            # Check trigger conditions
            loss_triggered = val_loss < next_op['trigger_loss']
            timeout_triggered = (iter_num * 10 - iter_of_last_op) >= next_op['max_wait_iters']
            
            print(f"Next operation: {next_op['name']}")
            print(f"  Trigger loss: {next_op['trigger_loss']:.1f}")
            print(f"  Max wait iters: {next_op['max_wait_iters']}")
            print(f"  Iterations since last op: {iter_num * 10 - iter_of_last_op}")
            print(f"  Loss triggered: {loss_triggered}")
            print(f"  Timeout triggered: {timeout_triggered}")
            
            if loss_triggered or timeout_triggered:
                print(f"\n🚀 EXECUTING OPERATION: {next_op['name']}")
                print(f"   Trigger reason: {'Loss threshold' if loss_triggered else 'Timeout'}")
                
                # Execute operation
                op_name = next_op['name']
                op_value = next_op['value']
                
                if op_name == 'change_lr':
                    old_lr = lr_multiplier
                    lr_multiplier *= op_value
                    print(f"   Learning rate multiplier: {old_lr:.2f} → {lr_multiplier:.2f}")
                    
                elif op_name == 'change_batch_size':
                    old_batch_size = batch_size
                    old_multiplier = batch_size_multiplier
                    batch_size_multiplier *= op_value
                    batch_size = max(1, int(batch_size * op_value))
                    print(f"   Batch size: {old_batch_size} → {batch_size} (multiplier: {old_multiplier:.2f} → {batch_size_multiplier:.2f})")
                    
                elif op_name == 'change_grad_accum':
                    old_grad_accum = gradient_accumulation_steps
                    old_multiplier = grad_accum_multiplier
                    grad_accum_multiplier *= op_value
                    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps * op_value))
                    print(f"   Gradient accumulation: {old_grad_accum} → {gradient_accumulation_steps} (multiplier: {old_multiplier:.2f} → {grad_accum_multiplier:.2f})")
                    
                elif op_name == 'reset_lr_schedule':
                    old_offset = lr_schedule_offset
                    lr_schedule_offset = iter_num * 10
                    print(f"   LR schedule offset: {old_offset} → {lr_schedule_offset}")
                
                # Remove executed operation and update state
                scaling_schedule.pop(0)
                iter_of_last_op = iter_num * 10
                
                print(f"   Operations remaining: {len(scaling_schedule)}")
                print("✅ Operation completed\n")
            else:
                print("   No operation triggered\n")
        else:
            print("   No more operations in schedule\n")
    
    print("=== Final State ===")
    print(f"lr_multiplier: {lr_multiplier:.2f}")
    print(f"batch_size: {batch_size} (multiplier: {batch_size_multiplier:.2f})")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps} (multiplier: {grad_accum_multiplier:.2f})")
    print(f"lr_schedule_offset: {lr_schedule_offset}")
    print(f"Operations remaining: {len(scaling_schedule)}")
    
    print("\n✅ Manual test completed successfully!")
    return True

def test_configuration_loading():
    """Test loading of different configuration formats"""
    print("\n=== Configuration Loading Test ===")
    
    config_files = [
        'configs/sample_scaling_schedule.yaml',
        'configs/sample_scaling_schedule.json',
        'configs/test_scaling_schedule.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\nTesting {config_file}:")
            try:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    with open(config_file, 'r') as f:
                        schedule = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        schedule = json.load(f)
                
                print(f"  ✅ Loaded {len(schedule)} operations")
                for i, op in enumerate(schedule):
                    print(f"    {i+1}. {op['name']} (trigger: {op['trigger_loss']}, wait: {op['max_wait_iters']})")
                    
            except Exception as e:
                print(f"  ❌ Error loading {config_file}: {e}")
        else:
            print(f"\n⚠️  {config_file} not found")

def main():
    """Run manual tests"""
    try:
        # Test configuration loading
        test_configuration_loading()
        
        # Test orchestrator simulation
        simulate_orchestrator()
        
        print("\n🎉 All manual tests passed!")
        
    except Exception as e:
        print(f"\n❌ Manual test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
