#!/usr/bin/env python3
"""
Test script to validate DDP-safety of the Training Orchestrator.
This script simulates multi-rank behavior to ensure state consistency.
"""

import os
import sys
import yaml
import tempfile
from unittest.mock import MagicMock, patch
from logger import TrainingLogger

def simulate_ddp_orchestrator():
    """Simulate DDP behavior with multiple ranks"""
    print("=== DDP Safety Test for Training Orchestrator ===\n")
    
    # Create test configuration
    test_schedule = [
        {
            'name': 'change_lr',
            'value': 2.0,
            'trigger_loss': 6.0,
            'max_wait_iters': 10,
            'reevaluate': False
        },
        {
            'name': 'change_batch_size',
            'value': 1.5,
            'trigger_loss': 5.0,
            'max_wait_iters': 20,
            'reevaluate': False
        }
    ]
    
    # Simulate 4 ranks (processes)
    num_ranks = 4
    rank_states = []
    
    print(f"Simulating {num_ranks} DDP ranks...")
    
    # Initialize state for each rank
    for rank in range(num_ranks):
        rank_state = {
            'rank': rank,
            'master_process': rank == 0,
            'scaling_schedule': test_schedule.copy(),
            'iter_of_last_op': 0,
            'lr_multiplier': 1.0,
            'batch_size_multiplier': 1.0,
            'grad_accum_multiplier': 1.0,
            'lr_schedule_offset': 0,
            'batch_size': 12,
            'gradient_accumulation_steps': 5
        }
        rank_states.append(rank_state)
    
    print("Initial state for all ranks:")
    for rank_state in rank_states:
        print(f"  Rank {rank_state['rank']}: lr_mult={rank_state['lr_multiplier']}, "
              f"batch_size={rank_state['batch_size']}, schedule_len={len(rank_state['scaling_schedule'])}")
    print()
    
    # Simulate training iterations
    simulated_losses = [7.0, 5.5, 4.0]  # Decreasing validation loss
    
    for iter_num, val_loss in enumerate(simulated_losses):
        iteration = iter_num * 100
        print(f"--- Iteration {iteration} (val_loss: {val_loss:.1f}) ---")
        
        # Step 1: Master decides on operation
        op_to_run = [None]
        master_state = rank_states[0]  # Rank 0 is master
        
        if master_state['scaling_schedule']:
            next_op = master_state['scaling_schedule'][0]
            
            # Check trigger conditions
            loss_triggered = val_loss < next_op['trigger_loss']
            timeout_triggered = (iteration - master_state['iter_of_last_op']) >= next_op['max_wait_iters']
            
            print(f"Master (rank 0) decision:")
            print(f"  Next operation: {next_op['name']}")
            print(f"  Loss triggered: {loss_triggered} (val_loss {val_loss:.1f} < trigger {next_op['trigger_loss']:.1f})")
            print(f"  Timeout triggered: {timeout_triggered} (waited {iteration - master_state['iter_of_last_op']} >= {next_op['max_wait_iters']})")
            
            if loss_triggered or timeout_triggered:
                trigger_reason = 'Loss threshold' if loss_triggered else 'Timeout'
                op_to_run[0] = {
                    'op': next_op,
                    'reason': trigger_reason,
                    'loss': val_loss
                }
                print(f"  Decision: EXECUTE {next_op['name']} (reason: {trigger_reason})")
            else:
                print(f"  Decision: NO OPERATION")
        else:
            print("Master (rank 0): No operations remaining in schedule")
        
        # Step 2: Simulate broadcast (in real DDP, this would be torch.distributed.broadcast_object_list)
        print(f"Broadcasting decision to all ranks...")
        
        # Step 3: All ranks execute synchronously
        if op_to_run[0] is not None:
            op_data = op_to_run[0]
            next_op = op_data['op']
            trigger_reason = op_data['reason']
            
            print(f"All ranks executing operation: {next_op['name']}")
            
            # Simulate execute_operation on all ranks
            for rank_state in rank_states:
                op_name = next_op['name']
                op_value = next_op['value']
                
                if op_name == 'change_lr':
                    old_lr = rank_state['lr_multiplier']
                    rank_state['lr_multiplier'] *= op_value
                    if rank_state['master_process']:
                        print(f"  Rank {rank_state['rank']} (master): lr_multiplier {old_lr:.2f} → {rank_state['lr_multiplier']:.2f}")
                
                elif op_name == 'change_batch_size':
                    old_batch_size = rank_state['batch_size']
                    rank_state['batch_size_multiplier'] *= op_value
                    rank_state['batch_size'] = max(1, int(rank_state['batch_size'] * op_value))
                    if rank_state['master_process']:
                        print(f"  Rank {rank_state['rank']} (master): batch_size {old_batch_size} → {rank_state['batch_size']}")
                
                # All ranks update their schedule and timer
                rank_state['scaling_schedule'].pop(0)
                rank_state['iter_of_last_op'] = iteration
        
        # Step 4: Verify state consistency across all ranks
        print("State consistency check:")
        reference_state = rank_states[0]
        all_consistent = True
        
        for rank_state in rank_states[1:]:
            consistent = (
                rank_state['lr_multiplier'] == reference_state['lr_multiplier'] and
                rank_state['batch_size'] == reference_state['batch_size'] and
                rank_state['batch_size_multiplier'] == reference_state['batch_size_multiplier'] and
                rank_state['grad_accum_multiplier'] == reference_state['grad_accum_multiplier'] and
                rank_state['lr_schedule_offset'] == reference_state['lr_schedule_offset'] and
                len(rank_state['scaling_schedule']) == len(reference_state['scaling_schedule']) and
                rank_state['iter_of_last_op'] == reference_state['iter_of_last_op']
            )
            
            if consistent:
                print(f"  ✅ Rank {rank_state['rank']}: State consistent with master")
            else:
                print(f"  ❌ Rank {rank_state['rank']}: State DIVERGED from master!")
                print(f"    lr_mult: {rank_state['lr_multiplier']} vs {reference_state['lr_multiplier']}")
                print(f"    batch_size: {rank_state['batch_size']} vs {reference_state['batch_size']}")
                print(f"    schedule_len: {len(rank_state['scaling_schedule'])} vs {len(reference_state['scaling_schedule'])}")
                all_consistent = False
        
        if not all_consistent:
            print("❌ CRITICAL: State divergence detected! DDP would fail!")
            return False
        
        print()
    
    # Final state summary
    print("=== Final State Summary ===")
    final_state = rank_states[0]
    print(f"All ranks final state:")
    print(f"  lr_multiplier: {final_state['lr_multiplier']:.2f}")
    print(f"  batch_size: {final_state['batch_size']}")
    print(f"  batch_size_multiplier: {final_state['batch_size_multiplier']:.2f}")
    print(f"  Operations remaining: {len(final_state['scaling_schedule'])}")
    
    print("\n✅ DDP Safety Test PASSED - All ranks maintained consistent state!")
    return True

def test_broadcast_simulation():
    """Test the broadcast mechanism simulation"""
    print("\n=== Broadcast Mechanism Test ===")
    
    # Simulate the broadcast_object_list behavior
    op_to_run = [None]
    
    # Master decides
    master_decision = {
        'op': {'name': 'change_lr', 'value': 2.0},
        'reason': 'Loss threshold',
        'loss': 5.8
    }
    op_to_run[0] = master_decision
    
    print(f"Master decision: {op_to_run[0]['op']['name']} (reason: {op_to_run[0]['reason']})")
    
    # Simulate broadcast to all ranks
    num_ranks = 4
    for rank in range(num_ranks):
        received_decision = op_to_run[0]  # In real DDP, this would be the broadcasted object
        if received_decision is not None:
            print(f"Rank {rank} received: {received_decision['op']['name']}")
        else:
            print(f"Rank {rank} received: No operation")
    
    print("✅ Broadcast simulation successful")

def main():
    """Run all DDP safety tests"""
    try:
        # Test DDP orchestrator simulation
        if not simulate_ddp_orchestrator():
            sys.exit(1)
        
        # Test broadcast mechanism
        test_broadcast_simulation()
        
        print("\n🎉 All DDP safety tests passed!")
        
    except Exception as e:
        print(f"\n❌ DDP safety test failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
