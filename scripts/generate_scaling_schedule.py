#!/usr/bin/env python3
"""
Generate scaling schedule JSON for layer growth testing.

This script creates a schedule that:
1. Starts with 1 layer, grows to target_layers
2. Trains for iters_per_layer iterations per layer
3. First freeze_iters after adding new layer: freeze all older layers and embeddings
4. After freeze_iters: set previous layer to LoRA 1/4, then 1/16
5. Runs merge_lora_weights before each growth
6. Adds extra_iters after final layer
"""

import json
import argparse
from typing import List, Dict, Any


def create_operation(name: str, value: Any, max_wait_iters: int, desc: str) -> Dict[str, Any]:
    """Create a standardized operation entry."""
    # merge_lora_weights should always have reevaluate: true
    reevaluate = True if name == "merge_lora_weights" else False
    
    return {
        "name": name,
        "value": value,
        "trigger_loss": 0,  # Always use time-based triggers
        "max_wait_iters": max_wait_iters,
        "desc": desc,
        "reevaluate": reevaluate,
        "completed": False
    }


def generate_scaling_schedule(
    target_layers: int = 6,
    iters_per_layer: int = 500,
    freeze_iters: int = 200,
    n_embd: int = 384,
    extra_iters: int = 500
) -> List[Dict[str, Any]]:
    """Generate the complete scaling schedule."""
    
    schedule = []
    
    # Step 1: Train first layer unfreezed for iters_per_layer iterations
    schedule.append(create_operation(
        "merge_lora_weights", None, iters_per_layer,
        f"Train first layer unfreezed for {iters_per_layer} iterations"
    ))
    
    # Step 2: Add layers 2 through target_layers
    for layer_num in range(2, target_layers + 1):
        current_iter = (layer_num - 1) * iters_per_layer
        
        # Create stack_layers array using simple rule:
        # Start with [0, 0], then for each new layer:
        # 1) add +1 to last element 2) add new element equal to last element
        # [0, 0] -> [0, 1, 1] -> [0, 1, 2, 2] -> [0, 1, 2, 3, 3] etc.
        if layer_num == 2:
            stack_config = [0, 0]
        else:
            # Get previous config and apply the rule
            prev_config = [0, 0]
            for i in range(3, layer_num + 1):
                # Rule: increment last element, then duplicate it
                prev_config[-1] += 1
                prev_config.append(prev_config[-1])
            stack_config = prev_config
        
        # Add the new layer
        schedule.append(create_operation(
            "stack_layers", stack_config, 1,
            f"Add layer {layer_num} after {current_iter} iterations total"
        ))
        
        # Freeze all previous layers (both attn and mlp) + embeddings
        for prev_layer in range(layer_num - 1):  # 0 to layer_num-2
            schedule.append(create_operation(
                "freeze_layer", f"attn.{prev_layer}", 1,
                f"Freeze layer {prev_layer} attention when adding layer {layer_num}"
            ))
            schedule.append(create_operation(
                "freeze_layer", f"mlp.{prev_layer}", 1,
                f"Freeze layer {prev_layer} MLP when adding layer {layer_num}"
            ))
        
        # Freeze embeddings
        schedule.append(create_operation(
            "freeze_layer", "wte", 1,
            f"Freeze embeddings when adding layer {layer_num}"
        ))
        
        # Wait freeze_iters iterations, then unfreeze all previously frozen layers
        # First unfreeze operation waits 200 iterations, rest are immediate
        first_unfreeze = True
        
        # Unfreeze all previous layers (both attn and mlp)
        for prev_layer in range(layer_num - 1):  # 0 to layer_num-2
            wait_time = freeze_iters if first_unfreeze else 1
            schedule.append(create_operation(
                "unfreeze_layer", f"attn.{prev_layer}", wait_time,
                f"Unfreeze layer {prev_layer} attention after {freeze_iters} iterations"
            ))
            first_unfreeze = False  # Only first operation waits
            
            schedule.append(create_operation(
                "unfreeze_layer", f"mlp.{prev_layer}", 1,
                f"Unfreeze layer {prev_layer} MLP after {freeze_iters} iterations"
            ))
        
        # Unfreeze embeddings
        schedule.append(create_operation(
            "unfreeze_layer", "wte", 1,
            f"Unfreeze embeddings after {freeze_iters} iterations"
        ))
        
        # Set LoRA ranks: most recent previous layer to 1/4, layer before that to 1/16
        most_recent_prev = layer_num - 2  # The layer we just finished training
        if most_recent_prev >= 0:
            rank_quarter = n_embd // 4  # 1/4 of embedding dim
            schedule.append(create_operation(
                "set_layer_lora_rank", [f"attn.{most_recent_prev}", rank_quarter], 1,
                f"Set layer {most_recent_prev} attention to LoRA rank 1/4 of {n_embd}"
            ))
            
            # Set layer before that to 1/16 if it exists
            if most_recent_prev > 0:
                prev_prev = most_recent_prev - 1
                rank_sixteenth = n_embd // 16  # 1/16 of embedding dim
                schedule.append(create_operation(
                    "set_layer_lora_rank", [f"attn.{prev_prev}", rank_sixteenth], 1,
                    f"Set layer {prev_prev} attention to LoRA rank 1/16 of {n_embd}"
                ))
        
        # Wait remaining iterations to complete iters_per_layer total, then merge LoRA
        remaining_iters = iters_per_layer - freeze_iters - 2  # -2 for the operations above
        if layer_num < target_layers:  # Don't merge after final layer
            schedule.append(create_operation(
                "merge_lora_weights", None, remaining_iters,
                f"Merge LoRA weights before next growth (at iter {current_iter + iters_per_layer})"
            ))
    
    # Add extra training iterations after final layer
    if extra_iters > 0:
        schedule.append(create_operation(
            "merge_lora_weights", None, extra_iters,
            f"Extra training after final layer for {extra_iters} iterations"
        ))
    
    return schedule


def main():
    parser = argparse.ArgumentParser(description="Generate scaling schedule JSON")
    parser.add_argument("--target-layers", type=int, default=6, 
                       help="Target number of layers (default: 6)")
    parser.add_argument("--iters-per-layer", type=int, default=500,
                       help="Iterations per layer (default: 500)")  
    parser.add_argument("--freeze-iters", type=int, default=200,
                       help="Iterations to freeze after adding layer (default: 200)")
    parser.add_argument("--n-embd", type=int, default=384,
                       help="Embedding dimension for LoRA rank calculation (default: 384)")
    parser.add_argument("--extra-iters", type=int, default=500,
                       help="Extra iterations after final layer (default: 500)")
    parser.add_argument("--output", type=str, 
                       default="configs/test_basic_shrunken_schedule.json",
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Generate the schedule
    schedule = generate_scaling_schedule(
        target_layers=args.target_layers,
        iters_per_layer=args.iters_per_layer, 
        freeze_iters=args.freeze_iters,
        n_embd=args.n_embd,
        extra_iters=args.extra_iters
    )
    
    # Write to file
    with open(args.output, 'w') as f:
        json.dump(schedule, f, indent=2)
    
    # Print summary
    total_iters = args.target_layers * args.iters_per_layer + args.extra_iters
    print(f"Generated scaling schedule:")
    print(f"  Target layers: {args.target_layers}")
    print(f"  Iterations per layer: {args.iters_per_layer}")
    print(f"  Freeze iterations: {args.freeze_iters}")
    print(f"  Total iterations: {total_iters}")
    print(f"  Operations: {len(schedule)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()