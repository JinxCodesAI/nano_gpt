#!/usr/bin/env python3
"""
Debug script to understand parameter counting with weight tying.
"""

import torch
import torch.nn as nn
from model import GPT, GPTConfig

def debug_parameter_counting():
    """Debug how PyTorch counts parameters with weight tying"""
    print("=" * 60)
    print("Debugging Parameter Counting with Weight Tying")
    print("=" * 60)
    
    # Create a small test model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=1,  # Single layer for simplicity
        n_head=2,
        n_embd=64,  # Smaller for easier counting
        dropout=0.0,
        bias=False,
        binary_classification=False  # Start as language model
    )
    
    model = GPT(config)
    
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Expected token embedding params: {config.vocab_size * config.n_embd}")
    print(f"Expected LM head params (if separate): {config.vocab_size * config.n_embd}")
    
    # Check initial state
    print("\n--- Initial State (Language Model with Weight Tying) ---")
    print(f"wte.weight shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head.weight shape: {model.lm_head.weight.shape}")
    print(f"Are they the same tensor? {model.transformer.wte.weight is model.lm_head.weight}")
    print(f"wte tensor ID: {id(model.transformer.wte.weight)}")
    print(f"lm_head tensor ID: {id(model.lm_head.weight)}")
    
    # Count parameters manually
    all_params = list(model.parameters())
    print(f"\nTotal parameter tensors: {len(all_params)}")
    
    total_manual = 0
    for i, param in enumerate(all_params):
        param_count = param.numel()
        total_manual += param_count
        print(f"  Param {i}: shape {param.shape}, count {param_count}")
    
    print(f"\nManual count: {total_manual}")
    print(f"get_num_params(): {model.get_num_params()}")
    print(f"get_trainable_param_count(): {model.get_trainable_param_count()}")
    
    # Switch to binary classification
    print("\n--- After Switch to Binary Classification ---")
    model.switch_to_binary_classification()
    
    print(f"wte.weight shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head.weight shape: {model.lm_head.weight.shape}")
    print(f"Are they the same tensor? {model.transformer.wte.weight is model.lm_head.weight}")
    print(f"wte tensor ID: {id(model.transformer.wte.weight)}")
    print(f"lm_head tensor ID: {id(model.lm_head.weight)}")
    
    # Count parameters manually again
    all_params = list(model.parameters())
    print(f"\nTotal parameter tensors: {len(all_params)}")
    
    total_manual = 0
    for i, param in enumerate(all_params):
        param_count = param.numel()
        total_manual += param_count
        print(f"  Param {i}: shape {param.shape}, count {param_count}")
    
    print(f"\nManual count: {total_manual}")
    print(f"get_num_params(): {model.get_num_params()}")
    print(f"get_trainable_param_count(): {model.get_trainable_param_count()}")
    
    # Switch back to language modeling
    print("\n--- After Switch Back to Language Modeling ---")
    model.switch_to_language_modeling(config.vocab_size)
    
    print(f"wte.weight shape: {model.transformer.wte.weight.shape}")
    print(f"lm_head.weight shape: {model.lm_head.weight.shape}")
    print(f"Are they the same tensor? {model.transformer.wte.weight is model.lm_head.weight}")
    print(f"wte tensor ID: {id(model.transformer.wte.weight)}")
    print(f"lm_head tensor ID: {id(model.lm_head.weight)}")
    
    # Count parameters manually again
    all_params = list(model.parameters())
    print(f"\nTotal parameter tensors: {len(all_params)}")
    
    total_manual = 0
    for i, param in enumerate(all_params):
        param_count = param.numel()
        total_manual += param_count
        print(f"  Param {i}: shape {param.shape}, count {param_count}")
    
    print(f"\nManual count: {total_manual}")
    print(f"get_num_params(): {model.get_num_params()}")
    print(f"get_trainable_param_count(): {model.get_trainable_param_count()}")

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducible results
    debug_parameter_counting()