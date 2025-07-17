#!/usr/bin/env python3
"""
Prepare reward model training data by generating mixed natural/synthetic sequences.

This script creates a dataset for training the reward model by:
1. Loading a pre-trained base model
2. Using the same train/val split as the base model to prevent data contamination
3. For each sequence, choosing a random crossover point K
4. Taking first K tokens from natural data, remaining from model generation
5. Creating target labels [K/N, (N-K)/N] for the reward model

Usage:
    python prepare_reward_data.py --model_path checkpoints/base_model_v1.pt --data_path data/shakespeare/input.txt
"""

import os
import argparse
import numpy as np
import torch
import tiktoken
from tqdm import tqdm
import random

from model import GPT, GPTConfig


def load_base_model(model_path, device):
    """Load the pre-trained base model for generating synthetic text."""
    print(f"Loading base model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config and create model
    config = checkpoint['config']
    # Ensure we're in generator mode
    config.mode = 'generator'
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model with {model.get_num_params()/1e6:.2f}M parameters")
    return model, config


def load_and_split_data(data_path, train_split=0.9):
    """
    Load raw text data and split it using the same ratio as the base model.
    This ensures perfect alignment with base model's train/val splits.
    """
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Use tiktoken GPT-2 BPE encoding (same as base model)
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(data)
    
    # Split using the same ratio as base model (90/10)
    n = len(tokens)
    split_idx = int(n * train_split)
    
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    return train_tokens, val_tokens, enc


def generate_completion(model, prompt_tokens, target_length, device, temperature=1.0, top_k=None):
    """Generate completion for a given prompt using the base model."""
    model.eval()
    
    with torch.no_grad():
        # Convert to tensor
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate completion
        completion_length = target_length - len(prompt_tokens)
        if completion_length <= 0:
            return prompt_tokens
        
        # Use the model's generate method
        generated = model.generate(
            prompt_tensor, 
            max_new_tokens=completion_length,
            temperature=temperature,
            top_k=top_k
        )
        
        # Return the full sequence as a list
        return generated[0].cpu().tolist()


def create_reward_samples(tokens, model, config, device, num_samples_per_chunk=100, temperature=1.0):
    """
    Create reward model training samples from a token sequence.
    
    For each sample:
    1. Choose a random crossover point K in [1, block_size-1]
    2. Take first K tokens from natural data
    3. Generate remaining (block_size - K) tokens using the model
    4. Create target label [K/block_size, (block_size-K)/block_size]
    """
    block_size = config.block_size
    samples_x = []  # Input sequences
    samples_y = []  # Target probability distributions
    
    # Calculate how many complete blocks we can create
    num_blocks = len(tokens) // block_size
    print(f"Creating samples from {num_blocks} complete blocks")
    
    for block_idx in tqdm(range(num_blocks), desc="Processing blocks"):
        # Extract a complete block from natural data
        start_idx = block_idx * block_size
        natural_block = tokens[start_idx:start_idx + block_size]
        
        # Create multiple samples per block with different crossover points
        samples_this_block = min(num_samples_per_chunk, block_size - 2)  # Leave room for K in [1, block_size-1]
        
        for _ in range(samples_this_block):
            # Choose random crossover point K (at least 1, at most block_size-1)
            K = random.randint(1, block_size - 1)
            
            # Take first K tokens from natural data
            natural_prefix = natural_block[:K]
            
            # Generate completion using the model
            try:
                mixed_sequence = generate_completion(
                    model, 
                    natural_prefix, 
                    block_size, 
                    device, 
                    temperature=temperature
                )
                
                # Ensure we have exactly block_size tokens
                if len(mixed_sequence) != block_size:
                    # Truncate or pad as needed
                    if len(mixed_sequence) > block_size:
                        mixed_sequence = mixed_sequence[:block_size]
                    else:
                        # Pad with the last token if too short
                        while len(mixed_sequence) < block_size:
                            mixed_sequence.append(mixed_sequence[-1])
                
                # Create target probability distribution [P(natural), P(synthetic)]
                p_natural = K / block_size
                p_synthetic = (block_size - K) / block_size
                target_probs = [p_natural, p_synthetic]
                
                samples_x.append(mixed_sequence)
                samples_y.append(target_probs)
                
            except Exception as e:
                print(f"Warning: Failed to generate sample at block {block_idx}, K={K}: {e}")
                continue
    
    print(f"Created {len(samples_x)} reward model samples")
    return samples_x, samples_y


def save_reward_dataset(samples_x, samples_y, output_dir, split_name):
    """Save the reward dataset to binary files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy arrays
    x_array = np.array(samples_x, dtype=np.uint16)  # Token IDs
    y_array = np.array(samples_y, dtype=np.float32)  # Probability distributions
    
    # Save to binary files
    x_path = os.path.join(output_dir, f'{split_name}_x.bin')
    y_path = os.path.join(output_dir, f'{split_name}_y.bin')
    
    x_array.tofile(x_path)
    y_array.tofile(y_path)
    
    print(f"Saved {split_name} data:")
    print(f"  X: {x_path} ({x_array.shape})")
    print(f"  Y: {y_path} ({y_array.shape})")
    
    # Save metadata
    metadata = {
        'num_samples': len(samples_x),
        'block_size': len(samples_x[0]) if samples_x else 0,
        'x_shape': x_array.shape,
        'y_shape': y_array.shape,
        'x_dtype': str(x_array.dtype),
        'y_dtype': str(y_array.dtype)
    }
    
    metadata_path = os.path.join(output_dir, f'{split_name}_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare reward model training data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained base model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/shakespeare/input.txt',
                        help='Path to the raw text data file')
    parser.add_argument('--output_dir', type=str, default='data/reward_dataset',
                        help='Output directory for reward dataset')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/validation split ratio (default: 0.9)')
    parser.add_argument('--samples_per_chunk', type=int, default=10,
                        help='Number of samples to generate per data chunk')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for text generation')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling for text generation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load base model
    model, config = load_base_model(args.model_path, device)
    
    # Load and split data (using same split as base model)
    train_tokens, val_tokens, tokenizer = load_and_split_data(args.data_path, args.train_split)
    
    print(f"\nGenerating reward dataset with block_size={config.block_size}")
    print(f"Samples per chunk: {args.samples_per_chunk}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    
    # Generate training samples from training data
    print("\n=== Generating Training Samples ===")
    train_x, train_y = create_reward_samples(
        train_tokens, model, config, device, 
        args.samples_per_chunk, args.temperature
    )
    
    # Generate validation samples from validation data
    print("\n=== Generating Validation Samples ===")
    val_x, val_y = create_reward_samples(
        val_tokens, model, config, device,
        args.samples_per_chunk, args.temperature
    )
    
    # Save datasets
    print(f"\n=== Saving to {args.output_dir} ===")
    save_reward_dataset(train_x, train_y, args.output_dir, 'train')
    save_reward_dataset(val_x, val_y, args.output_dir, 'val')
    
    print("\n=== Dataset Creation Complete ===")
    print(f"Training samples: {len(train_x)}")
    print(f"Validation samples: {len(val_x)}")
    print(f"Block size: {config.block_size}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()