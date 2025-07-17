#!/usr/bin/env python3
"""
Prepare reward model training data by generating mixed natural/synthetic sequences.

This script creates a dataset for training the reward model by:
1. Loading a pre-trained base model
2. Using the same train/val split as the base model to prevent data contamination
3. For each sequence, choosing a random crossover point K
4. Taking first K tokens from natural data, remaining from model generation
5. Creating target labels [K/N, (N-K)/N] for the reward model

Enhanced with configurable tokenization support and binary file reuse capabilities.

Usage:
    # Text mode with BPE tokenization (default)
    python prepare_reward_data.py --model_path checkpoints/base_model_v1.pt --data_path data/shakespeare/input.txt

    # Text mode with character tokenization
    python prepare_reward_data.py --model_path checkpoints/char_model.pt --input_mode text --data_path data/shakespeare_char/input.txt --tokenization char --meta_path data/shakespeare_char/meta.pkl

    # Binary mode (reuse existing train.bin/val.bin)
    python prepare_reward_data.py --model_path checkpoints/base_model.pt --input_mode binary --train_bin data/shakespeare/train.bin --val_bin data/shakespeare/val.bin
"""

import os
import argparse
import logging
import numpy as np
import torch
import tiktoken
from tqdm import tqdm
import random

from model import GPT, GPTConfig
from tokenization_manager import TokenizationManager, TokenizationError
from data_loader import DataLoader, DataLoadError
from reward_data_config import RewardDataConfig, ConfigurationValidator, RewardDataPrepError, TokenizationInfo


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


def load_data_with_config(config: RewardDataConfig) -> tuple:
    """
    Load data using the new configurable system.

    Args:
        config: Validated RewardDataConfig instance

    Returns:
        Tuple of (train_tokens, val_tokens, tokenization_manager)
    """
    # Initialize tokenization manager
    if config.tokenization == 'auto':
        # Auto-detect based on data path or meta path
        if config.input_mode == 'text':
            tokenization_manager = TokenizationManager(data_path=config.data_path)
        else:
            # For binary mode, try to detect from directory structure
            data_dir = os.path.dirname(config.train_bin) if config.train_bin else None
            tokenization_manager = TokenizationManager(data_path=data_dir, meta_path=config.meta_path)
    elif config.tokenization == 'char':
        tokenization_manager = TokenizationManager(meta_path=config.meta_path)
    else:  # bpe
        tokenization_manager = TokenizationManager()
        tokenization_manager.load_bpe_tokenization()

    # Initialize data loader
    data_loader = DataLoader(tokenization_manager)

    # Load data based on input mode
    if config.input_mode == 'text':
        train_tokens, val_tokens = data_loader.load_from_text(config.data_path, config.train_split)
    else:  # binary
        train_tokens, val_tokens = data_loader.load_from_binary(config.train_bin, config.val_bin)

    # Log data information
    data_info = data_loader.get_data_info(train_tokens, val_tokens)
    logging.info(f"Data loaded successfully:")
    logging.info(f"  Tokenization: {data_info['tokenization_method']}")
    logging.info(f"  Vocab size: {data_info['vocab_size']}")
    logging.info(f"  Train tokens: {data_info['train_size']:,}")
    logging.info(f"  Val tokens: {data_info['val_size']:,}")

    return train_tokens, val_tokens, tokenization_manager


def load_and_split_data(data_path, train_split=0.9):
    """
    Legacy function for backward compatibility.
    Load raw text data and split it using BPE tokenization.
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


def save_reward_dataset(samples_x, samples_y, output_dir, split_name, tokenization_info=None):
    """Save the reward dataset to binary files with optional tokenization metadata."""
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

    # Add tokenization information if provided
    if tokenization_info:
        metadata.update({
            'tokenization_method': tokenization_info.method,
            'vocab_size': tokenization_info.vocab_size,
            'meta_path': tokenization_info.meta_path or 'None'
        })

    metadata_path = os.path.join(output_dir, f'{split_name}_metadata.txt')
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare reward model training data with configurable tokenization')

    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained base model checkpoint')

    # Input mode parameters
    parser.add_argument('--input_mode', choices=['text', 'binary'], default='text',
                        help='Input mode: text (raw text file) or binary (existing .bin files)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the raw text data file (text mode only)')
    parser.add_argument('--train_bin', type=str, default=None,
                        help='Path to existing train.bin file (binary mode only)')
    parser.add_argument('--val_bin', type=str, default=None,
                        help='Path to existing val.bin file (binary mode only)')

    # Tokenization configuration
    parser.add_argument('--tokenization', choices=['auto', 'bpe', 'char'], default='auto',
                        help='Tokenization method (auto-detect, bpe, or char)')
    parser.add_argument('--meta_path', type=str, default=None,
                        help='Path to meta.pkl file for character tokenization')

    # Generation parameters (existing)
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

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Handle backward compatibility - if only data_path is provided, use text mode
    if args.data_path and not args.train_bin and not args.val_bin:
        if args.input_mode == 'binary':
            print("Warning: data_path provided but input_mode is binary. Switching to text mode.")
            args.input_mode = 'text'

    # Create configuration object
    config_obj = RewardDataConfig(
        model_path=args.model_path,
        input_mode=args.input_mode,
        data_path=args.data_path,
        train_bin=args.train_bin,
        val_bin=args.val_bin,
        tokenization=args.tokenization,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        samples_per_chunk=args.samples_per_chunk,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )

    # Validate configuration
    validator = ConfigurationValidator()
    if not validator.validate_config(config_obj):
        validator.print_validation_results(config_obj)
        suggestions = validator.suggest_fixes(config_obj)
        if suggestions:
            print("\nSuggested fixes:")
            for suggestion in suggestions:
                print(f"  SUGGESTION: {suggestion}")
        return 1

    # Print any warnings
    if config_obj.get_warnings():
        validator.print_validation_results(config_obj)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    if config_obj.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config_obj.device
    print(f"Using device: {device}")

    try:
        # Load base model
        model, model_config = load_base_model(config_obj.model_path, device)

        # Load data using new configurable system
        train_tokens, val_tokens, tokenization_manager = load_data_with_config(config_obj)

        # Create tokenization info for metadata
        tokenization_info = TokenizationInfo(
            method=tokenization_manager.tokenization_type,
            vocab_size=tokenization_manager.vocab_size,
            meta_path=tokenization_manager.meta_path
        )

        print(f"\nGenerating reward dataset with block_size={model_config.block_size}")
        print(f"Tokenization method: {tokenization_info.method}")
        print(f"Vocab size: {tokenization_info.vocab_size}")
        print(f"Samples per chunk: {config_obj.samples_per_chunk}")
        print(f"Temperature: {config_obj.temperature}")
        if config_obj.top_k:
            print(f"Top-k: {config_obj.top_k}")

        # Generate training samples from training data
        print("\n=== Generating Training Samples ===")
        train_x, train_y = create_reward_samples(
            train_tokens, model, model_config, device,
            config_obj.samples_per_chunk, config_obj.temperature
        )

        # Generate validation samples from validation data
        print("\n=== Generating Validation Samples ===")
        val_x, val_y = create_reward_samples(
            val_tokens, model, model_config, device,
            config_obj.samples_per_chunk, config_obj.temperature
        )

        # Save datasets with tokenization metadata
        print(f"\n=== Saving to {config_obj.output_dir} ===")
        save_reward_dataset(train_x, train_y, config_obj.output_dir, 'train', tokenization_info)
        save_reward_dataset(val_x, val_y, config_obj.output_dir, 'val', tokenization_info)

        print("\n=== Dataset Creation Complete ===")
        print(f"Training samples: {len(train_x)}")
        print(f"Validation samples: {len(val_x)}")
        print(f"Block size: {model_config.block_size}")
        print(f"Output directory: {config_obj.output_dir}")
        print(f"Tokenization: {tokenization_info.method} (vocab_size={tokenization_info.vocab_size})")

        return 0

    except (RewardDataPrepError, TokenizationError, DataLoadError) as e:
        print(f"\nERROR: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        logging.exception("Unexpected error occurred")
        return 1


if __name__ == '__main__':
    main()