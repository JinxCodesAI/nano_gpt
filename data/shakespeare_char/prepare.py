"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save pre-computed batch files containing x,y tensors with metadata,
and meta.pkl containing the encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import argparse
import torch

def generate_batches(data_ids, batch_size, block_size, target_size, batches_per_file, total_batches, output_dir, split_name):
    """Generate pre-computed batch files with x,y tensors and metadata."""

    # Calculate how many files we need
    num_files = (total_batches + batches_per_file - 1) // batches_per_file

    print(f"Generating {num_files} {split_name} files with {batches_per_file} batches each")
    print(f"Total batches: {total_batches}")

    batch_idx = 0
    for file_idx in range(num_files):
        # Calculate batches for this file
        remaining_batches = total_batches - batch_idx
        current_file_batches = min(batches_per_file, remaining_batches)

        # Generate batches for this file
        x_batches = []
        y_batches = []

        for _ in range(current_file_batches):
            # Generate random indices for this batch (circular reading)
            max_start_idx = len(data_ids) - block_size
            ix = torch.randint(max_start_idx, (batch_size,))

            # Create x and y tensors
            x = torch.stack([torch.from_numpy((data_ids[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data_ids[i+1:i+1+target_size]).astype(np.int64)) for i in ix])

            x_batches.append(x)
            y_batches.append(y)

        # Combine all batches in this file
        file_x = torch.cat(x_batches, dim=0)
        file_y = torch.cat(y_batches, dim=0)

        # Create metadata
        metadata = {
            'batch_size': batch_size,
            'block_size': block_size,
            'target_size': target_size,
            'num_batches': current_file_batches,
            'file_idx': file_idx,
            'split': split_name
        }

        # Save the file
        filename = f'{split_name}_batches_{file_idx:04d}.pt'
        filepath = os.path.join(output_dir, filename)

        torch.save({
            'x': file_x,
            'y': file_y,
            'metadata': metadata
        }, filepath)

        print(f"Saved {filepath} with {current_file_batches} batches")
        batch_idx += current_file_batches

def main():
    parser = argparse.ArgumentParser(description='Prepare Shakespeare dataset with batch generation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=256, help='Block size (input sequence length)')
    parser.add_argument('--target_size', type=int, default=None, help='Target size (defaults to block_size)')
    parser.add_argument('--batches_per_file', type=int, default=100, help='Number of batches per batch file')
    parser.add_argument('--total_batches', type=int, default=1000, help='Total number of batches to generate')
    parser.add_argument('--force_regenerate', action='store_true', help='Force regeneration even if files exist')

    args = parser.parse_args()

    # Set target_size to block_size if not specified
    if args.target_size is None:
        args.target_size = args.block_size

    output_dir = os.path.dirname(__file__)

    # Check if batch files already exist and we're not forcing regeneration
    train_batch_pattern = os.path.join(output_dir, 'train_batches_*.pt')
    val_batch_pattern = os.path.join(output_dir, 'val_batches_*.pt')

    import glob
    existing_train_files = glob.glob(train_batch_pattern)
    existing_val_files = glob.glob(val_batch_pattern)

    if existing_train_files and existing_val_files and not args.force_regenerate:
        print("Batch files already exist. Use --force_regenerate to recreate them.")
        print(f"Found {len(existing_train_files)} train files and {len(existing_val_files)} val files")
        return

    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Convert to numpy arrays
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    # Generate batch files
    print(f"\nGenerating batch files with:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  block_size: {args.block_size}")
    print(f"  target_size: {args.target_size}")
    print(f"  batches_per_file: {args.batches_per_file}")
    print(f"  total_batches: {args.total_batches}")

    # Calculate validation batches (10% of total)
    val_batches = max(1, args.total_batches // 10)
    train_batches = args.total_batches

    print(f"  train_batches: {train_batches}")
    print(f"  val_batches: {val_batches}")

    # Generate training batches
    generate_batches(train_ids, args.batch_size, args.block_size, args.target_size,
                    args.batches_per_file, train_batches, output_dir, 'train')

    # Generate validation batches
    generate_batches(val_ids, args.batch_size, args.block_size, args.target_size,
                    args.batches_per_file, val_batches, output_dir, 'val')

    # Also save the legacy bin files for compatibility
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'batch_size': args.batch_size,
        'block_size': args.block_size,
        'target_size': args.target_size,
        'batches_per_file': args.batches_per_file,
        'total_batches': args.total_batches,
        'train_batches': train_batches,
        'val_batches': val_batches,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nDataset preparation complete!")
    print(f"Generated batch files and saved metadata to meta.pkl")

if __name__ == '__main__':
    main()
