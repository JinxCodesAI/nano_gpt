#!/usr/bin/env python3
"""
Script to inspect batch files and decode their content using dataset metadata.
Usage: python inspect_batch.py <dataset_name> <batch_file_path>
Example: python inspect_batch.py char_diffusion train/1757250083194-000013-100.pt
"""

import sys
import os
import pickle
import torch
from pathlib import Path
from core.batch import Batch, unpack_batch


def load_meta(data_dir):
    """Load dataset metadata."""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta


def decode_tokens(tokens, itos):
    """Decode token IDs to characters/strings."""
    return ''.join([itos.get(token, f'<UNK:{token}>') for token in tokens])


def analyze_batch_file(dataset_name, batch_file_path):
    """Analyze a batch file and show decoded content."""
    
    # Construct data directory path
    data_dir = os.path.join('data', dataset_name)
    
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {data_dir}")
    print(f"Batch file: {batch_file_path}")
    print("=" * 80)
    
    # Load metadata
    try:
        meta = load_meta(data_dir)
        print("METADATA:")
        for key, value in meta.items():
            if key in ['stoi', 'itos']:
                print(f"  {key}: <vocab mapping with {len(value)} entries>")
            elif isinstance(value, (list, dict)) and len(str(value)) > 100:
                print(f"  {key}: <{type(value).__name__} with {len(value)} items>")
            else:
                print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Extract vocab mappings
    itos = meta.get('itos', {})
    if isinstance(itos, dict):
        # Convert string keys to int if needed
        itos = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in itos.items()}
    
    stoi = meta.get('stoi', {})
    vocab_size = meta.get('vocab_size', len(itos))
    mask_token_id = meta.get('mask_token_id', None)
    ignore_index = meta.get('ignore_index', -100)
    
    print("VOCABULARY INFO:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  mask_token_id: {mask_token_id}")
    print(f"  ignore_index: {ignore_index}")
    if mask_token_id is not None and mask_token_id in itos:
        print(f"  mask_token: '{itos[mask_token_id]}'")
    print()
    
    # Load batch file
    full_batch_path = os.path.join(data_dir, 'queue', batch_file_path)
    if not os.path.exists(full_batch_path):
        print(f"Batch file not found: {full_batch_path}")
        return
    
    try:
        batch_data = torch.load(full_batch_path, map_location='cpu')
        print("BATCH FILE STRUCTURE:")
        if isinstance(batch_data, dict) and 'batches' in batch_data:
            print(f"  batches: list[{len(batch_data['batches'])}] of per-batch entries")
            print(f"  metadata: dict with keys {list(batch_data.get('metadata', {}).keys())}")
        else:
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.dtype} {list(value.shape)}")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value)} = {value}")
        print()
    except Exception as e:
        print(f"Error loading batch file: {e}")
        return

    # Extract tensors (handle new array-of-batches format first)
    if isinstance(batch_data, dict) and 'batches' in batch_data:
        batches = batch_data.get('batches', [])
        if not batches:
            print("Error: 'batches' list is empty")
            return
        entry = batches[0]
        if not isinstance(entry, dict) or 'tensors' not in entry:
            print("Error: batch entry missing 'tensors'")
            return
        x_tensor, y_tensor = unpack_batch(entry['tensors'])
        print(f"Using batch entry 0/{len(batches)-1}")
    else:
        # Legacy single-entry files
        if 'tensors' in batch_data:
            x_tensor, y_tensor = unpack_batch(batch_data['tensors'])
        else:
            x_tensor, y_tensor = unpack_batch(batch_data)

    if x_tensor is None or y_tensor is None:
        print("Error: Could not find 'x' or 'y' tensors in batch file")
        print("Available keys:", list(batch_data.keys()))
        if 'tensors' in batch_data:
            print("Tensors keys:", list(batch_data['tensors'].keys()))
        return
    
    batch_size, seq_len = x_tensor.shape
    print(f"BATCH DIMENSIONS:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print()
    
    # Analyze a few examples
    num_examples = min(100, 100)
    print(f"EXAMPLE ROWS (first {num_examples}):")
    print()
    
    for i in range(num_examples):
        x_tokens = x_tensor[i].tolist()
        y_tokens = y_tensor[i].tolist()
        
        # Decode tokens
        x_decoded = decode_tokens(x_tokens, itos)
        
        # For y tokens, show original tokens for masked positions, ignore_index elsewhere
        y_decoded_parts = []
        masked_positions = []
        for j, (x_tok, y_tok) in enumerate(zip(x_tokens, y_tokens)):
            if y_tok != ignore_index:
                y_decoded_parts.append(itos.get(y_tok, f'<UNK:{y_tok}>'))
                masked_positions.append(j)
            else:
                y_decoded_parts.append('_')
        
        print(f"Example {i+1}:")
        print(f"  Input (x):  {repr(x_decoded)}")
        print(f"  Target (y): {''.join(y_decoded_parts)}")
        print(f"  Masked positions: {masked_positions}")
        
        # Show token-by-token breakdown for first few positions
        print("  Token breakdown (first 20 positions):")
        for j in range(min(20, seq_len)):
            x_tok = x_tokens[j]
            y_tok = y_tokens[j]
            x_char = itos.get(x_tok, f'<UNK:{x_tok}>')
            y_char = itos.get(y_tok, f'<UNK:{y_tok}>') if y_tok != ignore_index else '<IGN>'
            mask_indicator = '*' if y_tok != ignore_index else ' '
            print(f"    {j:2d}: x={x_tok:3d}('{x_char}') y={y_tok:4d}('{y_char}') {mask_indicator}")
        print()
    
    # Statistics
    total_tokens = batch_size * seq_len
    masked_tokens = (y_tensor != ignore_index).sum().item()
    mask_percentage = (masked_tokens / total_tokens) * 100
    
    print("STATISTICS:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Masked tokens: {masked_tokens}")
    print(f"  Mask percentage: {mask_percentage:.2f}%")
    
    # Analyze mask token usage in input
    if mask_token_id is not None:
        mask_token_count = (x_tensor == mask_token_id).sum().item()
        mask_token_percentage = (mask_token_count / masked_tokens) * 100 if masked_tokens > 0 else 0
        print(f"  [MASK] tokens in input: {mask_token_count} ({mask_token_percentage:.1f}% of masked positions)")
    
    # Check for vocabulary coverage
    unique_x_tokens = set(x_tensor.flatten().tolist())
    unique_y_tokens = set(y_tensor[y_tensor != ignore_index].tolist())
    print(f"  Unique tokens in x: {len(unique_x_tokens)}")
    print(f"  Unique tokens in y: {len(unique_y_tokens)}")
    
    out_of_vocab_x = [tok for tok in unique_x_tokens if tok not in itos]
    out_of_vocab_y = [tok for tok in unique_y_tokens if tok not in itos]
    
    if out_of_vocab_x:
        print(f"  WARNING: Out-of-vocab tokens in x: {out_of_vocab_x}")
    if out_of_vocab_y:
        print(f"  WARNING: Out-of-vocab tokens in y: {out_of_vocab_y}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python inspect_batch.py <dataset_name> <batch_file_path>")
        print("Example: python inspect_batch.py char_diffusion train/1757250083194-000013-100.pt")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    batch_file_path = sys.argv[2]
    
    try:
        analyze_batch_file(dataset_name, batch_file_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()