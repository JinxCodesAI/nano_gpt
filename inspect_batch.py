#!/usr/bin/env python3
"""
Script to inspect batch files and decode their content using dataset metadata.
Usage: python inspect_batch.py <dataset_name> <batch_file_path>
Example: python inspect_batch.py char_diffusion train/1757250083194-000013-100.pt
"""

import sys
import os
import pickle
import json
import torch
from pathlib import Path


def load_meta(data_dir):
    """Load dataset metadata."""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta


def load_tokenizer_info(data_dir):
    """Load tokenizer info from tokenizer.json if available."""
    tokenizer_path = os.path.join(data_dir, 'tokenizer.json')
    if not os.path.exists(tokenizer_path):
        return None
        
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Try to find vocab in model.vocab first
        vocab = {}
        if 'model' in data and 'vocab' in data['model']:
            vocab = data['model']['vocab']
            
        # Fallback/Update from added_tokens if needed (though usually in vocab)
        # added_tokens is a list of dicts
        if 'added_tokens' in data:
            for item in data['added_tokens']:
                if 'content' in item and 'id' in item:
                    vocab[item['content']] = item['id']
                    
        return vocab
    except Exception as e:
        print(f"Warning: Error loading tokenizer.json: {e}")
        return None



def decode_tokens(tokens, itos):
    """Decode token IDs to characters/strings."""
    return ''.join([itos.get(token, f'<UNK:{token}>') for token in tokens]).replace('Ġ', ' ')


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
                # print(f"  {key}: <vocab mapping with {len(value)} entries>")
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
    
    # Try to load from tokenizer.json first
    tokenizer_vocab = load_tokenizer_info(data_dir)
    
    mask_token_id = None
    pad_token_id = None
    
    if tokenizer_vocab:
        mask_token_id = tokenizer_vocab.get('[MASK]')
        pad_token_id = tokenizer_vocab.get('[PAD]')
        print(f"Loaded special tokens from tokenizer.json: MASK={mask_token_id}, PAD={pad_token_id}")
    
    # Fallback to meta if not found
    if mask_token_id is None:
        mask_token_id = meta.get('mask_token_id')
        if mask_token_id is None and 'stoi' in meta:
            mask_token_id = meta['stoi'].get('[MASK]')
            
    if pad_token_id is None:
        # meta usually doesn't store pad_token_id explicitly unless custom
        if 'stoi' in meta:
            pad_token_id = meta['stoi'].get('[PAD]')

    ignore_index = meta.get('ignore_index', -100)
    
    print("VOCABULARY INFO:")
    print("VOCABULARY INFO:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  mask_token_id: {mask_token_id}")
    print(f"  pad_token_id: {pad_token_id}")
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
    
    # Extract tensors (check both direct and nested structure)
    if 'tensors' in batch_data:
        tensors = batch_data['tensors']
        x_tensor = tensors.get('x', None)
        y_tensor = tensors.get('y', None)
    else:
        x_tensor = batch_data.get('x', None)
        y_tensor = batch_data.get('y', None)
    
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
    num_examples = min(3, batch_size)
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
        print(f"  Target (y): {''.join(y_decoded_parts).replace('Ġ', ' ')}")
        print(f"  Masked positions: {masked_positions}")
        
        # Show token-by-token breakdown for first few positions
        print("  Token breakdown (first 20 positions):")
        for j in range(min(20, seq_len)):
            x_tok = x_tokens[j]
            y_tok = y_tokens[j]
            x_char = itos.get(x_tok, f'<UNK:{x_tok}>').replace('Ġ', ' ')
            y_char = (itos.get(y_tok, f'<UNK:{y_tok}>').replace('Ġ', ' ') if y_tok != ignore_index else '<IGN>')
            mask_indicator = '*' if y_tok != ignore_index else ' '
            print(f"    {j:2d}: x={x_tok:3d}('{x_char}') y={y_tok:4d}('{y_char}') {mask_indicator}")
        print()
    
    # Statistics
    total_tokens = batch_size * seq_len
    
    # 1) Percentage of [PAD] token in input
    pad_percentage = 0.0
    if pad_token_id is not None:
        pad_count = (x_tensor == pad_token_id).sum().item()
        pad_percentage = (pad_count / total_tokens) * 100
    
    # 2) Percentage of [MASK] token in input
    mask_input_percentage = 0.0
    if mask_token_id is not None:
        mask_input_count = (x_tensor == mask_token_id).sum().item()
        mask_input_percentage = (mask_input_count / total_tokens) * 100

    # 3) Percentage of tokens where input is different than target
    # We only care where target is valid (not ignore_index), or maybe global?
    # User request: "percentage of tokens where input is different than targer"
    # Usually we ignore loss where y is ignore_index. 
    # But strictly "input != target" could mean everywhere.
    # However, usually target is -100 (ignore) where it's not trained.
    # Logic: count where (x != y) AND (y != ignore_index)
    valid_targets = (y_tensor != ignore_index)
    diff_mask = (x_tensor != y_tensor) & valid_targets
    diff_count = diff_mask.sum().item()
    diff_percentage = (diff_count / total_tokens) * 100
    
    print("STATISTICS:")
    print(f"  Total tokens: {total_tokens}")
    # print(f"  Masked tokens: {masked_tokens}") # Removed as requested
    # print(f"  Mask percentage: {mask_percentage:.2f}%") # Removed as requested
    
    if pad_token_id is not None:
        print(f"  [PAD] tokens in input: {pad_percentage:.2f}%")
    else:
        print(f"  [PAD] tokens in input: N/A (id not found)")

    if mask_token_id is not None:
        print(f"  [MASK] tokens in input: {mask_input_percentage:.2f}%")
    else:
        print(f"  [MASK] tokens in input: N/A (id not found)")

    print(f"  Input != Target percentage: {diff_percentage:.2f}%")
    
    # Per-sample statistics for Input != Target
    # valid_targets matches shape of x_tensor, y_tensor
    diff_mask_float = diff_mask.float()
    diff_per_sample = diff_mask_float.sum(dim=1)
    # We should normalize by the number of valid targets per sample, or total seq_len?
    # Usually seq_len is constant. If we want "percentage of tokens where input != target", 
    # it implies over the whole sequence.
    pct_per_sample = (diff_per_sample / seq_len) * 100
    
    p10 = torch.quantile(pct_per_sample, 0.1).item()
    p90 = torch.quantile(pct_per_sample, 0.9).item()
    
    print(f"  10th percentile: {p10:.2f}%")
    print(f"  90th percentile: {p90:.2f}%")
    
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