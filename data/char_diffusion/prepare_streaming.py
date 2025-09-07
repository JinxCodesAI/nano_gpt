"""
Streaming provider for character-level BERT-style diffusion training.
Applies BERT-style masking (80% [MASK], 10% random, 10% unchanged) on Shakespeare text.
Uses built-in Python libraries only for CPU-optimized processing.
"""
from __future__ import annotations

import os
import random
import pickle
from typing import Dict, List, Any

# Import torch only when available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create minimal torch-like interface for compatibility
    class MockTensor:
        def __init__(self, data, dtype=None):
            self.data = data if isinstance(data, list) else [data]
            self.dtype = dtype or 'int64'
            self.shape = [len(self.data)] if isinstance(data, list) else [1]
            
        def tolist(self):
            return self.data
            
        def to(self, dtype):
            return MockTensor(self.data, dtype)
            
        @staticmethod
        def from_list(data, dtype='int64'):
            return MockTensor(data, dtype)

from data.common.provider_base import DataProviderBase


def apply_bert_style_corruption_cpu(x: List[List[int]], mask: List[List[bool]], 
                                  mask_token_id: int, vocab_size: int) -> List[List[int]]:
    """
    CPU-optimized BERT-style corruption using pure Python.
    Applies 80/10/10 rule: 80% [MASK], 10% random token, 10% unchanged.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions to corrupt (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token
        vocab_size: Size of vocabulary for random token generation
        
    Returns:
        corrupted_x: Input with BERT-style corruption applied
    """
    batch_size = len(x)
    seq_len = len(x[0])
    corrupted_x = [row[:] for row in x]  # Deep copy
    
    for i in range(batch_size):
        for j in range(seq_len):
            if mask[i][j]:
                rand = random.random()
                if rand < 0.8:
                    # 80%: replace with [MASK] token
                    corrupted_x[i][j] = mask_token_id
                elif rand < 0.9:
                    # 10%: replace with random token
                    corrupted_x[i][j] = random.randint(0, vocab_size - 1)
                # 10%: keep original token (no action needed)
    
    return corrupted_x


def apply_random_masking_cpu(x: List[List[int]], mask_probability: float, 
                           mask_token_id: int, vocab_size: int) -> tuple[List[List[int]], List[List[bool]]]:
    """
    CPU-optimized random masking for BERT-style training using pure Python.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        mask_probability: Probability of masking each token (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token
        vocab_size: Size of vocabulary for random token generation
        
    Returns:
        corrupted_x: Input with masking applied
        mask: Boolean mask indicating which positions were selected for prediction
    """
    batch_size = len(x)
    seq_len = len(x[0])
    
    # Generate random mask
    mask = [[random.random() < mask_probability for _ in range(seq_len)] for _ in range(batch_size)]
    
    # Apply BERT-style corruption
    corrupted_x = apply_bert_style_corruption_cpu(x, mask, mask_token_id, vocab_size)
    
    return corrupted_x, mask


class CharDiffusionProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        # Extract diffusion-specific config
        self.mask_probability = kwargs.pop('mask_probability', 0.15)
        self.mask_token_id = kwargs.pop('mask_token_id', None)  # Will be set after vocab creation
        
        super().__init__(*args, **kwargs)
        
        # Load Shakespeare data - fail if not present
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        with open(input_file_path, 'r') as f:
            data = f.read()
        
        # Create vocabulary
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars) + 1  # +1 for [MASK] token
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Add [MASK] token
        self.mask_token_id = len(chars)
        self.stoi['[MASK]'] = self.mask_token_id
        self.itos[self.mask_token_id] = '[MASK]'
        
        # Create train/val splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        self.train_ids = [self.stoi[c] for c in train_data]
        self.val_ids = [self.stoi[c] for c in val_data]
        
        if self.verbose:
            print(f"CharDiffusionProvider initialized:")
            print(f"  vocab_size: {self.vocab_size} (including [MASK])")
            print(f"  mask_probability: {self.mask_probability}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  train_data_size: {len(self.train_ids)}")
            print(f"  val_data_size: {len(self.val_ids)}")

    def build_meta(self) -> Dict:
        """Build metadata for BERT-style masked language modeling."""
        if self.block_size is None:
            raise ValueError("block_size must be set for CharDiffusionProvider")
        
        return {
            "dataset_name": "char_diffusion",
            "training_type": "MLM",  # Masked Language Modeling
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "mask_probability": self.mask_probability,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "input_ids", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "labels", "dtype": "int64", "shape": [self.block_size], "role": "target"},
                {"name": "attention_mask", "dtype": "bool", "shape": [self.block_size], "role": "mask"},
            ],
        }

    def sample_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with BERT-style masking."""
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - self.block_size
        
        # Sample random starting positions using the provided RNG seed
        # Convert torch Generator to random seed for reproducibility
        if TORCH_AVAILABLE and hasattr(rng, 'initial_seed'):
            random.seed(rng.initial_seed() % (2**32))
        else:
            random.seed(1337)  # fallback
            
        ix = [random.randint(0, max_start_idx - 1) for _ in range(self.batch_size)]
        
        # Create sequences
        x = [ids[i : i + self.block_size] for i in ix]
        
        # Apply BERT-style masking
        corrupted_x, mask = apply_random_masking_cpu(
            x, self.mask_probability, self.mask_token_id, 
            self.vocab_size - 1  # Exclude [MASK] token from random generation
        )
        
        # Create labels: -100 for non-masked positions (ignored in loss), original token for masked
        labels = [[-100 if not mask[i][j] else x[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        
        # Create attention mask (all positions are valid for now)
        attention_mask = [[True for _ in range(len(x[i]))] for i in range(len(x))]
        
        # Convert to tensors if torch is available, otherwise return as lists/mock tensors
        if TORCH_AVAILABLE:
            return {
                "input_ids": torch.tensor(corrupted_x, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            }
        else:
            # Return mock tensors for compatibility
            return {
                "input_ids": MockTensor.from_list(corrupted_x, 'int64'),
                "labels": MockTensor.from_list(labels, 'int64'),
                "attention_mask": MockTensor.from_list(attention_mask, 'bool'),
            }


def main():
    """Standalone entrypoint for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Streaming char diffusion provider")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--mask_probability', type=float, default=0.15)
    parser.add_argument('--batches_per_file', type=int, default=100)
    parser.add_argument('--max_backlog_files', type=int, default=2)
    parser.add_argument('--sleep_seconds', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    data_dir = os.path.dirname(__file__)
    provider = CharDiffusionProvider(
        data_dir=data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        mask_probability=args.mask_probability,
        batches_per_file=args.batches_per_file,
        max_backlog_files=args.max_backlog_files,
        sleep_seconds=args.sleep_seconds,
        seed=args.seed,
        verbose=args.verbose,
    )
    provider.run()


# Explicit provider alias for prepare.py discovery
Provider = CharDiffusionProvider


if __name__ == "__main__":
    main()