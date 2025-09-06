"""
Streaming provider for Shakespeare char-level LM using DataProviderBase.
Produces batches into queue/train and queue/val while respecting backlog limits.
"""
from __future__ import annotations

import os
import requests
import numpy as np
import torch
from typing import Dict

from data.common.provider_base import DataProviderBase


class ShakespeareCharProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # load or download data
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)
        with open(input_file_path, 'r') as f:
            data = f.read()
        # vocab
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        # splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        self.train_ids = np.array([self.stoi[c] for c in train_data], dtype=np.int64)
        self.val_ids = np.array([self.stoi[c] for c in val_data], dtype=np.int64)

    def build_meta(self) -> Dict:
        # LM schema: x[int64, block_size], y[int64, target_size]
        if self.block_size is None:
            raise ValueError("block_size must be set for ShakespeareCharProvider")
        tgt = self.target_size if self.target_size is not None else self.block_size
        return {
            "dataset_name": "shakespeare_char",
            "training_type": "LM",
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [tgt], "role": "target"},
            ],
        }

    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, torch.Tensor]:
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - int(self.block_size)
        ix = torch.randint(max_start_idx, (self.batch_size,), generator=rng)
        # Create x and y tensors
        x = torch.stack([torch.from_numpy(ids[i : i + self.block_size]) for i in ix])
        tgt = self.target_size if self.target_size is not None else self.block_size
        y = torch.stack([torch.from_numpy(ids[i + 1 : i + 1 + tgt]) for i in ix])
        return {"x": x.to(torch.int64), "y": y.to(torch.int64)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Streaming Shakespeare provider")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--target_size', type=int, default=None)
    parser.add_argument('--batches_per_file', type=int, default=100)
    parser.add_argument('--max_backlog_files', type=int, default=2)
    parser.add_argument('--sleep_seconds', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    data_dir = os.path.dirname(__file__)
    provider = ShakespeareCharProvider(
        data_dir=data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        target_size=args.target_size,
        batches_per_file=args.batches_per_file,
        max_backlog_files=args.max_backlog_files,
        sleep_seconds=args.sleep_seconds,
        seed=args.seed,
        verbose=args.verbose,
    )
    provider.run()


if __name__ == "__main__":
    main()

