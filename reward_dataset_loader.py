#!/usr/bin/env python3
"""
Utility functions for loading and working with reward model datasets.
Enhanced with tokenization compatibility validation.
"""

import os
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from reward_data_config import TokenizationInfo


class RewardDataset(Dataset):
    """PyTorch Dataset for reward model training data with tokenization compatibility validation."""

    def __init__(self, data_dir, split='train', expected_tokenization_info=None):
        """
        Initialize the reward dataset.

        Args:
            data_dir: Directory containing the reward dataset files
            split: 'train' or 'val'
            expected_tokenization_info: TokenizationInfo to validate compatibility against
        """
        self.data_dir = data_dir
        self.split = split
        self.logger = logging.getLogger(__name__)

        # Load data
        x_path = os.path.join(data_dir, f'{split}_x.bin')
        y_path = os.path.join(data_dir, f'{split}_y.bin')

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(f"Dataset files not found in {data_dir}")

        # Load metadata to get shapes and tokenization info
        metadata_path = os.path.join(data_dir, f'{split}_metadata.txt')
        metadata = self._load_metadata(metadata_path)

        # Extract tokenization information from metadata
        self.tokenization_info = self._extract_tokenization_info(metadata)

        # Validate tokenization compatibility if expected info provided
        if expected_tokenization_info:
            self._validate_tokenization_compatibility(expected_tokenization_info)

        # Load binary data
        self.x_data = np.fromfile(x_path, dtype=np.uint16)
        self.y_data = np.fromfile(y_path, dtype=np.float32)

        # Reshape data
        self.num_samples = metadata['num_samples']
        self.block_size = metadata['block_size']

        self.x_data = self.x_data.reshape(self.num_samples, self.block_size)
        self.y_data = self.y_data.reshape(self.num_samples, 2)

        print(f"Loaded {split} dataset: {self.num_samples} samples, block_size={self.block_size}")
        if self.tokenization_info:
            print(f"  Tokenization: {self.tokenization_info.method} (vocab_size={self.tokenization_info.vocab_size})")
    
    def _load_metadata(self, metadata_path):
        """Load metadata from text file."""
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        key = key.strip()
                        value = value.strip()

                        # Try to convert to appropriate type
                        if key in ['num_samples', 'block_size', 'vocab_size']:
                            metadata[key] = int(value)
                        elif key in ['x_shape', 'y_shape']:
                            # Parse tuple string like "(1000, 1024)"
                            value = value.strip('()')
                            metadata[key] = tuple(map(int, value.split(', ')))
                        else:
                            metadata[key] = value

        return metadata

    def _extract_tokenization_info(self, metadata):
        """Extract tokenization information from metadata."""
        if 'tokenization_method' in metadata:
            return TokenizationInfo(
                method=metadata['tokenization_method'],
                vocab_size=metadata.get('vocab_size', 0),
                meta_path=metadata.get('meta_path') if metadata.get('meta_path') != 'None' else None
            )
        return None

    def _validate_tokenization_compatibility(self, expected_info):
        """Validate that dataset tokenization is compatible with expected tokenization."""
        if not self.tokenization_info:
            self.logger.warning("Dataset has no tokenization information - cannot validate compatibility")
            return

        if self.tokenization_info.method != expected_info.method:
            raise ValueError(
                f"Tokenization method mismatch: dataset uses '{self.tokenization_info.method}' "
                f"but expected '{expected_info.method}'"
            )

        if self.tokenization_info.vocab_size != expected_info.vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: dataset has {self.tokenization_info.vocab_size} "
                f"but expected {expected_info.vocab_size}"
            )

        self.logger.info(f"Tokenization compatibility validated: {expected_info.method} "
                        f"(vocab_size={expected_info.vocab_size})")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample."""
        x = torch.from_numpy(self.x_data[idx]).long()
        y = torch.from_numpy(self.y_data[idx]).float()
        return x, y
    
    def get_stats(self):
        """Get dataset statistics including tokenization information."""
        stats = {
            'num_samples': self.num_samples,
            'block_size': self.block_size,
            'y_min': self.y_data.min(axis=0),
            'y_max': self.y_data.max(axis=0),
            'y_mean': self.y_data.mean(axis=0),
            'y_std': self.y_data.std(axis=0),
        }

        # Check probability sums
        prob_sums = self.y_data.sum(axis=1)
        stats['prob_sum_min'] = prob_sums.min()
        stats['prob_sum_max'] = prob_sums.max()
        stats['prob_sum_mean'] = prob_sums.mean()

        # Add tokenization information
        if self.tokenization_info:
            stats['tokenization_method'] = self.tokenization_info.method
            stats['vocab_size'] = self.tokenization_info.vocab_size
            stats['meta_path'] = self.tokenization_info.meta_path

        return stats

    def get_tokenization_info(self):
        """Get tokenization information for this dataset."""
        return self.tokenization_info


def create_reward_dataloaders(data_dir, batch_size=32, num_workers=0, expected_tokenization_info=None):
    """
    Create train and validation dataloaders for reward model training.

    Args:
        data_dir: Directory containing reward dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        expected_tokenization_info: TokenizationInfo to validate compatibility against

    Returns:
        train_loader, val_loader
    """

    # Create datasets with tokenization validation
    train_dataset = RewardDataset(data_dir, 'train', expected_tokenization_info)
    val_dataset = RewardDataset(data_dir, 'val', expected_tokenization_info)

    # Verify tokenization consistency between train and val
    train_info = train_dataset.get_tokenization_info()
    val_info = val_dataset.get_tokenization_info()

    if train_info and val_info:
        if (train_info.method != val_info.method or
            train_info.vocab_size != val_info.vocab_size):
            raise ValueError(
                f"Tokenization mismatch between train and val datasets: "
                f"train={train_info.method}({train_info.vocab_size}) vs "
                f"val={val_info.method}({val_info.vocab_size})"
            )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def print_dataset_info(data_dir):
    """Print information about the reward dataset including tokenization details."""

    print(f"=== Reward Dataset Info: {data_dir} ===")

    # Check for tokenization consistency across splits
    tokenization_infos = {}

    for split in ['train', 'val']:
        try:
            dataset = RewardDataset(data_dir, split)
            stats = dataset.get_stats()
            tokenization_infos[split] = dataset.get_tokenization_info()

            print(f"\n{split.upper()} SET:")
            print(f"  Samples: {stats['num_samples']:,}")
            print(f"  Block size: {stats['block_size']}")

            # Show tokenization information
            if 'tokenization_method' in stats:
                print(f"  Tokenization: {stats['tokenization_method']} (vocab_size={stats['vocab_size']})")
                if stats.get('meta_path'):
                    print(f"    Meta path: {stats['meta_path']}")

            print(f"  Y statistics:")
            print(f"    Min: {stats['y_min']}")
            print(f"    Max: {stats['y_max']}")
            print(f"    Mean: {stats['y_mean']}")
            print(f"    Std: {stats['y_std']}")
            print(f"  Probability sums:")
            print(f"    Min: {stats['prob_sum_min']:.6f}")
            print(f"    Max: {stats['prob_sum_max']:.6f}")
            print(f"    Mean: {stats['prob_sum_mean']:.6f}")

        except FileNotFoundError:
            print(f"\n{split.upper()} SET: Not found")
        except Exception as e:
            print(f"\n{split.upper()} SET: Error loading - {e}")

    # Check tokenization consistency
    if len(tokenization_infos) > 1:
        train_info = tokenization_infos.get('train')
        val_info = tokenization_infos.get('val')

        if train_info and val_info:
            if (train_info.method == val_info.method and
                train_info.vocab_size == val_info.vocab_size):
                print(f"\n✅ Tokenization consistency: Both splits use {train_info.method} "
                      f"with vocab_size={train_info.vocab_size}")
            else:
                print(f"\n⚠️  Tokenization mismatch detected:")
                print(f"    Train: {train_info.method} (vocab_size={train_info.vocab_size})")
                print(f"    Val: {val_info.method} (vocab_size={val_info.vocab_size})")
        elif not train_info and not val_info:
            print(f"\n⚠️  No tokenization information found in dataset metadata")
        else:
            print(f"\n⚠️  Incomplete tokenization information across splits")


if __name__ == '__main__':
    # Test the dataset loader
    data_dir = 'data/reward_dataset'
    
    if os.path.exists(data_dir):
        print_dataset_info(data_dir)
        
        # Test creating dataloaders
        try:
            train_loader, val_loader = create_reward_dataloaders(data_dir, batch_size=4)
            
            print(f"\n=== DataLoader Test ===")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            
            # Test loading a batch
            for batch_x, batch_y in train_loader:
                print(f"Batch X shape: {batch_x.shape}")
                print(f"Batch Y shape: {batch_y.shape}")
                print(f"Batch Y sample: {batch_y[0]}")
                break
                
        except Exception as e:
            print(f"DataLoader test failed: {e}")
    else:
        print(f"Dataset directory {data_dir} not found.")
        print("Run prepare_reward_data.py first to generate the dataset.")