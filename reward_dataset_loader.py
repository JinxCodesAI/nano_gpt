#!/usr/bin/env python3
"""
Utility functions for loading and working with reward model datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RewardDataset(Dataset):
    """PyTorch Dataset for reward model training data."""
    
    def __init__(self, data_dir, split='train'):
        """
        Initialize the reward dataset.
        
        Args:
            data_dir: Directory containing the reward dataset files
            split: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load data
        x_path = os.path.join(data_dir, f'{split}_x.bin')
        y_path = os.path.join(data_dir, f'{split}_y.bin')
        
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(f"Dataset files not found in {data_dir}")
        
        # Load metadata to get shapes
        metadata_path = os.path.join(data_dir, f'{split}_metadata.txt')
        metadata = self._load_metadata(metadata_path)
        
        # Load binary data
        self.x_data = np.fromfile(x_path, dtype=np.uint16)
        self.y_data = np.fromfile(y_path, dtype=np.float32)
        
        # Reshape data
        self.num_samples = metadata['num_samples']
        self.block_size = metadata['block_size']
        
        self.x_data = self.x_data.reshape(self.num_samples, self.block_size)
        self.y_data = self.y_data.reshape(self.num_samples, 2)
        
        print(f"Loaded {split} dataset: {self.num_samples} samples, block_size={self.block_size}")
    
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
                        if key in ['num_samples', 'block_size']:
                            metadata[key] = int(value)
                        elif key in ['x_shape', 'y_shape']:
                            # Parse tuple string like "(1000, 1024)"
                            value = value.strip('()')
                            metadata[key] = tuple(map(int, value.split(', ')))
                        else:
                            metadata[key] = value
        
        return metadata
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample."""
        x = torch.from_numpy(self.x_data[idx]).long()
        y = torch.from_numpy(self.y_data[idx]).float()
        return x, y
    
    def get_stats(self):
        """Get dataset statistics."""
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
        
        return stats


def create_reward_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Create train and validation dataloaders for reward model training.
    
    Args:
        data_dir: Directory containing reward dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets
    train_dataset = RewardDataset(data_dir, 'train')
    val_dataset = RewardDataset(data_dir, 'val')
    
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
    """Print information about the reward dataset."""
    
    print(f"=== Reward Dataset Info: {data_dir} ===")
    
    for split in ['train', 'val']:
        try:
            dataset = RewardDataset(data_dir, split)
            stats = dataset.get_stats()
            
            print(f"\n{split.upper()} SET:")
            print(f"  Samples: {stats['num_samples']:,}")
            print(f"  Block size: {stats['block_size']}")
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