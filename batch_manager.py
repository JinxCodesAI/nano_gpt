"""
Batch management for training data loading with pre-computed batches.

This module provides the BatchManager class that handles loading and managing
pre-computed training batches with circular reading and backward compatibility
with legacy data formats.
"""

import os
import glob
import numpy as np
import torch


class BatchManager:
    """
    Manages batch loading from pre-computed batch files with circular reading.
    
    Supports both pre-computed batch files and legacy .bin files for backward
    compatibility. Handles metadata validation and efficient batch iteration.
    """
    
    def __init__(self, data_dir, batch_size, block_size, device_type='cuda'):
        """
        Initialize BatchManager.
        
        Args:
            data_dir (str): Directory containing the dataset
            batch_size (int): Batch size for training
            block_size (int): Block size (sequence length)
            device_type (str): Device type ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.device_type = device_type
        
        # Batch file management state
        self._batch_files = {'train': [], 'val': []}
        self._current_file_idx = {'train': 0, 'val': 0}
        self._current_batch_idx = {'train': 0, 'val': 0}
        self._loaded_data = {'train': None, 'val': None}
        self._initialized = False
        
    def _initialize_batch_files(self):
        """Initialize the list of available batch files for each split."""
        for split in ['train', 'val']:
            pattern = os.path.join(self.data_dir, f'{split}_batches_*.pt')
            files = sorted(glob.glob(pattern))
            if not files:
                # Fallback to legacy mode if no batch files found
                print(f"Warning: No pre-computed batch files found for {split}. Using legacy mode.")
                return False
            self._batch_files[split] = files
            print(f"Found {len(files)} {split} batch files")
        return True
    
    def _load_batch_file(self, split, file_idx):
        """Load a specific batch file."""
        if file_idx >= len(self._batch_files[split]):
            file_idx = 0  # Circular reading

        filepath = self._batch_files[split][file_idx]
        data = torch.load(filepath, map_location='cpu')

        # Validate metadata matches current training config
        metadata = data['metadata']
        if metadata['batch_size'] != self.batch_size:
            print(f"Warning: Batch file batch_size ({metadata['batch_size']}) != training batch_size ({self.batch_size})")
        if metadata['block_size'] != self.block_size:
            print(f"Warning: Batch file block_size ({metadata['block_size']}) != training block_size ({self.block_size})")

        return data
    
    def _get_legacy_batch(self, split, device):
        """Fallback to legacy implementation for backward compatibility."""
        if split == 'train':
            data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y
    
    def get_batch(self, split, device):
        """
        Load pre-computed batch data with circular reading, with backward compatibility.
        
        Args:
            split (str): Data split ('train' or 'val')
            device: PyTorch device to move tensors to
            
        Returns:
            tuple: (x, y) tensors for input and target sequences
        """
        # Initialize batch files on first call
        if not self._initialized:
            if not self._initialize_batch_files():
                # Fallback to legacy implementation for backward compatibility
                return self._get_legacy_batch(split, device)
            self._initialized = True

        # Check if we need to load a new file
        if self._loaded_data[split] is None:
            self._loaded_data[split] = self._load_batch_file(split, self._current_file_idx[split])
            self._current_batch_idx[split] = 0

        # Get current batch
        data = self._loaded_data[split]
        x_data = data['x']
        y_data = data['y']

        # Calculate batch boundaries
        start_idx = self._current_batch_idx[split] * self.batch_size
        end_idx = min(start_idx + self.batch_size, x_data.shape[0])

        # Check if we need to move to next file
        if start_idx >= x_data.shape[0]:
            # Move to next file (circular)
            self._current_file_idx[split] = (self._current_file_idx[split] + 1) % len(self._batch_files[split])
            self._loaded_data[split] = self._load_batch_file(split, self._current_file_idx[split])
            self._current_batch_idx[split] = 0

            # Get batch from new file
            x_data = self._loaded_data[split]['x']
            y_data = self._loaded_data[split]['y']
            start_idx = 0
            end_idx = min(self.batch_size, x_data.shape[0])

        # Extract batch
        x = x_data[start_idx:end_idx]
        y = y_data[start_idx:end_idx]

        # Move to next batch for next call
        self._current_batch_idx[split] += 1

        # Move to device
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y
    
    def reset_state(self, split=None):
        """
        Reset the batch loading state for a specific split or all splits.
        
        Args:
            split (str, optional): Split to reset ('train' or 'val'). 
                                 If None, resets all splits.
        """
        splits_to_reset = [split] if split else ['train', 'val']
        
        for s in splits_to_reset:
            self._current_file_idx[s] = 0
            self._current_batch_idx[s] = 0
            self._loaded_data[s] = None
    
    def get_stats(self):
        """
        Get statistics about the current batch loading state.
        
        Returns:
            dict: Statistics including current file indices, batch indices, 
                  and total number of files per split.
        """
        return {
            'train_files': len(self._batch_files['train']),
            'val_files': len(self._batch_files['val']),
            'current_train_file': self._current_file_idx['train'],
            'current_val_file': self._current_file_idx['val'],
            'current_train_batch': self._current_batch_idx['train'],
            'current_val_batch': self._current_batch_idx['val'],
            'initialized': self._initialized
        }
