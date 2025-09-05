"""
Unit tests for BatchManager class.
"""

import os
import tempfile
import shutil
import numpy as np
import torch
import pytest
from batch_manager import BatchManager


class TestBatchManager:
    """Test cases for BatchManager class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.batch_size = 4
        self.block_size = 8
        self.device_type = 'cpu'
        
    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_batch_file(self, split, file_idx, num_batches=3):
        """Create a test batch file with dummy data."""
        # Create dummy data
        total_samples = num_batches * self.batch_size
        x_data = torch.randint(0, 1000, (total_samples, self.block_size))
        y_data = torch.randint(0, 1000, (total_samples, self.block_size))
        
        metadata = {
            'batch_size': self.batch_size,
            'block_size': self.block_size,
            'target_size': self.block_size
        }
        
        data = {
            'x': x_data,
            'y': y_data,
            'metadata': metadata
        }
        
        filepath = os.path.join(self.temp_dir, f'{split}_batches_{file_idx:04d}.pt')
        torch.save(data, filepath)
        return filepath
    
    def _create_legacy_data_file(self, split, size=1000):
        """Create a legacy .bin data file for testing backward compatibility."""
        data = np.random.randint(0, 1000, size=size, dtype=np.uint16)
        filepath = os.path.join(self.temp_dir, f'{split}.bin')
        data.tofile(filepath)
        return filepath
    
    def test_initialization(self):
        """Test BatchManager initialization."""
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        assert batch_manager.data_dir == self.temp_dir
        assert batch_manager.batch_size == self.batch_size
        assert batch_manager.block_size == self.block_size
        assert batch_manager.device_type == self.device_type
        assert not batch_manager._initialized
    
    def test_batch_loading_with_precomputed_files(self):
        """Test batch loading with pre-computed batch files."""
        # Create test batch files
        self._create_test_batch_file('train', 0)
        self._create_test_batch_file('val', 0)
        
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        device = torch.device('cpu')
        
        # Test train batch loading
        x, y = batch_manager.get_batch('train', device)
        assert x.shape == (self.batch_size, self.block_size)
        assert y.shape == (self.batch_size, self.block_size)
        assert x.device == device
        assert y.device == device
        
        # Test val batch loading
        x_val, y_val = batch_manager.get_batch('val', device)
        assert x_val.shape == (self.batch_size, self.block_size)
        assert y_val.shape == (self.batch_size, self.block_size)
    
    def test_circular_reading(self):
        """Test circular reading across multiple batch files."""
        # Create multiple batch files
        self._create_test_batch_file('train', 0, num_batches=2)
        self._create_test_batch_file('train', 1, num_batches=2)
        self._create_test_batch_file('val', 0, num_batches=1)
        
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        device = torch.device('cpu')
        
        # Load batches to exhaust first file and move to second
        for _ in range(3):  # This should move us to the second file
            x, y = batch_manager.get_batch('train', device)
            assert x.shape == (self.batch_size, self.block_size)
        
        # Check that we moved to the next file
        stats = batch_manager.get_stats()
        assert stats['current_train_file'] == 1
    
    def test_legacy_fallback(self):
        """Test fallback to legacy .bin files when no batch files exist."""
        # Create legacy data files
        self._create_legacy_data_file('train', size=1000)
        self._create_legacy_data_file('val', size=1000)
        
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        device = torch.device('cpu')
        
        # Test that legacy loading works
        x, y = batch_manager.get_batch('train', device)
        assert x.shape == (self.batch_size, self.block_size)
        assert y.shape == (self.batch_size, self.block_size)
        
        # Test val split
        x_val, y_val = batch_manager.get_batch('val', device)
        assert x_val.shape == (self.batch_size, self.block_size)
        assert y_val.shape == (self.batch_size, self.block_size)
    
    def test_metadata_validation(self):
        """Test metadata validation warnings."""
        # Create batch file with different metadata
        total_samples = 2 * self.batch_size
        x_data = torch.randint(0, 1000, (total_samples, self.block_size))
        y_data = torch.randint(0, 1000, (total_samples, self.block_size))

        # Wrong batch_size in metadata
        metadata = {
            'batch_size': self.batch_size + 1,  # Different from manager
            'block_size': self.block_size,
            'target_size': self.block_size
        }

        data = {'x': x_data, 'y': y_data, 'metadata': metadata}
        filepath = os.path.join(self.temp_dir, 'train_batches_0000.pt')
        torch.save(data, filepath)

        # Also create val batch file to avoid fallback to legacy mode
        self._create_test_batch_file('val', 0)

        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )

        device = torch.device('cpu')

        # This should work but print warnings
        x, y = batch_manager.get_batch('train', device)
        assert x.shape[0] <= self.batch_size  # May be smaller due to slicing
        assert x.shape[1] == self.block_size
    
    def test_reset_state(self):
        """Test state reset functionality."""
        self._create_test_batch_file('train', 0)
        self._create_test_batch_file('val', 0)
        
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        device = torch.device('cpu')
        
        # Load some batches to change state
        batch_manager.get_batch('train', device)
        batch_manager.get_batch('val', device)
        
        # Reset train state
        batch_manager.reset_state('train')
        stats = batch_manager.get_stats()
        assert stats['current_train_file'] == 0
        assert stats['current_train_batch'] == 0
        
        # Reset all states
        batch_manager.reset_state()
        stats = batch_manager.get_stats()
        assert stats['current_train_file'] == 0
        assert stats['current_val_file'] == 0
        assert stats['current_train_batch'] == 0
        assert stats['current_val_batch'] == 0
    
    def test_get_stats(self):
        """Test statistics reporting."""
        self._create_test_batch_file('train', 0)
        self._create_test_batch_file('train', 1)
        self._create_test_batch_file('val', 0)
        
        batch_manager = BatchManager(
            self.temp_dir, self.batch_size, self.block_size, self.device_type
        )
        
        device = torch.device('cpu')
        
        # Load a batch to initialize
        batch_manager.get_batch('train', device)
        
        stats = batch_manager.get_stats()
        assert stats['train_files'] == 2
        assert stats['val_files'] == 1
        assert stats['initialized'] == True
        assert 'current_train_file' in stats
        assert 'current_val_file' in stats
        assert 'current_train_batch' in stats
        assert 'current_val_batch' in stats


if __name__ == '__main__':
    pytest.main([__file__])
