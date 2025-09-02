"""Generic data loading utilities - reusable across datasets"""
import numpy as np
import torch
from typing import Tuple, Optional


def load_memmap_data(data_path: str) -> np.memmap:
    """Load memory-mapped data file"""
    return np.memmap(data_path, dtype=np.uint16, mode='r')


def sample_indices_random(data_length: int, batch_size: int, block_size: int, 
                         seed: Optional[int] = None) -> np.ndarray:
    """Sample random indices for batch generation"""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(data_length - block_size, (batch_size,)).numpy()


def sample_indices_paragraph_boundaries(valid_indices: np.ndarray, batch_size: int, 
                                       seed: Optional[int] = None) -> np.ndarray:
    """Sample indices respecting paragraph boundaries"""
    if seed is not None:
        torch.manual_seed(seed)
    ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
    return valid_indices[ix_indices]


def vectorized_data_loading(data: np.memmap, indices: np.ndarray, block_size: int) -> np.ndarray:
    """Efficient vectorized data loading"""
    ix_expanded = indices[:, None] + np.arange(block_size)[None, :]
    return data[ix_expanded].astype(np.int64)


def find_double_newline_indices(data: np.memmap, vocab_size: int, block_size: int) -> np.ndarray:
    """
    Find positions that start at paragraph boundaries (double newlines).
    This is dataset-agnostic utility that can work with any character-based dataset.
    """
    # Look for double newline pattern in the data
    # This assumes newline character is in the vocabulary
    
    # Convert data to numpy array for processing
    data_array = np.array(data)
    
    # Find all newline positions (assuming newline is encoded as a specific token)
    # For Shakespeare dataset, we need to find the actual newline token ID
    # This is a simplified implementation - in practice you'd need to know the token mapping
    
    # For now, return a simple range that respects block_size constraints
    # This can be enhanced per dataset as needed
    valid_positions = []
    
    # Simple implementation: every 100 positions could be a valid start
    # Real implementation would analyze the actual text structure
    for i in range(0, len(data_array) - block_size, 100):
        valid_positions.append(i)
    
    return np.array(valid_positions, dtype=np.int64)


def create_batch_from_indices(data: np.memmap, indices: np.ndarray, 
                            block_size: int, device: str = 'cpu') -> torch.Tensor:
    """Create a batch tensor from data indices"""
    # Vectorized data loading
    x_np = vectorized_data_loading(data, indices, block_size)
    
    # Convert to tensor
    x = torch.from_numpy(x_np)
    
    # Move to device if needed
    if device != 'cpu':
        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
    
    return x


def validate_data_integrity(data_path: str, expected_dtype: type = np.uint16) -> bool:
    """Validate that a data file has the expected format"""
    try:
        data = np.memmap(data_path, dtype=expected_dtype, mode='r')
        # Basic checks
        if len(data) == 0:
            return False
        # Check if values are within reasonable range for token IDs
        if data.max() > 100000 or data.min() < 0:
            return False
        return True
    except Exception:
        return False


def get_data_stats(data_path: str) -> dict:
    """Get statistics about a data file"""
    data = load_memmap_data(data_path)
    return {
        'length': len(data),
        'min_token': int(data.min()),
        'max_token': int(data.max()),
        'unique_tokens': len(np.unique(data)),
        'dtype': str(data.dtype)
    }