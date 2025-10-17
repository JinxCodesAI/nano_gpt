"""
File utility functions for data provider operations.
"""
import os
import time
import torch
from typing import Dict


def write_file_atomic(data: Dict, directory: str, seq: int, batches_per_file: int, verbose: bool = False) -> str:
    """
    Write data to file atomically using temporary file and os.replace.
    
    Args:
        data: Dictionary containing tensors and metadata to save
        directory: Target directory for the file
        seq: Sequence number for filename
        batches_per_file: Number of batches in the file
        verbose: Whether to print verbose output
        
    Returns:
        final_path: Path of the written file
    """
    ts = int(time.time() * 1000)
    tmp_name = f".tmp-{ts}-{seq:06d}.pt"
    final_name = f"{ts}-{seq:06d}-{batches_per_file}.pt"
    tmp_path = os.path.join(directory, tmp_name)
    final_path = os.path.join(directory, final_name)
    
    torch.save(data, tmp_path)
    os.replace(tmp_path, final_path)
    
    if verbose:
        print(f"[provider] produced: {final_path}")
    
    return final_path


def ensure_queue_dirs(data_dir: str) -> tuple[str, str]:
    """
    Ensure queue directories exist and return their paths.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Tuple of (train_dir, val_dir) paths
    """
    queue_dir = os.path.join(data_dir, "queue")
    train_dir = os.path.join(queue_dir, "train")
    val_dir = os.path.join(queue_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    return train_dir, val_dir


def get_backlog_size(directory: str) -> int:
    """
    Get the number of completed batch files in a directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        Number of .pt files that don't start with .tmp-
    """
    if not os.path.exists(directory):
        return 0
    
    return len([
        fn for fn in os.listdir(directory) 
        if fn.endswith('.pt') and not fn.startswith('.tmp-')
    ])


def get_max_sequence_number(directory: str) -> int:
    """
    Get the maximum sequence number from existing files in directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Maximum sequence number found, or -1 if no files exist
    """
    if not os.path.exists(directory):
        return -1
        
    max_seq = -1
    existing = [
        fn for fn in os.listdir(directory) 
        if fn.endswith('.pt') and not fn.startswith('.tmp-')
    ]
    
    for fn in existing:
        try:
            # filename format: ts-seq-batches.pt
            parts = fn.split('-')
            seq = int(parts[1])
            max_seq = max(max_seq, seq)
        except (ValueError, IndexError):
            continue
            
    return max_seq