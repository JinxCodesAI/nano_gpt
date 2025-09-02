"""Generic dataset interface for all diffusion datasets"""
import os
import pickle
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from queue import Queue
import threading
import importlib.util


class DatasetConfig:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_dir = os.path.join('data', dataset_name)
        self.meta = self._load_meta()
        self.training_config = self._load_training_config()
        
        # Runtime caches (keep these in training layer)
        self._batch_cache = Queue(maxsize=self.meta.get('batch_cache_size', 1000))
        self._current_batch_file = None
        self._cached_data = {}
        
        # Validate dataset integrity
        self._validate_dataset()
    
    def _load_meta(self) -> Dict[str, Any]:
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Dataset {self.dataset_name} missing meta.pkl")
        with open(meta_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_training_config(self) -> Any:
        config_path = os.path.join(self.data_dir, 'training_config.py')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Dataset {self.dataset_name} missing training_config.py")
            
        spec = importlib.util.spec_from_file_location("training_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    
    def _validate_dataset(self):
        """Validate dataset has all required components"""
        required_files = ['meta.pkl', 'training_config.py', 'train.bin', 'val.bin']
        for file in required_files:
            filepath = os.path.join(self.data_dir, file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset {self.dataset_name} missing required file: {file}")
        
        # Validate prepared batches exist
        prepared_dir = os.path.join(self.data_dir, 'prepared_batches')
        if not os.path.exists(prepared_dir):
            raise FileNotFoundError(f"Dataset {self.dataset_name} missing prepared_batches directory. Run prepare.py first.")
    
    def validate_training_config(self, block_size: int, batch_size: Optional[int] = None):
        """Validate training parameters against dataset constraints"""
        if block_size != self.meta['block_size']:
            raise ValueError(f"Block size mismatch: dataset '{self.dataset_name}' requires {self.meta['block_size']}, got {block_size}")
        
        # Any batch_size should be supported through dynamic loading
        if batch_size and batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")
    
    def get_training_batch(self, iteration: int, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get training batch for specific iteration with runtime caching"""
        self.validate_training_config(block_size, batch_size)
        
        # Collect samples to form the requested batch size
        return self._collect_samples_for_batch(batch_size, iteration, 'train')
    
    def get_validation_batch(self, iteration: int, validation_sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get validation batch for specific iteration and sample index"""
        batch_file = self._find_batch_file_for_iteration(iteration, 'val')
        val_batches = torch.load(batch_file, map_location='cpu')
        
        # Return specific validation stage batch based on sample_idx
        stage_idx = validation_sample_idx % len(val_batches)
        return val_batches[stage_idx]
    
    def get_stage_config_for_iteration(self, iteration: int) -> Dict:
        """Get stage configuration for given iteration"""
        stages = self.training_config.UNMASKING_STAGES
        if not stages:
            return None
        
        # Simple mapping: divide iterations evenly across stages
        max_iters = 8000  # Default, should be configurable
        iterations_per_stage = max_iters // len(stages)
        stage_idx = min(iteration // iterations_per_stage, len(stages) - 1)
        
        return stages[stage_idx]
    
    def _find_batch_file_for_iteration(self, iteration: int, split: str) -> str:
        """Find the appropriate batch file for given iteration"""
        # Round down to nearest prepared iteration
        eval_interval = 200  # This should come from dataset config
        file_iteration = (iteration // eval_interval) * eval_interval
        filename = f"{split}_iter_{file_iteration:04d}.pt"
        filepath = os.path.join(self.data_dir, 'prepared_batches', filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prepared batch file not found: {filepath}")
        
        return filepath
    
    def _load_batch_file_to_cache(self, batch_file: str):
        """Load batch file samples into runtime cache"""
        # Load prepared data as-is, no modification
        prepared_samples = torch.load(batch_file, map_location='cpu')
        
        # Add individual samples to cache for serving
        if isinstance(prepared_samples, (list, tuple)) and len(prepared_samples) == 3:
            # Single batch format: (x, y, mask)
            x, y, mask = prepared_samples
            batch_size = x.size(0)
            for i in range(batch_size):
                sample = (x[i:i+1], y[i:i+1], mask[i:i+1])
                try:
                    self._batch_cache.put_nowait(sample)
                except:
                    break  # Cache full
        else:
            # Multiple samples format: list of (x, y, mask) samples
            for sample in prepared_samples:
                try:
                    self._batch_cache.put_nowait(sample)
                except:
                    break  # Cache full
    
    def _collect_samples_for_batch(self, target_batch_size: int, iteration: int, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect samples from cache and additional files to form requested batch size"""
        collected_x, collected_y, collected_mask = [], [], []
        
        # Collect samples from current cache
        while len(collected_x) < target_batch_size:
            try:
                sample_x, sample_y, sample_mask = self._batch_cache.get_nowait()
                collected_x.append(sample_x)
                collected_y.append(sample_y)
                collected_mask.append(sample_mask)
            except:
                # Cache empty, need to load more data
                break
        
        # If we need more samples, load additional files
        file_offset = 0
        while len(collected_x) < target_batch_size:
            try:
                # Find next batch file (increment iteration to get more data)
                next_batch_file = self._find_batch_file_for_iteration(iteration + file_offset, split)
                if next_batch_file == self._current_batch_file:
                    file_offset += 200  # Skip to next eval_interval
                    continue
                    
                # Load more samples
                self._load_batch_file_to_cache(next_batch_file)
                self._current_batch_file = next_batch_file
                
                # Collect newly loaded samples
                while len(collected_x) < target_batch_size:
                    try:
                        sample_x, sample_y, sample_mask = self._batch_cache.get_nowait()
                        collected_x.append(sample_x)
                        collected_y.append(sample_y)
                        collected_mask.append(sample_mask)
                    except:
                        break  # This file exhausted
                        
                file_offset += 200  # Move to next potential file
                
            except FileNotFoundError:
                # No more prepared files available
                if len(collected_x) == 0:
                    raise RuntimeError(f"No prepared data available for iteration {iteration}")
                break  # Use what we have
        
        # Combine collected samples into batch
        batch_x = torch.cat(collected_x[:target_batch_size], dim=0)
        batch_y = torch.cat(collected_y[:target_batch_size], dim=0)
        batch_mask = torch.cat(collected_mask[:target_batch_size], dim=0)
        
        return batch_x, batch_y, batch_mask