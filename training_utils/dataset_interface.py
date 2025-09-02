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
        required_files = ['meta.pkl', 'train.bin', 'val.bin']
        for file in required_files:
            filepath = os.path.join(self.data_dir, file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Dataset {self.dataset_name} missing required file: {file}")

        # Check if this is a training-specific dataset (has training_config.py and prepared batches)
        # or a general dataset (just has basic data files)
        training_config_path = os.path.join(self.data_dir, 'training_config.py')
        prepared_dir = os.path.join(self.data_dir, 'prepared_batches')

        if os.path.exists(training_config_path):
            # Training-specific dataset - validate prepared batches exist
            if not os.path.exists(prepared_dir):
                raise FileNotFoundError(f"Dataset {self.dataset_name} has training_config.py but missing prepared_batches directory. Run prepare.py first.")
        else:
            # General dataset - no training_config.py required
            print(f"Dataset {self.dataset_name} is a general dataset (no training_config.py found)")

        # Validate supported model modes
        supported_modes = self.meta.get('supported_model_modes', ['language_model'])
        if not isinstance(supported_modes, list) or len(supported_modes) == 0:
            raise ValueError(f"Dataset {self.dataset_name} meta.pkl must specify 'supported_model_modes' as a non-empty list")
    
    def validate_training_config(self, block_size: int, model_mode: str, batch_size: Optional[int] = None):
        """Validate training parameters against dataset constraints - called once at startup"""
        if block_size != self.meta['block_size']:
            raise ValueError(f"Block size mismatch: dataset '{self.dataset_name}' requires {self.meta['block_size']}, got {block_size}")

        # Validate model mode is supported by this dataset
        supported_modes = self.meta.get('supported_model_modes', ['language_model'])
        if model_mode not in supported_modes:
            raise ValueError(f"Dataset '{self.dataset_name}' does not support model_mode '{model_mode}'. Supported modes: {supported_modes}")

        # Validate data shapes if available in metadata
        if 'data_shapes' in self.meta:
            shapes = self.meta['data_shapes']
            print(f"Dataset data shapes - X: {shapes.get('X')}, Y: {shapes.get('Y')}, mask: {shapes.get('mask')}")

        print(f"✓ Dataset '{self.dataset_name}' validated for model_mode '{model_mode}' with block_size {block_size}")

        # Any batch_size should be supported through dynamic loading
        if batch_size and batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}")
    
    def get_training_batch(self, iteration: int, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get training batch for specific iteration with runtime caching"""
        # Validate block size only (model_mode validation happens once at startup)
        if block_size != self.meta['block_size']:
            raise ValueError(f"Block size mismatch: dataset '{self.dataset_name}' requires {self.meta['block_size']}, got {block_size}")

        # Check if this dataset has training-specific preparation
        if hasattr(self.training_config, 'UNMASKING_STAGES'):
            # Training-specific dataset with prepared batches
            return self._collect_samples_for_batch(batch_size, iteration, 'train')
        else:
            # General dataset - generate batch on-the-fly using pre-determined format
            return self._generate_batch_from_data(batch_size, block_size, 'train')
    
    def load_validation_set(self, eval_iters: int, device: str = 'cpu') -> List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Load fixed validation set ONCE at training start - dataset provides 'buffet', training decides consumption"""
        print("Loading fixed validation set...")

        # Load validation metadata
        validation_dir = os.path.join(self.data_dir, 'validation')
        validation_meta_path = os.path.join(validation_dir, 'validation_meta.pkl')

        if not os.path.exists(validation_meta_path):
            raise FileNotFoundError(f"Validation metadata not found: {validation_meta_path}. Run prepare.py first.")

        with open(validation_meta_path, 'rb') as f:
            validation_meta = pickle.load(f)

        validation_batches = []

        # Training script decides how many samples it needs per stage
        samples_needed_per_stage = eval_iters

        for stage_idx in range(validation_meta['num_stages']):
            stage_samples = []
            samples_collected = 0
            file_idx = 0

            while samples_collected < samples_needed_per_stage:
                # Load validation file
                file_path = os.path.join(validation_dir, f"stage_{stage_idx}_file_{file_idx}.pt")

                if not os.path.exists(file_path):
                    # If we've exhausted available files, repeat from beginning
                    if file_idx == 0:
                        raise FileNotFoundError(f"No validation files found for stage {stage_idx}")
                    file_idx = 0
                    file_path = os.path.join(validation_dir, f"stage_{stage_idx}_file_{file_idx}.pt")

                file_samples = torch.load(file_path, map_location=device)

                # Take what we need and ensure all tensors are on the correct device
                remaining_needed = samples_needed_per_stage - samples_collected
                samples_to_take = min(len(file_samples), remaining_needed)

                # Move each sample's tensors to the correct device
                for sample in file_samples[:samples_to_take]:
                    if isinstance(sample, (list, tuple)) and len(sample) == 3:
                        x, y, mask = sample
                        # Ensure all tensors are on the correct device
                        x = x.to(device) if hasattr(x, 'to') else x
                        y = y.to(device) if hasattr(y, 'to') else y
                        mask = mask.to(device) if hasattr(mask, 'to') else mask
                        stage_samples.append((x, y, mask))
                    else:
                        stage_samples.append(sample)

                samples_collected += samples_to_take
                file_idx += 1

            validation_batches.append(stage_samples)
            print(f"  Loaded {len(stage_samples)} validation samples for stage {stage_idx+1}/{validation_meta['num_stages']}")

        print(f"✓ Validation set loaded: {validation_meta['num_stages']} stages, {eval_iters} samples per stage")
        return validation_batches

    def get_validation_sample(self, validation_batches: List[List], stage_idx: int, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get specific validation sample from pre-loaded validation set"""
        if stage_idx >= len(validation_batches):
            raise IndexError(f"Stage index {stage_idx} out of range (max: {len(validation_batches)-1})")

        stage_samples = validation_batches[stage_idx]
        if sample_idx >= len(stage_samples):
            # Wrap around if we need more samples than available
            sample_idx = sample_idx % len(stage_samples)

        return stage_samples[sample_idx]
    
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

    def _generate_batch_from_data(self, batch_size: int, block_size: int, split: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate batch on-the-fly for general datasets using pre-determined format from metadata"""
        from .data_loading_utils import load_memmap_data, sample_indices_random

        # Load data
        data_path = os.path.join(self.data_dir, f'{split}.bin')
        data = load_memmap_data(data_path)

        # Sample indices
        indices = sample_indices_random(len(data), batch_size, block_size)

        # Load sequences
        ix_expanded = indices[:, None] + np.arange(block_size)[None, :]
        x_data = data[ix_expanded].astype(np.int64)
        x = torch.from_numpy(x_data)

        # Use the data format determined during dataset preparation/validation
        # For general datasets, default to language modeling format
        supported_modes = self.meta.get('supported_model_modes', ['language_model'])
        primary_mode = supported_modes[0]  # Use the first supported mode as the format

        if primary_mode == 'language_model':
            # Standard language modeling: predict next token
            y = x[:, 1:].contiguous()  # Shift by 1 for next token prediction
            x = x[:, :-1].contiguous()  # Remove last token from input
            mask = torch.ones_like(x)  # No masking for standard LM
        else:
            # For other modes, the dataset should have training-specific preparation
            raise NotImplementedError(f"General dataset {self.dataset_name} with mode {primary_mode} requires training-specific preparation.")

        return x, y, mask