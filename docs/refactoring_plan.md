# Data-Training Decoupling Refactoring Plan

## Status: ✅ COMPLETED - All validation batch construction issues resolved

## Problem Analysis

The current architecture has significant coupling issues between data generation and training logic that violate proper separation of concerns:

### Current Issues

1. **Massive `get_batch` function** (~200 lines in `training_utils/batch_generation.py`):
   - Handles data loading from .bin files
   - Applies masking strategies  
   - Creates validation sets on-the-fly
   - Manages stage progression
   - Handles prefetching and caching
   - Applies label smoothing
   - Contains dataset-specific logic mixed with generic utilities

2. **Misplaced configuration**:
   - `unmasking_stages` and `validation_stages` are defined in training configs (`config/shkspr_char_diff/optimal5.py`) 
   - These are dataset-specific configurations, not training hyperparameters
   - Same stages need to be duplicated across different training experiments

3. **Overly simple data preparation**:
   - `data/shakespeare_char/prepare.py` only converts text to tokens
   - No pre-computation of training-specific data structures
   - All complex processing happens at training time

4. **Training script complexity**:
   - `train_run.py` must understand masking strategies, stage definitions, validation creation
   - Should only focus on training loop mechanics
   - Contains dataset-specific logic that should be abstracted

## Proposed Architecture

### Core Principle: Data Preparation vs Training Execution

**Data Layer**: Responsible for all dataset-specific logic, pre-computation, and data structure creation
**Training Layer**: Responsible only for loading pre-prepared data and executing training loops

### 1. Dataset-Centric Configuration Structure

```
data/
├── shakespeare_char/                # Original dataset (unchanged)
│   ├── input.txt
│   ├── train.bin, val.bin
│   ├── meta.pkl
│   └── prepare.py
├── shakespeare_char_diffusion/      # NEW: Enhanced dataset for diffusion training
│   ├── input.txt                    # Symlink to ../shakespeare_char/input.txt
│   ├── train.bin, val.bin          # Generated tokenized data  
│   ├── meta.pkl                    # Enhanced with training metadata
│   ├── training_config.py          # Dataset-specific training configurations
│   ├── prepare.py                  # Enhanced preparation script
│   ├── data_utils.py               # Dataset-specific utilities
│   ├── cached_data/                # Runtime caches for efficient loading
│   │   ├── paragraph_boundaries.npy
│   │   └── valid_indices.npy
│   └── prepared_batches/           # Pre-generated training/validation data
│       ├── train_iter_0000.pt      # Data for iteration 0
│       ├── train_iter_0200.pt      # Data for iteration 200
│       ├── val_iter_0000.pt        # Validation data for iteration 0
│       └── ...                     # Abstracts away stage concepts
└── other_dataset_diffusion/
    ├── similar structure...
```

### 2. Enhanced Meta Information

Extend `meta.pkl` to include:
```python
meta = {
    # Current fields
    'vocab_size': 65,
    'itos': {...},
    'stoi': {...},
    
    # New fields
    'extended_vocab_size': 80,        # With special tokens
    'special_tokens': {
        'mask_token_id': 65,
        'wrong_token_id': 66,
        'remask_good_id': 67,
        'remask_wrong_id': 68,
    },
    'dataset_type': 'character',       # vs 'subword', 'word'
    'block_size': 1024,               # FIXED block size constraint for this dataset
    'supported_model_modes': ['language_model', 'token_classifier'],
    'default_training_config': 'unmasking_stages_v1'
}
```

### 3. Pre-computed Data Structures

Instead of generating data on-the-fly:
- Pre-generate validation sets for all stages
- Create indexed data structures for efficient sampling
- Cache expensive computations like paragraph boundaries
- Store training stage definitions with the dataset

## Detailed Implementation Plan

### Phase 1: Dataset Configuration Migration

**1.1 Create new dataset folder `data/shakespeare_char_diffusion/`**
- Copy structure from `shakespeare_char` 
- Symlink `input.txt` to avoid duplication
- This preserves the original dataset unchanged

**1.2 Create `data/shakespeare_char_diffusion/training_config.py`**
```python
# Moved from config/shkspr_char_diff/optimal5.py
UNMASKING_STAGES = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    # ... all 8 training stages from optimal5.py
]

VALIDATION_STAGES = [
    # ... all 14 validation stages from optimal5.py
]

# Dataset-specific parameters (NOT training hyperparameters)
BLOCK_SIZE = 1024                    # CONSTRAINT: Must match this exactly
USE_PARAGRAPH_BOUNDARIES = False
USE_ALL_STAGES_FOR_TRAINING = True

# Cache configuration
BATCH_CACHE_SIZE = 1000              # Number of batches to cache in memory
PREFETCH_BUFFER_SIZE = 10            # Runtime prefetch buffer
```

**1.3 Create `data/shakespeare_char_diffusion/data_utils.py`**
```python
"""Dataset-specific utilities for Shakespeare character diffusion training"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from ..training_utils.masking_strategies import apply_stage_masking

def validate_block_size(requested_block_size: int, dataset_block_size: int):
    """Validate that training block_size matches dataset constraint"""
    if requested_block_size != dataset_block_size:
        raise ValueError(f"Block size mismatch: dataset requires {dataset_block_size}, got {requested_block_size}")

def generate_batch_for_iteration(data: np.memmap, valid_indices: np.ndarray, 
                                iteration: int, batch_size: int, block_size: int,
                                stages_config: List[Dict], meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch for specific iteration using stage progression logic"""
    # Implementation moved from batch_generation.py
    pass

def create_iteration_mapping(max_iters: int, stages_config: List[Dict]) -> Dict[int, int]:
    """Create mapping from iteration number to stage index"""
    # Business logic for stage progression
    pass
```

**1.4 Enhance `data/shakespeare_char_diffusion/prepare.py`**
```python
import os
import pickle
import numpy as np
import torch
from typing import Dict, Any
from training_config import UNMASKING_STAGES, VALIDATION_STAGES, BLOCK_SIZE
from data_utils import generate_batch_for_iteration, create_iteration_mapping

def create_enhanced_meta() -> Dict[str, Any]:
    """Create enhanced metadata with training information"""
    # Load original meta from parent dataset
    parent_meta_path = os.path.join('..', 'shakespeare_char', 'meta.pkl')
    with open(parent_meta_path, 'rb') as f:
        base_meta = pickle.load(f)
    
    # Add training-specific metadata
    enhanced_meta = {
        **base_meta,
        'extended_vocab_size': base_meta['vocab_size'] + 15,
        'special_tokens': {
            'mask_token_id': base_meta['vocab_size'],
            'wrong_token_id': base_meta['vocab_size'] + 1,
            'remask_good_id': base_meta['vocab_size'] + 2,
            'remask_wrong_id': base_meta['vocab_size'] + 3,
        },
        'dataset_type': 'character',
        'block_size': BLOCK_SIZE,  # CONSTRAINT: Fixed for this dataset
        'supported_model_modes': ['language_model', 'token_classifier'],
        'training_stages': len(UNMASKING_STAGES),
        'validation_stages': len(VALIDATION_STAGES),
        'batch_cache_size': 1000,
    }
    return enhanced_meta

def pre_generate_training_data(max_iters: int, eval_interval: int):
    """Pre-generate training and validation data for all iterations"""
    os.makedirs('prepared_batches', exist_ok=True)
    
    # Create iteration-to-stage mapping
    iteration_mapping = create_iteration_mapping(max_iters, UNMASKING_STAGES)
    
    # Generate data for training iterations and validation points
    for iteration in range(0, max_iters, eval_interval):
        # Generate training batch
        train_batch = generate_batch_for_iteration(
            data, valid_indices, iteration, batch_size=192, 
            block_size=BLOCK_SIZE, stages_config=UNMASKING_STAGES, meta=meta
        )
        torch.save(train_batch, f'prepared_batches/train_iter_{iteration:04d}.pt')
        
        # Generate validation batches for all validation stages
        val_batches = []
        for val_stage_idx, val_stage in enumerate(VALIDATION_STAGES):
            val_batch = generate_validation_batch_for_stage(val_stage, iteration)
            val_batches.append(val_batch)
        torch.save(val_batches, f'prepared_batches/val_iter_{iteration:04d}.pt')

def main():
    # Setup directory structure
    os.makedirs('cached_data', exist_ok=True)
    os.makedirs('prepared_batches', exist_ok=True)
    
    # Create enhanced metadata
    meta = create_enhanced_meta()
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    # Pre-compute expensive operations
    prepare_cached_data()
    
    # Pre-generate training data (example for 8000 iterations)
    pre_generate_training_data(max_iters=8000, eval_interval=200)

if __name__ == '__main__':
    main()
```

**1.5 Validation of dataset constraints**
Add validation in training script:
```python
# In train_run.py, after loading dataset config
dataset_config = DatasetConfig(dataset)
if block_size != dataset_config.meta['block_size']:
    raise ValueError(f"Block size mismatch: dataset '{dataset}' requires block_size={dataset_config.meta['block_size']}, got {block_size}")
```

### Phase 2: Dataset Interface and Runtime Caching

**2.1 Create `training_utils/dataset_interface.py`**
```python
"""Generic dataset interface for all diffusion datasets"""
import os
import pickle
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from queue import Queue
import threading

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
        with open(meta_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_training_config(self) -> Any:
        import importlib.util
        config_path = os.path.join(self.data_dir, 'training_config.py')
        spec = importlib.util.spec_from_file_location("training_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module
    
    def _validate_dataset(self):
        """Validate dataset has all required components"""
        required_files = ['meta.pkl', 'training_config.py', 'train.bin', 'val.bin']
        for file in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file)):
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
        return self._collect_samples_for_batch(batch_size, iteration)
    
    def get_validation_batch(self, iteration: int, validation_sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get validation batch for specific iteration and sample index"""
        batch_file = self._find_batch_file_for_iteration(iteration, 'val')
        val_batches = torch.load(batch_file)
        
        # Return specific validation stage batch based on sample_idx
        stage_idx = validation_sample_idx % len(val_batches)
        return val_batches[stage_idx]
    
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
        prepared_samples = torch.load(batch_file)
        
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
    
    def _collect_samples_for_batch(self, target_batch_size: int, iteration: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                next_batch_file = self._find_batch_file_for_iteration(iteration + file_offset, 'train')
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
```

**2.2 Create `training_utils/batch_cache.py`**
```python
"""Runtime batch caching utilities - separate from dataset preparation"""
import torch
from queue import Queue
from typing import Tuple, Optional
import threading

class BatchCache:
    """Runtime cache for efficient batch serving"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = Queue(maxsize=cache_size)
        self.cache_size = cache_size
        self._lock = threading.Lock()
    
    def get_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get batch from cache if available"""
        try:
            return self.cache.get_nowait()
        except:
            return None
    
    def put_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """Add batch to cache"""
        try:
            self.cache.put_nowait(batch)
        except:
            pass  # Cache full, skip
    
    def clear(self):
        """Clear the cache"""
        with self._lock:
            while not self.cache.empty():
                try:
                    self.cache.get_nowait()
                except:
                    break
```

**2.3 Move reusable utilities to `training_utils/data_loading_utils.py`**
```python
"""Generic data loading utilities - reusable across datasets"""
import numpy as np
import torch
from typing import Tuple

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
```

### Phase 3: Decouple Batch Generation

**3.1 Drastically simplify `training_utils/batch_generation.py`**

Current (~325 lines) → Target (~30 lines):
```python
"""Simplified batch generation - just delegates to dataset interface"""
from .dataset_interface import DatasetConfig
from .training_config import TrainingContext

def get_batch(split: str, ctx: TrainingContext, validation_sample_idx=None):
    """Simplified batch loader - delegates to dataset interface"""
    if split == 'val':
        return ctx.dataset_config.get_validation_batch(ctx.iter_num, validation_sample_idx or 0)
    else:
        return ctx.dataset_config.get_training_batch(ctx.iter_num, ctx.batch_size, ctx.block_size)

# Remove all other functions:
# - get_batch_unmasking (325 lines) → moved to dataset preparation
# - get_batch_remasking_binary → moved to dataset preparation  
# - All caching logic → moved to dataset_interface.py
# - All masking logic → moved to dataset preparation
# - All validation set creation → moved to dataset preparation
```

**3.2 Update `training_utils/training_config.py`**
```python
# Add dataset_config field to TrainingContext
class TrainingContext:
    def __init__(self, dataset_config: DatasetConfig, **kwargs):
        self.dataset_config = dataset_config
        
        # Remove dataset-specific fields that now come from dataset_config:
        # - unmasking_stages (now in dataset_config.training_config.UNMASKING_STAGES)
        # - validation_stages (now in dataset_config.training_config.VALIDATION_STAGES) 
        # - mask_token_id (now in dataset_config.meta['special_tokens']['mask_token_id'])
        # - extended_vocab_size (now in dataset_config.meta['extended_vocab_size'])
        # - use_paragraph_boundaries (now in dataset_config.training_config.USE_PARAGRAPH_BOUNDARIES)
        # - use_all_stages_for_training (now in dataset_config.training_config.USE_ALL_STAGES_FOR_TRAINING)
        
        # Keep training hyperparameters:
        self.batch_size = kwargs['batch_size']
        self.block_size = kwargs['block_size'] 
        self.max_iters = kwargs['max_iters']
        self.device = kwargs['device']
        self.device_type = kwargs['device_type']
        
        # Keep training logic parameters (loss computation):
        self.weight_loss_by_mask_ratio = kwargs.get('weight_loss_by_mask_ratio', False)
        self.enable_entropy_penalty = kwargs.get('enable_entropy_penalty', False) 
        self.max_entropy_penalty = kwargs.get('max_entropy_penalty', 0.5)
        self.entropy_penalty_start_iter = kwargs.get('entropy_penalty_start_iter', 6000)
        self.uncertainty_factor = kwargs.get('uncertainty_factor', 0.0)
        # ... other training params
    
    def get_current_stage_config(self):
        """Delegate to dataset configuration"""
        return self.dataset_config.get_stage_config_for_iteration(self.iter_num)
```

**3.3 Keep minimal masking utilities in `training_utils/masking_strategies.py`**

Remove dataset-specific functions, keep only generic utilities:
```python
"""Generic masking utilities - no dataset-specific logic"""

# Keep these generic functions:
def apply_bert_style_corruption_gpu(x, mask, mask_token_id, meta_vocab_size):
    # Generic 80/10/10 corruption
    
def get_progressive_validation_iterations(eval_iters, max_iters):
    # Generic iteration calculation

# REMOVE these dataset-specific functions:
# - apply_random_masking_gpu → move to data/shakespeare_char_diffusion/data_utils.py
# - apply_stage_masking → move to data/shakespeare_char_diffusion/data_utils.py
# - apply_target_driven_sticky_masking_gpu → move to data/shakespeare_char_diffusion/data_utils.py
# - apply_sticky_corruption_gpu → move to data/shakespeare_char_diffusion/data_utils.py
# - apply_corruption_gpu → move to data/shakespeare_char_diffusion/data_utils.py
# - All shakespeare-specific logic → move to dataset folder
```

### Phase 4: Clean Training Scripts

**4.1 Major simplification of `train_run.py`**

**Remove these sections (lines ~220-300):**
```python
# REMOVE: Stage configuration creation
unmasking_stage_objects = []
for stage in unmasking_stages:
    # ... 20+ lines of stage creation logic

# REMOVE: Validation stage configuration  
validation_stage_objects = []
for stage in validation_stages:
    # ... 20+ lines of validation stage creation

# REMOVE: Complex TrainingContext creation with all dataset-specific params
training_ctx = TrainingContext(
    training_type=training_type,
    # ... 20+ dataset-specific parameters
)
```

**Replace with simple dataset loading:**
```python
# NEW: Simple dataset configuration loading
dataset_config = DatasetConfig(dataset)
training_ctx = TrainingContext(
    dataset_config=dataset_config,
    # Training hyperparameters:
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr,
    # Training logic parameters:
    weight_loss_by_mask_ratio=weight_loss_by_mask_ratio,
    enable_entropy_penalty=enable_entropy_penalty,
    max_entropy_penalty=max_entropy_penalty,
    entropy_penalty_start_iter=entropy_penalty_start_iter,
    uncertainty_factor=uncertainty_factor
)
```

**4.2 Create new config file `config/shakespeare_diffusion/experiment1.py`**

Replace `config/shkspr_char_diff/optimal5.py` with clean training-only config:
```python
"""Training configuration for Shakespeare diffusion experiment"""

# Dataset selection (loads all dataset-specific settings)
dataset = 'shakespeare_char_diffusion'

# I/O
out_dir = 'out'
init_from = 'scratch'
wandb_log = True
wandb_project = 'experiments_diffusion'  
wandb_run_name = 'shkspr_char_diff_experiment1'

# Training hyperparameters ONLY
batch_size = 192                    # Any batch_size supported via adaptation
gradient_accumulation_steps = 12
learning_rate = 1e-3
max_iters = 8000
warmup_iters = 2000
lr_decay_iters = 8000
min_lr = 3e-5
weight_decay = 2e-2
grad_clip = 0.0
decay_lr = True

# Training-specific loss computation parameters
weight_loss_by_mask_ratio = True    # Weight loss by sqrt(1.0 / mask_ratio)
enable_entropy_penalty = False     # Apply entropy penalty to loss
max_entropy_penalty = 0.5          # Maximum entropy penalty multiplier
entropy_penalty_start_iter = 6000  # Iteration to start applying entropy penalty
uncertainty_factor = 0.1           # Label smoothing factor for loss computation

# Model architecture 
n_layer = 6
n_head = 6  
n_embd = 384
dropout = 0.2
bias = False
attention_type = 'bidirectional'
use_rope = True

# Training process
eval_interval = 200
log_interval = 20
eval_iters = 20
eval_only = False
always_save_checkpoint = True
compile = True

# MOVED TO DATASET: Data generation configurations moved to data/shakespeare_char_diffusion/
# - unmasking_stages → data/shakespeare_char_diffusion/training_config.py
# - validation_stages → data/shakespeare_char_diffusion/training_config.py  
# - use_paragraph_boundaries → data/shakespeare_char_diffusion/training_config.py
# - use_all_stages_for_training → data/shakespeare_char_diffusion/training_config.py

# KEPT IN TRAINING CONFIG: Training logic parameters (control loss computation)
# - weight_loss_by_mask_ratio → stays in training config (modifies loss during training)
# - enable_entropy_penalty → stays in training config (modifies loss during training)
# - max_entropy_penalty → stays in training config (training hyperparameter)
# - entropy_penalty_start_iter → stays in training config (training schedule)
# - uncertainty_factor → stays in training config (modifies loss via label smoothing)
```

**4.3 Update training script initialization**

**In `train_run.py` lines ~162-200, replace:**
```python
# OLD: Complex dataset loading
ddp = int(os.environ.get('RANK', -1)) != -1
# ... DDP setup
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    # ... complex vocab setup
```

**With simple dataset configuration:**
```python
# NEW: Simple dataset initialization
from training_utils.dataset_interface import DatasetConfig

# Initialize dataset configuration
dataset_config = DatasetConfig(dataset)
dataset_config.validate_training_config(block_size, batch_size)

# DDP setup (unchanged)
ddp = int(os.environ.get('RANK', -1)) != -1
# ... DDP setup remains the same

# All dataset info now comes from dataset_config
meta_vocab_size = dataset_config.meta['vocab_size']
extended_vocab_size = dataset_config.meta['extended_vocab_size']
mask_token_id = dataset_config.meta['special_tokens']['mask_token_id']
```

**4.4 Simplify model initialization (lines ~311-320)**

**Replace:**
```python
# OLD: Manual vocab size calculation
model_args['vocab_size'] = extended_vocab_size if meta_vocab_size is not None else 65 + 15
```

**With:**
```python
# NEW: Use dataset configuration
model_args['vocab_size'] = dataset_config.meta['extended_vocab_size']
```

### Phase 5: Implementation Details and Specific Code Changes

**5.1 Exact file modifications required:**

1. **New files to create:**
   - `data/shakespeare_char_diffusion/` (entire new dataset folder)
   - `data/shakespeare_char_diffusion/training_config.py`
   - `data/shakespeare_char_diffusion/data_utils.py` 
   - `data/shakespeare_char_diffusion/prepare.py`
   - `training_utils/dataset_interface.py`
   - `training_utils/batch_cache.py`
   - `training_utils/data_loading_utils.py`
   - `config/shakespeare_diffusion/experiment1.py`

2. **Files to drastically simplify:**
   - `training_utils/batch_generation.py`: 325 lines → 30 lines
     - Remove `get_batch_unmasking()` function (150+ lines)
     - Remove `get_batch_remasking_binary()` function (100+ lines)  
     - Replace with simple delegation to dataset interface
   
   - `train_run.py`: Remove ~80 lines of dataset-specific setup
     - Lines 220-300: Remove stage configuration creation
     - Lines 162-200: Simplify dataset loading
     - Lines 311-320: Simplify model initialization

3. **Files to update with specific changes:**
   
   **`training_utils/training_config.py`:**
   - Remove these dataset-specific fields from `TrainingContext.__init__()`:
     ```python
     # REMOVE these dataset-specific parameters:
     unmasking_stages=None,
     validation_stages=None, 
     mask_token_id=None,
     extended_vocab_size=None,
     meta_vocab_size=None,
     use_paragraph_boundaries=False,
     use_all_stages_for_training=False,
     ```
   - Add `dataset_config: DatasetConfig` parameter
   - Update `get_current_stage_config()` to delegate to dataset_config
   - KEEP these training logic parameters:
     ```python
     # KEEP these training-specific parameters:
     weight_loss_by_mask_ratio=False,
     enable_entropy_penalty=False,
     max_entropy_penalty=0.5,
     entropy_penalty_start_iter=6000,
     uncertainty_factor=0.0,
     ```

   **`training_utils/masking_strategies.py`:**
   - Remove lines 16-65: `apply_random_masking_gpu()` 
   - Remove lines 67-161: `apply_target_driven_sticky_masking_gpu()`
   - Remove lines 200-375: `apply_sticky_corruption_gpu()`, `apply_corruption_gpu()`
   - Keep only generic utilities: `apply_bert_style_corruption_gpu()`, `get_progressive_validation_iterations()`

4. **Files with minor updates:**
   - `training_utils/validation_sets.py` - Update imports
   - `training_utils/entropy_utils.py` - No changes needed
   - `model.py` - No changes needed

**5.2 Migration strategy with specific steps:**

**Step 1: Create new dataset (preserves current functionality)**
```bash
# Create new dataset folder
mkdir -p data/shakespeare_char_diffusion/cached_data
mkdir -p data/shakespeare_char_diffusion/prepared_batches

# Copy  raw data
cp ../shakespeare_char/input.txt data/shakespeare_char_diffusion/input.txt

# Create dataset-specific files
# (implement training_config.py, data_utils.py, prepare.py as specified above)
```

**Step 2: Generate prepared data**
```bash
cd data/shakespeare_char_diffusion
python prepare.py  # Pre-generates all training/validation data
```

**Step 3: Create new training utilities**
```python
# Implement dataset_interface.py, batch_cache.py, data_loading_utils.py
# (as specified in Phase 2)
```

**Step 4: Update training infrastructure**
```python 
# Modify training_utils/training_config.py (add dataset_config field)
# Simplify training_utils/batch_generation.py (remove 90% of code)
# Clean training_utils/masking_strategies.py (keep only generic functions)
```

**Step 5: Create new config and update training script**
```python
# Create config/shakespeare_diffusion/experiment1.py
# Update train_run.py imports and initialization
# Test that training still works with new architecture
```

**Step 6: Remove old code**
```python
# Remove unused functions from batch_generation.py
# Remove unused parameters from TrainingContext
# Clean up imports
```

**5.3 Validation of implementation correctness:**

**Before refactoring - baseline behavior:**
```bash
python train_run.py config/shkspr_char_diff/optimal5.py
# Record: loss curves, validation metrics, model performance
```

**After refactoring - should match exactly:**
```bash  
python train_run.py config/shakespeare_diffusion/experiment1.py
# Verify: identical loss curves, validation metrics, model performance
```

**Key validation points:**
- ✅ Same stage progression behavior
- ✅ Identical masking ratios at each iteration
- ✅ Same validation set composition across stages
- ✅ Identical loss computation and optimization
- ✅ All entropy penalties and label smoothing preserved
- ✅ Same performance with any batch_size (16, 32, 64, 192, 256)
- ✅ Block size constraint properly enforced

**5.4 Performance improvements expected:**
- Faster training startup (pre-computed validation sets)
- Reduced memory usage (efficient caching)
- Faster iteration times (simplified batch generation)
- **Flexible batch sizes**: Any batch_size supported by loading exactly the required number of samples from prepared files, no shuffling/repeating
- **Deterministic data serving**: Samples served exactly as prepared, maintaining data integrity

## Benefits of This Refactoring

### 1. Clear Separation of Concerns
- **Data preparation**: Handles all dataset-specific logic
- **Training execution**: Focuses purely on training mechanics
- **Configuration**: Training vs dataset parameters clearly separated

### 2. Easy Dataset Addition
Adding a new dataset requires only:
- Create `data/new_dataset/` folder
- Implement `prepare.py` and `training_config.py`
- All training scripts work immediately

### 3. Better Performance
- Pre-computed validation sets eliminate runtime generation
- Cached data structures reduce repeated computations
- Optimized data loading without complex logic

### 4. Maintainable Codebase
- Small, focused functions with single responsibilities
- No more 200-line `get_batch` function
- Clear interfaces between components

### 5. Reusable Components
- Training utilities work with any properly prepared dataset
- Generic masking strategies
- Modular architecture allows easy testing and extension

## Validation of Correctness

All current functionalities will be preserved:
- ✅ Stage-based unmasking training
- ✅ Validation across all stages  
- ✅ Label smoothing
- ✅ Entropy penalties
- ✅ Progressive validation
- ✅ All existing training modes

The key difference: these features will be implemented through clean interfaces rather than monolithic functions.