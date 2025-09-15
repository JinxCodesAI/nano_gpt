# Sequence Scorer Dataset Implementation Plan

## Overview

This document outlines the implementation of a sequence scoring dataset that generates partially synthetic text using a pre-trained MLM model. The dataset will:

1. Start with original text input
2. Apply masking (similar to char_diffusion)
3. Use a pre-trained MLM model to predict masked tokens
4. Create "synthetic" text by replacing masks with model predictions
5. Generate sequence-level targets (0-1) representing synthenticity ratio: `target = 1 - masking_ratio`

## Current System Analysis

### Existing Architecture Components

1. **DataProviderBase** (`data/common/provider_base.py`):
   - Handles filesystem queue management
   - Provides deterministic RNG seeding
   - Manages batch file creation and atomic writes

2. **CharDiffusionProvider** (`data/char_diffusion/prepare_streaming.py`):
   - Character-level Shakespeare dataset
   - BERT-style masking (80% [MASK], 10% random, 10% unchanged)
   - Creates MLM training data with corrupted input and target labels

3. **DatasetConsumer** (`dataset_consumer.py`):
   - Consumes streaming batch files
   - Supports flexible batch schemas
   - Handles both (X,Y) and dict returns

4. **Prepare System** (`prepare.py`):
   - Unified config loading
   - Provider discovery by convention
   - Validation integration

## Implementation Plan

### Phase 1: Create Base Sequence Scoring Provider

#### 1.1 Directory Structure
Create the new dataset directory:
```
data/
├── sequence_scorer/
│   ├── __init__.py
│   ├── prepare_streaming.py        # Main provider implementation
│   ├── mlm_inference.py           # MLM model loading and inference
│   ├── synthetic_generation.py    # Synthetic text generation logic
│   ├── config/                    # Stage-based configurations (same as char_diffusion)
│   │   ├── __init__.py
│   │   ├── simple.py              # Simple masking configuration
│   │   └── complex.py             # Advanced multi-stage configuration
│   └── input.txt                  # Raw text data (Shakespeare)
```

#### 1.2 Provider Class Structure

**File**: `data/sequence_scorer/prepare_streaming.py`

```python
from typing import Dict, Any, Optional, List
import torch
import os
from data.common.provider_base import DataProviderBase
from .mlm_inference import MLMInferenceEngine
from .synthetic_generation import create_synthetic_text, create_stage_synthetic_text

class SequenceScorerProvider(DataProviderBase):
    def __init__(self, *args, **kwargs):
        # Extract sequence scorer specific config
        self.mlm_checkpoint_path = kwargs.pop('mlm_checkpoint_path')
        self.cls_token_id = kwargs.pop('cls_token_id', 0)  # [CLS] token position
        
        # Stage-based configuration (same as char_diffusion)
        self.use_all_stages_for_training = kwargs.pop('use_all_stages_for_training', None)
        self.unmasking_stages = kwargs.pop('unmasking_stages', None)
        self.validation_stages = kwargs.pop('validation_stages', None)
        
        # Fallback simple configuration for non-stage mode
        self.mask_probability_range = kwargs.pop('mask_probability_range', (0.1, 0.8))
        
        super().__init__(*args, **kwargs)
        
        # Initialize MLM inference engine
        self.mlm_engine = MLMInferenceEngine(
            checkpoint_path=self.mlm_checkpoint_path,
            device='cpu',  # Use CPU for data generation
            verbose=self.verbose
        )
        
        # Load text data and create vocabulary (similar to char_diffusion)
        self._load_text_data()
        
        # Validate and initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()
        
    def _validate_stage_config(self):
        """Validate stage configuration and raise exceptions for unsupported options."""
        if self.use_all_stages_for_training is not None:
            if not self.use_all_stages_for_training:
                raise NotImplementedError("use_all_stages_for_training=False is not yet implemented")
            
            if not self.unmasking_stages:
                raise ValueError("unmasking_stages must be provided when use_all_stages_for_training=True")
                
            if not self.validation_stages:
                raise ValueError("validation_stages must be provided when use_all_stages_for_training=True")
    
    def _initialize_stage_distribution(self):
        """Initialize stage distribution for batch generation."""
        if self.use_all_stages_for_training:
            # Calculate how many batches of each stage type to generate per file
            self.train_stage_distribution = self._calculate_stage_distribution(self.unmasking_stages)
            self.val_stage_distribution = self._calculate_stage_distribution(self.validation_stages)
        else:
            self.train_stage_distribution = None
            self.val_stage_distribution = None
    
    def _calculate_stage_distribution(self, stages: List[Dict]) -> List[Dict]:
        """
        Calculate how many samples of each stage type to generate per file.
        Same logic as char_diffusion provider.
        """
        total_stages = len(stages)
        total_samples = self.batches_per_file * self.batch_size
        samples_per_stage = total_samples // total_stages
        remainder = total_samples % total_stages
        
        distribution = []
        for i, stage in enumerate(stages):
            # Distribute remainder across first stages
            count = samples_per_stage + (1 if i < remainder else 0)
            if count > 0:  # Only include stages with samples
                distribution.append({
                    'config': stage,
                    'count': count
                })
        
        return distribution
        
    def build_meta(self) -> Dict:
        """Build metadata for sequence scoring task."""
        return {
            "dataset_name": "sequence_scorer",
            "training_type": "SEQUENCE_SCORING",
            "vocab_size": self.vocab_size,
            "cls_token_id": self.cls_token_id,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {
                    "name": "input_ids", 
                    "dtype": "int64", 
                    "shape": [self.block_size], 
                    "role": "input"
                },
                {
                    "name": "targets", 
                    "dtype": "float32", 
                    "shape": [], 
                    "role": "target"
                },
            ],
        }
    
    def sample_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        """Generate batch with synthetic text and synthenticity scores."""
        if self.use_all_stages_for_training:
            # For stage-based generation, we need to generate all samples for the file at once
            # This method will be called by the base class for each batch, but we need to 
            # coordinate across all batches in the file. We'll handle this differently.
            raise NotImplementedError("Stage-based sampling requires file-level generation")
        else:
            return self._sample_default_batch(split, rng)
    
    def produce_one_file(self, split: str, seq: int) -> None:
        """Override to handle stage-based generation at file level."""
        if self.use_all_stages_for_training:
            self._produce_stage_based_file(split, seq)
        else:
            # Use default file production for non-stage-based generation
            super().produce_one_file(split, seq)
```

### Phase 2: MLM Inference Engine

#### 2.1 Model Loading and Setup

**File**: `data/sequence_scorer/mlm_inference.py`

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from model import GPT, GPTConfig, ModelMode

class MLMInferenceEngine:
    """Handles loading and inference with pre-trained MLM models."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu', verbose: bool = False):
        self.device = device
        self.verbose = verbose
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract model configuration
        if 'model_args' in checkpoint:
            model_args = checkpoint['model_args']
        else:
            raise ValueError("Checkpoint missing model_args")
            
        # Ensure model is in language modeling mode
        model_args['mode'] = ModelMode.LANGUAGE_MODEL
        
        # Create and load model
        config = GPTConfig(**model_args)
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()
        
        # Extract vocabulary info from meta
        if 'meta' in checkpoint:
            meta = checkpoint['meta']
            self.vocab_size = meta.get('vocab_size')
            self.mask_token_id = meta.get('mask_token_id')
            self.stoi = meta.get('stoi', {})
            self.itos = meta.get('itos', {})
        else:
            raise ValueError("Checkpoint missing metadata")
            
        if self.verbose:
            print(f"MLMInferenceEngine initialized:")
            print(f"  Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
            print(f"  Vocab size: {self.vocab_size}")
            print(f"  Mask token ID: {self.mask_token_id}")
    
    @torch.no_grad()
    def predict_masked_tokens(
        self, 
        input_ids: torch.Tensor, 
        mask_positions: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict tokens for masked positions.
        
        Args:
            input_ids: Input sequence with [MASK] tokens [batch_size, seq_len]
            mask_positions: Boolean mask indicating positions to predict [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)
            
        Returns:
            Predicted token IDs for masked positions [batch_size, seq_len]
        """
        # Forward pass through model
        logits, _ = self.model(input_ids.to(self.device))
        
        # Extract logits for masked positions only
        masked_logits = logits[mask_positions]  # [num_masked_positions, vocab_size]
        
        if masked_logits.numel() == 0:
            return input_ids.clone()
        
        # Apply temperature scaling
        if temperature != 1.0:
            masked_logits = masked_logits / temperature
        
        # Top-k sampling if specified
        if top_k is not None:
            v, _ = torch.topk(masked_logits, min(top_k, masked_logits.size(-1)))
            masked_logits[masked_logits < v[:, [-1]]] = -float('inf')
        
        # Sample predictions
        probs = F.softmax(masked_logits, dim=-1)
        predicted_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Create result tensor
        result = input_ids.clone()
        result[mask_positions] = predicted_tokens.to(input_ids.device)
        
        return result
```

#### 2.2 Synthetic Text Generation

**File**: `data/sequence_scorer/synthetic_generation.py`

```python
import torch
from typing import Tuple, Dict, Any
from data.char_diffusion.masking_utils import apply_stage_masking

def apply_stage_masking_direct(
    x: torch.Tensor, 
    stage_config: Dict[str, Any], 
    mask_token_id: int, 
    rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply stage-specific masking with direct [MASK] token replacement (no BERT-style corruption).
    
    Args:
        x: Input tokens [batch_size, seq_len]
        stage_config: Stage configuration dictionary
        mask_token_id: Token ID for [MASK] token
        rng: Random number generator
        
    Returns:
        masked_x: Input with masked positions replaced by [MASK] tokens
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config['type']
    
    if stage_type == 'random':
        max_masked_ratio = stage_config['max_masked_ratio']
        batch_size = x.shape[0]
        
        # Generate different mask ratios for each sample
        mask_ratios = torch.rand(batch_size, generator=rng) * max_masked_ratio
        
        all_masked = []
        all_masks = []
        
        for i in range(batch_size):
            sample_x = x[i:i+1]
            mask_probs = torch.rand(sample_x.shape, generator=rng)
            mask = mask_probs < mask_ratios[i].item()
            
            masked_x = sample_x.clone()
            masked_x[mask] = mask_token_id
            
            all_masked.append(masked_x)
            all_masks.append(mask)
        
        return torch.cat(all_masked, dim=0), torch.cat(all_masks, dim=0)
        
    elif stage_type == 'sticky':
        from data.char_diffusion.masking_utils import apply_target_driven_sticky_masking_cpu
        # Use existing sticky masking but extract only mask positions
        _, mask = apply_target_driven_sticky_masking_cpu(
            x, stage_config['target_masked_ratio'], 
            stage_config['p1_probability'], stage_config['p2_probability'],
            mask_token_id, 0, rng  # vocab_size not used for direct masking
        )
        # Apply direct masking
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask
        
    elif stage_type == 'span':
        from data.char_diffusion.masking_utils import apply_span_masking_cpu
        # Use existing span masking but extract only mask positions
        _, mask = apply_span_masking_cpu(
            x, stage_config['spans_count'], mask_token_id, 0, rng
        )
        # Apply direct masking
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask
        
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")

def create_synthetic_text(
    original_text: torch.Tensor,
    mask_ratio: float,
    mlm_engine,
    mask_token_id: int,
    vocab_size: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: int = None
) -> Tuple[torch.Tensor, float]:
    """
    Create synthetic text by masking and predicting with MLM model.
    
    Args:
        original_text: Original text tokens [batch_size, seq_len]
        mask_ratio: Fraction of tokens to mask and predict
        mlm_engine: MLM inference engine
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size (excluding [MASK])
        rng: Random number generator
        sampling_temperature: Temperature for prediction sampling
        top_k: Top-k sampling parameter
        
    Returns:
        synthetic_text: Text with predicted tokens [batch_size, seq_len]
        actual_synthenticity: Actual ratio of synthetic tokens
    """
    batch_size, seq_len = original_text.shape
    
    # Generate random mask
    mask_probs = torch.rand(original_text.shape, generator=rng)
    mask = mask_probs < mask_ratio
    
    # Apply direct masking (replace masked positions with [MASK] token)
    corrupted_input = original_text.clone()
    corrupted_input[mask] = mask_token_id
    
    # Use MLM model to predict masked tokens
    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input, 
        mask,  # Use the original mask positions
        temperature=sampling_temperature,
        top_k=top_k
    )
    
    # Calculate actual synthenticity ratio
    # Count positions where we masked and regenerated tokens
    synthetic_positions = mask
    total_positions = seq_len * batch_size
    actual_synthetic_count = synthetic_positions.sum().item()
    actual_synthenticity = actual_synthetic_count / total_positions
    
    return predicted_text, actual_synthenticity

def create_stage_synthetic_text(
    original_text: torch.Tensor,
    stage_config: Dict[str, Any],
    mlm_engine,
    mask_token_id: int,
    vocab_size: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: int = None
) -> Tuple[torch.Tensor, float]:
    """
    Create synthetic text using stage-based masking configuration.
    
    Args:
        original_text: Original text tokens [batch_size, seq_len]
        stage_config: Stage configuration dictionary (sticky, random, or span)
        mlm_engine: MLM inference engine
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size (excluding [MASK])
        rng: Random number generator
        sampling_temperature: Temperature for prediction sampling
        top_k: Top-k sampling parameter
        
    Returns:
        synthetic_text: Text with predicted tokens [batch_size, seq_len]
        actual_synthenticity: Actual ratio of synthetic tokens
    """
    batch_size, seq_len = original_text.shape
    
    # Apply stage-specific masking (direct replacement with [MASK] tokens)
    corrupted_input, mask = apply_stage_masking_direct(
        original_text, stage_config, mask_token_id, rng
    )
    
    # Use MLM model to predict masked tokens
    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input, 
        mask,  # Use the original mask positions
        temperature=sampling_temperature,
        top_k=top_k
    )
    
    # Calculate actual synthenticity ratio
    # Count positions where we masked and regenerated tokens
    synthetic_positions = mask
    total_positions = seq_len * batch_size
    actual_synthetic_count = synthetic_positions.sum().item()
    actual_synthenticity = actual_synthetic_count / total_positions
    
    return predicted_text, actual_synthenticity

def add_cls_token(
    text: torch.Tensor, 
    cls_token_id: int,
    block_size: int
) -> torch.Tensor:
    """
    Add [CLS] token at the beginning of sequences.
    
    Args:
        text: Input text [batch_size, seq_len]
        cls_token_id: ID of [CLS] token  
        block_size: Target sequence length
        
    Returns:
        text_with_cls: Text with [CLS] token [batch_size, block_size]
    """
    batch_size, seq_len = text.shape
    
    # Create tensor with [CLS] at position 0
    result = torch.zeros((batch_size, block_size), dtype=text.dtype)
    result[:, 0] = cls_token_id
    
    # Copy text starting from position 1, truncating if necessary
    copy_len = min(seq_len, block_size - 1)
    result[:, 1:1+copy_len] = text[:, :copy_len]
    
    return result
```

### Phase 3: Complete Provider Implementation

#### 3.1 Data Loading and Vocabulary Setup

**File**: `data/sequence_scorer/prepare_streaming.py` (continued)

```python
def _load_text_data(self):
    """Load text data and create vocabulary compatible with MLM model."""
    input_file_path = os.path.join(self.data_dir, 'input.txt')
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input text file not found: {input_file_path}")
        
    with open(input_file_path, 'r') as f:
        data = f.read()
    
    # Use vocabulary from MLM model to ensure compatibility
    self.stoi = self.mlm_engine.stoi
    self.itos = self.mlm_engine.itos  
    self.vocab_size = self.mlm_engine.vocab_size
    self.mask_token_id = self.mlm_engine.mask_token_id
    
    # Convert text to token IDs
    self.train_ids = [self.stoi.get(c, 0) for c in data[:int(len(data) * 0.9)]]
    self.val_ids = [self.stoi.get(c, 0) for c in data[int(len(data) * 0.9):]]
    
    # Add [CLS] token if not in vocabulary
    if self.cls_token_id not in self.itos:
        self.itos[self.cls_token_id] = '[CLS]'
        self.stoi['[CLS]'] = self.cls_token_id

def _sample_default_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
    """Generate batch with simple mask ratio range (non-stage mode)."""
    ids = self.train_ids if split == "train" else self.val_ids
    max_start_idx = len(ids) - (self.block_size - 1)  # Reserve space for [CLS]
    
    # Sample random starting positions
    ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()
    
    # Create original sequences 
    original_sequences = []
    for i in ix:
        seq = ids[i : i + (self.block_size - 1)]  # Leave space for [CLS]
        original_sequences.append(seq)
    
    original_text = torch.tensor(original_sequences, dtype=torch.long)
    
    # Generate random mask ratios for each sequence
    min_ratio, max_ratio = self.mask_probability_range
    mask_ratios = torch.rand(self.batch_size, generator=rng) * (max_ratio - min_ratio) + min_ratio
    
    # Generate synthetic text and calculate targets
    batch_inputs = []
    batch_targets = []
    
    for i in range(self.batch_size):
        # Create synthetic version of this sequence
        synthetic_text, actual_synthenticity = create_synthetic_text(
            original_text[i:i+1],  # Single sequence
            mask_ratios[i].item(),
            self.mlm_engine,
            self.mask_token_id,
            0,  # vocab_size not needed for direct masking
            rng,
            sampling_temperature=1.0,
            top_k=50  # Use top-k sampling for diversity
        )
        
        # Add [CLS] token at the beginning
        input_with_cls = add_cls_token(
            synthetic_text, 
            self.cls_token_id, 
            self.block_size
        )
        
        batch_inputs.append(input_with_cls)
        batch_targets.append(actual_synthenticity)
    
    # Convert to tensors
    input_ids = torch.cat(batch_inputs, dim=0)
    targets = torch.tensor(batch_targets, dtype=torch.float32)
    
    return {
        "input_ids": input_ids,
        "targets": targets,
    }

def _produce_stage_based_file(self, split: str, seq: int) -> None:
    """Generate an entire file with stage-based sampling (same pattern as char_diffusion)."""
    import time
    
    rng = torch.Generator()
    # derive deterministic seed per split/seq (same as base class)
    per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
    rng.manual_seed(per_seed)
    
    ids = self.train_ids if split == "train" else self.val_ids
    max_start_idx = len(ids) - (self.block_size - 1)  # Reserve space for [CLS]
    
    # Get stage distribution for this split
    stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution
    
    # Generate all samples according to stage distribution
    all_inputs = []
    all_targets = []
    all_stage_info = []
    
    for stage_info in stage_distribution:
        stage_config = stage_info['config']
        count = stage_info['count']
        
        if count == 0:
            continue
            
        # Sample sequences for this stage
        ix = torch.randint(0, max_start_idx, (count,), generator=rng).tolist()
        original_sequences = []
        for i in ix:
            seq_data = ids[i : i + (self.block_size - 1)]  # Leave space for [CLS]
            original_sequences.append(seq_data)
        
        original_text = torch.tensor(original_sequences, dtype=torch.long)
        
        # Apply stage-specific synthetic text generation
        synthetic_text, actual_synthenticity = create_stage_synthetic_text(
            original_text,
            stage_config,
            self.mlm_engine,
            self.mask_token_id,
            0,  # vocab_size not needed for direct masking
            rng,
            sampling_temperature=1.0,
            top_k=50
        )
        
        # Add [CLS] token at the beginning for each sequence
        inputs_with_cls = []
        for i in range(count):
            input_with_cls = add_cls_token(
                synthetic_text[i:i+1], 
                self.cls_token_id, 
                self.block_size
            )
            inputs_with_cls.append(input_with_cls)
        
        # Calculate synthenticity targets per sequence (for stage-based, may be same for all in stage)
        stage_targets = [actual_synthenticity] * count
        
        all_inputs.extend(inputs_with_cls)
        all_targets.extend(stage_targets)
        
        # Track stage info for each sample
        all_stage_info.extend([stage_config] * count)
    
    # Concatenate all samples
    if all_inputs:
        combined_inputs = torch.cat(all_inputs, dim=0)
        combined_targets = torch.tensor(all_targets, dtype=torch.float32)
        
        # Randomly shuffle all samples to mix different stage types
        total_samples = combined_inputs.shape[0]
        shuffle_indices = torch.randperm(total_samples, generator=rng)
        
        shuffled_inputs = combined_inputs[shuffle_indices]
        shuffled_targets = combined_targets[shuffle_indices]
        
        # Create shuffled stage info
        shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]
        
        # Organize into batches
        tensors = {
            "input_ids": shuffled_inputs,
            "targets": shuffled_targets,
        }
        
        metadata = {
            "batch_size": self.batch_size,
            "num_batches": self.batches_per_file,
            "file_idx": seq,
            "split": split,
            "produced_at": int(time.time() * 1000),
            "stage_info": shuffled_stage_info,
            "stage_distribution": stage_distribution  # Include stage distribution info
        }
        
        # Write atomic
        d = self.train_dir if split == "train" else self.val_dir
        ts = metadata["produced_at"]
        tmp_name = f".tmp-{ts}-{seq:06d}.pt"
        final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
        tmp_path = os.path.join(d, tmp_name)
        final_path = os.path.join(d, final_name)
        torch.save({"tensors": tensors, "metadata": metadata}, tmp_path)
        os.replace(tmp_path, final_path)
        if self.verbose:
            print(f"[sequence_scorer] produced stage-based file: {final_path}")
    else:
        # Fallback to default generation
        super().produce_one_file(split, seq)
```

### Phase 4: Configuration Integration

#### 4.1 Simple Configuration (Non-Stage Mode)

**File**: `config/sequence_scorer_simple.py`

```python
"""Simple configuration for sequence scoring dataset with basic masking"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 16  # Smaller batch for sequence scoring
block_size = 256  # Reasonable context for scoring

# MLM model for synthetic text generation
mlm_checkpoint_path = 'checkpoints/char_diffusion_mlm.pt'  # Path to trained MLM model
mask_probability_range = (0.1, 0.7)  # Range of masking ratios to generate
cls_token_id = 0  # Position for [CLS] token (adjust based on vocab)

# No stage configuration (uses simple masking)
use_all_stages_for_training = None
unmasking_stages = None
validation_stages = None

# Model mode configuration  
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'

# Training settings optimized for sequence scoring
learning_rate = 1e-4
warmup_iters = 300
max_iters = 8000
eval_interval = 200

# Data generation settings
batches_per_file = 50  # Smaller files due to MLM inference cost
max_backlog_files = 3
sleep_seconds = 5.0  # Longer sleep due to inference cost

print("Simple sequence scorer configuration loaded")
```

#### 4.2 Stage-Based Configuration

**File**: `config/sequence_scorer_complex.py`

```python
"""Advanced configuration for sequence scoring dataset with stage-based generation"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 16
block_size = 256

# MLM model for synthetic text generation
mlm_checkpoint_path = 'checkpoints/char_diffusion_mlm.pt'
cls_token_id = 0

# Load composition configuration (same as char_diffusion)
composition_config = 'complex'  # refers to data/sequence_scorer/config/complex.py

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    config_path = os.path.join('data', 'sequence_scorer', 'config', f'{composition_config}.py')
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{composition_config}_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Import all global variables from the config
        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):
                globals()[attr_name] = getattr(config_module, attr_name)
        print(f"Loaded composition config from {config_path}")
    else:
        print(f"Warning: composition config file not found at {config_path}")
else:
    # Set default values when no composition config is used
    use_all_stages_for_training = None
    unmasking_stages = None
    validation_stages = None

# Model mode configuration  
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'

# Training settings optimized for sequence scoring
learning_rate = 1e-4
warmup_iters = 300
max_iters = 8000
eval_interval = 200

# Data generation settings (more conservative for stage-based)
batches_per_file = 30  # Smaller files due to MLM inference and stage complexity
max_backlog_files = 2
sleep_seconds = 8.0  # More time for MLM inference across stages

print("Complex sequence scorer configuration loaded")
```

#### 4.3 Stage Configuration Files

**File**: `data/sequence_scorer/config/complex.py`

```python
# Stage-based configuration for sequence scorer (mirrors char_diffusion complex.py)
use_all_stages_for_training = True

# Training stages - same configurations as char_diffusion but adapted for sequence scoring
unmasking_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type': 'span', 'spans_count': 20, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]

# Validation stages - extended set for comprehensive evaluation
validation_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]
```

#### 4.4 Prepare.py Integration

The existing `prepare.py` uses convention-based discovery and already passes stage-related configuration, so no changes needed. The system will automatically:
- Find module: `data.sequence_scorer.prepare_streaming`  
- Find class: `SequenceScorerProvider` (via `Provider` alias)
- Pass stage configuration through `use_all_stages_for_training`, `unmasking_stages`, `validation_stages` parameters

### Phase 5: Usage and Validation

#### 5.1 Training Pipeline Integration

**Simple Mode Usage:**

```bash
# 1. First train a char_diffusion MLM model (if not already available)
python train.py config/train_char_diffusion.py --out_dir=checkpoints/char_mlm

# 2. Generate sequence scoring data using simple configuration
python prepare.py config/sequence_scorer_simple.py

# 3. Train sequence scorer model
python train.py config/sequence_scorer_simple.py --out_dir=checkpoints/sequence_scorer_simple
```

**Stage-Based Mode Usage:**

```bash
# 1. First train a char_diffusion MLM model with stage-based configuration
python train.py config/train_char_diffusion.py --out_dir=checkpoints/char_mlm_complex

# 2. Generate sequence scoring data using complex stage-based configuration
python prepare.py config/sequence_scorer_complex.py

# 3. Train sequence scorer model with stage-based data
python train.py config/sequence_scorer_complex.py --out_dir=checkpoints/sequence_scorer_complex
```

**Hybrid Training (Recommended):**

```bash
# 1. Train MLM model with complex stage configuration for robustness
python train.py config/train_char_diffusion.py --out_dir=checkpoints/char_mlm_robust

# 2. Generate sequence scoring data with SAME stage configuration
python prepare.py config/sequence_scorer_complex.py

# 3. Train sequence scorer with stage-aware synthetic data
python train.py config/sequence_scorer_complex.py --out_dir=checkpoints/sequence_scorer_robust
```

#### 5.2 Expected Data Flow

**Simple Mode Data Flow:**
1. **Data Generation Phase** (prepare.py):
   - Load Shakespeare text 
   - For each batch:
     - Sample text sequences
     - Apply random masking (10%-70% of tokens with [MASK] token)
     - Use MLM model to regenerate masked positions
     - Add [CLS] token at beginning
     - Target = actual synthenticity ratio (proportion of regenerated tokens)
   
2. **Training Phase** (train.py):
   - Load synthetic text with [CLS] tokens
   - Extract [CLS] representation from sequence scorer model
   - Predict synthenticity score (0-1) with sigmoid
   - Optimize MSE loss against true synthenticity ratios

**Stage-Based Data Flow:**
1. **Data Generation Phase** (prepare.py):
   - Load Shakespeare text
   - For each file generation:
     - Distribute samples across all stages (9 training + 14 validation stages)
     - For each stage:
       - Apply stage-specific masking (sticky/random/span with [MASK] tokens)
       - Use MLM model to regenerate masked positions
       - Calculate stage-specific synthenticity ratio (proportion of regenerated tokens)
     - Shuffle all samples to mix stage types
     - Add [CLS] tokens and save with stage metadata

2. **Training Phase** (train.py):
   - Load mixed synthetic text with diverse masking patterns
   - Extract [CLS] representations across all stage types
   - Learn robust synthenticity detection across diverse regeneration patterns
   - Optimize MSE loss against curriculum of synthenticity ratios

#### 5.3 Validation Tests

Create test script to validate the pipeline:

**File**: `validate_sequence_scorer.py`

```python
"""Validation script for sequence scorer dataset"""

def test_mlm_inference():
    """Test MLM model loading and inference"""
    # Load MLM model and test prediction
    
def test_synthetic_generation():
    """Test synthetic text generation with various mask ratios"""
    # Test different mask ratios and validate synthenticity calculation
    
def test_data_generation():
    """Test complete data generation pipeline"""
    # Generate sample batches and validate schema
    
def test_training_integration():
    """Test integration with sequence scorer model"""
    # Test that generated data works with sequence scorer training

if __name__ == "__main__":
    # Run all validation tests
    pass
```

## Performance Considerations

### Memory and Compute Optimization

1. **CPU-based MLM Inference**: 
   - Use CPU for data generation to free GPU for training
   - Implement batched inference for efficiency
   - Consider quantization for faster inference

2. **Caching Strategy**:
   - Cache MLM predictions for repeated text segments
   - Use deterministic seeding for reproducible generation

3. **Disk Usage**:
   - Generate smaller batch files due to MLM inference cost
   - Monitor queue backlog to balance generation vs consumption

### Scalability Considerations

1. **Distributed Generation**:
   - MLM model loading per process (DDP friendly)
   - Separate data generation per GPU rank if needed

2. **Model Checkpointing**:
   - Ensure MLM checkpoint includes all necessary metadata
   - Version compatibility checks between MLM and scorer models

## Integration Benefits

This enhanced implementation provides:

1. **Complete Stage Compatibility**: Full support for char_diffusion's complex stage configurations
2. **Seamless Integration**: Uses existing DataProviderBase and masking utilities
3. **Flexible Configuration**: Both simple masking and advanced multi-stage curriculum learning
4. **Training Compatibility**: Works with existing sequence scorer model architecture
5. **Monitoring**: Detailed logging of synthenticity distributions across all stages
6. **Reproducibility**: Deterministic generation with proper RNG seeding per stage
7. **Curriculum Learning**: Enables progressive difficulty training like char_diffusion
8. **Configuration Reuse**: Same complex.py files can be used across datasets

## Key Advantages of Stage-Based Sequence Scoring

1. **Robust Synthetic Detection**: Trains on diverse regeneration patterns (sticky, random, span masking)
2. **Curriculum Learning**: Progressive difficulty from low to high synthenticity ratios  
3. **Cross-Dataset Consistency**: Same stage configurations as MLM training
4. **Evaluation Granularity**: Detailed validation across 14 different regeneration scenarios
5. **Research Alignment**: Matches char_diffusion experimental setup for fair comparison

The sequence scorer dataset will enable training models to detect AI-generated text by learning the relationship between local (token-level) and global (sequence-level) synthetic patterns across a comprehensive curriculum of regeneration strategies, mirroring the sophisticated training approach used in char_diffusion.