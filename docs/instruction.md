# Multi-Mode Transformer Development Guide

## Overview: Creating a Versatile Language Analysis System

This guide details how to transform your current diffusion-based transformer into a flexible, multi-mode system supporting three distinct training and inference paradigms. The current codebase already has sophisticated infrastructure for unmasking training - we'll extend this to support additional modes while maintaining backward compatibility.

### Current System Analysis

Your existing codebase has:
- **Advanced unmasking training** with stage-based progression (`training_type = 'unmasking'`)
- **Binary remasking training** for corruption detection (`training_type = 'remasking_binary'`)
- **Modular training utilities** in `training_utils/` package
- **Flexible model architecture** with bidirectional and causal attention support
- **Sophisticated batch generation** with caching and prefetching
- **Comprehensive evaluation** and validation systems

## The Three Target Modes

### **Mode 1: Language Modeling (Linguistic Detective) 🕵️‍♂️**
- **Current Status**: ✅ **Already Implemented** as `training_type = 'unmasking'`
- **Purpose**: Reconstruct original text from masked/corrupted versions
- **Training**: Uses sophisticated stage-based unmasking with sticky/random masking strategies
- **Architecture**: Bidirectional attention, vocabulary-sized output head
- **Loss**: Cross-entropy over vocabulary
- **Use Case**: Pre-training foundation models, text completion

### **Mode 2: Token-Level Classification (Inspector) �**
- **Current Status**: ⚠️ **Needs Refactoring** from `training_type = 'remasking_binary'`
- **Purpose**: Classify each token individually with flexible number of classes
- **Training**: Multi-class classification per token position (probability distribution over classes)
- **Architecture**: Bidirectional attention, N-class output head (configurable)
- **Loss**: Cross-entropy over N classes (with optional class balancing)
- **Use Case**: AI detection, authorship attribution, quality assessment per token, multi-class token tagging
- **Key Feature**: Supports 2+ classes, not limited to binary classification

### **Mode 3: Sequence-Level Scoring (Judge) ⚖️**
- **Current Status**: ❌ **Not Implemented** - needs to be built
- **Purpose**: Single quality score (0-1) for entire sequence
- **Training**: Regression to predict continuous score between 0 and 1
- **Architecture**: Bidirectional attention, single sigmoid output from [CLS] token
- **Loss**: MSE loss for continuous score prediction
- **Use Case**: Quality scoring, reward modeling for RL, content rating, sequence-level assessment
- **Key Feature**: Returns continuous score 0-1, not classification

---

## Implementation Plan

### Phase 1: Model Architecture Updates

#### Step 1: Define Model Mode Enum
**File**: `model.py`
**Location**: After imports, before `LayerNorm` class

```python
from enum import Enum

class ModelMode(Enum):
    """Defines the three operational modes for the transformer model"""
    LANGUAGE_MODEL = "language_model"      # Unmasking/reconstruction (current 'unmasking')
    TOKEN_CLASSIFIER = "token_classifier"  # Per-token multi-class classification (refactored from 'remasking_binary')
    SEQUENCE_SCORER = "sequence_scorer"    # Sequence-level scoring 0-1 (new)
```

#### Step 2: Update GPTConfig
**File**: `model.py`
**Location**: `GPTConfig` dataclass (around line 260)

**Changes needed**:
1. Replace `binary_classification: bool = False` with `mode: ModelMode = ModelMode.LANGUAGE_MODEL`
2. Add `num_token_classes: int = 2` for flexible token classification
3. Add `cls_token_id: int = None` for sequence scoring
4. Add transfer learning support parameters
5. Add `__post_init__` method to enforce bidirectional attention for classification modes

```python
@dataclass
class GPTConfig:
    # ... existing fields ...
    mode: ModelMode = ModelMode.LANGUAGE_MODEL
    num_token_classes: int = 2  # Number of classes for token classification (flexible, not just binary)
    cls_token_id: int = None  # Special token ID for [CLS] token in sequence scoring

    # Transfer learning support
    freeze_transformer: bool = False  # If True, freeze all transformer weights (feature extraction)
    init_from_checkpoint: str = None  # Path to pretrained checkpoint for transfer learning

    def __post_init__(self):
        # Enforce bidirectional attention for classification tasks
        if self.mode in [ModelMode.TOKEN_CLASSIFIER, ModelMode.SEQUENCE_SCORER]:
            if self.attention_type != 'bidirectional':
                print(f"WARNING: {self.mode.value} requires bidirectional attention. Changing from '{self.attention_type}' to 'bidirectional'")
                self.attention_type = 'bidirectional'
```

#### Step 3: Update GPT.__init__ Method
**File**: `model.py`
**Location**: `GPT.__init__` method (around line 295)

**Replace the binary_classification logic with**:
```python
# Create appropriate output head based on mode
if self.config.mode == ModelMode.LANGUAGE_MODEL:
    # Language modeling: predict next token from vocabulary
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    # Weight tying for language modeling
    self.transformer.wte.weight = self.lm_head.weight
elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
    # Token-level classification: flexible number of classes per token
    self.lm_head = nn.Linear(config.n_embd, config.num_token_classes, bias=False)
    print(f"Token classifier head: {config.num_token_classes} classes per token")
elif self.config.mode == ModelMode.SEQUENCE_SCORER:
    # Sequence-level scoring: single continuous score 0-1 from [CLS] token
    self.sequence_head = nn.Sequential(
        nn.Linear(config.n_embd, 1, bias=False),
        nn.Sigmoid()  # Ensure output is between 0 and 1
    )
    # Initialize with smaller weights to prevent gradient explosion
    torch.nn.init.normal_(self.sequence_head[0].weight, mean=0.0, std=0.002)
    print("Sequence scorer head: continuous score 0-1")
else:
    raise ValueError(f"Unknown model mode: {self.config.mode}")

# Transfer learning support: freeze transformer if requested
if config.freeze_transformer:
    print("Freezing transformer weights for feature extraction")
    for param in self.transformer.parameters():
        param.requires_grad = False
    # Keep head trainable
    if hasattr(self, 'lm_head'):
        for param in self.lm_head.parameters():
            param.requires_grad = True
    if hasattr(self, 'sequence_head'):
        for param in self.sequence_head.parameters():
            param.requires_grad = True
```

#### Step 4: Update GPT.forward Method
**File**: `model.py`
**Location**: `GPT.forward` method (around line 361)

**Replace the existing loss computation logic with**:
```python
# Mode-specific forward pass and loss computation
if self.config.mode == ModelMode.SEQUENCE_SCORER:
    # Extract [CLS] token representation (first token)
    cls_output = x[:, 0, :]  # (batch_size, n_embd)
    logits = self.sequence_head(cls_output).squeeze(-1)  # (batch_size,) with sigmoid applied

    if targets is not None:
        # MSE loss for continuous score prediction (0-1 range)
        loss = F.mse_loss(logits, targets.float())
    else:
        loss = None

elif self.config.mode in [ModelMode.LANGUAGE_MODEL, ModelMode.TOKEN_CLASSIFIER]:
    # Token-level predictions for all positions
    if targets is not None:
        # Training: compute logits for all positions
        logits = self.lm_head(x)
    else:
        # Inference: optimize based on attention type
        if self.config.mode == ModelMode.LANGUAGE_MODEL and getattr(self.config, 'attention_type', 'causal') == 'causal':
            # Causal language modeling: only need last position for generation
            logits = self.lm_head(x[:, [-1], :])
        else:
            # Token classification or bidirectional: need all positions
            logits = self.lm_head(x)

    if targets is not None:
        if self.config.mode == ModelMode.TOKEN_CLASSIFIER:
            # Multi-class token classification with flexible number of classes
            num_classes = self.config.num_token_classes

            if targets.dim() == 3:
                # Soft targets (probability distributions)
                loss = F.cross_entropy(logits.view(-1, num_classes), targets.view(-1, num_classes))
            else:
                # Hard targets with optional dynamic class weighting
                flattened_targets = targets.view(-1)
                valid_targets = flattened_targets[flattened_targets != -1]

                if len(valid_targets) > 0 and num_classes > 1:
                    # Dynamic class weighting for imbalanced datasets
                    unique, counts = torch.unique(valid_targets, return_counts=True)
                    n_samples = len(valid_targets)

                    class_weights = torch.zeros(num_classes, device=targets.device, dtype=logits.dtype)
                    for cls, count in zip(unique, counts):
                        if cls < num_classes:  # Ensure class index is valid
                            class_weights[cls] = n_samples / (num_classes * count)

                    loss = F.cross_entropy(logits.view(-1, num_classes), flattened_targets,
                                         weight=class_weights, ignore_index=-1)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), flattened_targets, ignore_index=-1)
        else:
            # Language modeling loss
            if targets.dim() == 3:
                # Soft targets (label smoothing)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1, logits.size(-1)))
            else:
                # Hard targets
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        loss = None
else:
    raise ValueError(f"Unknown model mode: {self.config.mode}")
```

---

### Phase 2: Training System Updates

#### Step 5: Add New Training Type for Sequence Classification
**File**: `training_utils/training_config.py`
**Location**: `TrainingContext` class

**Add new training type and token classification parameters**:
```python
# In TrainingContext class, update the training_type field and add new parameters
training_type: str = 'unmasking'  # Options: 'unmasking', 'token_classification', 'sequence_scoring'
num_token_classes: int = 2  # Number of classes for token classification (flexible)
freeze_transformer: bool = False  # For transfer learning: freeze transformer, train only head
```

#### Step 6: Update Batch Generation
**File**: `training_utils/batch_generation.py`
**Location**: `get_batch` function (line 33)

**Update the routing logic**:
```python
def get_batch(split, ctx: TrainingContext, validation_sample_idx=None):
    """Main batch generation function that delegates to specific training type functions"""
    if ctx.training_type == 'unmasking':
        return get_batch_unmasking(split, ctx, validation_sample_idx)
    elif ctx.training_type in ['remasking_binary', 'token_classification']:
        # Support both old and new names for backward compatibility
        return get_batch_token_classification(split, ctx, validation_sample_idx)
    elif ctx.training_type == 'sequence_scoring':
        return get_batch_sequence_scoring(split, ctx, validation_sample_idx)
    else:
        raise ValueError(f"Unsupported training type: {ctx.training_type}")
```

**Rename and update the existing remasking function**:
```python
def get_batch_token_classification(split, ctx: TrainingContext, validation_sample_idx=None):
    """
    Token-level classification training with flexible number of classes.
    Refactored from get_batch_remasking_binary to support multi-class classification.
    """
    # This function should be based on the existing get_batch_remasking_binary
    # but updated to support ctx.num_token_classes instead of hardcoded binary
    # The core logic remains the same, just the target generation changes

    # ... (use existing remasking logic but update target generation)

    # Instead of binary targets (remask_good_id, remask_wrong_id):
    # Generate targets based on ctx.num_token_classes
    # For backward compatibility, if num_token_classes == 2, use existing logic
    pass  # Implementation details...

def get_batch_sequence_scoring(split, ctx: TrainingContext, validation_sample_idx=None):
    """Sequence-level scoring training with [CLS] token prepending - returns continuous scores 0-1"""

    # Reuse existing data loading infrastructure
    data_cache = get_data_cache()
    valid_indices_cache = get_valid_indices_cache()

    # Initialize data cache if needed (same as other modes)
    if data_cache[split] is None:
        if split == 'train':
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data_cache[split] = np.memmap(os.path.join(ctx.data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        if ctx.use_paragraph_boundaries:
            print(f"Computing valid indices for {split} (paragraph boundaries)... (one-time cost)")
            valid_indices_cache[split] = find_double_newline_indices(data_cache[split], ctx.meta_vocab_size, ctx.block_size - 1)  # -1 for [CLS] token
        else:
            print(f"Using random sampling for {split} (no paragraph boundaries)")
            valid_indices_cache[split] = np.array([])
        print(f"Found {len(valid_indices_cache[split])} valid indices for {split}")

    # Generate data (similar to other modes but with block_size - 1)
    data = data_cache[split]
    valid_indices = valid_indices_cache[split]

    # Sample indices (leave room for [CLS] token)
    if len(valid_indices) == 0:
        if split == 'val':
            torch.manual_seed(42)
            ix_np = torch.randint(len(data) - (ctx.block_size - 1), (ctx.batch_size,)).numpy()
            torch.manual_seed(1337 + ctx.seed_offset)
        else:
            ix_np = torch.randint(len(data) - (ctx.block_size - 1), (ctx.batch_size,)).numpy()
    else:
        if split == 'val':
            torch.manual_seed(42)
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]
            torch.manual_seed(1337 + ctx.seed_offset)
        else:
            ix_indices = torch.randint(len(valid_indices), (ctx.batch_size,)).numpy()
            ix_np = valid_indices[ix_indices]

    # Load text data (block_size - 1 tokens)
    ix_expanded = ix_np[:, None] + np.arange(ctx.block_size - 1)[None, :]
    text_data = data[ix_expanded].astype(np.int64)

    # Prepend [CLS] token to each sequence
    cls_token_id = ctx.extended_vocab_size + 1  # Use next available token ID
    cls_tokens = np.full((ctx.batch_size, 1), cls_token_id, dtype=np.int64)
    x_np = np.concatenate([cls_tokens, text_data], axis=1)

    # Transfer to GPU
    x = torch.from_numpy(x_np)
    if ctx.device_type == 'cuda':
        x = x.pin_memory().to(ctx.device, non_blocking=True)
    else:
        x = x.to(ctx.device)

    # Generate sequence-level targets (placeholder - replace with actual labels)
    # For now, generate random scores between 0 and 1 for demonstration
    if split == 'val':
        torch.manual_seed(42 + (validation_sample_idx or 0))

    # TODO: Replace this with actual sequence-level labels from your dataset
    # This could be quality scores, sentiment labels, etc.
    y = torch.rand(ctx.batch_size, device=ctx.device)  # Random scores for demo

    if split == 'val':
        torch.manual_seed(1337 + ctx.seed_offset)

    # No mask needed for sequence classification
    mask = torch.zeros_like(x, dtype=torch.bool)

    return x, y, mask
```

#### Step 7: Update Training Script
**File**: `train_run.py`
**Location**: Multiple locations

**Add import** (after line 20):
```python
from model import GPTConfig, GPT, ModelMode
```

**Update model configuration** (around line 304):
```python
# Add mode selection to model_args
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type,
                  use_rope=use_rope, mode=ModelMode.SEQUENCE_SCORER,  # Updated name
                  num_token_classes=2, freeze_transformer=False)  # Add these lines
```

**Add training type configuration** (around line 44):
```python
# Add new training type options
training_type = 'sequence_scoring'  # Options: 'unmasking', 'token_classification', 'sequence_scoring'
num_token_classes = 2  # For token classification: number of classes (flexible)
freeze_transformer = False  # For transfer learning: freeze transformer weights
init_from_checkpoint = None  # Path to pretrained checkpoint for transfer learning
```

#### Step 8: Update Configuration System
**File**: `configurator.py` (if it exists) or config files

**Add new configuration options**:
```python
# Sequence classification specific configs
cls_token_id = None  # Will be set automatically
sequence_target_type = 'regression'  # 'regression' or 'classification'
num_sequence_classes = 1  # For classification mode
```

---

### Phase 3: Transfer Learning Support

#### Step 9: Add Transfer Learning to Model Initialization
**File**: `model.py`
**Location**: `GPT.__init__` method, after head creation

**Add transfer learning logic**:
```python
# Transfer learning: load pretrained weights if specified
if config.init_from_checkpoint is not None:
    print(f"Loading pretrained weights from {config.init_from_checkpoint}")
    checkpoint = torch.load(config.init_from_checkpoint, map_location='cpu', weights_only=False)

    # Load transformer weights (excluding heads)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Filter out head weights from pretrained checkpoint
    transformer_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('lm_head') and not k.startswith('sequence_head'):
            transformer_state_dict[k] = v

    # Load transformer weights
    missing_keys, unexpected_keys = self.load_state_dict(transformer_state_dict, strict=False)
    print(f"Loaded pretrained transformer weights:")
    print(f"  Missing keys: {len(missing_keys)} (expected for new heads)")
    print(f"  Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        print(f"  Missing (will be randomly initialized): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Unexpected (ignored): {unexpected_keys[:5]}...")
```

#### Step 10: Add Transfer Learning Configuration Validation
**File**: `train_run.py`
**Location**: After configuration loading

**Add transfer learning validation**:
```python
# Transfer learning validation
if init_from_checkpoint is not None:
    if not os.path.exists(init_from_checkpoint):
        raise FileNotFoundError(f"Transfer learning checkpoint not found: {init_from_checkpoint}")

    print(f"Transfer learning enabled:")
    print(f"  Checkpoint: {init_from_checkpoint}")
    print(f"  Freeze transformer: {freeze_transformer}")

    if freeze_transformer:
        print("  Mode: Feature extraction (frozen transformer + trainable head)")
    else:
        print("  Mode: Fine-tuning (all weights trainable)")

    # Add to model args
    model_args['init_from_checkpoint'] = init_from_checkpoint
    model_args['freeze_transformer'] = freeze_transformer
```

---

### Phase 4: Data Pipeline Integration

#### Step 11: Create Sequence Scoring Dataset Interface
**File**: `training_utils/dataset_interface.py` (new file)

```python
"""
Dataset interface for sequence-level classification tasks.
Handles loading and preprocessing of sequence-level labels.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

class SequenceDatasetInterface:
    """Interface for loading sequence-level classification datasets"""

    def __init__(self, data_dir: str, dataset_name: str = "quality_scores"):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.labels_cache = {}

    def load_sequence_labels(self, split: str) -> Dict[int, float]:
        """
        Load sequence-level labels for the given split.

        Expected format: JSON file with mapping from sequence_index -> label
        {
            "0": 0.85,
            "1": 0.23,
            "2": 0.91,
            ...
        }
        """
        if split in self.labels_cache:
            return self.labels_cache[split]

        labels_file = os.path.join(self.data_dir, f"{split}_labels.json")

        if not os.path.exists(labels_file):
            print(f"WARNING: No labels file found at {labels_file}")
            print("Using random labels for demonstration purposes")
            # Generate random labels as fallback
            data_file = os.path.join(self.data_dir, f"{split}.bin")
            if os.path.exists(data_file):
                data = np.memmap(data_file, dtype=np.uint16, mode='r')
                num_sequences = len(data) // 1024  # Approximate
                labels = {str(i): np.random.random() for i in range(num_sequences)}
            else:
                labels = {}
        else:
            with open(labels_file, 'r') as f:
                labels = json.load(f)
                # Convert string keys to int, values to float
                labels = {int(k): float(v) for k, v in labels.items()}

        self.labels_cache[split] = labels
        return labels

    def get_label_for_sequence(self, split: str, sequence_idx: int) -> float:
        """Get label for a specific sequence index"""
        labels = self.load_sequence_labels(split)
        return labels.get(sequence_idx, 0.5)  # Default to neutral score

    def get_labels_stats(self, split: str) -> Dict[str, float]:
        """Get statistics about the labels distribution"""
        labels = self.load_sequence_labels(split)
        values = list(labels.values())

        if not values:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
```

#### Step 12: Update Batch Generation with Real Labels
**File**: `training_utils/batch_generation.py`
**Location**: Update the `get_batch_sequence_scoring` function

**Replace the placeholder target generation**:
```python
# Replace this section in get_batch_sequence_scoring:
# TODO: Replace this with actual sequence-level labels from your dataset
# This could be quality scores, sentiment labels, etc.
# y = torch.rand(ctx.batch_size, device=ctx.device)  # Random scores for demo

# With this:
from .dataset_interface import SequenceDatasetInterface

# Initialize dataset interface (do this once, cache it)
if not hasattr(ctx, '_sequence_dataset'):
    ctx._sequence_dataset = SequenceDatasetInterface(ctx.data_dir)

# Get actual labels for the sampled sequences
labels = []
for i, seq_idx in enumerate(ix_np):
    # Convert data index to approximate sequence index
    sequence_idx = seq_idx // (ctx.block_size - 1)
    label = ctx._sequence_dataset.get_label_for_sequence(split, sequence_idx)
    labels.append(label)

y = torch.tensor(labels, dtype=torch.float32, device=ctx.device)
```

#### Step 13: Add Evaluation Metrics for Sequence Scoring
**File**: `training_utils/model_evaluation.py`
**Location**: Update `estimate_loss` function

**Add sequence scoring evaluation**:
```python
# Add this to the estimate_loss function after the existing loss computation:

if ctx.training_type == 'sequence_scoring':
    # Additional metrics for sequence scoring (continuous 0-1 scores)
    with torch.no_grad():
        # Compute correlation between predictions and targets
        if len(losses_list) > 0:
            all_preds = []
            all_targets = []

            for _ in range(min(eval_iters, 10)):  # Sample a few batches for correlation
                X, Y, mask = get_batch(split, ctx)
                with ctx_manager:
                    logits, loss = model(X, Y)

                all_preds.extend(logits.cpu().numpy().tolist())
                all_targets.extend(Y.cpu().numpy().tolist())

            if len(all_preds) > 1:
                correlation = np.corrcoef(all_preds, all_targets)[0, 1]
                out[f'{split}_correlation'] = correlation if not np.isnan(correlation) else 0.0

                # Mean absolute error (important for 0-1 scoring)
                mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
                out[f'{split}_mae'] = mae

                # R-squared for regression quality
                ss_res = np.sum((np.array(all_targets) - np.array(all_preds)) ** 2)
                ss_tot = np.sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                out[f'{split}_r2'] = r2
```

---

### Phase 5: Configuration and Testing

#### Step 14: Create Example Configuration Files

**File**: `configs/token_classification_example.py`
```python
"""
Example configuration for token-level classification training.
Supports flexible number of classes (not just binary).
"""

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False
attention_type = 'bidirectional'  # Required for classification
use_rope = True

# Training configuration
training_type = 'token_classification'
num_token_classes = 3  # Flexible: 2 for binary, 3+ for multi-class
batch_size = 16
block_size = 1024
max_iters = 10000
learning_rate = 5e-4

# Transfer learning (optional)
init_from_checkpoint = 'out/ckpt_unmasking_5000.pt'  # Path to pretrained model
freeze_transformer = False  # False = fine-tuning, True = feature extraction

# Output
out_dir = 'out_token_classification'
wandb_project = 'token_classification'
wandb_run_name = 'ai_detector_v1'
```

**File**: `configs/sequence_scoring_example.py`
```python
"""
Example configuration for sequence-level scoring training.
Returns continuous scores 0-1 for quality assessment.
"""

# Model configuration
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False
attention_type = 'bidirectional'  # Required for scoring
use_rope = True

# Training configuration
training_type = 'sequence_scoring'
batch_size = 16
block_size = 512  # Shorter sequences for sequence scoring
max_iters = 10000
learning_rate = 5e-4
warmup_iters = 1000
lr_decay_iters = 8000
min_lr = 5e-5

# Transfer learning (recommended for scoring tasks)
init_from_checkpoint = 'out/ckpt_unmasking_10000.pt'  # Use pretrained language model
freeze_transformer = False  # Fine-tune entire model

# Data configuration
dataset = 'your_quality_dataset'
use_paragraph_boundaries = True

# Evaluation
eval_interval = 100
eval_iters = 20
log_interval = 10

# Output
out_dir = 'out_sequence_scoring'
wandb_project = 'sequence_scoring'
wandb_run_name = 'quality_scorer_v1'
```

#### Step 15: Update Training Script Validation
**File**: `train_run.py`
**Location**: After configuration loading (around line 114)

**Add validation for all modes**:
```python
# Add validation for token classification mode
if training_type in ['token_classification', 'remasking_binary']:
    if attention_type != 'bidirectional':
        print("WARNING: Token classification requires bidirectional attention")
        attention_type = 'bidirectional'

    # Update model args
    model_args['mode'] = ModelMode.TOKEN_CLASSIFIER
    model_args['num_token_classes'] = num_token_classes

    print(f"Token classification mode enabled:")
    print(f"  - Attention type: {attention_type}")
    print(f"  - Number of classes: {num_token_classes}")

# Add validation for sequence scoring mode
elif training_type == 'sequence_scoring':
    if attention_type != 'bidirectional':
        print("WARNING: Sequence scoring requires bidirectional attention")
        attention_type = 'bidirectional'

    # Ensure we have a CLS token ID
    if not hasattr(locals(), 'cls_token_id') or cls_token_id is None:
        cls_token_id = extended_vocab_size + 1
        print(f"Setting cls_token_id = {cls_token_id}")

    # Update model args
    model_args['cls_token_id'] = cls_token_id
    model_args['mode'] = ModelMode.SEQUENCE_SCORER

    print(f"Sequence scoring mode enabled:")
    print(f"  - Attention type: {attention_type}")
    print(f"  - CLS token ID: {cls_token_id}")
    print(f"  - Block size: {block_size} (includes CLS token)")

# Language modeling mode (default)
elif training_type == 'unmasking':
    model_args['mode'] = ModelMode.LANGUAGE_MODEL
    print(f"Language modeling mode enabled (unmasking)")
```

---

## Implementation Checklist

### ✅ Already Working (No Changes Needed)
- [x] **Mode 1: Language Modeling** - Fully implemented as `training_type = 'unmasking'`
- [x] **Bidirectional attention** - Already supported
- [x] **Modular training utilities** - Well-structured in `training_utils/`
- [x] **Batch generation infrastructure** - Sophisticated caching and prefetching
- [x] **Model evaluation system** - Comprehensive loss estimation

### 🔧 Needs Implementation (New Work)
- [ ] **ModelMode enum** in `model.py`
- [ ] **Updated GPTConfig** with mode selection, transfer learning, and validation
- [ ] **Mode-specific model heads** in `GPT.__init__` with transfer learning support
- [ ] **Mode-specific forward pass** in `GPT.forward`
- [ ] **Refactor token classification** from binary to flexible multi-class
- [ ] **Sequence scoring batch generation** function
- [ ] **Dataset interface** for sequence-level labels
- [ ] **Training script updates** for mode selection and transfer learning
- [ ] **Configuration validation** for all modes
- [ ] **Evaluation metrics** for sequence scoring
- [ ] **Transfer learning integration** for loading pretrained weights

### 🧪 Testing Strategy

#### Test Mode 1 (Language Modeling)
```bash
# Should work without changes
python train_run.py --training_type=unmasking --attention_type=bidirectional
```

#### Test Mode 2 (Token Classification)
```bash
# Multi-class token classification (new)
python train_run.py --training_type=token_classification --num_token_classes=3 --attention_type=bidirectional

# Binary token classification (backward compatibility)
python train_run.py --training_type=remasking_binary --attention_type=bidirectional

# With transfer learning (feature extraction)
python train_run.py --training_type=token_classification --num_token_classes=2 \
    --init_from_checkpoint=out/ckpt_unmasking_5000.pt --freeze_transformer=True

# With transfer learning (fine-tuning)
python train_run.py --training_type=token_classification --num_token_classes=2 \
    --init_from_checkpoint=out/ckpt_unmasking_5000.pt --freeze_transformer=False
```

#### Test Mode 3 (Sequence Scoring)
```bash
# Sequence scoring from scratch
python train_run.py --training_type=sequence_scoring --attention_type=bidirectional --block_size=512

# With transfer learning (recommended)
python train_run.py --training_type=sequence_scoring --block_size=512 \
    --init_from_checkpoint=out/ckpt_unmasking_10000.pt --freeze_transformer=False
```

---

## Migration Path

### Option A: Gradual Migration (Recommended)
1. **Phase 1**: Implement ModelMode enum and update model.py (Steps 1-4)
2. **Phase 2**: Add sequence classification training type (Steps 5-8)
3. **Phase 3**: Implement data pipeline (Steps 9-11)
4. **Phase 4**: Add configuration and testing (Steps 12-13)
5. **Test each phase** before proceeding to the next

### Option B: All-at-Once Migration
1. Implement all changes simultaneously
2. Higher risk but faster if you're confident
3. Requires more careful testing afterward

---

## Key Design Decisions

### 1. **Backward Compatibility**
- All existing `training_type` values continue to work
- Existing model checkpoints remain loadable
- Configuration files need minimal updates

### 2. **Token Management**
- `[CLS]` token uses `extended_vocab_size + 1`
- Existing special tokens (mask, wrong, etc.) unchanged
- Block size reduced by 1 for sequence scoring to accommodate `[CLS]`

### 3. **Loss Functions**
- **Language Modeling**: Cross-entropy over vocabulary (unchanged)
- **Token Classification**: Cross-entropy with flexible class balancing (supports 2+ classes)
- **Sequence Scoring**: MSE for continuous 0-1 score prediction

### 4. **Attention Requirements**
- **Language Modeling**: Supports both causal and bidirectional
- **Token Classification**: Requires bidirectional (enforced)
- **Sequence Scoring**: Requires bidirectional (enforced)

### 5. **Data Format**
- **Language Modeling**: Text sequences with masking
- **Token Classification**: Text sequences with multi-class labels per token
- **Sequence Scoring**: Text sequences with single continuous score (0-1) per sequence

### 6. **Transfer Learning Support**
- **Feature Extraction**: Freeze transformer, train only classification head
- **Fine-tuning**: Load pretrained weights, train entire model
- **Checkpoint Loading**: Automatic filtering of incompatible head weights
- **Backward Compatibility**: Existing checkpoints remain loadable

---

## Next Steps

1. **Start with Phase 1** (model architecture updates)
2. **Test thoroughly** after each phase
3. **Create example datasets** for sequence classification
4. **Document configuration options** for each mode
5. **Add comprehensive unit tests** for new functionality

This implementation plan maintains the sophisticated infrastructure you've already built while cleanly extending it to support the three distinct training paradigms you need.