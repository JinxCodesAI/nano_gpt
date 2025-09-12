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
│   └── input.txt                  # Raw text data (Shakespeare)
```

#### 1.2 Provider Class Structure

**File**: `data/sequence_scorer/prepare_streaming.py`

```python
from typing import Dict, Any, Optional
import torch
import os
from data.common.provider_base import DataProviderBase
from .mlm_inference import MLMInferenceEngine
from .synthetic_generation import create_synthetic_text

class SequenceScorerProvider(DataProviderBase):
    def __init__(self, *args, **kwargs):
        # Extract sequence scorer specific config
        self.mlm_checkpoint_path = kwargs.pop('mlm_checkpoint_path')
        self.mask_probability_range = kwargs.pop('mask_probability_range', (0.1, 0.8))
        self.cls_token_id = kwargs.pop('cls_token_id', 0)  # [CLS] token position
        
        super().__init__(*args, **kwargs)
        
        # Initialize MLM inference engine
        self.mlm_engine = MLMInferenceEngine(
            checkpoint_path=self.mlm_checkpoint_path,
            device='cpu',  # Use CPU for data generation
            verbose=self.verbose
        )
        
        # Load text data and create vocabulary (similar to char_diffusion)
        self._load_text_data()
        
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
        # Implementation details in Phase 2
        pass
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
from typing import Tuple
from data.char_diffusion.masking_utils import apply_bert_style_corruption_cpu

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
    
    # Apply BERT-style corruption to create input for MLM model
    corrupted_input = apply_bert_style_corruption_cpu(
        original_text, mask, mask_token_id, vocab_size, rng
    )
    
    # Use MLM model to predict masked tokens
    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input, 
        corrupted_input == mask_token_id,  # Only predict actual [MASK] tokens
        temperature=sampling_temperature,
        top_k=top_k
    )
    
    # Calculate actual synthenticity ratio
    # Count positions where we actually replaced with predictions
    synthetic_positions = corrupted_input == mask_token_id
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

def sample_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
    """Generate batch with synthetic text and synthenticity scores."""
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
            self.vocab_size - 1,  # Exclude [MASK] from random generation
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
```

### Phase 4: Configuration Integration

#### 4.1 Update Sequence Scorer Config

**File**: `config/sequence_scorer_config.py`

```python
"""Configuration for sequence scoring dataset with synthetic text generation"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 16  # Smaller batch for sequence scoring
block_size = 256  # Reasonable context for scoring

# MLM model for synthetic text generation
mlm_checkpoint_path = 'checkpoints/char_diffusion_mlm.pt'  # Path to trained MLM model
mask_probability_range = (0.1, 0.7)  # Range of masking ratios to generate
cls_token_id = 0  # Position for [CLS] token (adjust based on vocab)

# Model mode configuration  
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'

# Transfer learning settings
freeze_transformer = True
unfreeze_at_iteration = 2000
init_from_checkpoint = 'checkpoints/pretrained_lm.pt'

# Training settings optimized for sequence scoring
learning_rate = 1e-4
warmup_iters = 300
max_iters = 8000
eval_interval = 200

# Loss modifiers (most will be filtered out for MSE loss)
loss_modifiers_enabled = True
entropy_modifier_enabled = False  # N/A for regression
target_smoothing_enabled = False  # N/A for regression

# Data generation settings
batches_per_file = 50  # Smaller files due to MLM inference cost
max_backlog_files = 3
sleep_seconds = 5.0  # Longer sleep due to inference cost

print("Sequence scorer configuration loaded")
```

#### 4.2 Update Prepare.py Discovery

The existing `prepare.py` uses convention-based discovery, so no changes needed. The system will automatically find:
- Module: `data.sequence_scorer.prepare_streaming`  
- Class: `SequenceScorerProvider` (via `Provider` alias)

### Phase 5: Usage and Validation

#### 5.1 Training Pipeline Integration

**Usage commands:**

```bash
# 1. First train a char_diffusion MLM model (if not already available)
python train.py --config=config/train_char_diffusion.py --out_dir=checkpoints/char_mlm

# 2. Generate sequence scoring data using trained MLM model
python prepare.py config/sequence_scorer_config.py

# 3. Train sequence scorer model
python train.py config/sequence_scorer_config.py --out_dir=checkpoints/sequence_scorer
```

#### 5.2 Expected Data Flow

1. **Data Generation Phase** (prepare.py):
   - Load Shakespeare text 
   - For each batch:
     - Sample text sequences
     - Apply random masking (10%-70% of tokens)
     - Use MLM model to predict masked tokens
     - Add [CLS] token at beginning
     - Target = actual synthenticity ratio
   
2. **Training Phase** (train.py):
   - Load synthetic text with [CLS] tokens
   - Extract [CLS] representation from sequence scorer model
   - Predict synthenticity score (0-1) with sigmoid
   - Optimize MSE loss against true synthenticity ratios

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

This implementation provides:

1. **Seamless Integration**: Uses existing DataProviderBase architecture
2. **Flexible Configuration**: Configurable mask ratios and sampling parameters  
3. **Training Compatibility**: Works with existing sequence scorer model
4. **Monitoring**: Detailed logging of synthenticity distributions
5. **Reproducibility**: Deterministic generation with proper RNG seeding

The sequence scorer dataset will enable training models to detect AI-generated text by learning the relationship between local (token-level) and global (sequence-level) synthetic patterns.