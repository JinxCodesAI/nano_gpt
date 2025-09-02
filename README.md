
# Discrete Diffusion LLM Training - Unmasking Framework

A training framework for discrete diffusion-based language models using unmasking training, derived from [nanoGPT](https://github.com/karpathy/nanoGPT). This repository implements progressive unmasking with stage-based curriculum learning for character-level language modeling.

Unmasking training uses a curriculum learning approach where the model learns to predict masked tokens, starting with easier masking patterns and progressively advancing to more challenging ones. The model uses bidirectional attention and learns through stages that automatically advance based on validation performance.

The framework is designed for character-level tokenization and includes masking strategies, validation systems, and automatic stage progression.

## install

```
pip install torch numpy wandb
```

Dependencies:

- [pytorch](https://pytorch.org) - Core deep learning framework
- [numpy](https://numpy.org/install/) - Numerical computations
- `wandb` - Optional logging and experiment tracking

Note: This framework is designed specifically for character-level tokenization and does not require transformers, datasets, or tiktoken packages.

## Dataset Preparation

The framework supports different model modes that require different data formats:

### Model Modes and Data Requirements

#### 1. Language Model (`language_model`)
**Purpose**: Next-token prediction (standard autoregressive language modeling)
**Data format**:
- Input: `X = tokens[:-1]` (all tokens except last)
- Target: `Y = tokens[1:]` (input shifted by 1)
- Mask: `mask = all_ones` (no masking)
**Example datasets**: `shakespeare_char` (general text)

#### 2. Token Classifier (`token_classifier`)
**Purpose**: Per-token binary/multi-class classification (e.g., masked language modeling)
**Data format**:
- Input: `X = masked_tokens` (some tokens replaced with `<MASK>`)
- Target: `Y = original_tokens` (ground truth for all positions)
- Mask: `mask = masking_pattern` (indicates which tokens to predict)
**Example datasets**: `shakespeare_char_diffusion` (unmasking training)

#### 3. Sequence Classifier (`sequence_classifier`)
**Purpose**: Sequence-level classification or regression
**Data format**:
- Input: `X = [CLS] + tokens` (sequence with special CLS token)
- Target: `Y = sequence_label` (single label for entire sequence)
- Mask: `mask = attention_mask` (sequence-level attention)
**Example datasets**: Custom datasets with sequence-level labels

---

### Language Model Dataset Setup

For standard next-token prediction:

#### Basic Structure
```
data/your_dataset_name/
├── prepare.py              # Data preparation script
├── input.txt              # Raw text data
├── train.bin              # Generated: tokenized training data
├── val.bin                # Generated: tokenized validation data
└── meta.pkl               # Generated: vocabulary and dataset metadata
```

#### Required meta.pkl Contents
```python
meta = {
    'vocab_size': 65,                    # Base vocabulary size
    'itos': {...},                       # Index to string mapping
    'stoi': {...},                       # String to index mapping
    'dataset_type': 'character',         # 'character', 'subword', 'word'
    'block_size': 1024,                  # Fixed sequence length
    'supported_model_modes': ['language_model'],  # Only supports language modeling
    'data_shapes': {
        'X': '(batch_size, block_size-1)',  # Input tokens
        'Y': '(batch_size, block_size-1)',  # Target tokens (shifted)
        'mask': '(batch_size, block_size-1)',  # All ones (no masking)
        'description': 'Language modeling: X=input[:-1], Y=input[1:]'
    }
}
```

#### Usage
```python
# In your training config file
dataset = 'shakespeare_char'
model_mode = 'language_model'
```

---

### Token Classifier Dataset Setup

For masked language modeling / unmasking training:

#### Advanced Structure
```
data/your_dataset_name/
├── prepare.py              # Enhanced preparation script
├── training_config.py      # Training-specific configuration
├── data_utils.py          # Dataset-specific utilities
├── input.txt              # Raw text data
├── train.bin              # Generated: tokenized training data
├── val.bin                # Generated: tokenized validation data
├── meta.pkl               # Generated: enhanced metadata
├── prepared_batches/       # Generated: pre-computed training batches
│   ├── train_iter_0000.pt
│   ├── train_iter_0200.pt
│   └── ...
└── validation/            # Generated: fixed validation pools
    ├── validation_meta.pkl
    ├── stage_00_file_00.pt
    └── ...
```

#### Required meta.pkl Contents
```python
meta = {
    'vocab_size': 65,                    # Base vocabulary size
    'itos': {...},                       # Index to string mapping
    'stoi': {...},                       # String to index mapping
    'extended_vocab_size': 80,           # With special tokens for masking
    'special_tokens': {                  # Special token IDs
        'mask_token_id': 65,
        'wrong_token_id': 66,
        'remask_good_id': 67,
        'remask_wrong_id': 68,
    },
    'dataset_type': 'character',         # 'character', 'subword', 'word'
    'block_size': 1024,                  # Fixed sequence length
    'supported_model_modes': ['token_classifier'],  # Only supports token classification
    'training_stages': 8,                # Number of curriculum stages
    'validation_stages': 14,             # Number of validation stages
    'data_shapes': {
        'X': '(batch_size, block_size)',     # Masked input tokens
        'Y': '(batch_size, block_size)',     # Original target tokens
        'mask': '(batch_size, block_size)',  # Masking pattern (0/1)
        'description': 'Token classification: X=masked_input, Y=original_targets, mask=masking_pattern'
    }
}
```

#### Training Configuration Example
```python
# training_config.py - Defines curriculum stages
UNMASKING_STAGES = [
    {
        "type": "sticky",
        "target_masked_ratio": 0.2,
        "p1_probability": 0.15,
        "p2_probability": 0.3,
        "val_loss_stale_count": 6
    },
    # ... more stages with increasing difficulty
]

VALIDATION_STAGES = [
    # ... validation configurations for each difficulty level
]

# Dataset constraints
BLOCK_SIZE = 1024
USE_PARAGRAPH_BOUNDARIES = True
VALIDATION_SAMPLES_PER_STAGE = 100
```

#### Usage
```python
# In your training config file
dataset = 'shakespeare_char_diffusion'
model_mode = 'token_classifier'
training_type = 'unmasking'
```

---

## Quick Start

### Language Modeling (Next-Token Prediction)
```python
# config/language_model.py
dataset = 'shakespeare_char'           # General text dataset
model_mode = 'language_model'          # Next-token prediction
batch_size = 16
max_iters = 5000
learning_rate = 1e-3
```
```sh
python train_run.py config/language_model.py
```

### Token Classification (Unmasking/Diffusion Training)
```python
# config/token_classifier.py
dataset = 'shakespeare_char_diffusion'  # Dataset with masking support
model_mode = 'token_classifier'         # Per-token classification
training_type = 'unmasking'             # Curriculum-based unmasking
batch_size = 16
max_iters = 8000
```
```sh
python train_run.py config/token_classifier.py
```

### Pre-configured Experiments
```sh
# Unmasking training with curriculum learning
python train_run.py config/shakespeare_diffusion/experiment1.py
```

## Unmasking Training Overview

**Unmasking training** is a curriculum learning approach where:

1. **Masking**: Random tokens in the input are replaced with a special `<MASK>` token
2. **Prediction**: The model learns to predict the original tokens at masked positions
3. **Bidirectional Context**: Unlike autoregressive models, the model can attend to both past and future tokens
4. **Stage Progression**: Training automatically advances through stages with increasing difficulty
5. **Curriculum Learning**: Starts with easy masking patterns, progresses to harder ones

### Masking Strategies

**Sticky Masking** (Recommended):
- Uses neighboring token context to create coherent masked regions
- `p1_probability`: Chance of masking when no neighbors are masked
- `p2_probability`: Chance of masking when neighbors are already masked
- `target_masked_ratio`: Target fraction of tokens to mask

**Random Masking**:
- Uniformly random token masking
- `max_masked_ratio`: Maximum fraction of tokens to mask
- Each sample gets a different random masking ratio (0 to max)

### Stage-Based Learning

Training progresses through predefined stages automatically:
- Each stage has its own masking configuration
- Stages advance when validation loss stops improving
- `val_loss_stale_count`: How many evaluations to wait before advancing
- Validation uses samples from ALL stages for consistent evaluation

## Model Architecture

The models use bidirectional attention with Rotary Position Embeddings (RoPE):
- **Attention**: Bidirectional self-attention (no causal masking)
- **Position**: RoPE instead of absolute position embeddings
- **Vocabulary**: Extended with special tokens for masking/corruption

## Quick Start Examples

### Standard Unmasking Training (GPU Recommended)

```sh
python train_run.py config/shakespeare_diffusion/experiment1.py
```

Or create your own config file with:
```python
# config/my_experiment.py
dataset = 'shakespeare_char_diffusion'
batch_size = 16
block_size = 1024
max_iters = 5000
learning_rate = 1e-3
training_type = 'unmasking'
```

### Transfer Learning

The framework supports transfer learning between model modes:

#### From Language Model to Token Classifier
```python
# config/transfer_to_classifier.py
init_from = 'resume'
pretrained_checkpoint_path = 'out/language_model.pt'  # Trained with language_model mode
dataset = 'shakespeare_char_diffusion'                # Dataset supporting token_classifier
model_mode = 'token_classifier'                       # Switch to classification
transfer_learning_mode = 'feature_extraction'        # or 'fine_tuning'
learning_rate = 1e-3
max_iters = 5000
```
```sh
python train_run.py config/transfer_to_classifier.py
```

#### Transfer Learning Modes
- **Feature Extraction**: Freeze backbone, train only the classification head
- **Fine-tuning**: Train all parameters with pretrained initialization

### CPU Training (Reduced Model)

```python
# config/cpu_training.py
dataset = 'shakespeare_char_diffusion'
training_type = 'unmasking'
device = 'cpu'
compile = False
batch_size = 8
block_size = 256
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 10000
learning_rate = 5e-4
```
```sh
python train_run.py config/cpu_training.py
```

### Apple Silicon (MPS)

```python
# config/mps_training.py
dataset = 'shakespeare_char_diffusion'
training_type = 'unmasking'
device = 'mps'
batch_size = 12
block_size = 512
max_iters = 25000
```
```sh
python train_run.py config/mps_training.py
```

## Detailed Configuration Guide

### Unmasking Stage Configuration

Stages are defined in dataset configuration files (e.g., `data/shakespeare_char_diffusion/training_config.py`). Each stage is a dictionary with:

#### Sticky Masking Stages
```python
# In data/your_dataset/training_config.py
UNMASKING_STAGES = [
    {
        "type": "sticky",
        "target_masked_ratio": 0.2,      # 20% of tokens masked
        "p1_probability": 0.3,           # 30% chance to start new masked region
        "p2_probability": 0.0,           # 0% chance to extend (no stickiness yet)
        "val_loss_stale_count": 5        # Advance after 5 evals without improvement
    },
    {
        "type": "sticky",
        "target_masked_ratio": 0.4,      # 40% of tokens masked
        "p1_probability": 0.2,           # 20% chance to start new regions
        "p2_probability": 0.8,           # 80% chance to extend existing regions
        "val_loss_stale_count": 5
    },
    {
        "type": "sticky",
        "target_masked_ratio": 0.6,      # 60% of tokens masked (hardest)
        "p1_probability": 0.1,           # 10% chance for new regions
        "p2_probability": 0.9,           # 90% chance to extend (very sticky)
        "val_loss_stale_count": 10       # More patience for final stage
    }
]
```

#### Random Masking Stages
```python
# In data/your_dataset/training_config.py
UNMASKING_STAGES = [
    {
        "type": "random",
        "max_masked_ratio": 0.3,         # Up to 30% tokens masked randomly
        "val_loss_stale_count": 5
    },
    {
        "type": "random",
        "max_masked_ratio": 0.7,         # Up to 70% tokens masked
        "val_loss_stale_count": 8
    }
]
```

### General LLM Training Parameters

#### Model Architecture
```python
# Model size (adjust based on compute/memory)
n_layer = 6           # Number of transformer layers (4-12 typical)
n_head = 6            # Number of attention heads (usually same as n_layer)
n_embd = 384          # Embedding dimension (128, 256, 384, 512, 768)
dropout = 0.2        # Dropout rate (0.0-0.1, lower for small models)
bias = False          # Use bias in Linear/LayerNorm (False often better)

# Context and batch
block_size = 1024     # Sequence length (256, 512, 1024, 2048)
batch_size = 16       # Batch size (adjust for GPU memory)
```

#### Training Schedule
```python
# Learning rate schedule
learning_rate = 1e-3  # Peak learning rate (1e-4 to 5e-3)
warmup_iters = 2000   # Warmup steps (usually 10-20% of total)
lr_decay_iters = 41000 # Decay until this iteration (usually 80-90% of max_iters)
min_lr = 1e-4         # Minimum LR (usually learning_rate/10)
max_iters = 50000     # Total training iterations

# Optimizer
beta1 = 0.9           # Adam beta1 (momentum)
beta2 = 0.99          # Adam beta2 (RMSprop-like)
weight_decay = 2e-2   # L2 regularization (1e-4 to 1e-2)
grad_clip = 1.0       # Gradient clipping (0.5 to 2.0)
```

#### Data Configuration
```python
# Data handling
dataset = 'shakespeare_char_diffusion'  # Character-level dataset
eval_interval = 200                     # How often to evaluate (iterations)
eval_iters = 20                         # Number of batches for evaluation
```

### Hardware-Specific Settings

#### For GPU Training
```python
device = 'cuda'              # Use GPU
compile = True               # Use torch.compile (PyTorch 2.0+)
dtype = 'float16'           # Mixed precision (faster, less memory)
batch_size = 16             # Larger batches
block_size = 1024           # Longer sequences
```

#### For CPU Training
```python
device = 'cpu'              # Use CPU
compile = False             # Disable compilation
dtype = 'float32'           # Full precision
batch_size = 4              # Smaller batches
block_size = 256            # Shorter sequences
n_layer = 4                 # Smaller model
n_embd = 128
```

#### Memory Optimization
```python
# If running out of memory, try:
batch_size = 8              # Reduce batch size
block_size = 512            # Reduce sequence length
n_layer = 4                 # Reduce model size
n_embd = 256
gradient_accumulation_steps = 2  # Simulate larger batches
```

## Validation and Monitoring

### Validation Strategy

Unmasking training uses a structured validation approach:

**Pre-created Validation Sets**:
- Generated once at training start
- Contains samples from ALL stages (not just current)
- Ensures consistent evaluation throughout training
- Distributed evenly across difficulty levels

**Per-Stage Validation**:
- Tracks performance on each stage separately
- Shows which difficulty levels are improving
- Helps identify training issues

### Key Metrics to Monitor

**Loss Metrics**:
- `val_loss`: Overall validation loss
- `val_stage_X_loss`: Loss on each specific stage
- `train_loss`: Training loss (should be lower than validation)

**Performance Metrics**:
- `val_model_vs_random`: How much better than random guessing
- `val_most_likely_accuracy`: Percentage of correct top-1 predictions
- `val_masked_token_ratio`: Actual masking ratios achieved

**Stage Progression**:
- Current stage number
- Validation loss stale count (resets when loss improves)
- Best validation loss for current stage

### Training Logs Example
```
step 4000: train loss 2.1234, val loss 2.3456, lr 0.000800
Stage 1 (sticky): target_ratio=0.4, p1=0.2, p2=0.8, stale_count=2
  val model vs random: 3.45x better
  val accuracy: 0.3421 (random baseline: 0.0154)
  Most likely guess correct P %: 34.2%
Per-stage validation losses:
  Stage 0 (sticky): 2.1234 (320 samples) - ratio=0.2
  Stage 1 (sticky): 2.4567 (320 samples) - ratio=0.4
  Stage 2 (sticky): 2.8901 (320 samples) - ratio=0.6
```

### When to Worry

**Bad Signs**:
- Validation loss not decreasing after many iterations
- Model vs random ratio < 2x (model not learning)
- NaN/Inf in losses (training instability)
- Stage never advancing (stale count keeps growing)

**Good Signs**:
- Validation loss steadily decreasing
- Model vs random ratio > 5x
- Stages advancing automatically
- Per-stage losses improving across all stages

## Transfer Learning Guide

### Overview

The framework supports transfer learning capabilities, allowing you to:

1. **Load pretrained models** trained with `binary_classification=False` (language modeling)
2. **Switch to binary classification** (`binary_classification=True`) with automatic head reinitialization  
3. **Choose training mode**:
   - **Feature Extraction**: Freeze backbone, train only the classification head
   - **Fine-tuning**: Train all parameters with pretrained initialization
4. **Train normally** using the existing robust training pipeline

### Transfer Learning Workflow

#### Step 1: Train a Pretrained Model (Language Modeling)

First, train a model with standard unmasking on your text data:

```python
# config/pretrain.py
dataset = 'shakespeare_char_diffusion'
training_type = 'unmasking'
batch_size = 16
max_iters = 20000
out_dir = 'pretrained_model'
```
```sh
python train_run.py config/pretrain.py
```

This creates a language model checkpoint (e.g., `pretrained_model/ckpt_unmasking_20000.pt`).

#### Step 2: Transfer Learning for Classification

**Feature Extraction Mode** (recommended first approach):
```python
# config/feature_extraction_transfer.py
init_from = 'resume'
pretrained_checkpoint_path = 'pretrained_model/ckpt_unmasking_20000.pt'
switch_to_binary = True
transfer_learning_mode = 'feature_extraction'
training_type = 'unmasking'
learning_rate = 1e-3
max_iters = 5000
out_dir = 'feature_extraction_model'
```
```sh
python train_run.py config/feature_extraction_transfer.py
```

**Fine-tuning Mode** (for better performance):
```python
# config/fine_tuning_transfer.py
init_from = 'resume'
pretrained_checkpoint_path = 'pretrained_model/ckpt_unmasking_20000.pt'
switch_to_binary = True
transfer_learning_mode = 'fine_tuning'
training_type = 'unmasking'
learning_rate = 1e-4
max_iters = 10000
out_dir = 'fine_tuning_model'
```
```sh
python train_run.py config/fine_tuning_transfer.py
```

### Transfer Learning Configuration

#### Required Parameters

In your configuration file:
- `init_from = 'resume'`: Use checkpoint loading mode
- `pretrained_checkpoint_path = 'PATH'`: Path to your pretrained language model checkpoint
- `switch_to_binary = True`: Switch from language modeling to binary classification head
- `transfer_learning_mode = 'MODE'`: Choose `'feature_extraction'` or `'fine_tuning'`

#### Transfer Learning Modes

**Feature Extraction (`feature_extraction`)**:
- **Freezes**: All transformer layers (backbone)
- **Trains**: Only the binary classification head
- **Use case**: When you have limited training data or want fast training
- **Learning rate**: Can use higher rates (1e-3 to 5e-3)
- **Training time**: Much faster due to fewer parameters

**Fine-tuning (`fine_tuning`)**:
- **Freezes**: Nothing - all parameters trainable
- **Trains**: Entire model with pretrained initialization
- **Use case**: When you have sufficient training data and want best performance
- **Learning rate**: Use lower rates (1e-4 to 1e-3) to avoid destroying pretrained features
- **Training time**: Longer but often achieves better results

### Transfer Learning Best Practices

#### Learning Rates
```python
# Feature extraction: Higher learning rates OK
transfer_learning_mode = 'feature_extraction'
learning_rate = 1e-3

# Fine-tuning: Lower learning rates to preserve pretrained features
transfer_learning_mode = 'fine_tuning'
learning_rate = 1e-4
```

#### Training Duration
```python
# Feature extraction: Shorter training usually sufficient
transfer_learning_mode = 'feature_extraction'
max_iters = 2000  # to 5000

# Fine-tuning: Longer training for convergence
transfer_learning_mode = 'fine_tuning'
max_iters = 5000  # to 15000
```

#### Model Architecture Compatibility
- **Compatible**: Pretrained and target models must have same `n_layer`, `n_head`, `n_embd`, `block_size`
- **Flexible**: `vocab_size` and `binary_classification` can differ (handled automatically)
- **Error handling**: Framework validates architecture compatibility and provides clear error messages

### Monitoring Transfer Learning

#### Key Metrics to Watch

**Parameter Counts**:
```
*** TRANSFER LEARNING OPTIMIZER SETUP ***
Optimizer parameter summary:
  Total parameters: 2,345,678
  Trainable parameters: 1,536 (feature extraction) or 2,345,678 (fine-tuning)
  Frozen parameters: 2,344,142 (feature extraction) or 0 (fine-tuning)
  Trainable percentage: 0.1% (feature extraction) or 100.0% (fine-tuning)
```

**Training Logs**:
```
*** TRANSFER LEARNING MODE ***
Loading pretrained weights from: pretrained_model/ckpt_unmasking_20000.pt
Transfer learning mode: feature_extraction
Switch to binary classification: True
✓ Pretrained weights loaded successfully
✓ Switched to binary classification
✓ Backbone frozen for feature extraction
```

#### Expected Behavior

**Feature Extraction**:
- Very few trainable parameters (typically < 1% of total)
- Fast training iterations
- Quick convergence (often within 1000-3000 iterations)

**Fine-tuning**:
- All parameters trainable
- Slower training iterations
- More gradual convergence
- Often achieves lower final loss than feature extraction

### Transfer Learning Examples

#### Example 1: Text Classification
```sh
# 1. Train language model on text corpus
python train_run.py config/shakespeare_diffusion/experiment1.py

# 2. Feature extraction for classification task
# config/classifier_features.py
init_from = 'resume'
pretrained_checkpoint_path = 'out/ckpt_unmasking_20000.pt'
switch_to_binary = True
transfer_learning_mode = 'feature_extraction'
max_iters = 3000
out_dir = 'classifier_features'

python train_run.py config/classifier_features.py

# 3. Fine-tuning for better performance
# config/classifier_finetuned.py
init_from = 'resume'
pretrained_checkpoint_path = 'out/ckpt_unmasking_20000.pt'
switch_to_binary = True
transfer_learning_mode = 'fine_tuning'
learning_rate = 5e-5
max_iters = 8000
out_dir = 'classifier_finetuned'

python train_run.py config/classifier_finetuned.py
```

#### Example 2: Domain Adaptation
```sh
# 1. Train on general text (e.g., Shakespeare)
python train_run.py config/shakespeare_diffusion/experiment1.py

# 2. Fine-tune for domain-specific classification
# config/domain_classifier.py
init_from = 'resume'
pretrained_checkpoint_path = 'out/latest.pt'
dataset = 'domain_specific_char_diffusion'
switch_to_binary = True
transfer_learning_mode = 'fine_tuning'
learning_rate = 1e-4
max_iters = 12000
out_dir = 'domain_classifier'

python train_run.py config/domain_classifier.py
```

### Transfer Learning Troubleshooting

#### Common Issues

**Architecture Mismatch**:
```
FileNotFoundError: Pretrained checkpoint file not found
```
- Check the checkpoint path is correct
- Ensure the checkpoint file exists

**Parameter Count Warnings**:
```
WARNING: Optimizer param count (X) != trainable param count (Y)
```
- This indicates an internal inconsistency (should not occur with the current implementation)
- Try restarting training or check for custom optimizer configurations

**Poor Transfer Performance**:
- **Try lower learning rates** for fine-tuning (1e-5 to 1e-4)
- **Train longer** - transfer learning can take time to converge
- **Check pretrained model quality** - ensure base model was well-trained
- **Verify data compatibility** - ensure training data is appropriate for the task

#### Performance Tips

**For Better Results**:
- Use well-trained pretrained models (>15k iterations)
- Start with feature extraction, then try fine-tuning
- Use appropriate learning rates (lower for fine-tuning)
- Monitor both training and validation metrics

**For Faster Training**:
- Use feature extraction mode for rapid prototyping
- Freeze more layers if you implement custom freezing
- Use larger batch sizes if memory allows

## Advanced Usage

### Resuming Training

Training automatically saves checkpoints and can be resumed:

```python
# config/resume_training.py
init_from = 'resume'
out_dir = 'your_checkpoint_dir'
# ... other parameters from original training
```
```sh
python train_run.py config/resume_training.py
```

Checkpoints preserve:
- Model and optimizer state
- Training iteration count
- Current stage and stage progression
- Best validation loss for current stage
- Validation loss stale count

The framework automatically finds the latest checkpoint (`ckpt_unmasking_XXXXX.pt`).

### Custom Stage Configurations

**Progressive Difficulty Example**:
```python
# In data/your_dataset/training_config.py
UNMASKING_STAGES = [
    # Stage 0: Very easy - scattered 15% masking
    {"type": "sticky", "target_masked_ratio": 0.15, "p1_probability": 0.4, "p2_probability": 0.1, "val_loss_stale_count": 3},

    # Stage 1: Easy - small regions, 30% masking
    {"type": "sticky", "target_masked_ratio": 0.30, "p1_probability": 0.3, "p2_probability": 0.6, "val_loss_stale_count": 5},

    # Stage 2: Medium - larger regions, 50% masking
    {"type": "sticky", "target_masked_ratio": 0.50, "p1_probability": 0.2, "p2_probability": 0.8, "val_loss_stale_count": 5},

    # Stage 3: Hard - very large regions, 70% masking
    {"type": "sticky", "target_masked_ratio": 0.70, "p1_probability": 0.1, "p2_probability": 0.9, "val_loss_stale_count": 8},
]
```

**Mixed Strategy Example**:
```python
# In data/your_dataset/training_config.py
UNMASKING_STAGES = [
    # Start with random masking
    {"type": "random", "max_masked_ratio": 0.2, "val_loss_stale_count": 3},

    # Progress to sticky masking
    {"type": "sticky", "target_masked_ratio": 0.3, "p1_probability": 0.3, "p2_probability": 0.7, "val_loss_stale_count": 5},
    {"type": "sticky", "target_masked_ratio": 0.5, "p1_probability": 0.2, "p2_probability": 0.8, "val_loss_stale_count": 8},
]
```

### Hyperparameter Tuning Guidelines

**Start with these safe defaults**:
```python
# Model (for Shakespeare-sized datasets)
n_layer = 6
n_head = 6  
n_embd = 384
block_size = 1024
batch_size = 16

# Learning
learning_rate = 1e-3
warmup_iters = 2000
weight_decay = 1e-3
```

**Scale up gradually**:
1. **More data**: Increase `max_iters`, `batch_size`
2. **More compute**: Increase `n_layer`, `n_embd`  
3. **Longer sequences**: Increase `block_size`
4. **Harder masking**: Higher `target_masked_ratio`, more stages

**Signs you can scale up**:
- Training loss still decreasing
- Model vs random ratio > 5x
- Stages advancing smoothly
- No memory issues

**Signs to scale down**:
- Out of memory errors
- Training loss plateauing quickly  
- Poor model vs random performance
- Stages not advancing

## Understanding Unmasking Training

### How It Differs from Autoregressive Training

**Traditional Autoregressive (GPT-style)**:
- Predicts next token given previous tokens only
- Causal attention (can't see future)
- Left-to-right generation
- Loss on all positions

**Unmasking Training**:
- Predicts masked tokens given all other tokens
- Bidirectional attention (can see past and future)
- Fill-in-the-blank style learning
- Loss only on masked positions
- Curriculum learning through masking difficulty

### Why Stage-Based Learning?

1. **Easy → Hard Progression**: Start with few masks, increase difficulty
2. **Stable Training**: Avoid overwhelming the model early
3. **Better Convergence**: Each stage builds on previous learning
4. **Automatic Advancement**: No manual intervention needed

### Masking Strategy Details

**Sticky Masking Benefits**:
- Creates coherent masked regions (words, phrases)
- More realistic than random masking
- Encourages understanding of context
- `p2_probability > p1_probability` creates "sticky" regions

**Example Progression**:
- **Stage 1**: 20% masked, no stickiness (scattered masks)
- **Stage 2**: 40% masked, some stickiness (small regions)
- **Stage 3**: 60% masked, very sticky (large regions)

### Character-Level Benefits

- **Fine-grained control**: Can mask individual characters
- **Spelling/morphology**: Learns character patterns
- **No tokenization issues**: No out-of-vocabulary problems
- **Language agnostic**: Works for any script

### Training Monitoring

Key metrics to watch:
- **Validation loss per stage**: Should decrease over time
- **Stage progression**: Automatic advancement when loss plateaus
- **Masked token ratio**: Should match target ratios
- **Model vs random**: Should improve significantly above random chance

## Performance Notes

The framework includes several optimizations for efficient unmasking training:

**Data Optimizations**:
- **Background prefetching**: Training batches prepared in parallel
- **Memory mapping**: Efficient data loading with numpy memmap
- **Vectorized masking**: GPU-optimized masking operations
- **Cached validation sets**: Pre-created validation data (no runtime generation)
- **Paragraph boundaries**: Optional alignment with text structure

**Model Optimizations**:
- **Flash Attention**: Automatic use when available (PyTorch 2.0+)
- **PyTorch compilation**: `torch.compile()` for ~2x speedup
- **Mixed precision**: `float16` training when using CUDA
- **Bidirectional attention**: Optimized for non-causal attention patterns

**Training Optimizations**:
- **Gradient accumulation**: Simulate larger batches
- **Automatic recovery**: Handles NaN/Inf without manual intervention
- **Stage caching**: Efficient validation across all difficulty levels

**Hardware-Specific Tips**:
- **CUDA**: Set `dtype = 'float16'`, `compile = True` in config
- **CPU**: Set `compile = False`, reduce model size in config
- **Apple Silicon**: Set `device = 'mps'` for GPU acceleration
- **Multi-GPU**: Framework supports DDP (set environment variables)

## Origin and Acknowledgements

This repository is derived from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. The original nanoGPT provides an excellent foundation for transformer training, which has been extensively modified here to support unmasking-based discrete diffusion training.

Key modifications from the original:
- **Bidirectional attention**: Allows attending to future tokens (essential for unmasking)
- **Stage-based curriculum learning**: Automatic progression through difficulty levels
- **Masking strategies**: Sticky and random masking with fine-grained control
- **Validation systems**: Pre-created validation sets with per-stage evaluation
- **Character-level focus**: Optimized for character-based tokenization
- **Instability detection**: Automatic recovery from training issues
- **Monitoring**: Detailed metrics and logging

This work explores a different paradigm from autoregressive language modeling, building upon nanoGPT's clean, readable foundation while implementing curriculum learning approaches to discrete diffusion.

## Troubleshooting

### Common Issues

**Training Not Starting**:
```
No unmasking stages defined, exiting...
```
- Solution: Define `UNMASKING_STAGES` list in your dataset's `training_config.py` file

**Memory Issues**:
```
CUDA out of memory
```
- Reduce `batch_size` (try 8, 4, 2)
- Reduce `block_size` (try 512, 256)
- Reduce model size (`n_layer=4, n_embd=256`)
- Use `gradient_accumulation_steps=2` to simulate larger batches

**Compilation Errors**:
```
torch.compile() not supported
```
- Set `compile = False` in your configuration file
- More common on older PyTorch versions or CPU training

**Training Instability**:
```
NaN detected in validation
```
- Lower learning rate (try 5e-4 instead of 1e-3)
- Increase warmup iterations
- Check for exploding gradients (reduce `grad_clip`)
- Framework includes automatic recovery from checkpoints

**Stage Not Advancing**:
- Check `val_loss_stale_count` - may need to be lower
- Validation loss may have plateaued naturally
- Try different masking ratios or probabilities

**Poor Performance**:
- Model vs random ratio < 2x means model isn't learning well
- Try longer training, different learning rate
- Check if data is properly formatted (character-level)
- Ensure `use_paragraph_boundaries=True` for better samples

### Performance Tuning Tips

**For Better Results**:
- Use sticky masking (more realistic than random)
- Start with lower masking ratios (0.1-0.2)
- Use paragraph boundaries for cleaner samples
- Train longer (50k+ iterations)
- Monitor per-stage validation losses

**For Faster Training**:
- Enable compilation: `compile = True` in config
- Use mixed precision: `dtype = 'float16'` in config
- Larger batch sizes if memory allows
- Use `device = 'mps'` on Apple Silicon

