# Multi-Mode GPT Implementation Guide

This document describes the new multi-mode functionality that extends the base GPT implementation with support for token classification and sequence scoring tasks, along with transfer learning capabilities.

## Overview

The implementation adds three operational modes while maintaining full backward compatibility:

1. **Language Model** - Standard autoregressive language modeling (existing functionality)
2. **Token Classifier** - Per-token classification with flexible number of classes  
3. **Sequence Scorer** - Sequence-level scoring (0-1 range) using [CLS] token

## Quick Start

### 1. Token Classification

```python
from model import GPT, GPTConfig, ModelMode

# Create token classification model
config = GPTConfig(
    n_layer=12, n_head=12, n_embd=768,
    vocab_size=50304, block_size=1024,
    
    # Classification specific
    mode=ModelMode.TOKEN_CLASSIFIER,
    num_token_classes=3,  # Adjust for your task
    attention_type='bidirectional',  # Auto-set for classification
    
    # Transfer learning (optional)
    init_from_checkpoint='path/to/pretrained/model.pt',
    freeze_transformer=True,
)

model = GPT(config)

# Use in training
input_ids = torch.randint(0, 50304, (2, 512))
targets = torch.randint(0, 3, (2, 512))  # Token class labels
logits, loss = model(input_ids, targets)
# logits shape: [batch, seq_len, num_classes]
```

### 2. Sequence Scoring

```python
# Create sequence scoring model  
config = GPTConfig(
    n_layer=12, n_head=12, n_embd=768,
    vocab_size=50304, block_size=1024,
    
    mode=ModelMode.SEQUENCE_SCORER,
    cls_token_id=101,  # [CLS] token ID from tokenizer
    attention_type='bidirectional',
    
    # Transfer learning
    init_from_checkpoint='path/to/pretrained/model.pt',
    freeze_transformer=True,
)

model = GPT(config)

# Use in training
input_ids = torch.randint(0, 50304, (2, 512))  
targets = torch.rand(2)  # Regression targets [0-1]
logits, loss = model(input_ids, targets)
# logits shape: [batch] with sigmoid activation
```

### 3. Using Existing Configuration Templates

```bash
# Token classification
python train.py --config=config/token_classifier_config.py \
    --init_from_checkpoint=checkpoints/pretrained.pt \
    --dataset=your_classification_data \
    --num_token_classes=5

# Sequence scoring
python train.py --config=config/sequence_scorer_config.py \
    --init_from_checkpoint=checkpoints/pretrained.pt \
    --dataset=your_scoring_data \
    --cls_token_id=101
```

## Key Features

### Transfer Learning

The implementation supports sophisticated transfer learning workflows:

**Feature Extraction Mode:**
- Freeze pretrained transformer weights
- Only train new classification/scoring head
- Faster training, less overfitting risk

**Full Fine-tuning Mode:**
- Unfreeze transformer weights after warmup
- Fine-tune entire model end-to-end  
- Better performance on target task

**Dynamic Unfreezing:**
```python
# Automatically unfreeze during training
config = GPTConfig(
    # ... other params ...
    freeze_transformer=True,
    unfreeze_at_iteration=2000,  # Unfreeze after 2000 steps
    unfreeze_lr_multiplier=0.1,  # Reduce LR when unfreezing
)
```

### Enhanced Loss Modifiers

Loss modifiers now work intelligently across all modes:

- **Language Model**: All modifiers supported
- **Token Classifier**: Entropy and target smoothing supported  
- **Sequence Scorer**: Most modifiers filtered out (MSE loss)

```python
from loss_modifiers import create_loss_modifier_pipeline

# Works automatically across all modes
pipeline = create_loss_modifier_pipeline({
    'loss_modifiers_enabled': True,
    'entropy_modifier_enabled': True,      # Good for classification
    'target_smoothing_enabled': True,      # Label smoothing  
    'mask_ratio_weight_enabled': False,    # Skip for classification
})

# Pipeline automatically filters compatible modifiers per mode
logits, loss = model(input_ids, targets, loss_modifiers=pipeline)
```

### Advanced Schedulers

New schedulers optimized for transfer learning:

```python
from core.scheduler import TransferLearningScheduler, WarmupOnlyScheduler, AdaptiveScheduler

# Transfer learning scheduler
scheduler = TransferLearningScheduler(
    base_lr=5e-5,
    head_lr_multiplier=10.0,  # Higher LR for new head
    feature_extraction_iters=2000,
    unfreeze_lr_drop=0.1,
)

# Returns different LRs for transformer vs head
lr_dict = scheduler.get_lr(iter_num, is_frozen=True)
# {'transformer': 0.0, 'head': 5e-4}
```

## Configuration Reference

### Model Modes

| Parameter | Language Model | Token Classifier | Sequence Scorer |
|-----------|---------------|------------------|-----------------|
| `mode` | `ModelMode.LANGUAGE_MODEL` | `ModelMode.TOKEN_CLASSIFIER` | `ModelMode.SEQUENCE_SCORER` |
| `attention_type` | `'causal'` | `'bidirectional'` (auto-set) | `'bidirectional'` (auto-set) |
| `num_token_classes` | N/A | Required (e.g., 3) | N/A |
| `cls_token_id` | N/A | N/A | Required (e.g., 101) |

### Transfer Learning Parameters

```python
GPTConfig(
    # Standard parameters...
    
    # Transfer learning
    init_from_checkpoint=None,      # Path to pretrained model
    freeze_transformer=False,       # Start with frozen transformer  
    unfreeze_at_iteration=None,     # When to unfreeze (None = never)
    unfreeze_lr_multiplier=0.1,     # LR reduction when unfreezing
)
```

### Loss Modifier Compatibility

| Modifier | Language Model | Token Classifier | Sequence Scorer |
|----------|---------------|------------------|-----------------|
| Entropy Modifier | ✅ | ✅ | ❌ |
| Target Smoothing | ✅ | ✅ | ❌ |  
| Mask Ratio Weight | ✅ | ✅ | ❌ |

## Examples

### Complete Token Classification Workflow

1. **Prepare Configuration:**
```python
# config/my_classification.py
model_mode = 'token_classifier'
num_token_classes = 4
init_from_checkpoint = 'checkpoints/pretrained_lm.pt'
freeze_transformer = True
unfreeze_at_iteration = 3000
learning_rate = 5e-5
max_iters = 10000

# Enable helpful modifiers
loss_modifiers_enabled = True
entropy_modifier_enabled = True
target_smoothing_enabled = True
target_smoothing_factor = 0.1
```

2. **Run Training:**
```bash
python train.py --config=config/my_classification.py \
    --dataset=my_token_data \
    --out_dir=checkpoints/token_classifier
```

3. **Monitor Training:**
The training will:
- Start with frozen transformer (feature extraction)
- Use entropy-based loss weighting for better classification
- Apply label smoothing to prevent overfitting
- Automatically unfreeze at iteration 3000
- Continue with full fine-tuning

### Sequence Scoring Workflow

```python
# config/my_scoring.py
model_mode = 'sequence_scorer'
cls_token_id = 0  # Adjust based on tokenizer
init_from_checkpoint = 'checkpoints/pretrained_lm.pt'
freeze_transformer = True
unfreeze_at_iteration = 2000
learning_rate = 1e-4

# Sequence scoring uses MSE loss, so most modifiers are filtered out
loss_modifiers_enabled = True  # But modifiers auto-filter based on mode
```

## Backward Compatibility

**All existing MLM training workflows work unchanged:**

```python
# This still works exactly as before
config = GPTConfig(n_layer=12, n_head=12, n_embd=768)  # Defaults to LANGUAGE_MODEL
model = GPT(config)
# ... existing training code unchanged
```

The new functionality is completely opt-in through the `mode` parameter.

## Advanced Usage

### Custom Transfer Learning

```python
# Manual control over freezing/unfreezing
model = GPT(config)

# Check frozen status
if model.get_frozen_status():
    print("Transformer is frozen")

# Manual unfreezing
model.unfreeze_transformer_weights()

# Manual freezing  
model.freeze_transformer_weights()
```

### Multi-Stage Training

```python
# Stage 1: Feature extraction (frozen)
config_stage1 = GPTConfig(
    mode=ModelMode.TOKEN_CLASSIFIER,
    freeze_transformer=True,
    # ... other params
)
model = GPT(config_stage1)
# Train with frozen transformer...

# Stage 2: Full fine-tuning (unfrozen)  
model.unfreeze_transformer_weights()
# Continue training with lower LR...
```

### Custom Schedulers

```python
from core.scheduler import AdaptiveScheduler

# LR adapts based on validation performance
scheduler = AdaptiveScheduler(
    initial_lr=1e-4,
    patience=3,      # Wait 3 evals before reducing
    factor=0.5,      # Reduce by half
    min_lr=1e-6,
)

# In training loop
if scheduler.step(val_loss):
    print(f"LR reduced to {scheduler.current_lr}")
```

## Troubleshooting

### Common Issues

1. **Wrong Attention Type:**
   - Classification modes auto-correct to bidirectional
   - Warning message will be printed

2. **Loss Modifier Compatibility:**
   - Incompatible modifiers are automatically filtered
   - Check logs for which modifiers are active

3. **Transfer Learning:**
   - Ensure checkpoint path exists and is accessible
   - Mismatched architectures will show missing/unexpected keys

4. **Memory Usage:**
   - Bidirectional attention uses more memory than causal
   - Consider reducing batch size for classification tasks

### Debugging

```python
# Check model configuration
print(f"Mode: {model.config.mode}")
print(f"Attention: {model.config.attention_type}")  
print(f"Frozen: {model.get_frozen_status()}")

# Check loss modifier pipeline
pipeline = create_loss_modifier_pipeline(config)
print(f"Enabled modifiers: {pipeline.get_enabled_modifier_names()}")
```

## Performance Tips

1. **Start with Feature Extraction:**
   - Faster training, less overfitting
   - Good baseline before full fine-tuning

2. **Use Label Smoothing:**
   - Helpful for classification tasks
   - Reduces overconfidence

3. **Dynamic Unfreezing:**
   - Let head train first, then unfreeze
   - Reduce LR when unfreezing

4. **Appropriate Batch Sizes:**
   - Smaller batches for fine-tuning (16-32)
   - Larger batches for feature extraction (64+)

## Migration Guide

To upgrade existing code:

1. **No changes needed** for existing language modeling
2. **For new modes:** Add mode-specific parameters to config
3. **For transfer learning:** Add checkpoint and freezing parameters  
4. **For enhanced scheduling:** Import new scheduler classes

The implementation maintains 100% backward compatibility with existing training pipelines.