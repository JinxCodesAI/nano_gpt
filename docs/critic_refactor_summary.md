# Critic Implementation Refactor Summary

## Overview
Refactored the critic head implementation to support three distinct modes with simplified configuration and cleaner code architecture.

## Key Changes

### 1. New CriticMode Enum
Added `CriticMode` enum in `model.py`:
- `NONE`: No critic (default behavior)
- `TARGETLESS`: Critic's normalized output directly weights the loss
- `TARGETED`: Critic trained against explicit confidence target (1 - normalized entropy)

### 2. Enhanced Critic Head Architecture
**Old**: Simple linear layer `nn.Linear(n_embd, 1, bias=False)`

**New**: MLP with Sigmoid activation
```python
nn.Sequential(
    nn.Linear(n_embd, n_embd // 2),
    nn.GELU(),
    nn.Linear(n_embd // 2, 1),
    nn.Sigmoid()  # Ensures [0, 1] confidence score
)
```

### 3. Simplified Configuration

**Removed parameters:**
- `add_critic_head` (replaced by `critic_mode`)
- `critic_target_scope` (no longer needed)
- `mask_token_id` (no longer needed)
- `pad_token_id` (no longer needed)

**New/Updated parameters:**
- `critic_mode: CriticMode = CriticMode.NONE`
- `critic_alpha: float = 0.5` (now only used in TARGETED mode)
- `start_critic_iteration: int = 0` (warmup start)
- `end_critic_iteration: int = 0` (warmup end)

### 4. Mode-Specific Behavior

#### NONE Mode
Standard cross-entropy loss without any critic involvement.

#### TARGETLESS Mode
- Critic predicts confidence scores for each token
- Scores are normalized to have mean=1 per sequence
- Normalization factor is detached to prevent gaming
- Loss weights = predicted_confidence / mean_confidence.detach()
- Gradient flows through weights to train the critic

#### TARGETED Mode
- Critic trained with explicit target: `true_confidence = 1.0 - normalized_entropy`
- Critic has separate MSE loss against this target
- Critic predictions (detached) weight the LM loss: `loss_weights = 1.0 + predicted_confidence.detach()`
- Total loss = `final_lm_loss + alpha_eff * loss_critic`
- Supports alpha warmup via `start_critic_iteration` and `end_critic_iteration`

### 5. Code Simplification

**Removed:**
- Dependency on `build_critic_artifacts_from_logits` from `sample_utils`
- Complex artifact building with masked token replacement
- Second forward pass through transformer for critic input
- Scope-based target construction logic

**Result:**
- ~70 lines of complex logic removed from `_forward_language_model`
- Single forward pass for all modes
- Cleaner, more maintainable code

### 6. Updated Files

**Core:**
- `model.py`: Added CriticMode enum, refactored critic head and forward logic
- `train.py`: Updated to use CriticMode instead of add_critic_head
- `core/training_step.py`: No changes needed (already compatible)

**Sampling:**
- `sample.py`: Updated critic checks to use critic_mode
- `sample_simple.py`: Updated critic checks to use critic_mode

**Configuration:**
- `config/train_char_diffusion_critic_head.py`: Updated to use critic_mode='targetless'
- `config/train_char_diffusion_critic_a40.py`: Updated to use critic_mode='targetless'

**Tests:**
- `tests/test_critic_toggle.py`: Completely rewritten to test new modes

### 7. Backward Compatibility

- Loss modifiers pipeline remains fully compatible
- Existing checkpoints can be loaded (critic_head structure is different but can be ignored)
- Old config files need minor updates (replace `add_critic_head=True` with `critic_mode='targetless'`)

### 8. Benefits

1. **Simpler**: Removed ~70 lines of complex logic
2. **Faster**: Single forward pass instead of two
3. **Cleaner**: No dependency on sample_utils in model.py
4. **More flexible**: Easy to add new critic modes
5. **Better architecture**: MLP critic head is more expressive
6. **Clearer semantics**: Explicit modes instead of boolean flags

## Migration Guide

### For Config Files
**Old:**
```python
add_critic_head = True
critic_alpha = 0.5
critic_target_scope = 'masked_and_ignore'
mask_token_id = None
pad_token_id = None
```

**New:**
```python
critic_mode = 'targetless'  # or 'targeted' or 'none'
critic_alpha = 0.5  # Only used in TARGETED mode
```

### For Code
**Old:**
```python
if getattr(model.config, 'add_critic_head', False):
    # ...
```

**New:**
```python
from model import CriticMode
if getattr(model.config, 'critic_mode', CriticMode.NONE) != CriticMode.NONE:
    # ...
```

## Testing

Run the updated test suite:
```bash
pytest tests/test_critic_toggle.py -v
```

Test different modes in training:
```bash
# TARGETLESS mode
python train.py config/train_char_diffusion_critic_head.py

# TARGETED mode (update config first)
# Set critic_mode = 'targeted' in config file
python train.py config/train_char_diffusion_critic_a40.py

# NONE mode (default)
python train.py config/train_char_diffusion.py
```

## Notes

- The new MLP critic head will need retraining (old checkpoints have different architecture)
- TARGETLESS mode is recommended as the default for most use cases
- TARGETED mode may be useful when you want explicit confidence calibration
- Alpha warmup is only active in TARGETED mode

