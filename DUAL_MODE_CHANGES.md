# Dual-Mode Architecture Changes

## Summary

This document describes the changes made to remove TOKEN_CLASSIFIER support and implement a dual-mode architecture where each model supports both LANGUAGE_MODEL and SEQUENCE_SCORER modes simultaneously, with runtime mode switching.

## Key Changes

### 1. ModelMode Enum (model.py)
- **Removed**: `TOKEN_CLASSIFIER` mode
- **Kept**: `LANGUAGE_MODEL` and `SEQUENCE_SCORER` modes only

### 2. GPTConfig (model.py)
**Removed fields:**
- `mode: ModelMode` - No longer config-level, now runtime switchable
- `num_token_classes: int` - TOKEN_CLASSIFIER specific
- `binary_classification: bool` - Legacy backward compatibility
- `__post_init__()` method - No longer needed for mode validation

**Kept fields:**
- `cls_token_id: int` - Still used for SEQUENCE_SCORER mode
- All other fields remain unchanged

### 3. GPT Model Architecture (model.py)

**Dual-Head Architecture:**
- Model now creates **both** heads simultaneously:
  - `lm_head`: Linear(n_embd, vocab_size) for LANGUAGE_MODEL mode
  - `sequence_head`: ScaledSigmoidHead(n_embd) for SEQUENCE_SCORER mode
- Weight tying still applies to `lm_head` and token embeddings
- CLS embedding is created if `cls_token_id` is specified (used in SEQUENCE_SCORER mode)

**Runtime Mode Switching:**
- Added `_current_mode` instance variable (default: LANGUAGE_MODEL)
- Added `set_mode(ModelMode)` method to switch modes at runtime
- Added `get_mode()` method to query current mode

**Example:**
```python
model = GPT(config)
# Default mode is LANGUAGE_MODEL
logits, loss = model(x, y)  # Uses lm_head

# Switch to SEQUENCE_SCORER
model.set_mode(ModelMode.SEQUENCE_SCORER)
scores, loss = model(x, y)  # Uses sequence_head

# Switch back
model.set_mode(ModelMode.LANGUAGE_MODEL)
```

### 4. Forward Pass (model.py)

**Removed:**
- `_forward_token_classifier()` method
- `_compute_weighted_classification_loss()` method

**Updated:**
- `forward()` now routes based on `self._current_mode` instead of `self.config.mode`
- `_forward_language_model()` and `_forward_sequence_scorer()` pass `self._current_mode` to loss modifiers

### 5. Critic Head (model.py)

**Updated:**
- Critic head is now created independently of mode (if `add_critic_head=True`)
- `critic_scores()` method no longer checks for LANGUAGE_MODEL mode
- Critic functionality works in both modes

### 6. Evaluator (core/evaluator.py)

**Updated:**
- Uses `raw_model.get_mode()` instead of `raw_model.config.mode`
- Mode detection is now runtime-based, not config-based

### 7. Loss Modifiers

**Updated support for all modifiers:**
- `EntropyModifier`: Now only supports LANGUAGE_MODEL
- `TargetSmoothingModifier`: Now only supports LANGUAGE_MODEL
- `MaskRatioWeightModifier`: Now only supports LANGUAGE_MODEL
- `SequenceScorerVarianceModifier`: Still SEQUENCE_SCORER only
- `SequenceScorerCorrelationModifier`: Still SEQUENCE_SCORER only
- `MetricsCollectorModifier`: Still supports all modes
- `SequenceJudgeWeightModifier`: Still LANGUAGE_MODEL only

**Removed:**
- All TOKEN_CLASSIFIER support from loss modifiers

### 8. Interface Changes

**Minimal interface changes:**
- No changes to `forward()` signature - still `forward(idx, targets=None, attention_mask=None, loss_modifiers=None)`
- Mode switching is explicit via `set_mode()` method
- Backward compatible: existing code continues to work in LANGUAGE_MODEL mode by default

## Migration Guide

### For Existing Code

**No changes needed** if you're using LANGUAGE_MODEL or SEQUENCE_SCORER modes. The model defaults to LANGUAGE_MODEL mode.

### For TOKEN_CLASSIFIER Users

TOKEN_CLASSIFIER mode has been removed. If you were using it:

1. **Option 1**: Use LANGUAGE_MODEL mode with bidirectional attention
   ```python
   config = GPTConfig(
       attention_type='bidirectional',
       # ... other settings
   )
   model = GPT(config)
   # Model defaults to LANGUAGE_MODEL mode
   ```

2. **Option 2**: Implement custom classification head on top of the transformer
   ```python
   model = GPT(config)
   # Add your own classification head
   classifier = nn.Linear(config.n_embd, num_classes)
   ```

### For Batch-Level Mode Switching

To switch modes per batch during training:

```python
# In your training loop
for batch in dataloader:
    # Determine mode from batch metadata
    if batch['mode'] == 'sequence_scorer':
        model.set_mode(ModelMode.SEQUENCE_SCORER)
    else:
        model.set_mode(ModelMode.LANGUAGE_MODEL)
    
    logits, loss = model(batch['x'], batch['y'])
    # ... rest of training step
```

## Files Modified

1. `model.py` - Core model changes
2. `core/evaluator.py` - Mode detection updates
3. `loss_modifiers/entropy_modifier.py` - Remove TOKEN_CLASSIFIER support
4. `loss_modifiers/target_smoothing_modifier.py` - Remove TOKEN_CLASSIFIER support
5. `loss_modifiers/mask_ratio_weight_modifier.py` - Remove TOKEN_CLASSIFIER support

## Files Not Modified (Backward Compatible)

- `core/training_step.py` - No changes needed
- `core/trainer.py` - No changes needed
- `dataset_consumer.py` - No changes needed
- `train.py` - No changes needed

## Testing

A test script `test_dual_mode.py` has been added to verify:
1. Default mode is LANGUAGE_MODEL
2. Model has both heads
3. Forward pass works in LANGUAGE_MODEL mode
4. Mode switching to SEQUENCE_SCORER works
5. Forward pass works in SEQUENCE_SCORER mode
6. Mode switching back to LANGUAGE_MODEL works
7. Inference mode (no targets) works in both modes
8. Invalid mode raises TypeError

## Benefits

1. **Simplified Architecture**: Only two modes instead of three
2. **Runtime Flexibility**: Can switch modes without recreating model
3. **Batch-Level Control**: Can process different batch types in same training run
4. **Reduced Complexity**: Removed conditional head creation logic
5. **Future-Proof**: Easy to add new modes or heads as needed

## Breaking Changes

1. **TOKEN_CLASSIFIER mode removed**: Code using this mode must be updated
2. **Config fields removed**: `mode`, `num_token_classes`, `binary_classification`
3. **Loss modifier compatibility**: TOKEN_CLASSIFIER support removed from all modifiers

## Obsolete Files

The following files reference TOKEN_CLASSIFIER and may need updates:
- `config/token_classifier_config.py` - Example config, now obsolete
- `examples/token_classification_example.py` - Example code, needs update
- `docs/MULTIMODE_USAGE.md` - Documentation, needs update
- `docs/loss_modifiers.md` - Documentation, needs update

