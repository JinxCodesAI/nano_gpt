# Batch-Level Mode Switching

## Overview

The batch-level mode switching feature enables a single model to process batches in different modes (LANGUAGE_MODEL or SEQUENCE_SCORER) within the same training run. Each batch can specify its desired model mode via metadata, and the training loop automatically switches the model to the appropriate mode before processing.

## Key Features

1. **Runtime Mode Switching**: Model switches between LANGUAGE_MODEL and SEQUENCE_SCORER modes based on batch metadata
2. **Single Model, Multiple Tasks**: Train one model on multiple tasks simultaneously
3. **Flexible Data Mixing**: Control the distribution and alternation of different batch types
4. **Backward Compatible**: Batches without mode metadata default to LANGUAGE_MODEL

## Architecture

### Model Side

The model now supports both modes simultaneously:
- Both `lm_head` and `sequence_head` are created at initialization
- `set_mode(ModelMode)` switches the active mode
- `get_mode()` returns the current mode
- Default mode: `LANGUAGE_MODEL`

### Data Side

Providers can specify mode per batch:
- `sample_batch()` returns a dict with optional `'model_mode'` key
- Valid values: `'language_model'` or `'sequence_scorer'`
- DataProviderBase collects `model_mode` from batches into file metadata
- DatasetConsumer exposes it as `'_model_mode'` in batch dict

### Training Side

TrainingStep handles mode switching:
- Checks for `'_model_mode'` in batch dict
- Calls `model.set_mode()` before forward pass
- If no mode specified, keeps current mode (defaults to LANGUAGE_MODEL)

## Usage

### Creating a Dual-Mode Provider

```python
from data.common.provider_base import DataProviderBase

class MyDualModeProvider(DataProviderBase):
    def sample_batch(self, split, rng):
        # Determine mode for this batch
        if self._should_use_language_model(split, rng):
            # Generate LANGUAGE_MODEL batch
            batch = {
                'x': ...,  # input tokens
                'y': ...,  # target tokens
                'model_mode': 'language_model',
            }
        else:
            # Generate SEQUENCE_SCORER batch
            batch = {
                'input_ids': ...,  # input with CLS token
                'targets': ...,    # scalar targets [0, 1]
                'model_mode': 'sequence_scorer',
            }
        
        return batch
```

### Using the DualModeProvider

The included `DualModeProvider` combines char_diffusion and sequence_scorer:

```python
# config/train_dual_mode.py
dataset = 'dual_mode'

# Mode distribution (50/50 by default)
mode_distribution = {
    'language_model': 0.5,
    'sequence_scorer': 0.5,
}

# Alternation frequency (1 = every batch, 2 = every 2 batches, etc.)
alternation_frequency = 1

# Other config...
```

Run training:
```bash
# Start data provider
python prepare.py config/train_dual_mode.py

# Start training (in another terminal)
python train.py config/train_dual_mode.py
```

### Training Flow

```
1. DataProvider generates batch with model_mode metadata
   ↓
2. Batch saved to queue with metadata
   ↓
3. DatasetConsumer loads batch, exposes _model_mode
   ↓
4. TrainingStep extracts _model_mode from batch
   ↓
5. TrainingStep calls model.set_mode(mode)
   ↓
6. Model forward pass uses appropriate head
   ↓
7. Loss computed and backpropagated
   ↓
8. Repeat for next batch (potentially different mode)
```

## DualModeProvider Details

### Architecture

The `DualModeProvider` (in `data/dual_mode/prepare_streaming.py`) combines two existing providers:
- **CharDiffusionProvider**: Generates LANGUAGE_MODEL batches (masked language modeling)
- **SequenceScorerProvider**: Generates SEQUENCE_SCORER batches (sequence quality scoring)

### Configuration

```python
# Mode distribution
mode_distribution = {
    'language_model': 0.5,   # 50% of batches
    'sequence_scorer': 0.5,  # 50% of batches
}

# Alternation frequency
alternation_frequency = 1  # Switch mode every N batches
```

### Mode Determination

```python
def _determine_mode_for_batch(self, batch_idx, rng):
    # Determine alternation window
    window_idx = batch_idx // self.alternation_frequency
    
    # Use window-based RNG for deterministic distribution
    window_rng = torch.Generator()
    window_rng.manual_seed(rng.initial_seed() + window_idx)
    
    # Sample mode based on distribution
    rand_val = torch.rand(1, generator=window_rng).item()
    lm_ratio = self.mode_distribution.get('language_model', 0.5)
    
    return 'language_model' if rand_val < lm_ratio else 'sequence_scorer'
```

### Batch Format

**LANGUAGE_MODEL batches:**
```python
{
    'x': torch.Tensor,        # shape: [batch_size, block_size], dtype: int64
    'y': torch.Tensor,        # shape: [batch_size, block_size], dtype: int64
    'model_mode': 'language_model',
}
```

**SEQUENCE_SCORER batches:**
```python
{
    'input_ids': torch.Tensor,  # shape: [batch_size, block_size], dtype: int64
    'targets': torch.Tensor,    # shape: [batch_size], dtype: float32
    'model_mode': 'sequence_scorer',
}
```

## Benefits

### 1. Shared Representations
- Single model learns from both tasks
- Shared transformer layers capture general patterns
- Task-specific heads specialize for each mode

### 2. Efficient Training
- No need to train separate models
- Shared computation for feature extraction
- Better GPU utilization

### 3. Flexible Task Mixing
- Control distribution of each task
- Adjust alternation frequency
- Easy to add more tasks

### 4. Improved Generalization
- Multi-task learning can improve generalization
- Tasks can complement each other
- Reduces overfitting on single task

## Implementation Details

### DataProviderBase Changes

```python
# In produce_one_file()
batch_metadata = {}
for k in batches[0].keys():
    if k not in tensor_keys:
        # Collect non-tensor metadata (including model_mode)
        values = [batch[k] for batch in batches if k in batch]
        batch_metadata[k] = values

metadata.update(batch_metadata)  # Includes model_mode list
```

### DatasetConsumer Changes

```python
# In get_batch()
if metadata and 'model_mode' in metadata:
    model_modes = metadata['model_mode']
    if isinstance(model_modes, list) and len(model_modes) > start_idx:
        batch_tensors['_model_mode'] = model_modes[start_idx]
```

### TrainingStep Changes

```python
# In execute_step()
if '_model_mode' in batch:
    mode_str = batch['_model_mode']
    if mode_str == 'language_model':
        raw_model.set_mode(ModelMode.LANGUAGE_MODEL)
    elif mode_str == 'sequence_scorer':
        raw_model.set_mode(ModelMode.SEQUENCE_SCORER)
```

## Testing

Run the test suite:
```bash
python test_batch_mode_switching.py
```

Tests cover:
1. Batch metadata handling
2. Mode switching in training step
3. Forward pass with mode switching
4. Dual-mode provider concept
5. Alternation frequency control

## Migration Guide

### For Existing Single-Mode Providers

No changes needed! Providers that don't specify `model_mode` continue to work:
- Batches default to LANGUAGE_MODEL mode
- Existing training configs work unchanged

### For New Dual-Mode Providers

1. Decide on mode distribution and alternation
2. Implement `sample_batch()` to return batches with `model_mode` key
3. Ensure vocab consistency across modes
4. Test with both modes

### For Training Configs

No changes needed! The model:
- Defaults to LANGUAGE_MODEL mode
- Switches based on batch metadata automatically
- Works with both single-mode and dual-mode providers

## Best Practices

1. **Vocab Consistency**: Ensure all sub-providers use the same vocabulary
2. **Special Tokens**: Coordinate special token IDs (CLS, MASK, PAD, etc.)
3. **Batch Size**: Use consistent batch_size across modes
4. **Block Size**: Use consistent block_size across modes
5. **Alternation**: Start with frequent alternation (freq=1), adjust based on results
6. **Distribution**: Start with 50/50, adjust based on task importance
7. **Monitoring**: Log mode distribution during training to verify balance

## Troubleshooting

### Issue: Model stuck in one mode
**Solution**: Check that batches include `model_mode` metadata. Verify DatasetConsumer exposes `_model_mode`.

### Issue: Vocab size mismatch
**Solution**: Ensure all sub-providers use the same vocab. Check `vocab_size` in meta.pkl.

### Issue: Loss spikes when switching modes
**Solution**: This is normal initially. Consider:
- Adjusting learning rate
- Using separate optimizers per mode (advanced)
- Increasing warmup iterations

### Issue: One mode dominates
**Solution**: Adjust `mode_distribution` to balance tasks. Monitor per-mode losses.

## Future Enhancements

Potential improvements:
1. **Per-mode learning rates**: Different LR for each mode
2. **Dynamic distribution**: Adjust mode distribution based on validation performance
3. **More modes**: Add support for additional task types
4. **Mode-specific loss weights**: Weight losses differently per mode
5. **Curriculum learning**: Start with one mode, gradually introduce others

## References

- `model.py`: Dual-mode model architecture
- `data/common/provider_base.py`: Base provider with metadata collection
- `dataset_consumer.py`: Consumer with metadata exposure
- `core/training_step.py`: Training step with mode switching
- `data/dual_mode/prepare_streaming.py`: Example dual-mode provider
- `docs/datasets_guide.md`: Dataset creation guide

