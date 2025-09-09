# Loss Modifiers System

The loss modifiers system provides a modular, extensible way to apply various loss modifications during training. All modifiers are **opt-in** and maintain full backward compatibility - the default behavior remains unchanged unless explicitly configured.

## Overview

The system consists of three main components:
- **Pipeline**: Orchestrates multiple modifiers in sequence
- **Base Interface**: Abstract class ensuring consistent behavior
- **Individual Modifiers**: Specific loss modification algorithms

## Quick Start

### Default Behavior (No Changes)
```bash
python train.py  # All modifiers disabled - identical to original behavior
```

### Enable Basic Modifiers
```bash
# Enable label smoothing
python train.py --loss_modifiers_enabled=True --target_smoothing_enabled=True --target_smoothing_factor=0.1

# Enable entropy-based weighting  
python train.py --loss_modifiers_enabled=True --entropy_modifier_enabled=True --entropy_modifier_use_for_weighting=True

# Combine multiple modifiers
python train.py --loss_modifiers_enabled=True \
  --target_smoothing_enabled=True --target_smoothing_factor=0.1 \
  --entropy_modifier_enabled=True --entropy_modifier_weight=1.2 \
  --mask_ratio_weight_enabled=True
```

## Available Modifiers

### 1. Entropy Modifier

**Purpose**: Weights loss based on the entropy of wrong answer distributions per position.

**Theory**: 
- **High entropy** (uniform wrong answers) = good signal-to-noise ratio
- **Low entropy** (concentrated wrong answers) = poor signal-to-noise ratio

**Configuration**:
```python
loss_modifiers_enabled = True
entropy_modifier_enabled = True
entropy_modifier_weight = 1.0                    # Base weight factor
entropy_modifier_use_for_weighting = False       # Use entropy for dynamic weighting
entropy_modifier_threshold = 0.0                 # Minimum entropy threshold
entropy_modifier_eps = 1e-8                      # Prevent log(0) errors
```

**Use Cases**:
- Focus training on positions with better signal quality
- Identify and down-weight positions with poor discrimination
- Analysis of model confidence patterns

**Metrics Logged**:
- `mean_entropy`: Average entropy across positions
- `max_entropy`, `min_entropy`: Entropy range
- `entropy_std`: Entropy standard deviation
- `entropy_weight`: Applied weight (if using for weighting)

### 2. Target Smoothing Modifier

**Purpose**: Applies label smoothing to target tokens to reduce overfitting.

**Theory**: Instead of hard targets (1 for correct, 0 for others), uses soft targets that assign some probability to incorrect tokens. This prevents the model from becoming overconfident.

**Configuration**:
```python
loss_modifiers_enabled = True
target_smoothing_enabled = True
target_smoothing_factor = 0.1                    # Smoothing strength (0.0 = no smoothing)
target_smoothing_special_tokens = []             # Token IDs to exclude from smoothing
target_smoothing_exclude_padding = True          # Exclude padding from loss
target_smoothing_padding_token = -100            # Padding token ID
```

**Use Cases**:
- Reduce overfitting on training data
- Improve model generalization
- Better calibrated prediction confidence

**Metrics Logged**:
- `original_loss`: Loss before smoothing
- `smoothed_loss`: Loss after smoothing
- `smoothing_factor`: Applied smoothing factor
- `valid_positions`: Number of valid (non-padding) positions

### 3. Mask Ratio Weight Modifier

**Purpose**: Weights loss inversely proportional to the square root of the mask ratio.

**Theory**: Sequences with fewer valid tokens should receive higher weights to compensate for having less signal in the loss calculation.

**Configuration**:
```python
loss_modifiers_enabled = True
mask_ratio_weight_enabled = True
mask_ratio_weight_power = 0.5                    # Power for inverse weighting
mask_ratio_weight_min = 0.1                      # Minimum weight (prevents extremes)
mask_ratio_weight_max = 10.0                     # Maximum weight (prevents extremes)
mask_ratio_weight_eps = 1e-8                     # Prevent division by zero
mask_ratio_weight_sequence_level = True          # Apply per-sequence vs per-batch
```

**Formula**: `weight = 1 / (mask_ratio + eps)^power`

**Use Cases**:
- Balance training with variable-length sequences
- Handle datasets with significant padding
- Compensate for uneven token distributions

**Metrics Logged**:
- `mean_mask_ratio`: Average mask ratio across batch
- `min_mask_ratio`, `max_mask_ratio`: Mask ratio range
- `mean_weight`: Average applied weight
- `weight_std`: Weight standard deviation

## Configuration Files

### Via Command Line
```bash
python train.py \
  --loss_modifiers_enabled=True \
  --entropy_modifier_enabled=True \
  --entropy_modifier_weight=1.5 \
  --target_smoothing_enabled=True \
  --target_smoothing_factor=0.15
```

### Via Config File
Create a config file (e.g., `config/loss_modifiers.py`):
```python
# Loss modifier configuration
loss_modifiers_enabled = True

# Entropy modifier
entropy_modifier_enabled = True
entropy_modifier_weight = 1.2
entropy_modifier_use_for_weighting = True
entropy_modifier_threshold = 0.1

# Target smoothing
target_smoothing_enabled = True
target_smoothing_factor = 0.1
target_smoothing_special_tokens = [0, 1, 2]  # Exclude special tokens

# Mask ratio weighting
mask_ratio_weight_enabled = False  # Disabled for this experiment
```

Then run: `python train.py config/loss_modifiers.py`

## Monitoring and Debugging

### WandB Integration
All modifier metrics are automatically logged to WandB under the `loss_modifiers/` prefix:
```
loss_modifiers/EntropyModifier.mean_entropy
loss_modifiers/TargetSmoothingModifier.smoothed_loss
loss_modifiers/MaskRatioWeightModifier.mean_weight
```

### Console Output
The system prints enabled modifiers at startup:
```
Enabled loss modifiers: EntropyModifier, TargetSmoothingModifier
```

## Advanced Usage

### Custom Modifier Development

To create a new modifier, extend `BaseLossModifier`:

```python
from loss_modifiers.base import BaseLossModifier
import torch

class CustomModifier(BaseLossModifier):
    def __init__(self, config):
        super().__init__(config)
        self.custom_param = config.get('custom_param', 1.0)
    
    def modify_loss(self, logits, targets, loss, **kwargs):
        if not self.enabled:
            return loss
        
        # Your custom loss modification logic here
        modified_loss = loss * self.custom_param
        
        # Store metrics for monitoring
        self._metrics = {
            'custom_metric': modified_loss.item(),
        }
        
        return modified_loss
    
    def get_metrics(self):
        return self._metrics.copy()
```

### Modifier Combinations

Modifiers are applied in sequence, so order matters:
1. **Target Smoothing**: Changes the loss function itself
2. **Entropy Modifier**: Weights based on prediction quality
3. **Mask Ratio Weight**: Adjusts for sequence-level characteristics

## Best Practices

### Starting Configuration
For most use cases, start with:
```python
loss_modifiers_enabled = True
target_smoothing_enabled = True
target_smoothing_factor = 0.1  # Conservative smoothing
```

### Experimentation Guidelines
1. **Enable one modifier at a time** initially to understand individual effects
2. **Monitor metrics closely** - especially original vs modified loss values
3. **Start with conservative parameters** and gradually increase
4. **Use validation loss** as the primary metric for hyperparameter tuning

### Performance Considerations
- **Zero overhead when disabled** (default state)
- **Minimal overhead when enabled** - optimized for efficiency
- **Memory usage** is negligible for most modifiers

## Troubleshooting

### Common Issues

**No effect observed**:
- Ensure `loss_modifiers_enabled = True`
- Check individual modifier `enabled` flags
- Verify parameters are within reasonable ranges

**Unstable training**:
- Reduce modifier strength (e.g., lower `target_smoothing_factor`)
- Check for extreme weights in mask ratio modifier
- Monitor gradient norms for unusual patterns

**Performance degradation**:
- Disable modifiers one by one to isolate the cause
- Check for excessive memory usage in custom modifiers
- Profile modifier execution time

### Debug Mode
Add debug prints by enabling verbose logging in individual modifiers:
```python
# In your config
entropy_modifier_debug = True  # Custom parameter for debugging
```

## Examples

### Language Model Fine-tuning
```bash
python train.py \
  --dataset=custom_text \
  --loss_modifiers_enabled=True \
  --target_smoothing_enabled=True \
  --target_smoothing_factor=0.1
```

### High-Quality Signal Training
```bash
python train.py \
  --loss_modifiers_enabled=True \
  --entropy_modifier_enabled=True \
  --entropy_modifier_use_for_weighting=True \
  --entropy_modifier_threshold=0.2
```

### Imbalanced Sequence Lengths
```bash
python train.py \
  --loss_modifiers_enabled=True \
  --mask_ratio_weight_enabled=True \
  --mask_ratio_weight_power=0.5 \
  --mask_ratio_weight_max=5.0
```

## References

- [Label Smoothing Paper](https://arxiv.org/abs/1512.00567)
- [Entropy in Machine Learning](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Loss Function Design Principles](https://distill.pub/2017/momentum/)