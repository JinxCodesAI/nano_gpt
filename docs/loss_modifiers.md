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

# Enable entropy-based entropy-based weighting (always dynamic)
python train.py --loss_modifiers_enabled=True --entropy_modifier_enabled=True

# Combine multiple modifiers
python train.py --loss_modifiers_enabled=True \
  --target_smoothing_enabled=True --target_smoothing_factor=0.1 \
  --entropy_modifier_enabled=True --entropy_modifier_weight=1.2 \
  --mask_ratio_weight_enabled=True
```

## Available Modifiers

### 1. Entropy Modifier

Purpose: Focus training on positions/samples where the model’s wrong‑answer distribution is more uniform (higher information content), and optionally down‑weight positions with low‑entropy wrong‑answer distributions.

Intuition:
- High entropy (wrong answers are spread out) → model isn’t fixating on a few wrong tokens → better signal‑to‑noise → give more weight.
- Low entropy (wrong answers concentrate on a few tokens) → potentially biased/confused predictions → give less or baseline weight.

Parameters (from config or CLI):
- entropy_modifier_weight (float, default 1.0): Base multiplicative weight applied to the dynamic batch weight.
- entropy_modifier_threshold (float, default 0.0): Minimum entropy floor when computing dynamic weights. Entropies below this are clamped up to the threshold to avoid zero/near‑zero weights.
- entropy_modifier_eps (float, default 1e-8): Numerical stability for logs/divisions.

Configuration example:
```python
loss_modifiers_enabled = True
entropy_modifier_enabled = True
entropy_modifier_weight = 1.0
entropy_modifier_threshold = 0.1
entropy_modifier_eps = 1e-8
```

Mechanics (how it works):
1) Compute softmax over logits to get per‑token probabilities p.
2) Zero out the probability of the correct token; renormalize remaining mass over wrong tokens only to get q (distribution over wrong answers).
3) Per‑position entropy: H = -Σ_i q_i log(q_i). If all mass is on the correct token (no wrong mass), H is treated as 0.
4) If a mask is provided, per‑position entropies are masked accordingly.
5) Always compute per‑sample entropy as the mean (or masked mean) across positions.
6) Clamp per‑sample entropies with entropy_threshold: H* = clamp(H, min=entropy_threshold).
7) Compute batch_weight = mean(H*) * entropy_modifier_weight.
8) Final loss = loss * batch_weight.

Batch-level illustration (sample weights):
Assume batch size = 4, per-sample entropies after per-position averaging are H = [0.2, 0.6, 1.0, 0.4], threshold=0.5, weight=1.0.
- Clamped entropies: H* = [0.5, 0.6, 1.0, 0.5]
- batch_weight = mean(H*) = 0.65
- All samples in this step share the same scalar multiplier 0.65; the modifier does not apply different multipliers per sample in the current implementation.


What is entropy_modifier_threshold?
- entropy_modifier_threshold: A floor applied during dynamic weighting to prevent very small entropies from collapsing the weight. It clamps each per‑sample entropy to at least this value before averaging.

Logged metrics:
- mean_entropy: Mean per‑position wrong‑answer entropy (masked if provided).
- max_entropy, min_entropy, entropy_std: Range and variability of entropies.
- entropy_weight: The final batch multiplier (mean clamped per‑sample entropy × entropy_modifier_weight).

Worked example:
Suppose batch_size=2, seq_len=2, vocab_size=4. Targets = [[2, 1], [0, 3]]. Consider the first token of sample 1 with logits [2.0, 1.0, 4.0, 0.5] and target=2.
- Softmax p ≈ [0.100, 0.037, 0.848, 0.015]. Zero out the correct token (index 2): wrong mass ≈ 0.152.
- Renormalize over wrong tokens: q ≈ [0.100/0.152, 0.037/0.152, 0.015/0.152, (target=2 is 0)] ≈ [0.658, 0.243, 0.099, 0.000].
- Entropy H = -Σ q_i log(q_i) ≈ -(0.658 ln 0.658 + 0.243 ln 0.243 + 0.099 ln 0.099) ≈ 0.90 nats.

If the average per‑sample entropy across its positions is, say, H_s1=0.8 and H_s2=0.4, with entropy_modifier_threshold=0.5 and entropy_modifier_weight=1.0:
- Clamp: H*_s1 = max(0.8, 0.5) = 0.8; H*_s2 = max(0.4, 0.5) = 0.5.
- batch_weight = mean([0.8, 0.5]) = 0.65.
- Final loss = original_loss * 0.65 (lower than baseline because average wrong‑answer entropy is modest; if entropies were higher, the weight would be higher).

Same batch, effect with always-dynamic weighting:
- Given H = [0.8, 0.4] and threshold=0.5:
  - H* = [0.8, 0.5]
  - batch_weight = mean(H*) = (0.8 + 0.5)/2 = 0.65
  - Final loss = original_loss × 0.65

Observation: With always-dynamic weighting, entropy directly influences the single batch scalar applied to the loss every step.

Use cases:
- Emphasize updates when the model’s wrong predictions are diverse (more informative errors).
- Down‑weight updates when the model fixates on specific wrong tokens (possibly noisy/confusing cases).
- Track confidence patterns via entropy metrics during training.

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

Purpose: Adjust loss based on how many valid (non-masked) tokens a sequence contains. Sequences with fewer valid tokens receive higher weights to compensate for lower signal.

Definitions:
- mask: Boolean tensor (batch_size, seq_len) where True indicates a valid position. If not provided, the modifier infers mask from targets using ignore_index (default -100).
- mask_ratio (per sequence): sum(mask_i) / seq_len.

Parameters:
- mask_ratio_weight_power (float, default 0.5): Exponent for inverse weighting; if 0.5, weight = 1/sqrt(mask_ratio).
- mask_ratio_weight_min, mask_ratio_weight_max (floats, defaults 0.1 and 10.0): Clamp bounds for weights.
- mask_ratio_weight_eps (float, default 1e-8): Numerical stability for divisions and zero ratios.

Computation (always sequence-level):
1) If mask is None, infer mask as (targets != ignore_index), where ignore_index defaults to -100 (or can be passed via modifier kwargs).
2) Compute mask_ratio per sequence: r_i = sum(mask_i) / (seq_len + eps)
3) Compute weight per sequence: w_i = clamp((r_i + eps)^(-power), min_weight, max_weight)
4) Compute per-position loss with reduction='none', reshape to [B, T].
5) Mask and average per sequence: L_i = sum(loss_i * mask_i) / (sum(mask_i) + eps)
6) Final loss = mean_i(w_i * L_i)

Examples (sequence-level):
- Assume batch_size=2, seq_len=4, power=0.5, min=0.1, max=10.0, ignore_index=-100
- targets_1 = [5, -100, -100, -100] → mask_1 = [1, 0, 0, 0] → r_1 = 1/4 = 0.25 → w_1 = 1/sqrt(0.25) = 2.0
- targets_2 = [3, 4, 1, 2] → mask_2 = [1, 1, 1, 1] → r_2 = 4/4 = 1.0 → w_2 = 1/sqrt(1.0) = 1.0
- Suppose per-sequence masked losses: L_1 = 2.0, L_2 = 1.0
- Final loss = mean([2.0*2.0, 1.0*1.0]) = mean([4.0, 1.0]) = 2.5

Use cases:
- Balance training with variable sequence lengths or heavy padding.
- Emphasize sequences with fewer valid tokens to avoid under-training them.

Metrics logged:
- mean_mask_ratio, min_mask_ratio, max_mask_ratio
- mean_weight, min_weight, max_weight, weight_std

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