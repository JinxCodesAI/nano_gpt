# Loss Modifiers System

The loss modifiers system provides a modular, extensible way to apply various loss modifications during training. All modifiers are **opt-in** and maintain full backward compatibility - the default behavior remains unchanged unless explicitly configured.

## Overview

The system consists of three main components:
- **Pipeline**: Orchestrates multiple modifiers in sequence. The pipeline now threads optional per-position loss tensors through modifiers for precise control, and aggregates to a scalar at the end when needed.
- **Base Interface**: Abstract class ensuring consistent behavior. Modifiers may return either a scalar loss or a dict with a replacement per-position loss.
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

Purpose: Focus training on samples where the model's wrong‑answer distribution is more uniform (higher information content), and penalize samples where wrong answer probability concentrates in few tokens. Uses **normalized entropy** that is vocabulary-size independent.

Intuition:
- **High normalized entropy (→1.0)**: wrong answers uniformly distributed → lower penalty (lower weight multiplier)
- **Low normalized entropy (→0.0)**: wrong answers concentrated on few tokens → higher penalty (higher weight multiplier)
- **Vocabulary-size independent**: entropy normalized to [0,1] range for consistent behavior across different models

Parameters (from config or CLI):
- entropy_modifier_weight (float, default 1.0): Base factor in the per-sample multiplier.
- entropy_modifier_threshold (float, default 0.0): Minimum normalized entropy floor when computing per-sample multipliers (range: [0,1]).
- entropy_modifier_eps (float, default 1e-8): Numerical stability for logs/divisions.
- entropy_modifier_verbose (bool, default False): Enable verbose logging of calculation irregularities and safety corrections.

Configuration example:
```python
loss_modifiers_enabled = True

entropy_modifier_enabled = True
entropy_modifier_weight = 0.1
entropy_modifier_threshold = 0.0
entropy_modifier_eps = 1e-8
entropy_modifier_verbose = False  # Set to True for debugging
```

Mechanics (how it works):
1) Compute softmax over logits to get per‑token probabilities p.
2) Zero out the probability of the correct token; renormalize remaining mass over wrong tokens only to get q (distribution over wrong answers).
3) Per‑position raw entropy: H_raw = -Σ_i q_i log(q_i). If all mass is on the correct token (no wrong mass), H_raw is treated as 0.
4) **Normalize entropy**: H_norm = H_raw / log(N_wrong) where N_wrong is the count of wrong tokens with non-negligible probability. This produces vocabulary-size independent values in [0,1].
5) If a mask is provided, per‑position normalized entropies are masked accordingly.
6) Compute per‑sample entropy as the mean (or masked mean) across positions.
7) Compute per-sample weights w_i = entropy_modifier_weight / (clamp(H_i, min=entropy_threshold) + eps).
8) Aggregate per-position loss to per-sample loss and multiply by w_i; final loss is mean over samples.

Per-sample illustration (sample weights):
Assume batch size = 4, per-sample **normalized** entropies after per-position averaging are H = [0.2, 0.6, 1.0, 0.4], threshold=0.5, weight=1.0.
- Clamped entropies: H* = [0.5, 0.6, 1.0, 0.5]
- Per-sample weights w = 1 / (H* + eps) ≈ [2.0, 1.667, 1.0, 2.0]
- Final loss = mean_i(w_i * L_i) where L_i is the per-sample masked mean loss.

**Note**: With normalized entropy, these values are now comparable across different vocabulary sizes.


What is entropy_modifier_threshold?
- entropy_modifier_threshold: A floor applied during dynamic weighting to prevent very small normalized entropies from collapsing the weight. It clamps each per‑sample entropy to at least this value (range [0,1]) before averaging. Now vocabulary-size independent!

Logged metrics:
- mean_entropy: Mean per‑position **normalized** wrong‑answer entropy [0,1] (masked if provided).
- max_entropy, min_entropy, entropy_std: Range and variability of normalized entropies [0,1].
- entropy_weight_mean: Mean per-sample weight multiplier derived from normalized entropy.

**All entropy metrics are now vocabulary-size independent and bounded in [0,1].**

### Verbose Mode Debug Logging

Enable verbose logging with `entropy_modifier_verbose = True` to monitor calculation irregularities:

```python
entropy_modifier_verbose = True  # Enable detailed logging
```

Verbose mode reports:
- **Positions with no wrong answer probability**: When model is extremely confident 
- **Positions with ≤1 wrong token**: Cannot compute meaningful entropy (forced to 0)
- **Entropy values clamped**: Values outside [0,1] range corrected
- **Non-finite values detected**: NaN/inf values replaced with safe defaults
- **Per-sample weight ranges**: Min/max weights applied to loss samples
- **Extreme weights clamped**: Very high/low weights bounded to reasonable range
- **Loss modification ratios**: Before/after loss comparison
- **Non-finite metrics corrected**: Safety fallbacks for logging values

Use verbose mode during development to ensure entropy calculation behaves as expected.

Worked example:
Suppose batch_size=2, seq_len=2, vocab_size=4. Targets = [[2, 1], [0, 3]]. Consider the first token of sample 1 with logits [2.0, 1.0, 4.0, 0.5] and target=2.
- Softmax p ≈ [0.100, 0.037, 0.848, 0.015]. Zero out the correct token (index 2): wrong mass ≈ 0.152.
- Renormalize over wrong tokens: q ≈ [0.100/0.152, 0.037/0.152, 0.015/0.152, (target=2 is 0)] ≈ [0.658, 0.243, 0.099, 0.000].
- Raw entropy H_raw = -Σ q_i log(q_i) ≈ -(0.658 ln 0.658 + 0.243 ln 0.243 + 0.099 ln 0.099) ≈ 0.90 nats.
- **Normalized entropy**: 3 wrong tokens have probability > eps, so max_entropy = ln(3) ≈ 1.099. H_norm = 0.90 / 1.099 ≈ **0.82**.

If the average per‑sample **normalized** entropy across positions is H_s1=0.7 and H_s2=0.3, with entropy_modifier_threshold=0.5 and entropy_modifier_weight=1.0:
- Clamp: H*_s1 = max(0.7, 0.5) = 0.7; H*_s2 = max(0.3, 0.5) = 0.5.
- Per-sample weights: w = [1/0.7, 1/0.5] ≈ [1.43, 2.0]
- Final loss = mean([1.43 * L_1, 2.0 * L_2]) where L_i are per-sample losses.

**Key improvement**: The normalized entropy value 0.82 is now meaningful regardless of vocabulary size!

Same batch with normalized entropy, old behavior comparison:
- **Before normalization** (vocabulary-dependent): Raw entropy values varied by vocab size
- **After normalization** (vocabulary-independent): H = [0.7, 0.3] are comparable across all vocabulary sizes
- Per-sample weights w = [1/0.7, 1/0.3] ≈ [1.43, 3.33] with threshold=0.0
- **Consistent behavior**: Same entropy threshold and weight settings work across different models!

Use cases:
- Emphasize updates when the model’s wrong predictions are diverse (more informative errors).
- Down‑weight updates when the model fixates on specific wrong tokens (possibly noisy/confusing cases).
- Track confidence patterns via entropy metrics during training.

### 2. Target Smoothing Modifier

**Purpose**: Applies label smoothing to target tokens to reduce overfitting.

**Theory**: Instead of hard targets (1 for correct, 0 for others), uses soft targets that assign some probability to incorrect tokens. This prevents the model from becoming overconfident.

**Mechanics**: Computes smoothed cross-entropy per position as the expectation of negative log-probability under the smoothed target distribution q: L = -Σ_y q_y log p_y. This is sometimes written as a KL-like form, but we do not subtract H(q) because it is constant wrt model parameters. The modifier now returns a per-position loss tensor to the pipeline, which performs final masking and aggregation.

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
1. **Target Smoothing** (per-position loss producer): Replaces the per-position loss with smoothed cross-entropy.
2. **Mask Ratio Weight** (weighter/aggregator): Uses the provided per-position loss and mask to compute weighted per-sequence losses and averages to a scalar.
3. **Entropy Modifier** (batch weight): Multiplies the resulting scalar loss by a batch-level weight derived from wrong-answer entropy.

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