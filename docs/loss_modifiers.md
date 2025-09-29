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
### Supported Modes (ModelMode)

- Single source of truth: ModelMode is defined in model.py. All modifiers refer to this definition.
- Each modifier declares a supports_mode(ModelMode) method; the pipeline automatically filters by the current model mode.

Supported modes per modifier:
- EntropyModifier: LANGUAGE_MODEL, TOKEN_CLASSIFIER
- TargetSmoothingModifier: LANGUAGE_MODEL, TOKEN_CLASSIFIER
- MaskRatioWeightModifier: LANGUAGE_MODEL, TOKEN_CLASSIFIER
- SequenceScorerVarianceModifier: SEQUENCE_SCORER only
- SequenceScorerCorrelationModifier: SEQUENCE_SCORER only
- MetricsCollectorModifier: all modes (metrics-only; does not change loss)

Note: If a modifier does not support the active mode, it is skipped with zero overhead.


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
8) **Normalize weights to preserve batch mean**: Scale weights so sum(w_i) = batch_size.
9) Aggregate per-position loss to per-sample loss and multiply by w_i; final loss is mean over samples.

Per-sample illustration (sample weights):
Assume batch size = 4, per-sample **normalized** entropies after per-position averaging are H = [0.2, 0.6, 1.0, 0.4], threshold=0.5, weight=1.0.
- Clamped entropies: H* = [0.5, 0.6, 1.0, 0.5]
- Raw weights w_raw = 1 / (H* + eps) ≈ [2.0, 1.667, 1.0, 2.0]
- Sum of raw weights: 6.667
- **Normalized weights**: w = w_raw × (4 / 6.667) ≈ [1.2, 1.0, 0.6, 1.2] (sum = 4.0)
- Final loss = mean_i(w_i * L_i) where L_i is the per-sample masked mean loss.

**Note**: Weight normalization preserves the original batch mean while applying entropy-based reweighting.


What is entropy_modifier_threshold?
- entropy_modifier_threshold: A floor applied during dynamic weighting to prevent very small normalized entropies from collapsing the weight. It clamps each per‑sample entropy to at least this value (range [0,1]) before averaging. Now vocabulary-size independent!

Logged metrics:

**Per-Position Entropy** (all sequence positions):
- per_position_mean_entropy: Mean normalized entropy across all positions [0,1]
- per_position_max_entropy, per_position_min_entropy: Range of position entropies [0,1]
- per_position_entropy_std: Standard deviation of position entropies

**Per-Sample Entropy** (averaged per sequence):
- sample_mean_entropy: Mean of per-sample entropies [0,1]
- sample_max_entropy, sample_min_entropy: Range of sample-averaged entropies [0,1]
- sample_entropy_std: Standard deviation of per-sample entropies

**Weight Statistics** (after normalization):
- weight_mean: Mean per-sample weight (should ≈ 1.0 due to normalization)
- weight_max, weight_min: Range of normalized per-sample weights
- weight_std: Standard deviation of per-sample weights

**Loss Modification**:
- original_loss: Input loss value before modification
- final_loss: Output loss value after entropy weighting
- loss_ratio: final_loss / original_loss (should ≈ 1.0 due to mean preservation)
- loss_fallback: Boolean, true if fallback to original loss occurred

**Batch Info**:
- batch_size: Number of samples in batch
- weight_sum: Sum of normalized weights (should ≈ batch_size)

**All entropy metrics are vocabulary-size independent and bounded in [0,1].**

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

### 4. Sequence Scorer Variance Modifier

Purpose: For SEQUENCE_SCORER mode, scale the scalar loss by a factor derived from the ratio of batch variances var(targets) / var(predictions). This emphasizes updates when targets are more diverse than current predictions and helps counteract output cramping.

Parameters (from config or CLI):
- sequence_variance_enabled (bool): Enable this modifier
- sequence_variance_scale (float): Upper cap for the multiplicative factor (>1.0). Example: 2.0
- sequence_variance_alpha (float): Growth rate for the nonlinear curve. Example: 1.5
- sequence_variance_eps (float): Numerical stability epsilon when dividing by small variance. Default 1e-8

Configuration example:
```python
loss_modifiers_enabled = True
sequence_variance_enabled = True
sequence_variance_scale = 2.0
sequence_variance_alpha = 1.5
sequence_variance_eps = 1e-8
```

Mechanics:
1) In SEQUENCE_SCORER mode, model returns logits shaped (B,) with values in [0,1].
2) Compute r = var(targets) / max(var(predictions), eps) across the batch (unbiased=False).
3) Compute factor via an aggressive but saturating nonlinearity:
   factor = 1 + (scale - 1) * (1 - exp(-alpha * max(r - 1, 0)))
   This equals 1 when r <= 1, increases rapidly for r > 1, and saturates at 'scale'.
4) Scale the scalar base loss: loss <- loss * factor

Metrics Logged:
- variance_pred, variance_target
- ratio_y_over_x (r), factor_applied
- alpha, scale_cap
- original_loss, final_loss, loss_ratio


### 5. Sequence Scorer Correlation Modifier

Purpose: For SEQUENCE_SCORER mode, scale the scalar loss by a non-linear mapping of the Pearson correlation between predictions and targets. Penalizes anti-correlation strongest, neutral moderately, perfect positive correlation not penalized.

Mapping (c ∈ [-1, 1]):
- c =  1 → factor = 1.0 (no change)
- c =  0 → factor = √alpha
- c = -1 → factor = alpha

Parameters:
- sequence_correlation_enabled (bool)
- sequence_correlation_alpha (float ≥ 1.0): Maximum factor at c = -1
- sequence_correlation_eps (float): Numerical stability for std computations

Configuration example:
```python
loss_modifiers_enabled = True
sequence_correlation_enabled = True
sequence_correlation_alpha = 4.0
sequence_correlation_eps = 1e-8
```

Mechanics:
1) Compute Pearson correlation over batch between preds and targets (detach preds for correlation).
2) Compute factor via quadratic A c^2 + B c + C that passes (1→1), (0→√alpha), (-1→alpha).
3) Scale base loss: loss <- loss * factor.

Metrics Logged:
- correlation (c), multiplier (factor)
- alpha, A, B, C
- original_loss, final_loss, loss_ratio

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

**Purpose**: Balance training when dealing with sequences that have varying amounts of valid (non-masked) tokens. Applies higher weights to sequences with fewer valid tokens to compensate for reduced signal in the loss calculation.

**Intuition**: Sequences with fewer valid tokens (higher masking) should receive proportionally higher weights during training to ensure they contribute meaningfully to parameter updates. The modifier uses inverse power weighting: `weight = 1 / (mask_ratio^power)`.

**Parameters** (from config or CLI):
- `mask_ratio_weight_power` (float, default 0.5): Power for inverse weighting. Common values:
  - 0.5: Square root weighting `1/√(mask_ratio)` - moderate emphasis on sparse sequences
  - 1.0: Linear inverse weighting `1/mask_ratio` - strong emphasis on sparse sequences
  - 0.25: Fourth root weighting - gentle emphasis on sparse sequences
- `mask_ratio_weight_min_weight` (float, default 0.1): Minimum weight clamp to prevent extreme deweighting
- `mask_ratio_weight_max_weight` (float, default 10.0): Maximum weight clamp to prevent extreme upweighting
- `mask_ratio_weight_eps` (float, default 1e-8): Numerical stability epsilon for divisions

**Configuration**:
```python
loss_modifiers_enabled = True

mask_ratio_weight_enabled = True
mask_ratio_weight_power = 0.5        # Square root inverse weighting
mask_ratio_weight_min_weight = 0.1   # Prevent extreme deweighting
mask_ratio_weight_max_weight = 10.0  # Prevent extreme upweighting
mask_ratio_weight_eps = 1e-8         # Numerical stability
```

**Mechanics** (how it works):
1) **Mask inference**: If no mask provided, infer from targets: `mask = (targets != ignore_index)` where `ignore_index` defaults to -100
2) **Ratio calculation**: Per-sequence mask ratio: `r_i = sum(mask_i) / seq_len`
3) **Weight computation**: Per-sequence weight: `w_i = clamp(1 / (r_i + eps)^power, min_weight, max_weight)`
4) **Loss aggregation**: Compute per-position CE loss, mask invalid positions, average per sequence: `L_i = sum(loss_i * mask_i) / (sum(mask_i) + eps)`
5) **Final weighting**: Apply sequence weights: `final_loss = mean_i(w_i * L_i)`

**Worked Example**:
Batch with 3 sequences, seq_len=4, power=0.5, min_weight=0.1, max_weight=10.0:

```python
# Sample sequences with different masking patterns
targets = [
    [5, 10, -100, -100],    # 50% valid tokens
    [3, 4, 1, 2],           # 100% valid tokens
    [7, -100, -100, -100]   # 25% valid tokens
]

# Mask ratios
mask_ratios = [0.5, 1.0, 0.25]

# Compute weights: w_i = 1 / sqrt(r_i)
weights = [
    1/√(0.5) ≈ 1.41,   # Moderate upweight
    1/√(1.0) = 1.0,    # No reweighting
    1/√(0.25) = 2.0    # Strong upweight
]

# If per-sequence losses are [1.5, 1.2, 2.0]:
# Final loss = mean([1.41*1.5, 1.0*1.2, 2.0*2.0])
#            = mean([2.12, 1.2, 4.0]) = 2.44
```

**Weight Distribution Analysis**:
- **High mask ratio (1.0)**: Full sequence valid → weight ≈ 1.0 (baseline)
- **Medium mask ratio (0.5)**: Half sequence valid → weight ≈ 1.41 (+41% emphasis)
- **Low mask ratio (0.25)**: Quarter sequence valid → weight = 2.0 (+100% emphasis)
- **Very low mask ratio (0.1)**: Heavy masking → weight ≈ 3.16 (clamped if > max_weight)

**Use Cases**:
- **Variable sequence padding**: Balance training when sequences have different amounts of padding
- **Masked language modeling**: Emphasize sequences with fewer unmasked tokens
- **Sparse supervision**: Give more weight to samples with limited supervision signal
- **Curriculum learning**: Gradually increase emphasis on heavily masked sequences

**Metrics Logged**:

**Mask Statistics**:
- `mean_mask_ratio`: Average ratio of valid tokens across batch [0,1]
- `min_mask_ratio`, `max_mask_ratio`: Range of mask ratios in batch [0,1]

**Weight Statistics**:
- `mean_weight`: Average applied weight across sequences
- `min_weight`, `max_weight`: Range of applied weights [min_weight, max_weight]
- `weight_std`: Standard deviation of sequence weights


### Sequence Scoring Judge Weight Modifier

Purpose: In LANGUAGE_MODEL mode, dynamically reweight each sequence’s loss using a separate SEQUENCE_SCORER “judge” model that outputs a per‑sample wrongness factor in [0,1] (lower is better). The modifier compares this wrongness to the transformed masking ratio y (y = 1 - (-x^4 + 2x^3 - 2x + 1)) and scales loss by (wrongness / y)^exponent with clamping.

How it works:
1) Sampling: From current LM logits (B, T, V), sample one token per position (single pass). No iterative decoding.
2) Judge input: If the judge config contains a CLS token id, prepend it; crop to the judge’s block_size.
3) Scoring: Run the judge in eval+no_grad to obtain wrongness per sample (shape (B,), values in [0,1]).
4) Masking ratio (x): Compute per sample from targets (or provided mask): mask = (targets != ignore_index); x = sum(mask_i)/T. Apply shared transform y = 1 - (-x^4 + 2x^3 - 2x + 1).
5) Factor per sample: factor_i = clamp((wrong_i / clamp(y_i, eps))^exponent, min_factor, max_factor).
6) Loss application:
   - If a per_position_loss tensor is present (e.g., produced by Target Smoothing), multiply it by factor_i per sample and let the pipeline aggregate.
   - Otherwise, multiply the scalar loss by mean(factor).

Supported modes: LANGUAGE_MODEL only.

Configuration (keys):
- loss_modifiers_enabled = True
- judge_weight_modifier_enabled (bool)
- judge_weight_checkpoint (str): Path to SEQUENCE_SCORER checkpoint; loaded eagerly (fail‑fast)
- judge_weight_exponent (float, default 1.0)
- judge_weight_min_factor (float, default 0.1)
- judge_weight_max_factor (float, default 10.0)
- judge_weight_eps (float, default 1e-6)
- device, dtype: reused from main training config

Example:
```python
loss_modifiers_enabled = True

# Sequence Scoring Judge Weight
judge_weight_modifier_enabled = True
judge_weight_checkpoint = "out/judge/ckpt_sequence_scorer.pt"
judge_weight_exponent = 1.0
judge_weight_min_factor = 0.1
judge_weight_max_factor = 10.0
judge_weight_eps = 1e-6
```

Metrics logged:
- mean_wrongness, min_wrongness, max_wrongness, wrongness_std
- mean_factor,   min_factor,   max_factor,   factor_std

Notes:
- No changes to base classes or pipeline are required; this is an additive modifier.
- Judge is loaded on startup to fail fast on configuration issues; runs on the same device/dtype settings as the main model.
- Sampling uses a single multinomial draw per position at temperature=1.

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