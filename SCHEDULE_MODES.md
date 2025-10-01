# Schedule Modes for Diffusion Sampling

This document describes the two schedule modes available for diffusion-based text generation: **ratio** and **threshold**.

## Overview

During diffusion generation, the model iteratively unmasks and remasks tokens. The schedule determines which tokens to remask at each iteration. Two modes are now supported:

1. **Ratio mode** (default): Masks a fixed percentage of tokens at each iteration
2. **Threshold mode**: Masks all tokens with wrongness scores above a threshold

## Ratio Mode (Default)

### Behavior
- Masks a **fixed percentage** of tokens at each iteration
- The percentage decreases linearly from `start_ratio` to `end_ratio`
- Uses critic scores to **rank** tokens and select the top-k worst ones
- Always completes all iterations
- Predictable and consistent masked token counts

### Example
With `start_ratio=0.95` and `end_ratio=0.05` over 15 iterations:
```
Iteration 1/15: 16384/16384 masked (100.0%)
Iteration 2/15: 14511/16384 masked (88.6%)
Iteration 3/15: 13458/16384 masked (82.1%)
...
Iteration 15/15: 819/16384 masked (5.0%)
```

### Usage
```bash
python sample_simple.py --schedule-mode ratio --start-ratio 0.95 --end-ratio 0.05
```

## Threshold Mode

### Behavior
- Masks **all tokens** with wrongness probability above a threshold
- The threshold increases linearly (inverted from ratio)
- Can **finish early** if no tokens exceed the threshold
- Masked count depends on actual token quality
- More adaptive to generation quality

### Threshold Calculation
The threshold is **inverted** from the ratio parameter:
- `start_ratio=0.95` → threshold=0.05 (mask tokens with wrongness > 0.05, i.e., almost everything)
- `end_ratio=0.05` → threshold=0.95 (mask tokens with wrongness > 0.95, i.e., only the worst)

This inversion ensures that:
- At the **beginning** (high ratio), the threshold is **low**, so many tokens are masked
- At the **end** (low ratio), the threshold is **high**, so few tokens are masked

### Example
With `start_ratio=0.95` and `end_ratio=0.05` over 15 iterations:
```
Iteration 1/15: 16384/16384 masked (100.0%)
Iteration 2/15: 16383/16384 masked (100.0%)
Iteration 3/15: 16365/16384 masked (99.9%)
Iteration 4/15: 16199/16384 masked (98.9%)
Iteration 5/15: 15494/16384 masked (94.6%)
Iteration 6/15: 13990/16384 masked (85.4%)
Iteration 7/15: 11455/16384 masked (69.9%)
Iteration 8/15: 8178/16384 masked (49.9%)
Iteration 9/15: 5002/16384 masked (30.5%)
Iteration 10/15: 2413/16384 masked (14.7%)
Iteration 11/15: 826/16384 masked (5.0%)
Iteration 12/15: 176/16384 masked (1.1%)
Iteration 13/15: 24/16384 masked (0.1%)
Iteration 14/15: 1/16384 masked (0.0%)
  -> Early termination: no tokens exceed threshold
```

### Usage
```bash
python sample_simple.py --schedule-mode threshold --start-ratio 0.95 --end-ratio 0.05
```

## Comparison

| Feature | Ratio Mode | Threshold Mode |
|---------|-----------|----------------|
| **Masking strategy** | Fixed percentage | All above threshold |
| **Masked count** | Predictable, linear decrease | Adaptive, depends on quality |
| **Iterations** | Always completes all | Can finish early |
| **Use case** | Consistent, controlled generation | Adaptive, quality-driven generation |
| **Critic requirement** | Optional (falls back to random) | Required for meaningful results |

## Implementation Details

### Wrongness Score
The wrongness score is computed differently depending on the remasking strategy:

1. **Critic-guided** (default in sample_simple.py):
   - Uses `model.critic_scores()` to get logits
   - Converts to probability: `wrongness = sigmoid(critic_logits)`
   - Higher wrongness = more likely to be incorrect

2. **Intelligent remasking**:
   - Uses model's prediction confidence
   - `wrongness = 1.0 - p_taken` (uncertainty)
   - Higher uncertainty = more likely to remask

3. **Remasking model**:
   - Uses separate model to predict wrongness
   - `wrongness = 1.0 - sigmoid(confidence)`

### Early Termination
In threshold mode, if no tokens have wrongness above the threshold, the function returns `None` to signal early termination. The generation loop detects this and stops iterating.

## Parameters

### Command-line Arguments (sample_simple.py)
- `--schedule-mode {ratio,threshold}`: Choose schedule mode (default: ratio)
- `--start-ratio FLOAT`: Initial mask ratio/threshold (default: 0.95)
- `--end-ratio FLOAT`: Final mask ratio/threshold (default: 0.05)
- `--iterations INT`: Number of diffusion iterations (default: 15)

### Function Parameters
```python
def apply_remasking_step(..., schedule_mode='ratio'):
    """
    Args:
        schedule_mode: 'ratio' (default, mask fixed percentage) or 
                      'threshold' (mask all above threshold)
    
    Returns:
        Remasked tokens, or None if threshold mode and no tokens to mask
    """
```

## Testing

Two test scripts are provided:

1. **test_schedule_modes.py**: Unit tests for both modes
2. **test_full_schedule.py**: Full generation simulation showing masked counts

Run tests:
```bash
python test_schedule_modes.py
python test_full_schedule.py
```

## Notes

- The threshold mode requires a model with critic head (`add_critic_head=True`) for meaningful results
- Without a critic head, threshold mode falls back to ratio-based masking
- The `randomness_strength` parameter is not applied in threshold mode (only in ratio mode)
- Custom masking ratios (`--masking-ratios`) work with both modes

