# Diffusion Text Generation User Guide

This guide explains the different generation modes available in `sample.py` and how to configure them for different results.

## Overview

The diffusion text generation system supports multiple generation modes with different remasking strategies. All generation starts with a completely masked sequence and iteratively unmasks and remasks tokens until convergence or maximum iterations.

## Generation Modes

### 1. Random Remasking (Basic Mode)
The simplest generation mode using random remasking patterns.

**Configuration:**
```python
use_intelligent_remasking = False
remasking_checkpoint_name = None  # No remasking model needed
```

**How it works:**
- Uses schedule-based remasking (linear or exponential)
- Randomly selects tokens to remask at each iteration
- Runs for exactly `diffusion_iterations` steps

### 2. Intelligent Schedule-Based Remasking
Uses a trained remasking model but follows a predefined schedule.

**Configuration:**
```python
use_intelligent_remasking = True
remasking_checkpoint_name = 'ckpt_remasking_naive_4400.pt'  # Path to remasking model
remasking_schedule = 'linear'  # or 'exponential'
diffusion_iterations = 30  # Number of iterations
```

**How it works:**
- Uses remasking model to intelligently select which tokens to remask
- Follows `remasking_schedule` to determine how many tokens to remask
- Runs for exactly `diffusion_iterations` steps
- At each step, selects the N tokens with highest [WRONG] probabilities

### 3. Threshold-Based Intelligent Remasking
Uses a trained remasking model with probability thresholds for natural convergence.

**Configuration:**
```python
use_intelligent_remasking = True
use_mixed_remasking = False
remasking_checkpoint_name = 'ckpt_remasking_naive_4400.pt'
remasking_confidence_threshold = 0.003  # Probability threshold
diffusion_iterations = 30  # Number of iterations
```

**How it works:**
- At each iteration, remasks ALL tokens with [WRONG] probability > threshold
- Stops early when no tokens exceed the threshold (natural convergence)
- Maximum of `diffusion_iterations` steps as safety limit
- Most adaptive and intelligent mode

### 4. Mixed Random+Intelligent Remasking (Recommended for Controlled Generation)
Combines schedule-based token count with intelligent candidate selection.

**Configuration:**
```python
use_intelligent_remasking = True  # Enable intelligent remasking
use_mixed_remasking = True        # Enable mixed mode
remasking_checkpoint_name = 'ckpt_remasking_naive_4400.pt'
remasking_confidence_threshold = 0.01  # Threshold for intelligent candidates
remasking_schedule = 'linear'          # or 'exponential'
diffusion_iterations = 30              # Fixed number of iterations
```

**How it works:**
1. **Calculate target count**: Uses schedule to determine how many tokens to remask
2. **Find intelligent candidates**: Gets all tokens with [WRONG] probability > threshold
3. **Smart selection**:
   - If candidates ≥ target: Select top N candidates by probability
   - If candidates < target: Use ALL candidates + random additional tokens
   - If no candidates: Fall back to pure random selection
4. **Fixed iterations**: Always runs for exactly `diffusion_iterations` steps
5. **Best of both worlds**: Combines intelligent selection with controlled generation length

## Key Parameters

### Model Configuration
- `checkpoint_name`: Main diffusion model (unmasking model)
- `remasking_checkpoint_name`: Remasking model checkpoint (optional)
- `device`: 'cpu', 'cuda', 'cuda:0', etc.
- `compile`: Use PyTorch 2.0 compilation for speed

### Generation Control
- `sequence_length`: Length of generated text (e.g., 1024 tokens)
- `num_samples`: Number of texts to generate
- `diffusion_iterations`: Maximum number of iterations (safety limit)
- `seed`: Random seed for reproducible results

### Remasking Strategy
- `use_intelligent_remasking`: Enable/disable intelligent remasking
- `use_mixed_remasking`: Enable mixed random+intelligent mode (requires `use_intelligent_remasking=True`)
- `remasking_confidence_threshold`: Probability threshold for intelligent remasking
- `remasking_schedule`: 'linear' or 'exponential' (for schedule-based modes)

### Schedule Parameters (Schedule-based mode only)
- `start_ratio`: Starting masking ratio (1.0 = all masked)
- `end_ratio`: Ending masking ratio (0.1 = 10% masked)

## Configuration Examples

### Example 1: High-Quality Adaptive Generation (Threshold-Based)
```python
use_intelligent_remasking = True
use_mixed_remasking = False
remasking_checkpoint_name = 'ckpt_remasking_naive_4400.pt'
remasking_confidence_threshold = 0.005  # Conservative threshold
diffusion_iterations = 100  # Safety limit
sequence_length = 1024
```
**Expected behavior:** Generates high-quality text, stops when model is confident about all tokens, typically converges in 20-50 iterations.

### Example 2: Controlled Mixed Generation (Recommended)
```python
use_intelligent_remasking = True
use_mixed_remasking = True
remasking_checkpoint_name = 'ckpt_remasking_naive_4400.pt'
remasking_confidence_threshold = 0.01   # Moderate threshold for candidates
remasking_schedule = 'linear'
diffusion_iterations = 30   # Fixed iterations
start_ratio = 1.0
end_ratio = 0.1
```
**Expected behavior:** Combines intelligent token selection with predictable generation length, runs exactly 30 iterations with decreasing remask count.

### Example 3: Fast Mixed Generation
```python
use_intelligent_remasking = True
use_mixed_remasking = True
remasking_confidence_threshold = 0.05   # Higher threshold for fewer candidates
remasking_schedule = 'exponential'
diffusion_iterations = 20   # Shorter generation
sequence_length = 512
```
**Expected behavior:** Faster generation with good quality, more aggressive remasking schedule with intelligent candidate selection.

### Example 4: Baseline Random Generation
```python
use_intelligent_remasking = False
use_mixed_remasking = False
remasking_schedule = 'linear'
diffusion_iterations = 50
start_ratio = 1.0
end_ratio = 0.1
```
**Expected behavior:** Simple random remasking, good for baseline comparisons.

## Command Line Usage

You can override any parameter via command line:

```bash
# High-quality adaptive generation (threshold-based)
python sample.py --use_intelligent_remasking=True --remasking_confidence_threshold=0.005

# Mixed random+intelligent generation
python sample.py --use_intelligent_remasking=True --use_mixed_remasking=True --remasking_confidence_threshold=0.01

# Fast generation with exponential schedule
python sample.py --use_mixed_remasking=True --remasking_schedule=exponential --diffusion_iterations=20

# Specific model and length
python sample.py --checkpoint_name=my_model.pt --sequence_length=512

# Multiple samples
python sample.py --num_samples=5
```

## Debugging and Monitoring

### Verbose Output
The system provides detailed logging when `verbose=True`:
- Iteration progress and token counts
- [WRONG] probability ranges
- Number of tokens remasked
- Convergence information

### Key Metrics to Watch:
- **Convergence iteration**: When generation stops naturally
- **[WRONG] probability ranges**: Should decrease over time
- **Tokens remasked per iteration**: Should generally decrease
- **Final text quality**: Coherence and fluency

## Model Requirements

### Unmasking Model (Required)
- Trained with `training_type='unmasking'`
- Learns to predict original tokens at masked positions
- Any attention type (causal or bidirectional)

### Remasking Model (Optional but Recommended)
- Trained with `training_type='remasking'` 
- Learns to identify tokens that should be remasked
- Must have identical architecture to unmasking model
- Predicts [WRONG] token at positions needing remasking

## Performance Tips

1. **Use GPU**: Set `device='cuda'` for faster generation
2. **Enable compilation**: Set `compile=True` for PyTorch 2.0 speedup
3. **Appropriate thresholds**: Balance quality vs speed with threshold tuning
4. **Batch processing**: Generate multiple samples in one run
5. **Model caching**: Keep models loaded for multiple generations

## Troubleshooting

### Generation doesn't stop (runs all iterations)
- **Cause**: Threshold too low, model always finds tokens to remask
- **Fix**: Increase `remasking_confidence_threshold`

### Generation stops too early (poor quality)
- **Cause**: Threshold too high
- **Fix**: Decrease `remasking_confidence_threshold`

### Out of memory errors
- **Cause**: Sequence too long or models too large
- **Fix**: Reduce `sequence_length` or use smaller models

### Poor text quality
- **Cause**: Poorly trained models or wrong parameters
- **Fix**: Use well-trained models, adjust thresholds, try different schedules

This guide should help you achieve optimal results with the diffusion text generation system!