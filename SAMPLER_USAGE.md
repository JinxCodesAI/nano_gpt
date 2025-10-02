# Sampler Head Usage Guide

## Overview

The sampler head provides **coherent parallel token generation** by conditioning each prediction on its immediate left and right neighbors using wavefront filling. This solves the problem of local incoherence when multiple tokens are predicted simultaneously.

## Training with Sampler Head

### 1. Configuration

Create or modify your training config to enable the sampler head:

```python
# config/train_char_diffusion_sampler.py

# Sampler head configuration
add_sampler_head = True
start_sampler_iteration = 1000  # Start training sampler after 1000 iterations
sampler_min_neighbors_ratio = 0.01  # Bootstrap 1% of tokens when no neighbors available

# Critic head configuration (optional, for three-stage training)
add_critic_head = True
start_critic_iteration = 3000  # Start critic after sampler is trained
end_critic_iteration = 5000
critic_alpha = 1.0
```

### 2. Three-Stage Training Schedule

When both sampler and critic are enabled:

- **Stage 1 (0-1000)**: Main model only
  ```
  iter 500: loss 4.5000 (main 4.5000), time 150.00ms
  ```

- **Stage 2 (1000-3000)**: Main model + Sampler
  ```
  iter 1500: loss 9.2000 (main 4.6000, sampler 4.6000), time 155.00ms
  ```

- **Stage 3 (3000+)**: Main model + Sampler + Critic
  ```
  iter 3500: loss 13.8000 (main 4.6000, sampler 4.6000, critic 4.6000), time 160.00ms
  ```

### 3. Start Training

```bash
python train.py config/train_char_diffusion_sampler.py
```

### 4. Expected Output

At model initialization, you should see:
```
Initializing a new model from scratch
Using bidirectional attention
Sampler head enabled (start_iter=1000)  ← IMPORTANT: Verify this appears!
Critic head enabled (alpha=1.0)
number of parameters: 10.XX M
```

If you don't see "Sampler head enabled", the sampler was not created. Check:
1. Config has `add_sampler_head = True`
2. Model mode is `LANGUAGE_MODEL`
3. Attention type is `bidirectional`
4. You're not resuming from an old checkpoint without sampler

## Sampling with Sampler Head

### Basic Usage

The sampler is **automatically detected** and used when available:

```bash
python sample_simple.py \
    --out-dir out-char-diffusion \
    --checkpoint-name ckpt.pt \
    --num-samples 16 \
    --sequence-length 1024 \
    --iterations 15
```

### Compare Sampler vs Naive Sampling

To compare the quality difference, use the `--disable-sampler` flag:

#### With Sampler (Coherent Sampling)
```bash
python sample_simple.py \
    --out-dir out-char-diffusion \
    --checkpoint-name ckpt.pt \
    --num-samples 16 \
    --save sampler_enabled.json
```

#### Without Sampler (Naive Sampling)
```bash
python sample_simple.py \
    --out-dir out-char-diffusion \
    --checkpoint-name ckpt.pt \
    --num-samples 16 \
    --save sampler_disabled.json \
    --disable-sampler  # ← Force naive sampling
```

Then compare the outputs to see the coherence improvement!

### Advanced Sampling Options

```bash
python sample_simple.py \
    --out-dir out-char-diffusion \
    --checkpoint-name ckpt.pt \
    --num-samples 16 \
    --sequence-length 1024 \
    --iterations 15 \
    --temperature 0.8 \
    --top-p 0.95 \
    --schedule-mode ratio \
    --start-ratio 0.95 \
    --end-ratio 0.05 \
    --seed-text "To be or not to be" \
    --seed-placement random \
    --quality-metric judge \
    --judge-checkpoint-name judge_model.pt \
    --save output.json \
    --verbose
```

## Checking Checkpoints

To verify if a checkpoint has a sampler head:

```python
import torch

ckpt = torch.load('out-char-diffusion/ckpt.pt', map_location='cpu', weights_only=False)

# Check model_args
print("add_sampler_head:", ckpt['model_args'].get('add_sampler_head', False))
print("start_sampler_iteration:", ckpt['model_args'].get('start_sampler_iteration', 'N/A'))

# Check state_dict
has_sampler = any('sampler_head' in k for k in ckpt['model'].keys())
print("Has sampler in state_dict:", has_sampler)

# List sampler parameters
if has_sampler:
    sampler_params = [k for k in ckpt['model'].keys() if 'sampler_head' in k]
    print(f"\nSampler parameters ({len(sampler_params)}):")
    for p in sampler_params[:5]:  # Show first 5
        print(f"  {p}")
```

## Troubleshooting

### Problem: "Sampler head enabled" doesn't appear during training

**Cause**: Sampler parameters not passed to model initialization.

**Solution**: Make sure you're using the latest `train.py` that includes sampler parameters in `model_args` (lines 122-124, 250-252).

### Problem: Training shows only main loss, no sampler loss

**Causes**:
1. Resuming from old checkpoint without sampler
2. Current iteration < `start_sampler_iteration`
3. Sampler not created (see above)

**Solutions**:
1. Delete old checkpoint or use different `out_dir`
2. Wait until iteration reaches `start_sampler_iteration`
3. Verify "Sampler head enabled" appears at startup

### Problem: Sampler loss is very high

**Expected**: Sampler loss starts high (~4.6 for random initialization) and decreases over time. This is normal!

The sampler is learning from scratch to predict tokens conditioned on neighbors. Give it time to train.

### Problem: Sample quality doesn't improve with sampler

**Possible causes**:
1. Sampler not trained enough (check iteration count)
2. Temperature too high (try 0.8 instead of 1.0)
3. Dataset doesn't benefit from local coherence
4. Need to tune `sampler_min_neighbors_ratio`

**Debug**:
1. Check sampler loss is decreasing during training
2. Compare with `--disable-sampler` to see difference
3. Try different sampling parameters
4. Visualize with `--save output.json` and inspect iteration data

## Architecture Details

### Gradient Isolation

The sampler is an **auxiliary network** - gradients from sampler loss do NOT affect:
- ❌ Transformer layers
- ❌ Token embeddings
- ❌ Position embeddings
- ❌ LM head

Only sampler_head parameters are updated by sampler loss.

This is verified by `test_sampler_gradient_isolation.py`.

### Wavefront Filling

During inference, the sampler uses wavefront-based filling:

1. **Wave 1**: Fill tokens with ≥1 non-masked neighbor
2. **Wave 2**: Fill newly unmasked tokens' neighbors
3. **Continue** until all tokens filled
4. **Bootstrap**: If no tokens have neighbors (all masked), fill top 1% by confidence

This ensures each prediction is conditioned on at least one real neighbor, improving local coherence.

### Loss Calculation

```python
total_loss = main_loss + sampler_loss + (critic_alpha * critic_loss)
```

- Main loss: Standard cross-entropy for language modeling
- Sampler loss: Standard cross-entropy for neighbor-conditioned prediction (no weight)
- Critic loss: Binary cross-entropy for quality prediction (weighted by alpha)

## Performance Impact

### Training Speed
- **Stage 1** (main only): Baseline speed
- **Stage 2** (main + sampler): ~3-5% slower (minimal overhead)
- **Stage 3** (main + sampler + critic): ~5-10% slower

### Inference Speed
- **Naive sampling**: Fast (single forward pass per iteration)
- **Sampler wavefront**: Slower (multiple forward passes per wave)
  - Typical: 2-5x slower depending on masking ratio
  - Trade-off: Better quality for slower generation

### Memory Usage
- Sampler adds ~2-3% more parameters
- Minimal memory overhead during training
- No additional memory during inference

## Best Practices

1. **Start sampler early**: `start_sampler_iteration = 1000` (after initial convergence)
2. **Train sampler before critic**: Give sampler time to learn (2000+ iterations)
3. **Use bidirectional attention**: Required for sampler to work
4. **Monitor sampler loss**: Should decrease steadily
5. **Compare with/without**: Use `--disable-sampler` to verify improvement
6. **Tune temperature**: Lower (0.7-0.9) often works better with sampler
7. **Save checkpoints frequently**: `always_save_checkpoint = True`

## References

- Full specification: `docs/sampler.md`
- Implementation summary: `SAMPLER_IMPLEMENTATION_SUMMARY.md`
- Review and fixes: `SAMPLER_REVIEW_AND_FIXES.md`
- Gradient isolation tests: `test_sampler_gradient_isolation.py`
- Integration tests: `test_sample_simple_integration.py`

