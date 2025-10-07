# GRPO Memory Optimization Guide

## Memory Usage Analysis

GRPO training is significantly more memory-intensive than standard supervised training due to:

### 1. **Multiple Forward Passes Per Micro-Step**

Standard training: **1 forward pass**
- Model forward on input batch

GRPO training: **3 forward passes**
- Sampling forward (with `torch.no_grad()`)
- Current policy forward (with gradients)
- Reference policy forward (with `torch.no_grad()`)

### 2. **Group Size Multiplier**

Standard training processes `batch_size` samples.

GRPO processes `batch_size × group_size` samples:
- Each input is repeated `group_size` times
- Generates `group_size` completions per input
- All completions processed through models

**Example:**
- Standard: `batch_size=4` → 4 samples
- GRPO: `batch_size=1, group_size=8` → 8 samples (2x more)
- GRPO: `batch_size=2, group_size=8` → 16 samples (4x more)

### 3. **Large Logits Tensors**

Each forward pass produces logits: `(batch_size × group_size, seq_len, vocab_size)`

**Memory per logits tensor:**
- Shape: `(B×k, T, V)` where B=batch_size, k=group_size, T=seq_len, V=vocab_size
- Example: `(8, 1024, 67)` = 8 × 1024 × 67 × 4 bytes = ~2.2 MB
- Three forward passes = ~6.6 MB per micro-step
- With gradient accumulation: multiply by `grad_accum_steps`

### 4. **Additional Tensors**

- Completions: `(B×k, T)`
- Rewards: `(B×k,)`
- Advantages: `(B×k,)`
- Log probabilities: `(B×k, T)`
- KL divergence: `(B×k,)`

## Memory Optimizations Implemented

### 1. **Don't Return Logits from Sampling** ✅

**Before:**
```python
completions, logits_gen = predict_and_sample_tokens(..., return_logits=True)
# logits_gen: (B*k, T, V) kept in memory but never used
```

**After:**
```python
completions = predict_and_sample_tokens(..., return_logits=False)
# No logits stored, saves ~2.2 MB per micro-step
```

**Savings:** ~2.2 MB per micro-step (for typical config)

### 2. **Explicit Tensor Deletion** ✅

**Added explicit `del` statements after tensor use:**
```python
# Compute log probs
log_probs_current = F.log_softmax(logits_current, dim=-1)
token_log_probs = torch.gather(log_probs_current, ...)
del logits_current, log_probs_current  # Free immediately

# After backward pass
del X, Y, mask, X_repeated, mask_repeated
del completions, rewards, advantages
del sequence_log_probs, kl_divergence
```

**Savings:** Ensures tensors are freed immediately, not waiting for Python GC

### 3. **Compute and Discard Pattern** ✅

**Pattern used throughout:**
1. Compute logits
2. Extract needed information (log-probs)
3. Delete logits immediately
4. Continue with smaller tensors

**Example:**
```python
logits_current, _ = self.generator(X_repeated, targets=None)
log_probs_current = F.log_softmax(logits_current, dim=-1)
token_log_probs = torch.gather(log_probs_current, ...)
del logits_current, log_probs_current  # Free large tensors
sequence_log_probs = (token_log_probs * mask).sum(dim=1)
del token_log_probs  # Free intermediate tensor
```

## Memory Comparison

### Sequence Scorer Training (Your Reference)

**Config:** `batch_size=4, gradient_accumulation_steps=8`
- Processes: 4 samples per micro-step
- Forward passes: 1 per micro-step
- Memory: ~70% GPU utilization

### GRPO Training (Current)

**Config:** `batch_size=1, group_size=8, gradient_accumulation_steps=16`
- Processes: 8 samples per micro-step (2x more than sequence scorer)
- Forward passes: 3 per micro-step (3x more than sequence scorer)
- **Effective memory multiplier: 2x × 3x = 6x**

**Why batch_size=1 is necessary:**
- 6x memory multiplier means we need 6x smaller batch
- Sequence scorer uses batch_size=4
- GRPO should use batch_size ≈ 4/6 ≈ 0.67 → round to 1

## Recommended Settings

### For Memory-Constrained Training

```python
batch_size = 1                      # Keep small
group_size = 4                      # Reduce from 8 to 4
gradient_accumulation_steps = 32   # Increase to maintain effective batch
```

**Effective batch size:** `1 × 4 × 32 = 128`

### For More Memory Available

```python
batch_size = 2                      # Double batch size
group_size = 8                      # Standard group size
gradient_accumulation_steps = 16   # Moderate accumulation
```

**Effective batch size:** `2 × 8 × 16 = 256`

### For Maximum Memory Efficiency

```python
batch_size = 1                      # Minimum
group_size = 4                      # Minimum for stable advantages
gradient_accumulation_steps = 64   # Maximum accumulation
```

**Effective batch size:** `1 × 4 × 64 = 256`

## Further Optimization Opportunities

### 1. **Gradient Checkpointing** (Not Implemented)

Trade compute for memory by recomputing activations during backward pass.

**Potential savings:** 30-50% memory reduction
**Cost:** 20-30% slower training

**Implementation:**
```python
# In model.py
self.transformer = torch.nn.ModuleList([
    torch.utils.checkpoint.checkpoint_wrapper(block)
    for block in blocks
])
```

### 2. **Mixed Precision Training** (Already Used)

Using `bfloat16` or `float16` reduces memory by 2x.

**Current:** Already using `dtype='bfloat16'` ✅

### 3. **Reduce Sequence Length**

Shorter sequences = less memory.

**Current:** `block_size=1024`
**Alternative:** `block_size=512` (saves 50% memory)

### 4. **Smaller Group Size**

Fewer completions per input = less memory.

**Current:** `group_size=8`
**Alternative:** `group_size=4` (saves 50% memory)
**Trade-off:** Less stable advantage estimates

### 5. **Flash Attention** (If Available)

More memory-efficient attention implementation.

**Requires:** `flash-attn` package and compatible GPU

## Monitoring Memory Usage

### During Training

Watch for these metrics in logs:
```
mem_alloc_mb: Current allocated memory
mem_reserved_mb: Reserved memory
mem_max_alloc_mb: Peak allocated memory
```

### Manual Check

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
```

## Troubleshooting OOM Errors

### If you get "CUDA out of memory" errors:

1. **Reduce `batch_size`** (most effective)
   - Try: `batch_size=1`

2. **Reduce `group_size`** (very effective)
   - Try: `group_size=4` or `group_size=2`

3. **Increase `gradient_accumulation_steps`** (no memory cost)
   - Compensates for smaller batch/group size
   - Try: `gradient_accumulation_steps=32` or `64`

4. **Reduce `block_size`** (effective but changes task)
   - Try: `block_size=512`

5. **Use smaller model** (if possible)
   - Reduce `n_layer`, `n_embd`, or `n_head`

## Summary

GRPO is inherently memory-intensive due to:
- 3 forward passes per micro-step
- `group_size` multiplier on batch size
- Large logits tensors

**Optimizations implemented:**
- Don't store unused logits from sampling
- Explicit tensor deletion after use
- Compute-and-discard pattern for large tensors

**Result:** Can run with `batch_size=1, group_size=8` on same hardware that runs sequence scorer with `batch_size=4`.

**Recommendation:** Start with conservative settings and gradually increase if memory allows.

