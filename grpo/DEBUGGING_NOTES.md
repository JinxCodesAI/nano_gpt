# GRPO Training Issues - Debugging Notes

## Observed Issues

### Issue 1: Loss is 0.0000

```
iter 0: loss 0.1235 (main 0.1235, critic 0.0000), time 21747.97ms
iter 1: loss 0.0000 (main 0.0000, critic 0.0000), time 103634.20ms
```

**Possible Causes:**

1. **All rewards are identical**
   - If judge gives same score to all completions
   - Advantages = rewards - baseline = 0 for all samples
   - pg_loss = -(log_probs × 0).mean() = 0

2. **Sequence log probs are zero**
   - If no masked positions in batch
   - If log probs computation is broken

3. **Gradient accumulation issue**
   - Loss scaled by 1/grad_accum_steps might be too small
   - But this shouldn't make it exactly 0

### Issue 2: Extremely Slow Training

```
iter 0: time 21747.97ms (~22 seconds)
iter 1: time 103634.20ms (~104 seconds)
```

**Possible Causes:**

1. **torch.compile on first iteration**
   - First iteration with compile=True is very slow (compilation)
   - Subsequent iterations should be faster
   - But iter 1 is even slower than iter 0!

2. **Three forward passes per micro-step**
   - Sampling: 1 forward pass
   - Current policy: 1 forward pass
   - Reference policy: 1 forward pass
   - With grad_accum_steps=16: 3 × 16 = 48 forward passes per iteration

3. **Group size multiplier**
   - batch_size=1, group_size=8
   - Processing 8 samples per micro-step
   - 8 × 16 = 128 samples per iteration

4. **Compilation overhead**
   - torch.compile may be recompiling on each iteration
   - Or compilation is not helping (may hurt for dynamic shapes)

## Diagnostic Logging Added

The code now logs after each major step:

```
[Micro 0] 1. Iteration start: mem=XXX MB, time=0.000s
[Micro 0] 2. Samples generated: mem=XXX MB, time=X.XXXs
[Micro 0] 3. Samples scored: mem=XXX MB, time=X.XXXs, rewards=X.XXXX±X.XXXX
[Micro 0] 4. Loss computed: mem=XXX MB, time=X.XXXs, pg_loss=X.XXXXXX, kl=X.XXXXXX, total=X.XXXXXX
5. Optimizer step finished: mem=XXX MB, time=X.XXXs total
```

This will help identify:
- Which step is slow
- Memory usage at each step
- Whether rewards vary
- Whether loss components are zero

## Expected Behavior

### Timing (without compile)

Rough estimates for batch_size=1, group_size=8, grad_accum_steps=16:

- Sampling: ~0.5-1s per micro-step
- Scoring: ~0.2-0.5s per micro-step
- Current policy forward: ~0.5-1s per micro-step
- Reference policy forward: ~0.5-1s per micro-step
- Backward: ~0.5-1s per micro-step

**Total per micro-step:** ~2-4 seconds
**Total per iteration (16 micro-steps):** ~32-64 seconds

### Timing (with compile, after warmup)

Should be 20-30% faster after compilation:
**Total per iteration:** ~25-50 seconds

### Loss Values

- **pg_loss:** Should be negative (we're maximizing)
- **kl_penalty:** Should be small positive (typically 0.001-0.1)
- **total loss:** Should be non-zero

If loss is exactly 0, something is broken.

## Debugging Steps

### Step 1: Check Rewards

Look for this in logs:
```
[Micro 0] 3. Samples scored: rewards=X.XXXX±X.XXXX
```

**If std is ~0:** All rewards are identical → advantages are zero → loss is zero

**Expected:** Rewards should vary (std > 0.01)

### Step 2: Check Advantages

Look for this warning:
```
WARNING: All advantages are near zero! rewards_grouped=[...]
```

**If this appears:** Judge is giving identical scores to all completions

**Possible fixes:**
- Check judge model is loaded correctly
- Verify judge is a sequence scorer (not language model)
- Check if completions are actually different

### Step 3: Check Log Probs

Look for this warning:
```
WARNING: All sequence_log_probs are near zero! num_masked=X.X
```

**If this appears:** Log probability computation is broken

**Possible fixes:**
- Check if masked positions exist (num_masked > 0)
- Verify model forward pass returns correct logits
- Check gather operation is working

### Step 4: Identify Slow Step

Look at timing logs to see which step takes longest:

**If "Samples generated" is slow:**
- predict_and_sample_tokens is the bottleneck
- May be due to torch.compile
- Try compile=False

**If "Samples scored" is slow:**
- Judge model forward pass is slow
- Check if judge is compiled (shouldn't be)

**If "Loss computed" is slow:**
- Current/reference policy forwards are slow
- May be due to torch.compile
- Try compile=False

### Step 5: Test Without Compile

Set in config:
```python
compile = False
```

This will:
- Remove compilation overhead
- Make timing more predictable
- Help identify if compile is the issue

## Recommended Actions

1. **Run with detailed logging** (already added)
2. **Check rewards variance** in logs
3. **Disable compile** temporarily: `compile = False`
4. **Reduce grad_accum_steps** for faster iteration: `gradient_accumulation_steps = 4`
5. **Check judge model** is working correctly

## Quick Test

To quickly test if GRPO logic is working:

```python
# In grpo_config.py
batch_size = 1
group_size = 4  # Reduce from 8
gradient_accumulation_steps = 1  # Reduce from 16
compile = False  # Disable compilation
max_iters = 10  # Just test a few iterations
```

This should complete in ~30-60 seconds and show if loss is non-zero.

