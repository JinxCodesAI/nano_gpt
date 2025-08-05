
# The Definitive `argmax`-Free Refactor of `train.py`

## 1. The Goal

To replace the entire complex, `argmax`-based loss calculation with a new, simplified, and mathematically sound system that is easier to control and debug. This involves:
1.  Replacing the five loss-related functions with a single, new `calculate_diffusion_loss` function.
2.  Updating the configuration parameters to match this simpler logic.
3.  Updating the main training loop to use a new, more stable feedback signal.
4.  Introducing a data generation curriculum to teach the model generative and corrective skills sequentially.

## 2. Step-by-Step Implementation

### Step 1: Update the Configuration Block

First, we will update the hyperparameters at the top of `train.py` to reflect our new, simpler loss structure.

**Action:** In `train.py`, replace your entire "Diffusion loss specific penalties" block with the following:

```python
# In train.py config section

# ... (model parameters) ...
model_type = 'gpt2' # 'gpt2' or 'diffusion'

# --- NEW SIMPLIFIED LOSS CONFIG ---
initial_mask_logit_bias = 3.5   # The starting value for our dynamic bias.
bias_update_strength = 0.01   # How quickly the bias adapts.
target_mask_prob = 0.5        # The desired output probability for [MASK] on un-masking tasks.

# Simple, task-based loss weights
weight_unmask_task = 1.0        # The base weight for the loss when the model must guess a word.
weight_remask_task = 1.0        # The base weight for the loss when the model must identify a corrupted word.

# Curriculum settings
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # Iterations for the penalty curriculum.
proofreading_warmup_iters = 2000 # NEW: Iterations to ramp up the "re-masking" task.

guaranteed_correct_factor = 0.01
# ...
```

### Step 2: Implement the New Data and Loss Functions

Next, we will replace all five of the old loss-related functions (`_calculate_diagnostic_logs`, `_log_diffusion_behavior`, `_calculate_loss_weights`, `calculate_diffusion_loss`, and your old `get_masking_penalties`) with just three new, clean functions.

**Action:** In `train.py`, **delete** all of your existing loss functions and replace them with this new, complete block of code. A good place is right before the main configuration block.

```python
# In train.py

def get_corruption_scheduler(it):
    """
    Controls the curriculum for both penalties and data generation.
    Returns the current penalty values and the re-masking task ratio.
    """
    # --- Penalty Curriculum ---
    if it >= masking_warmup_iters:
        current_penalty_mask_correct = penalty_mask_correct
    else:
        ratio = it / masking_warmup_iters
        current_penalty_mask_correct = 0.0 + ratio * (penalty_mask_correct - 0.0)

    # --- Data Generation Curriculum ---
    if it >= proofreading_warmup_iters:
        remask_ratio = 1.0
    else:
        remask_ratio = it / proofreading_warmup_iters
        
    return current_penalty_mask_correct, remask_ratio

def calculate_diffusion_loss(logits, targets, inputs, mask_token_id, 
                             current_penalty_mask_correct, weight_unmask_task, 
                             weight_remask_task, mask_logit_bias, log_diagnostics=False):
    """
    A simplified and robust loss function that does not use argmax for weighting.
    It applies weights based only on the task type (the Input-Target pair).
    """
    # --- Step 1: Apply the dynamic logit bias ---
    biased_logits = logits.clone()
    if mask_logit_bias != 0.0:
        biased_logits[:, :, mask_token_id] += mask_logit_bias

    # --- Step 2: Calculate the base cross-entropy loss ---
    flat_logits = biased_logits.view(-1, biased_logits.size(-1))
    flat_targets = targets.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='none')

    # --- Step 3: Apply weights based on the TASK TYPE only ---
    weights = torch.ones_like(per_token_loss)
    flat_inputs = inputs.view(-1)
    
    # Task 1: Un-masking (Input was [MASK], Target is a word)
    unmask_task_positions = (flat_inputs == mask_token_id) & (flat_targets != mask_token_id)
    weights[unmask_task_positions] = weight_unmask_task
    
    # Task 2: Re-masking (Input was a word, Target is [MASK])
    remask_task_positions = (flat_inputs != mask_token_id) & (flat_targets == mask_token_id)
    weights[remask_task_positions] = weight_remask_task
    
    # Task 3: State-Dependent Penalty for Destructive Edits
    with torch.no_grad():
        B, T = inputs.shape
        predictions_2d = torch.argmax(biased_logits, dim=-1)
        for i in range(B):
            epsilon = 1e-6
            input_words_mask = (inputs[i] != mask_token_id)
            num_input_words = input_words_mask.sum().item()
            corrupted_words_mask = (targets[i, input_words_mask] == mask_token_id)
            num_corrupted_words = corrupted_words_mask.sum().item()
            corruption_rate = num_corrupted_words / (num_input_words + epsilon)
            effective_penalty = current_penalty_mask_correct * (1.0 - corruption_rate)
            
            destructive_mask = (inputs[i] == targets[i]) & (inputs[i] != mask_token_id) & (predictions_2d[i] == mask_token_id)
            
            weights_2d = weights.view(B, T)
            weights_2d[i, destructive_mask] = effective_penalty
            
    # The final loss is the weighted average
    final_loss = (per_token_loss * weights).mean()

    # --- Step 4: Calculate the NEW Feedback Signal (argmax-free) ---
    avg_mask_prob = 0.0
    if unmask_task_positions.any():
        unmask_logits = biased_logits.view(-1, biased_logits.size(-1))[unmask_task_positions]
        unmask_probs = F.softmax(unmask_logits, dim=-1)
        avg_mask_prob = unmask_probs[:, mask_token_id].mean().item()

    if log_diagnostics:
        with torch.no_grad():
            avg_mask_logit = logits[:, :, mask_token_id].mean().item()
            avg_max_logit = logits.max(dim=-1).values.mean().item()
            print(f"[DIAG] Avg MASK Logit: {avg_mask_logit:7.2f} | Avg MAX Logit: {avg_max_logit:7.2f} | Avg MASK Prob: {avg_mask_prob:.2%}")

    return final_loss, avg_mask_prob
```

### Step 3: Update the `get_batch` Function

Now we will implement the data generation curriculum.

**Action:** In `train.py`, replace your `get_batch` function with this new version.

```python
# In train.py

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_clean = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if model_type == 'diffusion':
        assert mask_token_id is not None, "mask_token_id must be set globally"
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()

        # Get the current re-masking task ratio from the curriculum scheduler
        _, remask_ratio = get_corruption_scheduler(iter_num)

        b, t = x_corrupted.shape
        for i in range(b):
            max_corruption = 1 - guaranteed_correct_factor
            
            rate_mask = torch.rand(1) * max_corruption
            # The rate of re-masking tasks is now controlled by the curriculum
            rate_random = torch.rand(1) * rate_mask * remask_ratio
            # The final un-masking rate is what's left over
            rate_mask = rate_mask - rate_random

            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            x_corrupted[i, pos_mask] = mask_token_id
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            y_target[i, pos_random] = mask_token_id

        x, y = x_corrupted, y_target
    else:
        y_autoregressive = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x_clean, y_autoregressive

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

### Step 4: Update the Main Training Loop

Finally, we will update the main loop to use our new functions and the new feedback mechanism.

**Action:** In `train.py`, modify the `while True:` loop as follows.

```python
# In train.py

# ... (after DDP wrapping) ...

# --- NEW: Initialize the dynamic bias and its feedback signal ---
mask_logit_bias = initial_mask_logit_bias
running_avg_mask_prob = target_mask_prob # Use a moving average for stability
# --- END NEW ---

# ... (the estimate_loss function needs to be updated) ...
@torch.no_grad()
def estimate_loss(current_penalty_mask_correct, current_mask_logit_bias):
    global fixed_val_batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'val' and model_type == 'diffusion':
            if fixed_val_batches is None:
                # Note: The fixed val set uses the FINAL curriculum values, not the current ones
                _, final_remask_ratio = get_corruption_scheduler(proofreading_warmup_iters)
                fixed_val_batches = create_fixed_validation_set(final_remask_ratio) # Pass ratio here
        for k in range(eval_iters):
            if split == 'val' and model_type == 'diffusion':
                X, Y = fixed_val_batches[k]
            else:
                X, Y = get_batch(split)
            with ctx:
                logits, loss_from_model = model(X, Y)
                if model_type == 'diffusion':
                    # Get final penalty value for eval
                    final_penalty_mask_correct, _ = get_corruption_scheduler(masking_warmup_iters)
                    loss, _ = calculate_diffusion_loss(
                        logits, Y, X, mask_token_id, 
                        final_penalty_mask_correct,
                        weight_unmask_task, weight_remask_task,
                        current_mask_logit_bias,
                        log_diagnostics=False
                    )
                else:
                    loss = loss_from_model
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ... (and the create_fixed_validation_set function) ...
def create_fixed_validation_set(remask_ratio):
    # ... (function body is mostly the same, just use the passed-in remask_ratio) ...

# ... (delete the old get_masking_penalties function) ...

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
while True:
    # --- MODIFICATION: Use the new scheduler ---
    lr = get_lr(iter_num)
    current_penalty_mask_correct, _ = get_corruption_scheduler(iter_num)
    # --- END MODIFICATION ---

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(current_penalty_mask_correct, mask_logit_bias)
        # ... (rest of eval block) ...

    # forward backward update
    for micro_step in range(gradient_accumulation_steps):
        # ...
        with ctx:
            logits, loss_from_model = model(X, Y)
            if model_type == 'diffusion':
                # --- MODIFICATION: Use the new loss function and feedback signal ---
                loss, avg_mask_prob = calculate_diffusion_loss(
                    logits, Y, X, mask_token_id,
                    current_penalty_mask_correct,
                    weight_unmask_task, weight_remask_task,
                    mask_logit_bias,
                    log_diagnostics=(micro_step == gradient_accumulation_steps - 1)
                )
            else:
                loss = loss_from_model
                avg_mask_prob = target_mask_prob # Placeholder
            loss = loss / gradient_accumulation_steps
        # ... (rest of backward pass) ...

    # --- MODIFICATION: The new feedback loop ---
    if model_type == 'diffusion':
        running_avg_mask_prob = 0.99 * running_avg_mask_prob + 0.01 * avg_mask_prob
        error_signal = target_mask_prob - running_avg_mask_prob
        mask_logit_bias += bias_update_strength * error_signal
    # --- END MODIFICATION ---

    # timing and logging
    if iter_num % log_interval == 0 and master_process:
        # ...
        if model_type == 'diffusion':
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, logit_bias {mask_logit_bias:.2f}")
        # ...
    # ... (rest of loop) ...
```
