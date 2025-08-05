# Implementing a Curriculum for Masking Penalties

## 1. The Goal

The objective is to change the `penalty_keep_mask` and `penalty_mask_correct` from fixed values into dynamic values that are scheduled over an initial "warmup" period. This creates a curriculum for the model with two distinct phases:

1.  **Initial Phase (Exploration):** At the beginning of training, the model is strongly encouraged to make guesses and is not heavily penalized for being wrong.
    *   `penalty_keep_mask` starts at `1.0` (high penalty for not guessing).
    *   `penalty_mask_correct` starts at `0.0` (no penalty for incorrectly masking a correct word).

2.  **Final Phase (Refinement):** After the warmup period, the penalties settle at their final configured values, encouraging the model to be more precise and cautious.
    *   `penalty_keep_mask` decreases to its configured value (e.g., `0.25`), allowing the model to say "I don't know" more freely.
    *   `penalty_mask_correct` increases to its configured value (e.g., `0.5`), penalizing the model more for altering work that was already correct.

This transition will happen linearly over a configurable number of iterations (`masking_warmup_iters`).

## 2. Step-by-Step Implementation

Follow these instructions to modify your `train.py` file.

### Step 1: Add the New Configuration Parameter

First, we need to add the `masking_warmup_iters` parameter to the configuration section of the script.

**Action:** In `train.py`, add the new parameter alongside the other diffusion-specific penalties.

```python
# In train.py

# ... (inside the configuration block) ...
# Diffusion loss specific penalties
penalty_keep_mask = 0.25      # Final discount for failing to unmask, but keeping [MASK].
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # NEW: Number of iterations for the penalty curriculum.
guaranteed_correct_factor = 0.01
# ...
```

### Step 2: Create the Penalty Scheduler Function

This new function will calculate the dynamic penalty values based on the current training iteration, similar to how `get_lr` works.

**Action:** Add the following function definition in `train.py`, ideally right after the `get_lr` function.

```python
# In train.py

# ... (after get_lr function)

def get_masking_penalties(it):
    """
    Calculates the dynamic penalties for the diffusion loss based on the training iteration.
    Implements a linear curriculum over `masking_warmup_iters`.
    """
    # If we are past the warmup, return the final configured values
    if it >= masking_warmup_iters:
        return penalty_keep_mask, penalty_mask_correct

    # Calculate the progress ratio (from 0.0 to 1.0) through the warmup
    ratio = it / masking_warmup_iters

    # Linearly decrease penalty_keep_mask from 1.0 down to its final value
    # Formula: start + ratio * (end - start)
    current_penalty_keep_mask = 1.0 + ratio * (penalty_keep_mask - 1.0)

    # Linearly increase penalty_mask_correct from 0.0 up to its final value
    # Formula: start + ratio * (end - start)
    current_penalty_mask_correct = 0.0 + ratio * (penalty_mask_correct - 0.0)

    return current_penalty_keep_mask, current_penalty_mask_correct
```

### Step 3: Integrate the Scheduler into the Training Loop

Now, you must call this new function every iteration to get the current penalty values and pass them to your loss functions.

**Action:** Modify the main training loop (`while True:`) and the `estimate_loss` function.

1.  **Modify the main `while True:` loop:**

    ```python
    # In train.py
    
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0
    while True:
    
        # --- MODIFICATION START ---
        # Determine and set the learning rate and masking penalties for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        current_penalty_keep_mask, current_penalty_mask_correct = get_masking_penalties(iter_num)
        # --- MODIFICATION END ---
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            # --- MODIFICATION: Pass current penalties to estimate_loss ---
            losses = estimate_loss(current_penalty_keep_mask, current_penalty_mask_correct)
            # ...
        
        # ... (rest of the eval block) ...
    
        # forward backward update...
        for micro_step in range(gradient_accumulation_steps):
            # ...
            with ctx:
                logits, loss_from_model = model(X, Y)
                if model_type == 'diffusion':
                    # --- MODIFICATION: Use dynamic penalties in the loss calculation ---
                    loss = diffusion_loss_function(logits, Y, X, mask_token_id, current_penalty_keep_mask, current_penalty_mask_correct)
                else:
                    loss = loss_from_model
                loss = loss / gradient_accumulation_steps
            # ... (rest of the forward/backward block) ...
    
        # ... (rest of the loop) ...
    ```

2.  **Modify the `estimate_loss` function signature and its call to the loss function:**

    ```python
    # In train.py

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    # --- MODIFICATION: Add penalties as arguments ---
    def estimate_loss(current_penalty_keep_mask=penalty_keep_mask, current_penalty_mask_correct=penalty_mask_correct):
        global fixed_val_batches
        out = {}
        model.eval()
        for split in ['train', 'val']:
            # ... (logic to get/create batches) ...
            for k in range(eval_iters):
                # ... (get X, Y) ...
                with ctx:
                    logits, loss_from_model = model(X, Y)
                    if model_type == 'diffusion':
                        # --- MODIFICATION: Use penalty arguments in the loss calculation ---
                        loss = diffusion_loss_function(logits, Y, X, mask_token_id, current_penalty_keep_mask, current_penalty_mask_correct)
                    else:
                        loss = loss_from_model
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    ```
 