Of course. This is the perfect next step. A clear, actionable plan is essential.

This `implementation_plan.md` is designed to be a comprehensive, step-by-step guide for a skilled Python developer who may not be an expert in PyTorch or machine learning. It focuses on the "what" and "how," with clear code snippets and explanations for each change, translating the high-level concepts from our previous discussions into concrete actions.

---

# `implementation_plan.md`

## Subject: Step-by-Step Refactoring Guide for the Diffusion Model

### 1. Introduction

This document provides a detailed implementation plan to refactor the existing codebase into a new, robust diffusion-style language model. The core of this refactoring is an architectural shift from a single, ambiguous `[MASK]` token to a dual-token system (`[MASK]` and `[WRONG]`) to resolve fundamental learning instabilities.

This guide is structured as a series of sequential, isolated steps. Each step includes a rationale ("Why are we doing this?") and a clear set of actions with code snippets.

### 2. Pre-computation and Setup

Before we modify the training logic, we must update the model's architecture to support the new two-token system.

#### Step 2.1: Update the Model Configuration
*   **Why:** The model needs to be aware of two new special tokens, not just one. We will add IDs for both `[MASK]` and `[WRONG]`.
*   **Action:** In `model.py`, modify the `GPTConfig` dataclass.

```python
# In model.py

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # --- MODIFICATION: Replace single mask_token_id with two new ones ---
    model_type: str = 'gpt2'
    mask_token_id: int = None # The "unknown" token
    wrong_token_id: int = None # The "incorrect" token
```

### 3. Training Script (`train.py`) Refactoring

We will now implement the new training process, starting with data generation and moving to the loss function.

#### Step 3.1: Add New Configuration Hyperparameters
*   **Why:** Our new design requires a simpler set of penalties and a new curriculum scheduler. We will replace the old, complex parameters.
*   **Action:** In `train.py`, find the configuration block at the top of the file and replace the old "Diffusion loss specific penalties" with this new, cleaner set of parameters.

```python
# In train.py (configuration section)

# ... (model parameters) ...
model_type = 'gpt2' # 'gpt2' or 'diffusion'

# --- NEW SIMPLIFIED LOSS CONFIG ---
# Task-based loss weights
weight_unmask_task = 1.0        # Weight for the "fill-in-the-blank" task.
weight_remask_task = 1.0        # Weight for the "proofreading" task.

# Curriculum settings
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # Iterations to ramp up the penalty_mask_correct.
proofreading_warmup_iters = 2000 # Iterations to ramp up the "re-masking" task.

guaranteed_correct_factor = 0.01
# ...
```

#### Step 3.2: Implement the New Curriculum Scheduler
*   **Why:** We need a single function to control the gradual introduction of both the re-masking task and its associated penalties.
*   **Action:** In `train.py`, **delete** the old `diffusion_loss_function`. Then, add this new scheduler function near the top of the file.

```python
# In train.py

def get_corruption_scheduler(it):
    """
    Controls the curriculum for both penalties and data generation.
    Returns the current penalty values and the re-masking task ratio.
    """
    # --- Penalty Curriculum for "destructive editing" ---
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
```

#### Step 3.3: Implement the New, Simplified Loss Function
*   **Why:** This is the core of the refactoring. We are replacing the old, flawed, `argmax`-based loss function with a new one that is simpler and correctly uses the `[WRONG]` token.
*   **Action:** In `train.py`, add this new loss function.

```python
# In train.py

def calculate_diffusion_loss(logits, targets, inputs, mask_token_id, wrong_token_id,
                             current_penalty_mask_correct, weight_unmask_task, 
                             weight_remask_task, log_diagnostics=False):
    """
    A simplified and robust loss function for the two-token system.
    It applies weights based only on the task type (the Input-Target pair).
    """
    # --- Step 1: Calculate the base cross-entropy loss ---
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='none')

    # --- Step 2: Apply weights based on the TASK TYPE only ---
    weights = torch.ones_like(per_token_loss)
    flat_inputs = inputs.view(-1)
    
    # Task 1: Un-masking (Input was [MASK], Target is a word)
    unmask_task_positions = (flat_inputs == mask_token_id) & (flat_targets != wrong_token_id)
    weights[unmask_task_positions] = weight_unmask_task
    
    # Task 2: Re-masking (Input was a word, Target is [WRONG])
    remask_task_positions = (flat_inputs != mask_token_id) & (flat_targets == wrong_token_id)
    weights[remask_task_positions] = weight_remask_task
    
    # Task 3: State-Dependent Penalty for Destructive Edits
    with torch.no_grad():
        B, T = inputs.shape
        predictions_2d = torch.argmax(logits, dim=-1)
        for i in range(B):
            epsilon = 1e-6
            input_words_mask = (inputs[i] != mask_token_id)
            num_input_words = input_words_mask.sum().item()
            corrupted_words_mask = (targets[i, input_words_mask] == wrong_token_id)
            num_corrupted_words = corrupted_words_mask.sum().item()
            corruption_rate = num_corrupted_words / (num_input_words + epsilon)
            effective_penalty = current_penalty_mask_correct * (1.0 - corruption_rate)
            
            # Find positions where a correct word was wrongly changed to [WRONG]
            destructive_mask = (inputs[i] == targets[i]) & (inputs[i] != mask_token_id) & (predictions_2d[i] == wrong_token_id)
            
            weights_2d = weights.view(B, T)
            weights_2d[i, destructive_mask] = effective_penalty
            
    # The final loss is the weighted average
    final_loss = (per_token_loss * weights).mean()

    # (Optional: Add simplified logging here if desired)

    return final_loss
```

#### Step 3.4: Update the `get_batch` Function
*   **Why:** We need to implement the data curriculum and use the new `[WRONG]` token as a target.
*   **Action:** In `train.py`, replace the existing `get_batch` function with this one.

```python
# In train.py

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_clean = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if model_type == 'diffusion':
        assert mask_token_id is not None and wrong_token_id is not None, "Special tokens not initialized"
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()

        # Get the current re-masking task ratio from the curriculum scheduler
        _, remask_ratio = get_corruption_scheduler(iter_num)

        b, t = x_corrupted.shape
        for i in range(b):
            max_corruption = 1 - guaranteed_correct_factor
            rate_mask = torch.rand(1) * max_corruption
            rate_random = torch.rand(1) * rate_mask * remask_ratio
            rate_mask = rate_mask - rate_random

            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            x_corrupted[i, pos_mask] = mask_token_id
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            # --- MODIFICATION: The target for corrupted tokens is now [WRONG] ---
            y_target[i, pos_random] = wrong_token_id

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

#### Step 3.5: Update Model Initialization
*   **Why:** We need to handle a vocabulary size that is now `+2` larger and correctly initialize `mask_token_id` and `wrong_token_id`.
*   **Action:** In `train.py`, modify the `init_from = 'scratch'` block.

```python
# In train.py

# --- Make wrong_token_id a global variable like mask_token_id ---
mask_token_id = None
wrong_token_id = None

# ... (inside the `init_from = 'scratch'` block)
if model_type == 'diffusion':
    # Vocab size is now +2 for [MASK] and [WRONG]
    vocab_size = meta_vocab_size + 2 if meta_vocab_size is not None else 50306
    model_args['vocab_size'] = vocab_size
    # Assign the last two token IDs
    model_args['mask_token_id'] = vocab_size - 2
    model_args['wrong_token_id'] = vocab_size - 1
    # Make them globally available to get_batch
    global mask_token_id, wrong_token_id
    mask_token_id = model_args['mask_token_id']
    wrong_token_id = model_args['wrong_token_id']
else:
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
```
*(Remember to also update the `init_from = 'resume'` block to handle loading these two new `_token_id` values from checkpoints.)*

#### Step 3.6: Update the Main Training Loop
*   **Why:** We need to call our new scheduler and our new loss function.
*   **Action:** In `train.py`, modify the main `while True:` loop.

```python
# In train.py

# ... (inside the `while True:` loop)
# --- MODIFICATION: Use the new scheduler ---
lr = get_lr(iter_num) if decay_lr else learning_rate
current_penalty_mask_correct, _ = get_corruption_scheduler(iter_num)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
# ...
# (inside the `for micro_step...` loop)
with ctx:
    logits, loss_from_model = model(X, Y)
    if model_type == 'diffusion':
        # --- MODIFICATION: Call the new, simpler loss function ---
        loss = calculate_diffusion_loss(
            logits, Y, X, mask_token_id, wrong_token_id,
            current_penalty_mask_correct,
            weight_unmask_task, weight_remask_task
        )
    else:
        loss = loss_from_model
    loss = loss / gradient_accumulation_steps
# ...
```

### 4. Inference Script Refactoring

Finally, we need to update the inference logic to handle the new two-token system.

#### Step 4.1: Update `generate_diffusion` in `model.py`
*   **Why:** The inference loop needs to implement the new state transition rule: when the model predicts `[WRONG]`, we must change that token to `[MASK]` for the next iteration.
*   **Action:** In `model.py`, **replace** the existing `generate_diffusion` method with this new version.

```python
# In model.py

    @torch.no_grad()
    def generate_diffusion(self, idx, max_steps, temperature=1.0, top_k=None):
        """
        Iteratively refines a sequence using the [MASK] / [WRONG] token system.
        """
        assert self.config.model_type == 'diffusion', "This is for diffusion models"
        mask_token_id = self.config.mask_token_id
        wrong_token_id = self.config.wrong_token_id
        assert mask_token_id is not None and wrong_token_id is not None, "Special tokens not configured."
        self.eval()

        for _ in range(max_steps):
            if not (idx == mask_token_id).any():
                break # Converged

            logits, _ = self(idx)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(idx.shape)

            # --- NEW STATE UPDATE LOGIC ---
            # Start with a copy of the current state
            new_idx = idx.clone()
            
            # 1. Update positions that were [MASK] with the model's new prediction.
            unmask_positions = (idx == mask_token_id)
            new_idx[unmask_positions] = idx_next[unmask_positions]
            
            # 2. Find where the model flagged an error and turn it into a [MASK] for the next round.
            remask_positions = (idx != mask_token_id) & (idx_next == wrong_token_id)
            new_idx[remask_positions] = mask_token_id
            
            idx = new_idx
            # --- END NEW LOGIC ---
        
        # Finalization: force any remaining [MASK] tokens to be something else
        if (idx == mask_token_id).any():
            logits, _ = self(idx)
            # Forbid both special tokens as a final answer
            logits[:, :, mask_token_id] = -float('Inf')
            logits[:, :, wrong_token_id] = -float('Inf')
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs.view(--1, probs.size(-1)), num_samples=1).view(idx.shape)
            mask = (idx == mask_token_id)
            idx = torch.where(mask, idx_next, idx)

        self.train()
        return idx
```

#### Step 4.2: Update `generate_diffusion.py`
*   **Why:** The sampling script is mostly correct, but it needs to be aware of the two special tokens to decode the output correctly.
*   **Action:** No direct code changes are needed in `generate_diffusion.py`, as it correctly loads the `GPTConfig` from the checkpoint. However, you must ensure that your `meta.pkl` file (or other decoding logic) is updated to include mappings for the two new special tokens so the `decode` function doesn't crash.

This completes the full refactoring plan. By following these steps, you will transition your project from the old, buggy single-token system to the new, robust, and architecturally sound two-token system.