# Making the Validation Set Fixed and Reproducible

## 1. The Goal

The objective is to modify the validation process to be deterministic. Instead of generating a new, random set of validation tasks every time `estimate_loss()` is called, we will pre-generate a single, fixed set of validation tasks at the start of the training script. This ensures that the validation loss is always computed on the exact same data, making results from different experiments directly comparable.

To ensure this fixed set is comprehensive, it will contain `eval_iters` batches with a linearly increasing level of difficulty, from very low corruption to very high corruption.

## 2. The Plan

1.  **Create a New Function (`create_fixed_validation_set`)**: This function will run only once. It will read from `val.bin`, generate `eval_iters` batches of data (`X` and `Y`), and apply the new linearly increasing corruption scheme.
2.  **Store the Fixed Set**: The generated batches will be stored in a list in memory.
3.  **Update `estimate_loss()`**: The validation part of this function will be changed to iterate over the pre-generated list of batches instead of calling `get_batch('val')`.
4.  **Simplify `get_batch()`**: The logic for handling the `'val'` split inside `get_batch()` is now redundant and will be removed.

## 3. Step-by-Step Implementation

Follow these instructions to modify your `train.py` file.

### Step 1: Create the Pre-generation Function

First, we need the function that builds our fixed validation set.

**Action:** Add the following new function to your `train.py` script. A good place is right after the `get_batch` function definition.

```python
# In train.py

def create_fixed_validation_set():
    """
    Creates a fixed set of validation tasks with linearly increasing difficulty.
    This function is run once at the start of training.
    """
    print("Creating fixed validation set...")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # Use a fixed seed to ensure the same text snippets are chosen every time
    torch.manual_seed(1337)

    val_batches = []
    for k in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x_clean = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
        
        # --- Create corrupted X and target Y for this batch ---
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()
        b, t = x_corrupted.shape

        # Linearly interpolate the total corruption rate
        # Start with low corruption (high guaranteed_correct_factor)
        # End with high corruption (low guaranteed_correct_factor)
        progress = k / (eval_iters - 1) if eval_iters > 1 else 1.0
        start_corruption = guaranteed_correct_factor
        end_corruption = 1.0 - guaranteed_correct_factor
        current_max_corruption = start_corruption + progress * (end_corruption - start_corruption)

        for i in range(b):
            rate_mask = torch.rand(1) * current_max_corruption
            rate_random = torch.rand(1) * (current_max_corruption - rate_mask)
            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            x_corrupted[i, pos_mask] = mask_token_id
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            y_target[i, pos_random] = mask_token_id

        # Move to the correct device
        if device_type == 'cuda':
            x_corrupted = x_corrupted.pin_memory().to(device, non_blocking=True)
            y_target = y_target.pin_memory().to(device, non_blocking=True)
        else:
            x_corrupted, y_target = x_corrupted.to(device), y_target.to(device)
        
        val_batches.append((x_corrupted, y_target))
    
    # Reset the seed to not affect subsequent operations
    torch.manual_seed(1337 + seed_offset)
    print("Fixed validation set created.")
    return val_batches
```

### Step 2: Update `get_batch()`

Since the validation data is now pre-generated, we can remove the logic for the `'val'` split from the `get_batch()` function. This makes it simpler and solely responsible for generating training data.

**Action:** Replace your current `get_batch` function with this simplified version.

```python
# In train.py

def get_batch(split):
    # We only ever call this for the 'train' split now.
    # The 'val' split is handled by the pre-generated fixed set.
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_clean = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if model_type == 'diffusion':
        # ... (The existing diffusion corruption logic for training data remains unchanged) ...
        assert mask_token_id is not None, "mask_token_id must be set globally"
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()

        b, t = x_corrupted.shape
        for i in range(b):
            max_corruption = 1 - guaranteed_correct_factor
            rate_mask = torch.rand(1) * max_corruption
            rate_random = torch.rand(1) * (max_corruption - rate_mask)
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

### Step 3: Generate the Fixed Set and Update `estimate_loss()`

Now we will call our new function once and change `estimate_loss` to use the result.

**Action:**
1.  **Add a global variable** `fixed_val_batches = None` near the top of the script.
2.  **Call the creation function** right before the main training loop begins.
3.  **Modify `estimate_loss`** to use this fixed data.

```python
# In train.py

# ... (after model initialization and DDP wrapping) ...

# THIS IS THE NEW GLOBAL VARIABLE
fixed_val_batches = None

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    global fixed_val_batches # Use the global variable
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'val' and model_type == 'diffusion':
            # --- MODIFICATION: Create the fixed set if it doesn't exist ---
            if fixed_val_batches is None:
                fixed_val_batches = create_fixed_validation_set()
            # --- END MODIFICATION ---

        for k in range(eval_iters):
            # --- MODIFICATION: Use the correct data source ---
            if split == 'val' and model_type == 'diffusion':
                X, Y = fixed_val_batches[k]
            else:
                # For 'train' split or gpt2 mode, get a random batch as before
                X, Y = get_batch(split)
            # --- END MODIFICATION ---

            with ctx:
                logits, loss_from_model = model(X, Y)
                if model_type == 'diffusion':
                    loss = diffusion_loss_function(logits, Y, X, mask_token_id, penalty_keep_mask, penalty_mask_correct)
                else:
                    loss = loss_from_model
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ... (just before the `while True:` training loop starts) ...
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:
# ... (rest of the script)
```

With these changes, your script will now produce a comparable and robust validation loss metric for your diffusion model experiments.