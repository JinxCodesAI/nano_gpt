
### High-Level Goal: The "Grow-As-You-Go" Model

Think of this feature as a memory-saving trick. A significant portion of a language model's memory is used for its "dictionary" — the list of all possible words (tokens) it knows. For a large dictionary, this is very memory-intensive.

Our goal is to:
1.  **Start Small:** Begin training with a tiny, "shrunken" dictionary. All the rare words we're ignoring for now will be mapped to a single, special "unknown word" token (`RARE_TOKEN_ID`). This dramatically reduces the initial memory needed.
2.  **Grow Later:** Once the model has learned the basics of grammar and sentence structure using this small dictionary, we'll trigger an operation to "grow" its dictionary to the full size.
3.  **Fine-tune:** After the dictionary grows, we'll briefly focus the model's learning *only* on the new words to get them up to speed, without changing the core knowledge it has already gained.

This guide will walk you through modifying the two provided files, `train.py` (the main script that runs the training) and `model.py` (the definition of the GPT model itself), to achieve this.

---

### Step 1: Add New Settings to `train.py`

**Goal:** Inform the training script about the new options needed to control this feature.

**File to modify:** `train.py`

1.  **Locate the configuration section.** Find the block of code near the top of the file that starts with `out_dir = 'out'`.
2.  **Add the new configuration variables.** Add the following lines alongside the other settings.

    ```python
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    # Shrunken vocabulary training parameters
    shrunken_vocab_size = None # If set, enables training with a smaller vocab to save memory
    vocab_remapping_file = None # Path to .pt file with the remapping tensor, required if shrunken_vocab_size is set
    RARE_TOKEN_ID = None # The token ID in the shrunken vocab for all out-of-vocab tokens
    # -----------------------------------------------------------------------------
    # LoRA architectural parameters. These will be overridden by config files.
    embedding_mode = 'standard'
    ```

3.  **Make the new settings configurable.** Find the line `config_keys = [...]`. This list tells the program which variables can be set from the command line or a config file. Add our new variables to this list.

    ```python
    # ... (previous code) ...
    lora_alpha = 1.0 # scaling factor for LoRA layers
    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))] # Modified to include NoneType
    exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    ```
    *Developer Note: I've slightly modified the `config_keys` line to also include `None` values, which is good practice for your new optional parameters.*

### Step 2: Modify Model Initialization in `train.py`

**Goal:** When the program starts, if `shrunken_vocab_size` is active, create a model that is physically smaller.

**File to modify:** `train.py`

1.  **Locate the "model init" section.** Search for the comment `# model init`.
2.  **Adjust the vocabulary size.** We need to decide whether to tell the model to use the full vocabulary size or our new shrunken one. Modify the code as follows:

    ```python
    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout, n_hidden=n_hidden,
                    use_rotary_embeddings=use_rotary_embeddings,
                    rotary_base=rotary_base,
                    rotary_max_position_embeddings=rotary_max_position_embeddings,
                    # --- ADD THESE LINES ---
                    embedding_mode=embedding_mode,
                    embedding_rank=embedding_rank,
                    attn_lora_rank=attn_lora_rank,
                    lora_alpha=lora_alpha
                )
    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        # --- THIS BLOCK IS MODIFIED ---
        # Determine the active vocabulary size for model instantiation
        if shrunken_vocab_size is not None:
            print(f"Using shrunken vocabulary of size: {shrunken_vocab_size}")
            active_vocab_size = shrunken_vocab_size
        else:
            active_vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        
        model_args['vocab_size'] = active_vocab_size
        # --- END OF MODIFICATION ---
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
    ```
    *Developer Note: We are creating a new variable `active_vocab_size`. The model will now be created with this vocabulary size, which is how we achieve the memory savings.*

### Step 3: Implement Data Remapping in `train.py`

**Goal:** Load our "translation map" and use it to convert the training data on-the-fly to the smaller dictionary format.

**File to modify:** `train.py`

1.  **Load the remapping tensor.** Near the top of the file, after the DDP (Distributed Data Parallel) setup, add the logic to load the tensor from the file specified in the config.

    ```python
    # ... after DDP setup ...
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    # --- ADD THIS ENTIRE BLOCK ---
    # Vocabulary remapping setup
    remapping_vector = None
    remapping_active = False # Global flag to control remapping
    if master_process and shrunken_vocab_size is not None:
        if not vocab_remapping_file or not os.path.exists(vocab_remapping_file):
            raise ValueError("`shrunken_vocab_size` is set, but `vocab_remapping_file` is missing or invalid.")
        if RARE_TOKEN_ID is None:
            raise ValueError("`shrunken_vocab_size` is set, but `RARE_TOKEN_ID` is not.")
        
        print(f"Loading vocabulary remapping from {vocab_remapping_file}")
        remapping_vector = torch.load(vocab_remapping_file)
        remapping_active = True # Initially active if configured
    
    if ddp:
        # Broadcast the remapping_active flag and the tensor itself to all processes
        active_flag_tensor = torch.tensor([1.0 if remapping_active else 0.0], device=device)
        torch.distributed.broadcast(active_flag_tensor, src=0)
        remapping_active = active_flag_tensor.item() == 1.0

        if remapping_active:
            if ddp_rank != 0: # If not master, create a placeholder tensor
                remapping_vector = torch.zeros(meta['vocab_size'], dtype=torch.long)
            remapping_vector = remapping_vector.to(device)
            torch.distributed.broadcast(remapping_vector, src=0)
    elif remapping_vector is not None:
        remapping_vector = remapping_vector.to(device)
    # --- END OF ADDED BLOCK ---

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        training_logger.setup(config)
    ```

2.  **Apply the remapping in the training loop.** Find the main training loop (`while True:`). Inside it, find the `for micro_step...` loop. This is where we process each small batch of data. We'll add our translation logic here.

    ```python
    # ... inside while True loop ...
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        # --- ADD THIS BLOCK ---
        # Remap vocabulary on-the-fly if active
        if remapping_active:
            X, Y = remapping_vector[X], remapping_vector[Y]
        # --- END OF ADDED BLOCK ---
            
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train') # get the next batch
        scaler.scale(loss).backward()
    ```
    *Developer Note: We get a fresh batch (`X`, `Y`) at the end of the loop and remap it at the start of the next iteration. This ensures the original data remains clean.*

### Step 4: Add New Abilities to the Model in `model.py`

**Goal:** Teach the `GPT` class how to grow its vocabulary and how to enter the special "fine-tuning" mode.

**File to modify:** `model.py`

1.  **Add the `embedding_finetune_mode` flag.** In the `GPT` class `__init__` method, add a flag to track this state.

    ```python
    class GPT(nn.Module):
    
        def __init__(self, config):
            super().__init__()
            assert config.vocab_size is not None
            assert config.block_size is not None
            self.config = config
            self.embedding_finetune_mode = False # Add this line
    
            # ... rest of the __init__ method ...
    ```

2.  **Modify the optimizer configuration.** Find the `configure_optimizers` method in the `GPT` class. We need to add logic at the top to check our new flag. If it's `True`, we will only train the dictionary-related parts of the model.

    ```python
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # --- ADD THIS ENTIRE BLOCK ---
        # If in embedding finetune mode, only optimize the embedding and lm_head layers
        if self.embedding_finetune_mode:
            print("Optimizer configured for embedding fine-tuning mode.")
            param_dict = {pn: p for pn, p in param_dict.items() if 'wte' in pn or 'lm_head' in pn}
        # --- END OF ADDED BLOCK ---

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    ```

3.  **Add the new architectural methods.** At the end of the `GPT` class, after the `generate` method but before `stack_layers`, add the two new methods for resizing the vocabulary and setting the finetune mode.

    ```python
    # ... after generate() method ...
    @torch.no_grad()
    def get_merged_state_dict(self):
        # ... existing method body ...
        return final_sd

    def resize_vocabulary(self, new_vocab_size, source_token_id, noise_std=0.01):
        """
        Grows the vocabulary size of the model in a function-preserving way.
        New token embeddings are initialized from a source token plus noise.
        """
        print(f"Resizing vocabulary from {self.config.vocab_size} to {new_vocab_size}.")
        if new_vocab_size <= self.config.vocab_size:
            raise ValueError("New vocabulary size must be larger than the current one.")

        old_wte = self.transformer.wte
        old_lm_head = self.lm_head
        device = old_wte.weight.device
        
        # Create new, larger layers
        self.transformer.wte = nn.Embedding(new_vocab_size, self.config.n_embd).to(device)
        self.lm_head = nn.Linear(self.config.n_embd, new_vocab_size, bias=False).to(device)
        
        # Copy old weights
        self.transformer.wte.weight.data[:self.config.vocab_size, :] = old_wte.weight.data
        self.lm_head.weight.data[:self.config.vocab_size, :] = old_lm_head.weight.data
        
        # Initialize new weights from the source token
        source_embedding = old_wte.weight.data[source_token_id, :]
        source_lm_head = old_lm_head.weight.data[source_token_id, :]
        
        # Add noise to break symmetry
        noise_embedding = torch.randn(new_vocab_size - self.config.vocab_size, self.config.n_embd, device=device) * noise_std
        noise_lm_head = torch.randn(new_vocab_size - self.config.vocab_size, self.config.n_embd, device=device) * noise_std
        
        self.transformer.wte.weight.data[self.config.vocab_size:, :] = source_embedding + noise_embedding
        self.lm_head.weight.data[self.config.vocab_size:, :] = source_lm_head + noise_lm_head
        
        # Update config and re-tie weights
        self.config.vocab_size = new_vocab_size
        self.transformer.wte.weight = self.lm_head.weight
        
        print("Vocabulary resized successfully.")

    def set_embedding_finetune_mode(self, enabled: bool):
        """
        Sets the model to fine-tune only embedding-related layers.
        When enabled, all parameters except wte and lm_head are frozen.
        """
        self.embedding_finetune_mode = enabled
        
        # Freeze or unfreeze the model backbone
        for name, param in self.named_parameters():
            if 'wte' not in name and 'lm_head' not in name:
                param.requires_grad = not enabled
                
        status = "ENABLED" if enabled else "DISABLED"
        print(f"Embedding fine-tuning mode is now {status}.")

    def stack_layers(self, layer_map):
    # ... rest of the file ...
    ```

### Step 5: Wire Up New Operations in `train.py`

**Goal:** Allow the scaling schedule to call the new model functions we just wrote.

**File to modify:** `train.py`

1.  **Locate the `execute_operation` function.** This function is the "control panel" for all scheduled tasks.
2.  **Update the function** to handle our three new operations: `resize_vocabulary`, `set_embedding_finetune_mode`, and `disable_vocab_remapping`.

    ```python
    def execute_operation(op, trigger_reason, current_val_loss, iter_num, target_architecture_config):
        # Make globals mutable within this function
        # --- MODIFICATION ---
        global learning_rate, batch_size, gradient_accumulation_steps, warmup_iters, eval_iters, eval_interval
        global lr_schedule_offset, training_logger, master_process, model, optimizer, raw_model, unoptimized_model
        global remapping_active # Add this global
        # --- END MODIFICATION ---

        # ... (logging code) ...

        try:
            # Check if this is an architectural operation
            # --- MODIFICATION ---
            architectural_ops = ['stack_layers', 'widen_mlp', 'set_attn_lora_rank',
                                 'set_embedding_lora_rank', 'merge_lora_weights',
                                 'resize_vocabulary', 'set_embedding_finetune_mode']
            # --- END MODIFICATION ---

            if op_name in architectural_ops:
                if master_process:
                    print(f"Performing architectural operation: {op_name}")

                unwrapped_model = unoptimized_model if compile else (model.module if ddp else model)
                old_optimizer_state = optimizer.state_dict()
                old_param_dict = {name: p for name, p in unwrapped_model.named_parameters()}

                # --- Perform the absolute architectural operation ---
                if op_name == 'stack_layers':
                    unwrapped_model.stack_layers(op_value)
                elif op_name == 'widen_mlp':
                    unwrapped_model.widen_mlp(op_value)
                # ... (other architectural ops) ...
                elif op_name == 'merge_lora_weights':
                    unwrapped_model.merge_lora_weights()
                # --- ADD THESE BLOCKS ---
                elif op_name == 'resize_vocabulary':
                    # op_value is expected to be [source_token_id, noise_std]
                    source_token_id, noise_std = op_value
                    # The target vocab size is derived from the meta file
                    meta_path = os.path.join(data_dir, 'meta.pkl')
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    full_vocab_size = meta['vocab_size']
                    unwrapped_model.resize_vocabulary(full_vocab_size, source_token_id, noise_std)
                elif op_name == 'set_embedding_finetune_mode':
                    # op_value is expected to be True or False
                    unwrapped_model.set_embedding_finetune_mode(op_value)
                # --- END OF ADDED BLOCKS ---

                # --- Re-create optimizer and wrappers (this logic remains the same) ---
                # ... (rest of the if block) ...

            # --- Handle non-architectural (hyperparameter) operations ---
            else:
                if op_name == 'set_lr':
                    # ... (existing operations) ...
                elif op_name == 'reset_lr_schedule':
                    if master_process: print(f"Resetting LR schedule at iter {iter_num}")
                    lr_schedule_offset = iter_num
                # --- ADD THIS BLOCK ---
                elif op_name == 'disable_vocab_remapping':
                    if master_process: print("Disabling vocabulary remapping.")
                    remapping_active = False
                # --- END OF ADDED BLOCK ---
                else:
                    raise ValueError(f"Unknown operation '{op_name}'")
                if master_process: training_logger.log_operation_success(iter_num, op_name, {'new_value': op_value})
            
            return True
    ```

### Step 6: Update Validation Metrics in `train.py`

**Goal:** Calculate and log our new "Core Accuracy" metric to get a true sense of performance during the shrunken vocabulary phase.

**File to modify:** `train.py`

1.  **Locate the `estimate_loss` function.**
2.  **Modify it to calculate accuracy.** We'll add logic to check if remapping is active and, if so, calculate accuracy only on the "core" (non-rare) tokens.

    ```python
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            # --- ADD THIS ---
            core_token_correct = 0
            core_token_total = 0
            # --- END ADD ---

            for k in range(eval_iters):
                X, Y = get_batch(split)
                
                # --- THIS ENTIRE BLOCK IS MODIFIED ---
                # If remapping is active, we need to handle metrics carefully
                if remapping_active:
                    remapped_X = remapping_vector[X]
                    remapped_Y = remapping_vector[Y]
                    with ctx:
                        logits, loss = model(remapped_X, remapped_Y)
                    
                    # Calculate Core Accuracy on the validation set
                    if split == 'val':
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        preds = torch.argmax(probs, dim=-1)
                        
                        # Mask to ignore the RARE_TOKEN_ID in accuracy calculation
                        is_core_token = (remapped_Y != RARE_TOKEN_ID)
                        
                        core_token_correct += ((preds == remapped_Y) & is_core_token).sum().item()
                        core_token_total += is_core_token.sum().item()
                else:
                    # Standard loss calculation
                    with ctx:
                        logits, loss = model(X, Y)
                # --- END OF MODIFICATION ---
                        
                losses[k] = loss.item()

            out[split] = losses.mean()
            # --- ADD THIS ---
            if remapping_active and core_token_total > 0 and split == 'val':
                out['val_core_acc'] = core_token_correct / core_token_total
            # --- END ADD ---

        model.train()
        return out
    ```
3. **Log the new metric.** Find the `if iter_num % eval_interval == 0:` block and update the `wandb.log` call to include the new metric.

   ```python
   # ... inside if iter_num % eval_interval ...
   if wandb_log:
       elapsed_time_seconds = time.time() - start_time
       wandb_metrics = {
           "iter": iter_num,
           "train/loss": losses['train'],
           "val/loss": losses['val'],
           "lr": lr,
           "mfu": running_mfu*100,
           "time/elapsed_seconds": elapsed_time_seconds, # Log elapsed time
       }
       # --- ADD THIS BLOCK ---
       if 'val_core_acc' in losses:
           wandb_metrics["val/core_accuracy"] = losses['val_core_acc']
       # --- END ADD ---
   ```
