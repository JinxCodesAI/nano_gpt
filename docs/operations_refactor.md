

### The Goal: A Declarative and Absolute Scaling System

Our objective is to completely remove the intermediate layer of multipliers and divisors (`lr_multiplier`, `n_layer_divisor`, etc.) and have the scaling schedule directly set the absolute target values for the model's architecture and training hyperparameters.

*   **Before:** `{"name": "change_lr", "value": 0.5}` (Multiply current LR by 0.5)
*   **After:** `{"name": "set_lr", "value": 5e-4}` (Set the LR to exactly 5e-4)

*   **Before:** `{"name": "stack_layers", "value": 2}` (Duplicate all layers once)
*   **After:** `{"name": "stack_layers", "value": [0, 0, 1, 1, 1]}` (Create a new 5-layer model by taking two copies of original layer 0 and three copies of original layer 1)

This change will make your `scaling_schedule.json` a much more powerful and explicit blueprint for your training run.

---

### Step-by-Step Refactoring Guide

#### Step 1: Purge All Relative State Variables from `train.py`

The first step is to remove the entire system of multipliers and divisors. This will immediately simplify the global state.

**In `train.py`, locate the "Dynamic State Parameters" section (around line 100) and delete all the multiplier/divisor variables.**

```python
# In train.py

# -----------------------------------------------------------------------------
# DYNAMIC STATE PARAMETERS (TO BE DELETED)
# --- DELETE ALL OF THESE LINES ---
# attn_lora_rank_divisor = 0 
# vocab_lora_rank_divisor = 0
# lora_alpha_multiplier = 1.0 
# n_layer_divisor = 1
# n_hidden_divisor = 1
# batch_size_multiplier = 1.0
# grad_accum_multiplier = 1.0
# lr_multiplier = 1.0
# warmup_iters_multiplier = 1.0
# eval_iters_multiplier = 1.0
# eval_interval_multiplier = 1.0 
# -----------------------------------------------------------------------------

# --- ALSO DELETE THE INITIALIZATION LOGIC THAT USES DIVISORS ---
# DELETE THIS BLOCK:
# if n_hidden_divisor is not None and n_hidden_divisor>0:
#     if n_hidden is not None:
#         n_hidden = n_hidden // n_hidden_divisor
#     else:
#         n_hidden = 4 * n_embd // n_hidden_divisor
#
# if n_layer_divisor is not None and n_layer_divisor>0:
#     n_layer = n_layer // n_layer_divisor
```

With these variables gone, you also need to update the few places they were used.

1.  **Simplify `get_lr`:** This function no longer needs a multiplier.
    ```python
    # In train.py, modify get_lr
    def get_lr(it):
        effective_it = it - lr_schedule_offset
        # REMOVED: Multipliers are gone
        actual_warmup_iters = warmup_iters 
        actual_lr_decay_iters = lr_decay_iters
        
        # ... (rest of the logic is fine, just remove '* lr_multiplier' from return statements) ...
        if effective_it < actual_warmup_iters:
            # REMOVED multiplier
            return learning_rate * (effective_it + 1) / (actual_warmup_iters + 1)
        if effective_it > actual_lr_decay_iters:
            # REMOVED multiplier
            return min_lr 
        # ...
        # REMOVED multiplier
        return (min_lr + coeff * (learning_rate - min_lr))
    ```

2.  **Simplify `estimate_loss` and the main loop eval check:**
    ```python
    # In train.py, inside estimate_loss()
    # actual_eval_iters = int(eval_iters * eval_iters_multiplier) # OLD
    actual_eval_iters = eval_iters # NEW
    
    # In train.py, inside the main training loop (while True:)
    # if iter_num % actual_eval_interval == 0: # OLD with actual_eval_interval calculation
    if iter_num % eval_interval == 0: # NEW and simple
    ```

#### Step 2: Refactor Architectural Operations in `model.py` to be Absolute

Now, let's modify the model's methods to accept absolute values and perform strict validation.

1.  **Modify `widen_mlp` to accept a target dimension.**
    
    **In `model.py`, replace the `widen_mlp` method with this:**
    ```python
    # In model.py, inside the GPT class
    def widen_mlp(self, new_hidden_dim):
        """
        Widens MLP layers to a new absolute dimension using Net2WiderNet.
        Throws a ValueError if the new dimension is not larger than the current one.
        """
        print(f"Widening MLP hidden dimension to {new_hidden_dim}.")
        
        # Validation is the first step
        original_hidden_dim = self.config.n_hidden if self.config.n_hidden is not None else 4 * self.config.n_embd
        if not new_hidden_dim > original_hidden_dim:
            raise ValueError(f"New hidden dimension ({new_hidden_dim}) must be greater than the original ({original_hidden_dim}).")

        for block in self.transformer.h:
            mlp = block.mlp
            w_fc, b_fc, w_proj = mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight
            device = w_fc.device

            new_c_fc = nn.Linear(self.config.n_embd, new_hidden_dim, bias=self.config.bias).to(device)
            new_c_proj = nn.Linear(new_hidden_dim, self.config.n_embd, bias=self.config.bias).to(device)
            
            # Net2WiderNet mapping
            mapping = torch.randint(0, original_hidden_dim, (new_hidden_dim,), device=device)
            mapping[:original_hidden_dim] = torch.arange(original_hidden_dim, device=device)
            
            # Copy weights for the first part of the mapping
            new_c_fc.weight.data.copy_(w_fc.data[mapping])
            if b_fc is not None:
                new_c_fc.bias.data.copy_(b_fc.data[mapping])
            
            # Add noise to break symmetry for the new neurons
            noise = torch.randn_like(new_c_fc.weight.data[original_hidden_dim:]) * 1e-4
            new_c_fc.weight.data[original_hidden_dim:] += noise

            # Calculate replication factors for adjusting the output layer
            replication_factors = torch.zeros(original_hidden_dim, device=device)
            for i in range(original_hidden_dim):
                replication_factors[i] = (mapping == i).sum()
            
            # Copy and scale the output projection weights
            new_c_proj.weight.data.copy_(w_proj.data[:, mapping])
            new_c_proj.weight.data /= replication_factors[mapping].view(1, -1)
            
            mlp.c_fc = new_c_fc
            mlp.c_proj = new_c_proj
            
        self.config.n_hidden = new_hidden_dim
        print(f"MLP hidden dimension successfully widened to {new_hidden_dim}.")
    ```

2.  **Modify `stack_layers` to use an explicit layer map.**
    
    **In `model.py`, replace the `stack_layers` method with this:**
    ```python
    # In model.py, inside the GPT class
    def stack_layers(self, layer_map):
        """
        Reconstructs the transformer stack based on an explicit layer map.
        The map is a list of source indices. e.g., [0, 0, 1] creates a 3-layer model
        with two copies of the original layer 0 and one copy of the original layer 1.
        
        Throws a ValueError if any source index is out of bounds.
        """
        print(f"Re-stacking layers based on map: {layer_map}. New depth will be {len(layer_map)}.")
        original_n_layer = self.config.n_layer
        
        # --- Validation First ---
        if not layer_map: # Cannot stack to zero layers this way
            raise ValueError("Layer map cannot be empty.")
        if max(layer_map) >= original_n_layer:
            raise ValueError(f"Invalid layer map: index {max(layer_map)} is out of bounds for current model with {original_n_layer} layers.")

        # Deepcopy original layers to use as a clean source palette
        original_layers = copy.deepcopy(self.transformer.h)
        new_layers = nn.ModuleList()
        
        # Build the new stack layer by layer
        for source_idx in layer_map:
            new_layers.append(copy.deepcopy(original_layers[source_idx]))
            
        self.transformer.h = new_layers
        self.config.n_layer = len(new_layers)
        print(f"Model now has {self.config.n_layer} layers.")
    ```

#### Step 3: Overhaul the `execute_operation` Orchestrator in `train.py`

This is the biggest change. We'll rewrite the function to handle absolute values and new operation names. We will also add robust error handling.

**In `train.py`, replace the entire `execute_operation` function with the following:**
```python
# In train.py

def execute_operation(op, trigger_reason, current_val_loss, iter_num, target_architecture_config):
    # Make globals mutable within this function
    global learning_rate, batch_size, gradient_accumulation_steps, warmup_iters, eval_iters, eval_interval
    global lr_schedule_offset, training_logger, master_process, model, optimizer, raw_model, unoptimized_model
    
    op_desc = op.get('desc', '')
    op_name = op['name']
    op_label = f"{op_name} {op_desc}"
    op_value = op['value']
    
    if master_process:
        print(f"\n--- EXECUTING OPERATION: {op_label} | Value: {op_value} ---")
        log_details = {
            'trigger_reason': trigger_reason,
            'current_val_loss': current_val_loss,
            'trigger_loss': op['trigger_loss'],
            'max_wait_iters': op['max_wait_iters']
        }
        training_logger.log_operation_start(iter_num, op_label, op_value, log_details)
        
    try:
        # Check if this is an architectural operation
        architectural_ops = ['stack_layers', 'widen_mlp', 'set_attn_lora_rank', 
                             'set_embedding_lora_rank', 'merge_lora_weights']
        
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
            elif op_name == 'set_attn_lora_rank':
                unwrapped_model.resize_lora_rank(op_value)
            elif op_name == 'set_embedding_lora_rank':
                unwrapped_model.resize_embedding_rank(op_value)
            elif op_name == 'merge_lora_weights':
                unwrapped_model.merge_lora_weights()
            
            # --- Re-create optimizer and wrappers (this logic remains the same) ---
            log_detailed_params(unwrapped_model)
            if master_process: print("Re-configuring optimizer...")
            optimizer = unwrapped_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
            if master_process: print("Transferring optimizer state...")
            transfer_optimizer_state(optimizer, old_optimizer_state, old_param_dict, unwrapped_model)
            
            model = unwrapped_model
            if compile:
                if master_process: print("Re-compiling the model...")
                model = torch.compile(model)
            if ddp:
                if master_process: print("Re-wrapping model in DDP...")
                model = DDP(model, device_ids=[ddp_local_rank])
            
            raw_model = model.module if ddp else model
            log_model_architecture(raw_model, iter_num)
            if master_process: training_logger.log_operation_success(iter_num, op_name, {'new_config': raw_model.config.__dict__})

        # --- Handle non-architectural (hyperparameter) operations ---
        else:
            if op_name == 'set_lr':
                if master_process: print(f"Learning rate: {learning_rate:.6f} -> {op_value:.6f}")
                learning_rate = op_value
            elif op_name == 'set_batch_size':
                if master_process: print(f"Batch size: {batch_size} -> {op_value}")
                batch_size = op_value
            elif op_name == 'set_grad_accum':
                if master_process: print(f"Grad accum steps: {gradient_accumulation_steps} -> {op_value}")
                gradient_accumulation_steps = op_value
            elif op_name == 'set_warmup_iters':
                if master_process: print(f"Warmup iters: {warmup_iters} -> {op_value}")
                warmup_iters = op_value
            elif op_name == 'set_eval_iters':
                if master_process: print(f"Eval iters: {eval_iters} -> {op_value}")
                eval_iters = op_value
            elif op_name == 'set_eval_interval':
                if master_process: print(f"Eval interval: {eval_interval} -> {op_value}")
                eval_interval = op_value
            elif op_name == 'reset_lr_schedule':
                if master_process: print(f"Resetting LR schedule at iter {iter_num}")
                lr_schedule_offset = iter_num
            else:
                raise ValueError(f"Unknown operation '{op_name}'")
            if master_process: training_logger.log_operation_success(iter_num, op_name, {'new_value': op_value})

        return True

    except ValueError as e:
        # Catch validation errors from the model methods (e.g., widening to smaller dim)
        error_msg = f"Operation '{op_name}' failed validation: {e}"
        if master_process:
            print(f"ERROR: {error_msg}")
            training_logger.log_operation_error(iter_num, op_name, error_msg)
        # We will not mark this operation as complete and let the program exit or continue
        # depending on your desired failure mode. For safety, we return False.
        # Consider adding `sys.exit(1)` if failure should be fatal.
        return False
```

#### Step 4: Update Your Schedule Configuration File

Your schedule files now need to use the new absolute operation names and values. Here is a template based on your examples.

**Example `scaling_schedule.json`:**
```json
[
  {
    "name": "set_lr",
    "desc": "Initial learning rate",
    "value": 6e-4,
    "trigger_loss": 100.0,
    "max_wait_iters": 1,
    "reevaluate": false
  },
  {
    "name": "stack_layers",
    "desc": "Grow to 4 layers from initial 2",
    "value": [0, 0, 1, 1],
    "trigger_loss": 3.5,
    "max_wait_iters": 500,
    "reevaluate": true
  },
  {
    "name": "widen_mlp",
    "desc": "Widen MLP from 64 to 128",
    "value": 128,
    "trigger_loss": 3.2,
    "max_wait_iters": 500,
    "reevaluate": true
  },
  {
    "name": "set_attn_lora_rank",
    "desc": "Enable LoRA fine-tuning for attention",
    "value": 8,
    "trigger_loss": 2.8,
    "max_wait_iters": 1000,
    "reevaluate": true
  },
    {
    "name": "stack_layers",
    "desc": "Discard layer 0, keep only grown layers",
    "value": [1, 1, 2, 3],
    "trigger_loss": 2.5,
    "max_wait_iters": 1000,
    "reevaluate": true
  }
]
```

### Summary of Benefits

By completing this refactor, you have achieved a significantly more robust and intuitive system:

1.  **Clarity:** The code is simpler. There are no confusing multipliers to track. The `execute_operation` function directly sets state.
2.  **Declarative Configuration:** Your schedule file is now a human-readable plan that explicitly states the target architecture and hyperparameters at each stage.
3.  **Robustness:** The strict validation in the model methods (`stack_layers`, `widen_mlp`) prevents illegal architectural changes and provides clear error messages, failing fast instead of producing unexpected behavior.
4.  **Flexibility:** The new `stack_layers` is incredibly powerful, allowing not just for growth but also for selective pruning and reordering of layers, enabling much more sophisticated architectural experiments.