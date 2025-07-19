# Functional Specification: Dynamic Training Orchestrator

This document outlines the functional specification and implementation plan for a dynamic training orchestrator. The system will adjust a GPT model's architecture and hyperparameters during pre-training based on a predefined schedule triggered by performance milestones.

---

### 1. High-Level Goal

The primary objective is to improve the efficiency of LLM pre-training. Instead of training a large, static model from scratch, we will implement a curriculum-based approach where the model begins in a smaller, more parameter-efficient state. As the model learns and its validation loss improves, the orchestrator will trigger operations to progressively increase its capacity and adjust training parameters.

This strategy aims to:
* **Accelerate Convergence:** Achieve target performance levels using less compute time and fewer training tokens.
* **Improve Efficiency:** Reduce the overall computational resources required for pre-training.
* **Enable Experimentation:** Create a flexible framework to systematically test different growth strategies.

---

### 2. The Training Orchestrator Loop

The core of the system is a "Training Orchestrator" built into the main `train.py` script. It manages the dynamic operations based on a schedule.

#### 2.1. The `scaling_schedule`

The orchestrator's behavior is defined by a `scaling_schedule`, a FIFO queue of operations.

* **Format:** `(operation_name, operation_value, trigger_val_loss, reevaluate_after)`
    * `operation_name` (str): The name of the function to call (e.g., `'stack_layers'`).
    * `operation_value` (any): The argument for the function (e.g., `2` for `stack_layers`).
    * `trigger_val_loss` (float): The validation loss threshold that triggers the operation.
    * `reevaluate_after` (bool): A flag indicating if the operation is disruptive. `True` forces an immediate re-evaluation of the validation loss.

* **Example `scaling_schedule`:**
    ```python
    scaling_schedule = [
        ('change_lr', 0.5, 6.0, False), # Reduce LR by 50%
        ('stack_layers', 2, 5.5, True), # Double the number of layers
        ('increase_hidden_dim', 1.5, 5.2, True), # Increase MLP hidden dim by 50%
        ('decrease_attn_lora_scaling', 2, 4.8, True), # Effectively doubles the LoRA rank
        ('merge_lora_weights', None, 4.5, True), # Merge LoRA adapters and reset them
    ]
    ```

#### 2.2. Main Loop Logic

The `train.py` script's main loop will be modified as follows:

1.  **Train and Evaluate:** Perform training for `eval_interval` iterations and get the `current_val_loss`.
2.  **Check Schedule Trigger:**
    * If `scaling_schedule` is not empty and `current_val_loss < scaling_schedule[0][2]`, proceed.
3.  **Execute Operation:**
    * Pop the operation `op` from the schedule.
    * Call a master function, `execute_operation(op)`, which handles calling the correct method and managing all necessary state updates (optimizer, DDP wrappers, etc.).
4.  **Handle Re-evaluation:**
    * If `op[3]` (reevaluate_after) is `True`, immediately call `estimate_loss()` again to get a new `current_val_loss`.
5.  **Continue Training.**

---

### 3. System Parameters & Operations

#### 3.1. Dynamic State Parameters (`train.py`)

These parameters are defined at the top of `train.py` to hold the *current state* of the dynamic configuration. They are the single source of truth for all configurable values.

| Parameter Name                  | Initial Value | Description                                                                                             | Modified By Operation(s)             |
| ------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `attn_lora_scaling_f`           | 128           | **Divisor** for attention LoRA rank (`rank = n_embd / attn_lora_scaling_f`). A smaller value means a larger rank. If  1 no Lora is used            | `decrease_attn_lora_scaling`         |
| `vocab_lora_scaling_f`          | 64            | **Divisor** for embedding LoRA rank (`rank = n_embd / vocab_lora_scaling_f`). A smaller value means a larger rank. If  1 no Lora is used           | `decrease_vocab_lora_scaling`        |
| `n_layer_reduction_f`           | 2             | **Divisor** for model depth. `n_layer = final_n_layer / f`.                                               | `stack_layers` (implicitly)          |
| `n_hidden_reduction_f`          | 2             | **Divisor** for MLP width. `n_hidden = final_n_hidden / f`.                                               | `increase_hidden_dim` (implicitly)   |
| `n_batch_increase_f`            | 16            | **Divisor** for batch size. `batch_size = final_batch_size * f`.                                          | `change_batch_size`                  |
| `n_accu_step_reduction_f`       | 16            | **Divisor** for accumulation steps. `grad_accum = final_grad_accum * f`.                                | `change_grad_accum`                  |
| `lr_increase_f`                 | 10            | **Multiplier** for learning rate. `lr = final_lr * f`.                                                    | `change_lr`                          |
| `warmup_iters_decrease_f`       | 10            | **Divisor** for warmup iterations. `warmup_iters = final_warmup / f`.                                     | `change_warmup_iters`                |
| `eval_iters_decrease_f`         | 10            | **Divisor** for evaluation iterations. `eval_iters = final_eval_iters / f`.                               | `change_eval_params`                 |
| `eval_interval_decrease_f`      | 8             | **Divisor** for evaluation frequency. `eval_interval = final_eval_interval / f`.                          | `change_eval_params`                 |

#### 3.2. New `GPTConfig` Parameters (`model.py`)

These are added to the `GPTConfig` dataclass. Their values are derived from the state parameters in `train.py`.

* `embedding_mode` (str): `'standard'`, `'lora'`.
* `embedding_rank` (int): The rank `r` for the embedding layer's LoRA adapter.
* `attn_lora_rank` (int): The rank `r` for attention LoRA.

#### 3.3. Full List of Scheduled Operations

| Operation Name                  | `op_value` Meaning                     | `reevaluate` | Target Parameter(s)                 |
| ------------------------------- | -------------------------------------- | ------------ | ----------------------------------- |
| `change_lr`                     | Divisor for `lr_increase_f`            | `False`      | `lr_increase_f`                     |
| `change_batch_size`             | Divisor for `n_batch_increase_f`       | `False`      | `n_batch_increase_f`                |
| `change_grad_accum`             | Multiplier for `n_accu_step_reduction_f` | `False`      | `n_accu_step_reduction_f`           |
| `change_warmup_iters`           | Divisor for `warmup_iters_decrease_f`  | `False`      | `warmup_iters_decrease_f`           |
| `change_eval_params`            | `(interval_divisor, iters_divisor)`    | `False`      | `eval_interval_f`, `eval_iters_f`   |
| `stack_layers`                  | Multiplier for `n_layer`               | `True`       | `model.stack_layers()`              |
| `increase_hidden_dim`           | Multiplier for `n_hidden`              | `True`       | `model.widen_mlp()`                 |
| `decrease_attn_lora_scaling`    | Divisor for `attn_lora_scaling_f`      | `True`       | `model.resize_lora_rank()`          |
| `decrease_vocab_lora_scaling`   | Divisor for `vocab_lora_scaling_f`     | `True`       | `model.resize_embedding_rank()`     |
| `merge_lora_weights`            | `None`                                 | `True`       | `model.merge_lora_weights()`        |

---

### 4. Division of Responsibilities (`model.py` vs. `train.py`)

* **`model.py` (The Model):** Knows *how* to change its own architecture. Contains methods like `stack_layers`, `widen_mlp`, etc.
* **`train.py` (The Orchestrator):** Knows *when* to change the model and how to manage the training state. Defines the `scaling_schedule`, calls the model's methods, and handles re-creating the optimizer.

---

### 5. Detailed Operation Specifications

#### 5.1. Prerequisite: New Modules in `model.py`

**A. `LoRAEmbedding(nn.Module)`**
* **Goal:** Implement a LoRA-enabled embedding layer for iterative, function-preserving growth.
* **Implementation:**
    ```python
    class LoRAEmbedding(nn.Module):
        def __init__(self, vocab_size, n_embd, rank):
            super().__init__()
            self.main_weight = nn.Embedding(vocab_size, n_embd)
            self.lora_A = nn.Embedding(vocab_size, rank)
            self.lora_B = nn.Linear(rank, n_embd, bias=False)
            self.main_weight.requires_grad_(False)
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, idx):
            main_output = self.main_weight(idx)
            lora_output = self.lora_B(self.lora_A(idx))
            return main_output + lora_output

        def merge_and_reset(self):
            lora_update = self.lora_B.weight.t() @ self.lora_A.weight
            self.main_weight.weight.data += lora_update.t()
            nn.init.normal_(self.lora_A.weight, std=0.02)
            nn.init.zeros_(self.lora_B.weight)
    ```

**B. `LoRALinear(nn.Module)`**
* **Goal:** Implement a LoRA-enabled linear layer for iterative, function-preserving growth.
* **Implementation:**
    ```python
    class LoRALinear(nn.Module):
        def __init__(self, in_features, out_features, rank, bias=True):
            super().__init__()
            self.main_weight = nn.Linear(in_features, out_features, bias=bias)
            self.lora_A = nn.Linear(in_features, rank, bias=False)
            self.lora_B = nn.Linear(rank, out_features, bias=False)
            self.main_weight.requires_grad_(False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.main_weight(x) + self.lora_B(self.lora_A(x))

        def merge_and_reset(self):
            self.main_weight.weight.data += self.lora_B.weight @ self.lora_A.weight
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
    ```

#### 5.2. Architectural Operations (`model.py` methods)

**A. `stack_layers(self, num_copies)`**
* **Specification:** Implements the Gstack operator. Duplicates the model's layers `num_copies` times using `copy.deepcopy()` and replaces `self.transformer.h`. Updates `self.config.n_layer`. This is function-preserving as the duplicated layers simply repeat the function of the original block.
    ```python
    import copy
    # In GPT class
    def stack_layers(self, num_copies):
        if num_copies <= 1: return
        print(f"Stacking layers: current depth {self.config.n_layer}, creating {self.config.n_layer * num_copies} total layers.")
        original_layers = self.transformer.h
        new_layers = nn.ModuleList()
        for _ in range(num_copies):
            for layer in original_layers:
                new_layers.append(copy.deepcopy(layer))
        self.transformer.h = new_layers
        self.config.n_layer = len(new_layers)
    ```

**B. `widen_mlp(self, scale_factor)`**
* **Specification:** Implements the Net2WiderNet algorithm to widen the `c_fc` and `c_proj` layers in each `MLP` block while preserving the function. Updates `self.config.n_hidden`.
    ```python
    # In GPT class
    def widen_mlp(self, scale_factor):
        if scale_factor <= 1: return
        new_hidden_dim = 0
        for block in self.transformer.h:
            mlp = block.mlp
            w_fc, b_fc, w_proj = mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight
            original_hidden_dim = w_fc.shape[0]
            new_hidden_dim = int(original_hidden_dim * scale_factor)

            new_c_fc = nn.Linear(self.config.n_embd, new_hidden_dim, bias=self.config.bias)
            new_c_proj = nn.Linear(new_hidden_dim, self.config.n_embd, bias=self.config.bias)
            
            # Net2WiderNet mapping
            mapping = torch.randint(0, original_hidden_dim, (new_hidden_dim,))
            mapping[:original_hidden_dim] = torch.arange(original_hidden_dim)
            
            # Copy and scale weights
            new_c_fc.weight.data.copy_(w_fc.data[mapping])
            if b_fc is not None: new_c_fc.bias.data.copy_(b_fc.data[mapping])
            
            replication_factors = torch.zeros(original_hidden_dim, device=w_fc.device)
            for i in range(original_hidden_dim): replication_factors[i] = (mapping == i).sum()
            
            new_c_proj.weight.data.copy_(w_proj.data[:, mapping])
            new_c_proj.weight.data /= replication_factors[mapping].view(1, -1)
            
            mlp.c_fc, mlp.c_proj = new_c_fc, new_c_proj
        self.config.n_hidden = new_hidden_dim
    ```

**C. `resize_lora_rank(self, new_rank)`**
* **Specification (Function-Preserving):** Merges the existing LoRA adapter, then creates a new `LoRALinear` layer with the larger rank and a zero-initialized adapter.
    ```python
    # In GPT class
    def resize_lora_rank(self, new_rank):
        print(f"Resizing LoRA rank to {new_rank}.")
        self.config.attn_lora_rank = new_rank
        for block in self.transformer.h:
            if not isinstance(block.attn.c_attn, LoRALinear): continue
            
            # 1. Merge existing knowledge
            block.attn.c_attn.merge_and_reset()
            
            # 2. Create new layer with new rank
            new_c_attn = LoRALinear(self.config.n_embd, 3 * self.config.n_embd, rank=new_rank, bias=self.config.bias)
            
            # 3. Copy merged main weights
            new_c_attn.main_weight.load_state_dict(block.attn.c_attn.main_weight.state_dict())
            
            # 4. Replace layer
            block.attn.c_attn = new_c_attn.to(self.lm_head.weight.device)
    ```

**D. `resize_embedding_rank(self, new_rank)`**
* **Specification (Function-Preserving):** Merges the existing embedding LoRA, then creates a new `LoRAEmbedding` module with the larger rank.
    ```python
    # In GPT class
    def resize_embedding_rank(self, new_rank):
        if not isinstance(self.transformer.wte, LoRAEmbedding): return
        print(f"Resizing Embedding LoRA rank to {new_rank}.")
        self.config.embedding_rank = new_rank
        
        # 1. Merge existing knowledge
        self.transformer.wte.merge_and_reset()
        
        # 2. Create new module with new rank
        new_wte = LoRAEmbedding(self.config.vocab_size, self.config.n_embd, rank=new_rank)
        
        # 3. Copy merged main weights
        new_wte.main_weight.load_state_dict(self.transformer.wte.main_weight.state_dict())
        
        # 4. Replace module
        self.transformer.wte = new_wte.to(self.lm_head.weight.device)
    ```

**E. `merge_lora_weights(self)`**
* **Specification:** Performs an iterative update by merging the learned knowledge and then resetting the adapters to allow for a new phase of training.
    ```python
    # In GPT class
    def merge_lora_weights(self):
        print("Merging and resetting all LoRA adapters.")
        for block in self.transformer.h:
            if isinstance(block.attn.c_attn, LoRALinear):
                block.attn.c_attn.merge_and_reset()
        if isinstance(self.transformer.wte, LoRAEmbedding):
            self.transformer.wte.merge_and_reset()
    ```

#### 5.3. Training Parameter Operations (`train.py` logic)

These operations are handled entirely within the `execute_operation` function in `train.py`.

* **`change_lr(multiplier)`**: `lr_increase_f /= op_value`
* **`change_batch_size(divisor)`**: `n_batch_increase_f /= op_value`
* **`change_grad_accum(multiplier)`**: `n_accu_step_reduction_f *= op_value`
* **`change_warmup_iters(divisor)`**: `warmup_iters_decrease_f /= op_value`
* **`change_eval_params((interval_div, iters_div))`**: `eval_interval_decrease_f /= interval_div`, `eval_iters_decrease_f /= iters_div`

---

### 6. Optimizer State Management

**When to Re-create the Optimizer:**
The optimizer (`torch.optim.AdamW` in this case) must be re-created **any time the model's parameter graph changes**. This includes:
1.  **Adding parameters:** `stack_layers`, `widen_mlp`.
2.  **Replacing modules with new parameters:** `resize_lora_rank`, `resize_embedding_rank`.

The `merge_lora_weights` operation does *not* change the set of trainable parameters (it just updates weights and resets the same adapters), so it does not strictly require an optimizer reset. However, for simplicity and robustness, it is safest to re-create the optimizer after *any* architectural modification. Operations that only change training loop variables (like `change_lr`) do not require an optimizer reset.

**How to Re-create the Optimizer (The `execute_operation` workflow):**
This process is critical, especially when using `torch.compile` and DDP.

```python
# In train.py

def execute_operation(op_tuple):
    global model, optimizer, raw_model
    op_name, op_value, _, _ = op_tuple

    # 1. Get the raw, un-wrapped model instance
    # This is essential. Modifications must happen on the base model.
    unwrapped_model = model.module if ddp else model
    if compile:
        # If compiled, we have been keeping a reference to the original model
        unwrapped_model = unoptimized_model 

    # 2. Perform the operation
    if op_name in ['change_lr', 'change_batch_size', ...]:
        # Handle simple hyperparameter changes here...
        # These do NOT require optimizer reset.
        pass
    else:
        # Architectural changes that DO require an optimizer reset
        if op_name == 'stack_layers':
            unwrapped_model.stack_layers(op_value)
        elif op_name == 'widen_mlp':
            unwrapped_model.widen_mlp(op_value)
        # ... other architectural operations ...

        # 3. Re-create the optimizer with the new parameters
        # The model's .parameters() generator will now yield the new set of parameters.
        print("Re-configuring optimizer for new architecture...")
        optimizer = unwrapped_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

        # 4. Re-apply wrappers
        model = unwrapped_model
        if compile:
            print("Re-compiling the model...")
            model = torch.compile(model)
        if ddp:
            print("Re-wrapping model in DDP...")
            model = DDP(model, device_ids=[ddp_local_rank])
        
        # 5. Update the reference to the raw model
        raw_model = model.module if ddp else model
