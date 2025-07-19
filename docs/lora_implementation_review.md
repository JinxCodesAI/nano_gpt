

### **Detailed Instructions to Fix and Refactor `train.py`**

There are three main tasks:
1.  **Fix the LoRA configuration bug** to ensure LoRA settings are passed to the model.
2.  **Fix the `LoRAEmbedding` matrix multiplication bug** in `model.py`.
3.  **Refactor the parameter logging** into a reusable function and place the calls correctly.

---

### **Task 1: Fix LoRA Configuration Passing in `train.py`**

This is the bug that caused LoRA to be disabled.

#### **Step 1.1: Add Default LoRA Parameters**

In your `train.py` file, find the "Dynamic State Parameters" section (around line 85).

**ADD** the following lines to this section. This ensures your script has default values and won't crash if a LoRA config isn't provided.

```python
# In train.py, after the other dynamic parameters...

# --- ADD THESE LINES ---
# Concrete LoRA architectural parameters. These will be overridden by config files.
embedding_mode = 'standard'
attn_lora_rank = 0 # rank for attention LoRA, 0 disables
embedding_rank = 0 # rank for embedding LoRA, 0 disables
lora_alpha = 1.0 # scaling factor for LoRA layers
# --- END ADD ---```

#### **Step 1.2: Update the `model_args` Dictionary**

In `train.py`, find the `model init` section (around line 214). You need to add the LoRA parameters to this dictionary so they get passed to the `GPTConfig` constructor.

**REPLACE** the existing `model_args` definition:
```python
# OLD CODE TO REPLACE
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, n_hidden=n_hidden,
                  use_rotary_embeddings=use_rotary_embeddings,
                  rotary_base=rotary_base,
                  rotary_max_position_embeddings=rotary_max_position_embeddings)
```

**WITH THIS NEW CODE:**
```python
# NEW CODE
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, n_hidden=n_hidden,
    use_rotary_embeddings=use_rotary_embeddings,
    rotary_base=rotary_base,
    rotary_max_position_embeddings=rotary_max_position_embeddings,
    # Add these new keys to pass LoRA config to the model
    embedding_mode=embedding_mode,
    embedding_rank=embedding_rank,
    attn_lora_rank=attn_lora_rank,
    lora_alpha=lora_alpha
)
```

---

### **Task 2: Fix `LoRAEmbedding` and Parameter Freezing in `model.py`**

This task fixes the `RuntimeError` from matrix multiplication and the parameter counting bug.

#### **Step 2.1: Correct the `merge_and_reset` Method**

In your `model.py` file, find the `LoRAEmbedding` class.

**REPLACE** the `merge_and_reset` method with the following corrected version:
```python
# In model.py, class LoRAEmbedding

# OLD METHOD TO REPLACE
def merge_and_reset(self):
    if self.rank == 0 or self.lora_A is None or self.lora_B is None:
        return
    # Correct matrix multiplication order: A @ B
    lora_update = self.lora_A.weight @ self.lora_B.weight.T
    # The result is already (vocab_size, n_embd), so no transpose is needed.
    self.main_weight.weight.data += lora_update * (self.alpha / self.rank)
    # Reset LoRA weights
    nn.init.normal_(self.lora_A.weight, std=0.02)
    nn.init.zeros_(self.lora_B.weight)
```
*I notice you had a `.T` on `lora_B.weight` in your provided code, which might be a typo from a previous attempt. The shapes are `A:(V,R)` and `B:(E,R)`, so `B.T` is `(R,E)`. `A @ B.T` -> `(V,R) @ (R,E)` -> `(V,E)`, which is correct. My previous suggestion was `A @ B` which was incorrect. The version with `.T` is the right one.*

**Corrected `merge_and_reset` Method (Final Version):**
```python
# In model.py, class LoRAEmbedding

def merge_and_reset(self):
    if self.rank == 0 or self.lora_A is None or self.lora_B is None:
        return
    # Correct matrix multiplication order: A @ B.T
    # lora_A.weight is (V, R)
    # lora_B.weight is (E, R), so lora_B.weight.T is (R, E)
    # The result is (V, E) which matches the main_weight shape.
    lora_update = self.lora_A.weight @ self.lora_B.weight.T
    self.main_weight.weight.data += lora_update * (self.alpha / self.rank)
    # Reset LoRA weights
    nn.init.normal_(self.lora_A.weight, std=0.02)
    nn.init.zeros_(self.lora_B.weight)
```

#### **Step 2.2: Fix Weight Tying and Freezing**

In `model.py`, find the `GPT` class `__init__` method.

**REPLACE** the original weight tying block:
```python
# OLD CODE TO REPLACE
if isinstance(self.transformer.wte, LoRAEmbedding):
    # For LoRA embeddings, tie the main_weight with lm_head
    self.transformer.wte.main_weight.weight = self.lm_head.weight
else:
    # Standard weight tying for regular embeddings
    self.transformer.wte.weight = self.lm_head.weight
```

**WITH THIS NEW, CORRECTED BLOCK:**
```python
# NEW CODE
if isinstance(self.transformer.wte, LoRAEmbedding):
    # For LoRA embeddings, tie the main_weight with lm_head
    self.transformer.wte.main_weight.weight = self.lm_head.weight
    # CRITICAL FIX: After tying, explicitly freeze the lm_head as well,
    # since it now shares the same (supposedly frozen) weight tensor.
    self.lm_head.requires_grad_(False)
else:
    # Standard weight tying for regular embeddings
    self.transformer.wte.weight = self.lm_head.weight
```

---

### **Task 3: Refactor Parameter Logging in `train.py`**

This task fixes the `NameError` and provides logging before and after architectural changes.

#### **Step 3.1: ADD the `log_detailed_params` Helper Function**

In `train.py`, find a good place for helper functions, for example, just before the `execute_operation` function (around line 320).

**ADD** the following function definition:
```python
# In train.py

def log_detailed_params(model_to_log):
    """Logs the detailed parameter count of the provided model."""
    if master_process:
        print("\nDetailed parameter count:")
        detailed_params = model_to_log.get_detailed_param_count()
        for component, counts in detailed_params.items():
            total_str = f"{counts['total']:,}"
            trainable_str = f"{counts['trainable']:,}"
            print(f"  {component:<22} | Total: {total_str:>12} | Trainable: {trainable_str:>12}")
        print("-" * 60)
```

#### **Step 3.2: DELETE the Old Logging Block**

In `train.py`, find and **DELETE** the entire block that was causing the `NameError` (around line 234).
```python
# In train.py -- DELETE THIS BLOCK
if master_process:
    print("\nDetailed parameter count:")
    # ... entire for loop ...
```

#### **Step 3.3: INSERT Calls to the New Function**

**A. First Call (Initial State):** In `train.py`, find where `raw_model` is defined, just before the `while True:` loop (around line 575).

**INSERT** the function call right after `raw_model` is defined:
```python
# In train.py
raw_model = model.module if ddp else model
# --- INSERT THIS LINE ---
log_detailed_params(raw_model)
# --- END INSERT ---
running_mfu = -1.0
while True:
    # ...
```

**B. Second Call (After Architectural Change):** In `train.py`, inside the `execute_operation` function, find the section that handles `architectural_ops`.

**INSERT** the function call right after the model is modified and before the optimizer is re-created:
```python
# In execute_operation()
    if op_name in architectural_ops:
        # ... (logic to get unwrapped_model and perform operations) ...
        elif op_name == 'merge_lora_weights':
            unwrapped_model.merge_lora_weights()
            if master_process:
                training_logger.log_operation_success(iter_num, op_name, {'status': 'merged'})
        
        # --- INSERT THIS LINE ---
        log_detailed_params(unwrapped_model)
        # --- END INSERT ---
        
        # Re-create optimizer for the modified model
        # ...
```
