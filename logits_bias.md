
# Applying the Logit Bias Fix to Inference Code

## 1. The Goal

The objective is to ensure that the `mask_logit_bias`—the critical fix for the greedy behavior—is correctly applied during the inference process. This involves two steps:
1.  Making the `mask_logit_bias` a permanent part of the model's saved configuration.
2.  Updating the `generate_diffusion` and `sample.py` scripts to use this saved configuration value.

## 2. Step-by-Step Implementation

### Step 1: Update `GPTConfig` in `model.py`

First, the model's configuration class must be aware of the new hyperparameter so it can be saved and loaded with the model checkpoint.

**Action:** In `model.py`, add `mask_logit_bias` to the `GPTConfig` dataclass.

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
    # --- NEW ADDITIONS ---
    model_type: str = 'gpt2'
    mask_token_id: int = None
    mask_logit_bias: float = 0.0 # NEW: Add the bias to the config
```

### Step 2: Update `generate_diffusion` in `model.py`

The generation function must now read the `mask_logit_bias` from its own configuration and apply it to the logits before sampling.

**Action:** In `model.py`, replace your current `generate_diffusion` method with this corrected version.

```python
# In model.py, inside the GPT class

    @torch.no_grad()
    def generate_diffusion(self, idx, max_steps, temperature=1.0, top_k=None):
        """
        Iteratively refine a sequence using diffusion-style generation.
        This version applies the configured logit bias to fix greedy behavior.
        """
        assert self.config.model_type == 'diffusion', "This generation method is only for diffusion models"
        assert self.config.mask_token_id is not None, "mask_token_id must be configured."
        self.eval()

        mask_token_id = self.config.mask_token_id

        # --- Print a header for the logs ---
        print("-" * 100)
        header = f"{'Step':<12} | {'Re-masked':<15} | {'Un-masked':<15} | {'Confidence Ratio':<20} | {'Masks Left'}"
        print(header)
        print("-" * 100)

        for step in range(max_steps):
            num_masks_start_step = (idx == mask_token_id).sum().item()
            if num_masks_start_step == 0:
                print("Generation converged: no masks remaining.")
                break

            logits, _ = self(idx)

            # --- CRITICAL FIX FOR INFERENCE: APPLY LOGIT BIAS ---
            if self.config.mask_logit_bias != 0.0:
                logits[:, :, mask_token_id] += self.config.mask_logit_bias
            # --- END OF FIX ---
            
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(idx.shape)
            
            # (The rest of the function, including logging and update logic, remains unchanged)
            remasked_positions = (idx != mask_token_id) & (idx_next == mask_token_id)
            remasked_count = remasked_positions.sum().item()
            unmasked_positions = (idx == mask_token_id) & (idx_next != mask_token_id)
            unmasked_count = unmasked_positions.sum().item()

            epsilon = 1e-6
            pred_masks = (idx_next == mask_token_id).sum().item()
            pred_words = idx.numel() - pred_masks
            num_words_start_step = idx.numel() - num_masks_start_step
            prediction_ratio = pred_words / (pred_masks + epsilon)
            input_ratio = num_words_start_step / (num_masks_start_step + epsilon)
            confidence_ratio = prediction_ratio / (input_ratio + epsilon)

            new_idx = idx.clone()
            new_idx[unmasked_positions] = idx_next[unmasked_positions]
            new_idx[remasked_positions] = mask_token_id
            idx = new_idx
            
            num_masks_end_step = (idx == mask_token_id).sum().item()

            step_str = f"{step + 1}/{max_steps}"
            col1 = f"{step_str:<12}"
            col2 = f"{remasked_count:<15}"
            col3 = f"{unmasked_count:<15}"
            col4 = f"{confidence_ratio:<20.4f}"
            col5 = f"{num_masks_end_step}"
            print(f"{col1} | {col2} | {col3} | {col4} | {col5}")

        print("-" * 100)
        
        if (idx == mask_token_id).any():
            print("Finalizing generation: removing any remaining masks...")
            logits, _ = self(idx)

            # --- APPLY FINALIZATION BIAS AS WELL ---
            if self.config.mask_logit_bias != 0.0:
                logits[:, :, mask_token_id] += self.config.mask_logit_bias
            
            logits[:, :, mask_token_id] = -float('Inf') # Forbid predicting MASK
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(idx.shape)
            mask = (idx == mask_token_id)
            idx = torch.where(mask, idx_next, idx)

        self.train()
        return idx
```

### Step 3: Update `train.py` (Crucial for Saving Checkpoints)

Your training script must be told to save the new `mask_logit_bias` value into the checkpoint file.

**Action:** In `train.py`, find the `model_args` dictionary definition and add the `mask_logit_bias` to it. You also need to ensure it's loaded correctly when resuming.

1.  **In the `model init` block:**

    ```python
    # In train.py
    # ...
    # Add mask_logit_bias to the dictionary
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout, model_type=model_type,
                      mask_logit_bias=mask_logit_bias)
    # ...
    ```

2.  **In the `init_from = 'resume'` block:**

    ```python
    # In train.py
    # ...
    elif init_from == 'resume':
        # ... (loading checkpoint) ...
        # --- MODIFICATION: Add mask_logit_bias to the list of keys to load ---
        if 'mask_logit_bias' in checkpoint_model_args:
            model_args['mask_logit_bias'] = checkpoint_model_args['mask_logit_bias']
        # --- END MODIFICATION ---
        # ... (rest of the resume logic) ...
    ```

### Step 4: No Changes Needed for `sample.py`

This is the best part of this fix. Because you have correctly designed your project so that `sample.py` loads the `GPTConfig` directly from the checkpoint file:

```python
# In your sample.py
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
```

And because we have now added `mask_logit_bias` to that `GPTConfig` (Step 1) and ensured it's saved in the checkpoint (Step 3), the value will be **automatically loaded** into the model object.

When you call `model.generate_diffusion(...)`, the corrected function (from Step 2) will read `self.config.mask_logit_bias` and everything will work correctly. **Your inference script requires no direct changes.**

After making these changes to `model.py` and `train.py` and retraining your model, your inference script will now correctly apply the necessary bias and should produce much more coherent and well-behaved outputs.