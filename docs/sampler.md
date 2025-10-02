# Developer Guide: Implementing an Optional, Coherent Sampler Head

This document provides a complete, step-by-step guide for integrating a lightweight, bidirectional sampler head into the discrete diffusion model. The implementation is designed to be modular and optional, ensuring backward compatibility with existing models and providing a clear path for training and inference.

## 1. The Goal and Rationale

### 1.1 The Problem: Lack of Local Coherence

The core strength of our diffusion model is its ability to unmask multiple tokens in parallel. However, this introduces a challenge: the model's prediction for each `[MASK]` is independent of the other predictions made in the same step. This can lead to outputs that are locally incoherent.

For example, given the input `The [MASK] [MASK] on the [MASK]`, a parallel unmasker might plausibly predict `book` for the first mask and `shines` for the second, resulting in the nonsensical phrase "The book shines...". While each word is a reasonable prediction on its own, they don't form a coherent sequence together. The model needs a mechanism to ensure its parallel predictions are consistent with each other.

### 1.2 The Solution: A Bidirectional Sampler Head

We will solve this by implementing a lightweight, auxiliary sampler head. This module acts as an intelligent post-processor for the main model's predictions. Instead of sampling all tokens simultaneously from the initial logits, we will fill the masked positions sequentially, using the sampler to enforce local coherence.

For our bidirectional model, the sampler's prediction for a given token will be conditioned on three inputs:

  * The main model's hidden state (`Z`) for that position.
  * The embedding of the token's left neighbor.
  * The embedding of the token's right neighbor.

This ensures that every token sampled is contextually aware of the tokens that have just been placed around it, eliminating incoherent combinations and dramatically improving the quality of the generated text.

## 2. Implementation Guide: Model Architecture

The following changes should be made to your model definition file (e.g., `model.py`).

### 2.1. Making the Sampler Optional via Configuration

First, add a boolean flag to your `GPTConfig` dataclass. This will allow you to control whether a model is built with a sampler head, ensuring that you can still load and use models without one.

```python
# In your GPTConfig dataclass
@dataclass
class GPTConfig:
    # ... your existing parameters: n_layer, n_head, n_embd, etc.
    add_sampler_head: bool = False # <-- Add this flag
```

### 2.2. Creating the SamplerHead Module

Next, define the `SamplerHead` as a new `nn.Module`. It will be a simple two-layer MLP that processes the concatenated inputs.

```python
class SamplerHead(nn.Module):
    """
    A lightweight, bidirectional MLP that conditions a token prediction on its
    hidden state and its immediate left and right neighbors.
    """
    def __init__(self, config):
        super().__init__()
        # The input is the concatenation of three vectors:
        # [Embedding(left_neighbor), HiddenState(current), Embedding(right_neighbor)]
        input_dim = config.n_embd * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.n_embd, bias=config.bias),
            nn.SiLU(),
            LayerNorm(config.n_embd, bias=config.bias),
            nn.Linear(config.n_embd, config.n_embd, bias=config.bias),
            nn.SiLU(),
            LayerNorm(config.n_embd, bias=config.bias)
        )

    def forward(self, combined_input):
        # combined_input has shape (N, 3 * n_embd)
        return self.mlp(combined_input)
```

### 2.3. Conditionally Adding the Sampler to the GPT Model

Finally, modify your main `GPT` model's `__init__` method to create the sampler head only if the `add_sampler_head` flag is `True`. This is crucial for maintaining backward compatibility.

```python
# In your GPT.__init__ method
class GPT(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        # ... all your existing __init__ logic for self.transformer, etc. ...

        # Initialize sampler_head as None by default
        self.sampler_head = None
        # Conditionally create the sampler head based on the config
        if getattr(config, 'add_sampler_head', False):
            self.sampler_head = SamplerHead(config)
            self._log_info("Sampler head enabled for coherent multi-token sampling.")

        # ... rest of your __init__ logic ...
```

## 3. Implementation Guide: Training Strategy

To ensure stability, we will treat the sampler as an auxiliary network and introduce its training in a staged manner.

### 3.1. Add Configuration Parameters for Staged Training

In your training configuration file or dataclass, add the following parameters. This allows you to control the training schedule without changing the code.

```python
# In your training configuration
@dataclass
class TrainingConfig:
    # ... your existing training parameters ...

    # The iteration number at which to start training the sampler head.
    # Before this, only the main model's loss will be calculated.
    start_sampler_iteration: int = 5000 # Example: tune as needed

    # The weight to apply to the sampler's loss.
    sampler_loss_weight: float = 0.5 # Example: tune as needed
```

### 3.2. Modify the Training Step Logic

Update your main training function to conditionally compute and apply the sampler's loss. The key is to use `.detach()` to prevent the sampler's gradients from flowing back into and disrupting the main model's training process.

```python
# --- Conceptual Code for a Single Training Step ---

def train_step(model, batch, current_iteration, config):
    # Unpack batch data
    idx, targets = batch

    # --- Phase 1: Main Model Training (Always Active) ---
    
    # 1. Perform a forward pass through the main model.
    hidden_states = model._encode_tokens(idx)
    base_logits = model.lm_head(hidden_states)

    # 2. Calculate the primary loss for the main model.
    main_model_loss = F.cross_entropy(base_logits.view(-1, config.vocab_size), targets.view(-1), ignore_index=config.ignore_index)
    
    total_loss = main_model_loss

    # --- Phase 2: Sampler Head Training (Conditionally Active) ---

    # 3. Check if we should train the sampler in this step.
    if current_iteration >= config.start_sampler_iteration and model.sampler_head is not None:
        
        # 4. Prepare inputs for the sampler head. This function needs to be implemented.
        # It finds tokens that have valid neighbors and returns their indices and targets.
        sampler_target_indices, left_ids, right_ids, sampler_targets = prepare_sampler_inputs(hidden_states, targets)

        # 5. CRUCIAL: Detach all inputs to the sampler from the computation graph.
        # This isolates the sampler's training and protects the main model.
        hidden_states_for_sampler = hidden_states[sampler_target_indices].detach()
        left_embeddings = model.transformer.wte(left_ids).detach()
        right_embeddings = model.transformer.wte(right_ids).detach()
        
        # 6. Perform the sampler's forward pass.
        sampler_input = torch.cat([left_embeddings, hidden_states_for_sampler, right_embeddings], dim=1)
        sampler_features = model.sampler_head(sampler_input)
        sampler_logits = model.lm_head(sampler_features)
        
        # 7. Calculate the sampler's loss.
        sampler_loss = F.cross_entropy(sampler_logits, sampler_targets)
        
        # 8. Add the weighted sampler loss to the total loss.
        total_loss = total_loss + (config.sampler_loss_weight * sampler_loss)

    # --- Final Step: Backpropagation ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()
```

## 4. Implementation Guide: Inference Logic

Your inference script must now handle both models with and without a sampler.

### 4.1. Runtime Detection of the Sampler

In your main generation function (e.g., `diffusion_generate`), add a check at the beginning to determine which sampling path to take.

```python
# In your main inference/generation function
def diffusion_generate(model, initial_tokens, iterations, ...):
    
    # Check for the sampler's existence once at the start.
    sampler_exists = hasattr(model, 'sampler_head') and model.sampler_head is not None

    tokens = initial_tokens
    for iteration in range(iterations):
        if sampler_exists:
            # Use the new, coherent sampling method
            filled_tokens = coherent_sampling_step(model, tokens, ...)
        else:
            # Fall back to your original, naive sampling method
            # (e.g., using torch.multinomial on the initial logits)
            filled_tokens = naive_parallel_sampling_step(model, tokens, ...)
            
        # ... rest of your diffusion loop (critic remasking) ...
        tokens = run_critic_and_remask(model, filled_tokens, ...)
    
    return tokens
```

### 4.2. The Coherent Sampling Step

This function orchestrates the two-phase sampling process for models with a sampler.

```python
def coherent_sampling_step(model, Y, mask_token_id, pad_token_id, ...):
    """
    Fills all masked tokens in Y using a confidence-based, coherent sampling strategy.
    """
    # --- 1. Confidence Pass ---
    # Run one forward pass to get hidden states (Z) and initial logits for all tokens.
    with torch.no_grad():
        Z = model._encode_tokens(Y)
        initial_logits = model.lm_head(Z)

    # --- 2. Iterative Sampler Pass (Parallel Wavefront) ---
    # Use the parallel sampler pass to fill in masks. It may take a few waves
    # to fill all masks if they are in large, contiguous blocks.
    temp_Y = Y.clone()
    # Set a max number of waves to prevent rare infinite loops
    for _ in range(Y.shape[1]): # Max waves = sequence length
        num_masks_before = (temp_Y == mask_token_id).sum()
        if num_masks_before == 0:
            break
            
        temp_Y = parallel_sampler_pass(model, temp_Y, Z, mask_token_id, pad_token_id, ...)
        
        num_masks_after = (temp_Y == mask_token_id).sum()
        if num_masks_after == num_masks_before:
            # If no progress was made, break to avoid getting stuck.
            # You could fill remaining masks randomly or with the top-1 logit as a fallback.
            break
            
    return temp_Y
```

### 4.3. The Parallel Sampler Pass

This is the highly optimized function that performs one "wave" of unmasking on all eligible tokens in a single parallel batch.

```python
def parallel_sampler_pass(model, Y, Z, mask_token_id, pad_token_id, temperature=1.0, top_p=1.0):
    """
    Performs one parallel sampling pass on all eligible masked tokens that have at least
    one unmasked neighbor.
    """
    # 1. Identify all target masks that have at least one real neighbor
    is_real_token = (Y != mask_token_id)
    is_neighbor_to_real_token = torch.zeros_like(is_real_token, device=Y.device)
    is_neighbor_to_real_token[:, 1:] |= is_real_token[:, :-1]
    is_neighbor_to_real_token[:, :-1] |= is_real_token[:, 1:]
    sampler_target_mask = (Y == mask_token_id) & is_neighbor_to_real_token
    
    batch_indices, token_indices = sampler_target_mask.nonzero(as_tuple=True)
    
    if batch_indices.size(0) == 0:
        return Y # No eligible tokens to sample in this pass

    # 2. Gather all inputs for the sampler in a single parallel batch
    hidden_states = Z[batch_indices, token_indices]

    # Gather Left Neighbor Embeddings
    left_indices = (token_indices - 1).clamp(min=0)
    left_ids = Y[batch_indices, left_indices]
    left_ids[Y[batch_indices, left_indices] == mask_token_id] = pad_token_id
    left_embeddings = model.transformer.wte(left_ids)

    # Gather Right Neighbor Embeddings
    right_indices = (token_indices + 1).clamp(max=Y.shape[1] - 1)
    right_ids = Y[batch_indices, right_indices]
    right_ids[Y[batch_indices, right_indices] == mask_token_id] = pad_token_id
    right_embeddings = model.transformer.wte(right_ids)

    # 3. Run the Sampler Head
    sampler_input = torch.cat([left_embeddings, hidden_states, right_embeddings], dim=1)
    sampler_features = model.sampler_head(sampler_input)
    logits = model.lm_head(sampler_features)
    
    # 4. Sample from logits and update the main token tensor Y
    # (Apply temperature/top_p sampling logic here)
    probs = F.softmax(logits / temperature, dim=-1)
    # Your nucleus_sample function can be used here if it supports batched inputs
    new_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
    
    Y[batch_indices, token_indices] = new_token_ids
    return Y
```