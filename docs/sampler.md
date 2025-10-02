# Developer Guide: Implementing an Optional, Coherent Sampler Head

This document provides a complete, step-by-step guide for integrating a lightweight, bidirectional sampler head into the discrete diffusion model. The implementation is designed to be modular and optional, ensuring backward compatibility with existing models and providing a clear path for training and inference.

## 1. The Goal and Rationale

### 1.1 The Problem: Lack of Local Coherence

The core strength of our diffusion model is its ability to unmask multiple tokens in parallel. However, this introduces a challenge: the model's prediction for each `[MASK]` is independent of the other predictions made in the same step. This can lead to outputs that are locally incoherent.

For example, given the input `The [MASK] [MASK] on the [MASK]`, a parallel unmasker might plausibly predict `book` for the first mask and `shines` for the second, resulting in the nonsensical phrase "The book shines...". While each word is a reasonable prediction on its own, they don't form a coherent sequence together. The model needs a mechanism to ensure its parallel predictions are consistent with each other.

### 1.2 The Solution: A Bidirectional Sampler Head

We will solve this by implementing a lightweight, **auxiliary** sampler head. This module acts as an intelligent post-processor for the main model's predictions. Instead of sampling all tokens simultaneously from the initial logits, we will fill the masked positions in waves, using the sampler to enforce local coherence.

**Key Design Principles:**
- **Auxiliary Architecture**: The sampler is a separate, lightweight network trained independently from the main transformer
- **LANGUAGE_MODEL Mode Only**: Sampler head is only available for `LANGUAGE_MODEL` mode (like the critic head)
- **Bidirectional Attention Required**: The model must use `attention_type='bidirectional'` for the sampler to work correctly
- **Optional Feature**: The sampler can be disabled for performance-critical scenarios; models without it remain fully functional

For our bidirectional model, the sampler's prediction for a given token will be conditioned on three inputs:

  * The main model's hidden state (`Z`) for that position.
  * The embedding of the token's left neighbor.
  * The embedding of the token's right neighbor.

This ensures that every token sampled is contextually aware of the tokens that have just been placed around it, eliminating incoherent combinations and dramatically improving the quality of the generated text.

**Important note on neighbor handling**: When a token does not have a left or right neighbor (e.g., at sequence boundaries or when the neighbor is `[MASK]`), use a **zero embedding** for that missing neighbor position. This signals to the sampler that no context is available from that direction.

## 2. Implementation Guide: Model Architecture

The following changes should be made to your model definition file (e.g., `model.py`).

### 2.1. Making the Sampler Optional via Configuration

First, add configuration parameters to your `GPTConfig` dataclass. This will allow you to control whether a model is built with a sampler head, ensuring that you can still load and use models without one.

```python
# In your GPTConfig dataclass
@dataclass
class GPTConfig:
    # ... your existing parameters: n_layer, n_head, n_embd, etc.

    # Optional sampler head configuration (LANGUAGE_MODEL only)
    add_sampler_head: bool = False  # Enable/disable sampler head
    start_sampler_iteration: int = 0  # Iteration to start training sampler (0 = from beginning)
    sampler_min_neighbors_ratio: float = 0.01  # Minimum ratio of tokens to bootstrap when no neighbors available

    def __post_init__(self):
        # ... existing validation ...

        # Validate sampler head requirements
        if self.add_sampler_head:
            if self.mode != ModelMode.LANGUAGE_MODEL:
                raise ValueError("Sampler head only supported for LANGUAGE_MODEL mode")
            if self.attention_type != 'bidirectional':
                raise ValueError("Sampler head requires bidirectional attention (set attention_type='bidirectional')")
            if self.mask_token_id is None:
                raise ValueError("Sampler head requires mask_token_id to be configured")
```

### 2.2. Creating the SamplerHead Module

Next, define the `SamplerHead` as a new `nn.Module`. It will be a simple two-layer MLP that processes the concatenated inputs.

**Important**: The sampler head is an **auxiliary network** that is trained separately from the main transformer. It takes detached inputs during training to prevent gradients from flowing back into the main model.

```python
class SamplerHead(nn.Module):
    """
    A lightweight, bidirectional MLP that conditions a token prediction on its
    hidden state and its immediate left and right neighbors.

    This is an auxiliary network trained separately from the main transformer.
    During training, inputs are detached to isolate sampler training.

    Input: [left_neighbor_embedding, hidden_state, right_neighbor_embedding]
    Output: features (n_embd) to be passed to lm_head for token prediction

    When a neighbor is missing (boundary or [MASK]), use zero embedding.
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
        """
        Args:
            combined_input: (N, 3*n_embd) concatenated [left_emb, hidden_state, right_emb]

        Returns:
            features: (N, n_embd) features for lm_head to predict tokens
        """
        return self.mlp(combined_input)
```

### 2.3. Conditionally Adding the Sampler to the GPT Model

Finally, modify your main `GPT` model's `__init__` method to create the sampler head only if the `add_sampler_head` flag is `True`. This is crucial for maintaining backward compatibility.

**Note**: The sampler head should be created **after** the main model heads (lm_head) and **after** the critic head (if present), following the existing pattern in the codebase.

```python
# In your GPT.__init__ method
class GPT(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        # ... all your existing __init__ logic for self.transformer, etc. ...

        # Create mode-specific output heads (existing code)
        if self.config.mode == ModelMode.LANGUAGE_MODEL:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight
        # ... other modes ...

        # Optional critic head for LANGUAGE_MODEL multi-tasking (existing code)
        if getattr(self.config, 'add_critic_head', False) and self.config.mode == ModelMode.LANGUAGE_MODEL:
            self.critic_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self._log_info(f"Critic head enabled (alpha={self.config.critic_alpha})")

        # Optional sampler head for LANGUAGE_MODEL coherent sampling (NEW)
        if getattr(config, 'add_sampler_head', False):
            if config.mode != ModelMode.LANGUAGE_MODEL:
                raise ValueError("Sampler head only supported for LANGUAGE_MODEL mode")
            self.sampler_head = SamplerHead(config)
            self._log_info(f"Sampler head enabled (weight={config.sampler_loss_weight}, start_iter={config.start_sampler_iteration})")
        else:
            self.sampler_head = None

        # ... rest of your __init__ logic (weight initialization, etc.) ...
```

## 3. Implementation Guide: Training Strategy

The sampler is an **auxiliary network** trained **separately** from the main transformer. This is different from the critic head, which shares the transformer trunk with the main model.

### 3.1. Training Schedule: Three-Stage Approach

Training proceeds in three distinct stages, controlled by iteration thresholds:

**Stage 1: Main Model Only** (iterations 0 to `start_sampler_iteration`)
- Only the main language model is trained
- Sampler head exists but is not trained (no loss computed)
- Critic head exists but is not trained (alpha = 0)

**Stage 2: Main Model + Sampler** (iterations `start_sampler_iteration` to `start_critic_iteration`)
- Main model continues training
- Sampler head begins training with its own separate loss
- Critic head still inactive (alpha = 0)

**Stage 3: Main Model + Sampler + Critic** (iterations `start_critic_iteration` onwards)
- Main model continues training
- Sampler head continues training
- Critic head becomes active (alpha ramps up from 0 to configured value)

### 3.2. Configuration Parameters

The training schedule is controlled by these configuration parameters (already added to `GPTConfig` in section 2.1):

```python
@dataclass
class GPTConfig:
    # ... existing parameters ...

    # Sampler configuration
    add_sampler_head: bool = False
    start_sampler_iteration: int = 0  # When to start training sampler (0 = from beginning)
    sampler_min_neighbors_ratio: float = 0.01  # Bootstrap ratio when no neighbors (1%)

    # Critic configuration (existing)
    add_critic_head: bool = False
    critic_alpha: float = 0.5
    start_critic_iteration: int = 0  # When to start training critic
    end_critic_iteration: int = 0    # When critic reaches full alpha
```

**Recommended values:**
- `start_sampler_iteration = 0`: Train sampler from the beginning alongside main model
- `start_critic_iteration = 5000`: Start critic after sampler has stabilized
- `sampler_min_neighbors_ratio = 0.01`: Bootstrap 1% of tokens when no neighbors available

### 3.3. Helper Function: prepare_sampler_inputs

This function prepares sampler training data from the current batch. It is inspired by `build_critic_artifacts_from_logits` but serves a different purpose.

**Key differences from critic artifacts:**
- **Critic**: Creates synthetic inputs by sampling from logits to train wrongness prediction
- **Sampler**: Uses actual training data (input tokens and targets) to learn neighbor-conditioned token prediction

**Important**: During training, **every supervised position is suitable** for sampler training because neighbors are taken from the input/targets (not from masked positions). The sampler learns to predict the target token given the hidden state and the actual neighbor tokens from the input.

```python
def prepare_sampler_inputs(idx: torch.Tensor,
                          targets: torch.Tensor,
                          hidden_states: torch.Tensor,
                          wte_embedding: nn.Embedding,
                          mask_token_id: int,
                          ignore_index: int):
    """
    Prepare sampler training artifacts from current batch.

    Returns a dict with detached embeddings and training targets, similar to
    build_critic_artifacts_from_logits pattern.

    During training, every supervised position (targets != ignore_index) is eligible
    because neighbors come from the actual input tokens, not from masked positions.

    Args:
        idx: (B, T) input token IDs
        targets: (B, T) target token IDs
        hidden_states: (B, T, n_embd) hidden states from transformer
        wte_embedding: Token embedding layer (model.transformer.wte)
        mask_token_id: ID of the mask token
        ignore_index: Ignore index for loss computation (typically -100)

    Returns:
        dict with:
            'sampler_input': (N, 3*n_embd) concatenated [left_emb, hidden, right_emb] - DETACHED
            'sampler_targets': (N,) target token IDs for training
            'num_positions': int, number of eligible positions
    """
    B, T, n_embd = hidden_states.shape
    device = idx.device

    # Find supervised positions (not ignore_index)
    # During training, ALL supervised positions are eligible because neighbors
    # come from the input sequence, not from predictions
    valid = (targets != ignore_index)

    # Get indices of all valid positions
    batch_indices, pos_indices = valid.nonzero(as_tuple=True)
    N = len(batch_indices)

    if N == 0:
        # No eligible positions
        return None

    # Gather hidden states for all valid positions (DETACH for auxiliary training)
    h = hidden_states[batch_indices, pos_indices].detach()  # (N, n_embd)

    # Gather targets
    sampler_targets = targets[batch_indices, pos_indices]  # (N,)

    # Prepare left neighbor embeddings (DETACHED)
    # Use zero embedding when no left neighbor (position 0) or neighbor is [MASK]
    left_emb = torch.zeros(N, n_embd, device=device, dtype=h.dtype)
    left_exists = pos_indices > 0
    if left_exists.any():
        left_ids = idx[batch_indices[left_exists], pos_indices[left_exists] - 1]
        left_not_mask = left_ids != mask_token_id
        if left_not_mask.any():
            # Get embeddings for non-mask neighbors (DETACHED)
            left_emb[left_exists][left_not_mask] = wte_embedding(left_ids[left_not_mask]).detach()

    # Prepare right neighbor embeddings (DETACHED)
    # Use zero embedding when no right neighbor (position T-1) or neighbor is [MASK]
    right_emb = torch.zeros(N, n_embd, device=device, dtype=h.dtype)
    right_exists = pos_indices < (T - 1)
    if right_exists.any():
        right_ids = idx[batch_indices[right_exists], pos_indices[right_exists] + 1]
        right_not_mask = right_ids != mask_token_id
        if right_not_mask.any():
            # Get embeddings for non-mask neighbors (DETACHED)
            right_emb[right_exists][right_not_mask] = wte_embedding(right_ids[right_not_mask]).detach()

    # Concatenate inputs for sampler (all detached)
    sampler_input = torch.cat([left_emb, h, right_emb], dim=-1)  # (N, 3*n_embd)

    return {
        'sampler_input': sampler_input,
        'sampler_targets': sampler_targets,
        'num_positions': N,
    }
```

### 3.4. Training Loop Integration

The sampler loss is computed **in addition to** the main LM loss and added directly to the total loss. This is implemented in the model's `_forward_language_model` method, following the same pattern as the critic head.

```python
def _forward_language_model(self, x, targets, loss_modifiers, idx=None):
    """Language modeling forward pass with optional critic and sampler heads"""
    if targets is not None:
        # Main LM loss (existing code)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=self.config.ignore_index)

        # Track loss components for logging
        self._last_lm_loss = float(loss.detach().item())
        self._last_sampler_loss = 0.0
        self._last_critic_loss = 0.0

        # Optional sampler loss (NEW - Stage 2+)
        current_iter = getattr(self, '_current_iter', 0)
        if (getattr(self.config, 'add_sampler_head', False) and
            hasattr(self, 'sampler_head') and
            current_iter >= getattr(self.config, 'start_sampler_iteration', 0)):

            sampler_loss = self._compute_sampler_loss(x, idx, targets)
            if sampler_loss is not None:
                # Add sampler loss directly (no weight - standard cross-entropy)
                loss = loss + sampler_loss
                self._last_sampler_loss = float(sampler_loss.detach().item())

        # Optional critic loss (existing code - Stage 3+)
        alpha_eff = self._effective_critic_alpha()
        if (getattr(self.config, 'add_critic_head', False) and
            hasattr(self, 'critic_head') and
            alpha_eff > 0.0):
            # ... existing critic loss computation ...
            critic_loss = ...  # existing code
            loss = loss + float(alpha_eff) * critic_loss
            self._last_critic_loss = float(critic_loss.detach().item())

    else:
        # Inference mode (handled in section 4)
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss

def _compute_sampler_loss(self, hidden_states, idx, targets):
    """
    Compute sampler head loss using neighbor context.

    This is an auxiliary loss that trains the sampler independently.
    All inputs are detached to prevent gradients from affecting the main model.

    Returns standard cross-entropy loss (no special weighting).
    """
    artifacts = prepare_sampler_inputs(
        idx=idx,
        targets=targets,
        hidden_states=hidden_states,
        wte_embedding=self.transformer.wte,
        mask_token_id=self.config.mask_token_id,
        ignore_index=self.config.ignore_index
    )

    if artifacts is None:
        return None  # No eligible positions

    # All inputs are already detached by prepare_sampler_inputs
    sampler_input = artifacts['sampler_input']  # (N, 3*n_embd)
    sampler_targets = artifacts['sampler_targets']  # (N,)

    # Forward through sampler head
    sampler_features = self.sampler_head(sampler_input)  # (N, n_embd)
    sampler_logits = self.lm_head(sampler_features)  # (N, vocab_size)

    # Compute standard cross-entropy loss
    sampler_loss = F.cross_entropy(sampler_logits, sampler_targets)

    return sampler_loss
```

## 4. Implementation Guide: Inference Logic

During inference, the sampler head replaces the naive parallel sampling with a **wavefront-based coherent sampling** approach. This fills masked positions in waves, where each wave fills tokens that have at least one already-filled (non-masked) neighbor.

**Edge Case - Bootstrap**: When there are not enough tokens with neighbors (e.g., all tokens are masked), the sampler first performs a **bootstrap step**: it selects the top 1% (configurable via `sampler_min_neighbors_ratio`) of masked positions by highest logit confidence and fills them using naive sampling. This creates "seed" tokens that allow the wavefront to proceed.

### 4.1. Integration with Existing Diffusion Loop

The sampler integrates into the existing diffusion loop by replacing the sampling step in `build_critic_artifacts_from_logits`. Specifically, it replaces this line:

```python
# OLD: Naive parallel sampling
sampled = torch.multinomial(flat, num_samples=1).view(probs.size(0), probs.size(1))
```

With:

```python
# NEW: Sampler-based coherent sampling
if hasattr(model, 'sampler_head') and model.sampler_head is not None:
    sampled = sampler_wavefront_fill(model, idx, hidden_states, mask_token_id, temperature, top_p, ...)
else:
    sampled = torch.multinomial(flat, num_samples=1).view(probs.size(0), probs.size(1))
```

### 4.2. Wavefront-Based Coherent Sampling

The sampler fills masked positions in **waves**, where each wave fills tokens that have at least one non-masked neighbor. This ensures local coherence while maintaining reasonable computational efficiency.

**Two-Phase Approach:**
1. **Bootstrap Phase** (if needed): When no masked tokens have non-masked neighbors, fill the top 1% (by logit confidence) using naive sampling to create "seed" tokens
2. **Wavefront Phase**: Iteratively fill masked tokens that have at least one non-masked neighbor, using the sampler head for coherent predictions

**Performance Considerations:**
- The sampler head is much smaller and faster than the main transformer
- Multiple sampler passes may be needed, but each is lightweight
- The hope is that improved sampling quality reduces the number of diffusion iterations needed, offsetting the per-iteration cost
- This is why the sampler is **optional** - it needs empirical validation

```python
def sampler_wavefront_fill(model, tokens, hidden_states, mask_token_id,
                           temperature=1.0, top_p=1.0, vocab_size=None,
                           base_vocab_size=None, min_neighbors_ratio=0.01):
    """
    Fill masked tokens using wavefront-based coherent sampling.

    Fills tokens in waves, where each wave fills positions that have at least
    one non-masked neighbor. This ensures local coherence.

    Args:
        model: GPT model with sampler_head
        tokens: (B, T) current token sequence
        hidden_states: (B, T, n_embd) hidden states from _encode_tokens
        mask_token_id: ID of mask token
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        vocab_size: Full vocabulary size
        base_vocab_size: Base vocabulary size (excluding special tokens)
        min_neighbors_ratio: Minimum ratio of tokens with neighbors to proceed

    Returns:
        filled_tokens: (B, T) tokens with masks filled
    """
    B, T = tokens.shape
    device = tokens.device
    filled = tokens.clone()

    # Maximum waves = sequence length (to prevent infinite loops)
    max_waves = T

    for wave in range(max_waves):
        # Find masked positions
        is_masked = (filled == mask_token_id)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break  # All masks filled

        # Find masked positions with at least one non-masked neighbor
        has_left_neighbor = torch.zeros_like(is_masked)
        has_left_neighbor[:, 1:] = ~is_masked[:, :-1]

        has_right_neighbor = torch.zeros_like(is_masked)
        has_right_neighbor[:, :-1] = ~is_masked[:, 1:]

        # Eligible: masked AND has at least one non-masked neighbor
        eligible = is_masked & (has_left_neighbor | has_right_neighbor)
        num_eligible = eligible.sum().item()

        # Check if we have enough eligible tokens to proceed
        if num_eligible == 0:
            # EDGE CASE: No tokens with neighbors - need to bootstrap
            # This happens when all (or most) tokens are masked
            # Solution: Fill top 1% (min_neighbors_ratio) by confidence using naive sampling
            # This creates "seed" tokens that allow the wavefront to proceed
            eligible = _bootstrap_fill_by_confidence(
                filled, is_masked, hidden_states, model,
                min_neighbors_ratio, mask_token_id
            )
            num_eligible = eligible.sum().item()

            if num_eligible == 0:
                break  # Cannot make progress (should not happen)

        # Fill eligible positions using sampler
        batch_idx, pos_idx = eligible.nonzero(as_tuple=True)

        # Gather hidden states
        h = hidden_states[batch_idx, pos_idx]  # (N, n_embd)

        # Gather left neighbor embeddings (zero if missing or masked)
        left_emb = torch.zeros_like(h)
        left_exists = pos_idx > 0
        if left_exists.any():
            left_batch = batch_idx[left_exists]
            left_pos = pos_idx[left_exists] - 1
            left_ids = filled[left_batch, left_pos]
            left_not_mask = left_ids != mask_token_id
            if left_not_mask.any():
                left_emb[left_exists][left_not_mask] = model.transformer.wte(
                    left_ids[left_not_mask]
                )

        # Gather right neighbor embeddings (zero if missing or masked)
        right_emb = torch.zeros_like(h)
        right_exists = pos_idx < (T - 1)
        if right_exists.any():
            right_batch = batch_idx[right_exists]
            right_pos = pos_idx[right_exists] + 1
            right_ids = filled[right_batch, right_pos]
            right_not_mask = right_ids != mask_token_id
            if right_not_mask.any():
                right_emb[right_exists][right_not_mask] = model.transformer.wte(
                    right_ids[right_not_mask]
                )

        # Forward through sampler
        sampler_input = torch.cat([left_emb, h, right_emb], dim=-1)
        sampler_features = model.sampler_head(sampler_input)
        logits = model.lm_head(sampler_features)

        # Apply vocabulary restrictions
        if vocab_size is not None:
            logits[:, mask_token_id] = float('-inf')
            if base_vocab_size is not None and logits.shape[-1] > base_vocab_size:
                logits[:, base_vocab_size:] = float('-inf')

        # Sample tokens
        from sample_utils import nucleus_sample
        new_tokens = nucleus_sample(logits, top_p=top_p, temperature=temperature)

        # Update filled tokens
        filled[batch_idx, pos_idx] = new_tokens

    return filled

def _bootstrap_fill_by_confidence(filled, is_masked, hidden_states, model,
                                   min_ratio, mask_token_id):
    """
    Bootstrap filling when no masked tokens have non-masked neighbors.

    This handles the edge case where all (or most) tokens are masked, preventing
    the wavefront from starting. We select a small percentage (min_ratio, default 1%)
    of masked positions with the highest logit confidence and mark them for naive
    sampling. These "seed" tokens allow the wavefront to proceed.

    Args:
        filled: (B, T) current filled tokens
        is_masked: (B, T) boolean mask of masked positions
        hidden_states: (B, T, n_embd) hidden states from main model
        model: GPT model
        min_ratio: Ratio of masked tokens to bootstrap (default 0.01 = 1%)
        mask_token_id: ID of mask token

    Returns:
        eligible: (B, T) boolean mask of positions to fill in this bootstrap step
    """
    B, T = filled.shape
    num_masked = is_masked.sum().item()
    num_to_fill = max(1, int(num_masked * min_ratio))

    # Get logits for all masked positions using main lm_head
    batch_idx, pos_idx = is_masked.nonzero(as_tuple=True)
    h = hidden_states[batch_idx, pos_idx]
    logits = model.lm_head(h)

    # Get max logit (confidence) for each position
    # Higher max logit = model is more confident about its top prediction
    max_logits, _ = logits.max(dim=-1)

    # Select top-k positions by confidence
    if len(max_logits) <= num_to_fill:
        # If we have fewer masked tokens than num_to_fill, bootstrap all of them
        eligible = is_masked.clone()
    else:
        # Select top num_to_fill positions by confidence
        _, top_indices = torch.topk(max_logits, num_to_fill)
        eligible = torch.zeros_like(is_masked)
        eligible[batch_idx[top_indices], pos_idx[top_indices]] = True

    return eligible
```

### 4.3. Integration Point: Replacing Naive Sampling in build_critic_artifacts_from_logits

The sampler replaces the naive parallel sampling in `build_critic_artifacts_from_logits`. This function is used during training to create synthetic inputs for the critic.

**Location in codebase**: `sample_utils.py`, function `build_critic_artifacts_from_logits`

**Current naive sampling (line ~XXX)**:
```python
# Sample from the distribution
flat = probs.view(-1, probs.size(-1))
sampled = torch.multinomial(flat, num_samples=1).view(probs.size(0), probs.size(1))
```

**Replacement with sampler**:
```python
# Sample from the distribution
if hasattr(model, 'sampler_head') and model.sampler_head is not None:
    # Use sampler for coherent sampling
    # Note: We already have hidden states from the forward pass
    # hidden_states = model._encode_tokens(idx)  # Already computed earlier
    sampled = sampler_wavefront_fill(
        model=model,
        tokens=idx,  # Current input tokens
        hidden_states=hidden_states,  # From earlier forward pass
        mask_token_id=mask_token_id,
        temperature=1.0,  # Use temperature=1.0 for training
        top_p=1.0,  # No top-p filtering for training
        vocab_size=model.config.vocab_size,
        base_vocab_size=None,  # Allow all tokens during training
        min_neighbors_ratio=getattr(model.config, 'sampler_min_neighbors_ratio', 0.01)
    )
else:
    # Fallback to naive parallel sampling
    flat = probs.view(-1, probs.size(-1))
    sampled = torch.multinomial(flat, num_samples=1).view(probs.size(0), probs.size(1))
```

**Note**: The `build_critic_artifacts_from_logits` function will need to be modified to:
1. Accept the model as a parameter (to check for sampler_head)
2. Accept hidden_states as a parameter (to avoid recomputing)
3. Use the sampler when available

This ensures that during training, the critic sees samples generated with the same coherent sampling strategy that will be used during inference.

## 5. Summary: Key Implementation Points

### 5.1. Architecture Decisions

**Auxiliary Network Design:**
- Sampler is a separate, lightweight MLP trained independently from the main transformer
- All inputs to sampler are **detached** during training to prevent gradient flow to main model
- This is different from critic, which shares the transformer trunk

**Mode and Attention Requirements:**
- Sampler only works with `LANGUAGE_MODEL` mode
- Requires `attention_type='bidirectional'`
- Config validation enforces these requirements at model initialization

**Neighbor Handling:**
- Missing neighbors (boundaries or `[MASK]` tokens) use **zero embeddings**
- This signals to the sampler that no context is available from that direction

### 5.2. Training Schedule

**Three-Stage Training:**
1. **Stage 1** (0 to `start_sampler_iteration`): Main model only
2. **Stage 2** (`start_sampler_iteration` to `start_critic_iteration`): Main model + Sampler
3. **Stage 3** (`start_critic_iteration` onwards): Main model + Sampler + Critic

**Loss Computation:**
- Main loss: Always computed (standard cross-entropy)
- Sampler loss: Separate auxiliary loss (standard cross-entropy), added directly to total loss
- Critic loss: Added with weight `critic_alpha` (with warmup)
- Total loss = main_loss + sampler_loss + (critic_alpha * critic_loss)

**Recommended Configuration:**
```python
add_sampler_head = True
start_sampler_iteration = 0  # Train from beginning
sampler_min_neighbors_ratio = 0.01  # 1% bootstrap threshold

add_critic_head = True
critic_alpha = 0.5
start_critic_iteration = 5000  # Start after sampler stabilizes
end_critic_iteration = 10000  # Full alpha by iteration 10000
```

### 5.3. Inference Strategy

**Wavefront-Based Filling with Bootstrap:**
1. **Check for neighbors**: Identify masked tokens that have at least one non-masked neighbor
2. **Bootstrap if needed**: If no (or too few) tokens have neighbors, use `_bootstrap_fill_by_confidence` to:
   - Select top 1% (configurable) of masked positions by highest logit confidence
   - Fill these positions using naive sampling from lm_head logits
   - These "seed" tokens allow the wavefront to proceed
3. **Wavefront filling**: Fill masked tokens that have neighbors using sampler head for coherent predictions
4. **Iterate**: Repeat until all masks filled or no progress can be made

**Performance Considerations:**
- Multiple forward passes through sampler head (lightweight MLP)
- Bootstrap step uses main lm_head (no sampler) for seed tokens
- Hope: Improved quality reduces total diffusion iterations, offsetting per-iteration cost
- This is **optional** and needs empirical validation

**Integration Point:**
- Replaces naive sampling in `build_critic_artifacts_from_logits`
- Used during both training (for critic) and inference (for generation)

### 5.4. Implementation Checklist

**Model Changes (`model.py`):**
- [ ] Add sampler configuration to `GPTConfig`
- [ ] Add config validation in `__post_init__`
- [ ] Create `SamplerHead` class
- [ ] Add sampler head initialization in `GPT.__init__`
- [ ] Implement `_compute_sampler_loss` method
- [ ] Integrate sampler loss in `_forward_language_model`
- [ ] Add `_last_sampler_loss` tracking for logging

**Utility Functions (`sample_utils.py`):**
- [ ] Implement `prepare_sampler_inputs` function
- [ ] Implement `sampler_wavefront_fill` function
- [ ] Implement `_bootstrap_fill_by_confidence` helper
- [ ] Modify `build_critic_artifacts_from_logits` to use sampler
- [ ] Update function signatures to pass model and hidden_states

**Training Integration (`core/training_step.py` or `train.py`):**
- [ ] Ensure `_current_iter` is set on model for stage gating
- [ ] Add sampler loss to logging metrics
- [ ] Verify three-stage training schedule works correctly

**Testing:**
- [ ] Unit test: `SamplerHead` forward pass
- [ ] Unit test: `prepare_sampler_inputs` with various inputs
- [ ] Unit test: `sampler_wavefront_fill` with simple cases
- [ ] Integration test: Training with sampler enabled
- [ ] Integration test: Inference with sampler enabled
- [ ] Backward compatibility test: Models without sampler still work
- [ ] Checkpoint loading test: Old checkpoints load correctly

### 5.5. Open Questions and Future Work

**Questions for Empirical Validation:**
1. Does sampler improve generation quality measurably?
2. What is the performance impact (tokens/sec)?
3. Does improved quality reduce required diffusion iterations?
4. Should sampler have its own warmup schedule (like critic)?
5. What is the optimal `sampler_min_neighbors_ratio` for bootstrap?

**Potential Enhancements:**
1. Separate sampler heads for training vs inference
2. Adaptive wavefront ordering (beyond just "has neighbor")
3. Sampler-specific temperature/top-p parameters
4. Multi-scale sampler (different context windows)
5. Dynamic bootstrap ratio based on masking density

**Monitoring Metrics:**
- Sampler loss over training (should decrease as sampler learns)
- Bootstrap frequency (how often bootstrap is needed during inference)
- Ratio of tokens filled per wave during inference
- Number of waves needed to fill all masks
- Generation quality metrics (judge scores, human eval)
- Inference speed (tokens/sec, time per sample)