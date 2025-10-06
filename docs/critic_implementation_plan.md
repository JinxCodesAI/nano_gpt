The goal is to modify your `GPT` model so you can easily switch between and test three different behaviors for the `critic_head` in `LANGUAGE_MODEL` mode:

1.  **`NONE`**: The default, with no critic involved.
2.  **`TARGETLESS`**: Your idea, where the critic's normalized output directly weights the loss.
3.  **`TARGETED`**: My idea, where the critic is trained against an explicit confidence target.

-----

### Step 1: Configure the Experiment in `GPTConfig` ⚙️

First, we'll update the configuration class to be clean and simple.

1.  **Add a `CriticMode` Enum:** At the top of your file, near `ModelMode`, add this new `Enum` to define our three options.

    ```python
    class CriticMode(Enum):
        """Defines the training mode for the critic head"""
        NONE = "none"
        TARGETLESS = "targetless"
        TARGETED = "targeted"
    ```

2.  **Update `GPTConfig`:** Find your `GPTConfig` dataclass and replace the entire "Optional critic head configuration" section with the following. This removes the old, obsolete parameters and adds our new `critic_mode` selector.

    ```python
    # In GPTConfig...

    # --- DELETE THIS SECTION ---
    # add_critic_head: bool = False
    # critic_alpha: float = 0.5
    # critic_target_scope: str = 'masked_and_ignore'
    # mask_token_id: int = None
    # pad_token_id: int = None

    # --- ADD THIS NEW SECTION ---
    # Optional critic head configuration
    critic_mode: CriticMode = CriticMode.NONE
    critic_alpha: float = 0.5 # Weight for the critic's own loss ONLY in TARGETED mode

    # This part can stay as is
    # Critic alpha warmup
    start_critic_iteration: int = 0
    end_critic_iteration: int = 0
    ```

-----

### Step 2: Upgrade the Critic Head in `GPT.__init__`

We'll make the critic more expressive and ensure its output is a normalized `[0, 1]` confidence score.

1.  **Find the critic initialization:** In the `GPT.__init__` method, locate the section that initializes the `critic_head`.

2.  **Replace it:** Replace the existing `if` block and the simple `nn.Linear` with this new version, which creates a small MLP with a Sigmoid output.

    ```python
    # In GPT.__init__...

    # --- DELETE THIS LINE ---
    # if getattr(self.config, 'add_critic_head', False) and self.config.mode == ModelMode.LANGUAGE_MODEL:

    # --- REPLACE IT AND THE CRITIC HEAD WITH THIS ---
    # Optional critic head for LANGUAGE_MODEL multi-tasking
    if self.config.critic_mode != CriticMode.NONE and self.config.mode == ModelMode.LANGUAGE_MODEL:
        self.critic_head = nn.Sequential(
            nn.Linear(self.config.n_embd, self.config.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.config.n_embd // 2, 1),
            nn.Sigmoid() # This ensures the output is a [0, 1] confidence score
        )
        self._log_info(f"Critic head enabled (mode={self.config.critic_mode.value})")
    ```

-----

### Step 3: Implement the Core Logic in `_forward_language_model` 🧠

This is the main surgery. We will completely replace the old critic logic with the new, clean, branching logic.

1.  **Remove the old critic logic:** In `_forward_language_model`, find and **delete the entire `if` block** that begins with `if getattr(self.config, 'add_critic_head', False) ...`. This block contains the call to `build_critic_artifacts_from_logits` and all the associated logic.

2.  **Insert the new, unified logic:** In its place, insert the following code. It will handle all three critic modes.

    ```python
    # In _forward_language_model, right after the line:
    # logits = self.lm_head(x)

    # --- START OF NEW UNIFIED LOSS LOGIC ---

    # This is the base calculation, common to all modes.
    # It gets the standard per-token loss without any weighting.
    per_token_lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=self.config.ignore_index,
        reduction='none'
    ).view(targets.shape)

    mask = (targets != self.config.ignore_index)
    self._last_critic_loss = 0.0 # Will be updated by TARGETED mode if active

    # --- Branching logic based on the config setting ---

    if self.config.critic_mode == CriticMode.TARGETLESS:
        # ---- YOUR IDEA ----
        predicted_confidence = self.critic_head(x).squeeze(-1)
        
        # Normalize weights to have a mean of 1 across each sequence in the batch
        valid_confidences = predicted_confidence * mask.float()
        num_valid_tokens = mask.float().sum(dim=1, keepdim=True) + 1e-8
        mean_confidence = valid_confidences.sum(dim=1, keepdim=True) / num_valid_tokens
        
        # We detach the normalization factor to stop the model from indirectly
        # manipulating the mean to its advantage.
        loss_weights = predicted_confidence / mean_confidence.detach()

        # The gradient flows through 'loss_weights' to train the critic
        weighted_lm_loss = per_token_lm_loss * loss_weights
        loss = (weighted_lm_loss * mask.float()).sum() / mask.float().sum()

    elif self.config.critic_mode == CriticMode.TARGETED:
        # ---- MY IDEA ----
        # 1. Train the Critic: It must predict the model's true confidence (1 - entropy)
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * F.log_softmax(logits, dim=-1), dim=-1)
            normalized_entropy = entropy / math.log(self.config.vocab_size)
            true_confidence = 1.0 - normalized_entropy
        
        predicted_confidence = self.critic_head(x).squeeze(-1)
        # Critic has its own, separate loss against the explicit target
        loss_critic = F.mse_loss(predicted_confidence[mask], true_confidence[mask])
        self._last_critic_loss = float(loss_critic.detach().item())
        
        # 2. Use the Critic to weight the LM loss
        # We detach the critic's prediction so the LM can't cheat
        loss_weights = 1.0 + predicted_confidence.detach() 
        weighted_lm_loss = per_token_lm_loss * loss_weights
        final_lm_loss = (weighted_lm_loss * mask.float()).sum() / mask.float().sum()
        
        alpha_eff = self._effective_critic_alpha()
        loss = final_lm_loss + alpha_eff * loss_critic

    else: # This covers CriticMode.NONE
        # ---- NO CRITIC: Standard Cross Entropy Loss ----
        loss = per_token_lm_loss.sum() / mask.float().sum()
        
    # --- END OF NEW UNIFIED LOSS LOGIC ---

    # This section for logging can be updated to handle the new `final_lm_loss` variable
    try:
        lm_loss_val = final_lm_loss if self.config.critic_mode == CriticMode.TARGETED else loss
        self._last_lm_loss = float(lm_loss_val.detach().item())
    except Exception:
        self._last_lm_loss = 0.0
    ```

-----

### Step 4: Final Cleanup

1.  **Remove the unused import:** At the top of your file, delete the line:
    `from sample_utils import build_critic_artifacts_from_logits`
