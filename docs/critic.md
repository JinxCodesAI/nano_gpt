## The Rationale: From Heuristics to a Learned Policy

The core of your diffusion process is the re-masking strategy. Your current method uses a smart **heuristic**: re-mask the tokens the model is least confident about. While effective, this logic is fixed.

The goal of this implementation is to replace that fixed heuristic with a **learned policy**. You will train a second model head—a **"Critic"**—whose specific job is to look at a generated sequence and predict which tokens are likely errors. This Critic head can then directly guide the re-masking process during inference, using a data-driven strategy learned during training.

This is a multi-task learning setup where a shared transformer body learns to both **generate** text (the "Generator" head) and **critique** its own output (the "Critic" head).

-----

## Step 1: Modify the Model (`GPT` class in `model.py`)

First, we need to add the new Critic head to the model architecture.

### **Why?**

The model needs a dedicated, trainable component to output the error predictions. This head will take the final hidden states from the transformer and map them to a single score per token, indicating the likelihood of that token being an error.

### **How to Implement:**

1.  **Add the Critic Head in `__init__`:**
    In the `GPT` class `__init__` method, alongside the `lm_head` (our Generator), add a `critic_head`. It should be a simple linear layer that outputs one logit per token for binary classification.

    ```python
    class GPT(nn.Module):
        def __init__(self, config, logger=None):
            super().__init__()
            # ... (existing code for self.config, self.transformer) ...

            # Head for language modeling (The "Generator")
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight

            # New head for error detection (The "Critic")
            # Outputs a single logit per token for binary classification (correct vs. error)
            self.critic_head = nn.Linear(config.n_embd, 1, bias=False)

            # ... (rest of the __init__ method, weight initialization, etc.) ...
    ```

2.  **Update the Main `forward` Method:**
    Modify the `forward` method to handle routing to the different heads. We also need to accommodate the two separate forward passes that will happen during training. The logic for different `ModelMode`s should be refactored to handle this new multi-head setup.

    I suggest adding a `head_to_use` argument to select which head's output is required.

    ```python
    class GPT(nn.Module):
        # ... (__init__ and other methods) ...

        def forward(self, idx, targets=None, attention_mask=None, loss_modifiers=None, head_to_use='lm'):
            # ... (existing forward logic up to the final layer norm) ...
            # x = self.transformer.ln_f(x)

            if head_to_use == 'critic':
                # Return logits from the critic head for error prediction
                logits = self.critic_head(x)
                loss = None # Loss is calculated manually in the training loop for the critic
                return logits, loss

            # Default behavior is to use the language model head ('lm')
            else: # LANGUAGE_MODEL, TOKEN_CLASSIFIER, etc.
                if self.config.mode == ModelMode.SEQUENCE_SCORER:
                    return self._forward_sequence_scorer(x, targets, loss_modifiers)
                elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
                    return self._forward_token_classifier(x, targets, loss_modifiers)
                else:  # LANGUAGE_MODEL
                    return self._forward_language_model(x, targets, loss_modifiers)

    ```

    *Note: For simplicity, I've shown the critic as a separate path. You can integrate it more deeply with your existing `ModelMode` enum if you prefer.*

-----

## Step 2: Update the Training Loop

Now, we'll implement the two-part forward pass and the combined loss calculation in your training script.

### **Why?**

The model needs to be trained on both tasks simultaneously. We will first train the Generator to predict tokens, then use its (sampled) output to create a training target for the Critic in the same training step.

### **How to Implement:**

Inside your training loop, for each batch:

1.  **Forward Pass 1: Train the Generator**
    This is your standard training step. The model predicts the next token from a masked input.

    ```python
    # Assume X is masked input, Y is ground truth
    logits_gen, loss_gen = model(X, targets=Y, head_to_use='lm')
    ```

2.  **Sample from Generator Output**
    To create an input for the Critic, we need a complete (but likely imperfect) sequence. Sample from the Generator's logits. This step does not require gradients.

    ```python
    with torch.no_grad():
        probs = F.softmax(logits_gen, dim=-1)
        # Reshape for multinomial: (batch_size * sequence_length, vocab_size)
        probs_flat = probs.view(-1, probs.size(-1))
        predicted_tokens_flat = torch.multinomial(probs_flat, num_samples=1)
        # Reshape back to (batch_size, sequence_length)
        predicted_tokens = predicted_tokens_flat.view(X.size())

        # Fill in the original masked positions in X with the new predictions
        # to create a full sequence for the critic.
        critic_input = X.clone()
        mask = (X == mask_token_id) # Or however you track masked positions
        critic_input[mask] = predicted_tokens[mask]
    ```

3.  **Create the Target for the Critic**
    The Critic's job is to spot errors. The ground truth for this is where the `critic_input` differs from the original `Y`.

    ```python
    # Target is 1 where the prediction was an error, 0 otherwise
    critic_target = (critic_input != Y).float().unsqueeze(-1) # Add final dim to match critic output
    ```

4.  **Forward Pass 2: Train the Critic**
    Feed the `critic_input` back into the model, but this time, get the logits from the `critic_head`.

    ```python
    logits_critic, _ = model(critic_input, head_to_use='critic')
    ```

5.  **Calculate Critic Loss and Combine**
    Use a binary classification loss, like `BCEWithLogitsLoss`, which is numerically stable. Then, combine it with the generator loss.

    ```python
    # Use a mask to compute loss only on non-padded tokens if necessary
    loss_critic = F.binary_cross_entropy_with_logits(logits_critic, critic_target, reduction='mean')

    # Combine the losses. `alpha` is a hyperparameter to balance the two tasks.
    alpha = 0.5 # Example value, tune this
    total_loss = loss_gen + alpha * loss_critic
    ```

6.  **Backward Pass**
    Perform the backward pass on the combined `total_loss`. Autograd will handle the rest, sending the correct gradients to the shared body and each head.

    ```python
    total_loss.backward()
    optimizer.step()
    ```

-----

## Step 3: Update the Inference/Sampling Logic (`sample.py`)

Finally, use your new, trained Critic head to guide the re-masking process during diffusion generation.

### **Why?**

The whole point of training the Critic was to create a better re-masking policy. We now replace the `1 - confidence` heuristic with the direct output of the Critic head.

### **How to Implement:**

1.  **Modify the Re-masking Function:**
    Create a new function or modify `_select_tokens_with_confidence` to use the critic.

    ```python
    def _select_tokens_with_critic(tokens, num_to_mask, model, randomness_strength, device):
        """Select tokens to mask based on the learned Critic head."""
        model.eval()
        with torch.no_grad():
            # Get scores directly from the critic head
            critic_logits, _ = model(tokens, head_to_use='critic')
            # The critic outputs logits; higher values mean higher chance of being an error.
            uncertainty_scores = critic_logits.squeeze() # Shape: (batch_size, seq_len)

        # For simplicity, let's assume batch size of 1 for this logic
        uncertainty_scores = uncertainty_scores[0]

        # Blend with randomness (same as your original function)
        if randomness_strength > 0:
            random_scores = torch.rand_like(uncertainty_scores)
            combined_scores = (1 - randomness_strength) * uncertainty_scores + randomness_strength * random_scores
        else:
            combined_scores = uncertainty_scores

        # Select tokens with the highest predicted error scores
        # Ensure you don't re-mask already protected tokens (e.g., seed text)
        # This part requires integrating with your `unmaskable_indices` logic
        _, selected_indices = torch.topk(combined_scores, num_to_mask, largest=True)

        return selected_indices
    ```

2.  **Integrate into `diffusion_generate`:**
    In your main generation loop, after the sampling step, call your new critic-based re-masking function.

    ```python
    # Inside the diffusion_generate loop...
    # After predicting and sampling tokens into the `tokens` tensor...

    if iteration < iterations - 1:
        # Determine number of tokens to mask for this iteration based on schedule
        num_to_remask = ...

        # Use the critic to decide WHICH tokens to mask
        indices_to_mask = _select_tokens_with_critic(tokens, num_to_remask, model, ...)

        # Apply the new mask
        tokens[0, indices_to_mask] = mask_token_id
    ```