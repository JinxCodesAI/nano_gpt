> **Status:** Archived design memo. Superseded by the production
> implementation described in [`grpo/README.md`](../../grpo/README.md). This
> document remains for historical context and explains the initial plan that
> shaped the current pipeline.

Here is a detailed, step-by-step implementation plan in Markdown format,
explaining what to do, why it's necessary, and how to do it within your
existing codebase.

### **Phase 0: Conceptual Overview & Goal**

**What:** We will create a new training script that implements the Group-Relative Policy Optimization (GRPO) loop. This loop will not use a fixed dataset. Instead, at each step, it will generate its own training data.

**Why:** Your Generator model has learned the general patterns of Shakespeare, but you want to refine its quality beyond what the small, static dataset can teach it. GRPO allows the model to "explore" the space of possible Shakespearean text, and your Judge model will provide the "compass" that guides this exploration towards higher-quality outputs.

**How:** The core loop will be:
1.  **Prompt:** Create a starting point (e.g., a fully masked sequence or a `seed_text`).
2.  **Generate:** Use the Generator to produce a group of `k` candidate sequences from that prompt.
3.  **Judge:** Use the frozen Judge model to assign a reward score to each of the `k` candidates.
4.  **Learn:** Calculate the "advantage" for each candidate (how much better/worse it is than the group average). Use this advantage to update the Generator's weights, making it more likely to produce high-advantage sequences in the future.

---

### **Phase 1: Create the GRPO Training Script (`train_grpo.py`)**

**What:** A new Python script to orchestrate the GRPO training process. This script will be the main entry point for your new training regime.

**Why:** GRPO training has a fundamentally different structure than standard supervised training. It doesn't iterate over a dataset but rather generates its data on-the-fly. This requires a dedicated script to manage the two models (Generator and Judge) and the unique training loop.

**How:**
1.  **Copy and Modify `train.py`:** Create a new file `train_grpo.py` by copying your existing `train.py`.
2.  **Model Loading:**
    *   Load your **Generator** model exactly as you do now. This is the model we will be training, so we'll call it `generator_model`.
    *   Load your **Judge** model using `load_model_from_checkpoint`. We'll call it `judge_model`.
    *   **Crucially, freeze the Judge model:** After loading it, ensure its weights will not be updated.
        ```python
        # In train_grpo.py, after loading the judge_model
        judge_model.eval() # Set to evaluation mode
        for param in judge_model.parameters():
            param.requires_grad = False
        print("Judge model weights have been frozen.")
        ```
3.  **Optimizer Setup:** The optimizer should only be configured for the `generator_model`'s parameters.
    ```python
    # In train_grpo.py
    optimizer = generator_model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    ```
4.  **Remove Data Consumer:** You will not need the `DataConsumer` that loads batches from a file. Remove all code related to `get_batch`. The data will be generated inside the training step itself.
5.  **Instantiate a New `GRPOTrainingStep`:** The existing `TrainingStep` class is designed for supervised learning. We will create a new one for GRPO.
    ```python
    # In train_grpo.py, replace the existing TrainingStep
    from grpo_step import GRPOTrainingStep # We will create this file next

    grpo_step = GRPOTrainingStep(
        generator_model=generator_model,
        judge_model=judge_model,
        optimizer=optimizer,
        # ... add GRPO-specific config here ...
    )

    trainer = Trainer(
        model=generator_model, # The trainer still needs a reference to the model being trained
        optimizer=optimizer,
        training_step=grpo_step, # Use our new step logic
        # ... other trainer params ...
    )
    trainer.train()
    ```

---

### **Phase 2: Implement the Core Logic (`grpo_step.py`)**

**What:** A new file `grpo_step.py` containing a `GRPOTrainingStep` class. This class will encapsulate the entire GRPO logic for a single training iteration, cleanly separating it from the `Trainer` orchestration.

**Why:** This follows good software design. The `Trainer` class handles the high-level loop (timing, logging, checkpointing), while the `GRPOTrainingStep` class is responsible for the specific algorithm: generating data, calculating loss, and performing the backward pass.

**How:**

Create a new file `grpo_step.py`:
```python
import torch
from torch.nn import functional as F
from sample import diffusion_generate, calculate_judge_scores # Reuse your existing functions!

class GRPOTrainingStep:
    def __init__(self, generator_model, judge_model, optimizer, config):
        self.generator_model = generator_model
        self.judge_model = judge_model
        self.optimizer = optimizer
        self.config = config # Pass in GRPO configs like group_size, sequence_length, etc.

    def execute_step(self):
        """
        Performs one full GRPO training step.
        This will replace the logic of getting a batch and calculating cross-entropy loss.
        """
        self.optimizer.zero_grad()

        # STEP 1: GENERATE A GROUP OF CANDIDATES
        # The "prompt" for your diffusion model is a fully masked sequence.
        prompt = torch.full((self.config['group_size'], self.config['sequence_length']),
                            self.config['mask_token_id'], dtype=torch.long, device=self.config['device'])

        # Generate without tracking gradients for the generation process itself.
        with torch.no_grad():
            generated_tokens = diffusion_generate(
                model=self.generator_model,
                batch_size=self.config['group_size'],
                # ... other diffusion params from your config ...
            )

        # STEP 2: JUDGE THE GROUP
        with torch.no_grad():
            rewards = calculate_judge_scores(
                judge_model=self.judge_model,
                tokens=generated_tokens,
                device=self.config['device']
            )

        # STEP 3: CALCULATE ADVANTAGE
        baseline = rewards.mean()
        advantages = rewards - baseline
        # Optional but recommended: normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # STEP 4: CALCULATE THE GRPO LOSS
        # This requires getting the log-probability of each generated sequence.
        # We need to add a new method to your GPT model for this (see Phase 3).
        log_probs = self.generator_model.get_log_probabilities(prompt, generated_tokens)

        # The GRPO loss function. We use .detach() on advantages because they are
        # treated as fixed constants for this step; we do not want to backpropagate into the judge.
        loss = -torch.mean(log_probs * advantages.detach())

        # STEP 5: BACKPROPAGATION AND OPTIMIZER STEP
        loss.backward()
        # Optional: Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(self.generator_model.parameters(), self.config['grad_clip'])
        self.optimizer.step()

        return loss, None, None # Return loss and dummy X, Y to satisfy the Trainer's interface
```

---

### **Phase 3: Modify the Model (`model.py`)**

**What:** Add a new method `get_log_probabilities` to your `GPT` class.

**Why:** The GRPO loss function is `Loss = -log π(action) * Advantage`. The `log π(action)` term represents the log-probability of the model generating a specific sequence. Your current `forward` method calculates logits for training, but we need a dedicated function to efficiently calculate the probability of an entire generated sequence given a starting prompt.

**How:**

Add the following method inside the `GPT` class in `model.py`:
```python
    # Inside the GPT class in model.py

    def get_log_probabilities(self, inputs, targets):
        """
        Calculates the total log-probability of generating the target sequences,
        conditioned on the input sequences (which are fully masked for your use case).

        Args:
            inputs (Tensor): The input sequences to the model (b, t). In your case, the [MASK] prompt.
            targets (Tensor): The complete generated sequences (b, t).

        Returns:
            Tensor: A tensor of shape (b,) containing the sum of log-probabilities for each sequence.
        """
        # Get logits for all positions. Here we are re-using the standard forward pass.
        # This is a bit inefficient as it calculates logits for all vocab, but it's the simplest way.
        # The 'inputs' here are technically not needed if your model is purely bidirectional
        # and doesn't use the input tokens to predict masked ones, but it's good practice.
        # Since your diffusion model likely uses the full sequence, we pass `targets`.
        logits, _ = self.forward(targets) # Shape: (b, t, vocab_size)

        # We need to get the log-probability of the specific token that was generated at each position.
        log_softmax_logits = F.log_softmax(logits, dim=-1)

        # The `targets` tensor contains the token IDs for each position.
        # We can use `gather` to pick out the log-prob for the target token at each position.
        # We need to add a dimension to `targets` to match the shape for gather.
        target_log_probs = torch.gather(log_softmax_logits, 2, targets.unsqueeze(-1)).squeeze(-1) # Shape: (b, t)

        # The total log-probability of a sequence is the sum of the log-probabilities of its tokens.
        # We only care about the positions that were actually generated (not part of a fixed seed).
        # For simplicity now, we sum over all tokens. You can add a mask here later if needed.
        sequence_log_probs = target_log_probs.sum(dim=1) # Shape: (b,)

        return sequence_log_probs
```

### **Phase 4: Putting It All Together & Configuration**

1.  **Update Config:** Add new parameters to your configuration for GRPO training.
    *   `group_size`: The `k` value for GRPO (e.g., 8 or 16).
    *   `generator_checkpoint`: Path to the generator model to be trained.
    *   `judge_checkpoint`: Path to the frozen judge model.
    *   `grpo_learning_rate`: A separate, likely smaller, learning rate for GRPO fine-tuning.
    *   ... any other diffusion parameters needed by `diffusion_generate`.

2.  **Update `train_grpo.py`:** Make sure the script reads these new configuration values and passes them into the `GRPOTrainingStep` class.

3.  **Run `train_grpo.py`:** Execute your new script to begin GRPO training. Monitor the loss and periodically run `sample.py` with the updated generator checkpoints to see if the quality of generations is improving.

This structured approach reuses the maximum amount of your existing, working code (`diffusion_generate`, `calculate_judge_scores`) and encapsulates the new GRPO logic cleanly, making it easy to maintain and debug.