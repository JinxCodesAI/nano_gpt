Of course. A dedicated logging plan is a crucial part of any serious modeling effort. This document provides a clear set of guidelines and ready-to-use code snippets for implementing comprehensive diagnostics to monitor the health and behavior of the new `[MASK]` / `[WRONG]` model.

---

# `logging_guidelines.md`

## Subject: Diagnostic Logging Guidelines for the `[MASK]` / `[WRONG]` Diffusion Model

### 1. Introduction

This document outlines a comprehensive logging strategy to monitor the training process of our refactored, two-token diffusion model. The primary goal of this logging is to provide clear, quantitative insights into the model's internal state and behavior, allowing us to verify its health and diagnose any potential issues, such as the pathological "logit collapse" observed in previous versions.

The logging is designed to be added primarily within the `calculate_diffusion_loss` function in `train.py`, controlled by a `log_diagnostics` flag. This allows us to activate detailed logging for training steps without affecting the performance of the evaluation loop.

### 2. Core Diagnostic Logs

This section details the most critical metrics to implement, explaining what each one measures and how to interpret it.

#### 2.1. Logit Sanity Check (Detecting Logit Collapse)
*   **Goal:** To directly verify that the new two-token architecture has solved the "logit gap" problem. We need to monitor the raw logit values for our special tokens relative to the model's confident predictions.
*   **Metrics:**
    1.  `Avg MASK Logit`: The average raw logit value for the `[MASK]` token across all positions in the batch.
    2.  `Avg WRONG Logit`: The average raw logit value for the `[WRONG]` token.
    3.  `Avg MAX Logit`: The average of the maximum logit value at each position (i.e., the model's top guess).
*   **Interpretation:**
    *   **Healthy State:** All three values should be in a similar numerical range. We expect `Avg MAX Logit` to be the highest, but the gap between it and the special tokens should be small (e.g., < 5-10 points), not catastrophic (`> 50`).
    *   **Problem State (Logit Collapse):** If `Avg MASK Logit` or `Avg WRONG Logit` becomes a large negative number while `Avg MAX Logit` remains high, it signals that the model is relearning the old pathological bias.

#### 2.2. Un-masking Confidence and Accuracy
*   **Goal:** To understand the model's performance and confidence on the core generative task (`Input=[MASK]`, `Target=word`). We need to distinguish between low-confidence mistakes and high-confidence mistakes.
*   **Metrics:**
    1.  `Unmask Accuracy`: The percentage of un-masking tasks where the model's top prediction (`argmax`) was correct.
    2.  **`Avg Prob of WRONG Guess`**: When the model attempts to unmask but is incorrect, what was the average probability it assigned to its (wrong) top choice?
    3.  **`Skill vs Random`**: A ratio comparing the `Unmask Accuracy` to the accuracy of a random guess (`1 / vocab_size`).
*   **Interpretation:**
    *   **Healthy State:** We expect `Unmask Accuracy` to start low and gradually increase. `Avg Prob of WRONG Guess` should be relatively low. A low accuracy with low confidence is acceptable during early training. The model is "trying and failing softly."
    *   **Problem State:** A high `Avg Prob of WRONG Guess` (e.g., > 50%) combined with a low `Unmask Accuracy` is a major red flag. It means the model is "confidently wrong," which is a difficult state to learn from.

#### 2.3. Task Behavior and Token Preference
*   **Goal:** To get a high-level overview of the model's strategic choices. Is it favoring one action over another?
*   **Metrics:**
    1.  `Correct Unmasks`: Raw count of successful un-masking guesses.
    2.  `Incorrect Unmasks`: Raw count of failed un-masking guesses.
    3.  `Kept Mask`: Raw count of times the model's top guess for an un-masking task was to predict `[MASK]` again (i.e., being "lazy").
    4.  `Correct Re-masks`: Raw count of times the model correctly predicted `[WRONG]` for a corrupted input token.
    5.  **`WRONG Preference`**: A ratio of the model's output frequency of `[WRONG]` to the target frequency of `[WRONG]` in the batch.
*   **Interpretation:**
    *   **Healthy State:** We expect the `Kept Mask` count to be non-zero, indicating the model is capable of strategic inaction. `WRONG Preference` should hover around `1.0`, meaning the model is learning to predict the `[WRONG]` token at roughly the correct rate.
    *   **Problem State:** `Kept Mask` being consistently zero suggests a return to greedy behavior. `WRONG Preference` being very low (`< 0.1`) would indicate the model is struggling to learn the "proofreading" task.

### 3. Implementation Plan: A Unified Logging Function

To keep the main loss function clean, all diagnostic logic should be encapsulated in a single helper function.

**Action:** In `train.py`, this function can be used to replace any previous diagnostic helpers. It should be called from within `calculate_diffusion_loss` when `log_diagnostics=True`.

```python
# In train.py

def log_diffusion_diagnostics(logits, targets, inputs, mask_token_id, wrong_token_id, meta_vocab_size):
    """
    A unified function to calculate and print all key diagnostic metrics
    for the two-token diffusion model.
    """
    with torch.no_grad():
        # --- 1. Logit Sanity Check ---
        avg_mask_logit = logits[:, :, mask_token_id].mean().item()
        avg_wrong_logit = logits[:, :, wrong_token_id].mean().item()
        avg_max_logit, predictions = logits.max(dim=-1)
        avg_max_logit = avg_max_logit.mean().item()

        print(
            f"[LOGITS] Avg MASK: {avg_mask_logit:7.2f} | "
            f"Avg WRONG: {avg_wrong_logit:7.2f} | "
            f"Avg MAX: {avg_max_logit:7.2f}"
        )

        # --- Setup for Behavioral Analysis ---
        epsilon = 1e-6
        flat_inputs = inputs.view(-1)
        flat_targets = targets.view(-1)
        flat_predictions = predictions.view(-1)
        
        # --- 2. Un-masking Confidence and Accuracy ---
        unmask_task_mask = (flat_inputs == mask_token_id) & (flat_targets != wrong_token_id)
        total_unmask_tasks = unmask_task_mask.sum().item()
        
        if total_unmask_tasks > 0:
            # Get the model's predictions ONLY on the tokens that were part of an un-masking task
            unmask_predictions = flat_predictions[unmask_task_mask]
            unmask_targets = flat_targets[unmask_task_mask]
            
            # Accuracy Metrics
            correct_unmasks_mask = (unmask_predictions == unmask_targets)
            correct_unmasks_count = correct_unmasks_mask.sum().item()
            incorrect_unmasks_count = total_unmask_tasks - correct_unmasks_count
            
            unmask_accuracy = correct_unmasks_count / (total_unmask_tasks + epsilon)
            skill_vs_random = unmask_accuracy / (1.0 / meta_vocab_size if meta_vocab_size is not None else 1.0)

            # Confidence of WRONG Guesses
            incorrect_unmask_positions = unmask_task_mask.nonzero(as_tuple=True)[0][~correct_unmasks_mask]
            if len(incorrect_unmask_positions) > 0:
                # Get the probabilities for the model's (wrong) top guess at these positions
                incorrect_logits = logits.view(-1, logits.size(-1))[incorrect_unmask_positions]
                incorrect_probs = F.softmax(incorrect_logits, dim=-1)
                confidence_in_wrong_guess = incorrect_probs.max(dim=-1).values.mean().item()
            else:
                confidence_in_wrong_guess = 0.0
        else:
            unmask_accuracy, skill_vs_random, confidence_in_wrong_guess = 0.0, 0.0, 0.0

        print(
            f"[UNMASK] Accuracy: {unmask_accuracy:<6.2%} | "
            f"Skill: {skill_vs_random:<5.1f}x | "
            f"Conf. on Wrong: {confidence_in_wrong_guess:<6.2%}"
        )
        
        # --- 3. Task Behavior and Token Preference ---
        remask_task_mask = (flat_inputs != mask_token_id) & (flat_targets == wrong_token_id)
        total_remask_tasks = remask_task_mask.sum().item()
        
        # How many times did the model correctly predict [WRONG]?
        correct_remasks_count = (remask_task_mask & (flat_predictions == wrong_token_id)).sum().item()
        remask_accuracy = correct_remasks_count / (total_remask_tasks + epsilon)

        # How often does the model output [WRONG] compared to how often it should have?
        output_wrong_rate = (flat_predictions == wrong_token_id).float().mean().item()
        target_wrong_rate = (flat_targets == wrong_token_id).float().mean().item()
        wrong_preference = output_wrong_rate / (target_wrong_rate + epsilon)

        print(
            f"[REMASK] Accuracy: {remask_accuracy:<6.2%} | "
            f"WRONG Pref.: {wrong_preference:<5.2f}"
        )
```

**How to Integrate:**
1.  Add this function to `train.py`.
2.  In your new `calculate_diffusion_loss`, add a call to `log_diffusion_diagnostics(...)` inside the `if log_diagnostics:` block. Ensure you pass `meta_vocab_size` down to it.