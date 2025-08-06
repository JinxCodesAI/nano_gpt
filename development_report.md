# Development Summary Report: Diffusion-Style Language Model

## 1. Project Objective

The initial objective of this project was to adapt a standard autoregressive Generative Pre-trained Transformer (GPT) architecture into a non-autoregressive, iterative text generation model. The desired model would operate on principles analogous to diffusion models in computer vision.

The planned methodology involved the following key modifications:
1.  **Architectural Change:** Transition from unidirectional, causal self-attention to bidirectional self-attention.
2.  **Vocabulary Extension:** Introduce a `[MASK]` token to represent a state of generative uncertainty.
3.  **New Training Task:** Replace the next-token-prediction objective with a denoising objective. The model would be trained to reconstruct clean text from an artificially corrupted version. The corruption process included two distinct operations:
    *   **Masking:** Replacement of tokens with the `[MASK]` token (an "un-masking" task for the model).
    *   **Corruption:** Replacement of tokens with incorrect random tokens (a "re-masking" task for the model).
4.  **Iterative Inference:** Design a generation process that begins with a fully masked sequence and uses repeated forward passes of the model to iteratively refine the text.

## 2. Development Timeline and Iterations

The implementation of the initial concept revealed a series of emergent behaviors in the model that required iterative refinement of the training process.

### Iteration 1: Addressing "Greedy" Inference Behavior
*   **Observation:** The baseline model exhibited a "greedy" inference strategy, attempting to resolve all `[MASK]` tokens in a single step, resulting in low-quality output. The model did not effectively utilize the iterative refinement process.
*   **Hypothesis:** The standard cross-entropy loss function provided insufficient incentives to control the model's generative strategy.
*   **Action Taken:** A custom loss function was implemented with three distinct mechanisms:
    1.  A `penalty_keep_mask` to apply a discounted loss when the model correctly identified its uncertainty by outputting `[MASK]`.
    2.  A `penalty_mask_correct` to apply a discounted loss for the less severe error of masking an already-correct token.
    3.  A `scaling_factor`, derived from a `target_unmask_rate`, to dynamically increase the penalty for incorrect guesses if the rate of guessing was too high.

### Iteration 2: Correcting for `argmax`-Based Logic
*   **Observation:** The greedy behavior persisted despite the weighted loss function. Diagnostic logging revealed that the `scaling_factor` was saturated at its maximum value, indicating the incentive structure was ineffective.
*   **Root Cause Analysis:** A fundamental flaw was identified in the loss calculation. The weighting logic was based on the model's "hard" prediction, determined by `torch.argmax(logits)`. This practice discards the rich probabilistic information contained in the full logit distribution and creates a discontinuous, "gameable" loss landscape.
*   **Action Taken:** The loss function was refactored to remove all weighting logic dependent on `argmax`. The new design would rely on a different mechanism to control the greedy vs. lazy trade-off.

### Iteration 3: Diagnosing and Treating the "Logit Gap"
*   **Observation:** Even with the `argmax`-based weighting removed, the model remained pathologically greedy. Diagnostic logging of the raw logits was implemented. The logs revealed a severe "logit gap": the model's average raw logit for the `[MASK]` token was significantly lower than its average maximum logit for real words.
*   **Hypothesis:** The model had learned a strong internal bias against predicting the `[MASK]` token, rendering the penalty structure ineffective because the base cross-entropy loss for predicting `[MASK]` was pathologically high.
*   **Action Taken:** A self-regulating, dynamic `mask_logit_bias` was introduced.
    1.  A `logit_bias` is added directly to the `[MASK]` token's logit before the loss calculation.
    2.  This bias is dynamically adjusted at each training step via a feedback loop. The loop's goal is to steer the model's average output probability for `[MASK]` towards a configurable `target_mask_prob`.

### Iteration 4: Implementing a Data Generation Curriculum
*   **Observation:** The dynamic logit bias was mechanically functional but had to reach extreme values (`> 90.0`) to counteract the logit gap, indicating an underlying problem with the learning task itself.
*   **Hypothesis:** The training data was presenting two cognitively distinct tasks (generative un-masking and analytical re-masking) simultaneously from the start of training. This was overwhelming the model and causing it to learn a flawed initial strategy.
*   **Action Taken:** A data generation curriculum was implemented.
    1.  A `proofreading_warmup_iters` parameter was introduced.
    2.  The `get_batch` function was modified to control the ratio of re-masking tasks to un-masking tasks. Early in training, the model is presented almost exclusively with un-masking tasks. The more complex re-masking tasks are gradually introduced over the course of the warmup period.

### Iteration 5: Capping the Reward to Prevent Pathological Learning
*   **Observation:** The model continued to develop an extreme logit gap, even with the data curriculum in place.
*   **Root Cause Analysis:** A final, critical flaw was identified in the loss function: the weight for a correct un-masking guess was hard-coded to `0.0`. In the context of gradient descent, a zero loss is an infinitely powerful reward, incentivizing the model to adopt extreme weight configurations to maximize its chances of achieving this "jackpot."
*   **Action Taken:** The hard-coded `0.0` weight was replaced with a configurable, non-zero hyperparameter, `reward_correct_unmask` (e.g., `0.1`). This change ensures that a correct guess is still the best possible outcome but removes the pathological incentive created by an infinite reward.

## 3. Current System Architecture

The final, current state of the training system is a result of the iterative refinements described above. It is defined by the following key components:

1.  **An `argmax`-Free Loss Function:** The loss calculation is based on the full probability distribution from the logits. It applies simple, task-based weights (`weight_unmask_task`, `weight_remask_task`) rather than weights based on the model's performance.
2.  **A Dual Curriculum:** The model's learning is guided by two independent schedulers:
    *   **Data Curriculum:** Gradually increases the proportion of "re-masking" tasks in the training data over `proofreading_warmup_iters`.
    *   **Penalty Curriculum:** Gradually increases the `penalty_mask_correct` for destructive edits over `masking_warmup_iters`.
3.  **A Capped Reward Mechanism:** The incentive for a correct un-masking guess is a small, non-zero loss weight (`reward_correct_unmask`), preventing the model from developing extreme weights.
4.  **A Self-Regulating Dynamic Logit Bias:** The model's greedy vs. lazy behavior is controlled by a feedback loop that continuously adjusts the `mask_logit_bias` to steer the model's average output probability of `[MASK]` towards a `target_mask_prob`.
5.  **A Consistent Inference Process:** The `mask_logit_bias` is stored as part of the model's configuration and is correctly applied during the `generate_diffusion` inference method, ensuring consistency between training and generation.
