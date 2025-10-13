# Critic Calibration Discrepancy Review

## Observed Behaviour

Running `interactive_diffusion_explorer.py` on the `char_diffusion` stream shows that the critic assigns high probabilities to most incorrectly restored tokens (e.g., target=1 tokens have mean≈0.82 and 90th percentile≈0.98). The interactive unmasking accuracy for the same batch is ≈26 %.

In contrast, the current `critic_calibration.py` run over the same checkpoint/dataset reports near-uniform bucket probabilities around 0.02 for the first twenty critic-score buckets and an overall accuracy of ≈9.7 %, which contradicts the interactive inspection results.

## Root Causes

### 1. Critic input assembly diverges from training/interactive tooling

- The interactive explorer (and the training codepath in `model.py`) builds critic inputs via `build_critic_artifacts_from_logits`, which replaces **only** the masked tokens in the original sequence with LM samples before invoking the critic head.【F:interactive_diffusion_explorer.py†L901-L928】【F:sample_utils.py†L600-L640】
- `critic_calibration.py` instead feeds the critic head a tensor filled with multinomial samples for **every** position in the sequence (masked or not).【F:critic_calibration.py†L265-L279】 This produces unrealistic contexts—the critic sees random garbage in positions that should stay fixed—which drives its scores toward the ambiguous mid-range and destroys the correlation with actual correctness.

### 2. Calibration ignores the dataset’s critic mask semantics

- `build_critic_artifacts_from_logits` exposes both `critic_target` (0/1 error signal) and `critic_valid` (positions that should be counted for critic loss/stats, including masked tokens and, depending on scope, ignore-index positions).【F:sample_utils.py†L620-L635】 The interactive explorer respects this mask when summarising critic probabilities.【F:interactive_diffusion_explorer.py†L967-L980】
- The calibration script hardcodes its supervision mask to `targets != ignore_index`, silently discarding the ignore-index tokens that the critic was trained to treat as trivially correct and potentially mixing in non-masked positions if a dataset ever supplies explicit targets there.【F:critic_calibration.py†L287-L304】 Aligning with the shared artifacts would ensure we score the exact token set that the critic expects.

## Recommended Fixes

1. Reuse `build_critic_artifacts_from_logits` inside `critic_calibration.py`:
   - Pull the dataset’s `mask_token_id`, `ignore_index`, and optional `pad_token_id` from the model config/meta.
   - Call the helper to obtain `pred_tokens`, `critic_input`, `critic_target`, and `critic_valid`.
   - Invoke `model.critic_scores(critic_input, …)` so the critic evaluates the same filled-in sequence it saw during training/interactive inspection.

2. Drive the calibration statistics from the helper outputs:
   - Treat `pred_tokens` as the sampled predictions when comparing to `targets`.
   - Filter by `critic_valid` instead of `(targets != ignore_index)` so bucket counts match the critic’s definition of supervised positions.

Implementing the above should yield bucket curves that place low critic scores almost exclusively on correct tokens and restore the overall accuracy to the ≈25 % range observed in the explorer.
