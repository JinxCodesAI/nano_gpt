# Critic Calibration Discrepancy Review

## Observed Behaviour (pre-fix)

Before the latest revision, the verbose calibration output showed that all buckets below roughly 0.65 remained empty and inherited the same copied probability (≈0.51), while only a handful of mid/high buckets accumulated counts (e.g. bucket 68 had 88 693 hits with a 0.46 error rate and bucket 74 had 78 065 hits with a 0.63 error rate). In contrast, `interactive_diffusion_explorer.py` reported critic probabilities spanning the full range: target=0 tokens had a 10th percentile near 0.004 and target=1 tokens had a 90th percentile near 0.984. The explorer therefore observed both very low and very high critic scores, whereas the calibration script recorded almost no activity in the extreme buckets.

## Root Cause Summary

- `critic_calibration.py` forwarded both the language-model pass and the critic head with the streaming dataset's `attention_mask`. The char-diffusion queues zero out masked tokens in that mask, so the transformer ignored the very tokens we attempted to denoise and the critic never saw the substituted predictions. Training and the interactive explorer invoke the model without such a mask, letting the critic operate on the fully reconstructed sequence.【F:model.py†L890-L933】【F:interactive_diffusion_explorer.py†L901-L929】
- Because the critic input stayed masked, its logits collapsed toward the unconditional baseline (~0.5). The resulting calibration histogram never populated the low/high buckets and produced misleading error rates.

## Implemented Resolution

- Removed the dataset-provided `attention_mask` from both the language-model forward pass and the critic evaluation so the model mirrors the training graph during calibration.【F:critic_calibration.py†L260-L320】
- Continued to aggregate bucket statistics directly from `critic_target`/`critic_valid`, ensuring the saved calibration reflects the critic's notion of positive targets.

With the masking mismatch resolved, the calibration script now processes the same filled-in sequences as the interactive explorer and should recover the full critic-score dynamic range, yielding meaningful error probabilities across all buckets.
