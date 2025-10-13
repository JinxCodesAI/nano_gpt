# Critic Calibration Discrepancy Review

## Observed Behaviour

After integrating `build_critic_artifacts_from_logits`, the verbose calibration output still shows that all buckets below roughly 0.65 remain empty and inherit the same copied probability (≈0.51), while only a handful of mid/high buckets accumulate counts (e.g. bucket 68 has 88 693 hits with a 0.46 error rate, bucket 74 has 78 065 hits with a 0.63 error rate, and buckets ≥95 are mostly empty). In contrast, `interactive_diffusion_explorer.py` reports critic probabilities that span the full range: target=0 tokens have a 10th percentile near 0.004 and target=1 tokens have a 90th percentile near 0.984. The explorer therefore observes both very low and very high critic scores, whereas the calibration script records almost no activity in the extreme buckets.

## Current Root Causes

### 1. Critic is evaluated with the dataset attention mask

`critic_calibration.py` forwards both the language model and the critic head with the `attention_mask` coming from the streaming batches.【F:critic_calibration.py†L265-L314】 The dataset mask zeroes out the masked tokens (and padding), so when we subsequently call `model.critic_scores(critic_input, attention_mask=attention_mask)` the filled-in predictions are still suppressed. During training and in the interactive explorer the critic is always run without an attention mask, letting it see the substituted tokens in context.【F:model.py†L890-L933】【F:interactive_diffusion_explorer.py†L901-L929】 Feeding the critic a masked-out sequence collapses its logits toward the unconditional baseline (~0.5), which explains why the calibration histogram never visits the low/high buckets.

### 2. Language-model logits are generated under the same mask

The calibration loop also forwards the LM pass with `attention_mask=attention_mask` before sampling predictions.【F:critic_calibration.py†L276-L291】 The char-diffusion pipeline sets masked positions to zero in this mask, so the transformer ignores exactly the tokens we are trying to denoise. The interactive explorer (and the training graph inside `model.py`) calls the model without an attention mask when preparing critic artifacts.【F:model.py†L890-L933】【F:interactive_diffusion_explorer.py†L901-L929】 Masking them during calibration produces nearly context-free logits, which further flattens the critic scores toward the middle of the range.

## Recommended Fixes

1. Drop the dataset-provided `attention_mask` when generating logits and critic scores. Either omit the argument entirely or rebuild an all-ones mask so the model and critic process the filled tokens exactly like training and the explorer do. This should restore the full critic-score dynamic range and populate the low/high buckets.
2. Once the attention mask is removed, keep relying on `critic_target`/`critic_valid` from `build_critic_artifacts_from_logits` to aggregate error rates per bucket. With the critic now seeing the substituted tokens, the distribution should match the explorer: low buckets will predominantly contain target=0 counts, high buckets will align with target=1 counts, and the copied values for empty buckets will largely disappear.

Addressing the masking mismatch should reconcile the calibration histogram with the interactive explorer’s per-sample statistics and yield meaningful critic error probabilities across all score buckets.
