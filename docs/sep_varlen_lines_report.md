# feature/sep_varlen_lines — Intended Implementation, Expected Behavior, Differences vs diffusion_05_09, and Current Problem

This document summarizes, without opinions, what the feature/sep_varlen_lines branch was intended to implement, how it is expected to behave, how that differs from the diffusion_05_09 baseline, and what problem is currently observed.

## 1) Intended implementation on feature/sep_varlen_lines

Goal: switch data and model pipeline to support variable-length, line‑aligned sequences terminated by a [SEP] token and governed by an attention mask, while keeping BERT‑style corruption.

Key elements:
- Sequence construction
  - Each training sample starts at a line beginning.
  - Multiple complete lines are packed until context limit; sample ends with a [SEP] token.
  - Overlong single lines are truncated to fit; [SEP] is appended if space permits.
  - No PAD tokens used for loss; positions after [SEP] are excluded via attention_mask and ignored via labels == ignore_index.
- Special tokens and vocabulary
  - [MASK] and [SEP] are appended to the base vocabulary; ids are consistent across data generator and model.
- Attention mask semantics
  - attention_mask[b, t] = 1 for 0..valid_len (including [SEP]); 0 for positions after [SEP].
  - Model uses key padding mask derived from attention_mask to prevent attending to padded tail.
- Labels/targets and loss
  - BERT‑style 80/10/10 corruption over visible (attention_mask==1) positions only.
  - labels[b, t] = target token for supervised (corrupted) positions; labels == ignore_index elsewhere (including uncorrupted positions and all positions after [SEP]).
  - Loss is standard token cross‑entropy with ignore_index.
- Data pipeline changes
  - DatasetConsumer returns batches as dict[str, torch.Tensor] (e.g., {x, y, attention_mask}).
  - Data provider emits meta with block_size and special token ids; streaming generation continues to produce train/val batches.
- Model changes
  - Bidirectional attention path accepts attention_mask and converts to key‑padding mask for scaled_dot_product_attention.
  - Forward takes (x, y, attention_mask); logits and loss computed with ignore_index.
- Instrumentation
  - Diagnostics compute corruption breakdown (mask/random/unchanged), valid_len statistics, supervised_per_token ratio, and per‑sample masking ratio percentiles (batch snapshot and cumulative).

## 2) Expected behavior (feature/sep_varlen_lines)

- Correctness
  - Positions after [SEP] do not participate in attention or loss.
  - Only visible tokens (attention_mask==1) are candidates for corruption and supervision.
  - With standard BERT corruption settings, the supervised token density reflects the configured corruption policy.
- Logging
  - Diagnostics report mean valid length ~ up to block_size depending on packing.
  - Per‑sample mask ratio percentiles (p10, p50, p90) reflect the distribution of supervised tokens over visible positions within each batch; cumulative variants aggregate over all past batches.
- Performance
  - Vectorized packing and minimal Python loops; no significant regression vs baseline in throughput.

## 3) How this differs from diffusion_05_09

Baseline (diffusion_05_09):
- Fixed‑length windows without explicit attention_mask; no [SEP] token delimiting variable‑length content.
- Data represented predominantly for diffusion/unmasking tasks; stage configurations apply.
- Batches may not begin at line boundaries; no explicit line‑aligned packing.
- Batch interfaces not strictly dict‑only.

feature/sep_varlen_lines:
- Variable‑length, line‑aligned packing ending with [SEP]; positions beyond are masked out through attention_mask.
- Attention‑aware model path used (bidirectional) for MLM.
- Dataset/batch interfaces normalized to dict form and include attention_mask.
- Additional diagnostics for visibility/masking and attention behavior.
- Stage configurations are supported; behavior intended unchanged relative to diffusion_05_09.


## 4) Current observed problem

Empirical training logs on feature/sep_varlen_lines show:
- Training/validation loss decreases initially from ~4.3 to ~3.2 then plateaus around ~3.13, instead of continuing to ~2.1–2.4 as observed in diffusion_05_09 within similar steps.
- Diagnostics (examples from user logs):
  - supervised_per_token ≈ 0.47–0.52
  - valid_len_mean ≈ 1002–1007
  - Corruption breakdown: pct_mask ≈ 80%, pct_random ≈ 9–10%, pct_unchanged ≈ 9–10%
  - mask_ratio percentiles (batch snapshot) p50 ≈ 0.55 with wide spread; cumulative p10/p50/p90 initially fluctuated rather than stabilizing.
  - Attention ablation probe: loss_noattn_minus_attn small positive (~0.005–0.02), indicating attention mask is active but effect size is small in the probe subset.
- Cumulative masking ratio statistics initially updated only at log intervals, leading to sensitivity and oscillation of reported percentiles; this was adjusted so accumulation occurs every iteration and only percentile computation happens at log time.

This summarizes the intended design, differences from the baseline branch, and the problem as evidenced by the logs, without attributing cause.

