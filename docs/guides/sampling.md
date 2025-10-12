# Sampling and Diffusion Generation Guide

This guide explains how text generation works in the diffusion-first nanoGPT fork, covering the orchestration script (`sample.py`) and the reusable utilities in `sample_utils.py`. It documents how to run sampling jobs, the configuration surface, and the pathways that influence output quality and runtime.

## 1. Entry Points and When to Use Them

### `sample.py`: Iterative pipelines
`sample.py` is the main entry point for inference. It supports two strategies that operate on fully masked sequences:

| Method | When to use | Key characteristics |
| --- | --- | --- |
| Diffusion (`sampling_method='diffusion'`) | Default for diffusion-trained checkpoints | Iteratively fills masks, then remasks uncertain tokens according to a configurable schedule and guidance signals. |
| Multinomial (`sampling_method='multinomial'`) | Baseline or ablation runs | Re-samples the entire sequence every iteration without remasking heuristics. |

Both strategies share the same loader path, vocabulary handling, seed protection, and optional quality metrics.【F:sample.py†L559-L756】

### Utility module
`sample_utils.py` provides temperature/top-p sampling, remasking schedules, and scoring helpers that power both diffusion and multinomial modes. Understanding these utilities is critical when modifying sampling behavior.【F:sample_utils.py†L11-L661】

## 2. Running `sample.py`

1. **Prepare checkpoints and metadata.** The script expects a model checkpoint in `out_dir` and reads vocabulary metadata (including mask/pad IDs) from `data/<dataset>/meta.pkl`.【F:sample.py†L559-L603】
2. **Set configuration values.** Default parameters (e.g., `sampling_method`, `iterations`, `temperature`) live near the top of the file. They can be overridden at runtime via `configurator.py`:
   ```bash
   python sample.py --sampling_method=diffusion --iterations=32 --temperature=0.7
   ```
   Config files passed as positional arguments are executed before CLI key/value overrides, allowing reproducible experiment bundles.【F:sample.py†L1-L196】【F:configurator.py†L1-L40】
3. **Choose seed placement.** The optional `seed_text` is tokenized and protected either at the prefix or a random offset. Protected tokens are never remasked or resampled.【F:sample.py†L262-L299】【F:sample.py†L485-L555】
4. **Run generation.** The script wraps inference in `torch.no_grad()` and AMP contexts when CUDA is available. A `TimingAccumulator` provides per-stage latency stats if enabled.【F:sample.py†L680-L756】
5. **(Optional) Evaluate quality.** After sampling, the script can compute self-confidence scores or run a judge model configured for sequence scoring.【F:sample.py†L718-L756】

## 3. Diffusion Sampling Lifecycle

1. **Initialization.** The batch is filled with mask tokens, and seed text is injected and protected. Verbose mode prints sanity checks before diffusion begins.【F:sample.py†L343-L390】
2. **Iteration loop.** Each pass performs:
   - **Prediction:** `predict_and_sample_tokens` runs the model on the current tokens, collects logits for masked positions, applies temperature/top-p, excludes special tokens, and samples replacements. Sampling is batched per masked location to avoid re-decoding unmasked tokens.【F:sample.py†L392-L414】【F:sample_utils.py†L71-L206】
   - **Remasking (except the final pass):** `apply_remasking_step` determines which positions to mask before the next iteration, using either schedules, critic logits, intelligent uncertainty heuristics, or an external remasking model.【F:sample.py†L416-L451】【F:sample_utils.py†L430-L661】
3. **Schedules.** By default, a linear schedule decreases the remasking ratio from `start_ratio` to `end_ratio`. Custom per-iteration ratios are supported when `schedule_type='custom'`. The helper delegates to `linear_remasking_schedule` when running in ratio mode.【F:sample_utils.py†L11-L28】【F:sample_utils.py†L463-L495】
4. **Remasking strategies.** Inside `apply_remasking_step`:
   - **External remasking model:** Uses predicted confidence logits (binary by default) to mask the least confident positions, with optional randomness blending.【F:sample_utils.py†L498-L545】
   - **Critic-guided remasking:** If the main model exposes `add_critic_head`, the script queries `model.critic_scores()` to prioritize tokens with high predicted error probability.【F:sample.py†L384-L387】【F:sample_utils.py†L547-L590】
   - **Intelligent self-remasking:** Uses the diffusion logits to compute per-token uncertainty (1 – probability of the sampled token) and masks the highest-uncertainty positions.【F:sample_utils.py†L592-L642】
   - **Random fallback:** Selects a random subset when no guidance is available.【F:sample_utils.py†L644-L661】
   Threshold mode can replace ratio-based selection, masking all tokens whose wrongness exceeds a schedule-driven threshold. Each path signals early termination when nothing qualifies for remasking, letting the main loop exit before exhausting `iterations`.【F:sample_utils.py†L509-L625】
5. **Progress reporting.** `log_iteration_progress` prints the percentage of masked tokens and snapshots of decoded text for the first and final iterations when `show_progress` is enabled.【F:sample.py†L300-L341】

## 4. Multinomial Resampling Lifecycle

Multinomial mode keeps the same initialization and progress logging but bypasses remasking:
1. All positions are sampled each iteration using the full softmax distribution (after temperature and vocabulary masks are applied).【F:sample.py†L520-L555】
2. Protected seed tokens remain fixed across iterations, allowing prefixes/prompts to persist.【F:sample.py†L485-L555】
3. Without a remasking phase, the method behaves like a stochastic relaxation process; convergence depends solely on iteration count and temperature. This makes it useful for ablations but often lower quality than diffusion for the same compute budget.【F:sample.py†L453-L558】

## 5. Sampling Controls

| Control | Applies to | Effect |
| --- | --- | --- |
| `temperature` | Both | Divides logits before softmax; lower values sharpen distributions, higher values encourage diversity.【F:sample_utils.py†L71-L175】【F:sample.py†L529-L555】 |
| `top_p` | Diffusion only | Enables nucleus sampling that restricts sampling to the smallest token set whose cumulative probability exceeds `top_p`. Set to 1.0 to disable.【F:sample_utils.py†L31-L69】【F:sample.py†L400-L412】 |
| `iterations` | Both | Number of predict/remask or resample cycles. Diffusion can exit early if remasking schedules finish or no tokens qualify for masking.【F:sample.py†L392-L451】【F:sample.py†L512-L556】 |
| `start_ratio` / `end_ratio` | Diffusion | Define how aggressively masks are reintroduced over time. Higher `start_ratio` masks most tokens early for broad exploration; higher `end_ratio` keeps more tokens mutable toward the end.【F:sample_utils.py†L11-L28】【F:sample_utils.py†L463-L661】 |
| `randomness_strength` | Diffusion | Blends deterministic guidance with uniform noise when selecting tokens to remask, preventing the process from getting stuck in local optima.【F:sample_utils.py†L333-L373】【F:sample_utils.py†L498-L579】 |
| `schedule_mode` | Diffusion | Choose between masking a fixed ratio per iteration (`ratio`) or masking everything above a wrongness threshold (`threshold`). Threshold mode provides adaptive iteration counts via early termination signals.【F:sample_utils.py†L430-L661】 |
| `seed_placement` | Both | Chooses whether `seed_text` anchors the prefix or a random offset. Protected positions are never masked or resampled.【F:sample.py†L262-L299】【F:sample.py†L485-L555】 |
| `base_vocab_size`, `mask_token_id`, `pad_token_id` | Both | Guard rails that remove special tokens from sampling logits; values come from dataset metadata to stay consistent with training.【F:sample_utils.py†L153-L170】【F:sample.py†L559-L603】 |

## 6. Quality Metrics and Post-processing

- **Self confidence:** Computes the average log probability of generated tokens (excluding masks) to approximate internal certainty. Useful when no external judge is available.【F:sample_utils.py†L209-L269】【F:sample.py†L718-L733】
- **Judge model:** Loads a sequence scorer checkpoint, optionally prepends a CLS token, and interprets the output as an evaluation score (`1 - evaluation`). Throughput stats (tokens/sec) are reported for profiling.【F:sample_utils.py†L272-L305】【F:sample.py†L640-L756】
- Additional metrics can be integrated by following the pattern in the `QualityMetric` enum and branching after generation.

## 7. Extending Sampling Behavior

- **New remasking heuristics:** Add branches inside `apply_remasking_step` to compute a score tensor and mask accordingly. Ensure compatibility with both ratio and threshold modes, and consider randomness blending to avoid deterministic traps.【F:sample_utils.py†L430-L661】
- **Alternative schedules:** Replace or augment `linear_remasking_schedule` with custom functions; update `schedule_type` handling to select them based on configuration.【F:sample_utils.py†L11-L28】【F:sample_utils.py†L463-L495】
- **Custom quality metrics:** Mirror the judge/self-confidence implementations by adding new enum members, loading required models, and computing metrics within the post-generation block.【F:sample.py†L680-L756】
- **Diffusion loop instrumentation:** When instrumenting new heuristics or integrating with reinforcement pipelines, prefer to hook into `predict_and_sample_tokens` and `apply_remasking_step` so that diffusion-specific optimizations (mask-aware batching, remasking thresholds, critic integration) remain in effect.【F:sample.py†L392-L451】【F:sample_utils.py†L430-L661】

## 8. Performance Considerations

- `predict_and_sample_tokens` supports gradient-enabled execution (`no_grad=False`) for reinforcement learning scenarios (e.g., GRPO fine-tuning). The default `torch.no_grad()` keeps inference fast.【F:sample_utils.py†L71-L206】
- The script uses AMP (`torch.amp.autocast`) on CUDA to reduce memory bandwidth while keeping `torch.cuda` TF32 optimizations enabled.【F:sample.py†L52-L83】【F:sample.py†L680-L756】
- `TimingAccumulator` hooks (`get_global_timer`) wrap the prediction and remasking phases, helping diagnose bottlenecks when experimenting with new heuristics.【F:sample.py†L36-L76】【F:sample.py†L392-L451】

By following this guide, you can confidently run diffusion sampling jobs, tune their behavior, and extend the pipeline with new heuristics or evaluation metrics while understanding how each component affects both quality and runtime.
