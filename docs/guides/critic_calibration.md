## Critic Calibration Script

This guide explains how to run `critic_calibration.py` to measure how well the critic
head's confidence scores align with the critic targets (0 = correct, 1 = error)
generated from the model's sampled completions.

### Prerequisites
- A streaming dataset prepared under `data/<dataset_name>/` with queue files
  ready (e.g., run `python prepare.py config/train_char_diffusion.py`).
- A checkpoint produced by a model trained with `add_critic_head=True`.
- Access to the same vocabulary/config metadata that was saved alongside the
  dataset (the script reads `meta.pkl`). This metadata must expose the
  `mask_token_id` (and optionally `pad_token_id`) so the script can faithfully
  rebuild the critic inputs.

> **Tip:** The script automatically selects CUDA when available; otherwise it
> falls back to CPU execution.

### Basic usage
From the repository root:

```bash
python critic_calibration.py <dataset_or_config> <checkpoint_path> <num_sequences> \
  [--split train|val] [--verbose]
```

Arguments:
- `dataset_or_config`: Either the dataset name (e.g. `char_diffusion`), the
  path to the dataset directory, or the path to a Python config file that sets
  `dataset = "..."`. The script resolves relative paths against your current
  working directory as well as the repository root so it works whether you run
  it from inside or outside `data/`.
- `checkpoint_path`: Path to the `.pt` / `.pth` checkpoint that includes a critic
  head.
- `num_sequences`: Number of dataset sequences to evaluate. Each sequence is a
  full batch item (not individual tokens). The script stops after processing the
  requested number of sequences.
- `--split`: Optional dataset split to read from (`train` by default).
- `--verbose`: Emit intermediate bucket statistics (per-bucket totals, target-1
  counts, and probabilities) every 100 processed sequences. Useful for
  monitoring long runs.

Example:

```bash
python critic_calibration.py config/_diffusion_no_modifiers.py \
  checkpoints/char_diffusion/latest.pt \
  1024 --split train
```

The example above passes a training config; the script reads its `dataset`
attribute to locate `data/char_diffusion/`. You can also provide
`char_diffusion` or the absolute path to the dataset directory directly.

### Output
The script aggregates critic scores into 100 buckets, one for each 0.01 band on
`[0, 1]`, computes the empirical probability that the critic target equals 1
(`model error`) for each bucket, and fills empty buckets by copying the nearest
populated value towards 0.5 as requested in the original spec. Because the
critic was trained without an external attention mask, the script also ignores
any dataset-provided `attention_mask` so the transformer and critic process the
filled-in tokens exactly as they do during training and in the interactive
explorer.

For every supervised token the model samples a prediction from the softmax
distribution (via multinomial sampling) before querying the critic head on the
filled-in sequence produced by
`build_critic_artifacts_from_logits`. The helper mirrors the training and
interactive tooling: it replaces only the masked tokens, retains untouched input
tokens, and marks both masked tokens and ignore-index positions as valid for the
critic. This keeps the calibration aligned with how the critic was trained and
how `interactive_diffusion_explorer.py` visualises the scores.

Results are stored in a shared `calibration.json` file within the same directory
as the checkpoint. Each run updates (or creates) a key in that file named
after the provided checkpoint path, and the associated value is the array of 100
probabilities corresponding to the critic target being 1 (i.e. the token is
incorrect) for each bucket. Buckets with no direct observations inherit the
probability from the closest populated bucket as described above, so the saved
array exactly mirrors the verbose bucket report.

Both `sample.py` and `sample_simple.py` automatically read these tables when a
critic head is present. If a checkpoint has no dedicated entry in
`calibration.json`, the samplers fall back to a synthetic calibration ramp that
starts at 0.01 for bucket 0 and linearly increases to 0.99 for bucket 99. This
ensures critic-guided remasking remains functional even before a calibration
pass has been recorded.

When `--verbose` is active the script prints a table of 100 lines after every
100 processed sequences. Each line shows the bucket index, the number of critic
scores that landed in that bucket, how many of those had critic target 1, and
the resulting probability (or `0.0` if the bucket has no samples yet). These
figures help diagnose any unexpected distribution shifts.

After saving the JSON file the script also prints two summary statistics: the
overall critic-target positive rate (fraction of critic-valid positions with
target 1) and the same value recomputed from the aggregated bucket counts.
These numbers should match and offer a quick sanity check that the calibration
buckets were populated correctly.

### Troubleshooting
- **Missing dataset files:** Ensure the resolved dataset directory contains a
  `meta.pkl` file and queue batches for the requested split. If you pass a
  config file, double-check that its `dataset` attribute points to the prepared
  dataset.
- **Missing critic head:** The script validates that the checkpoint was created
  with `add_critic_head=True`. If this check fails, retrain or choose a
  compatible checkpoint.
- **Different batch schemas:** The script expects batches that expose either
  `input`/`target` or `x`/`y` tensors. Dataset-provided attention masks are
  ignored so the calibration matches the critic's training graph; if your
  batches use different field names, extend the script before running.
