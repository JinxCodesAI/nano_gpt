## Critic Calibration Script

This guide explains how to run `critic_calibration.py` to measure how well the critic
head's confidence scores match the model's actual token-level accuracy.

### Prerequisites
- A streaming dataset prepared under `data/<dataset_name>/` with queue files
  ready (e.g., run `python prepare.py config/train_char_diffusion.py`).
- A checkpoint produced by a model trained with `add_critic_head=True`.
- Access to the same vocabulary/config metadata that was saved alongside the
  dataset (the script reads `meta.pkl`).

> **Tip:** The script automatically selects CUDA when available; otherwise it
> falls back to CPU execution.

### Basic usage
From the repository root:

```bash
python critic_calibration.py <dataset_name> <checkpoint_path> <num_sequences> \
  [--split train|val] [--verbose]
```

Arguments:
- `dataset_name`: Folder under `data/` that contains the dataset to sample from
  (for example `char_diffusion`).
- `checkpoint_path`: Path to the `.pt` / `.pth` checkpoint that includes a critic
  head.
- `num_sequences`: Number of dataset sequences to evaluate. Each sequence is a
  full batch item (not individual tokens). The script stops after processing the
  requested number of sequences.
- `--split`: Optional dataset split to read from (`val` by default).
- `--verbose`: Emit intermediate bucket probabilities every 100 processed
  sequences. Useful for monitoring long runs.

Example:

```bash
python critic_calibration.py char_diffusion \
  checkpoints/char_diffusion/latest.pt \
  1024 --split val
```

### Output
The script aggregates critic scores into 100 buckets, one for each 0.01 band on
`[0, 1]`, computes the empirical accuracy for each bucket, and fills empty
buckets by copying the nearest populated value towards 0.5 as requested in the
original spec.

Results are written next to the checkpoint with the same filename but a `.json`
extension. In the example above, the script produces
`checkpoints/char_diffusion/latest.json`, containing an array with exactly 100
probabilities.

### Troubleshooting
- **Missing dataset files:** Ensure `data/<dataset_name>/meta.pkl` exists and
  that queue files are being produced by the streaming provider.
- **Missing critic head:** The script validates that the checkpoint was created
  with `add_critic_head=True`. If this check fails, retrain or choose a
  compatible checkpoint.
- **Different batch schemas:** The script expects batches that expose either
  `input`/`target` or `x`/`y` tensors along with an optional `attention_mask`.
  If your dataset uses a different naming convention, extend the script before
  running.
