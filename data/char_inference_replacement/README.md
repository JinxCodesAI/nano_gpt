# Char Inference Replacement Streaming Dataset

`CharInferenceReplacementProvider` extends the diffusion streaming pipeline by
replacing the random corruption branch with predictions produced by the latest
training checkpoint. Instead of injecting uniformly chosen characters, masked
positions are filled with model outputs, enabling self-bootstrapping / teacher
forcing while the provider keeps running alongside training.

## How It Works

- **Checkpoint watcher** – `CheckpointPredictionCorruptor` scans the target
  directory (typically `out_dir` from training) for the newest
  `ckpt_<training_type>_<iter>.pt` file and keeps the latest weights loaded
  on the desired device. If no versioned checkpoint has been saved yet it will
  fall back to `ckpt.pt` when present.
- **Prediction branch** – Whenever the corruption mixture routes positions to
  the prediction bucket, a forward pass is executed on the masked batch. Mask tokens
  are inserted temporarily, logits are optionally temperature-scaled, and the
  argmax tokens replace the masked positions.
- **Fallback** – If no checkpoint is available yet (e.g., early producer
  startup), an optional `RandomReplacementCorruptor` fallback can be enabled to
  keep batches flowing until a checkpoint appears.
- **Stage-aware mixing** – Fragment sampling, stage composition, label
  generation, and queue handling mirror `CharRandomReplacementProvider`, so
  existing configs remain compatible. Both train and validation splits use the
  same corruption mixture, drawing prediction targets and fragments from their
  respective data splits.

## Required Arguments

When invoking `prepare_streaming.py` directly or through `prepare.py`, set:

- `checkpoint_dir`: folder containing the checkpoints to load

Optional knobs include:

- `inference_device` / `inference_dtype` – control where inference runs
- `inference_refresh_seconds` – polling interval for new checkpoints
- `prediction_temperature` – scale logits before argmax
- `fallback_to_random` and related settings – enable/disable random fallback
- `train_corruption_mixture` – weights for (prediction, random, `[MASK]`, fragment)

## Example

```bash
python prepare.py config/train_char_inference_replacement_targets_full_agressive.py
```

The example config mirrors the aggressive random-replacement baseline but adds
checkpoint parameters so the provider can source predictions from the training
run in `out-char-random-replacement/`.

## Notes

- Periodic reloads are throttled (`inference_refresh_seconds`) to avoid
  constantly re-reading the checkpoint.
- The provider verifies that the checkpoint's `block_size` matches the dataset
  configuration to prevent silent shape mismatches.
- Metadata written to `meta.pkl` advertises the corruption type and mixture so
  consumers can adapt if needed.

## Understanding the Mask-Ratio Log Line

Every time a batch file is written the producer prints an extended summary:

```
[provider] mask ratios: initial min/max=0.0966/0.9004,
                         final min/max=0.0560/0.5276,
                         prediction total=0.0406/0.3618,
                         random total=0.0205/0.2127,
                         mask total=0.0167/0.1788,
                         fragment total=0.0392/0.3598,
                         prediction change frac=0.0420/0.4766,
                         random change frac=0.9834/0.9991,
                         mask change frac=1.0000/1.0000,
                         fragment change frac=0.1641/0.9565
```

The metrics are derived from all batches in the file:

- **initial min/max** – lowest and highest fraction of positions that were
  masked before any corruption was applied.
- **final min/max** – fraction of positions whose token changed after
  corruption (i.e., how many inputs differ from the original sequence).
- **prediction/random/mask/fragment total** – share of tokens routed to each
  corruption branch according to the mixture weights. These totals should
  roughly sum to the overall masked ratio on average.
- **prediction/random/mask/fragment change frac** – for each branch, the
  range of per-sample fractions of positions whose value actually changed
  relative to the original token. Prediction branches can partially agree with
  the model (fractions below 1.0). Random and mask branches always overwrite
  the source token, so their change fractions stay near or at 1.0.

Together these numbers confirm that non-prediction branches (random fallback,
mask token, and fragment copy) retain their intended behaviour while the prediction branch
captures how often the inference model disagrees with the source tokens.
