#!/usr/bin/env python3
"""Analyze critic calibration against ground-truth token accuracy.

This script loads a diffusion checkpoint with an enabled critic head and a
streaming dataset produced by the corresponding data provider. It runs a single
forward pass for randomly sampled batches from the dataset, evaluates the
critic score for each supervised token, and aggregates calibration statistics.

For every critic score bucket (width 0.01 across [0, 1]), the script computes
the empirical probability that the model's token prediction is correct.
Buckets with no observations inherit the value from the closest populated
bucket towards the center of the range (0.5) as requested.

The resulting array of 100 probabilities is written as JSON next to the
checkpoint, reusing the checkpoint's filename but with a ``.json`` suffix.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import List, Tuple

import torch

from dataset_consumer import DatasetConsumer
from model import GPT, GPTConfig, ModelMode


DEPRECATED_MODEL_FIELDS = {
    "mode",
    "num_token_classes",
    "binary_classification",
    "attention_type",
    "position_encoding",
}


def load_model(checkpoint_path: Path, device: torch.device) -> GPT:
    """Load a GPT model (with critic head) from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_args" not in checkpoint:
        raise ValueError("Checkpoint missing required 'model_args' entry")

    model_args = dict(checkpoint["model_args"])

    # Ignore deprecated fields to keep GPTConfig construction future-proof
    filtered_model_args = {
        k: v for k, v in model_args.items() if k not in DEPRECATED_MODEL_FIELDS
    }

    # Some checkpoints store legacy mode names – map them after instantiation
    legacy_mode = model_args.get("mode")

    gptconf = GPTConfig(**filtered_model_args)
    model = GPT(gptconf)

    if legacy_mode:
        if legacy_mode in (ModelMode.SEQUENCE_SCORER, "sequence_scorer"):
            model.set_mode(ModelMode.SEQUENCE_SCORER)
        else:
            # Default to language-model behaviour
            model.set_mode(ModelMode.LANGUAGE_MODEL)
    else:
        model.set_mode(ModelMode.LANGUAGE_MODEL)

    state_dict = checkpoint.get("model")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model' state dict")

    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if not getattr(model.config, "add_critic_head", False):
        raise RuntimeError(
            "Loaded model does not have a critic head (add_critic_head=False)."
        )

    return model


def load_dataset_consumer(dataset_name: str, device: torch.device) -> Tuple[DatasetConsumer, dict]:
    dataset_dir = Path("data") / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    meta_path = dataset_dir / "meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata not found: {meta_path}. Run the provider first."
        )

    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    batch_size = int(meta.get("batch_size", 1))
    block_size = meta.get("block_size")
    target_size = meta.get("target_size", block_size)

    device_type = "cuda" if device.type == "cuda" else "cpu"

    consumer = DatasetConsumer(
        data_dir=str(dataset_dir),
        batch_size=batch_size,
        block_size=block_size,
        target_size=target_size,
        device_type=device_type,
        prefer_queue=True,
        cache_files=1,
        wait_sleep_seconds=1.0,
        wait_timeout_seconds=30.0,
        verbose=False,
    )

    return consumer, meta


def bucket_index(score: float) -> int:
    # Clamp to [0, 1] to defend against potential numerical noise
    clamped = min(max(score, 0.0), 1.0)
    # Scores exactly equal to 1.0 belong to the final bucket
    if clamped == 1.0:
        return 99
    return int(math.floor(clamped * 100.0))


def fill_empty_buckets(probs: List[float | None]) -> List[float]:
    """Fill empty buckets by copying from the nearest populated bucket.

    Buckets < 0.5 copy from the next higher bucket, buckets >= 0.5 copy from the
    next lower bucket. If no populated bucket exists in that direction, the
    search falls back to the opposite direction. Remaining gaps default to 0.0.
    """

    filled = probs[:]
    for idx, value in enumerate(filled):
        if value is not None:
            continue
        direction = 1 if idx < 50 else -1
        neighbor = idx + direction
        found = None
        while 0 <= neighbor < 100:
            if filled[neighbor] is not None:
                found = filled[neighbor]
                break
            neighbor += direction
        if found is None:
            # Fall back to searching the opposite direction
            neighbor = idx - direction
            while 0 <= neighbor < 100:
                if filled[neighbor] is not None:
                    found = filled[neighbor]
                    break
                neighbor -= direction
        if found is None:
            found = 0.0
        filled[idx] = found
    return [float(v if v is not None else 0.0) for v in filled]


def analyze(
    model: GPT,
    consumer: DatasetConsumer,
    meta: dict,
    device: torch.device,
    num_samples: int,
    split: str,
) -> List[float]:
    ignore_index = int(getattr(model.config, "ignore_index", -100))

    correct_counts = torch.zeros(100, dtype=torch.long)
    total_counts = torch.zeros(100, dtype=torch.long)

    samples_processed = 0

    while samples_processed < num_samples:
        batch = consumer.get_batch(split, device)

        inputs = batch.get("input")
        targets = batch.get("target")
        if inputs is None or targets is None:
            # Fallback to alternative naming schemes for robustness
            inputs = batch.get("x")
            targets = batch.get("y")
        if inputs is None or targets is None:
            raise KeyError("Batch missing required input/target tensors")

        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs, dtype=torch.long)

        batch_mode = batch.get("_model_mode")
        if batch_mode in (ModelMode.SEQUENCE_SCORER, "sequence_scorer"):
            raise RuntimeError(
                "Sequence scorer batches are not supported for critic calibration"
            )
        model.set_mode(ModelMode.LANGUAGE_MODEL)

        with torch.no_grad():
            logits, _ = model(
                inputs,
                targets=targets,
                attention_mask=attention_mask,
            )
            critic_logits = model.critic_scores(inputs, attention_mask=attention_mask)
            critic_scores = torch.sigmoid(critic_logits)

        predictions = torch.argmax(logits, dim=-1)
        valid_mask = targets != ignore_index

        batch_size = inputs.size(0)
        sequence_limit = min(batch_size, num_samples - samples_processed)

        for batch_idx in range(sequence_limit):
            seq_valid = valid_mask[batch_idx]
            seq_scores = critic_scores[batch_idx]
            seq_preds = predictions[batch_idx]
            seq_targets = targets[batch_idx]

            valid_positions = torch.nonzero(seq_valid, as_tuple=False).view(-1)
            for pos in valid_positions:
                score = float(seq_scores[pos].item())
                bucket = bucket_index(score)
                is_correct = bool(seq_preds[pos].item() == seq_targets[pos].item())
                total_counts[bucket] += 1
                if is_correct:
                    correct_counts[bucket] += 1

        samples_processed += sequence_limit

    probabilities: List[float | None] = []
    for idx in range(100):
        total = int(total_counts[idx].item())
        if total > 0:
            correct = int(correct_counts[idx].item())
            probabilities.append(correct / total)
        else:
            probabilities.append(None)

    return fill_empty_buckets(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Critic calibration analysis")
    parser.add_argument("dataset", help="Dataset name (e.g., char_diffusion)")
    parser.add_argument("checkpoint", help="Path to the model checkpoint (.pt/.pth)")
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of dataset samples (sequences) to evaluate",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate (default: val)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    consumer, meta = load_dataset_consumer(args.dataset, device)

    probabilities = analyze(
        model=model,
        consumer=consumer,
        meta=meta,
        device=device,
        num_samples=max(1, int(args.num_samples)),
        split=args.split,
    )

    output_path = checkpoint_path.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(probabilities, f, indent=2)

    print(f"Saved critic calibration probabilities to {output_path}")


if __name__ == "__main__":
    main()
