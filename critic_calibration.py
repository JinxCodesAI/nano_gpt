#!/usr/bin/env python3
"""Analyze critic calibration against observed critic targets.

This script loads a diffusion checkpoint with an enabled critic head and a
streaming dataset produced by the corresponding data provider. It runs a single
forward pass for randomly sampled batches from the dataset, evaluates the
critic score for each supervised token, and aggregates calibration statistics.

For every critic score bucket (width 0.01 across [0, 1]), the script computes
the empirical probability that the critic target equals ``1`` (token judged
incorrect by the critic objective). Buckets with no observations inherit the
value from the closest populated bucket towards the center of the range (0.5)
as requested.

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
from sample_utils import build_critic_artifacts_from_logits


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


def resolve_dataset_directory(dataset_arg: str) -> Path:
    """Resolve a dataset identifier to an on-disk directory.

    ``dataset_arg`` may be a dataset name (e.g. ``"char_diffusion"``), a path to the
    dataset directory, or a path to a Python config file that defines a ``dataset``
    attribute. Paths are interpreted relative to the current working directory and
    the directory containing this script.
    """

    script_dir = Path(__file__).resolve().parent
    raw_path = Path(dataset_arg).expanduser()

    def candidate_dirs() -> List[Path]:
        bases = [Path.cwd(), script_dir]
        seen: set[Path] = set()
        for base in bases:
            for candidate in (base / dataset_arg, base / "data" / dataset_arg):
                try:
                    resolved = candidate.expanduser()
                except RuntimeError:
                    # ``expanduser`` can fail on some malformed inputs; skip them
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield resolved

    # Direct directory specification (absolute or relative)
    if raw_path.is_dir():
        return raw_path

    # Treat config files as pointers to the dataset name
    if raw_path.is_file() and raw_path.suffix == ".py":
        import runpy

        config_globals = runpy.run_path(str(raw_path))
        dataset_name = config_globals.get("dataset")
        if not dataset_name:
            raise ValueError(
                f"Config file {raw_path} does not define a 'dataset' attribute."
            )
        return resolve_dataset_directory(str(dataset_name))

    for candidate in candidate_dirs():
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Unable to locate dataset directory for argument "
        f"{dataset_arg!r}. Provide a dataset name, directory, or config file."
    )


def load_dataset_consumer(dataset_arg: str, device: torch.device) -> Tuple[DatasetConsumer, dict]:
    dataset_dir = resolve_dataset_directory(dataset_arg)

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
    verbose: bool = False,
) -> Tuple[List[float], float, float]:
    cfg = getattr(model, "config", object())

    mask_token_id = meta.get("mask_token_id", getattr(cfg, "mask_token_id", None))
    if mask_token_id is None:
        raise RuntimeError(
            "Critic calibration requires 'mask_token_id' in dataset meta or model config"
        )

    ignore_index = int(getattr(cfg, "ignore_index", meta.get("ignore_index", -100)))
    pad_token_id = meta.get("pad_token_id", getattr(cfg, "pad_token_id", None))
    critic_scope = getattr(cfg, "critic_target_scope", "masked_and_ignore")

    total_counts = torch.zeros(100, dtype=torch.long)
    target_one_counts = torch.zeros(100, dtype=torch.long)

    global_total = 0
    global_target_one = 0

    samples_processed = 0
    next_report = 100 if verbose else None

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
            )

            artifacts = build_critic_artifacts_from_logits(
                idx=inputs,
                logits=logits,
                targets=targets,
                mask_token_id=int(mask_token_id),
                ignore_index=ignore_index,
                pad_token_id=pad_token_id,
                scope=critic_scope,
            )

            critic_input = artifacts["critic_input"]
            critic_target = artifacts["critic_target"]
            critic_valid = artifacts["critic_valid"]

            critic_logits = model.critic_scores(critic_input)
            critic_scores = torch.sigmoid(critic_logits)

        batch_size = inputs.size(0)
        sequence_limit = min(batch_size, num_samples - samples_processed)

        if sequence_limit <= 0:
            break

        eval_scores = critic_scores[:sequence_limit]
        eval_targets = critic_target[:sequence_limit]
        eval_valid = critic_valid[:sequence_limit]

        flat_mask = eval_valid.reshape(-1)
        if flat_mask.any():
            flat_scores = eval_scores.reshape(-1)[flat_mask]
            flat_targets = eval_targets.reshape(-1)[flat_mask]

            clamped_scores = torch.clamp(flat_scores, 0.0, 1.0 - 1e-8)
            buckets = (clamped_scores * 100.0).long()
            buckets = torch.clamp(buckets, 0, 99)

            target_one = (flat_targets >= 0.5).to(torch.long)

            bucket_cpu = buckets.to("cpu")
            target_one_cpu = target_one.to("cpu")
            token_totals = torch.ones_like(bucket_cpu, dtype=torch.long)

            total_counts.index_add_(0, bucket_cpu, token_totals)
            target_one_counts.index_add_(0, bucket_cpu, target_one_cpu)

            global_total += int(token_totals.sum().item())
            global_target_one += int(target_one_cpu.sum().item())

        samples_processed += sequence_limit

        if verbose and next_report is not None:
            while samples_processed >= next_report:
                current: List[float | None] = []
                for idx in range(100):
                    total = int(total_counts[idx].item())
                    if total > 0:
                        positives = int(target_one_counts[idx].item())
                        current.append(positives / total)
                    else:
                        current.append(None)

                filled_current = fill_empty_buckets(current)

                print(f"[critic_calibration] {next_report} samples processed ->")
                for idx in range(100):
                    total = int(total_counts[idx].item())
                    positives = int(target_one_counts[idx].item())
                    prob = filled_current[idx] if total > 0 else 0.0
                    print(
                        f"  bucket {idx:02d}: hits={total}, target1={positives}, "
                        f"prob={prob:.6f}"
                    )
                next_report += 100

    if verbose:
        current: List[float | None] = []
        for idx in range(100):
            total = int(total_counts[idx].item())
            if total > 0:
                positives = int(target_one_counts[idx].item())
                current.append(positives / total)
            else:
                current.append(None)

        filled_current = fill_empty_buckets(current)

        print("[critic_calibration] final bucket statistics ->")
        for idx in range(100):
            total = int(total_counts[idx].item())
            positives = int(target_one_counts[idx].item())
            prob = filled_current[idx] if total > 0 else 0.0
            print(
                f"  bucket {idx:02d}: hits={total}, target1={positives}, "
                f"prob={prob:.6f}"
            )

    probabilities: List[float | None] = []
    for idx in range(100):
        total = int(total_counts[idx].item())
        if total > 0:
            positives = int(target_one_counts[idx].item())
            probabilities.append(positives / total)
        else:
            probabilities.append(None)

    filled = fill_empty_buckets(probabilities)

    overall_positive_rate = 0.0
    if global_total > 0:
        overall_positive_rate = global_target_one / global_total

    bucket_positive_rate = 0.0
    total_sum = int(total_counts.sum().item())
    if total_sum > 0:
        bucket_positive_rate = int(target_one_counts.sum().item()) / total_sum

    return filled, overall_positive_rate, bucket_positive_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Critic calibration analysis")
    parser.add_argument(
        "dataset",
        help="Dataset name, directory, or config file (e.g., char_diffusion or config/_diffusion.py)",
    )
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log bucket probabilities every 100 processed samples",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device)
    consumer, meta = load_dataset_consumer(args.dataset, device)

    probabilities, overall_positive_rate, bucket_positive_rate = analyze(
        model=model,
        consumer=consumer,
        meta=meta,
        device=device,
        num_samples=max(1, int(args.num_samples)),
        split=args.split,
        verbose=args.verbose,
    )

    output_path = checkpoint_path.with_suffix(".json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(probabilities, f, indent=2)

    print(f"Saved critic calibration probabilities to {output_path}")
    print(
        "Overall critic-target positive rate (target==1 fraction): "
        f"{overall_positive_rate:.6f}"
    )
    print(
        "Bucket-weighted positive rate (sum(target1_counts) / sum(total_counts)): "
        f"{bucket_positive_rate:.6f}"
    )


if __name__ == "__main__":
    main()
