#!/usr/bin/env python3
"""
Quick inspection tool for precomputed dataset batch files.

Usage:
    python inspect_batch_file.py path/to/batch.pt [--examples 3] [--limit 32]
                                       [--meta path/to/meta.pkl]
                                       [--ignore-index -100]
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def _split_payload(obj: Any) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
    """Extract tensor dict and optional metadata from a torch.load payload."""
    if isinstance(obj, dict):
        if "tensors" in obj and isinstance(obj["tensors"], dict):
            tensor_dict = {k: v for k, v in obj["tensors"].items() if isinstance(v, torch.Tensor)}
            meta_obj = obj.get("metadata")
            # fall back to other non-tensor entries if metadata key absent
            if meta_obj is None:
                meta_obj = {k: v for k, v in obj.items() if k not in ("tensors",)}
                if not meta_obj:
                    meta_obj = None
            return tensor_dict, meta_obj if meta_obj else None

        tensor_dict = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        meta_obj = {k: v for k, v in obj.items() if k not in tensor_dict}
        return tensor_dict, meta_obj if meta_obj else None

    raise TypeError(
        "Unsupported batch payload format. Expected a dict with tensors or a dict containing "
        "a 'tensors' entry."
    )


def _format_number(value: torch.Tensor) -> str:
    if value.numel() == 0:
        return "n/a"
    scalar = value.detach().cpu().reshape(-1)[0]
    return str(scalar.item())


def _summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    tensor_cpu = tensor.detach().cpu()
    summary_parts = [
        f"{name}",
        f"shape={tuple(tensor_cpu.shape)}",
        f"dtype={tensor_cpu.dtype}",
    ]
    try:
        summary_parts.append(f"min={_format_number(tensor_cpu.min())}")
        summary_parts.append(f"max={_format_number(tensor_cpu.max())}")
    except RuntimeError:
        # min/max not defined for this dtype (e.g., boolean)
        pass
    return ", ".join(summary_parts)


def _format_sample(tensor: torch.Tensor, index: int, limit: int) -> str:
    sample = tensor[index].detach().cpu()
    if sample.numel() == 0:
        return "[]"
    if sample.ndim == 0:
        return str(sample.item())
    flat = sample.view(-1)
    values = flat[:limit].tolist()
    formatted = ", ".join(str(v) for v in values)
    if flat.numel() > limit:
        formatted += ", ..."
    return f"shape={tuple(sample.shape)} [{formatted}]"


def _format_tensor_row(sample: torch.Tensor, limit: int) -> str:
    row = sample.detach().cpu()
    if row.ndim == 0:
        return str(row.item())
    flat = row.view(-1)
    values = flat[:limit].tolist()
    formatted = ", ".join(str(v) for v in values)
    if flat.numel() > limit:
        formatted += ", ..."
    return f"shape={tuple(row.shape)} [{formatted}]"


def _decode_row(
    tensor: torch.Tensor,
    decoder: Optional[Dict[int, str]],
    token_limit: int,
    ignore_index: Optional[int],
    replace_ignore_with: str = "_",
) -> str:
    if decoder is None:
        return _format_tensor_row(tensor, token_limit)

    values = tensor.detach().cpu().view(-1).tolist()
    decoded_tokens = []
    for idx, tok in enumerate(values):
        if ignore_index is not None and tok == ignore_index:
            decoded_tokens.append(replace_ignore_with)
        else:
            decoded_tokens.append(decoder.get(int(tok), f"<{tok}>"))
        if idx + 1 >= token_limit:
            break
    text = "".join(decoded_tokens).replace("\n", "\\n")
    if len(values) > token_limit:
        text += "..."
    return f"shape={tuple(tensor.shape)} text=\"{text}\""


def _find_meta_path(batch_path: str) -> Optional[str]:
    current = os.path.abspath(os.path.dirname(batch_path))
    root = os.path.abspath(os.sep)
    while True:
        candidate = os.path.join(current, "meta.pkl")
        if os.path.isfile(candidate):
            return candidate
        if current == root:
            return None
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _compute_target_token_stats(
    target: torch.Tensor, ignore_index: int
) -> Tuple[float, float, float]:
    counts = torch.count_nonzero(target.detach().cpu() != ignore_index, dim=1).to(torch.float32)
    mean = counts.mean().item()
    if counts.numel() == 0:
        return mean, float("nan"), float("nan")
    k = max(1, math.ceil(counts.numel() * 0.1))
    top_mean = counts.topk(k, largest=True).values.mean().item()
    bottom_mean = counts.topk(k, largest=False).values.mean().item()
    return mean, top_mean, bottom_mean


def inspect_batch(
    path: str,
    example_count: int,
    token_limit: int,
    meta_path: Optional[str],
    ignore_index_override: Optional[int],
) -> None:
    payload = torch.load(path, map_location="cpu")
    tensors, metadata = _split_payload(payload)

    if not tensors:
        raise RuntimeError("No tensors found in the loaded payload.")

    first_tensor = next(iter(tensors.values()))
    num_samples = first_tensor.shape[0] if first_tensor.ndim > 0 else 1

    print(f"File: {path}")
    print(f"Samples: {num_samples}")
    print(f"Tensors: {', '.join(tensors.keys())}")

    print()
    print("Tensor summaries:")
    for name, tensor in tensors.items():
        if tensor.shape[0] != num_samples and tensor.ndim > 0:
            print(f"  - WARNING: {name} has mismatched leading dimension {tensor.shape[0]} (expected {num_samples})")
        print(f"  - {_summarize_tensor(name, tensor)}")
    print()

    resolved_meta_path = meta_path or _find_meta_path(path)
    meta = None
    if resolved_meta_path:
        try:
            with open(resolved_meta_path, "rb") as meta_file:
                meta = pickle.load(meta_file)
            print(f"Loaded meta: {resolved_meta_path}")
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to read meta at {resolved_meta_path}: {exc}")

    if metadata:
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  - {key}: {value}")
        print()

    decoder = None
    if meta and isinstance(meta.get("itos"), dict):
        decoder = {int(k): str(v) for k, v in meta["itos"].items()}

    ignore_index: Optional[int] = None
    if meta and "ignore_index" in meta:
        ignore_index = int(meta["ignore_index"])
    if metadata and "ignore_index" in metadata:
        try:
            ignore_index = int(metadata["ignore_index"])  # type: ignore[index]
        except Exception:
            pass
    if ignore_index_override is not None:
        ignore_index = ignore_index_override
    if ignore_index is None:
        ignore_index = -100

    target_name = None
    if meta and isinstance(meta.get("batch_schema"), list):
        for entry in meta["batch_schema"]:
            if isinstance(entry, dict) and entry.get("role") == "target":
                target_name = entry.get("name")
                break
    if target_name is None and "y" in tensors:
        target_name = "y"

    if target_name and target_name in tensors and tensors[target_name].ndim >= 1:
        target_tensor = tensors[target_name]
        try:
            mean, top_mean, bottom_mean = _compute_target_token_stats(target_tensor, int(ignore_index))
            print(
                "Target token stats (non-ignore counts per sample): "
                f"avg={mean:.2f}, top10% avg={top_mean:.2f}, bottom10% avg={bottom_mean:.2f}"
            )
            print()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to compute target stats: {exc}")
            print()

    example_count = max(0, min(example_count, num_samples))
    if example_count == 0:
        print("No examples to display (try lowering --examples).")
        return

    print(f"Examples (showing up to {token_limit} values per tensor):")
    for idx in range(example_count):
        print(f"  Sample {idx}:")
        for name, tensor in tensors.items():
            if tensor.ndim == 0:
                value_repr = str(tensor.detach().cpu().item())
            else:
                if tensor.shape[0] <= idx:
                    value_repr = "<out of range>"
                else:
                    row = tensor[idx]
                    if decoder and tensor.dtype in (torch.int16, torch.int32, torch.int64, torch.long):
                        replace_ignore = name == target_name
                        value_repr = _decode_row(
                            row,
                            decoder,
                            token_limit,
                            int(ignore_index) if replace_ignore else None,
                        )
                    else:
                        value_repr = _format_sample(tensor, idx, token_limit)
            print(f"    {name}: {value_repr}")
        print()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a dataset batch .pt file.")
    parser.add_argument("path", help="Path to a .pt file produced by the dataset provider.")
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of sample rows to display (default: 3).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Maximum number of values to show per tensor for each sample (default: 32).",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Optional path to meta.pkl for decoding and stats (default: auto-discover).",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=None,
        help="Override ignore index value used when summarizing targets.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        inspect_batch(path, args.examples, args.limit, args.meta, args.ignore_index)
    except Exception as exc:  # noqa: BLE001 keep simple CLI
        print(f"Failed to inspect '{path}': {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
