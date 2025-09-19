#!/usr/bin/env python3
import os
import sys
import argparse
import glob
from typing import Tuple, Optional

import torch

def find_targets_tensor(payload) -> Optional[torch.Tensor]:
    """
    Return the targets tensor from a saved .pt payload.
    Supports formats used by DataProviderBase/SequenceScorerProvider:
    - {"tensors": {"input_ids": ..., "targets": ...}, "metadata": {...}}
    - {"x": ..., "y": ..., "metadata": {...}}
    - {<tensor-keys>, "metadata": {...}}
    """
    if isinstance(payload, dict):
        if "tensors" in payload and isinstance(payload["tensors"], dict):
            t = payload["tensors"].get("targets")
            if isinstance(t, torch.Tensor):
                return t
        # legacy
        if "y" in payload and isinstance(payload["y"], torch.Tensor):
            return payload["y"]
        # fallback: try to locate a 1D/2D float tensor likely to be targets
        for k, v in payload.items():
            if k == "metadata":
                continue
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
                # Heuristic: prefer 1D tensors
                return v
    return None


def count_zeros(t: torch.Tensor, tol: float = 1e-9) -> Tuple[int, int]:
    """Count zero entries in a tensor with a tolerance for float comparisons."""
    t_flat = t.detach().cpu().view(-1)
    if t_flat.dtype.is_floating_point:
        zeros = (t_flat.abs() <= tol).sum().item()
    else:
        zeros = (t_flat == 0).sum().item()
    total = t_flat.numel()
    return int(zeros), int(total)


def main():
    parser = argparse.ArgumentParser(description="Inspect .pt files and count zero-target samples per file")
    parser.add_argument("dir", help="Directory containing .pt files (e.g., data/sequence_scorer/queue/val)")
    parser.add_argument("--pattern", default="*.pt", help="Glob pattern to match files (default: *.pt)")
    parser.add_argument("--sort", choices=["name", "mtime"], default="name", help="Sort files by name or modification time")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files processed")
    parser.add_argument("--verbose", action="store_true", help="Print warnings for files without targets")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(2)

    pattern = os.path.join(args.dir, args.pattern)
    files = glob.glob(pattern)
    if not files:
        print(f"No files match: {pattern}")
        sys.exit(0)

    if args.sort == "name":
        files.sort()
    else:
        files.sort(key=lambda p: os.path.getmtime(p))

    if args.limit is not None:
        files = files[: args.limit]

    grand_zeros = 0
    grand_total = 0
    processed = 0

    print(f"Scanning {len(files)} file(s) in {args.dir}\n")
    print(f"{'file':60}  {'zeros':>10}  {'total':>10}  {'ratio':>8}")
    print("-" * 95)

    for path in files:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"{os.path.basename(path):60}  LOAD_ERROR: {e}")
            continue

        targets = find_targets_tensor(payload)
        if targets is None:
            if args.verbose:
                print(f"{os.path.basename(path):60}  WARN: no targets tensor found")
            continue

        z, n = count_zeros(targets)
        ratio = (z / n) if n > 0 else 0.0
        grand_zeros += z
        grand_total += n
        processed += 1
        print(f"{os.path.basename(path):60}  {z:10d}  {n:10d}  {ratio:8.3%}")

    print("-" * 95)
    if processed == 0:
        print("No files with detectable targets were processed.")
    else:
        grand_ratio = (grand_zeros / grand_total) if grand_total > 0 else 0.0
        print(f"TOTAL over {processed} file(s): zeros={grand_zeros} total={grand_total} ratio={grand_ratio:.3%}")


if __name__ == "__main__":
    main()

