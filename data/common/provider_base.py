"""
DataProviderBase: filesystem-queue-based producer for training batches.

Dataset scripts should subclass DataProviderBase and override build_meta() and
sample_batch(split, rng). The base handles:
- meta writing/validation
- directory structure (queue/train, queue/val)
- backlog/backpressure (max_backlog_files)
- atomic write via temp file + os.replace
- deterministic per-file RNG seeded by (base_seed, split, seq)
"""
from __future__ import annotations

import os
import time
import pickle
from typing import Dict, Iterable, Optional

import torch


class DataProviderBase:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: Optional[int] = None,
        target_size: Optional[int] = None,
        batches_per_file: int = 100,
        max_backlog_files: int = 2,
        sleep_seconds: float = 2.0,
        seed: int = 1337,
        verbose: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = int(batch_size)
        self.block_size = block_size
        self.target_size = target_size if target_size is not None else block_size
        self.batches_per_file = int(batches_per_file)
        self.max_backlog_files = int(max_backlog_files)
        self.sleep_seconds = float(sleep_seconds)
        self.seed = int(seed)
        self.verbose = verbose

        self.queue_dir = os.path.join(self.data_dir, "queue")
        self.train_dir = os.path.join(self.queue_dir, "train")
        self.val_dir = os.path.join(self.queue_dir, "val")
        self.meta_path = os.path.join(self.data_dir, "meta.pkl")

    # ---- to override ----
    def build_meta(self) -> Dict:
        """Return required metadata dict; must include training_type and batch_schema.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, torch.Tensor]:
        """Return a dict of tensors for one batch, each shaped [batch_size, ...]."""
        raise NotImplementedError

    # ---- base helpers ----
    def ensure_dirs(self) -> None:
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

    def backlog_size(self, split: str) -> int:
        d = self.train_dir if split == "train" else self.val_dir
        return len([fn for fn in os.listdir(d) if fn.endswith('.pt') and not fn.startswith('.tmp-')])

    def write_meta(self, meta: Dict) -> None:
        # strict keys presence
        required = ["training_type", "batch_schema"]
        for k in required:
            if k not in meta:
                raise ValueError(f"Missing required meta key: {k}")
        # enrich meta with defaults
        meta = dict(meta)
        meta.update({
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "target_size": self.target_size,
        })
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        if self.verbose:
            print(f"Wrote meta to {self.meta_path}")

    def produce_one_file(self, split: str, seq: int) -> None:
        rng = torch.Generator()
        # derive deterministic seed per split/seq
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        # collect batches
        batches = [self.sample_batch(split, rng) for _ in range(self.batches_per_file)]
        # concatenate along batch dimension
        keys = batches[0].keys()
        tensors: Dict[str, torch.Tensor] = {}
        for k in keys:
            tensors[k] = torch.cat([b[k] for b in batches], dim=0)
        metadata = {
            "batch_size": self.batch_size,
            "num_batches": self.batches_per_file,
            "file_idx": seq,
            "split": split,
            "produced_at": int(time.time() * 1000),
        }
        # write atomic
        d = self.train_dir if split == "train" else self.val_dir
        ts = metadata["produced_at"]
        tmp_name = f".tmp-{ts}-{seq:06d}.pt"
        final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
        tmp_path = os.path.join(d, tmp_name)
        final_path = os.path.join(d, final_name)
        torch.save({"tensors": tensors, "metadata": metadata}, tmp_path)
        os.replace(tmp_path, final_path)
        if self.verbose:
            print(f"Produced {final_path}")

    def run(self, splits: Iterable[str] = ("train", "val")) -> None:
        self.ensure_dirs()
        if not os.path.exists(self.meta_path):
            meta = self.build_meta()
            self.write_meta(meta)
        # initialize sequence counter by scanning existing files
        seq_counters = {}
        for split in splits:
            d = self.train_dir if split == "train" else self.val_dir
            os.makedirs(d, exist_ok=True)
            existing = [fn for fn in os.listdir(d) if fn.endswith('.pt') and not fn.startswith('.tmp-')]
            max_seq = -1
            for fn in existing:
                try:
                    # name: ts-seq-batches.pt
                    parts = fn.split('-')
                    seq = int(parts[1])
                    max_seq = max(max_seq, seq)
                except Exception:
                    continue
            seq_counters[split] = max_seq + 1
        # production loop (simple, single-threaded)
        try:
            while True:
                made_progress = False
                for split in splits:
                    if self.backlog_size(split) >= self.max_backlog_files:
                        continue
                    self.produce_one_file(split, seq_counters[split])
                    seq_counters[split] += 1
                    made_progress = True
                if not made_progress:
                    time.sleep(self.sleep_seconds)
        except KeyboardInterrupt:
            if self.verbose:
                print("Provider interrupted; exiting cleanly.")

