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
from typing import Dict, Iterable, Optional, Tuple, List

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
        raise RuntimeError("Subclasses must implement build_meta() and include required keys.")

    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, torch.Tensor]:
        """Return a dict of tensors for one batch, each shaped [batch_size, ...]."""
        raise RuntimeError("Subclasses must implement sample_batch(split, rng) and return tensor dicts shaped [batch_size, ...].")

    def default_model_mode(self) -> Optional[str]:
        """Optional per-provider fixed model mode.
        Subclasses can override to return 'language_model' or 'sequence_scorer'.
        Default None means no automatic mode injection.
        """
        return None

    # ---- shared data loading helpers ----
    def _load_input_text(self, filename: str = 'input.txt') -> str:
        """Load text data from input file in data_dir."""
        input_file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input text file not found: {input_file_path}")
        with open(input_file_path, 'r') as f:
            return f.read()

    def _create_char_vocab(self, data: str) -> Tuple[int, Dict[str, int], Dict[int, str]]:
        """Create character-level vocabulary from text data."""
        chars = sorted(list(set(data)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return vocab_size, stoi, itos

    def _create_train_val_split(self, data: str, train_ratio: float = 0.9) -> Tuple[str, str]:
        """Split text data into train and validation sets."""
        n = len(data)
        split_idx = int(n * train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        return train_data, val_data

    def _tokenize_text(self, text: str, stoi: Dict[str, int]) -> List[int]:
        """Convert text to list of token IDs using character vocabulary."""
        return [stoi[c] for c in text]

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
        """Write array-of-batches format with strict input/target tensors per batch.

        File layout:
          {
            'batches': [ {'tensors': {'input': Tensor, 'target': Tensor}, 'metadata': {...}}, ... ],
            'metadata': { file-level }
          }
        """
        rng = torch.Generator()
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        def _normalize_to_input_target(b: Dict) -> tuple[Dict[str, torch.Tensor], Dict]:
            # Map known schemas to unified names
            if 'input' in b and 'target' in b:
                inp, tgt = b['input'], b['target']
            elif 'x' in b and 'y' in b:
                inp, tgt = b['x'], b['y']
            elif 'input_ids' in b and 'targets' in b:
                inp, tgt = b['input_ids'], b['targets']
            else:
                raise ValueError("sample_batch must return either (input, target) or (x, y) or (input_ids, targets)")
            if not (isinstance(inp, torch.Tensor) and isinstance(tgt, torch.Tensor)):
                raise TypeError("Both input and target must be tensors")
            if inp.shape[0] != self.batch_size or tgt.shape[0] != self.batch_size:
                raise ValueError(f"Batch tensors must have batch dimension {self.batch_size}, got {inp.shape[0]} and {tgt.shape[0]}")
            tensors = {'input': inp, 'target': tgt}
            meta = {k: v for k, v in b.items() if not isinstance(v, torch.Tensor)}
            return tensors, meta

        batches_out = []
        for _ in range(self.batches_per_file):
            b = self.sample_batch(split, rng)
            tensors, meta = _normalize_to_input_target(b)
            # Inject fixed model mode when subclass defines it and batch didn't specify one
            mm = self.default_model_mode()
            if 'model_mode' not in meta and mm is not None:
                meta['model_mode'] = mm
            batches_out.append({'tensors': tensors, 'metadata': meta})

        file_meta = {
            'batch_size': self.batch_size,
            'num_batches': self.batches_per_file,
            'file_idx': seq,
            'split': split,
            'produced_at': int(time.time() * 1000),
        }

        d = self.train_dir if split == 'train' else self.val_dir
        ts = file_meta['produced_at']
        tmp_name = f".tmp-{ts}-{seq:06d}.pt"
        final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
        tmp_path = os.path.join(d, tmp_name)
        final_path = os.path.join(d, final_name)
        torch.save({'batches': batches_out, 'metadata': file_meta}, tmp_path)
        os.replace(tmp_path, final_path)
        if self.verbose:
            print(f"[provider] produced (array-of-batches): {final_path}")

    def run(self, splits: Iterable[str] = ("train", "val")) -> None:
        self.ensure_dirs()
        # Determine if we're starting from a fresh/empty queue (no .pt files yet)
        fresh_start = True
        for split in splits:
            d = self.train_dir if split == "train" else self.val_dir
            os.makedirs(d, exist_ok=True)
            existing = [fn for fn in os.listdir(d) if fn.endswith('.pt') and not fn.startswith('.tmp-')]
            if len(existing) > 0:
                fresh_start = False
        # (Re)write meta if it doesn't exist OR when starting from an empty queue
        if (not os.path.exists(self.meta_path)) or fresh_start:
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

