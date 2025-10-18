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

    # ---- optional hooks for subclasses ----
    def _on_file_start(self, split: str) -> None:
        """Hook invoked before sampling batches for a file."""
        return None

    def _file_report_lines(self, split: str):
        """Hook returning iterable of extra log lines per produced file."""
        return ()

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

        start_time = time.time()
        sample_time = 0.0
        device_copy_time = 0.0
        cpu_transfer_time = 0.0
        save_time = 0.0
        total_batches = self.batches_per_file
        if total_batches <= 0:
            raise ValueError("batches_per_file must be positive")

        self._on_file_start(split)

        t0 = time.perf_counter()
        first_batch = self.sample_batch(split, rng)
        sample_time += time.perf_counter() - t0
        keys = first_batch.keys()
        total_capacity = self.batch_size * total_batches

        device_buffers: Dict[str, torch.Tensor] = {}
        first_batch_size = next(iter(first_batch.values())).shape[0]

        for k in keys:
            first_tensor = first_batch[k]
            sample_shape = first_tensor.shape[1:]
            buffer = torch.empty(
                (total_capacity,) + sample_shape,
                dtype=first_tensor.dtype,
                device=first_tensor.device,
            )
            buffer[0:first_batch_size] = first_tensor
            device_buffers[k] = buffer

        offset = first_batch_size
        total_written = first_batch_size
        del first_batch

        for _ in range(1, total_batches):
            t0 = time.perf_counter()
            batch = self.sample_batch(split, rng)
            sample_time += time.perf_counter() - t0
            current_size = next(iter(batch.values())).shape[0]
            for k in keys:
                data = batch[k]
                buf = device_buffers[k]
                if data.device != buf.device:
                    t_copy_start = time.perf_counter()
                    data = data.to(buf.device)
                    device_copy_time += time.perf_counter() - t_copy_start
                end = offset + current_size
                if end > buf.shape[0]:
                    raise RuntimeError(
                        f"Buffer overflow when writing batch {seq}; "
                        f"expected capacity {buf.shape[0]}, attempting to write up to {end}"
                    )
                buf[offset:end] = data
            offset += current_size
            total_written += current_size
            del batch

        tensors: Dict[str, torch.Tensor] = {}
        for k, buf in device_buffers.items():
            if buf.device.type != "cpu":
                torch.cuda.synchronize() if buf.device.type == "cuda" else None
                t_cpu_start = time.perf_counter()
                tensors[k] = buf[:total_written].to("cpu", non_blocking=True)
                torch.cuda.synchronize() if buf.device.type == "cuda" else None
                cpu_transfer_time += time.perf_counter() - t_cpu_start
            else:
                tensors[k] = buf[:total_written]
        device_buffers.clear()
        try:
            del buf  # release reference to last buffer
        except UnboundLocalError:
            pass
        del device_buffers
        metadata = {
            "batch_size": self.batch_size,
            "num_batches": self.batches_per_file,
            "file_idx": seq,
            "split": split,
            "produced_at": int(time.time() * 1000),
            "total_samples": total_written,
        }
        # write atomic
        d = self.train_dir if split == "train" else self.val_dir
        ts = metadata["produced_at"]
        tmp_name = f".tmp-{ts}-{seq:06d}.pt"
        final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
        tmp_path = os.path.join(d, tmp_name)
        final_path = os.path.join(d, final_name)
        t_save_start = time.perf_counter()
        torch.save({"tensors": tensors, "metadata": metadata}, tmp_path)
        save_time += time.perf_counter() - t_save_start
        os.replace(tmp_path, final_path)
        if self.verbose:
            total_samples = total_written
            total_batches = self.batches_per_file
            elapsed = time.time() - start_time
            ms_total = elapsed * 1000.0
            ms_per_sample = ms_total / total_samples if total_samples else float("inf")
            ms_per_batch = ms_total / total_batches if total_batches else float("inf")
            print(f"[provider] produced: {final_path}")
            print(
                f"[provider] stats: {total_samples} samples, "
                f"{ms_total:.1f} ms total, {ms_per_sample:.3f} ms/sample, "
                f"{ms_per_batch:.1f} ms/batch "
                f"(sample={sample_time*1000:.1f} ms, "
                f"device_copy={device_copy_time*1000:.1f} ms, "
                f"cpu_transfer={cpu_transfer_time*1000:.1f} ms, "
                f"save={save_time*1000:.1f} ms)"
            )
        extra_lines = list(self._file_report_lines(split))
        if self.verbose:
            for line in extra_lines:
                print(line)

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
