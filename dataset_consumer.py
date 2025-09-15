"""
DatasetConsumer: streaming-friendly data consumer for precomputed batch files.

Features:
- Supports filesystem queue in data/<dataset>/queue/{train,val}
- Blocks and waits when no data is available
- Deletes fully-consumed files to signal provider (only in queue mode)
- Exposes meta.pkl and basic stats

Note: For now we maintain the (X, Y) return path when schema has x/y. If a
schema is present with other fields, get_batch returns a dict[str, Tensor].
"""
from __future__ import annotations

import os
import time
import glob
import pickle
from typing import Dict, List, Optional, Tuple, Union

import torch


class DatasetConsumer:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: Optional[int] = None,
        target_size: Optional[int] = None,
        device_type: str = "cuda",
        prefer_queue: bool = True,
        cache_files: int = 1,
        wait_sleep_seconds: float = 1.0,
        wait_timeout_seconds: Optional[float] = None,
        verbose: bool = False,  # keep for optional logging
    ) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.target_size = target_size if target_size is not None else block_size
        self.device_type = device_type
        self.prefer_queue = True  # enforce streaming-only mode
        self.cache_files = max(1, int(cache_files))
        self.wait_sleep_seconds = wait_sleep_seconds
        self.wait_timeout_seconds = wait_timeout_seconds
        self.verbose = verbose

        # batch state
        self._initialized = False
        self._mode = "queue"
        self._split_files: Dict[str, List[str]] = {"train": [], "val": []}
        self._current_file_idx: Dict[str, int] = {"train": 0, "val": 0}
        self._current_batch_idx: Dict[str, int] = {"train": 0, "val": 0}
        self._loaded_data: Dict[str, Optional[Dict[str, torch.Tensor]]] = {"train": None, "val": None}
        self._loaded_metadata: Dict[str, Optional[Dict]] = {"train": None, "val": None}

        # meta
        self._meta_path = os.path.join(self.data_dir, "meta.pkl")
        self._meta: Optional[Dict] = None

    # ---- meta and schema ----
    @property
    def meta(self) -> Dict:
        if self._meta is None:
            if not os.path.exists(self._meta_path):
                raise FileNotFoundError(f"meta.pkl not found at {self._meta_path}")
            with open(self._meta_path, "rb") as f:
                self._meta = pickle.load(f)
            # Ensure vocab_size accounts for CLS token if present
            try:
                cls_id = self._meta.get("cls_token_id", None)
                vs = self._meta.get("vocab_size", None)
                if cls_id is not None and vs is not None and cls_id >= vs:
                    self._meta["vocab_size"] = int(cls_id) + 1
            except Exception:
                # Fail safely; better to proceed than crash here
                pass
        return self._meta

    def schema(self, split: str) -> Optional[List[Dict]]:
        m = self.meta
        return m.get("batch_schema")

    # ---- initialization ----
    def _detect_mode(self) -> None:
        queue_dir = os.path.join(self.data_dir, "queue")
        if self.prefer_queue:
            if not os.path.isdir(queue_dir):
                raise FileNotFoundError(
                    f"Queue directory not found at {queue_dir}. "
                    f"Start the dataset provider (prepare.py <config>) to generate streaming data."
                )
            self._mode = "queue"
        if self.verbose:
            print(f"DatasetConsumer mode: {self._mode}")

    def _list_available_files(self, split: str) -> List[str]:
        if self._mode == "queue":
            d = os.path.join(self.data_dir, "queue", split)
            if not os.path.isdir(d):
                return []
            # ignore temp files that start with .tmp-
            files = [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith('.pt') and not fn.startswith('.tmp-')]
            files.sort()  # by filename; filenames include timestamp-seq for monotonicity
            return files
        else:
            pattern = os.path.join(self.data_dir, f"{split}_batches_*.pt")
            files = sorted(glob.glob(pattern))
            return files

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._detect_mode()
        for split in ("train", "val"):
            self._split_files[split] = self._list_available_files(split)
        self._initialized = True

    # ---- file operations ----
    def _load_file(self, split: str, file_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict]:
        files = self._split_files[split]
        if not files:
            raise FileNotFoundError(f"No batch files available for split={split} in mode={self._mode}")
        if file_idx >= len(files):
            file_idx = 0
            self._current_file_idx[split] = 0
        path = files[file_idx]
        data = torch.load(path, map_location="cpu")
        # normalize data dict
        tensors: Dict[str, torch.Tensor] = {}
        metadata: Dict = data.get("metadata", {})
        if "x" in data and "y" in data:
            tensors = {"x": data["x"], "y": data["y"]}
        elif "tensors" in data:
            tensors = data["tensors"]
        else:
            # allow only tensors besides metadata
            for k, v in data.items():
                if k == "metadata":
                    continue
                if isinstance(v, torch.Tensor):
                    tensors[k] = v
        return tensors, metadata

    def _maybe_delete_consumed(self, split: str, path: str) -> None:
        if self._mode != "queue":
            return
        # delete only in queue mode
        try:
            os.remove(path)
            if self.verbose:
                print(f"[consumer] deleted: {path}")
        except Exception as e:
            if self.verbose:
                print(f"[consumer] warning: failed to delete consumed file {path}: {e}")

    def _advance_to_next_file(self, split: str) -> None:
        self._current_file_idx[split] += 1
        if self._current_file_idx[split] >= len(self._split_files[split]):
            self._current_file_idx[split] = 0
        self._loaded_data[split] = None
        self._loaded_metadata[split] = None
        self._current_batch_idx[split] = 0

    def _ensure_data_available(self, split: str) -> None:
        self._initialize()
        start_time = time.time()
        while True:
            # refresh file list in queue mode or when none loaded
            if self._mode == "queue" or not self._split_files[split]:
                before = set(self._split_files[split])
                self._split_files[split] = self._list_available_files(split)
                after = set(self._split_files[split])
                added = sorted(list(after - before))
                if added and self.verbose:
                    for p in added:
                        print(f"[consumer] detected new file for {split}: {p}")
            if self._split_files[split]:
                return
            # no files available, wait
            if self.wait_timeout_seconds is not None and (time.time() - start_time) > self.wait_timeout_seconds:
                raise TimeoutError(f"Timed out waiting for data for split={split}")
            time.sleep(self.wait_sleep_seconds)

    # ---- public API ----
    def get_batch(self, split: str, device) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Fetch next batch for the split. Blocks if needed.

        Returns (X, Y) if keys 'x' and 'y' are available, else a dict[str, Tensor].
        """
        self._ensure_data_available(split)

        # Load current file if needed
        if self._loaded_data[split] is None:
            # choose file index; in queue mode pick the first available
            if self._mode == "queue":
                self._current_file_idx[split] = 0
            tensors, metadata = self._load_file(split, self._current_file_idx[split])
            self._loaded_data[split] = tensors
            self._loaded_metadata[split] = metadata
            self._current_batch_idx[split] = 0

        tensors = self._loaded_data[split]
        assert tensors is not None
        # derive total samples in this file from first tensor
        first_key = next(iter(tensors))
        total_samples = tensors[first_key].shape[0]

        start_idx = self._current_batch_idx[split] * self.batch_size
        end_idx = min(start_idx + self.batch_size, total_samples)

        # if consumed all samples, advance file
        if start_idx >= total_samples:
            # in queue mode: delete the file we just finished
            if self._mode == "queue":
                path = self._split_files[split][self._current_file_idx[split]]
                self._maybe_delete_consumed(split, path)
                # refresh file list for queue mode to get new arrivals
                self._split_files[split] = self._list_available_files(split)
            # advance to next file
            self._advance_to_next_file(split)
            # ensure availability again (may block)
            self._ensure_data_available(split)
            # load new file
            tensors, metadata = self._load_file(split, self._current_file_idx[split])
            self._loaded_data[split] = tensors
            self._loaded_metadata[split] = metadata
            self._current_batch_idx[split] = 0
            # recompute indices
            first_key = next(iter(tensors))
            total_samples = tensors[first_key].shape[0]
            start_idx = 0
            end_idx = min(self.batch_size, total_samples)

        # slice per field
        batch_tensors: Dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            batch_tensors[k] = v[start_idx:end_idx]

        cur_bs = next(iter(batch_tensors.values())).shape[0]
        if 0 < cur_bs < self.batch_size:
            need = self.batch_size - cur_bs

            def _repeat_to_fill(t: torch.Tensor, need: int) -> torch.Tensor:
                p = t.shape[0]
                reps = (need + p - 1) // p
                return t.repeat((reps,) + (1,) * (t.dim() - 1))[:need]

            for k, v in list(batch_tensors.items()):
                batch_tensors[k] = torch.cat([v, _repeat_to_fill(v, need)], dim=0)

        # advance
        self._current_batch_idx[split] += 1

        # move to device
        def _to_device(t: torch.Tensor) -> torch.Tensor:
            if self.device_type == "cuda":
                return t.pin_memory().to(device, non_blocking=True)
            return t.to(device)

        for k in list(batch_tensors.keys()):
            batch_tensors[k] = _to_device(batch_tensors[k])

        # return legacy tuple when possible
        if "x" in batch_tensors and "y" in batch_tensors:
            return batch_tensors["x"], batch_tensors["y"]
        # normalize common schema keys to tuple expected by train/evaluator
        if "input_ids" in batch_tensors and "targets" in batch_tensors:
            return batch_tensors["input_ids"], batch_tensors["targets"]
        return batch_tensors

    def reset_state(self, split: Optional[str] = None) -> None:
        splits = [split] if split else ["train", "val"]
        for s in splits:
            self._current_file_idx[s] = 0
            self._current_batch_idx[s] = 0
            self._loaded_data[s] = None
            self._loaded_metadata[s] = None

    def get_stats(self) -> Dict:
        return {
            "mode": self._mode,
            "train_files": len(self._split_files["train"]),
            "val_files": len(self._split_files["val"]),
            "current_train_file": self._current_file_idx["train"],
            "current_val_file": self._current_file_idx["val"],
            "current_train_batch": self._current_batch_idx["train"],
            "current_val_batch": self._current_batch_idx["val"],
            "initialized": self._initialized,
        }

