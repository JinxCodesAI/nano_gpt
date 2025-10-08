"""
DatasetConsumer: streaming-friendly data consumer for precomputed batch files.

Features:
- Supports filesystem queue in data/<dataset>/queue/{train,val}
- Blocks and waits when no data is available
- Deletes fully-consumed files to signal provider (only in queue mode)
- Exposes meta.pkl and basic stats

Note: get_batch always returns a dict[str, Tensor] (e.g., x, y, attention_mask).
"""
from __future__ import annotations

import os
import time
import glob
import pickle
from typing import Dict, List, Optional

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
        # When provider writes array-of-batches, this holds the list for current file
        self._loaded_batches: Dict[str, Optional[List[Dict]]] = {"train": None, "val": None}

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
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"DatasetConsumer failed to load batch file for split={split}: {path}. {type(e).__name__}: {e}") from e
        # Require array-of-batches format
        if isinstance(data, dict) and 'batches' in data:
            self._loaded_batches[split] = data.get('batches', [])
            metadata: Dict = data.get('metadata', {})
            return {}, metadata
        raise ValueError(f"DatasetConsumer expected 'batches' format in file {path}. Regenerate data with unified array-of-batches format.")

    def _filter_zero_supervision_samples(self, tensors: Dict[str, torch.Tensor], metadata: Dict) -> tuple[Dict[str, torch.Tensor], Dict]:
        """Filter out samples that have zero supervised targets (all targets == ignore_index)."""
        ignore_index = -100

        # Find target tensor - try common names
        target_tensor = None
        target_key = None
        for key in ['y', 'targets']:
            if key in tensors:
                target_tensor = tensors[key]
                target_key = key
                break

        if target_tensor is None:
            # No target tensor found, return as-is
            return tensors, metadata

        # Find samples with at least one supervised target
        # Handle both 1D targets (sequence_scorer) and 2D targets (MLM)
        if target_tensor.dim() == 1:
            # 1D targets: each sample has a single target value
            supervised_mask = (target_tensor != ignore_index)  # (batch_size,)
        else:
            # 2D targets: each sample has multiple target values
            supervised_mask = (target_tensor != ignore_index).any(dim=1)  # (batch_size,)
        num_supervised = supervised_mask.sum().item()
        total_samples = target_tensor.shape[0]

        if num_supervised == total_samples:
            # All samples have supervision, no filtering needed
            return tensors, metadata

        if num_supervised == 0:
            # No samples have supervision - this shouldn't happen but handle gracefully
            print(f"[consumer] WARNING: No samples with supervision found in batch file")
            return tensors, metadata

        # Filter all tensors to keep only supervised samples
        filtered_tensors = {}
        for key, tensor in tensors.items():
            filtered_tensors[key] = tensor[supervised_mask]

        # Update metadata if it contains stage_info or other per-sample data
        filtered_metadata = metadata.copy()
        if 'stage_info' in metadata and isinstance(metadata['stage_info'], list):
            # Filter stage_info to match filtered samples
            original_stage_info = metadata['stage_info']
            if len(original_stage_info) == total_samples:
                filtered_stage_info = [original_stage_info[i] for i in range(total_samples) if supervised_mask[i]]
                filtered_metadata['stage_info'] = filtered_stage_info

        print(f"[consumer] Filtered out {total_samples - num_supervised} samples with zero supervision (kept {num_supervised}/{total_samples})")

        return filtered_tensors, filtered_metadata

    def _maybe_delete_consumed(self, split: str, path: str) -> None:
        if self._mode != "queue":
            return
        # Keep validation files for circular reuse; delete only training files in queue mode
        if split == "val":
            return
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
        self._loaded_batches[split] = None
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
    def get_batch(self, split: str, device) -> Dict[str, torch.Tensor]:
        """Fetch next batch for the split. Blocks if needed.

        Always returns a dict[str, Tensor] (e.g., x, y, attention_mask).
        """
        self._ensure_data_available(split)

        # Load current file if needed (array-of-batches format only)
        if self._loaded_batches.get(split) is None:
            # choose file index; in queue mode pick the first available
            if self._mode == "queue":
                if split == "train":
                    self._current_file_idx[split] = 0
                else:
                    if self._current_file_idx[split] >= len(self._split_files[split]):
                        self._current_file_idx[split] = 0
            _, metadata = self._load_file(split, self._current_file_idx[split])
            self._loaded_metadata[split] = metadata
            self._current_batch_idx[split] = 0

        # If file uses array-of-batches format, serve one batch entry at a time
        if self._loaded_batches.get(split) is not None:
            batches_list = self._loaded_batches[split] or []
            idx = self._current_batch_idx[split]
            if idx >= len(batches_list):
                # advance to next file
                if self._mode == "queue":
                    path = self._split_files[split][self._current_file_idx[split]]
                    self._maybe_delete_consumed(split, path)
                    self._split_files[split] = self._list_available_files(split)
                self._advance_to_next_file(split)
                self._ensure_data_available(split)
                tensors, metadata = self._load_file(split, self._current_file_idx[split])
                self._loaded_data[split] = tensors if tensors else {}
                self._loaded_metadata[split] = metadata
                batches_list = self._loaded_batches[split] or []
                idx = 0

            entry = batches_list[idx]
            entry_tensors = dict(entry.get('tensors', {}))
            entry_meta = dict(entry.get('metadata', {}))

            # move to device
            def _to_device(t: torch.Tensor) -> torch.Tensor:
                if self.device_type == "cuda":
                    return t.pin_memory().to(device, non_blocking=True)
                return t.to(device)
            for k in list(entry_tensors.keys()):
                entry_tensors[k] = _to_device(entry_tensors[k])

            # attach mode (provider must set it in per-batch metadata)
            mm = entry_meta.get('model_mode', None)
            if mm is not None:
                entry_tensors['_model_mode'] = mm

            # advance
            self._current_batch_idx[split] += 1
            return entry_tensors

        # If we reach here, no batches are loaded which indicates a logic or format error
        raise RuntimeError(f"No 'batches' loaded for split={split}. Expected unified array-of-batches format.")

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

