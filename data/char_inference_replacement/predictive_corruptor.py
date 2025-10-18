"""Utilities for checkpoint-driven corruption."""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import torch

from checkpoint_manager import CheckpointManager
from model import GPT, GPTConfig


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_dtype(dtype: Optional[torch.dtype | str]) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; expected one of {list(mapping.keys())} or a torch.dtype"
        )
    return mapping[dtype]


class CheckpointPredictionCorruptor:
    """Replace masked tokens using predictions from the latest checkpoint.

    The corruptor eagerly loads an initial checkpoint and periodically checks
    for fresh weights. When a checkpoint is unavailable or fails to load, an
    optional fallback corruptor can be used instead (e.g., random replacement).
    """

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        mask_token_id: int,
        block_size: int,
        device: Optional[str],
        dtype: Optional[torch.dtype | str],
        refresh_seconds: float,
        temperature: float,
        fallback_corruptor: Optional[object],
        verbose: bool,
    ) -> None:
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be provided")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self._checkpoint_dir = os.path.abspath(checkpoint_dir)
        self._mask_token_id = int(mask_token_id)
        self._expected_block_size = int(block_size)
        self._device = _resolve_device(device)
        self._dtype = _resolve_dtype(dtype)
        self._refresh_seconds = float(refresh_seconds)
        self._temperature = float(temperature)
        self._fallback = fallback_corruptor
        self._verbose = bool(verbose)

        self._model: Optional[torch.nn.Module] = None
        self._last_checkpoint_path: Optional[str] = None
        self._last_checkpoint_mtime: Optional[float] = None
        self._last_refresh_attempt: float = 0.0

        # Attempt an eager load so the first corruption call does not stall.
        self._maybe_refresh_model(force=True)

    # ------------------------------------------------------------------ #
    # Public API

    def corrupt(self, x: torch.Tensor, mask: torch.Tensor, rng) -> torch.Tensor:
        """Return ``x`` with masked positions replaced by model predictions."""
        mask_bool = mask.to(dtype=torch.bool)
        if not mask_bool.any():
            return x.clone()

        model = self._ensure_model_available()
        if model is None:
            if self._fallback is None:
                raise RuntimeError(
                    "No checkpoint available for inference and no fallback provided."
                )
            if self._verbose:
                print(
                    "[prediction-corruptor] Falling back to random replacement; "
                    "checkpoint missing or failed to load."
                )
            return self._fallback.corrupt(x, mask_bool, rng)

        return self._predict_with_model(model, x, mask_bool)

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _discover_latest_checkpoint(self) -> Optional[str]:
        try:
            entries = os.listdir(self._checkpoint_dir)
        except FileNotFoundError:
            if self._verbose:
                print(
                    f"[prediction-corruptor] Checkpoint directory missing: {self._checkpoint_dir}"
                )
            return None
        except OSError as exc:
            if self._verbose:
                print(
                    f"[prediction-corruptor] Failed to list checkpoint directory "
                    f"{self._checkpoint_dir}: {exc}"
                )
            return None

        versioned: list[tuple[int, float, str]] = []
        fallback_path: Optional[str] = None
        for name in entries:
            if not name.endswith(".pt"):
                continue
            path = os.path.join(self._checkpoint_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            iteration = self._extract_iteration(name)
            if iteration is not None:
                versioned.append((iteration, mtime, path))
            elif name == "ckpt.pt":
                fallback_path = path

        if versioned:
            versioned.sort(key=lambda item: (item[0], item[1]))
            return versioned[-1][2]
        return fallback_path

    @staticmethod
    def _extract_iteration(filename: str) -> Optional[int]:
        stem = os.path.splitext(filename)[0]
        parts = stem.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return None

    def _ensure_model_available(self) -> Optional[torch.nn.Module]:
        self._maybe_refresh_model()
        return self._model

    def _maybe_refresh_model(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_refresh_attempt) < self._refresh_seconds:
            return
        self._last_refresh_attempt = now
        path = self._discover_latest_checkpoint()
        if path is None:
            return
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            return
        if (
            not force
            and self._last_checkpoint_path == path
            and self._last_checkpoint_mtime is not None
            and mtime <= self._last_checkpoint_mtime
        ):
            return
        loaded = self._load_checkpoint(path)
        if loaded:
            self._last_checkpoint_path = path
            self._last_checkpoint_mtime = mtime

    def _load_checkpoint(self, path: str) -> bool:
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:
            if self._verbose:
                print(f"[prediction-corruptor] Failed to load checkpoint {path}: {exc}")
            return False

        model_args = checkpoint.get("model_args")
        if model_args is None:
            if self._verbose:
                print(
                    "[prediction-corruptor] Checkpoint missing model_args; skipping reload."
                )
            return False
        gptconf = GPTConfig(**model_args)
        if gptconf.block_size != self._expected_block_size:
            raise ValueError(
                "Checkpoint block size mismatch: "
                f"{gptconf.block_size} (checkpoint) vs {self._expected_block_size} (dataset)"
            )

        old_model = self._model
        self._model = None
        if old_model is not None:
            del old_model
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

        model = GPT(gptconf)
        state_dict = checkpoint.get("model")
        if state_dict is None:
            if self._verbose:
                print(
                    "[prediction-corruptor] Checkpoint missing model state_dict; skipping reload."
                )
            return False
        cleaned = CheckpointManager._cleanup_state_dict_keys(dict(state_dict))
        model.load_state_dict(cleaned)
        model.to(self._device)
        if self._dtype is not None:
            model = model.to(self._dtype)
        model.eval()
        self._model = model
        if self._verbose:
            print(f"[prediction-corruptor] Loaded checkpoint from {path}")
        return True

    @torch.no_grad()
    def _predict_with_model(
        self, model: torch.nn.Module, x: torch.Tensor, mask_bool: torch.Tensor
    ) -> torch.Tensor:
        device = self._device

        x_device = x.to(device)
        mask_device = mask_bool.to(device)

        masked_inputs = x_device.clone()
        masked_inputs[mask_device] = self._mask_token_id

        logits, _ = model(masked_inputs)
        logits = logits / self._temperature

        predicted_ids = torch.argmax(logits, dim=-1)
        corrupted = x_device.clone()
        corrupted[mask_device] = predicted_ids[mask_device]

        return corrupted.to(x.device)


__all__: Sequence[str] = ["CheckpointPredictionCorruptor"]
