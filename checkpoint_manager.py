from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Encapsulates saving and loading training checkpoints.

    Checkpoint format mirrors the previous inline structure:
    {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,         # dict used to build GPTConfig
        'iter_num': iter_num,             # int
        'best_val_loss': best_val_loss,   # float
        'config': config,                 # dict of run-time config values
    }
    """

    def __init__(self, out_dir: str, filename: str = 'ckpt.pt') -> None:
        self.out_dir = out_dir
        self.filename = filename
        # Persistent state registered by the training script
        self._model: Optional[torch.nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._model_args: Optional[Dict[str, Any]] = None
        self._config: Optional[Dict[str, Any]] = None
        self._iter_num: int = 0
        self._best_val_loss: float = float('inf')
        self._last_loaded_path: Optional[str] = None

    @property
    def path(self) -> str:
        return os.path.join(self.out_dir, self.filename)

    @property
    def last_loaded_path(self) -> Optional[str]:
        return self._last_loaded_path

    def _get_training_type_str(self, strict: bool = True) -> Optional[str]:
        """Return training_type string from config.meta. If strict and missing, return None."""
        try:
            if self._config and isinstance(self._config, dict):
                meta = self._config.get('meta')
                if isinstance(meta, dict) and 'training_type' in meta:
                    return str(meta['training_type'])
                if not strict and 'dataset' in self._config:
                    return str(self._config['dataset'])
        except Exception:
            pass
        return None

    def _versioned_name(self) -> str:
        # training type and iteration in filename, fallback to 'unknown' ONLY for naming
        ttype = self._get_training_type_str(strict=False) or 'unknown'
        return f"ckpt_{ttype}_{self._iter_num}.pt"

    # Registration / state update API
    def register_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def register_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer

    def set_metadata(self, model_args: Dict[str, Any], config: Dict[str, Any]) -> None:
        self._model_args = model_args
        self._config = config

    def update_progress(self, iter_num: int, best_val_loss: float) -> None:
        self._iter_num = iter_num
        self._best_val_loss = best_val_loss

    def _meta_file_path(self) -> str:
        return os.path.join(self.out_dir, 'ckpt_meta.json')

    def _load_meta(self) -> Dict[str, Any]:
        p = self._meta_file_path()
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_meta(self, data: Dict[str, Any]) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self._meta_file_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def _list_versioned(self) -> list[str]:
        if not os.path.exists(self.out_dir):
            return []
        return [f for f in os.listdir(self.out_dir) if f.startswith('ckpt_') and f.endswith('.pt')]

    def _resolve_latest(self, training_type: str) -> Optional[str]:
        """Return the latest versioned checkpoint path for given training_type or None."""
        files = self._list_versioned()
        candidates = []
        for f in files:
            name = f[:-3]  # drop .pt
            parts = name.split('_')
            if len(parts) < 3:
                continue
            # expected format: ckpt_{training_type}_{iter}
            prefix = parts[0]
            iter_part = parts[-1]
            ttype = '_'.join(parts[1:-1])
            if prefix != 'ckpt':
                continue
            if ttype != training_type:
                continue
            try:
                it = int(iter_part)
            except ValueError:
                continue
            candidates.append((it, f))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return os.path.join(self.out_dir, candidates[-1][1])

    def resolve_load_path(self, training_type_hint: Optional[str] = None) -> str:
        """Return the exact path that would be loaded under strict rules (no load)."""
        ttype = training_type_hint or self._get_training_type_str(strict=True)
        if not ttype:
            raise ValueError("training_type missing in config.meta; cannot resolve checkpoint deterministically")
        meta = self._load_meta()
        if str(ttype) in meta:
            p = os.path.join(self.out_dir, meta[str(ttype)]['last_path'])
            if not os.path.exists(p):
                raise FileNotFoundError(f"Metadata points to missing checkpoint: {p}")
            return p
        cand = self._resolve_latest(str(ttype))
        if cand is None or not os.path.exists(cand):
            raise FileNotFoundError(f"No versioned checkpoint found for training_type={ttype} in {self.out_dir}")
        return cand

    def save(self) -> str:
        """
        Save checkpoint to disk using registered state.
        Expects model to be unwrapped (raw_model) if DDP is used.
        Returns the primary checkpoint path written. Also writes a versioned copy.
        """
        if self._model is None or self._optimizer is None or self._model_args is None or self._config is None:
            raise RuntimeError("CheckpointManager missing registered state (model/optimizer/model_args/config)")
        os.makedirs(self.out_dir, exist_ok=True)
        checkpoint = {
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'model_args': self._model_args,
            'iter_num': self._iter_num,
            'best_val_loss': self._best_val_loss,
            'config': self._config,
        }
        # write stable path (kept for compatibility, not used for resume decisions)
        torch.save(checkpoint, self.path)
        # write versioned filename with training type and iteration
        versioned = os.path.join(self.out_dir, self._versioned_name())
        torch.save(checkpoint, versioned)
        # update metadata for strict resume
        ttype = self._get_training_type_str(strict=True)
        if ttype:
            meta = self._load_meta()
            meta[str(ttype)] = {"last_path": os.path.basename(versioned), "iter": self._iter_num}
            self._save_meta(meta)
        return versioned

    def load(self, device: Optional[torch.device | str] = None, training_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Load checkpoint from disk using strict resolution rules:
        1) If metadata exists for training_type, load that path.
        2) Else, resolve latest versioned checkpoint for training_type.
        No fallback to the stable ckpt.pt.
        """
        p = self.resolve_load_path(training_type_hint=training_type_hint)
        self._last_loaded_path = p
        return torch.load(p, map_location=device, weights_only=False)

    @staticmethod
    def _cleanup_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clean up known unwanted prefixes from state_dict keys, e.g., '_orig_mod.'
        Returns a possibly modified dict (in-place update pattern preserved).
        """
        unwanted_prefix = '_orig_mod.'
        # Work on a copy of keys to avoid runtime dict-size change during iteration
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        return state_dict

    def load_model_state(self, model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Apply checkpoint model state into the provided model, after cleaning keys.
        """
        cleaned = self._cleanup_state_dict_keys(state_dict)
        model.load_state_dict(cleaned)

