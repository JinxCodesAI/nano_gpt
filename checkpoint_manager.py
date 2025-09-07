from __future__ import annotations
import os
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

    @property
    def path(self) -> str:
        return os.path.join(self.out_dir, self.filename)

    def _versioned_name(self) -> str:
        # training type and iteration in filename, fallback to 'unknown'
        ttype = 'unknown'
        try:
            if self._config and isinstance(self._config, dict):
                # prefer meta training_type if present
                meta = self._config.get('meta')
                if isinstance(meta, dict) and 'training_type' in meta:
                    ttype = str(meta['training_type'])
                elif 'dataset' in self._config:
                    # fallback to dataset name as a hint if training_type unknown
                    ttype = str(self._config['dataset'])
        except Exception:
            pass
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
        # write stable path
        torch.save(checkpoint, self.path)
        # write versioned filename with training type and iteration
        versioned = os.path.join(self.out_dir, self._versioned_name())
        torch.save(checkpoint, versioned)
        return versioned

    def load(self, device: Optional[torch.device | str] = None) -> Dict[str, Any]:
        """
        Load checkpoint from disk and return the raw checkpoint dict.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Checkpoint not found at {self.path}")
        ckpt = torch.load(self.path, map_location=device)
        return ckpt

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

