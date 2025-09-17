"""
Global config validator for training and data generation.

Usage from train.py and prepare.py:
    from config.validator import validate_config
    validate_config(globals())  # or a dict of config values

It raises ValueError on any inconsistency or missing required fields.
"""
from __future__ import annotations
from typing import Dict, Any


def _require(d: Dict[str, Any], key: str, typ, allow_none: bool = False):
    if key not in d:
        raise ValueError(f"Missing required config key: {key}")
    val = d[key]
    if val is None and allow_none:
        return
    if not isinstance(val, typ):
        raise ValueError(f"Config key {key} must be of type {typ}, got {type(val)}")


def validate_config(cfg: Dict[str, Any]) -> None:
    # dataset name required
    _require(cfg, 'dataset', str)
    # training core sizes
    _require(cfg, 'batch_size', int)
    _require(cfg, 'block_size', int)
    # model / training loop integrity
    _require(cfg, 'max_iters', int)
    _require(cfg, 'learning_rate', (float, int))
    # streaming consumer knobs (optional)
    # prefer_queue default True; caching/timeout/sleep optional
    # If streaming mode enforced, require provider to be able to produce data
    # No explicit checks here; DatasetConsumer will raise if queue missing.

    # Optional target_size defaults to block_size
    if 'target_size' in cfg and cfg['target_size'] is not None:
        if not isinstance(cfg['target_size'], int):
            raise ValueError("target_size must be int if provided")

    # Warn/error about conflicting vocab_size overrides
    if 'vocab_size' in cfg and cfg['vocab_size'] is not None:
        if not isinstance(cfg['vocab_size'], int):
            raise ValueError("vocab_size must be int if provided")

    # Additional hooks could validate training_type specific requirements once meta is available
    # Sequence scorer specific: ensure CLS id fits within vocab
    mm = cfg.get('model_mode') or cfg.get('mode')
    if isinstance(mm, str):
        mm_lower = mm.lower()
    else:
        mm_lower = str(mm).lower() if mm is not None else None
    if mm_lower == 'sequence_scorer':
        cls_id = cfg.get('cls_token_id', None)
        if cls_id is not None:
            if not isinstance(cls_id, int):
                raise ValueError(f"cls_token_id must be int, got {type(cls_id)}")
            vocab_size = cfg.get('vocab_size', None)
            if not isinstance(vocab_size, int):
                raise ValueError("sequence_scorer requires integer vocab_size when cls_token_id is set")
            if cls_id >= vocab_size:
                raise ValueError(
                    f"Config error: sequence_scorer requires vocab_size >= cls_token_id + 1. "
                    f"Got vocab_size={vocab_size}, cls_token_id={cls_id}. "
                    f"Set vocab_size to {cls_id + 1} or higher.")

    # For now we keep it minimal and strict on essentials.

