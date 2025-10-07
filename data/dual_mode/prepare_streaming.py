"""
Dual-mode streaming provider that combines LANGUAGE_MODEL and SEQUENCE_SCORER batches.

This provider generates batches for both modes in a single stream, alternating between
char_diffusion (LANGUAGE_MODEL) and sequence_scorer (SEQUENCE_SCORER) batches.
Each batch includes metadata indicating its model_mode, enabling runtime mode switching
during training.

Configuration:
- mode_distribution: dict with 'language_model' and 'sequence_scorer' ratios (default: 0.5 each)
- alternation_frequency: how often to switch modes (default: 1 = every batch)
"""
from __future__ import annotations

import os
from typing import Dict, Any

import torch

from data.common.provider_base import DataProviderBase
from model import ModelMode


class DualModeProvider(DataProviderBase):
    """
    Combines char_diffusion and sequence_scorer providers into a single dual-mode stream.
    
    Generates batches for both LANGUAGE_MODEL and SEQUENCE_SCORER modes, with each batch
    tagged with its model_mode for runtime switching during training.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        # Accept full config for dataset-specific parsing
        cfg = kwargs.pop('config', {}) or {}
        
        # Mode distribution configuration
        self.mode_distribution = cfg.get('mode_distribution', {
            'language_model': 0.5,
            'sequence_scorer': 0.5
        })
        
        # Alternation frequency (1 = alternate every batch, 2 = every 2 batches, etc.)
        self.alternation_frequency = cfg.get('alternation_frequency', 1)
        
        super().__init__(*args, **kwargs)
        
        # Initialize sub-providers for each mode
        self._init_language_model_provider(cfg)
        self._init_sequence_scorer_provider(cfg)
        
        # Batch counter for alternation
        self._batch_counter = 0
        
        if self.verbose:
            print(f"[dual_mode] DualModeProvider initialized:")
            print(f"  mode_distribution: {self.mode_distribution}")
            print(f"  alternation_frequency: {self.alternation_frequency}")
            print(f"  vocab_size: {self.vocab_size}")
    
    def _init_language_model_provider(self, cfg: Dict) -> None:
        """Initialize char_diffusion provider for LANGUAGE_MODEL batches."""
        from data.char_diffusion.prepare_streaming import CharDiffusionProvider
        
        # Create a sub-provider instance (reuse same data_dir, batch_size, etc.)
        # We'll call its sample_batch method directly
        self.lm_provider = CharDiffusionProvider(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            block_size=self.block_size,
            target_size=self.target_size,
            batches_per_file=self.batches_per_file,
            max_backlog_files=self.max_backlog_files,
            sleep_seconds=self.sleep_seconds,
            seed=self.seed,
            verbose=False,  # Suppress sub-provider logging
            config=cfg,
        )
        
        # Inherit vocab from char_diffusion provider
        self.vocab_size = self.lm_provider.vocab_size
        self.stoi = self.lm_provider.stoi
        self.itos = self.lm_provider.itos
        self.mask_token_id = self.lm_provider.mask_token_id
        self.pad_token_id = getattr(self.lm_provider, 'pad_token_id', None)
    
    def _init_sequence_scorer_provider(self, cfg: Dict) -> None:
        """Initialize sequence_scorer provider for SEQUENCE_SCORER batches."""
        from data.sequence_scorer.prepare_streaming import SequenceScorerProvider
        
        # Create a sub-provider instance
        self.ss_provider = SequenceScorerProvider(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            block_size=self.block_size,
            target_size=self.target_size,
            batches_per_file=self.batches_per_file,
            max_backlog_files=self.max_backlog_files,
            sleep_seconds=self.sleep_seconds,
            seed=self.seed + 1,  # Different seed for diversity
            verbose=False,  # Suppress sub-provider logging
            config=cfg,
        )
        
        # Verify vocab consistency
        if self.ss_provider.vocab_size != self.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: char_diffusion={self.vocab_size}, "
                f"sequence_scorer={self.ss_provider.vocab_size}"
            )
    
    def build_meta(self) -> Dict:
        """Build metadata for dual-mode dataset."""
        if self.block_size is None:
            raise ValueError("block_size must be set for DualModeProvider")
        
        return {
            "dataset_name": "dual_mode",
            "training_type": "DUAL_MODE",  # New training type
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "pad_token_id": self.pad_token_id,
            "cls_token_id": self.ss_provider.cls_token_id,
            "stoi": self.stoi,
            "itos": self.itos,
            "mode_distribution": self.mode_distribution,
            "alternation_frequency": self.alternation_frequency,
            "batch_schema": [
                # Schema is flexible - batches can have different fields based on mode
                # LANGUAGE_MODEL batches: x, y (both int64, shape [block_size])
                # SEQUENCE_SCORER batches: input_ids (int64, [block_size]), targets (float32, [])
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input", "mode": "language_model"},
                {"name": "y", "dtype": "int64", "shape": [self.target_size or self.block_size], "role": "target", "mode": "language_model"},
                {"name": "input_ids", "dtype": "int64", "shape": [self.block_size], "role": "input", "mode": "sequence_scorer"},
                {"name": "targets", "dtype": "float32", "shape": [], "role": "target", "mode": "sequence_scorer"},
            ],
        }
    
    def _determine_mode_for_batch(self, batch_idx: int, rng: torch.Generator) -> str:
        """
        Determine which mode to use for this batch.
        
        Uses alternation_frequency to control switching frequency.
        Within each alternation window, uses mode_distribution to decide.
        """
        # Determine alternation window
        window_idx = batch_idx // self.alternation_frequency
        
        # Use RNG seeded by window to ensure deterministic but varied distribution
        window_rng = torch.Generator()
        window_rng.manual_seed(rng.initial_seed() + window_idx)
        
        # Sample mode based on distribution
        rand_val = torch.rand(1, generator=window_rng).item()
        lm_ratio = self.mode_distribution.get('language_model', 0.5)
        
        if rand_val < lm_ratio:
            return 'language_model'
        else:
            return 'sequence_scorer'
    
    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, Any]:
        """
        Sample one batch, alternating between LANGUAGE_MODEL and SEQUENCE_SCORER modes.
        
        Returns a dict with tensors and metadata including 'model_mode'.
        """
        # Determine mode for this batch
        mode = self._determine_mode_for_batch(self._batch_counter, rng)
        self._batch_counter += 1
        
        # Generate batch from appropriate provider
        if mode == 'language_model':
            # Get batch from char_diffusion provider
            batch = self.lm_provider.sample_batch(split, rng)
            # Ensure batch has x, y keys (char_diffusion format)
            if 'x' not in batch or 'y' not in batch:
                raise ValueError("CharDiffusionProvider batch must have 'x' and 'y' keys")
        else:
            # Get batch from sequence_scorer provider
            batch = self.ss_provider.sample_batch(split, rng)
            # Ensure batch has input_ids, targets keys (sequence_scorer format)
            if 'input_ids' not in batch or 'targets' not in batch:
                raise ValueError("SequenceScorerProvider batch must have 'input_ids' and 'targets' keys")
        
        # Add model_mode metadata to batch
        batch['model_mode'] = mode
        
        return batch


# Export as Provider for prepare.py discovery
Provider = DualModeProvider

