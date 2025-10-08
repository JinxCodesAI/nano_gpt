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
        # Each sub-provider uses its own data directory and loads its own vocab/meta
        self._init_language_model_provider(cfg)
        self._init_sequence_scorer_provider(cfg)

        # Batch counter for alternation
        self._batch_counter = 0

        # RNG for mode selection (initialized once)
        self._mode_rng = torch.Generator()
        self._mode_rng.manual_seed(self.seed + 999)  # Different seed from data generation

        # Current mode (stays same for alternation_frequency batches)
        self._current_mode = None
        self._batches_in_current_mode = 0

        if self.verbose:
            print(f"[dual_mode] DualModeProvider initialized:")
            print(f"  mode_distribution: {self.mode_distribution}")
            print(f"  alternation_frequency: {self.alternation_frequency}")
            print(f"  vocab_size: {self.vocab_size}")
            print(f"  Special tokens:")
            print(f"    MASK: {self.mask_token_id}")
            print(f"    PAD:  {self.pad_token_id}")
            print(f"    CLS:  {self.cls_token_id}")
            print(f"  Special tokens:")
            print(f"    MASK: {self.mask_token_id}")
            print(f"    PAD:  {self.pad_token_id}")
            print(f"    CLS:  {self.cls_token_id}")
            print(f"  Batch normalization: all batches use (x, y) keys")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  pad_token_id: {self.pad_token_id}")
            print(f"  cls_token_id: {self.cls_token_id}")
            print(f"  Special tokens: MASK={self.mask_token_id}, PAD={self.pad_token_id}, CLS={self.cls_token_id}")

    def _init_language_model_provider(self, cfg: Dict) -> None:
        """Initialize char_diffusion provider for LANGUAGE_MODEL batches."""
        from data.char_diffusion.prepare_streaming import CharDiffusionProvider
        import os

        # Use char_diffusion's own data directory
        char_diffusion_dir = os.path.join(os.path.dirname(self.data_dir), 'char_diffusion')

        # Create a sub-provider instance
        # We'll call its sample_batch method directly (not run() method)
        self.lm_provider = CharDiffusionProvider(
            data_dir=char_diffusion_dir,
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
        self.cls_token_id = getattr(self.lm_provider, 'cls_token_id', None)
    
    def _init_sequence_scorer_provider(self, cfg: Dict) -> None:
        """Initialize sequence_scorer provider for SEQUENCE_SCORER batches."""
        from data.sequence_scorer.prepare_streaming import SequenceScorerProvider
        import os

        # Use sequence_scorer's own data directory
        sequence_scorer_dir = os.path.join(os.path.dirname(self.data_dir), 'sequence_scorer')

        # Create a sub-provider instance
        self.ss_provider = SequenceScorerProvider(
            data_dir=sequence_scorer_dir,
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
                # Mixed schema - batches have different fields based on mode
                # LANGUAGE_MODEL batches: x, y, attention_mask
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input", "mode": "language_model"},
                {"name": "y", "dtype": "int64", "shape": [self.target_size or self.block_size], "role": "target", "mode": "language_model"},
                {"name": "attention_mask", "dtype": "int64", "shape": [self.block_size], "role": "mask", "mode": "language_model"},
                # SEQUENCE_SCORER batches: input_ids, targets
                {"name": "input_ids", "dtype": "int64", "shape": [self.block_size], "role": "input", "mode": "sequence_scorer"},
                {"name": "targets", "dtype": "float32", "shape": [], "role": "target", "mode": "sequence_scorer"},
            ],
        }
    
    def _determine_mode_for_batch(self) -> str:
        """
        Determine which mode to use for this batch.

        Every alternation_frequency batches, randomly select a new mode.
        Keep the same mode for alternation_frequency consecutive batches.
        """
        # Check if we need to select a new mode
        if self._current_mode is None or self._batches_in_current_mode >= self.alternation_frequency:
            # Select new mode based on distribution
            lm_weight = self.mode_distribution.get('language_model', 0.5)
            ss_weight = self.mode_distribution.get('sequence_scorer', 0.5)
            total_weight = lm_weight + ss_weight

            # Random number from 0 to total_weight
            rand_val = torch.rand(1, generator=self._mode_rng).item() * total_weight

            # Select mode based on threshold
            if rand_val < lm_weight:
                self._current_mode = 'language_model'
            else:
                self._current_mode = 'sequence_scorer'

            # Reset counter for new mode
            self._batches_in_current_mode = 0

        # Increment counter
        self._batches_in_current_mode += 1

        return self._current_mode
    
    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, Any]:
        """
        Sample one batch, alternating between LANGUAGE_MODEL and SEQUENCE_SCORER modes.

        Returns a dict with tensors and metadata including 'model_mode'.
        Batch keys vary by mode (x/y for language_model, input_ids/targets for sequence_scorer).
        """
        # Determine mode for this batch
        mode = self._determine_mode_for_batch()

        # Generate batch from appropriate provider
        if mode == 'language_model':
            # Get batch from char_diffusion provider (keys: x, y, attention_mask)
            print(f"[dual_mode] Sampling language_model batch")
            batch = self.lm_provider.sample_batch(split, rng)
        else:
            # Get batch from sequence_scorer provider (keys: input_ids, targets)
            print(f"[dual_mode] Sampling sequence_scorer batch")
            batch = self.ss_provider.sample_batch(split, rng)

        # Add model_mode metadata to batch
        batch['model_mode'] = mode

        return batch


# Export as Provider for prepare.py discovery
Provider = DualModeProvider

