"""Streaming provider that iteratively applies sticky masking rounds.

This dataset reuses the Shakespeare character corpus from ``char_diffusion`` but
changes the masking strategy. For each sampled example we run the sticky masking
algorithm repeatedly, producing ``max_rounds`` progressively more corrupted
samples whose targets are always the original characters. This mirrors the
training objective of ``char_diffusion`` while giving the model visibility into
multiple diffusion steps.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Tuple

import torch

from data.char_diffusion.prepare_streaming import (
    CharDiffusionProvider,
    apply_bert_style_corruption_cpu,
)


class StickyRoundsProvider(CharDiffusionProvider):
    """Generate iterative sticky-masking samples from shared Shakespeare data."""

    def __init__(self, *args, **kwargs) -> None:
        cfg = kwargs.get("config", {}) or {}
        self.sticky_p1_probability: float = cfg.get("sticky_p1_probability", 0.15)
        self.sticky_p2_probability: float = cfg.get("sticky_p2_probability", 0.5)
        self.max_rounds: int = int(cfg.get("max_rounds", 3))
        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")

        default_source = os.path.join(os.path.dirname(__file__), "..", "char_diffusion")
        self.source_data_dir = cfg.get("source_data_dir", os.path.abspath(default_source))

        super().__init__(*args, **kwargs)

        self._base_samples_per_batch = math.ceil(self.batch_size / self.max_rounds)

    # ------------------------------------------------------------------
    # Data loading overrides
    # ------------------------------------------------------------------
    def _load_input_text(self, filename: str = "input.txt") -> str:  # type: ignore[override]
        """Load the shared Shakespeare corpus from ``char_diffusion``."""
        source_path = os.path.join(self.source_data_dir, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(
                f"Shared input text not found at {source_path}. Ensure char_diffusion data is prepared."
            )
        with open(source_path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def build_meta(self) -> Dict[str, Any]:  # type: ignore[override]
        meta = super().build_meta()
        meta.update(
            {
                "dataset_name": "char_diffusion_sticky_rounds",
                "sticky_p1_probability": self.sticky_p1_probability,
                "sticky_p2_probability": self.sticky_p2_probability,
                "max_rounds": self.max_rounds,
            }
        )
        return meta

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_batch(self, split: str, rng) -> Dict[str, Any]:  # type: ignore[override]
        if not self.enable_line_aligned_sequences:
            raise ValueError("StickyRoundsProvider requires line-aligned sequences to be enabled.")

        base_sequences, content_lengths = self._build_line_aligned_variable_length(
            split, self._base_samples_per_batch, rng
        )

        device = base_sequences.device
        x_out = torch.full(
            (self.batch_size, self.block_size),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        y_out = torch.full_like(x_out, self.ignore_index)
        round_index = torch.zeros(self.batch_size, dtype=torch.long, device=device)

        cursor = 0
        for base_idx in range(self._base_samples_per_batch):
            seq = base_sequences[base_idx]
            content_len = int(content_lengths[base_idx].item())
            seq_x, seq_y = self._generate_rounds_for_sequence(seq, content_len, rng)

            rounds_to_copy = min(seq_x.shape[0], self.batch_size - cursor)
            if rounds_to_copy <= 0:
                break

            next_cursor = cursor + rounds_to_copy
            x_out[cursor:next_cursor] = seq_x[:rounds_to_copy]
            y_out[cursor:next_cursor] = seq_y[:rounds_to_copy]
            round_index[cursor:next_cursor] = torch.arange(
                rounds_to_copy, dtype=torch.long, device=device
            )
            cursor = next_cursor

            if cursor >= self.batch_size:
                break

        if cursor != self.batch_size:
            raise RuntimeError(
                f"Expected to fill {self.batch_size} samples but produced {cursor}."
            )

        return {
            "x": x_out,
            "y": y_out,
            "round_index": round_index,
            "model_mode": "language_model",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_rounds_for_sequence(
        self, seq: torch.Tensor, content_len: int, rng: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = seq.device
        rounds_x = torch.full(
            (self.max_rounds, self.block_size),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        rounds_y = torch.full_like(rounds_x, self.ignore_index)

        if content_len <= 0:
            return rounds_x, rounds_y

        original = seq[:content_len].clone()
        current_tokens = original.clone()
        current_mask = torch.zeros(content_len, dtype=torch.bool, device=device)

        for round_idx in range(self.max_rounds):
            next_mask = self._sticky_round(current_mask, rng)

            if not next_mask.any():
                forced_index = int(torch.randint(0, content_len, (1,), generator=rng).item())
                next_mask[forced_index] = True

            new_positions = next_mask & ~current_mask
            if new_positions.any():
                new_mask_tensor = new_positions.unsqueeze(0)
                corrupted = apply_bert_style_corruption_cpu(
                    original.unsqueeze(0),
                    new_mask_tensor,
                    self.mask_token_id,
                    self.base_vocab_size,
                    rng,
                ).squeeze(0)
                current_tokens[new_positions] = corrupted[new_positions]

            current_mask = next_mask

            content_targets = torch.full(
                (content_len,), self.ignore_index, dtype=torch.long, device=device
            )
            content_targets[current_mask] = original[current_mask]

            rounds_x[round_idx, :content_len] = current_tokens
            rounds_y[round_idx, :content_len] = content_targets

        return rounds_x, rounds_y

    def _sticky_round(
        self, current_mask: torch.Tensor, rng: torch.Generator
    ) -> torch.Tensor:
        if current_mask.numel() == 0:
            return current_mask

        device = current_mask.device
        rand_vals = torch.rand(current_mask.shape, generator=rng, device=device)

        neighbor_masked = torch.zeros_like(current_mask)
        if current_mask.numel() > 1:
            neighbor_masked[1:] |= current_mask[:-1]
            neighbor_masked[:-1] |= current_mask[1:]

        probs = torch.full(
            current_mask.shape, self.sticky_p1_probability, dtype=rand_vals.dtype, device=device
        )
        if self.sticky_p2_probability != self.sticky_p1_probability:
            probs = probs.masked_fill(neighbor_masked, self.sticky_p2_probability)
        else:
            probs = probs

        new_masks = (rand_vals < probs) & ~current_mask
        return current_mask | new_masks


Provider = StickyRoundsProvider
