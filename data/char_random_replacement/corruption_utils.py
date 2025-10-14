"""Utilities for configurable random-replacement corruption."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


class RandomReplacementCorruptor:
    """Replace masked tokens with weighted random samples.

    Args:
        candidate_token_ids: Sequence of token IDs eligible for sampling.
        original_token_probability_multiplier: Multiplier applied to the
            original token's sampling weight. Values >1 favour keeping the
            original symbol, whereas values in (0, 1) discourage it.
    """

    def __init__(
        self,
        candidate_token_ids: Sequence[int],
        *,
        original_token_probability_multiplier: float = 1.0,
    ) -> None:
        if not candidate_token_ids:
            raise ValueError("candidate_token_ids must not be empty")
        if original_token_probability_multiplier < 0:
            raise ValueError("original_token_probability_multiplier must be non-negative")

        # Ensure deterministic ordering for reproducibility.
        sorted_candidates = sorted(int(token_id) for token_id in candidate_token_ids)
        self._candidate_tensor = torch.tensor(sorted_candidates, dtype=torch.long)
        self._candidate_index = {
            int(token_id): idx for idx, token_id in enumerate(sorted_candidates)
        }
        self._original_multiplier = float(original_token_probability_multiplier)

    @property
    def candidate_token_ids(self) -> torch.Tensor:
        return self._candidate_tensor

    def corrupt(self, x: torch.Tensor, mask: torch.Tensor, rng) -> torch.Tensor:
        """Apply random replacement to ``x`` respecting ``mask``."""
        if x.shape != mask.shape:
            raise ValueError("x and mask must share the same shape")

        corrupted = x.clone()
        masked_positions = mask.nonzero(as_tuple=False)
        if masked_positions.numel() == 0:
            return corrupted

        candidates = self._candidate_tensor.to(x.device)
        num_positions = masked_positions.size(0)
        num_candidates = candidates.size(0)

        # Base weights initialised to 1 for all candidates.
        weights = torch.ones((num_positions, num_candidates), dtype=torch.float32, device=x.device)

        if self._original_multiplier != 1.0:
            original_tokens = x[masked_positions[:, 0], masked_positions[:, 1]].tolist()
            for row_idx, token in enumerate(original_tokens):
                candidate_idx = self._candidate_index.get(int(token))
                if candidate_idx is not None:
                    weights[row_idx, candidate_idx] *= self._original_multiplier

        weight_sums = weights.sum(dim=1, keepdim=True)
        # Guard against degenerate configurations that zero out every weight.
        if torch.any(weight_sums == 0):
            raise ValueError("Sampling weights must be positive for every masked position")

        probabilities = weights / weight_sums
        sampled_indices = torch.multinomial(
            probabilities, 1, replacement=True, generator=rng
        ).squeeze(1)

        replacements = candidates[sampled_indices]
        corrupted[masked_positions[:, 0], masked_positions[:, 1]] = replacements
        return corrupted


def build_candidate_token_ids(
    vocab_size: int,
    *,
    excluded_token_ids: Optional[Iterable[int]] = None,
) -> Sequence[int]:
    """Construct the candidate set by excluding reserved tokens."""
    excluded = set(int(token_id) for token_id in (excluded_token_ids or ()))
    return [token_id for token_id in range(vocab_size) if token_id not in excluded]
