"""Utilities for configurable random-replacement corruption."""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

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
        self._original_multiplier = float(original_token_probability_multiplier)

    @property
    def candidate_token_ids(self) -> torch.Tensor:
        return self._candidate_tensor

    def corrupt(self, x: torch.Tensor, mask: torch.Tensor, rng) -> torch.Tensor:
        """Apply random replacement to ``x`` respecting ``mask``."""
        if x.shape != mask.shape:
            raise ValueError("x and mask must share the same shape")

        device = x.device
        corrupted = x.clone()
        masked_positions = mask.nonzero(as_tuple=False)
        if masked_positions.numel() == 0:
            return corrupted

        candidates = self._candidate_tensor.to(device)
        num_positions = masked_positions.size(0)
        num_candidates = candidates.size(0)

        if num_candidates == 0:
            raise ValueError("candidate_token_ids must contain at least one token")

        if num_candidates == 1:
            replacements = candidates.expand(num_positions)
            corrupted[masked_positions[:, 0], masked_positions[:, 1]] = replacements
            return corrupted

        original_tokens = x[masked_positions[:, 0], masked_positions[:, 1]]
        candidate_indices = torch.searchsorted(candidates, original_tokens, right=False)
        candidate_indices = torch.clamp(candidate_indices, max=num_candidates - 1)
        match_mask = candidates[candidate_indices] == original_tokens

        keep_prob = torch.zeros(num_positions, device=device, dtype=torch.float32)
        if self._original_multiplier > 0 and match_mask.any():
            denom = (num_candidates - 1) + self._original_multiplier
            keep_prob[match_mask] = self._original_multiplier / denom

        keep_draws = torch.rand(num_positions, generator=rng, device=device)
        keep_mask = keep_draws < keep_prob

        sampled_indices = torch.randint(
            0, num_candidates, (num_positions,), generator=rng, device=device
        )

        exclude_mask = match_mask & (~keep_mask)
        if exclude_mask.any():
            exclude_count = exclude_mask.sum()
            replacement_indices = torch.randint(
                0, num_candidates - 1, (exclude_count,), generator=rng, device=device
            )
            orig_indices = candidate_indices[exclude_mask]
            adjustment = (replacement_indices >= orig_indices).long()
            sampled_indices[exclude_mask] = replacement_indices + adjustment

        replacements = candidates[sampled_indices]
        if keep_mask.any():
            replacements[keep_mask] = original_tokens[keep_mask]

        corrupted[masked_positions[:, 0], masked_positions[:, 1]] = replacements
        return corrupted


FragmentSampler = Callable[[int, torch.Generator], torch.Tensor]


def apply_mixed_corruption(
    x: torch.Tensor,
    mask: torch.Tensor,
    rng,
    *,
    random_corruptor: RandomReplacementCorruptor,
    mask_token_id: int,
    fragment_sampler: FragmentSampler,
    mixture_weights: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> torch.Tensor:
    """Apply hybrid corruption with random, [MASK], and fragment strategies."""
    if x.shape != mask.shape:
        raise ValueError("x and mask must share the same shape")

    mask_bool = mask.to(dtype=torch.bool)
    if not mask_bool.any():
        return x.clone()

    total_weight = float(sum(mixture_weights))
    if total_weight <= 0:
        raise ValueError("mixture_weights must sum to a positive value")

    normalized = tuple(weight / total_weight for weight in mixture_weights)
    random_share, mask_share, _ = normalized
    random_threshold = random_share
    mask_threshold = random_share + mask_share

    selection = torch.rand(
        mask.shape, generator=rng, device=x.device, dtype=torch.float32
    )
    random_mask = mask_bool & (selection < random_threshold)
    mask_token_mask = mask_bool & (selection >= random_threshold) & (selection < mask_threshold)
    fragment_mask = mask_bool & ~(random_mask | mask_token_mask)

    corrupted = x.clone()

    if random_mask.any():
        random_corrupted = random_corruptor.corrupt(x, random_mask, rng)
        corrupted[random_mask] = random_corrupted[random_mask]

    if mask_token_mask.any():
        corrupted[mask_token_mask] = mask_token_id

    if fragment_mask.any():
        fragments = fragment_sampler(x.shape[0], rng)
        if fragments.shape != x.shape:
            raise ValueError(
                "Fragment sampler returned tensor with shape "
                f"{fragments.shape}, expected {x.shape}."
            )
        corrupted[fragment_mask] = fragments[fragment_mask]

    return corrupted


def build_candidate_token_ids(
    vocab_size: int,
    *,
    excluded_token_ids: Optional[Iterable[int]] = None,
) -> Sequence[int]:
    """Construct the candidate set by excluding reserved tokens."""
    excluded = set(int(token_id) for token_id in (excluded_token_ids or ()))
    return [token_id for token_id in range(vocab_size) if token_id not in excluded]
