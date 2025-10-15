"""Utility functions for sampling and diffusion-style decoding."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

__all__ = [
    "_interp_linear",
    "_interp_cosine",
    "compute_noise_ratio",
    "apply_re_noise",
    "blend_with_uniform",
]


def _interp_linear(iteration: int, total_iterations: int, start: float, end: float) -> float:
    """Return a linearly interpolated value for the current iteration."""
    if total_iterations <= 1:
        return start
    t = iteration / float(total_iterations - 1)
    return start + (end - start) * t


def _interp_cosine(iteration: int, total_iterations: int, start: float, end: float) -> float:
    """Return a cosine-annealed interpolation for the current iteration."""
    if total_iterations <= 1:
        return start
    t = iteration / float(total_iterations - 1)
    weight = 0.5 * (1.0 + math.cos(math.pi * t))
    return end + (start - end) * weight


def compute_noise_ratio(
    iteration: int,
    total_iterations: int,
    schedule: str,
    start: float,
    end: float,
) -> float:
    """Compute the noise (or temperature) ratio for the given iteration."""
    if schedule == "linear":
        return _interp_linear(iteration, total_iterations, start, end)
    if schedule == "cosine":
        return _interp_cosine(iteration, total_iterations, start, end)
    return 0.0


def _prepare_avoid_tensor(avoid_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    unique_ids = sorted(set(int(t) for t in avoid_ids))
    if not unique_ids:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(unique_ids, dtype=torch.long, device=device)


def apply_re_noise(
    x: torch.LongTensor,
    max_token_pos: int,
    initial_length: int,
    vocab_size: int,
    ratio: float,
    device: torch.device,
    avoid_ids: Sequence[int] = (),
) -> torch.LongTensor:
    """Re-sample a subset of active positions using random tokens.

    Args:
        x: Token tensor of shape ``(1, seq_len)`` that will be modified in-place.
        max_token_pos: End (exclusive) of the active decoding window.
        initial_length: Number of prompt tokens that should remain untouched.
        vocab_size: Size of the tokenizer vocabulary.
        ratio: Fraction of active positions to randomise in ``[0, 1]``.
        device: Torch device hosting ``x``.
        avoid_ids: Token ids to avoid sampling (e.g., padding).
    """
    if ratio <= 0.0:
        return x

    seq_len = x.size(1)
    max_idx = max(0, min(seq_len, max_token_pos))
    active_len = max(0, max_idx - initial_length)
    if active_len == 0:
        return x

    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    mask[0, initial_length:max_idx] = True

    bernoulli = torch.rand((1, seq_len), device=device) < ratio
    renoise_mask = mask & bernoulli
    if not renoise_mask.any():
        return x

    rand_tokens = torch.randint(low=0, high=vocab_size, size=x.shape, device=device)
    if avoid_ids:
        avoid_tensor = _prepare_avoid_tensor(avoid_ids, device)
        if avoid_tensor.numel() > 0:
            for _ in range(4):
                collisions = (rand_tokens[..., None] == avoid_tensor).any(-1)
                needs_resample = collisions & renoise_mask
                if not needs_resample.any():
                    break
                fresh_samples = torch.randint(0, vocab_size, x.shape, device=device)
                rand_tokens = torch.where(needs_resample, fresh_samples, rand_tokens)

    x = torch.where(renoise_mask, rand_tokens, x)
    return x


def blend_with_uniform(probs: torch.Tensor, blend_ratio: float) -> torch.Tensor:
    """Blend a probability tensor with a uniform distribution."""
    if blend_ratio <= 0.0:
        return probs
    vocab_size = probs.size(-1)
    uniform = torch.full_like(probs, 1.0 / vocab_size)
    return (1.0 - blend_ratio) * probs + blend_ratio * uniform
