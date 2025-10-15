"""Utility helpers for iterative sampling schedules and noise injection."""
from __future__ import annotations

import math
from typing import Iterable, Union

import torch


def _interp_linear(s: int, S: int, start: float, end: float) -> float:
    """Linearly interpolate between start and end for iteration ``s`` out of ``S``."""
    t = 0.0 if S <= 1 else s / (S - 1)
    return start + (end - start) * t


def _interp_cosine(s: int, S: int, start: float, end: float) -> float:
    """Cosine anneal from start to end across ``S`` iterations."""
    t = 0.0 if S <= 1 else s / (S - 1)
    w = 0.5 * (1.0 + math.cos(math.pi * t))
    return end + (start - end) * w


def compute_noise_ratio(
    iteration: int,
    max_iterations: int,
    schedule: str,
    start: float,
    end: float,
) -> float:
    """Return the noise ratio for a given iteration according to the schedule."""
    if schedule == 'linear':
        return _interp_linear(iteration, max_iterations, start, end)
    if schedule == 'cosine':
        return _interp_cosine(iteration, max_iterations, start, end)
    return 0.0


def apply_re_noise(
    x: torch.Tensor,
    max_token_pos: int,
    initial_length: int,
    vocab_size: int,
    ratio: float,
    device: Union[torch.device, str],
    avoid_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Apply re-noise to the active non-prompt region of ``x`` in-place."""
    if ratio <= 0.0:
        return x

    seq_len = x.size(1)
    clamped_end = max(0, min(seq_len, max_token_pos))
    active_len = max(0, clamped_end - initial_length)
    if active_len == 0:
        return x

    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    if active_len > 0:
        mask[0, initial_length:clamped_end] = True

    bern = torch.rand((1, seq_len), device=device) < ratio
    renoise_mask = mask & bern

    if not renoise_mask.any():
        return x

    if not avoid_ids:
        rand = torch.randint(low=0, high=vocab_size, size=x.shape, device=device)
    else:
        rand = torch.randint(low=0, high=vocab_size, size=x.shape, device=device)
        avoid = torch.tensor(sorted({int(i) for i in avoid_ids}), dtype=torch.long, device=device)
        for _ in range(4):
            bad = (rand[..., None] == avoid).any(-1)
            needs_resample = bad & renoise_mask
            if not needs_resample.any():
                break
            rand = torch.where(needs_resample, torch.randint(0, vocab_size, x.shape, device=device), rand)

    x = torch.where(renoise_mask, rand, x)
    return x


__all__ = [
    'compute_noise_ratio',
    'apply_re_noise',
]
