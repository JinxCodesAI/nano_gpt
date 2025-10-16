"""Utility helpers for iterative sampling schedules and structural edits."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

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
    replace_character_id: int,
    ratio: float,
    device: Union[torch.device, str],
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

    replacement = torch.full_like(x, fill_value=replace_character_id)
    x = torch.where(renoise_mask, replacement, x)
    return x


def uncertainty_from_logprobs(log_probs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Return per-position uncertainty ``1 - p(token)`` as a detached 1D tensor."""
    with torch.no_grad():
        if log_probs.dim() != 3 or tokens.dim() != 2:
            raise ValueError('Expected log_probs shape (1, T, V) and tokens shape (1, T).')
        if log_probs.size(0) != 1 or tokens.size(0) != 1:
            raise ValueError('uncertainty_from_logprobs only supports batch size 1.')

        T = tokens.size(1)
        if log_probs.size(1) < T:
            raise ValueError('log_probs shorter than tokens in sequence dimension.')

        idx = torch.arange(T, device=log_probs.device)
        p_self = log_probs[0, idx, tokens[0, idx]].exp()
        return (1.0 - p_self).detach()


def score_gaps_for_insertion(u: torch.Tensor) -> torch.Tensor:
    """Score gaps between tokens using neighbouring uncertainty."""
    if u.numel() < 2:
        return torch.empty(0, device=u.device, dtype=u.dtype)
    return torch.maximum(u[:-1], u[1:])


def right_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """Shift ``x[:, start:end)`` right by one and place ``fill_id`` at ``start``."""
    if end - start <= 0:
        return
    x[:, start + 1 : end] = x[:, start:end - 1]
    x[:, start] = fill_id


def left_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """Shift ``x[:, start:end)`` left by one and place ``fill_id`` at ``end-1``."""
    if end - start <= 0:
        return
    x[:, start:end - 1] = x[:, start + 1 : end]
    x[:, end - 1] = fill_id


def select_topk_mask(
    scores: torch.Tensor,
    k: int,
    *,
    forbid_lo: int = 0,
    forbid_hi: int = 0,
    additional_forbid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return boolean mask for the ``k`` highest scores respecting forbidden regions."""

    if scores.dim() != 1:
        raise ValueError('scores must be 1D tensor')

    if scores.numel() == 0 or k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    mask = torch.ones_like(scores, dtype=torch.bool)
    lo = max(0, forbid_lo)
    hi = min(scores.numel(), max(0, forbid_hi))
    if hi < lo:
        hi = lo
    if hi > lo:
        mask[lo:hi] = False

    if additional_forbid is not None:
        if additional_forbid.shape != scores.shape:
            raise ValueError('additional_forbid must match scores shape')
        mask &= ~additional_forbid

    available = int(mask.sum().item())
    if available <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    k = min(k, available)
    masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
    topk_idx = torch.topk(masked_scores, k=k).indices
    sel = torch.zeros_like(scores, dtype=torch.bool)
    sel[topk_idx] = True
    return sel


def deletion_scores_from_probs(
    probs: torch.Tensor,
    tokens: torch.Tensor,
    *,
    margin: float,
    lam: float,
) -> torch.Tensor:
    """Compute deletion desirability per position using lookahead heuristics."""
    if probs.dim() != 3 or tokens.dim() != 2:
        raise ValueError('Expected probs shape (1, T, V) and tokens shape (1, T).')
    if probs.size(0) != 1 or tokens.size(0) != 1:
        raise ValueError('deletion_scores_from_probs only supports batch size 1.')

    T = tokens.size(1)
    if probs.size(1) < T:
        raise ValueError('probs shorter than tokens in sequence dimension.')

    if T < 2:
        return torch.zeros(0, device=probs.device, dtype=probs.dtype)

    device = probs.device
    idx = torch.arange(T - 1, device=device)
    current = tokens[0, :-1]
    right = tokens[0, 1:]

    p_current = probs[0, idx, current]
    p_right = probs[0, idx, right]
    delta1 = p_right - p_current

    delta2 = torch.zeros_like(delta1)
    if T > 2:
        idx_next = torch.arange(1, T - 1, device=device)
        next_tokens = tokens[0, 2:]
        p_next = probs[0, idx_next, next_tokens]
        p_right_next = probs[0, idx_next, right[1:]]
        delta2[:-1] = p_next - p_right_next

    margin_tensor = torch.as_tensor(margin, device=device, dtype=probs.dtype)
    lam_tensor = torch.as_tensor(lam, device=device, dtype=probs.dtype)

    delta1 = torch.clamp(delta1 - margin_tensor, min=0.0)
    delta2 = torch.clamp(delta2 - margin_tensor, min=0.0)
    return delta1 + lam_tensor * delta2


def build_cooldown_mask(length: int, edits: Sequence[int], distance: int, device: torch.device) -> torch.Tensor:
    """Return boolean mask of indices to suppress due to cooldown."""
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if distance <= 0 or not edits:
        return mask

    for idx in edits:
        lo = max(0, idx - distance)
        hi = min(length, idx + distance + 1)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def apply_insertions(
    x: torch.Tensor,
    gap_indices: Sequence[int],
    *,
    max_token_pos: int,
    block_size: int,
    fill_id: int,
) -> Tuple[int, List[int]]:
    """Insert fill tokens at the requested gap indices from right to left."""
    applied: List[int] = []
    for g in sorted(gap_indices, reverse=True):
        if g < 0:
            continue
        start = min(g, block_size - 1)
        end = min(block_size, max_token_pos)
        if end - start <= 0:
            continue
        right_shift_(x, start=start, end=end, fill_id=fill_id)
        applied.append(start)
        max_token_pos = min(block_size, max_token_pos + 1)
    applied.reverse()
    return max_token_pos, applied


def apply_deletions(
    x: torch.Tensor,
    position_indices: Sequence[int],
    *,
    max_token_pos: int,
    initial_length: int,
    fill_id: int,
    prior_insertions: Sequence[int] | None = None,
) -> Tuple[int, List[int]]:
    """Delete characters at the requested indices from left to right."""

    applied: List[int] = []
    offset = 0
    insertions_sorted: List[int] = sorted(prior_insertions or [])
    ins_ptr = 0
    for idx in sorted(position_indices):
        while ins_ptr < len(insertions_sorted) and insertions_sorted[ins_ptr] <= idx:
            ins_ptr += 1
        adjustment = ins_ptr
        adj_idx = idx + adjustment - offset
        if adj_idx < initial_length or adj_idx >= max_token_pos:
            continue
        left_shift_(x, start=adj_idx, end=max_token_pos, fill_id=fill_id)
        applied.append(adj_idx)
        max_token_pos = max(initial_length, max_token_pos - 1)
        offset += 1
    return max_token_pos, applied


__all__ = [
    'apply_deletions',
    'apply_insertions',
    'apply_re_noise',
    'build_cooldown_mask',
    'compute_noise_ratio',
    'deletion_scores_from_probs',
    'left_shift_',
    'right_shift_',
    'score_gaps_for_insertion',
    'select_topk_mask',
    'uncertainty_from_logprobs',
]
