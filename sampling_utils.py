"""Utility helpers for iterative sampling schedules and structural edits."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import torch


def _interp_linear(s: int, S: int, start: float, end: float) -> float:
    """Linearly interpolate between endpoints for a scheduling helper.

    Args:
        s: Current iteration index, expected in ``[0, S)``.
        S: Total number of iterations in the schedule. Values ``<=1`` collapse to ``start``.
        start: Ratio applied at the beginning of the process.
        end: Ratio applied at the final iteration.

    Returns:
        The interpolated ratio for iteration ``s`` under a linear schedule.
    """
    t = 0.0 if S <= 1 else s / (S - 1)
    return start + (end - start) * t


def _interp_cosine(s: int, S: int, start: float, end: float) -> float:
    """Cosine-annealed interpolation for schedule helpers.

    Args:
        s: Current iteration index, expected in ``[0, S)``.
        S: Total number of iterations in the schedule. Values ``<=1`` collapse to ``start``.
        start: Ratio applied at the beginning of the process.
        end: Ratio applied at the final iteration.

    Returns:
        The interpolated ratio for iteration ``s`` using cosine decay.
    """
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
    """Map an iteration to a scalar ratio according to a named schedule.

    Args:
        iteration: Zero-based iteration currently being executed.
        max_iterations: Total number of iterations in the process.
        schedule: Either ``"linear"`` or ``"cosine"``; any other value disables the ratio.
        start: Ratio applied at iteration ``0``.
        end: Ratio applied at iteration ``max_iterations - 1``.

    Returns:
        The interpolated ratio for the provided iteration. Returns ``0.0`` for unknown schedules.
    """
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
    """Randomly overwrite active, non-prompt positions with a filler token.

    Args:
        x: Batch of token ids shaped ``(1, seq_len)`` being edited in-place.
        max_token_pos: Exclusive upper bound of the active window within ``x``.
        initial_length: Number of prompt tokens at the beginning of the sequence.
        replace_character_id: Token id used when re-noising positions.
        ratio: Probability of re-noising any eligible position.
        device: Torch device on which to allocate helper tensors.

    Returns:
        The modified tensor ``x`` with a subset of active positions replaced.
    """
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
    """Quantify token-level uncertainty from model log probabilities.

    Args:
        log_probs: Log-probabilities of shape ``(1, T, V)`` covering the active sequence.
        tokens: Token ids of shape ``(1, T)`` corresponding to the evaluated positions.

    Returns:
        A detached ``(T,)`` tensor where each element is ``1 - p(token)`` in ``[0, 1]``.
    """
    with torch.no_grad():
        T = tokens.size(1)
        idx = torch.arange(T, device=log_probs.device)
        p_self = log_probs[0, idx, tokens[0]].exp()
        return (1.0 - p_self).detach()


def score_gaps_for_insertion(u: torch.Tensor) -> torch.Tensor:
    """Estimate insertion desirability for each inter-token gap.

    Args:
        u: Uncertainty scores per token, shaped ``(T,)``.

    Returns:
        A tensor of length ``T - 1`` whose ``i``-th value reflects how uncertain either
        neighbor of the gap ``i`` is, using the maximum of the two uncertainties.
    """
    if u.numel() < 2:
        return torch.empty(0, device=u.device, dtype=u.dtype)
    return torch.maximum(u[:-1], u[1:])


def right_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """In-place helper to make room for an insertion in ``x``.

    Args:
        x: Sequence batch shaped ``(1, seq_len)`` modified in-place.
        start: Index where the vacant slot should appear.
        end: Exclusive end of the region to shift.
        fill_id: Token id used to populate the newly opened slot at ``start``.
    """
    if end - start <= 0:
        return
    slice_to_move = x[:, start:end - 1].clone()
    x[:, start + 1 : end] = slice_to_move
    x[:, start] = fill_id


def left_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """In-place helper to close a gap after deleting from ``x``.

    Args:
        x: Sequence batch shaped ``(1, seq_len)`` modified in-place.
        start: Inclusive beginning of the window being compacted.
        end: Exclusive end of the active sequence window.
        fill_id: Token id written into the vacated slot at ``end - 1``.
    """
    if end - start <= 0:
        return
    slice_to_move = x[:, start + 1 : end].clone()
    x[:, start:end - 1] = slice_to_move
    x[:, end - 1] = fill_id


def select_topk_mask(
    scores: torch.Tensor,
    k: int,
    *,
    forbid_lo: int = 0,
    forbid_hi: int = 0,
    additional_forbid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the top-``k`` scoring positions while respecting constraints.

    Args:
        scores: One-dimensional tensor of candidate scores.
        k: Number of selections to return.
        forbid_lo: Inclusive lower bound of an index range that must remain unselected.
        forbid_hi: Exclusive upper bound of the forbidden range.
        additional_forbid: Optional boolean tensor masking individual forbidden indices.

    Returns:
        A boolean tensor with ``True`` at the chosen indices and ``False`` elsewhere.
    """

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
    """Rank token positions by how beneficial a deletion appears.

    Args:
        probs: Token probabilities with shape ``(1, T, V)``.
        tokens: Token ids with shape ``(1, T)`` for the active sequence.
        margin: Minimum improvement threshold before a deletion is considered worthwhile.
        lam: Weight assigned to the second-order lookahead term.

    Returns:
        A ``(T - 1,)`` tensor scoring each position (except the last) for deletion.
    """
    T = tokens.size(1)
    if T < 2:
        return torch.zeros(1, device=tokens.device, dtype=probs.dtype)

    c = tokens[0]
    r = tokens[0, 1:]
    p_self = probs[0, torch.arange(T, device=probs.device), c]
    p_here_right = probs[0, torch.arange(T - 1, device=probs.device), r]
    d1 = p_here_right - p_self[:-1]

    if T >= 3:
        n = tokens[0, 2:]
        p_next_next = probs[0, 1:-1, n]
        p_next_right = probs[0, 1:-1, r[:-1]]
        d2_core = p_next_next - p_next_right
        d2 = torch.cat([d2_core, torch.zeros(1, device=tokens.device, dtype=probs.dtype)], dim=0)
    else:
        d2 = torch.zeros(T - 1, device=tokens.device, dtype=probs.dtype)

    d1 = torch.clamp_min(d1 - margin, 0.0)
    d2 = torch.clamp_min(d2 - margin, 0.0)
    return d1 + lam * d2


def build_cooldown_mask(length: int, edits: Sequence[int], distance: int, device: torch.device) -> torch.Tensor:
    """Mark indices that should be skipped due to recent edits.

    Args:
        length: Number of positions in the target mask.
        edits: Sequence of indices that were just edited.
        distance: Radius around each edit to mark as unavailable.
        device: Torch device for the returned mask.

    Returns:
        A boolean tensor where ``True`` denotes positions that should be suppressed.
    """
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
    """Insert filler tokens into the sequence at specified gaps.

    Args:
        x: Sequence batch shaped ``(1, seq_len)`` modified in-place.
        gap_indices: Indices of the gaps (between tokens) to populate.
        max_token_pos: Exclusive upper bound of the active region before insertion.
        block_size: Maximum number of tokens allowed in ``x``.
        fill_id: Token id written into the new slots.

    Returns:
        A tuple ``(new_max_token_pos, applied_indices)`` with the updated active length and
        the concrete token indices where insertions occurred (sorted ascending).
    """
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
    """Remove tokens from the sequence at specified indices.

    Args:
        x: Sequence batch shaped ``(1, seq_len)`` modified in-place.
        position_indices: Positions (relative to pre-insertion layout) proposed for deletion.
        max_token_pos: Exclusive upper bound of the active region before deletion.
        initial_length: Number of prompt tokens that must remain untouched.
        fill_id: Token id placed in vacated trailing slots after compaction.
        prior_insertions: Indices where insertions were applied this iteration, used to align
            deletion coordinates with the shifted sequence.

    Returns:
        A tuple ``(new_max_token_pos, applied_indices)`` describing the updated active length and
        the actual token indices that were deleted (sorted ascending).
    """

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
