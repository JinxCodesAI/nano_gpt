"""
Masking utilities for diffusion training based on reference implementation.
Contains masking functions for sticky, span, and random masking strategies.
"""
import torch
import math
from typing import Dict, Any


def _candidate_tensor(random_token_ids, device: torch.device) -> torch.Tensor:
    """Materialize the candidate token tensor on the requested device."""
    if isinstance(random_token_ids, torch.Tensor):
        return random_token_ids.to(device)
    return torch.tensor(list(random_token_ids), dtype=torch.long, device=device)


def apply_bert_style_corruption_cpu(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
    random_token_ids,
    rng,
) -> torch.Tensor:
    """
    Optimized BERT-style corruption using tensor operations.
    Applies 80/10/10 rule: 80% [MASK], 10% random token, 10% unchanged.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions to corrupt (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token
        random_token_ids: Iterable of token IDs eligible for random replacement
        rng: Torch random number generator for consistent randomization
        
    Returns:
        corrupted_x: Input with BERT-style corruption applied
    """
    corrupted_x = x.clone()
    
    # Generate random values for all masked positions at once
    rand_vals = torch.rand(mask.shape, generator=rng, device=mask.device)
    
    # Create masks for the three corruption types
    mask_positions = mask & (rand_vals < 0.8)  # 80%: [MASK] token
    random_positions = mask & (rand_vals >= 0.8) & (rand_vals < 0.9)  # 10%: random token
    # 10%: unchanged (no mask needed)
    
    # Apply [MASK] tokens
    corrupted_x[mask_positions] = mask_token_id
    
    # Apply random tokens
    if random_positions.any():
        num_random = random_positions.sum().item()
        candidates = _candidate_tensor(random_token_ids, mask.device)
        if candidates.numel() == 0:
            raise ValueError("random_token_ids must contain at least one token")
        indices = torch.randint(
            0, candidates.numel(), (num_random,), generator=rng, device=mask.device
        )
        random_tokens = candidates.index_select(0, indices)
        corrupted_x[random_positions] = random_tokens
    
    return corrupted_x


def apply_random_masking_cpu(
    x: torch.Tensor,
    mask_probability: float,
    mask_token_id: int,
    random_token_ids,
    rng=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized random masking for BERT-style training using tensor operations.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        mask_probability: Probability of masking each token (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token
        random_token_ids: Iterable of token IDs eligible for random replacement
        rng: Random number generator
        
    Returns:
        corrupted_x: Input with masking applied
        mask: Boolean mask indicating which positions were selected for prediction
    """
    # Generate random mask using provided RNG
    mask_tensor = torch.rand(x.shape, generator=rng, device=x.device)
    mask = mask_tensor < mask_probability
    
    # Apply BERT-style corruption
    corrupted_x = apply_bert_style_corruption_cpu(
        x, mask, mask_token_id, random_token_ids, rng
    )
    
    return corrupted_x, mask


def apply_target_driven_sticky_masking_cpu(
    x: torch.Tensor,
    target_masked_ratio: float,
    p1_probability: float,
    p2_probability: float,
    mask_token_id: int,
    random_token_ids,
    rng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-optimized target-driven sticky masking for unmasking training.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        target_masked_ratio: Target fraction of tokens to mask (0.0 to 1.0)
        p1_probability: Probability of masking when no neighbors are masked
        p2_probability: Probability of masking when neighbors are masked
        mask_token_id: Token ID to use for masking
        random_token_ids: Iterable of token IDs eligible for random replacement
        rng: Random number generator
        
    Returns:
        corrupted_x: Input with masked tokens replaced using BERT-style corruption
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate target number of masked tokens per sequence
    target_masked_count = max(1, math.ceil(target_masked_ratio * seq_len)) if target_masked_ratio > 0 else 0
    
    if target_masked_count == 0:
        # No masking needed - return early
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    current_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    current_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_tensor = torch.full((batch_size,), target_masked_count, dtype=torch.long, device=device)

    seq_len = x.shape[1]
    max_batch_add = max(1, seq_len // 8)

    while True:
        need = target_tensor - current_counts
        active_idx = torch.nonzero(need > 0, as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            break

        sub_mask = current_mask.index_select(0, active_idx)
        need_sub = need.index_select(0, active_idx)

        neighbor_masked = torch.zeros_like(sub_mask)
        neighbor_masked[:, 1:] |= sub_mask[:, :-1]
        neighbor_masked[:, :-1] |= sub_mask[:, 1:]

        weights = torch.where(neighbor_masked, p2_probability, p1_probability)
        weights = weights * (~sub_mask).float()

        available_counts = (weights > 0).sum(dim=1)
        need_sub = torch.minimum(need_sub, available_counts)
        chunk = torch.minimum(need_sub, torch.full_like(need_sub, max_batch_add))

        max_add = int(chunk.max().item())
        if max_add == 0:
            break

        weights = torch.where(
            weights > 0, weights, torch.full_like(weights, 1e-6)
        )
        scores = -torch.log(torch.rand(weights.shape, device=device, generator=rng)) / weights
        scores = scores.masked_fill(sub_mask, float("inf"))

        topk_vals, topk_idx = torch.topk(scores, max_add, dim=1, largest=False)
        row_idx = torch.arange(active_idx.size(0), device=device).unsqueeze(1).expand(-1, max_add)
        selection_mask = torch.arange(max_add, device=device).unsqueeze(0) < chunk.unsqueeze(1)

        if selection_mask.any():
            chosen_rows = row_idx[selection_mask]
            chosen_cols = topk_idx[selection_mask]
            sub_mask[chosen_rows, chosen_cols] = True

        current_mask.index_copy_(0, active_idx, sub_mask)
        current_counts[active_idx] += chunk

    remaining = target_tensor - current_counts
    remaining_idx = torch.nonzero(remaining > 0, as_tuple=False).squeeze(1)
    if remaining_idx.numel() > 0:
        sub_mask = current_mask.index_select(0, remaining_idx)
        remaining_sub = remaining.index_select(0, remaining_idx)
        available = (~sub_mask)
        available_counts = available.sum(dim=1)
        remaining_sub = torch.minimum(remaining_sub, available_counts)
        max_add = int(remaining_sub.max().item())
        if max_add > 0:
            scores = torch.rand(sub_mask.shape, device=device, generator=rng)
            scores = scores.masked_fill(~available, float("inf"))
            topk_val, topk_idx = torch.topk(scores, max_add, dim=1, largest=False)
            row_idx = torch.arange(remaining_idx.size(0), device=device).unsqueeze(1).expand(-1, max_add)
            selection_mask = torch.arange(max_add, device=device).unsqueeze(0) < remaining_sub.unsqueeze(1)
            if selection_mask.any():
                chosen_rows = row_idx[selection_mask]
                chosen_cols = topk_idx[selection_mask]
                sub_mask[chosen_rows, chosen_cols] = True
            current_mask.index_copy_(0, remaining_idx, sub_mask)
            current_counts[remaining_idx] += remaining_sub

    corrupted_x = apply_bert_style_corruption_cpu(
        x, current_mask, mask_token_id, random_token_ids, rng
    )

    return corrupted_x, current_mask


def apply_span_masking_cpu(
    x: torch.Tensor, spans_count: int, mask_token_id: int, rng
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-optimized span masking for unmasking training.
    Masks spans_count continuous areas in the input.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        spans_count: Number of continuous spans to mask
        mask_token_id: Token ID to use for masking
        rng: Random number generator
        
    Returns:
        masked_x: Input with masked spans directly replaced with mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    if spans_count <= 0 or seq_len <= 1:
        # No masking needed - return original
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Generate all random positions at once: (batch_size, 2*spans_count)
    random_positions = torch.rand(batch_size, 2 * spans_count, device=device, generator=rng)
    
    # Scale by sequence length and round down
    scaled_positions = (random_positions * seq_len).long()
    
    # Sort the positions to get proper start/end pairs
    sorted_positions, _ = torch.sort(scaled_positions, dim=1)
    
    # Split into start and end positions: (batch_size, spans_count)
    start_positions = sorted_positions[:, 0::2]  # indices 0, 2, 4, ...
    end_positions = sorted_positions[:, 1::2]    # indices 1, 3, 5, ...
    
    # Create position indices for all positions in sequence: (seq_len,)
    position_indices = torch.arange(seq_len, device=device)
    
    # Expand dimensions for broadcasting: 
    # position_indices: (1, 1, seq_len)
    # start_positions: (batch_size, spans_count, 1) 
    # end_positions: (batch_size, spans_count, 1)
    position_indices = position_indices.unsqueeze(0).unsqueeze(0)
    start_positions = start_positions.unsqueeze(2)
    end_positions = end_positions.unsqueeze(2)
    
    # Create mask for each span: (batch_size, spans_count, seq_len)
    span_masks = (position_indices >= start_positions) & (position_indices <= end_positions)
    
    # Combine all spans for each batch: (batch_size, seq_len)
    mask = span_masks.any(dim=1)
    
    # Apply masking directly with mask token (no BERT-style corruption for spans)
    masked_x = x.clone()
    masked_x[mask] = mask_token_id
    
    return masked_x, mask


def apply_stage_masking(
    x: torch.Tensor,
    stage_config: Dict[str, Any],
    mask_token_id: int,
    random_token_ids,
    rng,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: Stage configuration dictionary
        mask_token_id: Token ID to use for masking
        random_token_ids: Iterable of token IDs eligible for random replacement
        rng: Random number generator
        
    Returns:
        masked_x: Input with masked tokens
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config['type']
    
    if stage_type == 'random':
        max_masked_ratio = stage_config['max_masked_ratio']
        # For random masking, each sample gets a different random ratio up to max
        batch_size, seq_len = x.shape
        mask_ratios = torch.rand(batch_size, generator=rng, dtype=torch.float, device=x.device) * max_masked_ratio
        thresholds = mask_ratios.unsqueeze(1)
        rand_vals = torch.rand((batch_size, seq_len), generator=rng, dtype=torch.float, device=x.device)
        mask = rand_vals < thresholds
        corrupted_x = apply_bert_style_corruption_cpu(
            x, mask, mask_token_id, random_token_ids, rng
        )
        return corrupted_x, mask
        
    elif stage_type == 'sticky':
        target_masked_ratio = stage_config['target_masked_ratio']
        p1_probability = stage_config['p1_probability'] 
        p2_probability = stage_config['p2_probability']
        
        return apply_target_driven_sticky_masking_cpu(
            x,
            target_masked_ratio,
            p1_probability,
            p2_probability,
            mask_token_id,
            random_token_ids,
            rng,
        )
        
    elif stage_type == 'span':
        spans_count = stage_config['spans_count']
        return apply_span_masking_cpu(x, spans_count, mask_token_id, rng)
        
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")
