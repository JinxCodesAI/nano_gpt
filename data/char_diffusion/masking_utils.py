"""
Masking utilities for diffusion training based on reference implementation.
Contains masking functions for sticky, span, and random masking strategies.
"""
import torch
import math
from typing import Dict, Any


def apply_bert_style_corruption_cpu(x: torch.Tensor, mask: torch.Tensor, 
                                  mask_token_id: int, vocab_size: int, rng) -> torch.Tensor:
    """
    Optimized BERT-style corruption using tensor operations.
    Applies 80/10/10 rule: 80% [MASK], 10% random token, 10% unchanged.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions to corrupt (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token
        vocab_size: Size of vocabulary for random token generation
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
        random_tokens = torch.randint(
            0, vocab_size, (num_random,), generator=rng, device=mask.device
        )
        corrupted_x[random_positions] = random_tokens
    
    return corrupted_x


def apply_random_masking_cpu(x: torch.Tensor, mask_probability: float, 
                           mask_token_id: int, vocab_size: int, rng=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized random masking for BERT-style training using tensor operations.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        mask_probability: Probability of masking each token (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token
        vocab_size: Size of vocabulary for random token generation
        rng: Random number generator
        
    Returns:
        corrupted_x: Input with masking applied
        mask: Boolean mask indicating which positions were selected for prediction
    """
    # Generate random mask using provided RNG
    mask_tensor = torch.rand(x.shape, generator=rng, device=x.device)
    mask = mask_tensor < mask_probability
    
    # Apply BERT-style corruption
    corrupted_x = apply_bert_style_corruption_cpu(x, mask, mask_token_id, vocab_size, rng)
    
    return corrupted_x, mask


def apply_target_driven_sticky_masking_cpu(x: torch.Tensor, target_masked_ratio: float, 
                                         p1_probability: float, p2_probability: float, 
                                         mask_token_id: int, vocab_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-optimized target-driven sticky masking for unmasking training.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        target_masked_ratio: Target fraction of tokens to mask (0.0 to 1.0)
        p1_probability: Probability of masking when no neighbors are masked
        p2_probability: Probability of masking when neighbors are masked
        mask_token_id: Token ID to use for masking
        vocab_size: Size of vocabulary for random token generation
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
    neighbor_masked = torch.zeros_like(x, dtype=torch.bool, device=device)
    current_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_tensor = torch.full((batch_size,), target_masked_count, dtype=torch.long, device=device)

    max_rounds = min(1000, target_masked_count * 10)

    for _ in range(max_rounds):
        sequences_need_more = current_counts < target_tensor
        if not sequences_need_more.any():
            break

        neighbor_masked.zero_()
        neighbor_masked[:, 1:] |= current_mask[:, :-1]
        neighbor_masked[:, :-1] |= current_mask[:, 1:]

        rand_vals = torch.rand(x.shape, generator=rng, device=device)
        mask_probs = torch.where(neighbor_masked, p2_probability, p1_probability)
        new_masks = (rand_vals < mask_probs) & (~current_mask)
        new_masks &= sequences_need_more.unsqueeze(1)

        if not new_masks.any():
            break

        additions = new_masks.sum(dim=1)
        current_mask |= new_masks
        current_counts += additions

    final_counts = current_counts.clone()
    exceeded_sequences = torch.where(final_counts > target_tensor)[0]

    if exceeded_sequences.numel() > 0:
        for batch_idx in exceeded_sequences.tolist():
            excess = (final_counts[batch_idx] - target_tensor[batch_idx]).item()
            if excess <= 0:
                continue
            seq_mask = current_mask[batch_idx]
            masked_positions = torch.where(seq_mask)[0]
            if masked_positions.numel() == 0:
                continue
            perm_indices = torch.randperm(
                masked_positions.size(0), generator=rng, device=device
            )[:excess]
            positions_to_unmask = masked_positions[perm_indices]
            current_mask[batch_idx, positions_to_unmask] = False

    masked_x = x.clone()
    masked_x[current_mask] = mask_token_id

    corrupted_x = apply_bert_style_corruption_cpu(x, current_mask, mask_token_id, vocab_size, rng)

    return corrupted_x, current_mask


def apply_span_masking_cpu(x: torch.Tensor, spans_count: int, 
                         mask_token_id: int, vocab_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CPU-optimized span masking for unmasking training.
    Masks spans_count continuous areas in the input.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        spans_count: Number of continuous spans to mask
        mask_token_id: Token ID to use for masking
        vocab_size: Size of original vocabulary (not used for span masking)
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


def apply_stage_masking(x: torch.Tensor, stage_config: Dict[str, Any], 
                       mask_token_id: int, vocab_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: Stage configuration dictionary
        mask_token_id: Token ID to use for masking
        vocab_size: Size of original vocabulary (for random token generation, excluding special tokens)
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
        corrupted_x = apply_bert_style_corruption_cpu(x, mask, mask_token_id, vocab_size, rng)
        return corrupted_x, mask
        
    elif stage_type == 'sticky':
        target_masked_ratio = stage_config['target_masked_ratio']
        p1_probability = stage_config['p1_probability'] 
        p2_probability = stage_config['p2_probability']
        
        return apply_target_driven_sticky_masking_cpu(
            x, target_masked_ratio, p1_probability, p2_probability, 
            mask_token_id, vocab_size, rng
        )
        
    elif stage_type == 'span':
        spans_count = stage_config['spans_count']
        return apply_span_masking_cpu(x, spans_count, mask_token_id, vocab_size, rng)
        
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")
