"""
Per-sample loss processing utilities for diffusion training.
"""

import torch
import math
from .entropy_utils import calculate_wrong_answer_entropy_per_sample, get_current_entropy_penalty

def calculate_per_sample_losses(logits, targets, mask):
    """
    Calculate cross-entropy loss per sample without aggregation.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) or (batch_size, seq_len, vocab_size)
        mask: (batch_size, seq_len) - boolean mask

    Returns:
        per_sample_losses: (batch_size,) - average loss per sample
        per_sample_mask_counts: (batch_size,) - number of masked positions per sample
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape tensors
    logits_reshaped = logits.view(-1, vocab_size)
    mask_reshaped = mask.view(-1)

    if targets.dim() == 3:
        # Soft targets
        targets_reshaped = targets.view(-1, vocab_size)
        position_losses = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='none'
        )
    else:
        # Hard targets
        targets_reshaped = targets.view(-1)
        position_losses = torch.nn.functional.cross_entropy(
            logits_reshaped[mask_reshaped],
            targets_reshaped[mask_reshaped],
            reduction='none'
        )

    # Map masked positions back to sample indices
    mask_sample_indices = torch.arange(batch_size, device=mask.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)[mask_reshaped]

    # Aggregate losses per sample
    per_sample_losses = torch.zeros(batch_size, device=logits.device)
    per_sample_mask_counts = torch.zeros(batch_size, device=logits.device)

    per_sample_losses.scatter_add_(0, mask_sample_indices, position_losses)
    per_sample_mask_counts.scatter_add_(0, mask_sample_indices, torch.ones_like(position_losses))

    # Average loss per sample (replicates reduction='mean' behavior per sample)
    per_sample_losses = per_sample_losses / (per_sample_mask_counts + 1e-8)

    return per_sample_losses, per_sample_mask_counts

def apply_per_sample_modifications(per_sample_losses, logits, targets, mask, training_ctx, iter_num, wrongness_factor=None):
    """
    Apply per-sample modifications: entropy penalty and wrongness factor scaling.

    Args:
        per_sample_losses: (batch_size,) - base losses per sample
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len) or (batch_size, seq_len, vocab_size)
        mask: (batch_size, seq_len)
        training_ctx: TrainingContext
        iter_num: current iteration
        wrongness_factor: (batch_size,) - external wrongness scaling factor, default=None

    Returns:
        modified_losses: (batch_size,) - modified losses per sample
    """
    batch_size = per_sample_losses.shape[0]
    modified_losses = per_sample_losses.clone()

    # 1. Apply entropy penalty per sample (reuses existing entropy calculation)
    if training_ctx.enable_entropy_penalty:
        current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)

        if current_entropy_penalty > 0:
            # Calculate per-sample entropy penalties using existing function
            per_sample_entropies = calculate_wrong_answer_entropy_per_sample(
                logits, targets, mask, training_ctx.extended_vocab_size
            )

            # Calculate entropy multipliers (same logic as original)
            max_wrong_entropy = math.log(training_ctx.extended_vocab_size - 1)
            entropy_penalty_factors = (max_wrong_entropy - per_sample_entropies) / max_wrong_entropy
            entropy_multipliers = 1.0 + current_entropy_penalty * entropy_penalty_factors

            modified_losses = modified_losses * entropy_multipliers

    # 2. Apply wrongness factor scaling
    if wrongness_factor is not None:
        # Ensure wrongness_factor is the right shape and on the right device
        if wrongness_factor.shape[0] != batch_size:
            raise ValueError(f"wrongness_factor shape {wrongness_factor.shape} doesn't match batch_size {batch_size}")

        wrongness_factor = wrongness_factor.to(modified_losses.device)
        modified_losses = modified_losses * wrongness_factor

    return modified_losses
