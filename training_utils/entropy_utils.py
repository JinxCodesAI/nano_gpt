"""
Entropy penalty and label smoothing utilities for diffusion training.
Contains functions for entropy-based regularization and target smoothing.
"""

import torch

from .training_config import TrainingContext


def calculate_wrong_answer_entropy(logits, targets, vocab_size):
    """
    Calculate entropy of wrong answer distributions for entropy penalty.
    
    HIGH entropy (uniform wrong answers) = GOOD (high signal-to-noise ratio)
    LOW entropy (concentrated wrong answers) = BAD (low signal-to-noise ratio)
    
    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        targets: Target tokens (batch_size, seq_len)
        vocab_size: Size of vocabulary
        
    Returns:
        avg_entropy: Average entropy of wrong answer distributions across all positions
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device
    epsilon = 1e-9 # Use a slightly larger epsilon for stability
    
    # Get probabilities from logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Flatten for easier processing
    probs_flat = probs.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    # --- START FIX ---
    
    # Create a mask to zero out the correct answer probabilities
    wrong_probs = probs_flat.clone()
    wrong_probs[range(len(targets_flat)), targets_flat] = 0.0
    
    # Calculate the sum of the remaining "wrong" probabilities
    # This sum is (1.0 - p_correct)
    sum_wrong_probs = wrong_probs.sum(dim=1, keepdim=True)
    
    # Avoid division by zero for positions where p_correct was close to 1.0
    # If sum_wrong_probs is near zero, entropy is also zero, so we can ignore these.
    # We create a mask for safe normalization.
    safe_mask = sum_wrong_probs.squeeze() > epsilon
    
    if not safe_mask.any():
        # Handle the edge case where no positions have significant wrong probabilities
        return torch.tensor(0.0, device=device)
        
    # Re-normalize the wrong probabilities so they sum to 1
    # This creates a true probability distribution over the incorrect tokens
    normalized_wrong_probs = wrong_probs[safe_mask] / sum_wrong_probs[safe_mask]
    
    # Calculate entropy on the properly normalized distribution
    log_probs = torch.log(normalized_wrong_probs + epsilon)
    entropies = -(normalized_wrong_probs * log_probs).sum(dim=1)
    
    # --- END FIX ---
    
    # Return average entropy across all valid positions
    return entropies.mean()


def get_current_entropy_penalty(iter_num, ctx: TrainingContext):
    """
    Calculate current entropy penalty based on iteration number.
    
    Args:
        iter_num: Current iteration number
        ctx: Training context with penalty parameters
        
    Returns:
        current_penalty: Current entropy penalty multiplier (0 to max_entropy_penalty)
    """
    if not ctx.enable_entropy_penalty:
        return 0.0
    
    if iter_num < ctx.entropy_penalty_start_iter:
        return 0.0
    
    if iter_num >= ctx.max_iters:
        return ctx.max_entropy_penalty
    
    # Linear increase from start_iter to max_iters
    progress = (iter_num - ctx.entropy_penalty_start_iter) / (ctx.max_iters - ctx.entropy_penalty_start_iter)
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
    
    return progress * ctx.max_entropy_penalty


def calculate_wrong_answer_entropy_per_sample(logits, targets, mask, vocab_size):
    """
    Calculate entropy of wrong answer distributions per sample (reuses existing logic).

    Args:
        logits: Model logits (batch_size, seq_len, vocab_size)
        targets: Target tokens (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len) - only calculate for masked positions
        vocab_size: Size of vocabulary

    Returns:
        per_sample_entropies: (batch_size,) - entropy per sample
    """
    batch_size, seq_len, vocab_size_logits = logits.shape
    device = logits.device
    per_sample_entropies = torch.zeros(batch_size, device=device)

    for sample_idx in range(batch_size):
        sample_mask = mask[sample_idx]  # (seq_len,)
        if not sample_mask.any():
            continue

        # Extract masked positions for this sample
        sample_logits = logits[sample_idx:sample_idx+1, :, :]  # Keep batch dim: (1, seq_len, vocab_size)
        sample_targets = targets[sample_idx:sample_idx+1, :]   # Keep batch dim: (1, seq_len)
        sample_mask_expanded = sample_mask.unsqueeze(0)        # (1, seq_len)

        # Apply mask to get only masked positions
        masked_logits = sample_logits[:, sample_mask, :]       # (1, num_masked_positions, vocab_size)
        masked_targets = sample_targets[:, sample_mask]        # (1, num_masked_positions)

        # Reuse existing entropy calculation logic
        if masked_logits.numel() > 0:
            # Reshape to match existing function expectations
            reshaped_logits = masked_logits.view(1, -1, vocab_size_logits)
            reshaped_targets = masked_targets.view(1, -1)

            # Call existing function (it will return scalar for this single sample)
            sample_entropy = calculate_wrong_answer_entropy(reshaped_logits, reshaped_targets, vocab_size)
            per_sample_entropies[sample_idx] = sample_entropy

    return per_sample_entropies


def update_entropy_multiplier_ema(ctx: TrainingContext, current_multiplier: float):
    """
    Update the exponential moving average of entropy multiplier.

    Args:
        ctx: Training context
        current_multiplier: Current entropy multiplier value
    """
    if ctx.enable_entropy_penalty:
        # EMA update: ema = alpha * ema + (1-alpha) * current_value
        alpha = ctx.entropy_multiplier_ema_factor
        ctx.entropy_multiplier_ema = alpha * ctx.entropy_multiplier_ema + (1 - alpha) * current_multiplier


def apply_label_smoothing(targets, uncertainty_factor, vocab_size, special_token_ids=None, device=None):
    """
    Apply label smoothing to target tokens.
    
    Args:
        targets: Target token IDs (batch_size, seq_len)
        uncertainty_factor: Label smoothing factor (0.0 = no smoothing, >0 = apply smoothing)
        vocab_size: Size of vocabulary
        special_token_ids: List of special token IDs to exclude from smoothing (optional)
        device: Device to create tensors on
        
    Returns:
        smoothed_targets: Probability distribution targets (batch_size, seq_len, vocab_size)
    """
    if uncertainty_factor <= 0.0:
        # No smoothing, return one-hot encoded targets
        return torch.nn.functional.one_hot(targets, num_classes=vocab_size).float()
    
    if device is None:
        device = targets.device
    
    batch_size, seq_len = targets.shape
    
    # Create smoothed probability distribution
    smoothed_targets = torch.zeros(batch_size, seq_len, vocab_size, device=device)
    
    # Set correct answer probability to (1 - uncertainty_factor)
    correct_prob = 1.0 - uncertainty_factor
    smoothed_targets.scatter_(2, targets.unsqueeze(-1), correct_prob)
    
    # Calculate incorrect answer probability: u / (vocab_size - 1)
    # But we need to exclude special tokens from getting smoothed probability
    if special_token_ids is None:
        special_token_ids = []
        
    incorrect_prob = uncertainty_factor / (vocab_size - len(special_token_ids))
    
    # Add smoothing probability to all positions except the correct answer
    smoothed_targets += incorrect_prob
    
    # Remove the extra probability that was added to correct answers
    smoothed_targets.scatter_(2, targets.unsqueeze(-1), correct_prob)
    
    # Handle special tokens - set their probability to 0 (except when they are the correct answer)
    if special_token_ids is not None:
        for special_id in special_token_ids:
            if special_id < vocab_size:
                # Create mask for positions where special_id is NOT the correct answer
                not_correct_mask = (targets != special_id).unsqueeze(-1)
                # Zero out probability for this special token where it's not correct
                special_mask = torch.zeros(batch_size, seq_len, vocab_size, device=device)
                special_mask[:, :, special_id] = 1.0
                smoothed_targets = smoothed_targets * (1 - special_mask * not_correct_mask.float())
    
    sum_probs = smoothed_targets.sum(dim=-1, keepdim=True)
    # Renormalize to ensure probabilities sum to 1
    smoothed_targets = smoothed_targets / sum_probs
    
    return smoothed_targets