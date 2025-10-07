"""
Utility functions for diffusion-based text generation using iterative demasking
"""
import math
import torch
import torch.nn.functional as F
from collections import Counter
from contextlib import nullcontext


def linear_remasking_schedule(iteration, total_iterations, start_ratio, end_ratio):
    """
    Generate linear remasking schedule

    Args:
        iteration: Current iteration (0-based)
        total_iterations: Total number of iterations
        start_ratio: Starting masking ratio
        end_ratio: Ending masking ratio

    Returns:
        float: Masking ratio for this iteration
    """
    if total_iterations <= 1:
        return end_ratio

    progress = iteration / (total_iterations - 1)
    return start_ratio - progress * (start_ratio - end_ratio)


def nucleus_sample(logits, top_p=1.0, temperature=1.0):
    """
    Apply nucleus (top-p) sampling to logits

    Args:
        logits: Logits tensor (..., vocab_size)
        top_p: Nucleus parameter (1.0 = disabled)
        temperature: Temperature for scaling

    Returns:
        Sampled indices
    """
    if temperature != 1.0:
        logits = logits / temperature

    if top_p < 1.0:
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to keep (cumulative prob <= top_p)
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set logits of tokens to remove to -inf
        logits_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(logits_to_remove, -float('inf'))

    # Sample from the distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def predict_and_sample_tokens(model, tokens, mask_token_id, temperature=1.0,
top_p=1.0, vocab_size=None,
                            device='cuda', debug_logging_fn=None, itos=None, stoi=None,
                            verbose=False, log_debug=False, return_logits=False,
                            pad_token_id=None, base_vocab_size=None, no_grad=True):
    """
    Predict and sample new tokens for masked positions

    Args:
        model: The language model
        tokens: Current token sequence (batch_size, seq_len)
        mask_token_id: ID of the mask token
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        vocab_size: Vocabulary size
        device: Device to run on
        debug_logging_fn: Optional debug logging function
        itos: Index to string mapping
        stoi: String to index mapping
        verbose: Enable verbose logging
        log_debug: Enable debug logging
        return_logits: If True, return logits along with tokens
        pad_token_id: ID of the pad token to exclude from sampling
        base_vocab_size: Base vocabulary size (exclude special tokens beyond this)
        no_grad: If True (default), run forward pass without gradients. Set to False for GRPO training.

    Returns:
        tuple: (updated_tokens, prediction_tokens) or (updated_tokens, prediction_tokens, logits) if return_logits=True
    """
    batch_size, seq_len = tokens.shape


    # Find masked positions
    mask_positions = (tokens == mask_token_id)

    if not mask_positions.any():
        if return_logits:
            # Need to return logits even if no masks
            dummy_targets = torch.zeros_like(tokens)
            grad_context = torch.no_grad() if no_grad else nullcontext()
            with grad_context:
                logits, _ = model(tokens, targets=dummy_targets)
            return tokens, logits
        else:
            return tokens

    # Forward pass through the model
    # Pass dummy targets to get logits for all positions (not just the last one)
    dummy_targets = torch.zeros_like(tokens)

    # Conditionally use no_grad context
    grad_context = torch.no_grad() if no_grad else nullcontext()
    with grad_context:
        logits, _ = model(tokens, targets=dummy_targets)


    # Extract logits for masked positions only
    prediction_tokens = tokens.clone()

    for batch_idx in range(batch_size):
        batch_mask_positions = mask_positions[batch_idx]
        if not batch_mask_positions.any():
            continue

        # Get mask indices for this batch
        mask_indices = torch.nonzero(batch_mask_positions).squeeze(-1)

        # Extract logits for masked positions
        masked_logits = logits[batch_idx, mask_indices, :]  # (num_masked, vocab_size)

        # Exclude special tokens from sampling - only sample from base vocabulary
        if vocab_size is not None:
            # Set mask token logit to -inf so it's never sampled
            masked_logits[:, mask_token_id] = float('-inf')

            # Set pad token logit to -inf if it exists
            if pad_token_id is not None:
                masked_logits[:, pad_token_id] = float('-inf')

            # If we have base_vocab_size, only allow sampling from base vocabulary
            if base_vocab_size is not None:
                # Set all special tokens (beyond base vocab) to -inf
                if masked_logits.shape[-1] > base_vocab_size:
                    masked_logits[:, base_vocab_size:] = float('-inf')

            # Ensure we don't access beyond vocabulary
            if masked_logits.shape[-1] > vocab_size:
                masked_logits = masked_logits[:, :vocab_size]

        # Sample new tokens
        new_tokens = nucleus_sample(masked_logits, top_p=top_p, temperature=temperature)

        # Debug logging
        if debug_logging_fn and batch_idx == 0:
            debug_logging_fn(
                sample_idx=batch_idx,
                logits=logits[batch_idx],
                mask_indices=mask_indices,
                masked_logits=masked_logits,
                new_tokens=new_tokens,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size or masked_logits.shape[-1],
                itos=itos,
                stoi=stoi,
                log_debug=log_debug
            )

        # Update prediction tokens
        prediction_tokens[batch_idx, mask_indices] = new_tokens

    if return_logits:
        return prediction_tokens, logits
    else:
        return prediction_tokens


def calculate_selfconfidence_ratio(model, tokens, mask_token_id, device='cuda', ctx=None):
    """
    Calculate self-confidence score for generated tokens

    Args:
        model: The language model
        tokens: Generated token sequences (batch_size, seq_len)
        mask_token_id: ID of the mask token
        device: Device to run on
        ctx: Context manager for autocast

    Returns:
        List of confidence scores (log probabilities)
    """
    batch_size, seq_len = tokens.shape
    confidence_scores = []

    # The mask token is part of vocabulary, so we can pass it directly
    # Pass dummy targets to get logits for all positions (not just the last one)
    dummy_targets = torch.zeros_like(tokens)
    with torch.no_grad():
        if ctx is not None:
            with ctx:
                logits, _ = model(tokens, targets=dummy_targets)
        else:
            logits, _ = model(tokens, targets=dummy_targets)

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    for batch_idx in range(batch_size):
        batch_tokens = tokens[batch_idx]
        batch_probs = probs[batch_idx]

        # Calculate average log probability of generated tokens (excluding masks)
        valid_positions = (batch_tokens != mask_token_id)

        if not valid_positions.any():
            confidence_scores.append(float('-inf'))
            continue

        # Get valid tokens and their positions
        valid_tokens = batch_tokens[valid_positions]
        valid_probs = batch_probs[valid_positions]

        # Calculate probability of each valid token
        token_probs = []
        for i, token in enumerate(valid_tokens):
            if token < valid_probs.shape[-1]:  # Ensure token is within vocab range
                prob = valid_probs[i, token]
                token_probs.append(prob)

        if len(token_probs) == 0:
            confidence_scores.append(float('-inf'))
            continue

        # Calculate average log probability
        avg_log_prob = torch.log(torch.stack(token_probs) + 1e-10).mean().item()
        confidence_scores.append(avg_log_prob)

    return confidence_scores


def calculate_judge_scores(judge_model, tokens, device='cuda', ctx=None):
    """
    Evaluate generated sequences with a sequence scorer (judge) model in batch.
    Returns torch.Tensor of shape (batch_size,) with scores in [0,1], computed as 1 - evaluation.
    The judge model is expected to output values in [0,1] (ScaledSigmoidHead).
    """
    import torch
    from torch.nn import functional as F  # noqa: F401 (future use)

    judge_model.eval()

    # Optionally prepend CLS token if judge config defines it
    cls_id = getattr(judge_model.config, 'cls_token_id', None)
    idx = tokens
    if cls_id is not None:
        # Prepend CLS and ensure block size
        b, t = tokens.shape
        cls_col = torch.full((b, 1), int(cls_id), dtype=tokens.dtype, device=tokens.device)
        idx = torch.cat([cls_col, tokens], dim=1)
        max_t = getattr(judge_model.config, 'block_size', idx.size(1))
        if idx.size(1) > max_t:
            idx = idx[:, -max_t:]

    with torch.no_grad():
        if ctx is not None:
            with ctx:
                logits, _ = judge_model(idx, targets=None)
        else:
            logits, _ = judge_model(idx, targets=None)

    # logits in [0,1]; convert to score = 1 - evaluation
    evals = logits.clamp(0.0, 1.0)
    scores = 1.0 - evals
    return scores



def apply_remasking(tokens, mask_ratio, mask_token_id, remasking_model=None,
                   randomness_strength=0.5, base_model=None, intelligent_remasking=False,
                   protected_mask=None):
    """
    Apply remasking to tokens based on confidence or random selection

    Args:
        tokens: Current tokens (batch_size, seq_len)
        mask_ratio: Ratio of tokens to mask
        mask_token_id: ID of the mask token
        remasking_model: Optional remasking model for intelligent selection
        randomness_strength: Balance between random and model-guided remasking
        base_model: Base model for intelligent remasking
        intelligent_remasking: Enable intelligent remasking with base model

    Returns:
        Remasked tokens
    """
    if mask_ratio <= 0:
        return tokens

    batch_size, seq_len = tokens.shape
    device = tokens.device

    # Calculate number of tokens to mask per sequence
    num_to_mask = int(seq_len * mask_ratio)
    if num_to_mask <= 0:
        return tokens

    remasked_tokens = tokens.clone()

    for batch_idx in range(batch_size):
        batch_tokens = tokens[batch_idx]

        # Find non-masked positions and exclude protected positions (e.g., prefix)
        unmaskable_positions = (batch_tokens != mask_token_id)
        if protected_mask is not None:
            unmaskable_positions = unmaskable_positions & (~protected_mask[batch_idx])
        unmaskable_indices = torch.nonzero(unmaskable_positions).squeeze(-1)

        if len(unmaskable_indices) == 0:
            continue

        # Determine which tokens to mask
        actual_num_to_mask = min(num_to_mask, len(unmaskable_indices))

        if remasking_model is not None:
            # Use remasking model for intelligent selection
            positions_to_mask = _select_tokens_with_remasking_model(
                tokens[batch_idx:batch_idx+1], unmaskable_indices, actual_num_to_mask,
                remasking_model, randomness_strength, device
            )
        elif intelligent_remasking and base_model is not None:
            # Use base model for intelligent self-confidence based remasking
            positions_to_mask = _select_tokens_with_confidence(
                tokens[batch_idx:batch_idx+1], unmaskable_indices, actual_num_to_mask,
                base_model, randomness_strength, device
            )
        else:
            # Random selection
            selected_indices = torch.randperm(len(unmaskable_indices))[:actual_num_to_mask]
            positions_to_mask = unmaskable_indices[selected_indices]

        # Apply masking
        remasked_tokens[batch_idx, positions_to_mask] = mask_token_id

    return remasked_tokens


def _select_tokens_with_remasking_model(tokens, unmaskable_indices, num_to_mask,
                                      remasking_model, randomness_strength, device):
    """Select tokens to mask using a dedicated remasking model"""
    with torch.no_grad():
        logits, _ = remasking_model(tokens)

        # Get confidence scores for unmaskable positions
        confidence_scores = logits[0, unmaskable_indices, 1]  # Assuming binary classification

        # Blend with random selection based on randomness_strength
        if randomness_strength > 0:
            random_scores = torch.rand(len(confidence_scores), device=device)
            combined_scores = (1 - randomness_strength) * confidence_scores + randomness_strength * random_scores
        else:
            combined_scores = confidence_scores

        # Select tokens with lowest confidence (most likely to be wrong)
        _, selected_indices = torch.topk(combined_scores, num_to_mask, largest=False)

        return unmaskable_indices[selected_indices]


def _select_tokens_with_confidence(tokens, unmaskable_indices, num_to_mask,
                                 base_model, randomness_strength, device):
    """Select tokens to mask based on model confidence"""


    # Pass dummy targets to get logits for all positions
    dummy_targets = torch.zeros_like(tokens)
    with torch.no_grad():
        logits, _ = base_model(tokens, targets=dummy_targets)
        probs = F.softmax(logits, dim=-1)

        # Get confidence scores for unmaskable positions
        token_probs = probs[0, unmaskable_indices, tokens[0, unmaskable_indices]]

        # Convert to uncertainty (1 - confidence)
        uncertainty_scores = 1.0 - token_probs

        # Blend with random selection based on randomness_strength
        if randomness_strength > 0:
            random_scores = torch.rand(len(uncertainty_scores), device=device)
            combined_scores = (1 - randomness_strength) * uncertainty_scores + randomness_strength * random_scores
        else:
            combined_scores = uncertainty_scores

        # Select tokens with highest uncertainty
        _, selected_indices = torch.topk(combined_scores, num_to_mask, largest=True)

        return unmaskable_indices[selected_indices]


def apply_remasking_step(tokens, prediction_tokens, iteration, iterations, schedule_type='linear',
                        masking_ratios=None, start_ratio=0.9, end_ratio=0.1, remasking_model=None,
                        randomness_strength=0.5, mask_token_id=None, device='cuda', base_model=None,
                        intelligent_remasking=False, verbose=False, logits_from_predict=None,
                        protected_mask=None, schedule_mode='ratio'):
    """
    Apply remasking step with different scheduling options

    Args:
        tokens: Current tokens
        prediction_tokens: Predicted tokens
        iteration: Current iteration
        iterations: Total iterations
        schedule_type: 'linear' or 'custom'
        masking_ratios: Custom masking ratios for each iteration
        start_ratio: Starting mask ratio for linear schedule
        end_ratio: Ending mask ratio for linear schedule
        remasking_model: Optional remasking model
        randomness_strength: Randomness factor
        mask_token_id: Mask token ID
        device: Device
        base_model: Base model for intelligent remasking
        intelligent_remasking: Enable intelligent remasking
        verbose: Enable verbose output
        schedule_mode: 'ratio' (default, mask fixed percentage) or 'threshold' (mask all above threshold)

    Returns:
        tuple: (remasked_tokens, min_wrongness, remasked_indices) where:
               - remasked_tokens: tokens with remasking applied
               - min_wrongness: minimum wrongness of masked tokens
               - remasked_indices: list of (batch_idx, position) tuples for remasked positions
               Returns (None, None, None) if threshold mode and no tokens to mask
    """
    # Determine mask ratio/threshold for next iteration
    if schedule_type == 'custom' and masking_ratios is not None:
        next_iteration = iteration + 1
        if next_iteration < len(masking_ratios):
            mask_ratio = masking_ratios[next_iteration]
        else:
            mask_ratio = end_ratio
    else:
        # Linear schedule
        mask_ratio = linear_remasking_schedule(iteration + 1, iterations, start_ratio, end_ratio)

    if verbose:
        if schedule_mode == 'threshold':
            print(f"  Remasking with threshold: {mask_ratio:.3f}")
        else:
            print(f"  Remasking with ratio: {mask_ratio:.3f}")

    # Vectorized remasking
    batch_size, seq_len = prediction_tokens.shape
    if mask_token_id is None:
        raise ValueError("mask_token_id must be provided")

    unmaskable = (prediction_tokens != mask_token_id)
    if protected_mask is not None:
        unmaskable = unmaskable & (~protected_mask)

    # For ratio mode: determine how many to mask per row
    if schedule_mode == 'ratio':
        k = int(seq_len * mask_ratio)
        if k <= 0:
            return prediction_tokens, None, []
    else:
        # For threshold mode, k will be determined dynamically
        k = None

    if remasking_model is not None:
        # Batched remasking model forward once
        with torch.no_grad():
            logits_r, _ = remasking_model(prediction_tokens)
        # Assume binary: take class-1 confidence; fall back to last dim if shape ambiguous
        if logits_r.dim() == 3 and logits_r.size(-1) > 1:
            confidence = logits_r[:, :, 1]
        else:
            # Single logit per position; treat larger as more confident
            confidence = logits_r.squeeze(-1)

        if schedule_mode == 'threshold':
            # Threshold mode: mask all tokens with low confidence (high wrongness)
            # Convert confidence to wrongness probability
            wrongness = 1.0 - torch.sigmoid(confidence)
            # Invert mask_ratio: start_ratio=0.95 means threshold=0.05
            threshold = 1.0 - mask_ratio
            wrongness_masked = wrongness.masked_fill(~unmaskable, -float('inf'))
            select = (wrongness_masked > threshold) & unmaskable

            if not select.any():
                return None, None, []  # Signal early termination

            remasked_tokens = prediction_tokens.clone()
            remasked_tokens[select] = mask_token_id
            min_wrongness = wrongness[select].min().item() if select.any() else None
            remasked_indices = select.nonzero(as_tuple=False).tolist()
            return remasked_tokens, min_wrongness, remasked_indices
        else:
            # Ratio mode: mask top-k tokens
            # Lower confidence => more likely to mask: score = -confidence
            wrongness = 1.0 - torch.sigmoid(confidence)
            scores = -confidence
            scores = scores.masked_fill(~unmaskable, torch.finfo(scores.dtype).min)
            if randomness_strength > 0:
                scores = (1 - randomness_strength) * scores + randomness_strength * torch.rand_like(scores)

            if k > 0:
                topk_idx = scores.topk(k, dim=1, largest=True).indices
                select = torch.zeros_like(scores, dtype=torch.bool)
                select.scatter_(1, topk_idx, True)
                select &= unmaskable
                remasked_tokens = prediction_tokens.clone()
                remasked_tokens[select] = mask_token_id
                min_wrongness = wrongness[select].min().item() if select.any() else None
                remasked_indices = select.nonzero(as_tuple=False).tolist()
                return remasked_tokens, min_wrongness, remasked_indices
            return prediction_tokens, None, []

    # Critic-guided remasking path: precedence after remasking_model and before intelligent_remasking
    if base_model is not None and getattr(getattr(base_model, 'config', object()), 'add_critic_head', False) and not intelligent_remasking:
        with torch.no_grad():
            critic_logits = base_model.critic_scores(prediction_tokens)
        # Higher critic logit => higher error likelihood (wrongness)
        # Convert to probabilities for threshold mode
        wrongness_probs = torch.sigmoid(critic_logits)

        if schedule_mode == 'threshold':
            # Threshold mode: mask all tokens with wrongness above threshold
            # mask_ratio represents the wrongness threshold (0-1)
            # Higher mask_ratio = higher threshold = fewer tokens masked
            # We invert it: start_ratio=0.95 means threshold=0.05 (mask almost everything)
            threshold = 1.0 - mask_ratio
            wrongness_probs_masked = wrongness_probs.masked_fill(~unmaskable, -float('inf'))
            select = (wrongness_probs_masked > threshold) & unmaskable

            # Check if any tokens need masking
            if not select.any():
                return None, None, []  # Signal early termination

            remasked_tokens = prediction_tokens.clone()
            remasked_tokens[select] = mask_token_id
            min_wrongness = wrongness_probs[select].min().item() if select.any() else None
            remasked_indices = select.nonzero(as_tuple=False).tolist()
            return remasked_tokens, min_wrongness, remasked_indices
        else:
            # Ratio mode: mask top-k tokens based on critic scores (logits)
            scores = critic_logits.clone()
            scores = scores.masked_fill(~unmaskable, torch.finfo(scores.dtype).min)
            if randomness_strength > 0:
                scores = (1 - randomness_strength) * scores + randomness_strength * torch.rand_like(scores)

            if k > 0:
                topk_idx = scores.topk(k, dim=1, largest=True).indices
                select = torch.zeros_like(scores, dtype=torch.bool)
                select.scatter_(1, topk_idx, True)
                select &= unmaskable
                remasked_tokens = prediction_tokens.clone()
                remasked_tokens[select] = mask_token_id
                min_wrongness = wrongness_probs[select].min().item() if select.any() else None
                remasked_indices = select.nonzero(as_tuple=False).tolist()
                return remasked_tokens, min_wrongness, remasked_indices
            return prediction_tokens, None, []

    if intelligent_remasking:
        if logits_from_predict is None:
            # Fallback: per-sample (slower) path using base_model
            # Note: threshold mode not supported in this fallback path
            return apply_remasking(
                tokens=prediction_tokens,
                mask_ratio=mask_ratio,
                mask_token_id=mask_token_id,
                remasking_model=None,
                randomness_strength=randomness_strength,
                base_model=base_model,
                intelligent_remasking=True,
                protected_mask=protected_mask,
            )
        # Use logits from main forward, batched
        probs = F.softmax(logits_from_predict, dim=-1)
        p_taken = probs.gather(-1, prediction_tokens.unsqueeze(-1)).squeeze(-1)
        uncertainty = 1.0 - p_taken  # uncertainty (0-1, higher = more uncertain)

        if schedule_mode == 'threshold':
            # Threshold mode: mask all tokens with uncertainty above threshold
            # Invert mask_ratio: start_ratio=0.95 means threshold=0.05 (mask almost everything)
            threshold = 1.0 - mask_ratio
            uncertainty_masked = uncertainty.masked_fill(~unmaskable, -float('inf'))
            select = (uncertainty_masked > threshold) & unmaskable

            if not select.any():
                return None, None, []  # Signal early termination

            remasked_tokens = prediction_tokens.clone()
            remasked_tokens[select] = mask_token_id
            min_wrongness = uncertainty[select].min().item() if select.any() else None
            remasked_indices = select.nonzero(as_tuple=False).tolist()
            return remasked_tokens, min_wrongness, remasked_indices
        else:
            # Ratio mode: mask top-k tokens
            scores = uncertainty.masked_fill(~unmaskable, torch.finfo(uncertainty.dtype).min)
            if randomness_strength > 0:
                scores = (1 - randomness_strength) * scores + randomness_strength * torch.rand_like(scores)

            if k > 0:
                topk_idx = scores.topk(k, dim=1, largest=True).indices
                select = torch.zeros_like(scores, dtype=torch.bool)
                select.scatter_(1, topk_idx, True)
                select &= unmaskable
                remasked_tokens = prediction_tokens.clone()
                remasked_tokens[select] = mask_token_id
                min_wrongness = uncertainty[select].min().item() if select.any() else None
                remasked_indices = select.nonzero(as_tuple=False).tolist()
                return remasked_tokens, min_wrongness, remasked_indices
            return prediction_tokens, None, []

    # Random remasking fallback (threshold mode not supported here)
    if schedule_mode == 'threshold':
        # For threshold mode without critic/intelligent remasking, fall back to ratio mode
        k = int(seq_len * mask_ratio)
        if k <= 0:
            return prediction_tokens, None, []

    remasked_tokens = apply_remasking(
        tokens=prediction_tokens,
        mask_ratio=mask_ratio,
        mask_token_id=mask_token_id,
        remasking_model=None,
        randomness_strength=randomness_strength,
        base_model=None,
        intelligent_remasking=False,
        protected_mask=protected_mask,
    )
    return remasked_tokens, None, []


def build_critic_artifacts_from_logits(idx: torch.Tensor,
                                       logits: torch.Tensor,
                                       targets: torch.Tensor,
                                       mask_token_id: int,
                                       ignore_index: int,
                                       pad_token_id: int | None = None,
                                       scope: str = 'masked_and_ignore'):
    """
    Build critic sampling artifacts from LM logits and inputs.
    Returns a dict with:
      - pred_tokens: (B, T) sampled tokens from logits
      - critic_input: (B, T) input with masked positions filled by pred_tokens
      - critic_target: (B, T) float tensor, 0 for correct, 1 for error, with ignore positions set to 0 in masked_and_ignore scope
      - critic_valid: (B, T) bool tensor indicating which positions count for critic loss/stats
    Sampling uses multinomial over softmax(logits) (no temperature/top-p here; match training/eval usage).
    """
    if idx is None:
        raise RuntimeError("build_critic_artifacts_from_logits: idx is required")
    if logits is None:
        raise RuntimeError("build_critic_artifacts_from_logits: logits is required")
    if targets is None:
        raise RuntimeError("build_critic_artifacts_from_logits: targets is required")
    if mask_token_id is None:
        raise RuntimeError("build_critic_artifacts_from_logits: mask_token_id is required")

    masked_positions = (idx == int(mask_token_id))

    # Sample predictions from logits; flatten for efficiency then reshape back
    with torch.no_grad():
        probs = F.softmax(logits.detach(), dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled = torch.multinomial(flat, num_samples=1).view(probs.size(0), probs.size(1))
        pred_tokens = sampled

    critic_input = idx.clone()
    critic_input[masked_positions] = pred_tokens[masked_positions]

    # Base target: 1 when critic_input token != ground truth Y, else 0
    critic_target = (critic_input != targets).float()

    # Valid mask and ignore handling per scope
    if scope == 'masked_and_ignore':
        critic_valid = masked_positions | (targets == int(ignore_index))
        # For ignore_index positions, target is always 0
        critic_target = torch.where((targets == int(ignore_index)), torch.zeros_like(critic_target), critic_target)
    elif scope == 'masked_only':
        critic_valid = masked_positions
    else:
        raise RuntimeError(f"Unsupported critic_target_scope: {scope}. Use 'masked_and_ignore' or 'masked_only'.")

    if pad_token_id is not None:
        critic_valid = critic_valid & (idx != int(pad_token_id))

    return {
        'pred_tokens': pred_tokens,
        'critic_input': critic_input,
        'critic_target': critic_target,
        'critic_valid': critic_valid,
    }
