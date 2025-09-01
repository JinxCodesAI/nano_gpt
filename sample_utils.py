"""
Utility functions for diffusion-based text generation
"""
import math
import torch
from collections import Counter


def linear_remasking_schedule(total_iterations, current_iteration, start_ratio, end_ratio):
    """Linear decrease in remask ratio from start_ratio to end_ratio"""
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    return start_ratio - progress * (start_ratio - end_ratio)


def nucleus_sample(logits, top_p=1.0, temperature=1.0, recent_tokens=None, repetition_penalty=1.0):
    """
    Nucleus (top-p) sampling from logits with optional repetition penalty
    
    Args:
        logits: Tensor of shape (..., vocab_size) containing logits
        top_p: Float between 0 and 1. If < 1.0, only sample from tokens whose 
               cumulative probability is within top_p
        temperature: Temperature for scaling logits
        recent_tokens: Recent tokens to apply repetition penalty to (optional)
        repetition_penalty: Penalty for repeating recent tokens (>1.0 = discourage)
    
    Returns:
        Sampled token indices
    """
    # Apply repetition penalty if provided
    if recent_tokens is not None and repetition_penalty != 1.0 and len(recent_tokens) > 0:
        for token_id in set(recent_tokens):
            if token_id < logits.shape[-1]:
                logits[..., token_id] = logits[..., token_id] / repetition_penalty
    if temperature != 1.0:
        logits = logits / temperature
    
    if top_p >= 1.0:
        # Standard sampling without nucleus filtering
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # Apply nucleus sampling
    probs = torch.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff: indices where cumulative probability exceeds top_p
    cutoff_mask = cumulative_probs > top_p
    
    # Keep at least one token (the most probable one)
    cutoff_mask[..., 0] = False
    
    # Zero out probabilities beyond the cutoff
    sorted_probs[cutoff_mask] = 0.0
    
    # Sample from the filtered distribution
    if sorted_probs.sum(dim=-1, keepdim=True).min() > 0:
        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        # Map back to original indices
        batch_indices = torch.arange(sorted_indices.shape[0], device=sorted_indices.device).unsqueeze(-1)
        if len(sorted_indices.shape) > 2:
            # Handle multi-dimensional case
            sample_indices = sampled_sorted_indices.unsqueeze(-1)
            result = torch.gather(sorted_indices, -1, sample_indices).squeeze(-1)
        else:
            result = sorted_indices[batch_indices, sampled_sorted_indices.unsqueeze(-1)].squeeze(-1)
        return result
    else:
        # Fallback: sample from original distribution if filtering removed everything
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, mask_token_id, device, verbose=False):
    """
    Apply remasking using either random selection or model-guided selection
    
    Args:
        tokens: Current token sequence (batch_size, seq_len)
        remask_ratio: Fraction of tokens to remask
        remasking_model: Optional remasking_binary model
        randomness_strength: Balance between random (1.0) and model-guided (0.0)
        mask_token_id: ID of mask token
        device: Device to run on
        verbose: Whether to print debug info
    
    Returns:
        tokens: Updated token sequence with remasked positions
    """
    batch_size, seq_len = tokens.shape
    
    # Calculate target number of masks based on total sequence length
    target_masked = int(seq_len * remask_ratio)
    
    # Count currently masked positions
    currently_masked = (tokens == mask_token_id).sum(dim=1)  # Count per sample
    
    # Calculate how many additional masks needed per sample
    additional_masks_needed = target_masked - currently_masked
    additional_masks_needed = torch.clamp(additional_masks_needed, min=0)  # Don't unmask
    
    # Check if any samples need additional masks
    if additional_masks_needed.sum() == 0:
        if verbose:
            print(f"  No tokens to remask (ratio: {remask_ratio:.3f})")
        return tokens
    
    if remasking_model is None:
        # Pure random remasking
        if verbose:
            avg_additional = additional_masks_needed.float().mean().item()
            avg_currently_masked = currently_masked.float().mean().item()
            print(f"  Using pure random remasking: adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
        
        for batch_idx in range(batch_size):
            num_additional = additional_masks_needed[batch_idx].item()
            if num_additional > 0:
                # Only select from unmasked positions
                unmasked_positions = (tokens[batch_idx] != mask_token_id)
                unmasked_indices = torch.where(unmasked_positions)[0]
                if len(unmasked_indices) > 0:
                    num_to_select = min(num_additional, len(unmasked_indices))
                    selected_positions = torch.randperm(len(unmasked_indices), device=device)[:num_to_select]
                    remask_indices = unmasked_indices[selected_positions]
                    tokens[batch_idx, remask_indices] = mask_token_id
    else:
        # Model-guided remasking with randomness
        if verbose:
            avg_additional = additional_masks_needed.float().mean().item()
            avg_currently_masked = currently_masked.float().mean().item()
            print(f"  Using model-guided remasking: randomness={randomness_strength:.2f}, adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
        
        with torch.no_grad():
            # Get model predictions for all samples
            logits, _ = remasking_model(tokens, None)
            
            # Get probabilities for "remask" class (class 1 for remasking_binary)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, 2)
            wrong_token_probs = probs[:, :, 1]  # (batch_size, seq_len)
            
            for batch_idx in range(batch_size):
                num_additional = additional_masks_needed[batch_idx].item()
                if num_additional > 0:
                    # Only consider unmasked positions for remasking
                    unmasked_positions = (tokens[batch_idx] != mask_token_id)
                    unmasked_indices = torch.where(unmasked_positions)[0]
                    if len(unmasked_indices) > 0:
                        # Get probabilities only for unmasked positions
                        unmasked_model_probs = wrong_token_probs[batch_idx, unmasked_indices]
                        unmasked_rand_probs = torch.rand(len(unmasked_indices), device=device)
                        
                        # Combine random and model probabilities
                        combined_probs = randomness_strength * unmasked_rand_probs + (1 - randomness_strength) * unmasked_model_probs
                        
                        # Select top positions by combined probability
                        num_to_select = min(num_additional, len(unmasked_indices))
                        _, top_indices = torch.topk(combined_probs, num_to_select)
                        remask_indices = unmasked_indices[top_indices]
                        tokens[batch_idx, remask_indices] = mask_token_id
                        
                        if verbose and batch_idx == 0:  # Show stats for first sample
                            model_prob_range = f"{unmasked_model_probs.min().item():.3f}-{unmasked_model_probs.max().item():.3f}"
                            selected_model_probs = unmasked_model_probs[top_indices]
                            selected_rand_probs = unmasked_rand_probs[top_indices]
                            avg_model_prob = selected_model_probs.mean().item()
                            avg_rand_prob = selected_rand_probs.mean().item()
                            print(f"    Model prob range: {model_prob_range}")
                            print(f"    Selected positions - avg model prob: {avg_model_prob:.3f}, avg rand prob: {avg_rand_prob:.3f}")
    
    return tokens


def predict_and_sample_tokens(model, tokens, mask_token_id, temperature, top_p, repetition_penalty, 
                            repetition_window, vocab_size, device, debug_logging_fn=None, 
                            itos=None, stoi=None, verbose=False, log_debug=False):
    """
    Predict and sample tokens for all masked positions
    
    Args:
        model: The trained diffusion model
        tokens: Current token sequence (batch_size, seq_len)
        mask_token_id: ID of mask token
        temperature: Temperature for sampling
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repeating recent tokens
        repetition_window: Window size for repetition penalty
        vocab_size: Size of vocabulary
        device: Device to run on
        debug_logging_fn: Function for debug logging (optional)
        itos: Index to string mapping (for debug logging)
        stoi: String to index mapping (for debug logging)
        verbose: Whether to print progress
        log_debug: Whether to do detailed debug logging
        
    Returns:
        (updated_tokens, prediction_tokens): Updated tokens and tokens after prediction step
    """
    batch_size, seq_len = tokens.shape
    masked_positions = (tokens == mask_token_id)
    total_masked = masked_positions.sum().item()
    
    if total_masked > 0:
        with torch.no_grad():
            # Get predictions for all samples
            dummy_targets = torch.zeros_like(tokens)
            logits, _ = model(tokens, dummy_targets)
            
            # Sample new tokens for masked positions
            for sample_idx in range(batch_size):
                sample_masked = masked_positions[sample_idx]
                if sample_masked.sum() > 0:
                    mask_indices = torch.where(sample_masked)[0]
                    masked_logits = logits[sample_idx, mask_indices]
                    
                    # Get recent tokens for repetition penalty
                    if repetition_penalty != 1.0 and repetition_window > 0:
                        # Get recent unmasked tokens for this sample
                        sample_tokens = tokens[sample_idx]
                        unmasked_tokens = sample_tokens[sample_tokens != mask_token_id]
                        recent_tokens = unmasked_tokens[-repetition_window:].tolist() if len(unmasked_tokens) > 0 else []
                    else:
                        recent_tokens = None
                    
                    # Sample using nucleus sampling with repetition penalty
                    new_tokens = nucleus_sample(masked_logits, top_p=top_p, temperature=temperature,
                                              recent_tokens=recent_tokens, repetition_penalty=repetition_penalty)
                    
                    # Debug logging if function provided
                    if debug_logging_fn and verbose and log_debug:
                        debug_logging_fn(sample_idx, logits, mask_indices, masked_logits, new_tokens, 
                                       mask_token_id, vocab_size, itos, stoi, log_debug)
                    
                    tokens[sample_idx, mask_indices] = new_tokens
    
    # Save tokens after prediction for accurate display
    prediction_tokens = tokens.clone()
    
    if verbose:
        predicted_masks = (prediction_tokens == mask_token_id).sum().item()
        if predicted_masks > 0:
            print(f"  ⚠️  Model predicted {predicted_masks} mask tokens during prediction step!")
    
    return tokens, prediction_tokens


def apply_remasking_step(tokens, prediction_tokens, iteration, iterations, schedule_type, masking_ratios,
                        start_ratio, end_ratio, remasking_model, randomness_strength, mask_token_id, 
                        device, verbose=False):
    """
    Apply remasking step with scheduling
    
    Args:
        tokens: Current token sequence
        prediction_tokens: Tokens after prediction (for tracking)
        iteration: Current iteration number
        iterations: Total iterations
        schedule_type: 'linear' or 'custom'
        masking_ratios: Custom masking ratios (if schedule_type='custom')
        start_ratio: Starting ratio for linear schedule
        end_ratio: Ending ratio for linear schedule
        remasking_model: Optional remasking model
        randomness_strength: Balance between random and model-guided remasking
        mask_token_id: ID of mask token
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Updated tokens after remasking
    """
    if schedule_type == 'custom':
        # Use the next ratio from the custom schedule
        remask_ratio = masking_ratios[iteration + 1]
    else:
        # Use linear schedule
        remask_ratio = linear_remasking_schedule(iterations, iteration + 1, start_ratio, end_ratio)
    
    # Debug: Track masks before remasking (using actual prediction results)
    masks_before = (prediction_tokens == mask_token_id).sum().item()
    
    tokens = apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, 
                           mask_token_id, device, verbose)
    
    # Debug: Track masks after remasking
    if verbose:
        masks_after = (tokens == mask_token_id).sum().item()
        mask_change = masks_after - masks_before
        print(f"  Masks: {masks_before} → {masks_after} (change: {mask_change:+d})")
    
    return tokens


def calculate_selfconfidence_ratio(model, tokens, mask_token_id, device, ctx):
    """
    Calculate self-confidence score for generated tokens by running the model
    on the final output and computing the average log-probability of selected tokens.
    
    Args:
        model: The trained diffusion model
        tokens: Final generated token sequence (batch_size, seq_len)
        mask_token_id: ID of mask token (should not be present in final tokens)
        device: Device to run on
        ctx: Context manager for autocast
        
    Returns:
        List of average log-probabilities for each sample in the batch
    """
    batch_size, seq_len = tokens.shape
    
    with torch.no_grad():
        with ctx:
            # Create dummy targets (not used in inference)
            dummy_targets = torch.zeros_like(tokens)
            
            # Get model predictions for the final tokens
            logits, _ = model(tokens, dummy_targets)
            
            # Convert logits to log-probabilities (more numerically stable than softmax)
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
            
            # Create batch and position indices for advanced indexing
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            pos_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Extract log-probabilities for actual tokens using advanced indexing
            selected_log_probs = log_probs[batch_indices, pos_indices, tokens]  # (batch_size, seq_len)
            
            # Create mask for valid tokens (not mask_token_id and within vocab bounds)
            valid_mask = (tokens != mask_token_id) & (tokens < log_probs.shape[-1])
            
            # Calculate average log-probability for valid tokens in each sample (vectorized)
            # Set invalid positions to 0 for averaging (they'll be excluded by count)
            masked_log_probs = torch.where(valid_mask, selected_log_probs, torch.zeros_like(selected_log_probs))
            
            # Sum log-probabilities and count valid tokens per sample
            log_prob_sums = masked_log_probs.sum(dim=1)  # (batch_size,)
            valid_counts = valid_mask.sum(dim=1).float()  # (batch_size,)
            
            # Calculate average log-probability per sample
            # Use a small epsilon to avoid division by zero
            avg_log_probs = torch.where(valid_counts > 0, log_prob_sums / valid_counts, 
                                       torch.full_like(log_prob_sums, float('-inf')))
            
    return avg_log_probs.cpu().tolist()