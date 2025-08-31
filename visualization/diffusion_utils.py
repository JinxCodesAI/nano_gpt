"""
Diffusion utility functions extracted from sample.py
This avoids executing sample.py at import time while keeping code DRY
"""

import torch


# Global variables that will be set by the calling script
start_ratio = 0.99
end_ratio = 0.05

def linear_remasking_schedule(total_iterations, current_iteration):
    """Linear decrease in remask ratio from start_ratio to end_ratio"""
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    return start_ratio - progress * (start_ratio - end_ratio)


def apply_remasking(tokens, remask_ratio, remasking_model, randomness_strength, mask_token_id, device, 
                   base_model=None, intelligent_remasking=False, verbose=False, protected_positions=None):
    """
    Apply remasking using either random selection or model-guided selection
    
    Args:
        tokens: Current token sequence (batch_size, seq_len)
        remask_ratio: Fraction of tokens to remask
        remasking_model: Optional remasking_binary model
        randomness_strength: Balance between random (1.0) and model-guided (0.0)
        mask_token_id: ID of mask token
        device: Device to run on
        base_model: Base model for intelligent remasking when remasking_model is None
        intelligent_remasking: Use base model for intelligent remasking
        verbose: Whether to print debug info
        protected_positions: Set of sequence positions that should not be remasked
    
    Returns:
        tokens: Updated token sequence with remasked positions
    """
    batch_size, seq_len = tokens.shape
    
    # Initialize protected_positions if not provided
    if protected_positions is None:
        protected_positions = set()
    
    if verbose:
        print(f"  DEBUG: apply_remasking called with protected_positions: {sorted(protected_positions) if protected_positions else 'None'}")
    
    def get_remaskable_indices(batch_idx, device):
        """Get indices that are unmasked and not protected"""
        unmasked_positions = (tokens[batch_idx] != mask_token_id)
        unmasked_indices = torch.where(unmasked_positions)[0]
        
        # Filter out protected positions
        if protected_positions:
            protected_tensor = torch.tensor(list(protected_positions), device=device, dtype=torch.long)
            # Keep only indices that are not in protected_positions
            mask = ~torch.isin(unmasked_indices, protected_tensor)
            remaskable_indices = unmasked_indices[mask]
            
            if verbose and batch_idx == 0:
                print(f"    DEBUG: batch {batch_idx}: unmasked={len(unmasked_indices)}, protected={len(protected_positions)}, remaskable={len(remaskable_indices)}")
                print(f"    DEBUG: protected positions: {sorted(protected_positions)}")
                print(f"    DEBUG: filtered out {len(unmasked_indices) - len(remaskable_indices)} protected positions")
        else:
            remaskable_indices = unmasked_indices
            
        return remaskable_indices
    
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
        if intelligent_remasking and base_model is not None:
            # Base model intelligent remasking
            if verbose:
                avg_additional = additional_masks_needed.float().mean().item()
                avg_currently_masked = currently_masked.float().mean().item()
                print(f"  Using base model intelligent remasking: randomness={randomness_strength:.2f}, adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
            
            with torch.no_grad():
                # Get base model predictions for all samples
                dummy_targets = torch.zeros_like(tokens)
                logits, _ = base_model(tokens, dummy_targets)
                
                # Get probabilities and compute wrong_token_probs = 1 - current_token_prob
                probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                current_token_probs = probs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)  # (batch_size, seq_len)
                wrong_token_probs = 1 - current_token_probs  # (batch_size, seq_len)
                
                for batch_idx in range(batch_size):
                    num_additional = additional_masks_needed[batch_idx].item()
                    if num_additional > 0:
                        # Only consider unmasked, non-protected positions for remasking
                        remaskable_indices = get_remaskable_indices(batch_idx, device)
                        if len(remaskable_indices) > 0:
                            # Get probabilities only for remaskable positions
                            remaskable_model_probs = wrong_token_probs[batch_idx, remaskable_indices]
                            remaskable_rand_probs = torch.rand(len(remaskable_indices), device=device)
                            
                            # Combine random and model probabilities
                            combined_probs = randomness_strength * remaskable_rand_probs + (1 - randomness_strength) * remaskable_model_probs
                            
                            # Select top positions by combined probability
                            num_to_select = min(num_additional, len(remaskable_indices))
                            _, top_indices = torch.topk(combined_probs, num_to_select)
                            remask_indices = remaskable_indices[top_indices]
                            tokens[batch_idx, remask_indices] = mask_token_id
                            
                            if verbose and batch_idx == 0:  # Show stats for first sample
                                model_prob_range = f"{remaskable_model_probs.min().item():.3f}-{remaskable_model_probs.max().item():.3f}"
                                selected_model_probs = remaskable_model_probs[top_indices]
                                selected_rand_probs = remaskable_rand_probs[top_indices]
                                avg_model_prob = selected_model_probs.mean().item()
                                avg_rand_prob = selected_rand_probs.mean().item()
                                print(f"    Model prob range: {model_prob_range}")
                                print(f"    Selected positions - avg model prob: {avg_model_prob:.3f}, avg rand prob: {avg_rand_prob:.3f}")
        else:
            # Pure random remasking
            if verbose:
                avg_additional = additional_masks_needed.float().mean().item()
                avg_currently_masked = currently_masked.float().mean().item()
                print(f"  Using pure random remasking: adding {avg_additional:.1f} masks (currently {avg_currently_masked:.1f}/{seq_len}, target {target_masked})")
            
            for batch_idx in range(batch_size):
                num_additional = additional_masks_needed[batch_idx].item()
                if num_additional > 0:
                    # Only select from unmasked, non-protected positions
                    remaskable_indices = get_remaskable_indices(batch_idx, device)
                    if len(remaskable_indices) > 0:
                        num_to_select = min(num_additional, len(remaskable_indices))
                        selected_positions = torch.randperm(len(remaskable_indices), device=device)[:num_to_select]
                        remask_indices = remaskable_indices[selected_positions]
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
                    # Only consider unmasked, non-protected positions for remasking
                    remaskable_indices = get_remaskable_indices(batch_idx, device)
                    if len(remaskable_indices) > 0:
                        # Get probabilities only for remaskable positions
                        remaskable_model_probs = wrong_token_probs[batch_idx, remaskable_indices]
                        remaskable_rand_probs = torch.rand(len(remaskable_indices), device=device)
                        
                        # Combine random and model probabilities
                        combined_probs = randomness_strength * remaskable_rand_probs + (1 - randomness_strength) * remaskable_model_probs
                        
                        # Select top positions by combined probability
                        num_to_select = min(num_additional, len(remaskable_indices))
                        _, top_indices = torch.topk(combined_probs, num_to_select)
                        remask_indices = remaskable_indices[top_indices]
                        tokens[batch_idx, remask_indices] = mask_token_id
                        
                        if verbose and batch_idx == 0:  # Show stats for first sample
                            model_prob_range = f"{remaskable_model_probs.min().item():.3f}-{remaskable_model_probs.max().item():.3f}"
                            selected_model_probs = remaskable_model_probs[top_indices]
                            selected_rand_probs = remaskable_rand_probs[top_indices]
                            avg_model_prob = selected_model_probs.mean().item()
                            avg_rand_prob = selected_rand_probs.mean().item()
                            print(f"    Model prob range: {model_prob_range}")
                            print(f"    Selected positions - avg model prob: {avg_model_prob:.3f}, avg rand prob: {avg_rand_prob:.3f}")
    
    return tokens