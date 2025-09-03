"""
Masking and corruption strategies for diffusion training.
Contains all masking functions used in unmasking and remasking training.
"""

import torch
from model import GPTConfig, GPT

from .training_config import TrainingContext, UnmaskingStage, UnmaskingStageType


# Global synthetic model for remasking
synthetic_model = None


def apply_random_masking_gpu(x, max_masked_ratio, mask_token_id, meta_vocab_size):
    """
    GPU-optimized random masking for unmasking training.
    Each sample in the batch gets a different random masking probability.
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    mask_probs = torch.rand(batch_size, device=device) * max_masked_ratio
    rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
    mask_probs_expanded = mask_probs.unsqueeze(1).expand(-1, seq_len)
    
    # Step 1: Generate the boolean mask of positions to predict (this logic is unchanged)
    mask = rand_vals < mask_probs_expanded
    
    # Step 2: Apply the 80/10/10 corruption using the new function
    # NOTE: We use meta_vocab_size to avoid generating special tokens randomly
    corrupted_x = apply_bert_style_corruption_gpu(x, mask, mask_token_id, meta_vocab_size)
    
    return corrupted_x, mask


def apply_span_masking_gpu(x, spans_count, mask_token_id, meta_vocab_size):
    """
    GPU-optimized span masking for unmasking training.
    Masks spans_count continuous areas in the input.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        spans_count: Number of continuous spans to mask
        mask_token_id: Token ID to use for masking (NOT used for span masking as per requirement)
        meta_vocab_size: Size of original vocabulary (NOT used for span masking)
        
    Returns:
        masked_x: Input with masked spans directly replaced with mask_token_id (no BERT-style corruption)
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    if spans_count <= 0 or seq_len <= 1:
        # No masking needed - return original
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Generate all random positions at once: (batch_size, 2*spans_count)
    random_positions = torch.rand(batch_size, 2 * spans_count, device=device)
    
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
    
    # Apply masking
    masked_x = x.clone()
    masked_x[mask] = mask_token_id
    
    return masked_x, mask


def apply_stage_masking(x, stage_config: UnmaskingStage, mask_token_id, meta_vocab_size):
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: UnmaskingStage configuration
        mask_token_id: Token ID to use for masking
        meta_vocab_size: Size of original vocabulary (for random token generation, excluding special tokens)
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config.get_stage_type()
    
    if stage_type == UnmaskingStageType.RANDOM:
        config = stage_config.config
        return apply_random_masking_gpu(x, config.max_masked_ratio, mask_token_id, meta_vocab_size)
    elif stage_type == UnmaskingStageType.STICKY:
        config = stage_config.config
        return apply_target_driven_sticky_masking_gpu(
            x, config.target_masked_ratio, config.p1_probability,
            config.p2_probability, mask_token_id, meta_vocab_size
        )
    elif stage_type == UnmaskingStageType.SPAN:
        config = stage_config.config
        return apply_span_masking_gpu(x, config.spans_count, mask_token_id, meta_vocab_size)
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def apply_target_driven_sticky_masking_gpu(x, target_masked_ratio, p1_probability, p2_probability, mask_token_id, meta_vocab_size):
    """
    GPU-optimized target-driven sticky masking for unmasking training.

    Args:
        x: Input tokens (batch_size, seq_len)
        target_masked_ratio: Target fraction of tokens to mask (0.0 to 1.0)
        p1_probability: Probability of masking when no neighbors are masked
        p2_probability: Probability of masking when neighbors are masked
        mask_token_id: Token ID to use for masking
        meta_vocab_size: Size of original vocabulary (for random token generation, excluding special tokens)

    Returns:
        masked_x: Input with masked tokens replaced using BERT-style 80/10/10 corruption
        mask: Boolean mask indicating which positions were masked
    """
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate target number of masked tokens per sequence
    target_masked_count = int(target_masked_ratio * seq_len)
    
    if target_masked_count == 0:
        # No masking needed - return early
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Start with no masks
    masked_x = x.clone()
    
    # Pre-allocate tensors to avoid repeated allocations
    current_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    neighbor_masked = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Continue masking until we reach the target for each sequence
    max_rounds = min(1000, target_masked_count * 10)  # Adaptive safety limit
    target_tensor = torch.tensor(target_masked_count, device=device, dtype=torch.long)
    
    for round_idx in range(max_rounds):
        # Update current mask state
        current_mask = (masked_x == mask_token_id)
        
        # Check if we've reached target for all sequences (GPU-only operation)
        current_counts = current_mask.sum(dim=1)  # (batch_size,)
        sequences_need_more = current_counts < target_tensor
        
        if not sequences_need_more.any():
            break  # All sequences reached target
        
        # Find neighbor positions for sticky masking (reuse buffer)
        neighbor_masked.zero_()
        
        # Check left and right neighbors (vectorized)
        neighbor_masked[:, 1:] |= current_mask[:, :-1]  # Left neighbor
        neighbor_masked[:, :-1] |= current_mask[:, 1:]  # Right neighbor
        
        # Generate random values for masking decision (single GPU call)
        rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
        
        # Apply different probabilities based on neighbor status (vectorized)
        mask_probs = torch.where(neighbor_masked, p2_probability, p1_probability)
        new_masks = (rand_vals < mask_probs) & ~current_mask
        
        # Only mask sequences that haven't reached target yet (vectorized)
        sequences_need_more_expanded = sequences_need_more.unsqueeze(1).expand(-1, seq_len)
        new_masks &= sequences_need_more_expanded
        
        # Apply new masks (vectorized) - just mark positions, don't corrupt yet
        masked_x[new_masks] = mask_token_id

    # Final adjustment: remove excess masks with fully vectorized approach
    final_mask = (masked_x == mask_token_id)
    final_counts = final_mask.sum(dim=1)  # (batch_size,)

    # Only process sequences that exceeded target (minimize CPU-GPU sync)
    exceeded_sequences = torch.where(final_counts > target_tensor)[0]

    if exceeded_sequences.numel() > 0:
        # Process exceeded sequences with minimal loops
        for batch_idx in exceeded_sequences:
            excess = (final_counts[batch_idx] - target_tensor).item()
            if excess > 0:
                # Find masked positions (keep on GPU)
                seq_mask = final_mask[batch_idx]
                masked_positions = torch.where(seq_mask)[0]

                # Randomly select positions to unmask (single GPU operation)
                perm_indices = torch.randperm(masked_positions.size(0), device=device)[:excess]
                positions_to_unmask = masked_positions[perm_indices]

                # Restore original tokens (vectorized)
                masked_x[batch_idx, positions_to_unmask] = x[batch_idx, positions_to_unmask]

    # Get final mask state and apply BERT-style corruption
    final_mask = (masked_x == mask_token_id)

    # Apply BERT-style 80/10/10 corruption to the selected positions
    corrupted_x = apply_bert_style_corruption_gpu(x, final_mask, mask_token_id, meta_vocab_size)

    return corrupted_x, final_mask


def load_synthetic_model(checkpoint_path, device, extended_vocab_size):
    """Load the synthetic model for generating fake data in remasking training"""
    global synthetic_model
    
    if not checkpoint_path or synthetic_model is not None:
        return
    
    try:
        print(f"Loading synthetic model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model arguments from checkpoint
        checkpoint_model_args = checkpoint['model_args']
        
        # Create synthetic model with same architecture as checkpoint
        synthetic_gptconf = GPTConfig(**checkpoint_model_args)
        synthetic_model = GPT(synthetic_gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        # Fix keys if needed (same as main model loading)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        synthetic_model.load_state_dict(state_dict)
        synthetic_model.to(device)
        synthetic_model.eval()  # Always in eval mode
        
        print(f"Synthetic model loaded successfully (vocab_size: {synthetic_model.config.vocab_size})")
        
    except Exception as e:
        print(f"Warning: Could not load synthetic model from {checkpoint_path}: {e}")
        synthetic_model = None


def apply_sticky_corruption_gpu(x, target_masked_ratio, p1_probability, p2_probability, meta_vocab_size, debug=True):
    """Sticky corruption for remasking training with target-driven masking"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate target number of masked tokens per sequence
    target_masked_count = int(target_masked_ratio * seq_len)
    
    if target_masked_count == 0:
        # No masking needed - return original unchanged
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Start with original text
    corrupted_x = x.clone()
    
    # Pre-allocate tensors to avoid repeated allocations
    current_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    neighbor_masked = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Continue masking until we reach the target for each sequence
    max_rounds = min(1000, target_masked_count * 10)  # Adaptive safety limit
    target_tensor = torch.tensor(target_masked_count, device=device, dtype=torch.long)
    
    for round_idx in range(max_rounds):
        # Update current mask state (positions that are corrupted)
        current_mask = (corrupted_x != x)  # True where tokens have been corrupted
        
        # Check if we've reached target for all sequences (GPU-only operation)
        current_counts = current_mask.sum(dim=1)  # (batch_size,)
        sequences_need_more = current_counts < target_tensor
        
        if not sequences_need_more.any():
            break  # All sequences reached target
        
        # Find neighbor positions for sticky masking (reuse buffer)
        neighbor_masked.zero_()
        
        # Check left and right neighbors (vectorized)
        neighbor_masked[:, 1:] |= current_mask[:, :-1]  # Left neighbor
        neighbor_masked[:, :-1] |= current_mask[:, 1:]  # Right neighbor
        
        # Generate random values for masking decision (single GPU call)
        rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
        
        # Apply different probabilities based on neighbor status (vectorized)
        mask_probs = torch.where(neighbor_masked, p2_probability, p1_probability)
        new_masks = (rand_vals < mask_probs) & ~current_mask
        
        # Only mask sequences that haven't reached target yet (vectorized)
        sequences_need_more_expanded = sequences_need_more.unsqueeze(1).expand(-1, seq_len)
        new_masks &= sequences_need_more_expanded
        
        # Apply corruption to newly masked positions (vectorized)
        if new_masks.any():
            # Replace with random tokens from vocabulary
            random_tokens = torch.randint(0, meta_vocab_size, new_masks.sum().shape, device=device)
            corrupted_x[new_masks] = random_tokens
    
    # Final adjustment: remove excess corruptions with fully vectorized approach
    final_mask = (corrupted_x != x)
    final_counts = final_mask.sum(dim=1)  # (batch_size,)
    
    # Only process sequences that exceeded target (minimize CPU-GPU sync)
    exceeded_sequences = torch.where(final_counts > target_tensor)[0]
    
    if exceeded_sequences.numel() > 0:
        # Process exceeded sequences with minimal loops
        for batch_idx in exceeded_sequences:
            excess = (final_counts[batch_idx] - target_tensor).item()
            if excess > 0:
                # Find corrupted positions (keep on GPU)
                seq_mask = final_mask[batch_idx]
                corrupted_positions = torch.where(seq_mask)[0]
                
                # Randomly select positions to restore (single GPU operation)
                perm_indices = torch.randperm(corrupted_positions.size(0), device=device)[:excess]
                positions_to_restore = corrupted_positions[perm_indices]
                
                # Restore original tokens (vectorized)
                corrupted_x[batch_idx, positions_to_restore] = x[batch_idx, positions_to_restore]
    
    # Return final mask state
    final_mask = (corrupted_x != x)
    return corrupted_x, final_mask


def apply_random_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, sticky_transition_start, sticky_transition_end, meta_vocab_size, random_mask_warmup, debug=True):
    """Random corruption for remasking training with iteration-based masking probability"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # FIXED: Convert "unmasked" parameters to actual corruption probabilities
    # guaranteed_unmasked_max=0.9 means 90% unmasked -> 10% corrupted
    # guaranteed_unmasked_min=0.6 means 60% unmasked -> 40% corrupted
    corruption_min = 1.0 - guaranteed_unmasked_max  # Start: 10% corruption
    corruption_max = 1.0 - guaranteed_unmasked_min  # End: 40% corruption
    
    # Calculate masking probability based on iteration
    if iter_num < random_mask_warmup:
        # During warmup, gradually increase corruption from min to max
        progress = iter_num / random_mask_warmup
        mask_prob = corruption_min + progress * (corruption_max - corruption_min)
    elif iter_num < sticky_transition_start:
        mask_prob = corruption_max  # Maximum corruption (40%)
    elif iter_num < sticky_transition_end:
        # Keep maximum corruption during transition
        mask_prob = corruption_max
    else:
        mask_prob = corruption_max  # Stay at maximum corruption
    
    # Apply random masking
    rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
    mask = rand_vals < mask_prob
    
    # Debug: Print corruption rate occasionally during training only (not validation set creation)
    if debug and iter_num % 1000 == 0 and iter_num > 0:
        actual_mask_ratio = mask.float().mean().item()
        print(f"DEBUG: iter {iter_num}, target_corruption={mask_prob:.3f} ({mask_prob*100:.1f}%), actual={actual_mask_ratio:.3f} ({actual_mask_ratio*100:.1f}%)")
        print(f"  corruption_min={corruption_min:.3f}, corruption_max={corruption_max:.3f}, warmup={random_mask_warmup}")
    
    # Create corrupted version by randomly replacing masked tokens
    corrupted_x = x.clone()
    if mask.any():
        # Replace with random tokens from vocabulary
        random_tokens = torch.randint(0, meta_vocab_size, mask.sum().shape, device=device)
        corrupted_x[mask] = random_tokens
    
    return corrupted_x, mask


def apply_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, sticky_transition_start, sticky_transition_end, meta_vocab_size, random_mask_warmup, p1_p2_ratio=1.0, debug=True):
    """Unified corruption function that chooses between random and sticky masking based on p1_p2_ratio"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # Calculate current corruption probability based on iteration
    corruption_min = 1.0 - guaranteed_unmasked_max  # Start: 10% corruption
    corruption_max = 1.0 - guaranteed_unmasked_min  # End: 40% corruption
    
    if iter_num < random_mask_warmup:
        # During warmup, gradually increase corruption from min to max
        progress = iter_num / random_mask_warmup
        target_corruption_rate = corruption_min + progress * (corruption_max - corruption_min)
    elif iter_num < sticky_transition_start:
        target_corruption_rate = corruption_max  # Maximum corruption
    elif iter_num < sticky_transition_end:
        target_corruption_rate = corruption_max  # Keep maximum corruption during transition
    else:
        target_corruption_rate = corruption_max  # Stay at maximum corruption
    
    # Choose masking strategy based on p1_p2_ratio
    if p1_p2_ratio == 1.0:
        # Use random corruption
        return apply_random_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, 
                                         sticky_transition_start, sticky_transition_end, meta_vocab_size, 
                                         random_mask_warmup, debug=debug)
    else:
        # Use sticky corruption
        # Calculate p1 and p2 based on ratio, with max(p1, p2) = target_corruption_rate / 4
        max_prob = target_corruption_rate / 4.0
        
        if p1_p2_ratio > 1.0:
            # p1 is larger
            p1_probability = max_prob
            p2_probability = max_prob / p1_p2_ratio
        else:
            # p2 is larger
            p2_probability = max_prob
            p1_probability = max_prob * p1_p2_ratio
        
        if debug and iter_num % 1000 == 0 and iter_num > 0:
            print(f"DEBUG: iter {iter_num}, sticky masking: target_corruption={target_corruption_rate:.3f} ({target_corruption_rate*100:.1f}%)")
            print(f"  p1_p2_ratio={p1_p2_ratio:.3f}, p1={p1_probability:.3f}, p2={p2_probability:.3f}")
        
        return apply_sticky_corruption_gpu(x, target_corruption_rate, p1_probability, p2_probability, meta_vocab_size, debug=debug)


def apply_bert_style_corruption_gpu(x, mask, mask_token_id, meta_vocab_size):
    """
    Applies the 80/10/10 corruption strategy from BERT to the selected positions.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions selected for prediction (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token.
        meta_vocab_size: The size of the original vocabulary for generating random tokens (excluding special tokens).
        
    Returns:
        corrupted_x: The input tokens after applying the 80/10/10 rule.
    """
    corrupted_x = x.clone()
    
    # Generate random numbers to decide on the corruption type for each masked position
    rand = torch.rand(x.shape, device=x.device)
    
    # Determine the positions for each case based on the main mask
    # 80% of the time, we replace with [MASK]
    mask_token_positions = mask & (rand < 0.8)
    
    # 10% of the time, we replace with a random token (0.8 <= rand < 0.9)
    random_token_positions = mask & (rand >= 0.8) & (rand < 0.9)
    
    # 10% of the time, we keep the original token (rand >= 0.9) - no action needed for these
    
    # Apply the [MASK] tokens
    corrupted_x[mask_token_positions] = mask_token_id
    
    # Apply the random tokens
    num_random = random_token_positions.sum()
    if num_random > 0:
        random_tokens = torch.randint(0, meta_vocab_size, (num_random,), device=x.device)
        corrupted_x[random_token_positions] = random_tokens
        
    return corrupted_x


def get_progressive_validation_iterations(eval_iters, max_iters):
    """Generate validation iterations for progressive validation"""
    # Create a range of iterations from early to late training
    iterations = []
    for i in range(eval_iters):
        progress = i / (eval_iters - 1) if eval_iters > 1 else 0
        iter_val = int(progress * max_iters)
        iterations.append(iter_val)
    return iterations