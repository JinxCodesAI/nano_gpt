"""Dataset-specific utilities for Shakespeare character diffusion training"""
import torch
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Any

# Add parent directories to import training utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from training_utils.masking_strategies import apply_bert_style_corruption_gpu


def validate_block_size(requested_block_size: int, dataset_block_size: int):
    """Validate that training block_size matches dataset constraint"""
    if requested_block_size != dataset_block_size:
        raise ValueError(f"Block size mismatch: dataset requires {dataset_block_size}, got {requested_block_size}")


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


def apply_target_driven_sticky_masking_gpu(x, target_masked_ratio, p1_probability, p2_probability, mask_token_id):
    """
    GPU-optimized target-driven sticky masking for unmasking training.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        target_masked_ratio: Target fraction of tokens to mask (0.0 to 1.0)
        p1_probability: Probability of masking when no neighbors are masked
        p2_probability: Probability of masking when neighbors are masked
        mask_token_id: Token ID to use for masking
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
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
        
        # Apply new masks (vectorized)
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
    
    # Return final mask state
    final_mask = (masked_x == mask_token_id)
    return masked_x, final_mask


def apply_stage_masking(x, stage_config: Dict, mask_token_id, meta_vocab_size):
    """
    Apply masking based on stage configuration type.
    
    Args:
        x: Input tokens (batch_size, seq_len)
        stage_config: Stage configuration dictionary
        mask_token_id: Token ID to use for masking
        meta_vocab_size: Size of original vocabulary (for random token generation, excluding special tokens)
        
    Returns:
        masked_x: Input with masked tokens replaced by mask_token_id
        mask: Boolean mask indicating which positions were masked
    """
    stage_type = stage_config['type']
    
    if stage_type == 'random':
        return apply_random_masking_gpu(x, stage_config['max_masked_ratio'], mask_token_id, meta_vocab_size)
    elif stage_type == 'sticky':
        return apply_target_driven_sticky_masking_gpu(
            x, stage_config['target_masked_ratio'], stage_config['p1_probability'], 
            stage_config['p2_probability'], mask_token_id
        )
    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def generate_batch_for_iteration(data: np.memmap, valid_indices: np.ndarray, 
                                iteration: int, batch_size: int, block_size: int,
                                stages_config: List[Dict], meta: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch for specific iteration using stage progression logic"""
    # Implementation will be added during prepare.py creation
    # This is a placeholder for the batch generation logic that will be moved here
    pass


def create_iteration_mapping(max_iters: int, stages_config: List[Dict]) -> Dict[int, int]:
    """Create mapping from iteration number to stage index"""
    # Simple mapping: divide iterations evenly across stages
    iterations_per_stage = max_iters // len(stages_config)
    mapping = {}
    
    for iteration in range(max_iters):
        stage_idx = min(iteration // iterations_per_stage, len(stages_config) - 1)
        mapping[iteration] = stage_idx
    
    return mapping


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


def apply_corruption_gpu(x, iter_num, guaranteed_unmasked_max, guaranteed_unmasked_min, 
                        sticky_transition_start, sticky_transition_end, meta_vocab_size, 
                        random_mask_warmup, p1_p2_ratio=1.0, debug=True):
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
        rand_vals = torch.rand_like(x, dtype=torch.float, device=device)
        mask = rand_vals < target_corruption_rate
        
        # Create corrupted version by randomly replacing masked tokens
        corrupted_x = x.clone()
        if mask.any():
            # Replace with random tokens from vocabulary
            random_tokens = torch.randint(0, meta_vocab_size, mask.sum().shape, device=device)
            corrupted_x[mask] = random_tokens
        
        return corrupted_x, mask
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