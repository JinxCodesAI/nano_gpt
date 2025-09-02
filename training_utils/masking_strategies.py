"""
Masking and corruption strategies for diffusion training.
Contains all masking functions used in unmasking and remasking training.
"""

import torch
from model import GPTConfig, GPT

from .training_config import TrainingContext, UnmaskingStage, UnmaskingStageType


# Global synthetic model for remasking
synthetic_model = None


# REMOVED: Dataset-specific masking functions moved to data/<dataset>/data_utils.py
# - apply_random_masking_gpu → moved to data/shakespeare_char_diffusion/data_utils.py
# - apply_target_driven_sticky_masking_gpu → moved to data/shakespeare_char_diffusion/data_utils.py
# - apply_stage_masking → moved to data/shakespeare_char_diffusion/data_utils.py


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


# REMOVED: Dataset-specific corruption functions moved to data/<dataset>/data_utils.py
# - apply_sticky_corruption_gpu → moved to data/shakespeare_char_diffusion/data_utils.py
# - apply_random_corruption_gpu → moved to data/shakespeare_char_diffusion/data_utils.py  
# - apply_corruption_gpu → moved to data/shakespeare_char_diffusion/data_utils.py


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