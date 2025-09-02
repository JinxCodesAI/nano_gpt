"""
Dataset-specific training configurations for Shakespeare character diffusion.
Moved from config/shkspr_char_diff/optimal5.py - these are dataset constraints, not training hyperparameters.
"""

# Moved from config/shkspr_char_diff/optimal5.py
UNMASKING_STAGES = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]

VALIDATION_STAGES = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]

# Dataset-specific parameters (NOT training hyperparameters)
BLOCK_SIZE = 1024                    # CONSTRAINT: Must match this exactly
USE_PARAGRAPH_BOUNDARIES = False     # Dataset-specific data structure
USE_ALL_STAGES_FOR_TRAINING = True   # Dataset-specific training approach

# Validation data configuration (dataset-layer responsibility)
VALIDATION_SAMPLES_PER_STAGE = 200   # Fixed number of validation samples per stage
VALIDATION_SAMPLES_PER_FILE = 50     # How many samples in each .pt file
VALIDATION_BATCH_SIZE = 64           # Standard batch size for validation sample generation

# Cache configuration
BATCH_CACHE_SIZE = 1000              # Number of batches to cache in memory
PREFETCH_BUFFER_SIZE = 10            # Runtime prefetch buffer

# Special token configuration  
SPECIAL_TOKEN_OFFSET = 15            # Number of special tokens to reserve