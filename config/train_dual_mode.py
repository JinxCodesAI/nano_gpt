"""
Configuration for dual-mode training combining LANGUAGE_MODEL and SEQUENCE_SCORER.

This config trains a single model that switches between language modeling and sequence
scoring modes at runtime, based on batch metadata.
"""

# Dataset configuration
dataset = 'dual_mode'
batch_size = 16
block_size = 256
target_size = 256

# Dual-mode specific configuration
mode_distribution = {
    'language_model': 0.5,      # 50% language model batches
    'sequence_scorer': 0.5,     # 50% sequence scorer batches
}
alternation_frequency = 1  # Alternate every batch

# Data provider configuration
batches_per_file = 100
max_backlog_files = 3
sleep_seconds = 2.0

# Model architecture (dual-mode: has both heads)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

# Attention and position encoding
attention_type = 'bidirectional'  # Required for both modes
position_encoding = 'absolute'

# Sequence scorer configuration
cls_token_id = None  # CLS token for sequence scoring
mlm_checkpoint_path = 'out-char-diffusion/1.69_MLM_8500.pt'  # Path to MLM model for unmasking

# Char diffusion (LANGUAGE_MODEL) configuration
mask_probability = 0.15
mask_token_id = None  # Will be set by provider based on vocab

# Training configuration
max_iters = 10000
eval_interval = 500
eval_iters = 100
log_interval = 10

# Optimizer
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 10000
min_lr = 3e-5

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False

# Gradient accumulation
gradient_accumulation_steps = 4

# Output
out_dir = 'out/dual_mode'
eval_only = False
always_save_checkpoint = False

# Logging
wandb_log = False
wandb_project = 'nano_gpt'
wandb_run_name = 'dual_mode'

# Note: Model mode is NOT specified in config anymore
# The model defaults to LANGUAGE_MODEL and switches based on batch metadata

