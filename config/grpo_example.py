"""
Example GRPO Configuration

This is an example configuration for GRPO training.
Copy and modify this file for your specific use case.

Usage:
    python grpo/train_grpo.py config/grpo_example.py
"""

# -----------------------------------------------------------------------------
# Model checkpoints (REQUIRED - update these paths)
# -----------------------------------------------------------------------------

# Generator model to fine-tune
generator_checkpoint = 'out-char-diffusion/7250_1.77_pad_no_entropy.pt'

# Judge model for reward signal (must be a sequence scorer)
judge_checkpoint = 'out-char-diffusion/padded_judge_0.0155.pt'

# Reference model (optional, defaults to generator_checkpoint)
reference_checkpoint = None

# -----------------------------------------------------------------------------
# GRPO hyperparameters
# -----------------------------------------------------------------------------

# Group sampling
group_size = 8              # Number of completions per input
                            # Larger = more stable but slower

# KL divergence penalty
kl_beta = 0.1               # Coefficient for KL penalty
                            # Higher = more conservative

# Sampling
temperature = 0.8           # Sampling temperature
top_p = 0.95                # Nucleus sampling

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

# Learning rate (typically 10-100x lower than pretraining)
learning_rate = 1e-5
min_lr = 1e-6

# Schedule
max_iters = 10000
warmup_iters = 100
lr_decay_iters = 10000

# Optimization
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Batch size
batch_size = 16             # Number of unique inputs
                            # Effective batch = batch_size * group_size

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

dataset = 'char_diffusion'
block_size = 1024

# -----------------------------------------------------------------------------
# Logging and checkpointing
# -----------------------------------------------------------------------------

log_interval = 10
save_interval = 1000
sample_interval = 500       # Set to 0 to disable

# WandB (optional)
wandb_log = False
wandb_project = 'grpo'
wandb_run_name = 'grpo_example'

# Output
out_dir = 'out-grpo'

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

device = 'cuda'
dtype = 'bfloat16'
compile = False             # Disable torch.compile for GRPO

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------

# Key hyperparameters to tune:
# 1. group_size: Start with 8, increase for stability
# 2. kl_beta: Start with 0.1, increase if model diverges
# 3. learning_rate: Start with 1e-5, adjust based on loss
# 4. temperature: Start with 0.8, adjust for diversity

