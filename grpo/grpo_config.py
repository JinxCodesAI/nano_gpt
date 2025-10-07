"""
GRPO Configuration: Hyperparameters and settings for GRPO training.

This file contains default configuration values for GRPO training.
Override these values via command line or config file using configurator.py.
"""

# -----------------------------------------------------------------------------
# GRPO-specific hyperparameters
# -----------------------------------------------------------------------------

# Group sampling
group_size = 8              # Number of completions per input (k in GRPO)
                            # Effective batch size = batch_size * group_size

# KL divergence penalty
kl_beta = 0.1               # Coefficient for KL divergence penalty
                            # Higher values keep model closer to reference
                            # Lower values allow more exploration

# Sampling parameters
temperature = 0.8           # Sampling temperature for generation
top_p = 0.95                # Nucleus sampling threshold

# -----------------------------------------------------------------------------
# Model checkpoints
# -----------------------------------------------------------------------------

# Generator model (will be trained)
generator_checkpoint = 'out-char-diffusion/checkpoint.pt'

# Judge model (frozen, provides reward signal)
judge_checkpoint = 'out-char-diffusion/judge.pt'

# Reference model (optional, if None will copy from generator_checkpoint)
# This is a frozen copy of the initial generator for KL penalty
reference_checkpoint = None

# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------

# Learning rate (typically lower than pretraining)
learning_rate = 1e-5        # Max learning rate for GRPO fine-tuning
min_lr = 1e-6               # Minimum learning rate

# Training schedule
max_iters = 10000           # Total number of GRPO iterations
warmup_iters = 100          # Warmup iterations for learning rate
lr_decay_iters = 10000      # LR decay schedule length

# Optimization
weight_decay = 1e-1         # Weight decay
beta1 = 0.9                 # Adam beta1
beta2 = 0.95                # Adam beta2
grad_clip = 1.0             # Gradient clipping threshold

# Batch size
batch_size = 16             # Number of unique inputs per iteration
                            # Total samples processed = batch_size * group_size

gradient_accumulation_steps = 1  # Gradient accumulation

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

dataset = 'char_diffusion'  # Dataset name (must have masked inputs)
block_size = 1024           # Sequence length
target_size = None          # Target size (defaults to block_size)

# -----------------------------------------------------------------------------
# Logging and checkpointing
# -----------------------------------------------------------------------------

log_interval = 10           # Log metrics every N iterations
save_interval = 1000        # Save checkpoint every N iterations
sample_interval = 500       # Generate samples for monitoring every N iterations
                            # Set to 0 to disable sampling

# WandB logging
wandb_log = False           # Enable WandB logging
wandb_project = 'grpo'      # WandB project name
wandb_run_name = 'grpo_run' # WandB run name

# Output directory
out_dir = 'out-grpo'        # Directory for checkpoints and logs

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------

device = 'cuda'             # Device: 'cpu', 'cuda', 'cuda:0', etc.
dtype = 'bfloat16'          # Data type: 'float32', 'bfloat16', 'float16'
compile = False             # Use torch.compile (may cause issues with GRPO)

# DDP settings
backend = 'nccl'            # DDP backend

# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------

init_from = 'resume'        # Always 'resume' for GRPO (loads generator checkpoint)
seed = 1337                 # Random seed

# -----------------------------------------------------------------------------
# Data streaming
# -----------------------------------------------------------------------------

data_prefer_queue = True
data_cache_files = 1
data_wait_sleep_seconds = 1.0
data_wait_timeout_seconds = None
data_stream_verbose = False

# -----------------------------------------------------------------------------
# Evaluation (disabled for GRPO)
# -----------------------------------------------------------------------------

eval_interval = 999999999   # Effectively disable evaluation
eval_iters = 0              # No evaluation iterations
eval_only = False

# -----------------------------------------------------------------------------
# Notes
# -----------------------------------------------------------------------------

# GRPO training differs from standard supervised training:
# 1. No fixed targets - generates its own training data
# 2. Uses judge model for reward signal
# 3. Optimizes for group-relative advantages
# 4. Includes KL penalty to reference policy
#
# Key hyperparameters to tune:
# - group_size: Larger = more stable but slower
# - kl_beta: Higher = more conservative, lower = more exploration
# - learning_rate: Typically 10-100x lower than pretraining
# - temperature: Controls diversity of generations

