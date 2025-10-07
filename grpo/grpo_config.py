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
generator_checkpoint = 'out-char-diffusion/7250_1.77_pad_no_entropy.pt'

# Judge model (frozen, provides reward signal)
judge_checkpoint = 'out-char-diffusion/scoring_p90_0.0128.pt'

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
# MEMORY NOTE: GRPO is memory-intensive due to:
# - 3 forward passes per micro-step (sampling, current policy, reference policy)
# - group_size multiplier (processes batch_size * group_size samples)
# - Effective batch = batch_size * group_size * gradient_accumulation_steps
#
# Memory usage comparison:
# - Standard training with batch_size=4: processes 4 samples
# - GRPO with batch_size=1, group_size=8: processes 8 samples (2x more)
# - GRPO with batch_size=2, group_size=8: processes 16 samples (4x more)
#
# Recommended settings for memory-constrained training:
# - batch_size=1-2, group_size=4-8, gradient_accumulation_steps=8-16
# - Increase gradient_accumulation_steps rather than batch_size
batch_size = 1             # Number of unique inputs per iteration
                            # Total samples processed = batch_size * group_size

gradient_accumulation_steps = 16  # Gradient accumulation (increase this for larger effective batch)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

dataset = 'char_diffusion'  # Dataset name (must have masked inputs)
block_size = 1024           # Sequence length
target_size = None          # Target size (defaults to block_size)


composition_config = 'complex' # refers to data/char_diffusion/config/complex.py  use None if config is not defined

# Initialize stage variables before loading composition config
use_all_stages_for_training = None
unmasking_stages = None
validation_stages = None

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    import sys
    config_path = os.path.join('data', 'char_diffusion', 'config', f'{composition_config}.py')
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{composition_config}_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Import all global variables from the config
        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):
                globals()[attr_name] = getattr(config_module, attr_name)
        print(f"Loaded composition config from {config_path}")
    else:
        print(f"Warning: composition config file not found at {config_path}")

# -----------------------------------------------------------------------------
# Logging and checkpointing
# -----------------------------------------------------------------------------

log_interval = 1           # Log metrics every N iterations
save_interval = 100        # Save checkpoint every N iterations
sample_interval = 0      # Generate samples for monitoring every N iterations
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
compile = True             # Use torch.compile (may cause issues with GRPO)

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

# Data streaming config (must match prepare.py expectations)
batches_per_file = 100  # Samples per file
max_backlog_files = 3   # Maximum files in queue
sleep_seconds = 1.0     # Sleep between file checks
data_stream_verbose = True  # Enable verbose logging

# Consumer config (for training)
data_prefer_queue = True
data_cache_files = 1
data_wait_sleep_seconds = 1.0
data_wait_timeout_seconds = None

# Ignore index for loss computation
ignore_index = -100

# Model mode (required by data provider)
model_mode = 'language_model'

# Enable line-aligned sequences (default for char_diffusion)
enable_line_aligned_sequences = True

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

