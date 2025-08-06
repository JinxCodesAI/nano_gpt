# Shakespeare character-level diffusion model training configuration
# This config trains a small diffusion model on the Shakespeare dataset

# Model initialization and type
init_from = 'scratch'
model_type = 'diffusion'  # Use the two-token diffusion system

# I/O and output
out_dir = 'out-shakespeare-char-diffusion'
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

# WandB logging
wandb_log = False  # override via command line if you like
wandb_project = 'shakespeare-char-diffusion'  # Updated for diffusion
wandb_run_name = 'mini-diffusion-gpt'

# Data settings
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 512
block_size = 256  # context of up to 256 previous characters

# Model architecture (baby GPT model)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False  # False is better and faster

# Two-token diffusion system parameters
# Dynamic task-based loss weights (linearly interpolated over training)
weight_unmask_task_min = 1.5    # Initial weight for the "fill-in-the-blank" task (emphasize early)
weight_unmask_task_max = 1.0    # Final weight for the "fill-in-the-blank" task (balance later)
weight_remask_task_min = 0.1    # Initial weight for the "proofreading" task (lower priority early)  
weight_remask_task_max = 1.0    # Final weight for the "proofreading" task (emphasize later)

# Curriculum learning settings
penalty_mask_correct = 0.5      # Final discount for wrongly masking a correct token
masking_warmup_iters = 1000     # Iterations to ramp up the penalty_mask_correct
proofreading_warmup_iters = 2000  # Iterations to ramp up the "re-masking" task

# Data corruption settings
guaranteed_correct_factor = 0.01  # Fraction of tokens guaranteed to remain uncorrupted (1%)

# Diagnostic logging settings
log_diagnostics_interval = 100  # Log detailed diagnostics every N iterations
enable_diagnostics = True       # Master toggle for diagnostic logging

# Adaptive task weighting (experimental)
adaptive_task_weights = False   # Enable dynamic task weight adjustment
adaptive_weight_factor = 0.2    # How much to adjust weights

# Optimizer settings
learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-5  # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
weight_decay = 1e-1
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 500  # shorter warmup for small dataset

# Device settings (uncomment for specific devices)
# device = 'cpu'      # run on cpu only
# compile = False     # do not torch compile the model
# dtype = 'float32'   # use float32 for better stability on some devices