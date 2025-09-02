"""Training configuration for Shakespeare diffusion experiment"""

# Dataset selection (loads all dataset-specific settings)
dataset = 'shakespeare_char_diffusion'

# I/O
out_dir = 'out'
init_from = 'scratch'
wandb_log = True
wandb_project = 'experiments_diffusion'  
wandb_run_name = 'shkspr_char_diff_experiment1'

# Training hyperparameters ONLY
batch_size = 192                    # Any batch_size supported via adaptation
gradient_accumulation_steps = 12
learning_rate = 1e-3
max_iters = 8000
warmup_iters = 2000
lr_decay_iters = 8000
min_lr = 3e-5
weight_decay = 2e-2
grad_clip = 0.0
decay_lr = True

# Training-specific loss computation parameters
weight_loss_by_mask_ratio = True    # Weight loss by sqrt(1.0 / mask_ratio)
enable_entropy_penalty = False     # Apply entropy penalty to loss
max_entropy_penalty = 0.5          # Maximum entropy penalty multiplier
entropy_penalty_start_iter = 6000  # Iteration to start applying entropy penalty
uncertainty_factor = 0.1           # Label smoothing factor for loss computation

# Model architecture 
n_layer = 6
n_head = 6  
n_embd = 384
dropout = 0.2
bias = False
attention_type = 'bidirectional'
use_rope = True

# Training process
eval_interval = 200
log_interval = 20
eval_iters = 20
eval_only = False
always_save_checkpoint = True
compile = True

# Training type for diffusion
training_type = 'unmasking'  # 'unmasking', 'remasking', or 'remasking_binary' - type of training

# MOVED TO DATASET: Data generation configurations moved to data/shakespeare_char_diffusion/
# - unmasking_stages → data/shakespeare_char_diffusion/training_config.py
# - validation_stages → data/shakespeare_char_diffusion/training_config.py  
# - use_paragraph_boundaries → data/shakespeare_char_diffusion/training_config.py
# - use_all_stages_for_training → data/shakespeare_char_diffusion/training_config.py

# KEPT IN TRAINING CONFIG: Training logic parameters (control loss computation)
# - weight_loss_by_mask_ratio → stays in training config (modifies loss during training)
# - enable_entropy_penalty → stays in training config (modifies loss during training)
# - max_entropy_penalty → stays in training config (training hyperparameter)
# - entropy_penalty_start_iter → stays in training config (training schedule)
# - uncertainty_factor → stays in training config (modifies loss via label smoothing)