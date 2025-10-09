"""Training config for the iterative sticky masking dataset."""

# General training setup
out_dir = 'out-char-diffusion-sticky-rounds'
eval_interval = 250
eval_iters = 50
log_interval = 10

# Checkpointing
always_save_checkpoint = False

# Weights & Biases logging
wandb_log = True
wandb_project = 'char-diffusion'
wandb_run_name = 'bert-char-sticky-rounds'

# Dataset selection
#   Uses the Shakespeare corpus with iterative sticky masking rounds
dataset = 'char_diffusion_sticky_rounds'

# Optimizer / schedule
gradient_accumulation_steps = 1
batch_size = 48
block_size = 1024

learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500

dtype = 'float16'
dropout = 0.1

# Model architecture (bidirectional attention for BERT-style MLM)
n_layer = 6
n_head = 6
n_embd = 384
attention_type = 'bidirectional'
position_encoding = 'rotary'

# Dataset- and masking-specific parameters
mask_probability = 0.15
ignore_index = -100

# Sticky masking configuration
enable_line_aligned_sequences = True
sticky_p1_probability = 0.15
sticky_p2_probability = 0.5
max_rounds = 3

# Data streaming settings
batches_per_file = 100
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True

# Loss modifiers
loss_modifiers_enabled = True
entropy_modifier_enabled = True
entropy_modifier_weight = 0.3
entropy_modifier_threshold = 0.1
entropy_modifier_eps = 1e-8
entropy_modifier_verbose = True

# Target smoothing
target_smoothing_enabled = True
target_smoothing_factor = 0.1
target_smoothing_special_tokens = "65"
target_smoothing_exclude_padding = True
target_smoothing_padding_token = -100

# Mask ratio weighting
mask_ratio_weight_enabled = True
mask_ratio_weight_power = 0.5
mask_ratio_weight_min_weight = 0.1
mask_ratio_weight_max_weight = 10.0
mask_ratio_weight_eps = 1e-8

# Debug overrides (commented out)
# device = 'cpu'
# compile = False
# batch_size = 12
# block_size = 256
