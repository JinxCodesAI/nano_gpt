# Test config for universal checkpoint compatibility
# This config tests loading standard checkpoints into LoRA models and vice versa

wandb_log = False
wandb_project = 'checkpoint-compatibility-test'
wandb_run_name = 'universal-checkpoint-test'

# Make the model very small for fast testing
n_layer = 2
n_head = 2
n_embd = 64
n_hidden = 128
block_size = 32

# Very short training for quick testing
max_iters = 50
lr_decay_iters = 50
gradient_accumulation_steps = 1
batch_size = 2

# Frequent evaluation to trigger checkpoint saving
eval_interval = 10
eval_iters = 5
log_interval = 5

# Always save checkpoints for testing
always_save_checkpoint = True

# Use a specific output directory for this test
out_dir = 'out-checkpoint-compatibility-test'

# Disable compile for clearer debugging
compile = False

# Use CPU for consistent testing
device = 'cpu'
dtype = 'float32'

# Use small dataset for quick testing
dataset = 'shakespeare_char'

# Standard model configuration (no LoRA initially)
embedding_mode = 'standard'
attn_lora_rank = 0
embedding_rank = 0
lora_alpha = 1.0

# Learning parameters
learning_rate = 1e-3
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 5
min_lr = 1e-4
