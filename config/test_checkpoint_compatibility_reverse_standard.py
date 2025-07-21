# Test config for reverse universal checkpoint compatibility with standard model
# This config tests resuming from LoRA checkpoints with standard model

wandb_log = False
wandb_project = 'checkpoint-compatibility-test'
wandb_run_name = 'reverse-checkpoint-standard-test'

# Same model architecture
n_layer = 2
n_head = 2
n_embd = 64
n_hidden = 128
block_size = 32

# Short training for testing resume functionality
max_iters = 100
lr_decay_iters = 100
gradient_accumulation_steps = 1
batch_size = 2

# Frequent evaluation
eval_interval = 10
eval_iters = 5
log_interval = 5

# Always save checkpoints
always_save_checkpoint = True

# Use the same output directory to test resume functionality
out_dir = 'out-checkpoint-compatibility-reverse-test'

# Resume from the LoRA checkpoint
init_from = 'resume'

# Disable compile for clearer debugging
compile = False

# Use CPU for consistent testing
device = 'cpu'
dtype = 'float32'

# Use same dataset
dataset = 'shakespeare_char'

# Switch to standard model (no LoRA)
embedding_mode = 'standard'
attn_lora_rank = 0
embedding_rank = 0
lora_alpha = 1.0

# Same learning parameters
learning_rate = 1e-3
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay
decay_lr = True
warmup_iters = 5
min_lr = 1e-4
