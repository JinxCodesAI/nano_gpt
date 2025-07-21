# Test config for reverse universal checkpoint compatibility
# This config tests loading LoRA checkpoints into standard models

wandb_log = False
wandb_project = 'checkpoint-compatibility-test'
wandb_run_name = 'reverse-checkpoint-test'

# Same model architecture
n_layer = 2
n_head = 2
n_embd = 64
n_hidden = 128
block_size = 32

# Short training for testing
max_iters = 50
lr_decay_iters = 50
gradient_accumulation_steps = 1
batch_size = 2

# Frequent evaluation
eval_interval = 10
eval_iters = 5
log_interval = 5

# Always save checkpoints
always_save_checkpoint = True

# Use a different output directory for reverse test
out_dir = 'out-checkpoint-compatibility-reverse-test'

# Disable compile for clearer debugging
compile = False

# Use CPU for consistent testing
device = 'cpu'
dtype = 'float32'

# Use same dataset
dataset = 'shakespeare_char'

# Start with LoRA enabled
embedding_mode = 'lora'
attn_lora_rank = 8
embedding_rank = 8
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
