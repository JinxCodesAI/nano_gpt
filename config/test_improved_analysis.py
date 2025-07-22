# Test configuration for improved analysis features
# This config runs a short training session to validate the new async analysis features

# I/O
out_dir = 'out-test-improved-analysis'
eval_interval = 10  # Evaluate frequently to test async analysis
log_interval = 1
eval_iters = 5  # Keep evaluation quick
eval_only = False
always_save_checkpoint = False  # Don't save checkpoints for this test
init_from = 'scratch'

# file logging
log_dir = 'logs'
file_logging = True

# wandb logging - disabled for testing
wandb_log = False
wandb_project = 'test-improved-analysis'
wandb_run_name = 'test-run'

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 4  # Small batch size for quick testing
block_size = 64  # Small context for quick testing

# model - very small model for quick testing
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 1e-3
max_iters = 50  # Very short training for testing
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 5
lr_decay_iters = 40
min_lr = 1e-4

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Disable compilation for testing to avoid potential issues

# LoRA settings - enable LoRA for more interesting analysis
use_lora = True
lora_rank = 4
lora_alpha = 8
lora_dropout = 0.1
lora_targets = ['wte', 'c_attn', 'c_proj', 'c_fc']

# Scaling schedule - empty for this test
scaling_schedule = []
