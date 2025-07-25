# Minimal test for shrunken vocabulary feature using shakespeare_char dataset
import torch

# Basic settings
wandb_log = False
eval_interval = 50
log_interval = 10
eval_iters = 5
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
device = 'cuda'

# Use shakespeare_char dataset (small and complete)
dataset = 'shakespeare_char'

# Very small model for quick testing
batch_size = 1024
block_size = 256
gradient_accumulation_steps = 1
n_layer = 1
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

# Extended training for scaling schedule
learning_rate = 1e-3
max_iters = 3500  # 6 layers * 500 iters + 500 extra = 3500
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 3000
min_lr = 1e-4

# System
dtype = 'float32'  # Use float32 for compatibility
compile = False

# LoRA disabled
embedding_mode = 'standard'
attn_lora_rank = 0
embedding_rank = 0
lora_alpha = 1.0

# Shrunken vocabulary test - use a small subset
# Shakespeare char dataset has vocab_size = 65, let's shrink to 32
#shrunken_vocab_size = 32
#vocab_remapping_file = 'data/shakespeare_remapping.pt'
RARE_TOKEN_ID = 31  # Last token in shrunken vocab

# Scaling schedule for testing layer growth
scaling_schedule_file = 'configs/test_basic_shrunken_schedule.json'

# Output
out_dir = 'out-test-basic-shrunken'
