# File: config/L4_dynamic_start.py
# Config for starting a dynamic training run for GPT-2 124M on a 24GB GPU.

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124M-L4-dynamic-growth'

# --- Final Target Architectural Parameters (for reference) ---
# n_layer = 12
# n_head = 12
# n_embd = 768
# n_hidden = 3072 # 4 * 768

# --- Initial Architectural State (Small Model with LoRA) ---
n_layer = 3      # Start with 3 layers
n_head = 12
n_embd = 768
n_hidden = 768   # Start with a narrow MLP (1 * n_embd)

# Enable LoRA on the embedding layer with a moderate rank
embedding_mode = 'lora'
embedding_rank = 64      # A good starting point for the large embedding matrix
attn_lora_rank = 0       # Keep attention blocks standard for now
lora_alpha = 1.0

# --- Initial Training State (Aggressive & Fast) ---
max_iters = 600000
lr_decay_iters = 600000
block_size = 1024
weight_decay = 1e-1

# Start with a larger batch size that fits the small model in 24GB VRAM
batch_size = 16
# Start with a moderate grad accum
gradient_accumulation_steps = 2 # Effective batch size = 16 * 2 = 32

# Evaluate frequently to trigger operations quickly
eval_interval = 100
eval_iters = 20
log_interval = 10

# High learning rate with a fast warmup
learning_rate = 6e-4
warmup_iters = 200

# --- Link to the Schedule ---
scaling_schedule_file = 'configs/L4_growth_schedule.json'

# --- Set default multipliers (not used for initial state, but good practice) ---
lr_multiplier = 1.0
grad_accum_multiplier = 1.0
warmup_iters_multiplier = 1.0
eval_interval_multiplier = 1.0
eval_iters_multiplier = 1.0
batch_size_multiplier = 1.0

use_rotary_embeddings = True
rotary_base = 100.0
rotary_max_position_embeddings = 128