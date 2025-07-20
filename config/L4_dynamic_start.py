      
# File: config/L4_dynamic_start_final.py
# Final, robust starting config for dynamic training on a 24GB GPU.

wandb_log = True
wandb_project = 'owt'

# --- Final Target Architectural Parameters (for reference) ---
# n_layer = 12, n_head = 12, n_embd = 768, n_hidden = 3072

# --- Initial Architectural State (Tiny, Fully LoRA-enabled Model) ---
n_layer = 3
n_head = 12
n_embd = 768

# Enable LoRA on EVERYTHING with a low initial rank
embedding_mode = 'lora'
attn_lora_rank_divisor = 16 # Initial attn rank = 768 / 32 = 24
vocab_lora_rank_divisor = 16 # Initial vocab rank = 768 / 16 = 48
n_hidden_divisor = 4
lr_multiplier = 10
lora_alpha = 1.0
use_rotary_embeddings = True

# Convert divisors to concrete ranks for GPTConfig
attn_lora_rank = n_embd // attn_lora_rank_divisor if attn_lora_rank_divisor > 0 else 0
embedding_rank = n_embd // vocab_lora_rank_divisor if vocab_lora_rank_divisor > 0 else 0

# --- Initial Training State (Aggressive & Fast) ---
max_iters = 600000
lr_decay_iters = 600000
block_size = 1024
weight_decay = 1e-1

# Start with a batch size that fits the small model
batch_size = 32
gradient_accumulation_steps = 2 # Effective batch size = 16 * 2 = 32

# Evaluate frequently to check loss triggers
eval_interval = 100
eval_iters = 10
log_interval = 10

# High learning rate with a fast warmup
learning_rate = 1e-3
warmup_iters = 200

# --- Link to the Schedule ---
scaling_schedule_file = 'configs/L4_TrainGrowAdapt_schedule2.json'

    