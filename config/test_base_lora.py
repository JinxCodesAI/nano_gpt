# File: config/test_base_lora.py

# Base config for fast, small-scale testing of LoRA operations.

wandb_log = False
wandb_project = 'owt-test'
wandb_run_name='orchestrator-lora-test'

# Make the model very small for fast initialization and training
n_layer = 4
n_head = 4
n_embd = 128
n_hidden = 256
block_size = 64

# Make the training loop very short
max_iters = 3000
lr_decay_iters = 300
gradient_accumulation_steps = 4
batch_size = 8

# Evaluate very frequently to check triggers quickly
eval_interval = 50
eval_iters = 10
log_interval = 5

# For debugging, it's often better to disable compile to get clearer stack traces
compile = False

# --- LoRA Specific Settings ---
embedding_mode = 'lora'

# Calculate rank based on the small n_embd.
# For n_embd=128, a divisor of 8 gives a rank of 16.
attn_lora_rank_divisor = 8
vocab_lora_rank_divisor = 8
lora_alpha_multiplier = 1.0

# The following lines are needed to correctly initialize the model with LoRA.
# They won't be in globals() from train.py, so we add them here.
# Calculate the concrete rank values needed for GPTConfig
attn_lora_rank = n_embd // attn_lora_rank_divisor if attn_lora_rank_divisor > 0 else 0
embedding_rank = n_embd // vocab_lora_rank_divisor if vocab_lora_rank_divisor > 0 else 0
lora_alpha = 1.0 * lora_alpha_multiplier