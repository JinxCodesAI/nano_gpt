# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-fast'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
scaling_schedule_file = 'configs/first_scaling_schedule.json'
#overrides

use_rotary_embeddings = True


# --- Multiplier Overrides for Initial "Slingshot" State ---

# Start with a very high LR, but for a very short time.
# Initial LR will be 10 * 6e-4 = 6e-3.
lr_multiplier = 10.0

# Start with a very short warmup period (0.05 * 2000 = 100 iters).
warmup_iters_multiplier = 1

# Start with a smaller effective batch size for more frequent updates.
# This makes the initial per-GPU grad_accum steps = 40 / 8 * 0.4 = 2.
# Initial total batch size = 12 * (2 * 8) = 192.
grad_accum_multiplier = 0.01

# Start with more frequent evaluations to closely monitor the initial volatile phase.
# Initial eval interval = 1000 * 0.1 = 100 iters.
eval_interval_multiplier = 0.04
# Use fewer eval iters initially to speed things up.
# Initial eval iters = 200 * 0.2 = 40 iters.
eval_iters_multiplier = 0.05
