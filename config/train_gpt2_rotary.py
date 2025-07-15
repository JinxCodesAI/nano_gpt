# config for training GPT-2 (124M) with rotary embeddings
# launch as: torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_rotary.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M-rotary'

# rotary embedding settings
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 2048

# training settings (same as standard GPT-2)
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8
max_iters = 600000
lr_decay_iters = 600000

eval_interval = 1000
eval_iters = 200
log_interval = 10
weight_decay = 1e-1