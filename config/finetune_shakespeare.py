import time

"""Character-level finetuning config for diffusion checkpoints.

Historically this initialized from GPT-2 weights, but the bidirectional
architecture now expects checkpoints produced by this repository.
Update `out_dir` to point at the run you want to continue training from.
"""

# Directory containing the checkpoint to resume and where new checkpoints will be written.
out_dir = 'out-shakespeare'

eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'

# Continue training from an existing diffusion checkpoint; GPT-2 weights are no longer supported.
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
