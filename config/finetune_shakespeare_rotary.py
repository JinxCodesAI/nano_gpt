import time

out_dir = 'out-shakespeare-rotary'
eval_interval = 5
eval_iters = 40
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-rotary-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl'

# enable rotary embeddings for fine-tuning
use_rotary_embeddings = True
rotary_base = 10000.0
rotary_max_position_embeddings = 1024

always_save_checkpoint = False
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20
learning_rate = 3e-5
decay_lr = False