out_dir = 'out_exp'
init_from = 'scratch' # 'scratch' or 'resume'
wandb_log = True # disabled by default
wandb_project = 'experiments_diffusion'
wandb_run_name = 'shkspr_char_diff_easy_first' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
use_paragraph_boundaries = False # if True, start samples at paragraph boundaries (double newlines)
# diffusion training config
training_type = 'unmasking'  # 'unmasking', 'remasking', or 'remasking_binary' - type of training

# For unmasking: stage-based training with direct probability control

unmasking_stages = [
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20}
]

# adamw optimizer
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 50000 # it's just experiment, no need to decay
min_lr = 1e-4 # learning_rate / 10 usually
weight_decay=1e-3
dropout = 0.01 # for pretraining 0 is good, for finetuning try 0.1+

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # it's just experiment, no need to decay