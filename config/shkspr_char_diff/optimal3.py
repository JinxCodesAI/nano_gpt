out_dir = 'out'
init_from = 'scratch' # 'scratch' or 'resume'
wandb_log = False # disabled by default
wandb_project = 'experiments_diffusion'
wandb_run_name = 'shkspr_char_diff_moderate_first' # 'run' + str(time.time())
batch_size = 16
gradient_accumulation_steps = 16
# data
dataset = 'shakespeare_char'
use_paragraph_boundaries = False # if True, start samples at paragraph boundaries (double newlines)
# diffusion training config
training_type = 'unmasking'  # 'unmasking', 'remasking', or 'remasking_binary' - type of training
use_all_stages_for_training = True
weight_loss_by_mask_ratio = False
enable_entropy_penalty = False

# For unmasking: stage-based training with direct probability control

unmasking_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]


validation_stages = [
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.5, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 8},
    {'type':'sticky','target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 8},
    {'type':'random','max_masked_ratio': 0.2, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 4},
    {'type':'sticky','target_masked_ratio': 0.4, 'p1_probability': 0.15, 'p2_probability': 0.3, 'val_loss_stale_count': 6},
    {'type':'sticky','target_masked_ratio': 0.55, 'p1_probability': 0.1, 'p2_probability': 0.6, 'val_loss_stale_count': 10},
    {'type':'sticky','target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 15},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
    {'type':'sticky','target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 20},
]

# adamw optimizer
learning_rate = 3e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
warmup_iters = 600 # how many steps to warm up for
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
weight_decay=1e-2
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # it's just experiment, no need to decay

max_entropy_penalty = 3 # loss = loss * (1 + current_entropy_penalty * wrong_answers_entropy)

entropy_penalty_start_iter = 2500 # start increasing entropy penalty after this many iterations