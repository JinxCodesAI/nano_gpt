# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

init_from = 'scratch'
model_type = 'diffusion'

initial_mask_logit_bias = 3.5   # The starting value for our dynamic bias.
bias_update_strength = 0.01   # How quickly the bias adapts.
target_mask_prob = 0.5        # The desired output probability for [MASK] on un-masking tasks.

# Simple, task-based loss weights
weight_unmask_task = 1.0        # The base weight for the loss when the model must guess a word.
weight_remask_task = 1.0        # The base weight for the loss when the model must identify a corrupted word.

# Curriculum settings
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # Iterations for the penalty curriculum.
proofreading_warmup_iters = 2000 # NEW: Iterations to ramp up the "re-masking" task.

guaranteed_correct_factor = 0.01 

out_dir = 'out-shakespeare-char-diffusion'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 512
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 1000 # not super necessary potentially
masking_warmup_iters = 1000

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
