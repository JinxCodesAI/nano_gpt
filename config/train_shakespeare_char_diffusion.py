# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

init_from = 'scratch'
model_type = 'diffusion'
penalty_keep_mask = 0.25  # Penalty multiplayer when model correctly keeps `[MASK]` for unknown tokens
penalty_mask_correct= 0.5
guaranteed_correct_factor=0.2
target_unmask_rate=0.1
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
