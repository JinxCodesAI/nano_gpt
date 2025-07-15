# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

# rotary embedding settings
use_rotary_embeddings = True
rotary_base = 100.0
rotary_max_position_embeddings = 128

out_dir = 'out-shakespeare-char'
eval_interval = 150 # keep frequent because we'll overfit
eval_iters = 30
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.2
n_hidden = 512 # feed forward hidden dimension, defaults to 4 * n_embd = 2048

learning_rate = 4e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 4e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 20 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
