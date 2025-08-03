# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
out_dir = 'out-shakespeare-char'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 50
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 256 # context of up to 256 previous characters

# enhanced data augmentation settings
# uncomment to experiment with enhanced data to reduce overfitting
enhanced_data_probability = 0.2 # 50% enhanced data
min_prefix_length = 20 # shorter prefixes for character-level
max_prefix_length = 40 # shorter prefixes for character-level  
enhanced_generation_temperature = 0.3 # slightly more creative
enhanced_buffer_size = 512 # smaller buffer for this small dataset
enhanced_generation_batch_size =64


# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
