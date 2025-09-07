# train a character-level BERT model for diffusion/unmasking tasks
# based on shakespeare data with BERT-style masking

out_dir = 'out-char-diffusion'
eval_interval = 250
eval_iters = 200
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'bert-char'

dataset = 'char_diffusion'

gradient_accumulation_steps = 1
batch_size = 16  # Slightly larger batch size for BERT training
block_size = 512 # Context size for masking

# BERT training typically uses lower learning rates
learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500  # More warmup for BERT

# Model architecture - bidirectional for BERT
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
attention_type = 'bidirectional' # Critical for BERT-style training
position_encoding = 'absolute'

# Training type for masked language modeling
training_type = 'MLM' # Masked Language Modeling

# Diffusion/masking specific config
mask_probability = 0.15  # Standard BERT masking rate
mask_token_id = None  # Will be set from dataset meta

# Data streaming config
batches_per_file = 50  # Smaller files for faster iteration
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True

# For debugging on smaller machines, uncomment:
# device = 'cpu'
# compile = False
# batch_size = 4
# block_size = 128