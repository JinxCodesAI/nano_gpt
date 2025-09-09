# train a character-level BERT model for diffusion/unmasking tasks
# based on shakespeare data with BERT-style masking

out_dir = 'out-char-diffusion'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'bert-char'

dataset = 'char_diffusion'

composition_config = 'complex' # refers to data/char_diffusion/config/complex.py  use None if config is not defined

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    import sys
    config_path = os.path.join('data', 'char_diffusion', 'config', f'{composition_config}.py')
    if os.path.exists(config_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{composition_config}_config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Import all global variables from the config
        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):
                globals()[attr_name] = getattr(config_module, attr_name)
        print(f"Loaded composition config from {config_path}")
    else:
        print(f"Warning: composition config file not found at {config_path}")
else:
    # Set default values when no composition config is used
    use_all_stages_for_training = None
    unmasking_stages = None
    validation_stages = None

gradient_accumulation_steps = 4
batch_size = 32  # Slightly larger batch size for BERT training
block_size = 1024 # Context size for masking

# BERT training typically uses lower learning rates
learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500  # More warmup for BERT

# Model architecture - bidirectional for BERT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
attention_type = 'bidirectional' # Critical for BERT-style training
position_encoding = 'rotary'
dtype = 'float16'

# Training type for masked language modeling
training_type = 'MLM' # Masked Language Modeling

# Diffusion/masking specific config
mask_probability = 0.15  # Standard BERT masking rate
mask_token_id = None  # Will be set from dataset meta

# Data streaming config
batches_per_file = 100  # Smaller files for faster iteration
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True
ignore_index = -100  # Default PyTorch ignore index

# For debugging on smaller machines, uncomment:
# device = 'cpu'
# compile = False
# batch_size = 4
# block_size = 128