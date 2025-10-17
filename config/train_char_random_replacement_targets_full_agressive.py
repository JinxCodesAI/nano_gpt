"""Training configuration for the char_random_replacement dataset."""

# train a character-level model for diffusion/unmasking tasks
# based on shakespeare data with random replacement corruption

out_dir = 'out-char-random-replacement'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = True
compile = True

wandb_log = True # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'random-replacement-char-full-t'

dataset = 'char_random_replacement'
data_stream_verbose = True
dataset_partial_targets = False # IF True targets of unchanged positions are set to ignore_index, this is common practice in BERT training but do not work here 
original_token_probability_multiplier = 1.0  # Increase to bias toward keeping the original token during random replacement.
train_corruption_mixture = (0.8, 0.2, 0.0)  # (random, mask token, fragment) weights used for training corruption.

composition_config = 'example'  # refers to data/char_random_replacement/config/example.py; use None if config is not defined

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        dataset,
        'config',
        f'{composition_config}.py',
    )
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

gradient_accumulation_steps = 1 # Increase if batch_size had to be reduced to keep same effective batch size
batch_size = 384  # fits on A40 with 48 GB of RVAM, adjust for other machines
block_size = 1024 # Context size for masking

learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500  

# Model architecture  
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
dtype = 'float16'

# Training type for masked language modeling
training_type = 'MLM' # Masked Language Modeling 

# Data streaming config
batches_per_file = 10  # Smaller files for faster iteration
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True
ignore_index = -100  # Default PyTorch ignore index

# For debugging on smaller machines, uncomment:
# device = 'cpu'
# compile = False
# batch_size = 4
# block_size = 128
