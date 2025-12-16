"""Training configuration for the cosmopedia dataset."""

# Train a custom BPE tokenizer and model on Cosmopedia
# Configurable vocab size and tokenizer training data limit

out_dir = 'out-cosmopedia'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = True
compile = True # set to True for speedup if available

wandb_log = False # override via command line if you like
wandb_project = 'cosmopedia-gpt'
wandb_run_name = 'cosmopedia-4096'

dataset = 'cosmopedia'
data_stream_verbose = True

# Tokenizer / Data settings
vocab_size = 32768
tokenizer_train_samples = 100000 # Configurable limit for tokenizer training
bpe_dropout = 0.1 # BPE Dropout probability for training

# Training settings
gradient_accumulation_steps = 4
batch_size = 128
block_size = 512

learning_rate = 1e-3
max_iters = 100000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500

# Model architecture
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.1
dtype = 'float16'

training_type = 'MLM' 

# Corruption settings for Discrete Diffusion
original_token_probability_multiplier = 1.0  
train_corruption_mixture = (0.8, 0.2, 0.0) # (random, mask, fragment)
dataset_partial_targets = False # Full targets for training

composition_config = 'example'  # refers to data/cosmopedia/config/example.py

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
 

# Data streaming config
batches_per_file = 10 
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True
ignore_index = -100

device = 'cuda'
