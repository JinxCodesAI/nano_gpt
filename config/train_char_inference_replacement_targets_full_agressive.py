"""Training configuration for the char_inference_replacement dataset."""

# train a character-level model for diffusion/unmasking tasks
# using checkpoint-guided corruption instead of random replacement

out_dir = 'out-char-random-replacement'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = True
compile = True

wandb_log = True  # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'inference-replacement-char-full-t'

dataset = 'char_inference_replacement'
data_stream_verbose = True
dataset_partial_targets = False  # Full identity targets for training split.
train_corruption_mixture = (0.2, 0.2, 0.2, 0.4)  # (prediction, random, mask token, fragment) weights.

# Checkpoint inference settings
checkpoint_dir = out_dir  # provider discovers the latest checkpoint within this directory
inference_refresh_seconds = 20.0
prediction_temperature = 1.0
fallback_to_random = True
fallback_original_token_probability_multiplier = 0.0
fallback_extra_special_token_ids = None

composition_config = 'example'  # refers to data/char_random_replacement/config/example.py; use None if config is not defined

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    config_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'char_random_replacement',
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

gradient_accumulation_steps = 2  # Increase if batch_size had to be reduced
batch_size = 192  # fits on A40 with 48 GB of VRAM, adjust for other machines
block_size = 1024  # Context size for masking

learning_rate = 1e-3
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
training_type = 'MLM'  # Masked Language Modeling

# Data streaming config
batches_per_file = 10  # Smaller files for faster iteration
max_backlog_files = 3
sleep_seconds = 1.0
ignore_index = -100  # Default PyTorch ignore index

# For debugging on smaller machines, uncomment:
# device = 'cpu'
# compile = False
# batch_size = 4
# block_size = 128
