"""Training configuration with encoder guidance enabled for the char_random_replacement dataset."""

# train a character-level model for diffusion/unmasking tasks
# based on shakespeare data with random replacement corruption

out_dir = 'out-char-random-replacement-enc'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = True
compile = True

wandb_log = True # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'random-replacement-char-full-t-enc'

dataset = 'char_random_replacement'
data_stream_verbose = True
dataset_partial_targets = False # keep full supervision so encoder sees clean labels where available
original_token_probability_multiplier = 0.0
train_corruption_mixture = (0.6, 0.2, 0.2)  # (random, mask token, fragment)

composition_config = 'example'

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

        for attr_name in dir(config_module):
            if not attr_name.startswith('_'):
                globals()[attr_name] = getattr(config_module, attr_name)
        print(f"Loaded composition config from {config_path}")
    else:
        print(f"Warning: composition config file not found at {config_path}")
else:
    use_all_stages_for_training = None
    unmasking_stages = None
    validation_stages = None

gradient_accumulation_steps = 1
batch_size = 8  # slightly lower for encoder overhead; adjust per hardware
block_size = 1024

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500

# Model architecture
n_layer = 12
n_head = 8
n_embd = 384
dropout = 0.1
dtype = 'float16'

# Decoder LoRA adapters
use_lora_attn = True
use_lora_mlp = True
lora_rank = 64
lora_alpha = 16.0
lora_dropout = 0.05

# Encoder guidance + LoRA (independent from decoder weights)
use_encoder_guidance = True
enc_n_layer = 12
enc_n_head = 8
enc_n_embd = 384
enc_dropout = 0.05
enc_use_lora_attn = True
enc_use_lora_mlp = True
enc_lora_rank = 64
enc_lora_alpha = 16.0
enc_lora_dropout = 0.05

# FiLM adapter rank/scale for conditioning
film_rank = 16
guidance_scale = 1.0

# Training type for masked language modeling
training_type = 'MLM'

# Data streaming config
batches_per_file = 10
max_backlog_files = 3
sleep_seconds = 1.0
ignore_index = -100

# Device/debug overrides
# device = 'cpu'
# compile = False
# batch_size = 4
# block_size = 128
