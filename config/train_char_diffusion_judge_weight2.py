# train a character-level BERT model for diffusion/unmasking tasks
# based on shakespeare data with BERT-style masking

out_dir = 'out-char-diffusion'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'bert-char-judge-weight-epoch-3b'

dataset = 'char_diffusion'

composition_config = 'complex'  # refers to data/char_diffusion/config/complex.py

# Load global variables from composition config if specified
if composition_config is not None:
    import os
    import importlib.util
    config_path = os.path.join('data', 'char_diffusion', 'config', f'{composition_config}.py')
    if os.path.exists(config_path):
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
    # Set default values when no composition config is used
    use_all_stages_for_training = None
    unmasking_stages = None
    validation_stages = None

gradient_accumulation_steps = 1
batch_size = 16
block_size = 1024  # Context size for masking

# BERT-style training
learning_rate = 5e-4
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99
warmup_iters = 500

# Model architecture - bidirectional for BERT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
dtype = 'float16'

# Model mode for masked language modeling (BERT-style)
model_mode = 'language_model'

# Diffusion/masking specific config
mask_probability = 0.15  # Standard BERT masking rate
mask_token_id = None  # Will be set from dataset meta

# Data streaming config
batches_per_file = 100
max_backlog_files = 3
sleep_seconds = 1.0
data_stream_verbose = True
ignore_index = -100

# Loss modifiers
loss_modifiers_enabled = True

# Keep existing modifiers from the base config
entropy_modifier_enabled = False
entropy_modifier_weight = 0.3
entropy_modifier_threshold = 0.1
entropy_modifier_eps = 1e-8
entropy_modifier_verbose = True

# Target smoothing config
target_smoothing_enabled = True
target_smoothing_factor = 0.1
# Comma-delimited string accepted by factory; adjust to exclude your [MASK] id if needed
target_smoothing_special_tokens = "65"
target_smoothing_exclude_padding = True
target_smoothing_padding_token = -100

# Mask ratio weighting (can be combined with judge-weight)
mask_ratio_weight_enabled = True
mask_ratio_weight_power = 0.5
mask_ratio_weight_min_weight = 0.1
mask_ratio_weight_max_weight = 10.0
mask_ratio_weight_eps = 1e-8

# NEW: Sequence Scoring Judge Weight Modifier
# Uses a SEQUENCE_SCORER judge checkpoint to compute wrongness and scale LM loss.
judge_weight_modifier_enabled = True
# NOTE: Adjust path if the checkpoint is not located at repo root.
judge_weight_checkpoint = "out-char-diffusion\padded_judge_0.0155.pt"
# Exponent applied to (wrongness / mask_ratio)
judge_weight_exponent = 2.0
judge_start_iter = 2000
judge_max_iter = 5000
# Clamps for the final multiplier
judge_weight_min_factor = 0.0333
judge_weight_max_factor = 30.0
judge_weight_eps = 1e-6

