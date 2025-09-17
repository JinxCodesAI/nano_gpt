"""Advanced configuration for sequence scoring dataset with stage-based generation"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 4
gradient_accumulation_steps = 4
block_size = 1024
eval_interval = 250
eval_iters = 10
log_interval = 10

# MLM model for synthetic text generation
mlm_checkpoint_path = 'out/7250_1.76_all_LMod_enabled.pt'  # adjust to your MLM checkpoint
init_from_checkpoint = 'out/7250_1.76_all_LMod_enabled.pt'
freeze_transformer = True
unfreeze_at_iteration = 1000
cls_token_id = 66
max_backlog_files = 10

# Load composition configuration (same as char_diffusion)
composition_config = 'complex'  # reuses data/char_diffusion/config/complex.py

# Load globals from composition config if present
if composition_config is not None:
    import os
    config_path = os.path.join('data', 'char_diffusion', 'config', f'{composition_config}.py')
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

# Model mode configuration
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'

# BERT training typically uses lower learning rates
learning_rate = 5e-4
warmup_iters = 500
max_iters = 10000
lr_decay_iters = 10000
min_lr = 5e-5
beta2 = 0.99

# Model architecture - bidirectional for BERT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
position_encoding = 'rotary'
dtype = 'float16'

# Data generation settings
batches_per_file = 100
max_backlog_files = 20
sleep_seconds = 1.0

print("Complex sequence scorer configuration loaded")

