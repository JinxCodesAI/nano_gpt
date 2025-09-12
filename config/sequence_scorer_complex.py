"""Advanced configuration for sequence scoring dataset with stage-based generation"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 16
block_size = 256

# MLM model for synthetic text generation
mlm_checkpoint_path = 'out/ckpt_MLM_0.pt'  # adjust to your MLM checkpoint
cls_token_id = 0

# Load composition configuration (same as char_diffusion)
composition_config = 'complex'  # refers to data/sequence_scorer/config/complex.py

# Load globals from composition config if present
if composition_config is not None:
    import os
    config_path = os.path.join('data', 'sequence_scorer', 'config', f'{composition_config}.py')
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

# Training settings optimized for sequence scoring
learning_rate = 1e-4
warmup_iters = 300
max_iters = 8000
eval_interval = 200

# Data generation settings
batches_per_file = 30
max_backlog_files = 2
sleep_seconds = 8.0

print("Complex sequence scorer configuration loaded")

