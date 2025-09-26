



"""Advanced configuration for sequence scoring dataset with stage-based generation"""
wandb_log = True
wandb_project = 'char-diffusion'
wandb_run_name = 'sequence_scorer_complex_epoch_1'

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 4
gradient_accumulation_steps = 8
block_size = 1024
eval_interval = 250
eval_iters = 10
log_interval = 10

# MLM model for synthetic text generation
mlm_checkpoint_path = 'out-char-diffusion/7250_1.77_pad_no_entropy.pt'  # adjust to your MLM checkpoint
init_from_checkpoint = 'out-char-diffusion/7250_1.77_pad_no_entropy.pt'
freeze_transformer = True
unfreeze_at_iteration = 500
cls_token_id = 66
max_backlog_files = 10
vocab_size = 67

#init_from = 'resume'

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
learning_rate = 1e-5
warmup_iters = 500
max_iters = 10000
lr_decay_iters = 10000
min_lr = 1e-5
beta2 = 0.99

# Model architecture - bidirectional for BERT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
position_encoding = 'rotary'
dtype = 'float16'

# Loss modifiers configuration (sequence scoring)
loss_modifiers_enabled = True
# Emphasize batches with higher residual variance
sequence_variance_enabled = True
sequence_variance_scale = 5.0         # cap (>1.0) for aggressive scaling
sequence_variance_alpha = 1.5         # growth rate for nonlinear curve
sequence_variance_eps = 1e-8

# Correlation-based scaling (Pearson)
sequence_correlation_enabled = True
sequence_correlation_alpha = 4.0
sequence_correlation_eps = 1e-8



# Data generation settings
batches_per_file = 100
max_backlog_files = 20
sleep_seconds = 1.0

# Evaluation controls for sequence_scorer
# Ensure zero-only stats are always available and deterministic
# (These do not affect val loss; they only draw extra batches for zero-only metrics.)
eval_zero_stats_min_zeros = 16
# cap the number of extra batches to avoid long evals if zeros are rare
eval_zero_stats_max_extra_batches = 100
# reset validation stream at the start of each eval for determinism
eval_reset_val_stream = True

print("Complex sequence scorer configuration loaded")

