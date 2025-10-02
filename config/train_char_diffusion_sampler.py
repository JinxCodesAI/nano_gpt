# train a character-level BERT model with sampler head for coherent parallel generation
# based on shakespeare data with BERT-style masking

out_dir = 'out-char-diffusion'
eval_interval = 250
eval_iters = 50
log_interval = 10

# save checkpoints when validation improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'char-diffusion'
wandb_run_name = 'bert-char-sampler-test'

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
batch_size = 8  # Slightly larger batch size for BERT training
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
attention_type = 'bidirectional' # Critical for BERT-style training and REQUIRED for sampler
position_encoding = 'rotary'
dtype = 'float16'

# Model mode for masked language modeling (BERT-style)
model_mode = 'language_model'

# Diffusion/masking specific config
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

# ============================================================================
# SAMPLER HEAD CONFIGURATION (NEW)
# ============================================================================
# The sampler head provides coherent parallel token generation by conditioning
# each prediction on its immediate left and right neighbors using wavefront filling.
# This is an auxiliary network trained separately from the main transformer.

add_sampler_head = True
start_sampler_iteration = 1000  # Start training sampler after 1000 iterations (Stage 2)
sampler_min_neighbors_ratio = 0.01  # Bootstrap 1% of tokens when no neighbors available

# ============================================================================
# CRITIC HEAD CONFIGURATION
# ============================================================================
# The critic head evaluates the quality of generated tokens.
# When both sampler and critic are enabled, we have three-stage training:
#   Stage 1 (0-1000): Main model only
#   Stage 2 (1000-3000): Main model + Sampler
#   Stage 3 (3000+): Main model + Sampler + Critic

add_critic_head = True
start_critic_iteration = 3000  # Start critic after sampler is trained (Stage 3)
end_critic_iteration = 5000
critic_alpha = 1.0  # Weight for critic loss
critic_target_scope = 'masked_and_ignore'  # 'masked_only' or 'masked_and_ignore'

# ============================================================================
# LOSS MODIFIERS
# ============================================================================
loss_modifiers_enabled = True

# Entropy modifier - disabled for this test
entropy_modifier_enabled = False
entropy_modifier_weight = 0.3
entropy_modifier_threshold = 0.1 
entropy_modifier_eps = 1e-8
entropy_modifier_verbose = True

# Target smoothing config
target_smoothing_enabled = True
target_smoothing_factor = 0.1                    # Smoothing strength (0.0 = no smoothing)
target_smoothing_special_tokens = "65"           # Comma delimited Token IDs to exclude from smoothing
target_smoothing_exclude_padding = True          # Exclude padding from loss
target_smoothing_padding_token = -100            # Padding token ID

# Mask ratio weight modifier
mask_ratio_weight_enabled = True
mask_ratio_weight_power = 0.5        # Square root inverse weighting
mask_ratio_weight_min_weight = 0.1   # Prevent extreme deweighting
mask_ratio_weight_max_weight = 10.0  # Prevent extreme upweighting  
mask_ratio_weight_eps = 1e-8         # Numerical stability

# ============================================================================
# TRAINING SCHEDULE SUMMARY
# ============================================================================
# Iteration 0-1000:     Main model only (Stage 1)
#                       - Train base language model
#                       - Loss = LM loss + modifiers
#
# Iteration 1000-3000:  Main model + Sampler (Stage 2)
#                       - Continue training main model
#                       - Start training sampler head (auxiliary)
#                       - Loss = LM loss + Sampler loss + modifiers
#
# Iteration 3000-5000:  Main model + Sampler + Critic warmup (Stage 3a)
#                       - Continue training main + sampler
#                       - Critic alpha increases linearly from 0 to 1.0
#                       - Loss = LM loss + Sampler loss + (alpha * Critic loss) + modifiers
#
# Iteration 5000+:      Main model + Sampler + Critic full (Stage 3b)
#                       - All components active
#                       - Loss = LM loss + Sampler loss + Critic loss + modifiers
# ============================================================================

