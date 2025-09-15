"""Simple configuration for sequence scoring dataset with basic masking"""

# Dataset configuration
dataset = 'sequence_scorer'
batch_size = 16
block_size = 256

# MLM model for synthetic text generation
mlm_checkpoint_path = 'out/ckpt_MLM_0.pt'  # adjust path to trained char_diffusion MLM checkpoint
mask_probability_range = (0.1, 0.7)
cls_token_id = 0

# No stage configuration (uses simple masking)
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
batches_per_file = 50
max_backlog_files = 3
sleep_seconds = 5.0

print("Simple sequence scorer configuration loaded")

