"""Configuration template for token classification"""

# Override default config for token classification
model_mode = 'token_classifier'
attention_type = 'bidirectional'  # Required for classification
num_token_classes = 2  # Adjust as needed
freeze_transformer = True  # Start with feature extraction
unfreeze_at_iteration = 5000  # Unfreeze after warmup
init_from_checkpoint = None  # Path to pretrained model (set this!)

# Classification-specific training settings
learning_rate = 5e-5  # Lower LR for fine-tuning
warmup_iters = 1000
max_iters = 20000
eval_interval = 500

# Enable compatible loss modifiers for classification
loss_modifiers_enabled = True
entropy_modifier_enabled = True  # Works well for classification
target_smoothing_enabled = True  # Label smoothing for classification
mask_ratio_weight_enabled = False  # Not typically used for classification

# Data settings for classification
batch_size = 16  # Smaller batch for fine-tuning
block_size = 512  # May be smaller for classification tasks

print("Token classifier configuration loaded")