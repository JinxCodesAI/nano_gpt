"""Configuration template for sequence scoring"""

# Override default config for sequence scoring
model_mode = 'sequence_scorer'
attention_type = 'bidirectional'  # Required for classification
cls_token_id = 101  # [CLS] token ID from tokenizer (adjust as needed)
freeze_transformer = True  # Start with feature extraction
unfreeze_at_iteration = 3000  # Unfreeze after initial training
init_from_checkpoint = None  # Path to pretrained model (set this!)

# Sequence scoring specific settings
learning_rate = 1e-4  # Higher LR for small head
warmup_iters = 500
max_iters = 15000
eval_interval = 250

# Most loss modifiers don't apply to sequence scoring (MSE loss)
loss_modifiers_enabled = True
entropy_modifier_enabled = False  # N/A for MSE loss
target_smoothing_enabled = False  # N/A for regression
mask_ratio_weight_enabled = False  # N/A for sequence-level task

# Data settings for sequence scoring
batch_size = 32  # Can be larger for sequence scoring
block_size = 256  # May be smaller since we only need [CLS] representation

print("Sequence scorer configuration loaded")