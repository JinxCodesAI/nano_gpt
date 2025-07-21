# Test configuration for analyzer integration
# This config creates a very small model and runs for just a few iterations to test the analyzer

# I/O
out_dir = 'out-test-analyzer'
eval_interval = 5  # Evaluate every 5 steps for quick testing
log_interval = 1
eval_iters = 2  # Just 2 evaluation iterations
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# wandb logging
wandb_log = False  # Disable for testing

# data
dataset = 'shakespeare_char'  # Use small dataset
gradient_accumulation_steps = 1
batch_size = 4  # Small batch size
block_size = 64  # Small block size

# model - very small for quick testing
n_layer = 2
n_head = 2
n_embd = 64
dropout = 0.0
bias = False
n_hidden = None

# rotary embeddings
use_rotary_embeddings = False

# optimizer
learning_rate = 1e-3
max_iters = 20  # Just 20 iterations for testing
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = False  # Disable for testing
warmup_iters = 0

# system
device = 'cpu'  # Use CPU for testing to avoid GPU issues
dtype = 'float32'
compile = False  # Disable compilation for testing

# Dynamic State Parameters - all set to 1.0 for no scaling
embedding_mode = 'standard'
lora_alpha = 1.0

# scaling schedule configuration
scaling_schedule_file = None  # No scaling for testing
