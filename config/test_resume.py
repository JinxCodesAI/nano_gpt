# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out_resume'
eval_interval = 10
log_interval = 1
eval_iters = 5
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# file logging
log_dir = 'logs' # directory for log files
file_logging = True # enable file logging
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128
# model
n_layer = 2
n_head = 2
n_embd = 16
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
n_hidden = None # feed forward hidden dimension, defaults to 4 * n_embd if None
# rotary embeddings
use_rotary_embeddings = False
rotary_base = 10000.0
rotary_max_position_embeddings = 2048
# adamw optimizer
learning_rate = 6e-3 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 10 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Dynamic State Parameters for Training Orchestrator
# These parameters hold the current state of the dynamic configuration
# They are multipliers/divisors that modify the final values
      
# In train.py, Dynamic State Parameters section

# ... (after lora_alpha_multiplier) ...
# --- ADD THESE LINES ---
# Concrete LoRA architectural parameters. These will be overridden by config files.
embedding_mode = 'standard'
attn_lora_rank = 0 # rank for attention LoRA, 0 disables
embedding_rank = 0 # rank for embedding LoRA, 0 disables
lora_alpha = 1.0 # scaling factor for LoRA layers
attn_lora_rank_divisor = 0 # Divisor for attention LoRA rank (0 disables LoRA)
vocab_lora_rank_divisor = 0 # Divisor for embedding LoRA rank (0 disables LoRA)
lora_alpha_multiplier = 1.0 # Multiplier for LoRA alpha
n_layer_divisor = 1 # Divisor for model depth
n_hidden_divisor = 1 # Divisor for MLP width
batch_size_multiplier = 1.0 # Multiplier for batch size
grad_accum_multiplier = 1.0 # Multiplier for accumulation steps
lr_multiplier = 1.0 # Multiplier for learning rate
warmup_iters_multiplier = 1.0 # Multiplier for warmup iterations
eval_iters_multiplier = 1.0 # Multiplier for evaluation iterations
eval_interval_multiplier = 1.0 # Multiplier for evaluation frequency
# scaling schedule configuration
scaling_schedule_file = 'configs/resume_schedule.json' # Path to scaling schedule config file (YAML/JSON)