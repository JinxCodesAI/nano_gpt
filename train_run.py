
"""
Main training script runner for diffusion training.
Uses train_utils.py for all function definitions.
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import threading
from queue import Queue

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, ModelMode
from utils import Timer, log_masking_stats
from training_utils import (
    get_batch, estimate_loss, get_lr, load_synthetic_model,
    start_prefetch, stop_prefetch, TrainingContext, UnmaskingStage, update_stage_progress,
    create_unmasking_validation_set, create_sequence_scoring_validation_set, UnmaskingStageType, StickyStageConfig, RandomStageConfig, SpanStageConfig,
    calculate_wrong_answer_entropy, get_current_entropy_penalty, update_entropy_multiplier_ema,
    apply_label_smoothing
)

torch._dynamo.config.suppress_errors = True

# Global timer instance
timer = Timer()

def print_and_flush(msg):
    """Print message and immediately flush stdout for real-time logging"""
    print(msg)
    sys.stdout.flush()

# -----------------------------------------------------------------------------
# default config values 
# I/O
out_dir = 'out'
training_type = 'unmasking'  
eval_interval = 200
log_interval = 20
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume'
ckpt_filename = '34.5_58.4_UM.pt' # Specific checkpoint to load (if not latest)
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.01 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'bidirectional' # 'causal' or 'bidirectional' - type of attention to use (bidirectional recommended for diffusion)
use_rope = True # use Rotary Position Embeddings instead of absolute position embeddings
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'diffusion'
wandb_run_name = '13k_UN_noise_0.2' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
use_paragraph_boundaries = False # if True, start samples at paragraph boundaries (double newlines)
# training mode config - support for all 3 modes
training_type = 'unmasking'  # Options: 'unmasking', 'token_classification', 'sequence_scoring', 'remasking_binary' (backward compatibility)
num_token_classes = 2  # For token classification: number of classes (flexible, not just binary)

# transfer learning config
init_from_checkpoint = ""  # Path to pretrained checkpoint for transfer learning

# dynamic unfreezing for two-stage training
unfreeze_at_iteration = 1  # Iteration to unfreeze transformer (e.g., 2000 for two-stage training)
unfreeze_lr_multiplier = 1  # Reduce learning rate when unfreezing to avoid instability

# sequence scoring config
unmasking_model_checkpoint = ""  # Path to pretrained unmasking model for sequence scoring

# unmasking training config
unmasking_stages = [] # override in config file
validation_stages = [] # override in config file
use_all_stages_for_training = False # if True, generate training batches from all stages like validation
weight_loss_by_mask_ratio = False # if True, weight loss by sqrt(1.0 / mask_ratio) to balance gradient magnitude across masking ratios
enable_entropy_penalty = False # if True, apply entropy penalty to incentivize uniform wrong answer distributions
max_entropy_penalty = 0.5 # maximum entropy penalty multiplier (penalizes concentrated wrong answers)
entropy_penalty_start_iter = 6000 # iteration to start applying entropy penalty
# label smoothing config
uncertainty_factor = 0.0 # if > 0, apply label smoothing: correct answer gets (1-u), wrong answers get u/(vocab_size-1)

# adamw optimizer
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 50000
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 41000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay=1e-3

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rat
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster - disabled due to Triton issues
start_iter_num = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file

# Auto-configure freezing: True only for sequence_scoring with unfreezing enabled
freeze_transformer = (training_type == 'sequence_scoring' and unfreeze_at_iteration is not None)

print_and_flush(f"Freeze transformer: {freeze_transformer} at {unfreeze_at_iteration}")

if len(unmasking_stages) == 0 or unmasking_stages is None:
    print_and_flush("No unmasking stages defined, exiting...")
    exit()

# Update wandb run name after configuration is loaded
if training_type == 'unmasking':
    wandb_run_name = f'{wandb_run_name}_unmasking'

config = {k: globals()[k] for k in config_keys} # will be useful for logging

# Print source code and global variables on startup
print("=" * 80)
print("SOURCE CODE:")
print("=" * 80)

import sys
import os

# Get all local Python files that are imported
local_files = set()
for module_name, module in sys.modules.items():
    if hasattr(module, '__file__') and module.__file__:
        file_path = module.__file__
        # Only include .py files in current directory (not packages/libraries)
        if file_path.endswith('.py') and os.path.dirname(file_path) == os.getcwd():
            local_files.add(os.path.basename(file_path))

# Always include the main script
local_files.add('train_run.py')

# Convert to sorted list for consistent output
local_files = sorted(local_files)

for filename in local_files:
    print(f"\n--- {filename} ---")
    try:
        with open(filename, 'r') as f:
            print(f.read())
    except FileNotFoundError:
        print(f"File {filename} not found")

print("\n" + "=" * 80)
print("GLOBAL VARIABLES:")
print("=" * 80)
for name, value in sorted(globals().items()):
    if not name.startswith('_') and not callable(value):
        print(f"{name} = {value}")

print("\n" + "=" * 80)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print_and_flush(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
checkpoint_training_context = None  # For restoring training context state

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print_and_flush(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # Set special token ID for unmasking training
    mask_token_id = meta_vocab_size
    extended_vocab_size = meta_vocab_size + 15  # Reserve 15 special tokens for future finetuning
    print_and_flush(f"mask_token_id = {mask_token_id}, extended_vocab_size = {extended_vocab_size}")
else:
    mask_token_id = 65
    extended_vocab_size = 65 + 15  # Reserve 15 special tokens

# Create training context with all parameters
# Convert unmasking_stages dict to UnmaskingStage objects
unmasking_stage_objects = None
if training_type in ['unmasking', 'sequence_scoring']:
    unmasking_stage_objects = []
    for stage in unmasking_stages:
        stage_type = stage['type']
        if stage_type == 'sticky':
            config = StickyStageConfig(
                target_masked_ratio=stage['target_masked_ratio'],
                p1_probability=stage['p1_probability'],
                p2_probability=stage['p2_probability'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        elif stage_type == 'random':
            config = RandomStageConfig(
                max_masked_ratio=stage['max_masked_ratio'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        elif stage_type == 'span':
            config = SpanStageConfig(
                spans_count=stage['spans_count'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        else:
            raise ValueError(f"Unknown stage type: {stage_type}")
        
        unmasking_stage_objects.append(UnmaskingStage(config))

# Convert validation_stages dict to UnmaskingStage objects (if different from training stages)
validation_stage_objects = None
if training_type in ['unmasking', 'sequence_scoring'] and len(validation_stages) > 0:
    validation_stage_objects = []
    for stage in validation_stages:
        stage_type = stage['type']
        if stage_type == 'sticky':
            config = StickyStageConfig(
                target_masked_ratio=stage['target_masked_ratio'],
                p1_probability=stage['p1_probability'],
                p2_probability=stage['p2_probability'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        elif stage_type == 'random':
            config = RandomStageConfig(
                max_masked_ratio=stage['max_masked_ratio'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        elif stage_type == 'span':
            config = SpanStageConfig(
                spans_count=stage['spans_count'],
                val_loss_stale_count=stage['val_loss_stale_count']
            )
        else:
            raise ValueError(f"Unknown stage type: {stage_type}")
        
        validation_stage_objects.append(UnmaskingStage(config))

training_ctx = TrainingContext(
    training_type=training_type,
    num_token_classes=num_token_classes,
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    seed_offset=seed_offset,
    data_dir=data_dir,
    meta_vocab_size=meta_vocab_size,
    mask_token_id=mask_token_id,
    extended_vocab_size=extended_vocab_size,
    iter_num=iter_num,
    unmasking_stages=unmasking_stage_objects,
    validation_stages=validation_stage_objects,
    eval_iters=eval_iters,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr,
    use_paragraph_boundaries=use_paragraph_boundaries,
    use_all_stages_for_training=use_all_stages_for_training,
    weight_loss_by_mask_ratio=weight_loss_by_mask_ratio,
    enable_entropy_penalty=enable_entropy_penalty,
    max_entropy_penalty=max_entropy_penalty,
    entropy_penalty_start_iter=entropy_penalty_start_iter,
    uncertainty_factor=uncertainty_factor,
    # transfer learning parameters
    freeze_transformer=freeze_transformer,
    init_from_checkpoint=init_from_checkpoint,
    unfreeze_at_iteration=unfreeze_at_iteration,
    unfreeze_lr_multiplier=unfreeze_lr_multiplier
)

# Load unmasking model for sequence scoring if required
if training_type == 'sequence_scoring':
    print(f"Loading unmasking model for sequence scoring from {unmasking_model_checkpoint}")
    
    # Load the unmasking model checkpoint
    unmasking_checkpoint = torch.load(unmasking_model_checkpoint, map_location=device, weights_only=False)
    unmasking_model_args = unmasking_checkpoint['model_args']
    
    # Create unmasking model with same architecture
    unmasking_gptconf = GPTConfig(**unmasking_model_args)
    unmasking_model = GPT(unmasking_gptconf)
    
    # Load the state dict
    unmasking_state_dict = unmasking_checkpoint['model']
    # Handle potential _orig_mod prefix issues
    unwanted_prefix = '_orig_mod.'
    for k, v in list(unmasking_state_dict.items()):
        if k.startswith(unwanted_prefix):
            unmasking_state_dict[k[len(unwanted_prefix):]] = unmasking_state_dict.pop(k)
    
    unmasking_model.load_state_dict(unmasking_state_dict)
    unmasking_model.to(device)
    unmasking_model.eval()  # Set to eval mode for inference
    
    # Add to training context
    training_ctx.unmasking_model = unmasking_model
    
    print(f"Unmasking model loaded successfully:")
    print(f"  - Model parameters: {unmasking_model.get_num_params()/1e6:.2f}M")
    print(f"  - Vocab size: {unmasking_model_args.get('vocab_size', 'unknown')}")
    print(f"  - Block size: {unmasking_model_args.get('block_size', 'unknown')}")
    
    # Verify compatibility
    if unmasking_model_args.get('vocab_size') != extended_vocab_size:
        print(f"WARNING: Unmasking model vocab size ({unmasking_model_args.get('vocab_size')}) != current vocab size ({extended_vocab_size})")
    
    if unmasking_model_args.get('block_size', 1024) < (block_size - 1):
        print(f"WARNING: Unmasking model block size ({unmasking_model_args.get('block_size')}) < required size ({block_size - 1})")

# Apply restored training context state if resuming from checkpoint
print(f"DEBUG: init_from='{init_from}', checkpoint_training_context={checkpoint_training_context}")
if init_from == 'resume' and checkpoint_training_context is not None:
    print("Applying restored training context state...")
    training_ctx.current_stage = checkpoint_training_context.get('current_stage', 0)
    training_ctx.val_loss_stale_count = checkpoint_training_context.get('val_loss_stale_count', 0)
    training_ctx.best_val_loss_this_stage = checkpoint_training_context.get('best_val_loss_for_stage', float('inf'))
    training_ctx.entropy_multiplier_ema = checkpoint_training_context.get('entropy_multiplier_ema', 1.0)
    print(f"Training context restored: stage={training_ctx.current_stage}, stale_count={training_ctx.val_loss_stale_count}, entropy_ema={training_ctx.entropy_multiplier_ema:.4f}")
else:
    print(f"DEBUG: NOT applying training context. init_from='{init_from}', checkpoint_training_context={checkpoint_training_context is not None}")

# mode validation and configuration
cls_token_id = None

# Add validation for token classification mode
if training_type in ['token_classification', 'remasking_binary']:
    if attention_type != 'bidirectional':
        print("WARNING: Token classification requires bidirectional attention")
        attention_type = 'bidirectional'

    print(f"Token classification mode enabled:")
    print(f"  - Attention type: {attention_type}")
    print(f"  - Number of classes: {num_token_classes}")

    # Set model mode
    model_mode = ModelMode.TOKEN_CLASSIFIER

# Add validation for sequence scoring mode
elif training_type == 'sequence_scoring':
    if attention_type != 'bidirectional':
        print("WARNING: Sequence scoring requires bidirectional attention")
        attention_type = 'bidirectional'

    # Ensure we have a CLS token ID within the reserved special token range
    cls_token_id = meta_vocab_size + 5 if meta_vocab_size is not None else 70  # First special token after mask_token_id
    print(f"Setting cls_token_id = {cls_token_id} (using reserved special token slot)")

    # Update training context with CLS token ID
    training_ctx.cls_token_id = cls_token_id

    # Validate unmasking model checkpoint for sequence scoring
    if unmasking_model_checkpoint is None:
        raise ValueError("Sequence scoring requires unmasking_model_checkpoint to be specified")
    
    if not os.path.exists(unmasking_model_checkpoint):
        raise FileNotFoundError(f"Unmasking model checkpoint not found: {unmasking_model_checkpoint}")

    print(f"Sequence scoring mode enabled:")
    print(f"  - Attention type: {attention_type}")
    print(f"  - CLS token ID: {cls_token_id}")
    print(f"  - Block size: {block_size} (includes CLS token)")
    print(f"  - Unmasking model: {unmasking_model_checkpoint}")

    # Set model mode
    model_mode = ModelMode.SEQUENCE_SCORER

# Language modeling mode (default)
elif training_type == 'unmasking':
    print(f"Language modeling mode enabled (unmasking)")
    model_mode = ModelMode.LANGUAGE_MODEL
else:
    raise ValueError(f"Unknown training type: {training_type}")

# Transfer learning validation
if init_from_checkpoint is not None and init_from_checkpoint != "":
    if not os.path.exists(init_from_checkpoint):
        raise FileNotFoundError(f"Transfer learning checkpoint not found: {init_from_checkpoint}")

    print(f"Transfer learning enabled:")
    print(f"  Checkpoint: {init_from_checkpoint}")
    print(f"  Freeze transformer: {freeze_transformer}")

    if freeze_transformer:
        print("  Mode: Feature extraction (frozen transformer + trainable head)")
    else:
        print("  Mode: Fine-tuning (all weights trainable)")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type, 
                  use_rope=use_rope, mode=model_mode, num_token_classes=num_token_classes, 
                  cls_token_id=cls_token_id, freeze_transformer=freeze_transformer,
                  init_from_checkpoint=init_from_checkpoint, unfreeze_at_iteration=unfreeze_at_iteration,
                  unfreeze_lr_multiplier=unfreeze_lr_multiplier) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print_and_flush("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args['vocab_size'] = extended_vocab_size if meta_vocab_size is not None else 65 + 15
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print_and_flush(f"Resuming {training_type} training from {out_dir}")
    # resume training from a checkpoint.
    # Find the latest checkpoint file for the current training type
    if ckpt_filename is None:
        import glob
        
        # Generate checkpoint pattern based on training type
        if training_type == 'unmasking':
            ckpt_pattern = os.path.join(out_dir, 'ckpt_unmasking_*.pt')
            fallback_pattern = os.path.join(out_dir, 'ckpt_*unmasking*.pt')  # backward compatibility
        elif training_type in ['token_classification', 'remasking_binary']:
            ckpt_pattern = os.path.join(out_dir, 'ckpt_token_classifier_*.pt')
            fallback_pattern = os.path.join(out_dir, 'ckpt_*remasking*.pt')  # backward compatibility
        elif training_type == 'sequence_scoring':
            ckpt_pattern = os.path.join(out_dir, 'ckpt_sequence_scorer_*.pt')
            fallback_pattern = None
        else:
            ckpt_pattern = os.path.join(out_dir, f'ckpt_{training_type}_*.pt')
            fallback_pattern = None
        
        ckpt_files = glob.glob(ckpt_pattern)
        
        # Try fallback pattern if no files found
        if not ckpt_files and fallback_pattern:
            ckpt_files = glob.glob(fallback_pattern)
        
        if not ckpt_files:
            # Final fallback to old naming convention
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"No {training_type} checkpoint files found in {out_dir}")
        else:
            # Extract iteration numbers and find the latest
            def extract_iter_num(filename):
                basename = os.path.basename(filename)
                # Extract number from ckpt_{type}_{XXX}.pt
                parts = basename.split('_')
                for part in parts:
                    if part.replace('.pt', '').isdigit():
                        return int(part.replace('.pt', ''))
                return 0

            latest_ckpt = max(ckpt_files, key=extract_iter_num)
            ckpt_path = latest_ckpt
        print_and_flush(f"Loading latest checkpoint: {os.path.basename(ckpt_path)}")
    else:
        ckpt_path = os.path.join(out_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    training_ctx.extended_vocab_size = checkpoint_model_args['vocab_size']
    print_and_flush(f"Checkpoint vocab size: {training_ctx.extended_vocab_size}")
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Also restore use_rope setting if it exists in checkpoint
    if 'use_rope' in checkpoint_model_args:
        model_args['use_rope'] = checkpoint_model_args['use_rope']
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    start_iter_num = iter_num
    best_val_loss = checkpoint['best_val_loss']
    
    # Restore training context state if available
    if 'training_context' in checkpoint:
        ctx_state = checkpoint['training_context']
        print_and_flush(f"Restoring training context state:")
        print_and_flush(f"  Stage: {ctx_state.get('current_stage', 0)}")
        print_and_flush(f"  Val loss stale count: {ctx_state.get('val_loss_stale_count', 0)}")
        print_and_flush(f"  Best val loss for stage: {ctx_state.get('best_val_loss_for_stage', float('inf'))}")
        
        # These will be set on the training_ctx after it's created
        checkpoint_training_context = ctx_state
    else:
        checkpoint_training_context = None

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# No synthetic model loading needed for unmasking training

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print_and_flush("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process and not eval_only:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Function to reload model and optimizer from checkpoint during training
def reload_from_checkpoint():
    """Reload model and optimizer from the latest checkpoint"""
    global model, optimizer, iter_num, best_val_loss, training_ctx, raw_model
    
    print(f"\n*** RELOADING FROM CHECKPOINT ***")
    
    # Find the latest checkpoint file for the current training type
    import glob
    
    # Generate checkpoint pattern based on training type
    if training_type == 'unmasking':
        ckpt_pattern = os.path.join(out_dir, 'ckpt_unmasking_*.pt')
        fallback_pattern = os.path.join(out_dir, 'ckpt_*unmasking*.pt')  # backward compatibility
    elif training_type in ['token_classification', 'remasking_binary']:
        ckpt_pattern = os.path.join(out_dir, 'ckpt_token_classifier_*.pt')
        fallback_pattern = os.path.join(out_dir, 'ckpt_*remasking*.pt')  # backward compatibility
    elif training_type == 'sequence_scoring':
        ckpt_pattern = os.path.join(out_dir, 'ckpt_sequence_scorer_*.pt')
        fallback_pattern = None
    else:
        ckpt_pattern = os.path.join(out_dir, f'ckpt_{training_type}_*.pt')
        fallback_pattern = None
    
    ckpt_files = glob.glob(ckpt_pattern)
    
    # Try fallback pattern if no files found
    if not ckpt_files and fallback_pattern:
        ckpt_files = glob.glob(fallback_pattern)
    
    if not ckpt_files:
        print(f"No {training_type} checkpoint files found for recovery - cannot continue")
        return False
    
    # Extract iteration numbers and find the latest
    def extract_iter_num(filename):
        basename = os.path.basename(filename)
        # Extract number from ckpt_unmasking_XXX.pt
        parts = basename.split('_')
        for part in parts:
            if part.replace('.pt', '').isdigit():
                return int(part.replace('.pt', ''))
        return 0
    
    latest_ckpt = max(ckpt_files, key=extract_iter_num)
    ckpt_path = latest_ckpt
    print(f"Reloading from checkpoint: {os.path.basename(ckpt_path)}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Reload model state - handle compiled vs non-compiled model mismatches
    model_state = checkpoint['model']
    
    # Check if current model expects _orig_mod prefix but checkpoint doesn't have it
    current_keys = set(raw_model.state_dict().keys())
    checkpoint_keys = set(model_state.keys())
    
    # Determine if we need to add or remove _orig_mod prefix
    if any(k.startswith('_orig_mod.') for k in current_keys) and not any(k.startswith('_orig_mod.') for k in checkpoint_keys):
        # Current model is compiled (has _orig_mod prefix), but checkpoint doesn't - add prefix
        print("Adding _orig_mod prefix to checkpoint keys for compiled model")
        new_state = {}
        for k, v in model_state.items():
            new_state[f'_orig_mod.{k}'] = v
        model_state = new_state
    elif not any(k.startswith('_orig_mod.') for k in current_keys) and any(k.startswith('_orig_mod.') for k in checkpoint_keys):
        # Current model is not compiled, but checkpoint has _orig_mod prefix - remove prefix
        print("Removing _orig_mod prefix from checkpoint keys for non-compiled model")
        unwanted_prefix = '_orig_mod.'
        for k, v in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix):]] = model_state.pop(k)
    
    raw_model.load_state_dict(model_state)
    
    # Reload optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update iteration and loss tracking
    # Step back iteration to avoid immediately hitting the same problematic iteration
    iter_num = checkpoint['iter_num'] - 1
    best_val_loss = checkpoint['best_val_loss']
    
    # Restore training context state if available
    if 'training_context' in checkpoint:
        ctx_state = checkpoint['training_context']
        training_ctx.current_stage = ctx_state.get('current_stage', 0)
        training_ctx.val_loss_stale_count = ctx_state.get('val_loss_stale_count', 0)
        training_ctx.best_val_loss_this_stage = ctx_state.get('best_val_loss_for_stage', float('inf'))
        training_ctx.entropy_multiplier_ema = ctx_state.get('entropy_multiplier_ema', 1.0)
        print(f"Training context restored: stage={training_ctx.current_stage}, entropy_ema={training_ctx.entropy_multiplier_ema:.4f}")
    
    
    print(f"Model and optimizer reloaded from iteration {iter_num}")
    print("*** CHECKPOINT RELOAD COMPLETE ***\n")
    return True

# training loop
X, Y, mask = get_batch('train', training_ctx) # fetch the very first batch


t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Add initialization tracking for frozen state
if freeze_transformer and hasattr(raw_model, 'get_frozen_status'):
    print(f"Initial frozen status: {raw_model.get_frozen_status()}")
    raw_model.print_parameter_status()

# Show initial stage configuration for unmasking training
if training_ctx.training_type == 'unmasking':
    stage_config = training_ctx.get_current_stage_config()
    print(f"\n*** STAGE-BASED UNMASKING TRAINING INITIALIZED ***")
    print(f"Starting at Stage {training_ctx.current_stage}:")
    stage_type = stage_config.get_stage_type()
    print(f"  Stage type: {stage_type.value}")
    if stage_type == UnmaskingStageType.STICKY:
        config = stage_config.config
        print(f"  Target masked ratio: {config.target_masked_ratio}")
        print(f"  P1 probability: {config.p1_probability}")
        print(f"  P2 probability: {config.p2_probability}")
    elif stage_type == UnmaskingStageType.RANDOM:
        config = stage_config.config
        print(f"  Max masked ratio: {config.max_masked_ratio}")
    elif stage_type == UnmaskingStageType.SPAN:
        config = stage_config.config
        print(f"  Spans count: {config.spans_count}")
    print(f"  Val loss stale count limit: {stage_config.get_val_loss_stale_count()}")
    print(f"Total stages configured: {len(training_ctx.unmasking_stages)}")
    print("*** STAGE INITIALIZATION COMPLETE ***\n")
    
    # Pre-create validation set with equal representation from all stages
    print("Pre-creating validation set...")
    create_unmasking_validation_set(training_ctx)
    
    # Training batches will be generated fresh each time from all stages when flag is enabled
    if training_ctx.use_all_stages_for_training:
        print("Training will generate fresh batches from all stages each iteration")

# Show initial configuration for sequence scoring training
elif training_ctx.training_type == 'sequence_scoring':
    print(f"Sequence scoring training initialized with {len(training_ctx.validation_stages)} validation stages")

    # Pre-create validation set using validation_stages
    create_sequence_scoring_validation_set(training_ctx)

print_and_flush("Starting training loop...")
just_recovered = False
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, training_ctx) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Dynamic unfreezing logic - add this in the training loop before validation
    if (unfreeze_at_iteration is not None and
        iter_num == unfreeze_at_iteration and
        hasattr(raw_model, 'get_frozen_status') and
        raw_model.get_frozen_status()):

        print(f"\n*** DYNAMIC UNFREEZING at iteration {iter_num} ***")
        print("Switching from feature extraction to fine-tuning mode")

        # Unfreeze transformer weights
        raw_model.unfreeze_transformer_weights()

        # Reduce learning rate to avoid instability
        if unfreeze_lr_multiplier < 1.0:
            old_lr = learning_rate
            new_lr = old_lr * unfreeze_lr_multiplier
            print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")

            # Update learning rate in optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            # Update base learning rate for scheduler
            learning_rate = new_lr

        # Print parameter status
        raw_model.print_parameter_status()

        # Log to wandb if enabled
        if wandb_log and master_process:
            wandb.log({
                "unfreezing_iteration": iter_num,
                "old_lr": old_lr if unfreeze_lr_multiplier < 1.0 else learning_rate,
                "new_lr": learning_rate,
                "lr_multiplier": unfreeze_lr_multiplier
            })

        print("*** UNFREEZING COMPLETE ***\n")

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 and master_process and not just_recovered) or eval_only:
        print_and_flush(f"\n--- Starting validation at iteration {iter_num} ---")
        with timer.time_function('validation'):
            # Update training context with current iteration
            training_ctx.iter_num = iter_num
            losses = estimate_loss(model, ctx, timer, training_ctx)

        # VALIDATION INSTABILITY DETECTION
        train_loss_finite = math.isfinite(losses['train'])
        val_loss_finite = math.isfinite(losses['val'])
        if not train_loss_finite or not val_loss_finite:
            print_and_flush(f"\n*** VALIDATION INSTABILITY at iter {iter_num} ***")
            print_and_flush(f"Train loss: {losses['train']} ({'finite' if train_loss_finite else 'NaN/Inf'})")
            print_and_flush(f"Val loss: {losses['val']} ({'finite' if val_loss_finite else 'NaN/Inf'})")
            print_and_flush("NaN detected in validation - model has become unstable")
            print_and_flush("*** TERMINATING TRAINING ***")
            break
        
        # Print basic losses
        print_and_flush(f"--- Validation complete ---")
        print_and_flush(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        
        # Print entropy penalty information if enabled
        if training_ctx.enable_entropy_penalty:
            current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
            print_and_flush(f"entropy penalty: {current_entropy_penalty:.4f}, multiplier EMA: {training_ctx.entropy_multiplier_ema:.4f}")
        
        # Print stage information for unmasking training
        if 'current_stage' in losses:
            stage_config = training_ctx.get_current_stage_config()
            stage_type = stage_config.get_stage_type()
            stage_info = f"Stage {losses['current_stage']} ({stage_type.value}): "
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                stage_info += f"target_ratio={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f}"
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                stage_info += f"max_ratio={config.max_masked_ratio:.1f}"
            elif stage_type == UnmaskingStageType.SPAN:
                config = stage_config.config
                stage_info += f"spans_count={config.spans_count}"
            stage_info += f", stale_count={losses.get('val_loss_stale_count', 0)}"
            print_and_flush(stage_info)

        # Print model vs random statistics if available
        if 'val_model_vs_random' in losses:
            print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
            print(f"  val avg correct prob: {losses['val_avg_correct_prob']:.4f} (random: {1.0/training_ctx.extended_vocab_size:.4f})")
            if 'val_signal_to_noise' in losses:
                print(f"  val signal to noise: {losses['val_signal_to_noise']:.2f} (median: {losses.get('val_signal_to_noise_median', 0.0):.2f})")
            if 'val_most_likely_accuracy' in losses:
                print(f"  Most likely guess correct P %: {losses['val_most_likely_accuracy']:.1f}%")
        
        # Update stage progress for unmasking training
        stage_advanced = update_stage_progress(training_ctx, losses['val'])
        if stage_advanced:
            print(f"Advanced to stage {training_ctx.current_stage} - validation set remains consistent across all stages")
        
        print()  # Add blank line for readability

        if wandb_log and master_process and not eval_only:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "model vs random": losses.get('val_model_vs_random', 0.0),
                "signal to noise": losses.get('val_signal_to_noise', 0.0),
                "signal to noise median": losses.get('val_signal_to_noise_median', 0.0),
                "mfu": running_mfu*100, # convert to percentage
                "masked_token_ratio": losses.get('train_masked_token_ratio', 0.0),
                "min_masked_token_ratio": losses.get('train_min_masked_token_ratio', 0.0),
                "max_masked_token_ratio": losses.get('train_max_masked_token_ratio', 0.0),
            }
            
            # Add entropy penalty to validation wandb logging if enabled
            if training_ctx.enable_entropy_penalty:
                current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
                log_dict["entropy_penalty"] = current_entropy_penalty
                log_dict["entropy_multiplier_ema"] = training_ctx.entropy_multiplier_ema
            
            # Add per-stage validation losses for unmasking training
            for stage_idx in range(len(training_ctx.validation_stages or [])):
                stage_loss_key = f'val_stage_{stage_idx}_loss'
                stage_samples_key = f'val_stage_{stage_idx}_samples'
                if stage_loss_key in losses:
                    log_dict[f'val/stage_{stage_idx}_loss'] = losses[stage_loss_key]
                    log_dict[f'val/stage_{stage_idx}_samples'] = losses[stage_samples_key]

            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                
                # Save training context state for proper resumption
                checkpoint['training_context'] = {
                    'current_stage': training_ctx.current_stage,
                    'val_loss_stale_count': training_ctx.val_loss_stale_count,
                    'best_val_loss_for_stage': training_ctx.best_val_loss_this_stage,
                    'entropy_multiplier_ema': training_ctx.entropy_multiplier_ema
                }
                # Generate checkpoint filename based on training type
                if training_type == 'unmasking':
                    ckpt_filename = f'ckpt_unmasking_{iter_num}.pt'
                elif training_type in ['token_classification', 'remasking_binary']:
                    ckpt_filename = f'ckpt_token_classifier_{iter_num}.pt'
                elif training_type == 'sequence_scoring':
                    ckpt_filename = f'ckpt_sequence_scorer_{iter_num}.pt'
                else:
                    ckpt_filename = f'ckpt_{training_type}_{iter_num}.pt'  # fallback
                    
                if start_iter_num != iter_num:
                    print(f"saving checkpoint to {out_dir}/{ckpt_filename}")
                    torch.save(checkpoint, os.path.join(out_dir, ckpt_filename))
    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    with timer.time_function('gradient_accumulation_loop'):
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                with timer.time_function('forward_pass'):
                    # Combined forward pass and loss computation for efficiency
                    logits, loss = model(X, Y)
                
                with timer.time_function('instability_detection'):
                    # TRAINING INSTABILITY DETECTION
                    if not torch.isfinite(logits).all():
                        print(f"\n*** INSTABILITY DETECTED at iter {iter_num} ***")
                        print(f"Logits contain NaN/Inf: {torch.isnan(logits).sum().item()} NaN, {torch.isinf(logits).sum().item()} Inf")
                        print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
                        print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                        if reload_from_checkpoint():
                            # Reset local state and restart iteration completely
                            local_iter_num = 0
                            running_mfu = -1.0
                            training_ctx.iter_num = iter_num
                            # Generate new batch to avoid same problematic data
                            X, Y, mask = get_batch('train', training_ctx)
                            # Reset scaler state and start fresh iteration
                            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                            optimizer.zero_grad(set_to_none=True)
                            just_recovered = True
                            t0 = time.time()
                            continue
                        else:
                            print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                            break
                    
                    if not torch.isfinite(loss):
                        print(f"\n*** LOSS INSTABILITY at iter {iter_num} ***")
                        print(f"Loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
                        print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                        if reload_from_checkpoint():
                            # Reset local state and restart iteration completely
                            local_iter_num = 0
                            running_mfu = -1.0
                            training_ctx.iter_num = iter_num
                            # Generate new batch to avoid same problematic data
                            X, Y, mask = get_batch('train', training_ctx)
                            # Reset scaler state and start fresh iteration
                            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                            optimizer.zero_grad(set_to_none=True)
                            just_recovered = True
                            t0 = time.time()
                            continue
                        else:
                            print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                            break
                
                # Apply masking for unmasking training only (most efficient path)
                if training_ctx.training_type == 'unmasking' and mask.any():
                    # Fast path: reshape once and use boolean indexing
                    # Cross-entropy handles both hard targets (indices) and soft targets (probabilities)
                    logits_reshaped = logits.view(-1, logits.size(-1))
                    mask_reshaped = mask.view(-1)
                    
                    if Y.dim() == 3:
                        # Soft targets (probability distributions)
                        targets_reshaped = Y.view(-1, Y.size(-1))
                        loss = torch.nn.functional.cross_entropy(
                            logits_reshaped[mask_reshaped], 
                            targets_reshaped[mask_reshaped], 
                            reduction='mean'
                        )
                    else:
                        # Hard targets (token indices)
                        targets_reshaped = Y.view(-1)
                        loss = torch.nn.functional.cross_entropy(
                            logits_reshaped[mask_reshaped], 
                            targets_reshaped[mask_reshaped], 
                            reduction='mean'
                        )
                    
                    # Apply mask ratio weighting if enabled
                    if training_ctx.weight_loss_by_mask_ratio:
                        mask_ratio = mask.float().mean().item()
                        if mask_ratio > 0:
                            weight = (1.0 / mask_ratio) ** 0.5  # sqrt(1.0 / mask_ratio)
                            loss = loss * weight
                else:
                    if training_ctx.training_type == 'unmasking':
                        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                
                with timer.time_function('loss_processing'):
                    # Apply entropy penalty if enabled (works for all training types)
                    if training_ctx.enable_entropy_penalty:
                        current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
                        if current_entropy_penalty > 0:
                            # Calculate entropy of wrong answer distributions
                            wrong_answer_entropy = calculate_wrong_answer_entropy(logits, Y, training_ctx.extended_vocab_size)
                            
                            # Calculate max possible entropy for wrong answers: log(vocab_size - 1)
                            max_wrong_entropy = math.log(training_ctx.extended_vocab_size - 1)
                            
                            # Penalty for LOW entropy (concentrated wrong answers)
                            # When entropy is low (bad) -> high penalty
                            # When entropy is high (good) -> low penalty  
                            entropy_penalty_factor = (max_wrong_entropy - wrong_answer_entropy) / max_wrong_entropy
                            entropy_multiplier = 1.0 + current_entropy_penalty * entropy_penalty_factor
                            loss = loss * entropy_multiplier
                            
                            # Update EMA of entropy multiplier
                            update_entropy_multiplier_ema(training_ctx, entropy_multiplier)
                        else:
                            # No penalty applied, multiplier is 1.0
                            update_entropy_multiplier_ema(training_ctx, 1.0)

                # For remasking variants, model's internal loss is already correct
                
                # UNIVERSAL: Check final loss after any training-type-specific processing
                if not torch.isfinite(loss):
                    print(f"\n*** FINAL LOSS INSTABILITY at iter {iter_num} ***")
                    print(f"Final loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
                    print(f"Training type: {training_ctx.training_type}")
                    if hasattr(mask, 'float'):  # Check if mask exists
                        print(f"Mask ratio: {mask.float().mean().item():.4f}")
                    print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                    if reload_from_checkpoint():
                        # Reset local state and restart iteration completely
                        local_iter_num = 0
                        running_mfu = -1.0
                        training_ctx.iter_num = iter_num
                        # Generate new batch to avoid same problematic data
                        X, Y, mask = get_batch('train', training_ctx)
                        # Reset scaler state and start fresh iteration
                        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                        optimizer.zero_grad(set_to_none=True)
                        just_recovered = True
                        t0 = time.time()
                        continue
                    else:
                        print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                        break
                        
                loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        with timer.time_function('data_generation'):
            # Update training context with current iteration for the next batch
            training_ctx.iter_num = iter_num
            X, Y, mask = get_batch('train', training_ctx)
            
            
            # backward pass, with gradient scaling if training in fp16
            with timer.time_function('backward_pass'):
                scaler.scale(loss).backward()
    
    # GRADIENT PROCESSING AND CLIPPING
    with timer.time_function('gradient_processing'):
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            # Monitor gradient norms before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
            # Check for true instability (NaN/Inf gradients)
            if not torch.isfinite(grad_norm):
                # At iteration 0 with lr=0, infinite gradients indicate model/loss issues
                if iter_num == 0:
                    print(f"\n*** INITIALIZATION PROBLEM at iter {iter_num} ***")
                    print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else 'Inf'}")
                    print(f"Learning rate: {lr:.6f}")
                    print("This suggests model initialization or loss computation issues")
                    
                    # Check a few key statistics
                    print("\nModel parameter stats:")
                    for name, param in list(model.named_parameters())[:3]:  # First 3 params
                        print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                        if param.grad is not None:
                            print(f"    grad: mean={param.grad.data.mean().item():.6f}, std={param.grad.data.std().item():.6f}")
                else:
                    print(f"\n*** GRADIENT INSTABILITY at iter {iter_num} ***")
                    print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else 'Inf'}")
                
                # Check individual parameter gradients
                nan_params = 0
                inf_params = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_params += 1
                        if torch.isinf(param.grad).any():
                            inf_params += 1
                print(f"Parameters with NaN gradients: {nan_params}, with Inf gradients: {inf_params}")
                print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                if reload_from_checkpoint():
                    # Reset local state and restart iteration completely
                    local_iter_num = 0
                    running_mfu = -1.0
                    training_ctx.iter_num = iter_num
                    # Generate new batch to avoid same problematic data
                    X, Y, mask = get_batch('train', training_ctx)
                    # Reset scaler state and start fresh iteration
                    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                    optimizer.zero_grad(set_to_none=True)
                    just_recovered = True
                    t0 = time.time()
                    continue
                else:
                    print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                    break
            
            # Only warn about large gradients after initial iterations (when lr > 0)
            if iter_num > 10 and grad_norm > grad_clip * 10:
                print(f"WARNING: Large gradient norm at iter {iter_num}: {grad_norm.item():.4f} (clip threshold: {grad_clip})")
        else:
            # Still check gradient norms even without clipping
            total_norm = 0.0
            nan_grads = False
            inf_grads = False
            
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    if torch.isnan(param_norm):
                        nan_grads = True
                    if torch.isinf(param_norm):
                        inf_grads = True
                    total_norm += param_norm.item() ** 2
            
            total_norm = total_norm ** (1. / 2)
            
            if nan_grads or inf_grads:
                print(f"\n*** GRADIENT INSTABILITY at iter {iter_num} (no clipping) ***")
                print(f"NaN gradients: {nan_grads}, Inf gradients: {inf_grads}")
                print(f"Total gradient norm: {total_norm:.6f}")
                print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                if reload_from_checkpoint():
                    # Reset local state and restart iteration completely
                    local_iter_num = 0
                    running_mfu = -1.0
                    training_ctx.iter_num = iter_num
                    # Generate new batch to avoid same problematic data
                    X, Y, mask = get_batch('train', training_ctx)
                    # Reset scaler state and start fresh iteration
                    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                    optimizer.zero_grad(set_to_none=True)
                    just_recovered = True
                    t0 = time.time()
                    continue
                else:
                    print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                    break
    
    # OPTIMIZER OPERATIONS  
    with timer.time_function('optimizer_operations'):
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
    
    # PARAMETER STABILITY DETECTION
    with timer.time_function('parameter_stability_check'):
        nan_params = 0
        inf_params = 0
        param_names_with_issues = []
        
        for name, param in model.named_parameters():
            if param.data is not None:
                if torch.isnan(param.data).any():
                    nan_params += 1
                    param_names_with_issues.append(f"{name}(NaN)")
                if torch.isinf(param.data).any():
                    inf_params += 1
                    param_names_with_issues.append(f"{name}(Inf)")
        
        if nan_params > 0 or inf_params > 0:
            print(f"\n*** PARAMETER INSTABILITY at iter {iter_num} ***")
            print(f"Parameters with NaN values: {nan_params}, with Inf values: {inf_params}")
            print(f"Affected parameters: {param_names_with_issues[:10]}")  # Show first 10
            if len(param_names_with_issues) > 10:
                print(f"... and {len(param_names_with_issues) - 10} more")
            print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
            if reload_from_checkpoint():
                # Reset local state and restart iteration completely
                local_iter_num = 0
                running_mfu = -1.0
                training_ctx.iter_num = iter_num
                # Generate new batch to avoid same problematic data
                X, Y, mask = get_batch('train', training_ctx)
                # Reset scaler state and start fresh iteration
                scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                optimizer.zero_grad(set_to_none=True)
                just_recovered = True
                t0 = time.time()
                continue
            else:
                print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                break
    
    # CLEANUP OPERATIONS
    with timer.time_function('cleanup_operations'):
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # GPU SYNCHRONIZATION OPERATIONS
        with timer.time_function('gpu_synchronization'):
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        # Enhanced logging with detailed timing - use recent measurements only
        data_time = timer.get_recent_average('data_generation') * 1000
        forward_time = timer.get_recent_average('forward_pass') * 1000
        loss_time = timer.get_recent_average('loss_computation') * 1000
        backward_time = timer.get_recent_average('backward_pass') * 1000
        grad_accum_time = timer.get_recent_average('gradient_accumulation_loop') * 1000
        grad_proc_time = timer.get_recent_average('gradient_processing') * 1000
        optimizer_time = timer.get_recent_average('optimizer_operations') * 1000
        param_check_time = timer.get_recent_average('parameter_stability_check') * 1000
        cleanup_time = timer.get_recent_average('cleanup_operations') * 1000
        gpu_sync_time = timer.get_recent_average('gpu_synchronization') * 1000
        loss_proc_time = timer.get_recent_average('loss_processing') * 1000
        instability_time = timer.get_recent_average('instability_detection') * 1000

        # Calculate total of measured components (avoid double-counting nested timers)
        # grad_accum_time already contains ALL nested operations: data, forward, backward, loss_proc, instability
        measured_total = grad_accum_time + grad_proc_time + optimizer_time + param_check_time + cleanup_time + gpu_sync_time
        total_time = dt * 1000
        unaccounted_time = total_time - measured_total

        # Calculate sequence scoring ratio for relative prediction error
        if training_ctx.training_type == 'sequence_scoring' and Y.dim() == 1:
            with torch.no_grad():
                # Get predictions from the model
                logits, _ = model(X, None)
                if not hasattr(model, 'sequence_head'):
                    raise RuntimeError(
                        f"Model is missing 'sequence_head' attribute for sequence_scoring training! "
                        f"This indicates a model architecture mismatch. Expected ModelMode.SEQUENCE_SCORER "
                        f"but got model with mode {getattr(model.config, 'mode', 'unknown')}. "
                        f"Check your checkpoint loading and model configuration."
                    )

                # For sequence scoring, logits IS already the final predictions (batch_size,)
                predictions = logits  # Already processed through sequence_head in model.forward()

                # Calculate absolute error: abs(target - prediction)
                absolute_errors = torch.abs(Y - predictions)
                avg_absolute_error = absolute_errors.mean().item()

                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, ratio {avg_absolute_error:.3f}")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"  data: {data_time:.1f}ms, grad_accum: {grad_accum_time:.1f}ms (fw: {forward_time:.1f}ms, bw: {backward_time:.1f}ms)")
        print(f"  grad_proc: {grad_proc_time:.1f}ms, optimizer: {optimizer_time:.1f}ms, param_check: {param_check_time:.1f}ms")
        print(f"  loss_proc: {loss_proc_time:.1f}ms, instability: {instability_time:.1f}ms")
        print(f"  cleanup: {cleanup_time:.1f}ms, gpu_sync: {gpu_sync_time:.1f}ms")
        print(f"  measured: {measured_total:.1f}ms, unaccounted: {unaccounted_time:.1f}ms ({unaccounted_time/total_time*100:.1f}%)")
        
        # Log mask statistics for unmasking only (disabled for sequence_scoring to reduce log noise)
        if training_ctx.training_type == 'unmasking' and mask is not None:
            mask_counts_per_seq = mask.sum(dim=1).cpu()
            mask_mean = mask_counts_per_seq.float().mean().item()
            mask_var = mask_counts_per_seq.float().var().item() if mask_counts_per_seq.numel() > 1 else 0.0
            mask_min = mask_counts_per_seq.min().item()
            mask_max = mask_counts_per_seq.max().item()
            print(f"  mask_counts ({training_ctx.training_type}): mean={mask_mean:.6f}, var={mask_var:.6f}, min={mask_min:.6f}, max={mask_max:.6f}")
        
        # Add entropy penalty logging if enabled
        if training_ctx.enable_entropy_penalty:
            current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
            if current_entropy_penalty > 0:
                print(f"  entropy_penalty: {current_entropy_penalty:.4f}, multiplier_ema: {training_ctx.entropy_multiplier_ema:.4f} (max: {training_ctx.max_entropy_penalty})")

        # Validation timing (when applicable)
        if iter_num % eval_interval == 0:
            val_time = timer.get_average('validation') * 1000
            val_data_time = timer.get_average('validation_data_generation') * 1000
            val_forward_time = timer.get_average('validation_forward_pass') * 1000
            val_loss_time = timer.get_average('validation_loss_computation') * 1000
            print(f"  validation: {val_time:.2f}ms (data: {val_data_time:.2f}ms, forward: {val_forward_time:.2f}ms, loss: {val_loss_time:.2f}ms)")

        # Add masking statistics logging for unmasking only (disabled for sequence_scoring to reduce log noise)
        stage_config = training_ctx.get_current_stage_config()
        if stage_config and iter_num % (log_interval * 10) == 0 and training_ctx.training_type != 'sequence_scoring':
            mask_ratio = mask.float().mean().item()
            stage_type = stage_config.get_stage_type()
            stage_info = f"Masking: stage={training_ctx.current_stage} ({stage_type.value}), actual_ratio={mask_ratio:.3f}"
            if stage_type == UnmaskingStageType.STICKY:
                config = stage_config.config
                stage_info += f", target={config.target_masked_ratio:.1f}, p1={config.p1_probability:.1f}, p2={config.p2_probability:.1f}"
            elif stage_type == UnmaskingStageType.RANDOM:
                config = stage_config.config
                stage_info += f", max={config.max_masked_ratio:.1f}"
            elif stage_type == UnmaskingStageType.SPAN:
                config = stage_config.config
                stage_info += f", spans={config.spans_count}"
            print(stage_info)
        
        if wandb_log and master_process and not eval_only:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100 # convert to percentage
            }
            
            # Add entropy penalty to wandb logging if enabled
            if training_ctx.enable_entropy_penalty:
                current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
                log_dict["entropy_penalty"] = current_entropy_penalty
                log_dict["entropy_multiplier_ema"] = training_ctx.entropy_multiplier_ema

            wandb.log(log_dict)
    iter_num += 1
    local_iter_num += 1
    just_recovered = False  # Reset recovery flag after successful iteration

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Cleanup prefetch thread
stop_prefetch()
