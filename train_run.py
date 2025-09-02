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
    estimate_loss, get_lr, load_synthetic_model, 
    TrainingContext, update_stage_progress,
    calculate_wrong_answer_entropy, get_current_entropy_penalty, update_entropy_multiplier_ema,
    apply_label_smoothing
)
from training_utils.batch_generation import get_batch

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

# transfer learning config
transfer_learning_mode = 'from_scratch'  # 'from_scratch', 'feature_extraction', 'fine_tuning'
pretrained_checkpoint_path = None  # Path to pretrained checkpoint for transfer learning
model_mode = 'language_model'  # 'language_model', 'token_classifier', or 'sequence_classifier' - target mode after loading pretrained weights

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
compile = True # use PyTorch 2.0 to compile the model to be faster
start_iter_num = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file

# Note: unmasking_stages are now defined in dataset configuration, not in training config
# The check will be done after dataset_config is loaded

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

# NEW: Simple dataset initialization using dataset interface
from training_utils.dataset_interface import DatasetConfig

# Initialize dataset configuration
dataset_config = DatasetConfig(dataset)
dataset_config.validate_training_config(block_size, batch_size)

# Validate that dataset has required stages for unmasking training
if training_type == 'unmasking':
    if not hasattr(dataset_config.training_config, 'UNMASKING_STAGES') or len(dataset_config.training_config.UNMASKING_STAGES) == 0:
        print_and_flush(f"No unmasking stages defined in dataset '{dataset}', exiting...")
        exit()
    print_and_flush(f"✓ Dataset has {len(dataset_config.training_config.UNMASKING_STAGES)} unmasking stages")

# Load fixed validation set ONCE at startup - dataset provides 'buffet', training decides consumption
print_and_flush("Loading fixed validation set...")
validation_batches = dataset_config.load_validation_set(eval_iters)
print_and_flush(f"✓ Validation set loaded with {eval_iters} samples per stage")

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
checkpoint_training_context = None  # For restoring training context state

# All dataset info now comes from dataset_config
meta_vocab_size = dataset_config.meta['vocab_size']
extended_vocab_size = dataset_config.meta['extended_vocab_size']
mask_token_id = dataset_config.meta['special_tokens']['mask_token_id']

print_and_flush(f"Dataset: {dataset}")
print_and_flush(f"Block size constraint: {dataset_config.meta['block_size']}")
print_and_flush(f"Original vocab_size: {meta_vocab_size}")
print_and_flush(f"Extended vocab_size: {extended_vocab_size}")
print_and_flush(f"Mask token ID: {mask_token_id}")
print_and_flush(f"Training stages: {dataset_config.meta['training_stages']}")
print_and_flush(f"Validation stages: {dataset_config.meta['validation_stages']}")

# NEW: Simple training context creation with dataset configuration
training_ctx = TrainingContext(
    dataset_config=dataset_config,
    # Training hyperparameters:
    training_type=training_type,
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    seed_offset=seed_offset,
    iter_num=iter_num,
    eval_iters=eval_iters,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr,
    # Training logic parameters:
    weight_loss_by_mask_ratio=weight_loss_by_mask_ratio,
    enable_entropy_penalty=enable_entropy_penalty,
    max_entropy_penalty=max_entropy_penalty,
    entropy_penalty_start_iter=entropy_penalty_start_iter,
    uncertainty_factor=uncertainty_factor,
    transfer_learning_mode=transfer_learning_mode,
    pretrained_checkpoint_path=pretrained_checkpoint_path,
    model_mode=model_mode
)

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

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type, use_rope=use_rope) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print_and_flush("Initializing a new model from scratch")
    # NEW: Use dataset configuration
    model_args['vocab_size'] = dataset_config.meta['extended_vocab_size']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Handle transfer learning case
    if pretrained_checkpoint_path is not None:
        print_and_flush("*** TRANSFER LEARNING MODE ***")
        print_and_flush(f"Loading pretrained weights from: {pretrained_checkpoint_path}")
        print_and_flush(f"Transfer learning mode: {transfer_learning_mode}")
        print_and_flush(f"Target model mode: {model_mode}")
        
        # Load pretrained checkpoint
        if not os.path.exists(pretrained_checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint file {pretrained_checkpoint_path} not found")
        
        pretrained_checkpoint = torch.load(pretrained_checkpoint_path, map_location=device, weights_only=False)
        pretrained_model_args = pretrained_checkpoint['model_args']
        
        print_and_flush(f"Pretrained model architecture:")
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            if k in pretrained_model_args:
                print_and_flush(f"  {k}: {pretrained_model_args[k]}")
        
        # Use pretrained architecture but adapt for current training needs
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
            model_args[k] = pretrained_model_args[k]
        
        # Also restore use_rope setting if it exists in checkpoint
        if 'use_rope' in pretrained_model_args:
            model_args['use_rope'] = pretrained_model_args['use_rope']
        
        # Set vocab size and mode for the new model
        model_args['vocab_size'] = training_ctx.extended_vocab_size
        model_args['mode'] = ModelMode.LANGUAGE_MODEL  # Always load as language model first
        
        # Create model with pretrained architecture
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load pretrained state dict (will have language model head)
        pretrained_state = pretrained_checkpoint['model']
        
        # Remove _orig_mod prefix if present
        unwanted_prefix = '_orig_mod.'
        for k, v in list(pretrained_state.items()):
            if k.startswith(unwanted_prefix):
                pretrained_state[k[len(unwanted_prefix):]] = pretrained_state.pop(k)
        
        print_and_flush("Loading pretrained weights (allowing head size mismatch)...")
        # Use strict=False to allow head size mismatches
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_state, strict=False)
        
        if missing_keys:
            print_and_flush(f"Missing keys (expected for vocab size changes): {len(missing_keys)} keys")
        if unexpected_keys:
            print_and_flush(f"Unexpected keys: {len(unexpected_keys)} keys")
        
        print_and_flush("✓ Pretrained weights loaded successfully")
        
        # Switch to target mode if different from language modeling
        if model_mode == 'token_classifier':
            print_and_flush("Switching to token classification mode...")
            model.switch_to_token_classification()
            print_and_flush("✓ Switched to token classification mode")
        elif model_mode == 'sequence_classifier':
            print_and_flush("Switching to sequence classification mode...")
            model.switch_to_sequence_classification()
            print_and_flush("✓ Switched to sequence classification mode")
        elif model_mode == 'language_model':
            print_and_flush("Keeping language modeling mode (no switching needed)")
        else:
            raise ValueError(f"Unknown model_mode: {model_mode}. Must be 'language_model', 'token_classifier', or 'sequence_classifier'")
        
        # Set transfer learning mode (freeze/unfreeze)
        if transfer_learning_mode == 'feature_extraction':
            print_and_flush("Setting feature extraction mode (freezing backbone)...")
            model.freeze_backbone()
            print_and_flush("✓ Backbone frozen for feature extraction")
        elif transfer_learning_mode == 'fine_tuning':
            print_and_flush("Setting fine-tuning mode (all parameters trainable)...")
            model.unfreeze_all()
            print_and_flush("✓ All parameters unfrozen for fine-tuning")
        else:  # from_scratch
            print_and_flush("Using pretrained weights with all parameters trainable")
        
        # Print parameter status
        model.print_parameter_status()
        
        # Initialize fresh training state for transfer learning
        iter_num = 0
        start_iter_num = 0
        best_val_loss = 1e9
        checkpoint_training_context = None
        
        print_and_flush("*** TRANSFER LEARNING SETUP COMPLETE ***")
        
    else:
        # Regular resume training from a checkpoint
        print_and_flush(f"Resuming unmasking training from {out_dir}")
        # resume training from a checkpoint.
        # Find the latest unmasking checkpoint file
        if ckpt_filename is None:
            import glob
            ckpt_pattern = os.path.join(out_dir, 'ckpt_*unmasking*.pt')
            ckpt_files = glob.glob(ckpt_pattern)
            if not ckpt_files:
                # Fallback to old naming convention
                ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"No unmasking checkpoint files found in {out_dir}")
            else:
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

# Transfer learning specific optimizer adjustments and logging
if pretrained_checkpoint_path is not None:
    # Transfer learning: don't load optimizer state, start fresh
    print_and_flush("*** TRANSFER LEARNING OPTIMIZER SETUP ***")
    
    trainable_params = model.get_trainable_param_count()
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print_and_flush(f"Optimizer parameter summary:")
    print_and_flush(f"  Total parameters: {total_params:,}")
    print_and_flush(f"  Trainable parameters: {trainable_params:,}")  
    print_and_flush(f"  Frozen parameters: {frozen_params:,}")
    print_and_flush(f"  Trainable percentage: {trainable_params/total_params*100:.1f}%")
    
    if transfer_learning_mode == 'feature_extraction':
        print_and_flush(f"Feature extraction mode: optimizer will only update {trainable_params:,} head parameters")
    elif transfer_learning_mode == 'fine_tuning':
        print_and_flush(f"Fine-tuning mode: optimizer will update all {trainable_params:,} parameters")
    
    # Verify optimizer only has gradients for trainable parameters
    optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    if optimizer_param_count != trainable_params:
        print_and_flush(f"⚠ WARNING: Optimizer param count ({optimizer_param_count:,}) != trainable param count ({trainable_params:,})")
    else:
        print_and_flush(f"✓ Optimizer correctly configured for {trainable_params:,} trainable parameters")
    
    print_and_flush("*** TRANSFER LEARNING OPTIMIZER READY ***")
    
elif init_from == 'resume':
    # Regular resume: load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    print_and_flush("Loaded optimizer state from checkpoint")

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
    
    # Find the latest unmasking checkpoint file
    import glob
    ckpt_pattern = os.path.join(out_dir, 'ckpt_*unmasking*.pt')
    ckpt_files = glob.glob(ckpt_pattern)
    
    if not ckpt_files:
        print("No unmasking checkpoint files found for recovery - cannot continue")
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
X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Show initial stage configuration for unmasking training
if training_ctx.training_type == 'unmasking':
    stage_config = training_ctx.get_current_stage_config()
    print(f"\n*** STAGE-BASED UNMASKING TRAINING INITIALIZED ***")
    print(f"Starting at Stage {training_ctx.current_stage}:")
    stage_type = stage_config['type']
    print(f"  Stage type: {stage_type}")
    if stage_type == 'sticky':
        print(f"  Target masked ratio: {stage_config['target_masked_ratio']}")
        print(f"  P1 probability: {stage_config['p1_probability']}")
        print(f"  P2 probability: {stage_config['p2_probability']}")
    elif stage_type == 'random':
        print(f"  Max masked ratio: {stage_config['max_masked_ratio']}")
    print(f"  Val loss stale count limit: {stage_config.get('val_loss_stale_count', 10)}")
    print(f"Total stages configured: {len(dataset_config.training_config.UNMASKING_STAGES)}")
    print("*** STAGE INITIALIZATION COMPLETE ***\n")
    
    # Validation set is already loaded from pre-generated files above
    print("✓ Using pre-generated validation set")
    
    # Training batches will be generated fresh each time from all stages when flag is enabled
    if dataset_config.training_config.USE_ALL_STAGES_FOR_TRAINING:
        print("Training will generate fresh batches from all stages each iteration")

print_and_flush("Starting training loop...")
just_recovered = False
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, training_ctx) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if (iter_num % eval_interval == 0 and master_process and not just_recovered) or eval_only:
        print_and_flush(f"\n--- Starting validation at iteration {iter_num} ---")
        with timer.time_function('validation'):
            # Update training context with current iteration
            training_ctx.iter_num = iter_num
            losses = estimate_loss(model, ctx, timer, training_ctx, dataset_config, validation_batches)

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
            stage_type = stage_config['type']
            stage_info = f"Stage {losses['current_stage']} ({stage_type}): "
            if stage_type == 'sticky':
                stage_info += f"target_ratio={stage_config['target_masked_ratio']:.1f}, p1={stage_config['p1_probability']:.1f}, p2={stage_config['p2_probability']:.1f}"
            elif stage_type == 'random':
                stage_info += f"max_ratio={stage_config['max_masked_ratio']:.1f}"
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
                ckpt_filename = f'ckpt_unmasking_{iter_num}.pt'
                    
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
                            X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
                            X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
                        X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
            X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
                    X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
                    X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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
                X, Y, mask = get_batch('train', dataset_config, training_ctx.iter_num, training_ctx.batch_size, training_ctx.block_size)
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

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"  data: {data_time:.1f}ms, grad_accum: {grad_accum_time:.1f}ms (fw: {forward_time:.1f}ms, bw: {backward_time:.1f}ms)")
        print(f"  grad_proc: {grad_proc_time:.1f}ms, optimizer: {optimizer_time:.1f}ms, param_check: {param_check_time:.1f}ms")
        print(f"  loss_proc: {loss_proc_time:.1f}ms, instability: {instability_time:.1f}ms")
        print(f"  cleanup: {cleanup_time:.1f}ms, gpu_sync: {gpu_sync_time:.1f}ms")
        print(f"  measured: {measured_total:.1f}ms, unaccounted: {unaccounted_time:.1f}ms ({unaccounted_time/total_time*100:.1f}%)")
        
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

        # Add masking statistics logging for unmasking
        stage_config = training_ctx.get_current_stage_config()
        if stage_config and iter_num % (log_interval * 10) == 0:
            mask_ratio = mask.float().mean().item()
            stage_type = stage_config['type']
            stage_info = f"Masking: stage={training_ctx.current_stage} ({stage_type}), actual_ratio={mask_ratio:.3f}"
            if stage_type == 'sticky':
                stage_info += f", target={stage_config['target_masked_ratio']:.1f}, p1={stage_config['p1_probability']:.1f}, p2={stage_config['p2_probability']:.1f}"
            elif stage_type == 'random':
                stage_info += f", max={stage_config['max_masked_ratio']:.1f}"
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