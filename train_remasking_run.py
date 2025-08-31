"""
Remasking training script runner for binary classification of tokens to remask.
Uses train_utils.py for all function definitions and focuses specifically on remasking training.
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

from model import GPTConfig, GPT
from utils import Timer, log_masking_stats
from train_utils import (
    get_batch, estimate_loss, get_lr, 
    start_prefetch, stop_prefetch, TrainingContext, create_remasking_validation_set
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
out_dir = 'out_remasking'
training_type = 'remasking_binary'  # Focus on remasking binary classification
eval_interval = 200
log_interval = 20
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.01 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'bidirectional' # 'causal' or 'bidirectional' - type of attention to use (bidirectional recommended for diffusion)
use_rope = True # use Rotary Position Embeddings instead of absolute position embeddings
binary_classification = True # Use binary classification head (2 outputs instead of full vocab)

# wandb logging
wandb_log = True # disabled by default
wandb_project = 'diffusion_remasking'
wandb_run_name = 'remasking_binary' # 'run' + str(time.time())

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 16 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
use_paragraph_boundaries = False # if True, start samples at paragraph boundaries (double newlines)

# remasking configuration - random and sticky corruption supported
guaranteed_unmasked_max = 0.9  # At start: 10% of tokens corrupted
guaranteed_unmasked_min = 0.6   # At end: 40% of tokens corrupted
random_mask_warmup = 7000  # Warmup iterations for mask probability
sticky_transition_start = 1000  # Start of transition (unused but needed for compatibility)  
sticky_transition_end = 6000    # End of transition (unused but needed for compatibility)
p1_p2_ratio = 1.0  # If 1.0: random masking, if != 1.0: sticky masking with p1/p2 ratio

# adamw optimizer
learning_rate = 1e-5 # Lower learning rate for binary classification to prevent gradient explosion
max_iters = 10000
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 41000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay=1e-3

grad_clip = 100.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rat
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file

# Update wandb run name after configuration is loaded
wandb_run_name = f'{wandb_run_name}_binary_remasking'

config = {k: globals()[k] for k in config_keys} # will be useful for logging

# Print source code and global variables on startup
print("=" * 80)
print("REMASKING BINARY CLASSIFICATION TRAINING")
print("=" * 80)

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

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print_and_flush(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # For binary classification, we only need mask_token_id and don't need the other special tokens
    # But we reserve 15 special tokens for future finetuning consistency 
    mask_token_id = meta_vocab_size
    extended_vocab_size = meta_vocab_size + 15  # Reserve 15 special tokens for future finetuning
    print_and_flush(f"mask_token_id = {mask_token_id}, extended_vocab_size = {extended_vocab_size}")
else:
    print_and_flush("No meta.pkl found, using default GPT-2 vocab")
    mask_token_id = 50304
    extended_vocab_size = 50304 + 15  # Reserve 15 special tokens

# Create training context with remasking parameters
training_ctx = TrainingContext(
    training_type=training_type,
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    seed_offset=seed_offset,
    data_dir=data_dir,
    guaranteed_unmasked_max=guaranteed_unmasked_max,
    guaranteed_unmasked_min=guaranteed_unmasked_min,
    random_mask_warmup=random_mask_warmup,
    sticky_transition_start=sticky_transition_start,
    sticky_transition_end=sticky_transition_end,
    meta_vocab_size=meta_vocab_size,
    mask_token_id=mask_token_id,
    wrong_token_id=mask_token_id + 1,  # For remasking: corrupted positions
    remask_good_id=0,  # Binary: 0 = keep token
    remask_wrong_id=1,  # Binary: 1 = remask token  
    extended_vocab_size=extended_vocab_size,  # Only mask token needed for input
    iter_num=iter_num,
    eval_iters=eval_iters,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr,
    use_paragraph_boundaries=use_paragraph_boundaries,
    p1_p2_ratio=p1_p2_ratio  # For sticky masking configuration
)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type, 
                  use_rope=use_rope, binary_classification=binary_classification)
if init_from == 'scratch':
    # init a new model from scratch
    print_and_flush("Initializing a new binary classification model from scratch")
    model_args['vocab_size'] = extended_vocab_size if meta_vocab_size is not None else 50304 + 15
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print_and_flush(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    # Find the latest checkpoint file
    import glob
    ckpt_pattern = os.path.join(out_dir, 'ckpt_*remasking*.pt')
    ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        # Fallback to old naming convention
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No remasking checkpoint files found in {out_dir}")
    else:
        # Extract iteration numbers and find the latest
        def extract_iter_num(filename):
            basename = os.path.basename(filename)
            # Extract number from ckpt_remasking_XXX.pt
            parts = basename.split('_')
            for part in parts:
                if part.replace('.pt', '').isdigit():
                    return int(part.replace('.pt', ''))
            return 0

        latest_ckpt = max(ckpt_files, key=extract_iter_num)
        ckpt_path = latest_ckpt
        print_and_flush(f"Loading latest remasking checkpoint: {os.path.basename(ckpt_path)}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Also restore use_rope and binary_classification settings if they exist in checkpoint
    if 'use_rope' in checkpoint_model_args:
        model_args['use_rope'] = checkpoint_model_args['use_rope']
    if 'binary_classification' in checkpoint_model_args:
        model_args['binary_classification'] = checkpoint_model_args['binary_classification']
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# No synthetic model needed for remasking (supports both random and sticky masking)

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
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Remove the custom batch generation function - use standard get_batch instead

# Function to reload model and optimizer from checkpoint during training
def reload_from_checkpoint():
    """Reload model and optimizer from the latest checkpoint"""
    global model, optimizer, iter_num, best_val_loss, training_ctx, raw_model
    
    print(f"\n*** RELOADING FROM CHECKPOINT ***")
    
    # Find the latest checkpoint file
    import glob
    ckpt_pattern = os.path.join(out_dir, 'ckpt_*remasking*.pt')
    ckpt_files = glob.glob(ckpt_pattern)
    
    if not ckpt_files:
        print("No remasking checkpoint files found for recovery - cannot continue")
        return False
    
    # Extract iteration numbers and find the latest
    def extract_iter_num(filename):
        basename = os.path.basename(filename)
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
    
    # Reload model state
    model_state = checkpoint['model']
    
    # Handle compiled vs non-compiled model mismatches
    current_keys = set(raw_model.state_dict().keys())
    checkpoint_keys = set(model_state.keys())
    
    if any(k.startswith('_orig_mod.') for k in current_keys) and not any(k.startswith('_orig_mod.') for k in checkpoint_keys):
        print("Adding _orig_mod prefix to checkpoint keys for compiled model")
        new_state = {}
        for k, v in model_state.items():
            new_state[f'_orig_mod.{k}'] = v
        model_state = new_state
    elif not any(k.startswith('_orig_mod.') for k in current_keys) and any(k.startswith('_orig_mod.') for k in checkpoint_keys):
        print("Removing _orig_mod prefix from checkpoint keys for non-compiled model")
        unwanted_prefix = '_orig_mod.'
        for k, v in list(model_state.items()):
            if k.startswith(unwanted_prefix):
                model_state[k[len(unwanted_prefix):]] = model_state.pop(k)
    
    raw_model.load_state_dict(model_state)
    
    # Reload optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Update iteration and loss tracking
    iter_num = checkpoint['iter_num'] - 1
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Model and optimizer reloaded from iteration {iter_num}")
    print("*** CHECKPOINT RELOAD COMPLETE ***\n")
    return True

# training loop - use standard batch generation
X, Y, mask = get_batch('train', training_ctx) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Pre-create validation set for consistent validation
print("Pre-creating validation set...")
create_remasking_validation_set(training_ctx, force_recreate=True)

print_and_flush("Starting binary remasking training loop...")
print_and_flush(f"Binary classification mode: {model.config.binary_classification}")
print_and_flush(f"LM head output size: {model.lm_head.out_features}")
print_and_flush(f"Masking strategy: {'sticky' if p1_p2_ratio != 1.0 else 'random'} (p1_p2_ratio={p1_p2_ratio})")
if p1_p2_ratio != 1.0:
    print_and_flush(f"Sticky masking will use dynamic p1/p2 probabilities based on corruption schedule")

just_recovered = False
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, training_ctx) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and not just_recovered:
        print_and_flush(f"\n--- Starting validation at iteration {iter_num} ---")
        with timer.time_function('validation'):
            # Update training context with current iteration
            training_ctx.iter_num = iter_num
            # For binary classification validation, we need to modify estimate_loss
            # For now, use the existing estimate_loss which should work with binary targets
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
        
        # Print model vs random statistics if available
        if 'val_model_vs_random' in losses:
            print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
            print(f"  val accuracy: {losses['val_avg_correct_prob']:.4f}")
            if 'val_most_likely_accuracy' in losses:
                print(f"  Binary classification accuracy: {losses['val_most_likely_accuracy']:.1f}%")
        
        print()  # Add blank line for readability

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "val/model_vs_random": losses.get('val_model_vs_random', 0.0),
                "val/avg_correct_prob": losses.get('val_avg_correct_prob', 0.0),
                "mfu": running_mfu*100, # convert to percentage
                "masked_token_ratio": losses.get('train_masked_token_ratio', 0.0),
                "val/binary_accuracy": losses.get('val_most_likely_accuracy', 0.0),
            }
            
            # Add detailed per-class validation metrics for remasking_binary
            if training_ctx.training_type == 'remasking_binary':
                log_dict.update({
                    "val/corruption_ratio": losses.get('val_corruption_ratio', 0.0),
                    "val/random_baseline": losses.get('val_random_baseline', 0.0),
                    "val/accuracy_no_mask": losses.get('val_accuracy_no_mask', 0.0),
                    "val/accuracy_mask": losses.get('val_accuracy_mask', 0.0),
                    "val/class_dist_no_mask": losses.get('val_class_dist_no_mask', 0.0),
                    "val/class_dist_mask": losses.get('val_class_dist_mask', 0.0),
                    "val/avg_prob_right_p0": losses.get('val_avg_prob_right_p0', 0.0),
                    "val/avg_prob_right_p1": losses.get('val_avg_prob_right_p1', 0.0),
                })
            
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
                
                ckpt_filename = f'ckpt_remasking_binary_{iter_num}.pt'
                print(f"saving checkpoint to {out_dir}/{ckpt_filename}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_filename))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            with timer.time_function('forward_pass'):
                # Binary classification forward pass
                logits, loss = model(X, Y)
                
                # TRAINING INSTABILITY DETECTION
                if not torch.isfinite(logits).all():
                    print(f"\n*** INSTABILITY DETECTED at iter {iter_num} ***")
                    print(f"Logits contain NaN/Inf: {torch.isnan(logits).sum().item()} NaN, {torch.isinf(logits).sum().item()} Inf")
                    print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
                    print("*** ATTEMPTING RECOVERY FROM CHECKPOINT ***")
                    if reload_from_checkpoint():
                        local_iter_num = 0
                        running_mfu = -1.0
                        training_ctx.iter_num = iter_num
                        X, Y, mask = get_batch('train', training_ctx)
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
                        local_iter_num = 0
                        running_mfu = -1.0
                        training_ctx.iter_num = iter_num
                        X, Y, mask = get_batch('train', training_ctx)
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
    
    # GRADIENT INSTABILITY DETECTION
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        if not torch.isfinite(grad_norm):
            if iter_num == 0:
                print(f"\n*** INITIALIZATION PROBLEM at iter {iter_num} ***")
                print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else grad_norm.item()}")
                print(f"Learning rate: {lr:.6f}")
                print("This suggests model initialization or loss computation issues")
                
                print("\nModel parameter stats:")
                for name, param in list(model.named_parameters())[:3]:  # First 3 params
                    print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                    if param.grad is not None:
                        print(f"    grad: mean={param.grad.data.mean().item():.6f}, std={param.grad.data.std().item():.6f}")
            else:
                print(f"\n*** GRADIENT INSTABILITY at iter {iter_num} ***")
                print(f"Gradient norm is {grad_norm.item()}: {'NaN' if torch.isnan(grad_norm) else grad_norm.item()}")
            
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
                local_iter_num = 0
                running_mfu = -1.0
                training_ctx.iter_num = iter_num
                X, Y, mask = get_batch('train', training_ctx)
                scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                optimizer.zero_grad(set_to_none=True)
                just_recovered = True
                t0 = time.time()
                continue
            else:
                print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                break
        
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
                local_iter_num = 0
                running_mfu = -1.0
                training_ctx.iter_num = iter_num
                X, Y, mask = get_batch('train', training_ctx)
                scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
                optimizer.zero_grad(set_to_none=True)
                just_recovered = True
                t0 = time.time()
                continue
            else:
                print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
                break
                
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    
    # PARAMETER STABILITY DETECTION
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
            local_iter_num = 0
            running_mfu = -1.0
            training_ctx.iter_num = iter_num
            X, Y, mask = get_batch('train', training_ctx)
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
            optimizer.zero_grad(set_to_none=True)
            just_recovered = True
            t0 = time.time()
            continue
        else:
            print("*** RECOVERY FAILED - TERMINATING TRAINING ***")
            break
    
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        # Enhanced logging with detailed timing
        data_time = timer.get_average('data_generation') * 1000
        forward_time = timer.get_average('forward_pass') * 1000
        backward_time = timer.get_average('backward_pass') * 1000

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"  data: {data_time:.2f}ms, forward: {forward_time:.2f}ms, backward: {backward_time:.2f}ms")

        # Validation timing (when applicable)
        if iter_num % eval_interval == 0:
            val_time = timer.get_average('validation') * 1000
            val_data_time = timer.get_average('validation_data_generation') * 1000
            val_forward_time = timer.get_average('validation_forward_pass') * 1000
            val_loss_time = timer.get_average('validation_loss_computation') * 1000
            print(f"  validation: {val_time:.2f}ms (data: {val_data_time:.2f}ms, forward: {val_forward_time:.2f}ms, loss: {val_loss_time:.2f}ms)")

        # Add masking statistics and accuracy logging for remasking
        log_masking_stats(mask, iter_num, log_interval)
        
        # Calculate and log per-class accuracy for remasking_binary every log_interval
        if training_ctx.training_type == 'remasking_binary':
            with torch.no_grad():
                # Get predictions from current batch
                logits, _ = model(X, Y)
                
                # Calculate average probabilities for correct predictions by class
                probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, 2)
                probs_flat = probs.view(-1, 2)  # (batch_size * seq_len, 2)
                targets_flat = Y.view(-1)  # (batch_size * seq_len,)
                
                # Get probabilities for correct predictions separated by class
                class_0_mask = (targets_flat == 0)  # no-mask targets
                class_1_mask = (targets_flat == 1)  # mask targets
                
                # Average probability of correct answer for class 0 positions
                if class_0_mask.sum() > 0:
                    class_0_correct_probs = probs_flat[class_0_mask, 0]  # P(class=0) where target=0
                    avg_p_right_p0 = class_0_correct_probs.mean().item()
                else:
                    avg_p_right_p0 = 0.0
                
                # Average probability of correct answer for class 1 positions  
                if class_1_mask.sum() > 0:
                    class_1_correct_probs = probs_flat[class_1_mask, 1]  # P(class=1) where target=1
                    avg_p_right_p1 = class_1_correct_probs.mean().item()
                else:
                    avg_p_right_p1 = 0.0
                
                
                predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
                predictions_flat = predictions.view(-1)
                targets_flat = Y.view(-1)
                
                # Calculate relative prediction bias vs actual class distribution
                pred_0_count = (predictions_flat == 0).sum().item()
                pred_1_count = (predictions_flat == 1).sum().item()
                total_preds = predictions_flat.numel()
                pred_0_pct = (pred_0_count / total_preds) * 100
                pred_1_pct = (pred_1_count / total_preds) * 100
                
                # Calculate relative bias: prediction_rate / actual_rate
                pred_bias_0 = pred_0_pct / class_0_pct if class_0_pct > 0 else 0.0
                pred_bias_1 = pred_1_pct / class_1_pct if class_1_pct > 0 else 0.0
                
                # Calculate per-class accuracy
                class_0_mask = (targets_flat == 0)  # no-mask targets
                class_1_mask = (targets_flat == 1)  # mask targets
                
                if class_0_mask.sum() > 0:
                    class_0_correct = ((predictions_flat == 0) & class_0_mask).sum().item()
                    class_0_total = class_0_mask.sum().item()
                    class_0_acc = (class_0_correct / class_0_total) * 100
                else:
                    class_0_acc = 0.0
                
                if class_1_mask.sum() > 0:
                    class_1_correct = ((predictions_flat == 1) & class_1_mask).sum().item()
                    class_1_total = class_1_mask.sum().item()
                    class_1_acc = (class_1_correct / class_1_total) * 100
                else:
                    class_1_acc = 0.0
                
                # Overall accuracy
                total_correct = (predictions_flat == targets_flat).sum().item()
                total_samples = targets_flat.numel()
                overall_acc = (total_correct / total_samples) * 100
                
                # Class distribution
                class_0_pct = (class_0_total / total_samples) * 100
                class_1_pct = (class_1_total / total_samples) * 100
                
                print(f"  Training batch accuracy: no-mask {class_0_acc:.1f}%, mask {class_1_acc:.1f}%, overall {overall_acc:.1f}%")
                print(f"  Training class dist: no-mask {class_0_pct:.1f}%, mask {class_1_pct:.1f}%")
                print(f"  Training corruption rate: {class_1_pct:.1f}% (target progression: 10% -> 40%)")
                print(f"  Model probabilities: avg_p_right_p0={avg_p_right_p0:.3f}, avg_p_right_p1={avg_p_right_p1:.3f}")
                print(f"  Prediction bias: {pred_bias_0:.2f}x class0, {pred_bias_1:.2f}x class1 (1.0 = unbiased)")
                
                # Store for wandb logging below
                train_class_0_acc = class_0_acc
                train_class_1_acc = class_1_acc
                train_overall_acc = overall_acc
                train_class_0_pct = class_0_pct
                train_class_1_pct = class_1_pct
                train_avg_p_right_p0 = avg_p_right_p0
                train_avg_p_right_p1 = avg_p_right_p1
        else:
            # For non-remasking_binary training types, no per-class accuracy
            train_class_0_acc = None
            train_class_1_acc = None
            train_avg_p_right_p0 = None
            train_avg_p_right_p1 = None
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100 # convert to percentage
            }
            
            # Add per-class accuracy for remasking_binary
            if training_ctx.training_type == 'remasking_binary' and train_class_0_acc is not None:
                log_dict.update({
                    "train/accuracy_no_mask": train_class_0_acc,
                    "train/accuracy_mask": train_class_1_acc,
                    "train/accuracy_overall": train_overall_acc,
                    "train/class_dist_no_mask": train_class_0_pct,
                    "train/class_dist_mask": train_class_1_pct,
                    "train/corruption_rate": train_class_1_pct,
                    "train/avg_prob_right_p0": train_avg_p_right_p0,
                    "train/avg_prob_right_p1": train_avg_p_right_p1,
                    "train/pred_bias_0": pred_bias_0,
                    "train/pred_bias_1": pred_bias_1,
                })

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