"""
Main training script runner for diffusion training.
Uses train_utils.py for all function definitions.
"""

import os
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
    get_batch, estimate_loss, get_lr, load_synthetic_model, 
    start_prefetch, stop_prefetch, TrainingContext, UnmaskingStage, update_stage_progress
)

torch._dynamo.config.suppress_errors = True

# Global timer instance
timer = Timer()

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 20
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'diffusion'
wandb_run_name = '13k_UN_noise_0.2' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# diffusion training config
training_type = 'unmasking'  # 'unmasking', 'remasking', or 'remasking_binary' - type of training

# For unmasking: stage-based training with direct probability control
unmasking_stages = [
    {'target_masked_ratio': 0.2, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'target_masked_ratio': 0.4, 'p1_probability': 0.3, 'p2_probability': 0.0, 'val_loss_stale_count': 2},
    {'target_masked_ratio': 0.4, 'p1_probability': 0.1, 'p2_probability': 0.5, 'val_loss_stale_count': 2},
    {'target_masked_ratio': 0.6, 'p1_probability': 0.3, 'p2_probability': 0.1, 'val_loss_stale_count': 4},
    {'target_masked_ratio': 0.6, 'p1_probability': 0.1, 'p2_probability': 0.8, 'val_loss_stale_count': 4},
    {'target_masked_ratio': 0.7, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 4},
    {'target_masked_ratio': 0.8, 'p1_probability': 0.2, 'p2_probability': 0.4, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.8, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 6},
    {'target_masked_ratio': 0.9, 'p1_probability': 0.1, 'p2_probability': 0.9, 'val_loss_stale_count': 6},
]

# For remasking/remasking_binary: corruption strategy configuration
remasking_corruption_strategy = 'mixed'  # 'random', 'sticky', 'fragment', 'mixed', 'synthetic' - corruption strategy for remasking
remasking_strategy_weights = [0.25, 0.4, 0.25, 0.1]  # weights for [random, sticky, fragment, synthetic] when using 'mixed'
synthetic_checkpoint_name = '32.08_0.0.pt'  # Path to unmasking model checkpoint for synthetic data generation (only for 'synthetic' strategy)
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'bidirectional' # 'causal' or 'bidirectional' - type of attention to use (bidirectional recommended for diffusion)
# adamw optimizer
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 25000
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 21000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay=1e-1

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
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
if training_type == 'remasking':
    wandb_run_name = f'{wandb_run_name}_remasking'
elif training_type == 'remasking_binary':
    wandb_run_name = f'{wandb_run_name}_remasking_binary'

config = {k: globals()[k] for k in config_keys} # will be useful for logging
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
print(f"tokens per iteration will be: {tokens_per_iter:,}")

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
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # Set special token IDs for different training types
    mask_token_id = meta_vocab_size
    wrong_token_id = meta_vocab_size + 1  # For remasking: corrupted positions
    remask_good_id = meta_vocab_size + 2  # For remasking_binary: uncorrupted positions  
    remask_wrong_id = meta_vocab_size + 3  # For remasking_binary: corrupted positions
    extended_vocab_size = meta_vocab_size + 4  # Add 4 special tokens
    print(f"mask_token_id = {mask_token_id}, wrong_token_id = {wrong_token_id}")
    print(f"remask_good_id = {remask_good_id}, remask_wrong_id = {remask_wrong_id}, extended_vocab_size = {extended_vocab_size}")
else:
    print("No meta.pkl found, using default GPT-2 vocab")
    mask_token_id = 50304
    wrong_token_id = 50305
    remask_good_id = 50306
    remask_wrong_id = 50307
    extended_vocab_size = 50308

# Create training context with all parameters
# Convert unmasking_stages dict to UnmaskingStage objects
unmasking_stage_objects = None
if training_type == 'unmasking':
    unmasking_stage_objects = [
        UnmaskingStage(
            target_masked_ratio=stage['target_masked_ratio'],
            p1_probability=stage['p1_probability'],
            p2_probability=stage['p2_probability'],
            val_loss_stale_count=stage['val_loss_stale_count']
        ) for stage in unmasking_stages
    ]

training_ctx = TrainingContext(
    training_type=training_type,
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    seed_offset=seed_offset,
    data_dir=data_dir,
    meta_vocab_size=meta_vocab_size,
    mask_token_id=mask_token_id,
    wrong_token_id=wrong_token_id,
    remask_good_id=remask_good_id,
    remask_wrong_id=remask_wrong_id,
    extended_vocab_size=extended_vocab_size,
    iter_num=iter_num,
    unmasking_stages=unmasking_stage_objects,
    remasking_corruption_strategy=remasking_corruption_strategy,
    remasking_strategy_weights=remasking_strategy_weights,
    eval_iters=eval_iters,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr
)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = extended_vocab_size if meta_vocab_size is not None else 50305
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    # Find the latest checkpoint file
    import glob
    ckpt_pattern = os.path.join(out_dir, 'ckpt_*.pt')
    ckpt_files = glob.glob(ckpt_pattern)

    if not ckpt_files:
        # Fallback to old naming convention
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint files found in {out_dir}")
    else:
        # Extract iteration numbers and find the latest
        def extract_iter_num(filename):
            basename = os.path.basename(filename)
            # Extract number from ckpt_XXX.pt
            return int(basename.split('_')[2].split('.')[0])

        latest_ckpt = max(ckpt_files, key=extract_iter_num)
        ckpt_path = latest_ckpt
        print(f"Loading latest checkpoint: {os.path.basename(ckpt_path)}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
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
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# Load synthetic model if needed for remasking or remasking_binary training
if training_type in ['remasking', 'remasking_binary'] and synthetic_checkpoint_name:
    # Load synthetic model if strategy is 'synthetic' or 'mixed' (which can use synthetic)
    if remasking_corruption_strategy == 'synthetic' or remasking_corruption_strategy == 'mixed':
        load_synthetic_model(os.path.join(out_dir, synthetic_checkpoint_name), device, extended_vocab_size)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, mask = get_batch('train', training_ctx) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

# Show initial stage configuration for unmasking training
if training_ctx.training_type == 'unmasking':
    stage_config = training_ctx.get_current_stage_config()
    print(f"\n*** STAGE-BASED UNMASKING TRAINING INITIALIZED ***")
    print(f"Starting at Stage {training_ctx.current_stage}:")
    print(f"  Target masked ratio: {stage_config.target_masked_ratio}")
    print(f"  P1 probability: {stage_config.p1_probability}")
    print(f"  P2 probability: {stage_config.p2_probability}")
    print(f"  Val loss stale count limit: {stage_config.val_loss_stale_count}")
    print(f"Total stages configured: {len(training_ctx.unmasking_stages)}")
    print("*** STAGE INITIALIZATION COMPLETE ***\n")

print("Starting training loop...")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, training_ctx) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        print(f"\n--- Starting validation at iteration {iter_num} ---")
        with timer.time_function('validation'):
            # Update training context with current iteration
            training_ctx.iter_num = iter_num
            losses = estimate_loss(model, ctx, timer, training_ctx)

        # VALIDATION INSTABILITY DETECTION
        train_loss_finite = math.isfinite(losses['train'])
        val_loss_finite = math.isfinite(losses['val'])
        if not train_loss_finite or not val_loss_finite:
            print(f"\n*** VALIDATION INSTABILITY at iter {iter_num} ***")
            print(f"Train loss: {losses['train']} ({'finite' if train_loss_finite else 'NaN/Inf'})")
            print(f"Val loss: {losses['val']} ({'finite' if val_loss_finite else 'NaN/Inf'})")
            print("NaN detected in validation - model has become unstable")
            print("*** TERMINATING TRAINING ***")
            break
        
        # Print basic losses
        print(f"--- Validation complete ---")
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        
        # Print stage information for unmasking training
        if training_ctx.training_type == 'unmasking':
            if 'current_stage' in losses:
                print(f"Stage {losses['current_stage']}: masked_ratio={losses.get('target_masked_ratio', 0.0):.1f}, p1={losses.get('p1_probability', 0.0):.1f}, p2={losses.get('p2_probability', 0.0):.1f}, stale_count={losses.get('val_loss_stale_count', 0)}")
        else:
            print(f"Progressive validation used {training_ctx.eval_iters * training_ctx.batch_size} samples representing full training difficulty range")

        # Print model vs random statistics if available
        if 'val_model_vs_random' in losses:
            if training_ctx.training_type in ['remasking_binary', 'remasking']:
                print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
                print(f"  val accuracy: {losses['val_avg_correct_prob']:.4f} (random baseline: {losses.get('val_random_baseline', 0.0):.4f})")
                print(f"  val corruption ratio: {losses.get('val_corruption_ratio', 0.0):.4f}")
                if 'val_most_likely_accuracy' in losses:
                    print(f"  Most likely guess correct P %: {losses['val_most_likely_accuracy']:.1f}%")
            else:
                print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
                print(f"  val avg correct prob: {losses['val_avg_correct_prob']:.4f} (random: {1.0/training_ctx.extended_vocab_size:.4f})")
                if 'val_most_likely_accuracy' in losses:
                    print(f"  Most likely guess correct P %: {losses['val_most_likely_accuracy']:.1f}%")
        
        # Update stage progress for unmasking training
        if training_ctx.training_type == 'unmasking':
            stage_advanced = update_stage_progress(training_ctx, losses['val'])
            if stage_advanced:
                # Clear validation cache after stage change to use new stage parameters
                from train_utils import clear_validation_cache
                clear_validation_cache()
        else:
            print(f"Progressive validation: {training_ctx.eval_iters * training_ctx.batch_size} samples (difficulty: 0 → {training_ctx.max_iters} iters)")
        
        print()  # Add blank line for readability

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "model vs random": losses.get('val_model_vs_random', 0.0),
                "mfu": running_mfu*100, # convert to percentage
                "masked_token_ratio": losses.get('train_masked_token_ratio', 0.0),
                "min_masked_token_ratio": losses.get('train_min_masked_token_ratio', 0.0),
                "max_masked_token_ratio": losses.get('train_max_masked_token_ratio', 0.0),
            }

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
                if training_ctx.training_type == 'remasking':
                    ckpt_filename = f'ckpt_remasking_{iter_num}.pt'
                elif training_ctx.training_type == 'remasking_binary':
                    ckpt_filename = f'ckpt_remasking_binary_{iter_num}.pt'
                else:
                    ckpt_filename = f'ckpt_unmasking_{iter_num}.pt'
                    
                print(f"saving checkpoint to {out_dir}/{ckpt_filename}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_filename))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
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
                
                # TRAINING INSTABILITY DETECTION
                if not torch.isfinite(logits).all():
                    print(f"\n*** INSTABILITY DETECTED at iter {iter_num} ***")
                    print(f"Logits contain NaN/Inf: {torch.isnan(logits).sum().item()} NaN, {torch.isinf(logits).sum().item()} Inf")
                    print(f"Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
                    print("*** TERMINATING TRAINING ***")
                    break
                
                if not torch.isfinite(loss):
                    print(f"\n*** LOSS INSTABILITY at iter {iter_num} ***")
                    print(f"Loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
                    print("*** TERMINATING TRAINING ***")
                    break
                
                # Apply masking for unmasking training only (most efficient path)
                if training_ctx.training_type == 'unmasking' and mask.any():
                    # Fast path: reshape once and use boolean indexing
                    logits_reshaped = logits.view(-1, logits.size(-1))
                    targets_reshaped = Y.view(-1)
                    mask_reshaped = mask.view(-1)
                    
                    # Use boolean mask directly - most efficient
                    loss = torch.nn.functional.cross_entropy(
                        logits_reshaped[mask_reshaped], 
                        targets_reshaped[mask_reshaped], 
                        reduction='mean'
                    )
                # For remasking variants, model's internal loss is already correct
                
                # UNIVERSAL: Check final loss after any training-type-specific processing
                if not torch.isfinite(loss):
                    print(f"\n*** FINAL LOSS INSTABILITY at iter {iter_num} ***")
                    print(f"Final loss is {loss.item()}: {'NaN' if torch.isnan(loss) else 'Inf'}")
                    print(f"Training type: {training_ctx.training_type}")
                    if hasattr(mask, 'float'):  # Check if mask exists
                        print(f"Mask ratio: {mask.float().mean().item():.4f}")
                    print("*** TERMINATING TRAINING ***")
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
            print("*** TERMINATING TRAINING ***")
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
            print("*** TERMINATING TRAINING ***")
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
        print("*** TERMINATING TRAINING ***")
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
        loss_time = timer.get_average('loss_computation') * 1000
        backward_time = timer.get_average('backward_pass') * 1000

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"  data: {data_time:.2f}ms, forward: {forward_time:.2f}ms, loss: {loss_time:.2f}ms, backward: {backward_time:.2f}ms")

        # Validation timing (when applicable)
        if iter_num % eval_interval == 0:
            val_time = timer.get_average('validation') * 1000
            val_data_time = timer.get_average('validation_data_generation') * 1000
            val_forward_time = timer.get_average('validation_forward_pass') * 1000
            val_loss_time = timer.get_average('validation_loss_computation') * 1000
            print(f"  validation: {val_time:.2f}ms (data: {val_data_time:.2f}ms, forward: {val_forward_time:.2f}ms, loss: {val_loss_time:.2f}ms)")

        # Add masking statistics logging
        if training_ctx.training_type == 'unmasking':
            # For stage-based unmasking, show current stage info
            stage_config = training_ctx.get_current_stage_config()
            if stage_config and iter_num % (log_interval * 10) == 0:
                mask_ratio = mask.float().mean().item()
                print(f"Masking: stage={training_ctx.current_stage}, actual_ratio={mask_ratio:.3f}, target={stage_config.target_masked_ratio:.1f}, p1={stage_config.p1_probability:.1f}, p2={stage_config.p2_probability:.1f}")
        else:
            # For remasking types, use simple masking stats
            log_masking_stats(mask, iter_num, log_interval)
        
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "mfu": running_mfu*100 # convert to percentage
            }

            wandb.log(log_dict)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Cleanup prefetch thread
stop_prefetch()