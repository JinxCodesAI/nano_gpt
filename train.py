"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils import (Timer, log_masking_stats, apply_sticky_masking,
                   get_phase_info, create_soft_targets, compute_loss_with_soft_targets,
                   EntropyPenaltyModifier)
torch._dynamo.config.suppress_errors = True

# Global timer instance
timer = Timer()

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 20
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# diffusion training config
guaranteed_unmasked = 0.05       # Guaranteed fraction of tokens to keep unmasked

# sticky masking configuration - gradual transition from independent to sticky
sticky_transition_start = 1000   # When to start introducing sticky masking
sticky_transition_end = 3000     # When to reach full sticky masking
sticky_rounds = 10                # Number of sticky masking rounds
sticky_p1_p2_multiplier = 10.0    # Multiplier for sticky_p2 = sticky_p1 * multiplier

# multi-phase training configuration
n1_iterations = 500     # Phase 1: Identity task
n2_iterations = 500    # Phase 2: Gradual target increase
n3_iterations = 7500    # Phase 3: Standard training
n4_iterations = 7500   # Phase 4: Entropy penalty (overlaps 2-3, N4 < N2 + N3)

entropy_multiplier_max = 0.0    # Maximum entropy penalty multiplier
entropy_multiplier_min = 0.0    # Minimum entropy penalty multiplier
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'causal' # 'causal' or 'bidirectional' - type of attention to use (bidirectional recommended for diffusion)
# adamw optimizer

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 8000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta1 = 0.9
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
weight_decay=1e-1

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16'
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
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

# Cache for consistent validation batches
_val_batch_cache = None

def get_batch(split):
    global _val_batch_cache, iter_num

    # For validation, use cached batch to ensure consistency
    if split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    if split == 'val':
        # For validation, use fixed seed to ensure reproducible indices
        torch.manual_seed(42)
        ix = torch.randint(len(data) - block_size, (batch_size,))
        # Reset to original seed
        torch.manual_seed(1337 + seed_offset)
    else:
        # For training, use random indices as before
        ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    if split == 'val':
        # For validation, use fixed sticky masking with 0.5 ratio for consistency
        torch.manual_seed(42)

        # Apply mixed masking strategy with fixed 0.5 sticky ratio
        current_batch_size = x.shape[0]
        num_sticky_batches = int(current_batch_size * 0.5)  # Fixed 50% sticky ratio

        masked_x = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)

        # Apply independent masking to first part of batch
        if num_sticky_batches < current_batch_size:
            masking_prob = 0.5 * (1.0 - guaranteed_unmasked)  # Fixed at middle of range
            indep_mask = torch.rand(x[:current_batch_size-num_sticky_batches].shape) < masking_prob
            masked_x[:current_batch_size-num_sticky_batches][indep_mask] = mask_token_id
            mask[:current_batch_size-num_sticky_batches] = indep_mask

        # Apply sticky masking to remaining part of batch
        if num_sticky_batches > 0:
            sticky_masked_x, sticky_mask = apply_sticky_masking(
                x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier
            )
            masked_x[-num_sticky_batches:] = sticky_masked_x
            mask[-num_sticky_batches:] = sticky_mask

        # Reset to original seed
        torch.manual_seed(1337 + seed_offset)
    else:
        # For training, apply mixed masking strategy based on current iteration
        # Calculate sticky masking ratio based on current training iteration
        if iter_num < sticky_transition_start:
            # Pure independent masking
            sticky_ratio = 0.0
        elif iter_num >= sticky_transition_end:
            # Pure sticky masking
            sticky_ratio = 1.0
        else:
            # Gradual transition from independent to sticky
            progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
            sticky_ratio = progress

        # Apply mixed masking strategy
        if sticky_ratio == 0.0:
            # Pure independent masking
            masking_prob = torch.rand(1).item() * (1.0 - guaranteed_unmasked)
            mask = torch.rand(x.shape) < masking_prob
            masked_x = x.clone()
            masked_x[mask] = mask_token_id

        elif sticky_ratio == 1.0:
            # Pure sticky masking
            masked_x, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)

        else:
            # Mixed strategy: some batches independent, some sticky
            current_batch_size = x.shape[0]
            num_sticky_batches = int(current_batch_size * sticky_ratio)

            masked_x = x.clone()
            mask = torch.zeros_like(x, dtype=torch.bool)

            # Apply independent masking to first part of batch
            if num_sticky_batches < current_batch_size:
                masking_prob = torch.rand(1).item() * (1.0 - guaranteed_unmasked)
                indep_mask = torch.rand(x[:current_batch_size-num_sticky_batches].shape) < masking_prob
                masked_x[:current_batch_size-num_sticky_batches][indep_mask] = mask_token_id
                mask[:current_batch_size-num_sticky_batches] = indep_mask

            # Apply sticky masking to remaining part of batch
            if num_sticky_batches > 0:
                sticky_masked_x, sticky_mask = apply_sticky_masking(
                    x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier
                )
                masked_x[-num_sticky_batches:] = sticky_masked_x
                mask[-num_sticky_batches:] = sticky_mask

    # Target is original x, loss computed only on masked positions
    y = x.clone()

    # Move to device
    if device_type == 'cuda':
        masked_x = masked_x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        masked_x, y, mask = masked_x.to(device), y.to(device), mask.to(device)

    # Cache validation batch for consistency
    if split == 'val':
        _val_batch_cache = (masked_x, y, mask)

    return masked_x, y, mask

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
    
    # Set mask_token_id to be vocab_size (next available ID)
    mask_token_id = meta_vocab_size
    extended_vocab_size = meta_vocab_size + 1  # Add 1 for mask token
    print(f"mask_token_id = {mask_token_id}, extended_vocab_size = {extended_vocab_size}")
else:
    print("No meta.pkl found, using default GPT-2 vocab")
    mask_token_id = 50304
    extended_vocab_size = 50305

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
            return int(basename.split('_')[1].split('.')[0])

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

# initialize entropy penalty modifier
entropy_penalty_modifier = EntropyPenaltyModifier(
    penalty_strength=entropy_multiplier_max,  # Will be updated dynamically
    vocab_size=extended_vocab_size
)

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

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'val':
            # For validation, also track model vs random performance
            model_probs = []
            random_prob = 1.0 / extended_vocab_size  # Random chance probability

        for k in range(eval_iters):
            with timer.time_function('validation_data_generation'):
                X, Y, mask = get_batch(split)  # Updated to return mask
            with ctx:
                with timer.time_function('validation_forward_pass'):
                    # Pass dummy targets to force full sequence logits, ignore returned loss
                    logits, _ = model(X, torch.zeros_like(Y))
                with timer.time_function('validation_loss_computation'):
                    # Use same multi-phase loss computation as training
                    val_phase_info = get_phase_info(iter_num, n1_iterations, n2_iterations, n3_iterations,
                                                   n4_iterations, entropy_multiplier_max, entropy_multiplier_min)

                    # Create soft targets based on current phase
                    val_soft_targets, val_is_soft = create_soft_targets(
                        Y, extended_vocab_size, mask, mask_token_id,
                        val_phase_info['phase'], iter_num, n1_iterations, n2_iterations
                    )

                    # Compute base loss with soft/hard targets
                    if val_phase_info['phase'] == 1:
                        # Phase 1: Loss on all positions (identity task for unmasked, uniform for masked)
                        loss = compute_loss_with_soft_targets(logits, val_soft_targets, val_is_soft)
                    elif val_phase_info['phase'] == 2:
                        # Phase 2: Loss on all positions (identity for unmasked, gradual for masked)
                        loss = compute_loss_with_soft_targets(logits, val_soft_targets, val_is_soft)
                    else:
                        # Phase 3: Standard masked language modeling loss
                        loss = compute_loss_with_soft_targets(logits, Y, False, mask)

                # For validation, compute model vs random statistics on masked tokens only
                if split == 'val':
                    # Get probabilities from logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

                    # Flatten mask to match targets
                    mask_flat = mask.view(-1)  # (batch_size * seq_len,)

                    # Get probabilities for correct tokens at masked positions only
                    masked_positions = mask_flat.bool()
                    if masked_positions.sum() > 0:  # Only if there are masked tokens
                        probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
                        targets_flat = Y.view(-1)  # (batch_size * seq_len,)
                        correct_token_probs = probs_flat[masked_positions, targets_flat[masked_positions]]
                        model_probs.extend(correct_token_probs.cpu().tolist())

            losses[k] = loss.item()

        out[split] = losses.mean()

        # Add model vs random comparison for validation
        if split == 'val' and model_probs:
            avg_model_prob = sum(model_probs) / len(model_probs)
            prob_ratio = avg_model_prob / random_prob
            out[f'{split}_model_vs_random'] = prob_ratio
            out[f'{split}_avg_correct_prob'] = avg_model_prob

    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y, mask = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
print("Starting training loop...")
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        with timer.time_function('validation'):
            losses = estimate_loss()

        # Print basic losses
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")

        # Print model vs random statistics if available
        if 'val_model_vs_random' in losses:
            print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
            print(f"  val avg correct prob: {losses['val_avg_correct_prob']:.4f} (random: {1.0/extended_vocab_size:.4f})")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }

            # Add model vs random statistics to wandb if available
            if 'val_model_vs_random' in losses:
                log_dict["val/model_vs_random_ratio"] = losses['val_model_vs_random']
                log_dict["val/avg_correct_prob"] = losses['val_avg_correct_prob']
                log_dict["val/random_prob"] = 1.0 / extended_vocab_size

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
                ckpt_filename = f'ckpt_{iter_num}.pt'
                print(f"saving checkpoint to {out_dir}/{ckpt_filename}")
                torch.save(checkpoint, os.path.join(out_dir, ckpt_filename))
    if iter_num == 0 and eval_only:
        break

    # Get current phase information (outside the micro-step loop for logging)
    phase_info = get_phase_info(iter_num, n1_iterations, n2_iterations, n3_iterations,
                              n4_iterations, entropy_multiplier_max, entropy_multiplier_min)
    penalty_diagnostics = {}

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
                # Pass dummy targets to force full sequence logits, ignore returned loss
                logits, _ = model(X, torch.zeros_like(Y))
            with timer.time_function('loss_computation'):
                # Create soft targets based on current phase
                soft_targets, is_soft = create_soft_targets(
                    Y, extended_vocab_size, mask, mask_token_id,
                    phase_info['phase'], iter_num, n1_iterations, n2_iterations
                )

                # Compute base loss with soft/hard targets
                if phase_info['phase'] == 1:
                    # Phase 1: Loss on all positions (identity task for unmasked, uniform for masked)
                    loss = compute_loss_with_soft_targets(logits, soft_targets, is_soft)
                elif phase_info['phase'] == 2:
                    # Phase 2: Loss on all positions (identity for unmasked, gradual for masked)
                    loss = compute_loss_with_soft_targets(logits, soft_targets, is_soft)
                else:
                    # Phase 3: Standard masked language modeling loss
                    loss = compute_loss_with_soft_targets(logits, Y, False, mask)

                # Apply entropy penalty if in penalty phase
                if phase_info['has_entropy_penalty']:
                    # Update entropy penalty strength
                    entropy_penalty_modifier.penalty_strength = phase_info['entropy_strength']

                    # Apply entropy penalty
                    loss, penalty_diagnostics = entropy_penalty_modifier.apply_penalty(
                        loss, logits, soft_targets if is_soft else Y, X, mask_token_id, is_soft
                    )

                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        with timer.time_function('data_generation'):
            X, Y, mask = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        with timer.time_function('backward_pass'):
            scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
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

        # Add masking statistics logging with transition tracking
        log_masking_stats(mask, iter_num, log_interval, sticky_transition_start, sticky_transition_end)

        # Add phase monitoring
        print(f"  {phase_info['phase_name']}")

        # Add entropy penalty diagnostics if available
        if penalty_diagnostics and 'avg_entropy_penalty_factor' in penalty_diagnostics:
            print(f"  entropy penalty: {penalty_diagnostics['avg_entropy_penalty_factor']:.4f} avg factor, "
                  f"strength {penalty_diagnostics['penalty_entropy_strength']:.2f}, "
                  f"multiplier {penalty_diagnostics.get('penalty_multiplier', 1.0):.3f}x")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
