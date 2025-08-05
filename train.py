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
from torch.nn import functional as F
# In train.py

def get_corruption_scheduler(it):
    """
    Controls the curriculum for both penalties and data generation.
    Returns the current penalty values and the re-masking task ratio.
    """
    # --- Penalty Curriculum ---
    if it >= masking_warmup_iters:
        current_penalty_mask_correct = penalty_mask_correct
    else:
        ratio = it / masking_warmup_iters
        current_penalty_mask_correct = 0.0 + ratio * (penalty_mask_correct - 0.0)

    # --- Data Generation Curriculum ---
    if it >= proofreading_warmup_iters:
        remask_ratio = 1.0
    else:
        remask_ratio = it / proofreading_warmup_iters
        
    return current_penalty_mask_correct, remask_ratio

def calculate_diffusion_loss(logits, targets, inputs, mask_token_id, 
                             current_penalty_mask_correct, weight_unmask_task, 
                             weight_remask_task, mask_logit_bias, log_diagnostics=False):
    """
    A simplified and robust loss function that does not use argmax for weighting.
    It applies weights based only on the task type (the Input-Target pair).
    """
    # --- Step 1: Apply the dynamic logit bias ---
    biased_logits = logits.clone()
    if mask_logit_bias != 0.0:
        biased_logits[:, :, mask_token_id] += mask_logit_bias

    # --- Step 2: Calculate the base cross-entropy loss ---
    flat_logits = biased_logits.view(-1, biased_logits.size(-1))
    flat_targets = targets.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='none')

    # --- Step 3: Apply weights based on the TASK TYPE only ---
    weights = torch.ones_like(per_token_loss)
    flat_inputs = inputs.view(-1)
    
    # Task 1: Un-masking (Input was [MASK], Target is a word)
    unmask_task_positions = (flat_inputs == mask_token_id) & (flat_targets != mask_token_id)
    weights[unmask_task_positions] = weight_unmask_task
    
    # Task 2: Re-masking (Input was a word, Target is [MASK])
    remask_task_positions = (flat_inputs != mask_token_id) & (flat_targets == mask_token_id)
    weights[remask_task_positions] = weight_remask_task
    
    # Task 3: State-Dependent Penalty for Destructive Edits
    with torch.no_grad():
        B, T = inputs.shape
        predictions_2d = torch.argmax(biased_logits, dim=-1)
        for i in range(B):
            epsilon = 1e-6
            input_words_mask = (inputs[i] != mask_token_id)
            num_input_words = input_words_mask.sum().item()
            corrupted_words_mask = (targets[i, input_words_mask] == mask_token_id)
            num_corrupted_words = corrupted_words_mask.sum().item()
            corruption_rate = num_corrupted_words / (num_input_words + epsilon)
            effective_penalty = current_penalty_mask_correct * (1.0 - corruption_rate)
            
            destructive_mask = (inputs[i] == targets[i]) & (inputs[i] != mask_token_id) & (predictions_2d[i] == mask_token_id)
            
            weights_2d = weights.view(B, T)
            weights_2d[i, destructive_mask] = effective_penalty
            
    # The final loss is the weighted average
    final_loss = (per_token_loss * weights).mean()

    # --- Step 4: Calculate the NEW Feedback Signal (argmax-free) ---
    avg_mask_prob = 0.0
    if unmask_task_positions.any():
        unmask_logits = biased_logits.view(-1, biased_logits.size(-1))[unmask_task_positions]
        unmask_probs = F.softmax(unmask_logits, dim=-1)
        avg_mask_prob = unmask_probs[:, mask_token_id].mean().item()

    # --- PRESERVED DIAGNOSTIC LOGGING ---
    if log_diagnostics:
        with torch.no_grad():
            # [LOGITS] logging - preserved from _calculate_diagnostic_logs
            avg_mask_logit = logits[:, :, mask_token_id].mean().item()
            targets_for_gather = targets.clone()
            valid_targets_mask = targets_for_gather != -1
            targets_for_gather[~valid_targets_mask] = 0
            correct_target_logits = logits.gather(-1, targets_for_gather.unsqueeze(-1)).squeeze(-1)
            avg_correct_target_logit = correct_target_logits[valid_targets_mask].mean().item()
            avg_max_logit = logits.max(dim=-1).values.mean().item()
            
            print(
                f"[LOGITS] Avg MASK: {avg_mask_logit:7.2f} | "
                f"Avg CORRECT: {avg_correct_target_logit:7.2f} | "
                f"Avg MAX: {avg_max_logit:7.2f}"
            )
            
            # [BEHAVIOR] logging - adapted for new system
            epsilon = 1e-6
            predictions_flat = torch.argmax(biased_logits, dim=-1).view(-1)
            
            # Calculate behavior metrics compatible with new system
            is_unmask_task = unmask_task_positions
            total_unmask_tasks = is_unmask_task.sum().item()
            
            attempted_to_unmask = is_unmask_task & (predictions_flat != mask_token_id)
            correct_unmasks = (attempted_to_unmask & (predictions_flat == flat_targets)).sum().item()
            incorrect_unmasks = (attempted_to_unmask & (predictions_flat != flat_targets)).sum().item()
            kept_mask_incorrectly = (is_unmask_task & (predictions_flat == mask_token_id) & (flat_targets != mask_token_id)).sum().item()
            
            num_unmask_attempts = correct_unmasks + incorrect_unmasks
            unmask_accuracy = correct_unmasks / (num_unmask_attempts + epsilon)
            skill_vs_random = unmask_accuracy / (1.0 / meta_vocab_size if meta_vocab_size is not None else 1.0)
            
            input_mask_rate = is_unmask_task.float().mean().item()
            output_mask_rate = (predictions_flat == mask_token_id).float().mean().item()
            mask_preference = output_mask_rate / (input_mask_rate + epsilon)
            
            print(
                f"[BEHAVIOR] Correct: {correct_unmasks:<4} | "
                f"Incorrect: {incorrect_unmasks:<4} | "
                f"Kept Mask: {kept_mask_incorrectly:<4} | "
                f"Total Unmask Tasks: {total_unmask_tasks:<5}"
            )
            print(
                f"           Unmask Acc: {unmask_accuracy:<5.1%} | "
                f"Skill vs Random: {skill_vs_random:<5.1f}x | "
                f"Mask Preference: {mask_preference:<5.2f} | "
                f"Avg MASK Prob: {avg_mask_prob:.2%}"
            )

    return final_loss, avg_mask_prob
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
model_type = 'gpt2' # 'gpt2' or 'diffusion'
# --- NEW SIMPLIFIED LOSS CONFIG ---
initial_mask_logit_bias = 3.5   # The starting value for our dynamic bias.
bias_update_strength = 0.01   # How quickly the bias adapts.
target_mask_prob = 0.5        # The desired output probability for [MASK] on un-masking tasks.

# Simple, task-based loss weights
weight_unmask_task = 1.0        # The base weight for the loss when the model must guess a word.
weight_remask_task = 1.0        # The base weight for the loss when the model must identify a corrupted word.

# Curriculum settings
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # Iterations for the penalty curriculum.
proofreading_warmup_iters = 2000 # NEW: Iterations to ramp up the "re-masking" task.

guaranteed_correct_factor = 0.01 
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
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
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

mask_logit_bias = initial_mask_logit_bias
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
mask_token_id = None # Global variable to be set after model init

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        # For diffusion models, validation data is handled by the fixed set
        # For regular models, we still need to handle val split
        if model_type == 'diffusion':
            # This should not be called for val split in diffusion mode
            raise ValueError("get_batch should not be called with 'val' split for diffusion models")
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_clean = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if model_type == 'diffusion':
        assert mask_token_id is not None, "mask_token_id must be set globally"
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()

        # Get the current re-masking task ratio from the curriculum scheduler
        _, remask_ratio = get_corruption_scheduler(iter_num)

        b, t = x_corrupted.shape
        for i in range(b):
            max_corruption = 1 - guaranteed_correct_factor
            
            rate_mask = torch.rand(1) * max_corruption
            # The rate of re-masking tasks is now controlled by the curriculum
            rate_random = torch.rand(1) * rate_mask * remask_ratio
            # The final un-masking rate is what's left over
            rate_mask = rate_mask - rate_random

            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            x_corrupted[i, pos_mask] = mask_token_id
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            y_target[i, pos_random] = mask_token_id

        x, y = x_corrupted, y_target
    else:
        y_autoregressive = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x_clean, y_autoregressive

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def create_fixed_validation_set(remask_ratio):
    """
    Creates a fixed set of validation tasks with linearly increasing difficulty.
    This function is run once at the start of training for diffusion models only.
    """
    print("Creating fixed validation set...")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # Use a fixed seed to ensure the same text snippets are chosen every time
    torch.manual_seed(1337)

    val_batches = []
    for k in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x_clean = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
        
        # --- Create corrupted X and target Y for this batch ---
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone()
        b, t = x_corrupted.shape

        # Linearly interpolate the total corruption rate
        # Start with low corruption (high guaranteed_correct_factor)
        # End with high corruption (low guaranteed_correct_factor)
        progress = k / (eval_iters - 1) if eval_iters > 1 else 1.0
        start_corruption = guaranteed_correct_factor
        end_corruption = 1.0 - guaranteed_correct_factor
        current_max_corruption = start_corruption + progress * (end_corruption - start_corruption)

        for i in range(b):
            rate_mask = torch.rand(1) * current_max_corruption
            # Use the passed-in remask_ratio for validation set
            rate_random = torch.rand(1) * rate_mask * remask_ratio
            # The final un-masking rate is what's left over
            rate_mask = rate_mask - rate_random
            
            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            x_corrupted[i, pos_mask] = mask_token_id
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            y_target[i, pos_random] = mask_token_id

        # Move to the correct device
        if device_type == 'cuda':
            x_corrupted = x_corrupted.pin_memory().to(device, non_blocking=True)
            y_target = y_target.pin_memory().to(device, non_blocking=True)
        else:
            x_corrupted, y_target = x_corrupted.to(device), y_target.to(device)
        
        val_batches.append((x_corrupted, y_target))
    
    # Reset the seed to not affect subsequent operations
    torch.manual_seed(1337 + seed_offset)
    print("Fixed validation set created.")
    return val_batches

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

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, model_type=model_type,
                  mask_logit_bias=mask_logit_bias) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # --- MODIFICATION ---
    if model_type == 'diffusion':
        vocab_size = meta_vocab_size + 1 if meta_vocab_size is not None else 50305
        model_args['vocab_size'] = vocab_size
        model_args['mask_token_id'] = vocab_size - 1
        mask_token_id = model_args['mask_token_id']
    else:
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    # --- END MODIFICATION ---
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # --- FIX START: Correctly load all necessary model arguments ---
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    keys_to_force = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    for k in keys_to_force:
        model_args[k] = checkpoint_model_args[k]
    
    # Explicitly check for and load diffusion-specific arguments
    if 'model_type' in checkpoint_model_args:
        model_args['model_type'] = checkpoint_model_args['model_type']
    if 'mask_token_id' in checkpoint_model_args:
        model_args['mask_token_id'] = checkpoint_model_args['mask_token_id']
    # --- MODIFICATION: Add mask_logit_bias to the list of keys to load ---
    if 'mask_logit_bias' in checkpoint_model_args:
        model_args['mask_logit_bias'] = checkpoint_model_args['mask_logit_bias']
    # --- END MODIFICATION ---
    
    # After loading, update the global variables that control the script's behavior
    if model_args.get('model_type') == 'diffusion':
        print("Resuming in diffusion mode.")
        model_type = 'diffusion'
        mask_token_id = model_args.get('mask_token_id')
        assert mask_token_id is not None, "Resumed a diffusion model but mask_token_id is missing."
    # --- FIX END ---
        
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

# --- NEW: Initialize the dynamic bias and its feedback signal ---
mask_logit_bias = initial_mask_logit_bias
running_avg_mask_prob = target_mask_prob # Use a moving average for stability
# --- END NEW ---

# Global variable to store fixed validation batches for diffusion models
fixed_val_batches = None

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(current_penalty_mask_correct, current_mask_logit_bias):
    global fixed_val_batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        if split == 'val' and model_type == 'diffusion':
            if fixed_val_batches is None:
                # Note: The fixed val set uses the FINAL curriculum values, not the current ones
                _, final_remask_ratio = get_corruption_scheduler(proofreading_warmup_iters)
                fixed_val_batches = create_fixed_validation_set(final_remask_ratio) # Pass ratio here

        for k in range(eval_iters):
            if split == 'val' and model_type == 'diffusion':
                X, Y = fixed_val_batches[k]
            else:
                X, Y = get_batch(split)

            with ctx:
                logits, loss_from_model = model(X, Y)
                if model_type == 'diffusion':
                    # Get final penalty value for eval
                    final_penalty_mask_correct, _ = get_corruption_scheduler(masking_warmup_iters)
                    loss, _ = calculate_diffusion_loss(
                        logits, Y, X, mask_token_id, 
                        final_penalty_mask_correct,
                        weight_unmask_task, weight_remask_task,
                        current_mask_logit_bias,
                        log_diagnostics=False
                    )
                else:
                    loss = loss_from_model
                
            losses[k] = loss.item()
        out[split] = losses.mean()
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
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # --- MODIFICATION: Use the new scheduler ---
    lr = get_lr(iter_num) if decay_lr else learning_rate
    current_penalty_mask_correct, _ = get_corruption_scheduler(iter_num)
    # --- END MODIFICATION ---
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(current_penalty_mask_correct, mask_logit_bias)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
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
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}_g.pt'))
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
            logits, loss_from_model = model(X, Y)
            if model_type == 'diffusion':
                # --- MODIFICATION: Use the new loss function and feedback signal ---
                loss, avg_mask_prob = calculate_diffusion_loss(
                    logits, Y, X, mask_token_id,
                    current_penalty_mask_correct,
                    weight_unmask_task, weight_remask_task,
                    mask_logit_bias,
                    log_diagnostics=(micro_step == gradient_accumulation_steps - 1)
                )
            else:
                loss = loss_from_model
                avg_mask_prob = target_mask_prob # Placeholder
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
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

    # --- MODIFICATION: The new feedback loop ---
    if model_type == 'diffusion':
        running_avg_mask_prob = 0.99 * running_avg_mask_prob + 0.01 * avg_mask_prob
        error_signal = target_mask_prob - running_avg_mask_prob
        mask_logit_bias += bias_update_strength * error_signal
    # --- END MODIFICATION ---

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
        
        # --- MODIFICATION: Add the dynamic bias to the log message to monitor it ---
        if model_type == 'diffusion':
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, logit_bias {mask_logit_bias:.2f}")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        # --- END MODIFICATION ---
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
