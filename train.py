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
import threading
from queue import Queue

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from utils import Timer, log_masking_stats, apply_sticky_masking
torch._dynamo.config.suppress_errors = True

# Global timer instance
timer = Timer()

# Global synthetic model for remasking
synthetic_model = None

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
wandb_log = True # disabled by default
wandb_project = 'diffusion'
wandb_run_name = '10k_RE_Bi' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# diffusion training config
training_type = 'remasking'  # 'unmasking', 'remasking', or 'remasking_binary' - type of training
remasking_corruption_strategy = 'mixed'  # 'random', 'sticky', 'fragment', 'mixed', 'synthetic' - corruption strategy for remasking
remasking_strategy_weights = [0.3, 0.4, 0.3, 0.0]  # weights for [random, sticky, fragment, synthetic] when using 'mixed'
synthetic_checkpoint_name = '14.6_unmasking_no_noise.pt'  # Path to unmasking model checkpoint for synthetic data generation (only for 'synthetic' strategy)
guaranteed_unmasked = 0.0       # Guaranteed fraction of tokens to keep unmasked
noise_max_ratio = 0.05            # Maximum ratio of unmasked tokens to corrupt with random noise (0.0 to 1.0) - only for unmasking training

# sticky masking configuration - gradual transition from independent to sticky
sticky_transition_start = 500   # When to start introducing sticky masking
sticky_transition_end = 12000     # When to reach full sticky masking
sticky_rounds = 10                # Number of sticky masking rounds
sticky_p1_p2_multiplier = 10.0    # Multiplier for sticky_p2 = sticky_p1 * multiplier
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
attention_type = 'bidirectional' # 'causal' or 'bidirectional' - type of attention to use (bidirectional recommended for diffusion)
# adamw optimizer

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 8000 # make equal to max_iters usually
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
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
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

# Cache for consistent validation batches and data prefetching
_val_batch_cache = None
_data_cache = {'train': None, 'val': None}  # Cache memory-mapped data
_valid_indices_cache = {'train': None, 'val': None}  # Cache expensive index computation
_prefetch_enabled = True
_prefetch_queue = Queue(maxsize=2)  # Background batch preparation
_prefetch_thread = None
_prefetch_active = False

def find_double_newline_indices(data, meta_vocab_size):
    """Find all valid starting indices that begin with double newlines (\n\n)"""
    # Get the token IDs for newlines
    if meta_vocab_size is not None:
        # For Shakespeare character-level data, newline is token 0
        newline_id = 0
    else:
        # For GPT-2 style tokenization, this would be different
        newline_id = 198  # GPT-2 newline token
    
    # Find positions where we have \n\n (two consecutive newlines)
    valid_indices = []
    for i in range(len(data) - block_size - 1):  # -1 to ensure we can check i+1
        if i >= 1 and data[i] == newline_id and data[i+1] == newline_id:
            valid_indices.append(i)
    
    return np.array(valid_indices)

def _prepare_batch_data_only(split):
    """Background function to prepare raw batch data (CPU only)"""
    global _data_cache, _valid_indices_cache
    
    # Ensure data is cached
    if _data_cache[split] is None:
        return None
        
    data = _data_cache[split]
    valid_indices = _valid_indices_cache[split]
    
    # Fast index sampling - all on CPU
    if len(valid_indices) == 0:
        ix_np = torch.randint(len(data) - block_size, (batch_size,)).numpy()
    else:
        ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
        ix_np = valid_indices[ix_indices]

    # VECTORIZED DATA LOADING - Load entire batch at once
    x_np = np.zeros((batch_size, block_size), dtype=np.int64)
    for i, start_idx in enumerate(ix_np):
        x_np[i] = data[start_idx:start_idx+block_size].astype(np.int64)
    
    return x_np

def _prefetch_worker():
    """Background thread worker for data prefetching"""
    global _prefetch_active
    while _prefetch_active:
        try:
            # Prepare next batch in background
            x_np = _prepare_batch_data_only('train')
            if x_np is not None:
                _prefetch_queue.put(x_np, timeout=1.0)
        except:
            # Queue full or other error, just continue
            time.sleep(0.001)

def start_prefetch():
    """Start background data prefetching"""
    global _prefetch_thread, _prefetch_active
    if _prefetch_thread is None and _prefetch_enabled:
        _prefetch_active = True
        _prefetch_thread = threading.Thread(target=_prefetch_worker, daemon=True)
        _prefetch_thread.start()

def stop_prefetch():
    """Stop background data prefetching"""
    global _prefetch_thread, _prefetch_active
    _prefetch_active = False
    if _prefetch_thread is not None:
        _prefetch_thread.join(timeout=1.0)
        _prefetch_thread = None

def get_batch(split):
    if training_type == 'remasking':
        return get_batch_remasking(split)
    elif training_type == 'remasking_binary':
        return get_batch_remasking_binary(split)
    
    # Ultra-fast unmasking implementation with aggressive caching + prefetching
    global _val_batch_cache, iter_num, _data_cache, _valid_indices_cache

    # For validation, use cached batch to ensure consistency
    if split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Cache memory-mapped data and valid indices - MAJOR SPEEDUP
    if _data_cache[split] is None:
        if split == 'train':
            _data_cache[split] = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            _data_cache[split] = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        # Cache the expensive valid indices computation
        print(f"Computing valid indices for {split}... (one-time cost)")
        _valid_indices_cache[split] = find_double_newline_indices(_data_cache[split], meta_vocab_size)
        print(f"Found {len(_valid_indices_cache[split])} valid indices for {split}")
        
        # Start prefetching for training data
        if split == 'train':
            start_prefetch()

    # Try to get prefetched data for training
    x_np = None
    if split == 'train' and _prefetch_enabled:
        try:
            x_np = _prefetch_queue.get_nowait()
        except:
            pass  # Queue empty, generate normally
    
    # Generate data if not prefetched
    if x_np is None:
        data = _data_cache[split]
        valid_indices = _valid_indices_cache[split]
        
        # Fast index sampling - all on CPU to avoid GPU-CPU sync
        if len(valid_indices) == 0:
            if split == 'val':
                torch.manual_seed(42)
                ix_np = torch.randint(len(data) - block_size, (batch_size,)).numpy()
                torch.manual_seed(1337 + seed_offset)
            else:
                ix_np = torch.randint(len(data) - block_size, (batch_size,)).numpy()
        else:
            if split == 'val':
                torch.manual_seed(42)
                ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]
                torch.manual_seed(1337 + seed_offset)
            else:
                ix_indices = torch.randint(len(valid_indices), (batch_size,)).numpy()
                ix_np = valid_indices[ix_indices]

        # VECTORIZED DATA LOADING - Load entire batch at once
        x_np = np.zeros((batch_size, block_size), dtype=np.int64)
        for i, start_idx in enumerate(ix_np):
            x_np[i] = data[start_idx:start_idx+block_size].astype(np.int64)
    
    # Single GPU transfer with pinned memory
    x = torch.from_numpy(x_np)
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)

    # GPU-accelerated masking operations (already on GPU)
    if split == 'val':
        torch.manual_seed(42)
        masked_x, mask = apply_gpu_masking_validation(x, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier, guaranteed_unmasked)
        torch.manual_seed(1337 + seed_offset)
    else:
        masked_x, mask = apply_gpu_masking_training(x, iter_num, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier, 
                                                  guaranteed_unmasked, sticky_transition_start, sticky_transition_end)

    # Apply random noise to unmasked positions (already on GPU)
    masked_x = apply_random_noise_to_unmasked_gpu(masked_x, mask, noise_max_ratio, meta_vocab_size)
    
    # Target is original x
    y = x.clone()

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

def apply_random_corruption(x, corruption_prob, guaranteed_unmasked, meta_vocab_size):
    """Strategy 1: Random token corruption (original method)"""
    mask = torch.rand(x.shape) < (corruption_prob * (1.0 - guaranteed_unmasked))
    corrupted_x = x.clone()
    
    if mask.sum() > 0:
        vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
        random_tokens = torch.randint(0, vocab_size_to_use, (mask.sum().item(),))
        corrupted_x[mask] = random_tokens
    
    return corrupted_x, mask

def apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size):
    """Strategy 2: Sticky-style corruption without transitions"""
    # Use sticky masking logic but replace mask tokens with random tokens
    sticky_corrupted_x, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)
    
    # Replace mask tokens with random tokens
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    for batch_idx in range(sticky_corrupted_x.shape[0]):
        for pos_idx in range(sticky_corrupted_x.shape[1]):
            if sticky_corrupted_x[batch_idx, pos_idx] == mask_token_id:
                sticky_corrupted_x[batch_idx, pos_idx] = torch.randint(0, vocab_size_to_use, (1,)).item()
    
    return sticky_corrupted_x, mask

def apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id):
    """Strategy 3: Fragment-based corruption using real text segments"""
    # Use sticky masking to get corruption patterns
    _, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)
    
    corrupted_x = x.clone()
    
    # For each sequence in the batch, get a different source fragment
    for batch_idx in range(x.shape[0]):
        batch_mask = mask[batch_idx]
        if batch_mask.sum() > 0:
            # Sample a random fragment from training data
            fragment_start = torch.randint(0, len(data) - block_size, (1,)).item()
            fragment = torch.from_numpy(data[fragment_start:fragment_start + block_size].astype(np.int64))
            
            # Replace corrupted positions with tokens from the fragment
            corrupted_x[batch_idx][batch_mask] = fragment[batch_mask]
    
    return corrupted_x, mask

def apply_gpu_masking_validation(x, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier, guaranteed_unmasked):
    """GPU-optimized validation masking with fixed 0.5 sticky ratio"""
    current_batch_size = x.shape[0]
    num_sticky_batches = int(current_batch_size * 0.5)  # Fixed 50% sticky ratio
    
    # Pre-allocate tensors on GPU
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    
    # Apply independent masking to first part of batch (vectorized)
    if num_sticky_batches < current_batch_size:
        masking_prob = 0.5 * (1.0 - guaranteed_unmasked)  # Fixed at middle of range
        indep_mask = torch.rand(x[:current_batch_size-num_sticky_batches].shape, device=x.device) < masking_prob
        masked_x[:current_batch_size-num_sticky_batches][indep_mask] = mask_token_id
        mask[:current_batch_size-num_sticky_batches] = indep_mask
    
    # Apply GPU-accelerated sticky masking to remaining part of batch
    if num_sticky_batches > 0:
        sticky_masked_x, sticky_mask = apply_sticky_masking_gpu(
            x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier
        )
        masked_x[-num_sticky_batches:] = sticky_masked_x
        mask[-num_sticky_batches:] = sticky_mask
    
    return masked_x, mask

def apply_gpu_masking_training(x, iter_num, mask_token_id, sticky_rounds, sticky_p1_p2_multiplier, 
                              guaranteed_unmasked, sticky_transition_start, sticky_transition_end):
    """GPU-optimized training masking with dynamic sticky ratio"""
    # Calculate sticky masking ratio based on current training iteration
    if iter_num < sticky_transition_start:
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        sticky_ratio = 1.0
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress
    
    # Pre-allocate tensors on GPU
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
    
    if sticky_ratio == 0.0:
        # Pure independent masking (fully vectorized)
        masking_prob = torch.rand(1, device=x.device).item() * (1.0 - guaranteed_unmasked)
        mask = torch.rand(x.shape, device=x.device) < masking_prob
        masked_x[mask] = mask_token_id
    elif sticky_ratio == 1.0:
        # Pure sticky masking
        masked_x, mask = apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)
    else:
        # Mixed strategy: some batches independent, some sticky
        current_batch_size = x.shape[0]
        num_sticky_batches = int(current_batch_size * sticky_ratio)
        
        # Apply independent masking to first part of batch (vectorized)
        if num_sticky_batches < current_batch_size:
            masking_prob = torch.rand(1, device=x.device).item() * (1.0 - guaranteed_unmasked)
            indep_mask = torch.rand(x[:current_batch_size-num_sticky_batches].shape, device=x.device) < masking_prob
            masked_x[:current_batch_size-num_sticky_batches][indep_mask] = mask_token_id
            mask[:current_batch_size-num_sticky_batches] = indep_mask
        
        # Apply GPU-accelerated sticky masking to remaining part of batch
        if num_sticky_batches > 0:
            sticky_masked_x, sticky_mask = apply_sticky_masking_gpu(
                x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier
            )
            masked_x[-num_sticky_batches:] = sticky_masked_x
            mask[-num_sticky_batches:] = sticky_mask
    
    return masked_x, mask

def apply_sticky_masking_gpu(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier):
    """GPU-optimized sticky masking with parallel batch processing"""
    batch_size, seq_len = x.shape
    device = x.device
    
    # Pre-allocate result tensors on GPU
    masked_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool, device=device)
    
    # Vectorized sticky masking parameters
    sticky_p1 = torch.rand(batch_size, device=device) * 0.3 + 0.1  # Range: 0.1 to 0.4
    sticky_p2 = sticky_p1 * sticky_p1_p2_multiplier  # Higher transition probability
    
    # Process all batches in parallel
    for round_idx in range(sticky_rounds):
        # Generate random values for all positions at once
        rand_vals = torch.rand(batch_size, seq_len, device=device)
        
        if round_idx == 0:
            # Initial masking: independent for all positions
            new_mask = rand_vals < sticky_p1.unsqueeze(1)
        else:
            # Sticky masking: higher probability near existing masks
            # Compute neighbor influence in parallel
            padded_mask = torch.nn.functional.pad(mask.float(), (1, 1), value=0)
            left_neighbors = padded_mask[:, :-2]
            right_neighbors = padded_mask[:, 2:]
            has_masked_neighbor = (left_neighbors + right_neighbors) > 0
            
            # Vectorized probability assignment
            probs = torch.where(has_masked_neighbor, 
                              sticky_p2.unsqueeze(1), 
                              sticky_p1.unsqueeze(1))
            new_mask = rand_vals < probs
        
        # Update mask and apply masking
        mask = mask | new_mask
        masked_x[new_mask] = mask_token_id
    
    return masked_x, mask

def apply_random_noise_to_unmasked_gpu(x, mask, noise_max_ratio, meta_vocab_size):
    """GPU-optimized random noise application to unmasked positions"""
    global iter_num
    
    # Calculate progressive noise ratio based on training iteration
    if iter_num < sticky_transition_start:
        progressive_noise_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        progressive_noise_ratio = noise_max_ratio
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        progressive_noise_ratio = progress * noise_max_ratio
    
    if progressive_noise_ratio <= 0.0:
        return x
    
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    
    # All operations on GPU, fully vectorized
    unmasked_positions = ~mask
    batch_size = x.shape[0]
    
    # Generate noise ratios for all batch elements at once
    noise_ratios = torch.rand(batch_size, device=x.device) * progressive_noise_ratio
    noise_ratios_expanded = noise_ratios.unsqueeze(1).expand(-1, x.shape[1])
    
    # Generate random probabilities for all positions at once
    random_probs = torch.rand_like(x, dtype=torch.float, device=x.device)
    
    # Determine which positions to noise (fully vectorized)
    should_noise = unmasked_positions & (random_probs < noise_ratios_expanded)
    
    # Apply noise in-place if any positions need noising
    if should_noise.any():
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=x.device)
        x = torch.where(should_noise, random_tokens, x)
    
    return x

def apply_random_noise_to_unmasked(x, mask, noise_max_ratio, meta_vocab_size):
    """Apply random token noise to unmasked positions in input for unmasking training.
    
    This helps the model learn to unmask tokens even when some context tokens are noisy.
    Only applied during training_type='unmasking' to improve robustness.
    """
    global iter_num
    
    # Calculate progressive noise ratio based on training iteration
    if iter_num < sticky_transition_start:
        # No noise during early training
        progressive_noise_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        # Full noise ratio after transition
        progressive_noise_ratio = noise_max_ratio
    else:
        # Gradual increase from 0 to noise_max_ratio
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        progressive_noise_ratio = progress * noise_max_ratio
    
    if progressive_noise_ratio <= 0.0:
        return x
    
    noisy_x = x.clone()
    vocab_size_to_use = meta_vocab_size if meta_vocab_size is not None else 50257
    
    # Get unmasked positions for all batch elements
    unmasked_positions = ~mask  # Shape: (batch_size, seq_len)
    
    # Sample noise ratios for each batch element using progressive ratio
    batch_size = x.shape[0]
    noise_ratios = torch.rand(batch_size, device=x.device) * progressive_noise_ratio  # Shape: (batch_size,)
    
    # For each position, determine if it should be noised
    # First, generate random values for all unmasked positions
    random_probs = torch.rand_like(x, dtype=torch.float, device=x.device)  # Shape: (batch_size, seq_len)
    
    # Create noise mask: position gets noised if it's unmasked AND random_prob < noise_ratio
    noise_ratios_expanded = noise_ratios.unsqueeze(1).expand(-1, x.shape[1])  # Shape: (batch_size, seq_len)
    should_noise = unmasked_positions & (random_probs < noise_ratios_expanded)
    
    # Generate random tokens for all positions that should be noised
    if should_noise.any():
        random_tokens = torch.randint(0, vocab_size_to_use, x.shape, device=x.device)
        noisy_x = torch.where(should_noise, random_tokens, noisy_x)
    
    return noisy_x

def load_synthetic_model(checkpoint_path, device, extended_vocab_size):
    """Load the synthetic model for generating fake data in remasking training"""
    global synthetic_model
    
    if not checkpoint_path or synthetic_model is not None:
        return
    
    try:
        print(f"Loading synthetic model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model arguments from checkpoint
        checkpoint_model_args = checkpoint['model_args']
        
        # Create synthetic model with same architecture as checkpoint
        synthetic_gptconf = GPTConfig(**checkpoint_model_args)
        synthetic_model = GPT(synthetic_gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        # Fix keys if needed (same as main model loading)
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        synthetic_model.load_state_dict(state_dict)
        synthetic_model.to(device)
        synthetic_model.eval()  # Always in eval mode
        
        print(f"Synthetic model loaded successfully (vocab_size: {synthetic_model.config.vocab_size})")
        
    except Exception as e:
        print(f"Warning: Could not load synthetic model from {checkpoint_path}: {e}")
        synthetic_model = None

def apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size):
    """Strategy 4: Synthetic corruption using loaded unmasking model"""
    global synthetic_model
    
    if synthetic_model is None:
        print("Warning: Synthetic model not loaded, falling back to sticky corruption")
        return apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    
    # Use sticky masking to get corruption patterns
    _, mask = apply_sticky_masking(x, sticky_rounds, mask_token_id, sticky_p1_p2_multiplier)
    
    # Create input for synthetic model: replace masked positions with mask tokens
    synthetic_input = x.clone()
    synthetic_input[mask] = mask_token_id
    
    # Generate synthetic data using the loaded model
    with torch.no_grad():
        # Move to device if needed
        if synthetic_input.device != next(synthetic_model.parameters()).device:
            synthetic_input = synthetic_input.to(next(synthetic_model.parameters()).device)
        
        # Get logits from synthetic model
        logits, _ = synthetic_model(synthetic_input, None)
        
        # Sample from the model's distribution
        # Use temperature sampling for more realistic synthetic data
        temperature = 0.8
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        
        # Sample tokens for masked positions - vectorized approach
        corrupted_x = x.clone()
        
        # Create a sampling mask and sample all at once
        if mask.any():
            # Get flattened indices where mask is True
            mask_flat = mask.view(-1)
            probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
            
            # Sample for all masked positions at once
            masked_probs = probs_flat[mask_flat]  # (num_masked_total, vocab_size)
            sampled_tokens = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
            
            # Move sampled tokens to same device as corrupted_x
            sampled_tokens = sampled_tokens.to(corrupted_x.device)
            
            # Place sampled tokens back
            corrupted_x_flat = corrupted_x.view(-1)
            corrupted_x_flat[mask_flat] = sampled_tokens
            corrupted_x = corrupted_x_flat.view_as(x)
    
    return corrupted_x, mask

def get_batch_remasking(split):
    global _val_batch_cache, iter_num

    # For validation, use cached batch to ensure consistency
    if split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Load data same as unmasking
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Find valid starting indices that begin with double newlines
    valid_indices = find_double_newline_indices(data, meta_vocab_size)
    
    if len(valid_indices) == 0:
        # Fallback to original random sampling if no double newlines found
        print("Warning: No double newlines found, falling back to random sampling")
        if split == 'val':
            torch.manual_seed(42)
            ix = torch.randint(len(data) - block_size, (batch_size,))
            torch.manual_seed(1337 + seed_offset)
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        # Sample from valid double-newline starting positions
        if split == 'val':
            # For validation, use fixed seed to ensure reproducible indices
            torch.manual_seed(42)
            ix_indices = torch.randint(len(valid_indices), (batch_size,))
            ix = torch.from_numpy(valid_indices[ix_indices.numpy()])
            # Reset to original seed
            torch.manual_seed(1337 + seed_offset)
        else:
            # For training, use random indices from valid positions
            ix_indices = torch.randint(len(valid_indices), (batch_size,))
            ix = torch.from_numpy(valid_indices[ix_indices.numpy()])

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    # Select corruption strategy based on configuration
    if remasking_corruption_strategy == 'random':
        corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)
    elif remasking_corruption_strategy == 'sticky':
        corrupted_x, mask = apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    elif remasking_corruption_strategy == 'fragment':
        corrupted_x, mask = apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id)
    elif remasking_corruption_strategy == 'synthetic':
        corrupted_x, mask = apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    elif remasking_corruption_strategy == 'mixed':
        # Select strategy based on weights
        strategy_choice = np.random.choice(['random', 'sticky', 'fragment', 'synthetic'], p=remasking_strategy_weights)
        
        if strategy_choice == 'random':
            corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)
        elif strategy_choice == 'sticky':
            corrupted_x, mask = apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
        elif strategy_choice == 'fragment':
            corrupted_x, mask = apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id)
        else:  # synthetic
            corrupted_x, mask = apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    else:
        # Default fallback to random
        corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)

    # Target: original tokens at correct positions, wrong_token_id at corrupted positions
    y = x.clone()
    y[mask] = wrong_token_id

    # Move to device
    if device_type == 'cuda':
        corrupted_x = corrupted_x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        corrupted_x, y, mask = corrupted_x.to(device), y.to(device), mask.to(device)

    # Cache validation batch for consistency
    if split == 'val':
        _val_batch_cache = (corrupted_x, y, mask)

    return corrupted_x, y, mask

def get_batch_remasking_binary(split):
    """Remasking binary training: symmetric task with remask_good_id and remask_wrong_id targets"""
    global _val_batch_cache, iter_num

    # For validation, use cached batch to ensure consistency
    if split == 'val' and _val_batch_cache is not None:
        return _val_batch_cache

    # Load data same as other training types
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Find valid starting indices that begin with double newlines
    valid_indices = find_double_newline_indices(data, meta_vocab_size)
    
    if len(valid_indices) == 0:
        # Fallback to original random sampling if no double newlines found
        print("Warning: No double newlines found, falling back to random sampling")
        if split == 'val':
            torch.manual_seed(42)
            ix = torch.randint(len(data) - block_size, (batch_size,))
            torch.manual_seed(1337 + seed_offset)
        else:
            ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        # Sample from valid double-newline starting positions
        if split == 'val':
            # For validation, use fixed seed to ensure reproducible indices
            torch.manual_seed(42)
            ix_indices = torch.randint(len(valid_indices), (batch_size,))
            ix = torch.from_numpy(valid_indices[ix_indices.numpy()])
            # Reset to original seed
            torch.manual_seed(1337 + seed_offset)
        else:
            # For training, use random indices from valid positions
            ix_indices = torch.randint(len(valid_indices), (batch_size,))
            ix = torch.from_numpy(valid_indices[ix_indices.numpy()])

    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])

    # Select corruption strategy based on configuration (same as remasking)
    if remasking_corruption_strategy == 'random':
        corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)
    elif remasking_corruption_strategy == 'sticky':
        corrupted_x, mask = apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    elif remasking_corruption_strategy == 'fragment':
        corrupted_x, mask = apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id)
    elif remasking_corruption_strategy == 'synthetic':
        corrupted_x, mask = apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    elif remasking_corruption_strategy == 'mixed':
        # Select strategy based on weights
        strategy_choice = np.random.choice(['random', 'sticky', 'fragment', 'synthetic'], p=remasking_strategy_weights)
        
        if strategy_choice == 'random':
            corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)
        elif strategy_choice == 'sticky':
            corrupted_x, mask = apply_sticky_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
        elif strategy_choice == 'fragment':
            corrupted_x, mask = apply_fragment_corruption(x, data, block_size, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id)
        else:  # synthetic
            corrupted_x, mask = apply_synthetic_corruption(x, sticky_rounds, sticky_p1_p2_multiplier, mask_token_id, meta_vocab_size)
    else:
        # Default fallback to random
        corrupted_x, mask = apply_random_corruption(x, 0.5, guaranteed_unmasked, meta_vocab_size)

    # Binary targets: remask_good_id for uncorrupted, remask_wrong_id for corrupted
    y = torch.full_like(x, remask_good_id)  # Initialize all positions as "good"
    y[mask] = remask_wrong_id  # Mark corrupted positions as "wrong"

    # Move to device
    if device_type == 'cuda':
        corrupted_x = corrupted_x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        corrupted_x, y, mask = corrupted_x.to(device), y.to(device), mask.to(device)

    # Cache validation batch for consistency
    if split == 'val':
        _val_batch_cache = (corrupted_x, y, mask)

    return corrupted_x, y, mask

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
            # For binary classification and remasking, track corruption statistics
            if training_type in ['remasking_binary', 'remasking']:
                total_positions = 0
                corrupted_positions = 0
            else:
                random_prob = 1.0 / extended_vocab_size  # Random chance probability

        for k in range(eval_iters):
            with timer.time_function('validation_data_generation'):
                X, Y, mask = get_batch(split)  # Updated to return mask
            with ctx:
                with timer.time_function('validation_forward_pass'):
                    # This is handled in validation_loss_computation section
                    pass
                with timer.time_function('validation_loss_computation'):
                    # Get logits without internal loss computation
                    logits, _ = model(X, None)
                    
                    # Ensure we have full sequence logits for both attention types
                    if logits.size(1) == 1:  # Only last position (causal inference mode)
                        # Force full sequence by passing targets
                        logits, _ = model(X, Y)
                    
                    # Compute loss based on training type
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = Y.view(-1)
                    mask_flat = mask.view(-1)
                    
                    if training_type == 'unmasking':
                        # Unmasking: compute loss only on masked positions
                        if mask_flat.sum() > 0:
                            masked_logits = logits_flat[mask_flat]
                            masked_targets = targets_flat[mask_flat]
                            loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
                        else:
                            # Fallback if no masked tokens (shouldn't happen)
                            loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                            
                    elif training_type == 'remasking':
                        # Compute loss on all positions (always for remasking)
                        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                        
                    elif training_type == 'remasking_binary':
                        # Binary classification: compute loss on all positions
                        # Symmetric task: predict remask_good_id or remask_wrong_id
                        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

                # For validation, compute model vs random statistics
                if split == 'val':
                    # Get probabilities from logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
                    probs_flat = probs.view(-1, probs.size(-1))  # (batch_size * seq_len, vocab_size)
                    
                    if training_type == 'remasking_binary':
                        # For binary classification, compute accuracy on all positions
                        # Track corruption statistics for proper baseline
                        total_positions += targets_flat.numel()
                        corrupted_positions += (targets_flat == remask_wrong_id).sum().item()
                        
                        # Get probabilities for correct binary classification
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                    elif training_type == 'remasking':
                        # For remasking, compute accuracy on ALL positions (corrupted + uncorrupted)
                        # Track corruption statistics for proper baseline
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        total_positions += targets_flat.numel()
                        corrupted_positions += mask_flat.sum().item()  # mask indicates corrupted positions
                        
                        # Get probabilities for correct predictions at ALL positions
                        correct_token_probs = probs_flat[range(len(targets_flat)), targets_flat]
                        model_probs.extend(correct_token_probs.cpu().tolist())
                    else:
                        # For unmasking, compute on masked positions only
                        mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                        masked_positions = mask_flat.bool()
                        if masked_positions.sum() > 0:  # Only if there are masked tokens
                            correct_token_probs = probs_flat[masked_positions, targets_flat[masked_positions]]
                            model_probs.extend(correct_token_probs.cpu().tolist())

            losses[k] = loss.item()

        out[split] = losses.mean()

        # Add model vs random comparison for validation
        if split == 'val' and model_probs:
            avg_model_prob = sum(model_probs) / len(model_probs)
            
            if training_type == 'remasking_binary':
                # For binary classification, compare against distribution-aware random baseline
                corruption_ratio = corrupted_positions / total_positions if total_positions > 0 else 0.0
                # Random classifier matching the distribution would get:
                # P(correct) = P(guess_good) * P(actual_good) + P(guess_wrong) * P(actual_wrong)
                # With optimal random strategy: P(guess_good) = P(actual_good), P(guess_wrong) = P(actual_wrong)
                random_accuracy = (1 - corruption_ratio) ** 2 + corruption_ratio ** 2
                prob_ratio = avg_model_prob / random_accuracy if random_accuracy > 0 else float('inf')
                out[f'{split}_model_vs_random'] = prob_ratio
                out[f'{split}_avg_correct_prob'] = avg_model_prob
                out[f'{split}_corruption_ratio'] = corruption_ratio
                out[f'{split}_random_baseline'] = random_accuracy
            elif training_type == 'remasking':
                # For remasking, the task is corruption detection + appropriate response
                corruption_ratio = corrupted_positions / total_positions if total_positions > 0 else 0.0
                # Optimal random baseline: always guess the majority class
                # With corruption_ratio=0.2: always guess "uncorrupted" → 80% accuracy
                # General: max(corruption_ratio, 1-corruption_ratio)
                random_accuracy = max(corruption_ratio, 1 - corruption_ratio)
                prob_ratio = avg_model_prob / random_accuracy if random_accuracy > 0 else float('inf')
                out[f'{split}_model_vs_random'] = prob_ratio
                out[f'{split}_avg_correct_prob'] = avg_model_prob
                out[f'{split}_corruption_ratio'] = corruption_ratio
                out[f'{split}_random_baseline'] = random_accuracy
            else:
                # For unmasking, use uniform random baseline
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
            if training_type in ['remasking_binary', 'remasking']:
                print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
                print(f"  val accuracy: {losses['val_avg_correct_prob']:.4f} (random baseline: {losses.get('val_random_baseline', 0.0):.4f})")
                print(f"  val corruption ratio: {losses.get('val_corruption_ratio', 0.0):.4f}")
            else:
                print(f"  val model vs random: {losses['val_model_vs_random']:.2f}x better")
                print(f"  val avg correct prob: {losses['val_avg_correct_prob']:.4f} (random: {1.0/extended_vocab_size:.4f})")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "model vs random": losses.get('val_model_vs_random', 0.0),
                "mfu": running_mfu*100, # convert to percentage
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
                if training_type == 'remasking':
                    ckpt_filename = f'ckpt_remasking_{iter_num}.pt'
                elif training_type == 'remasking_binary':
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
                # This is handled in loss_computation section
                pass
            with timer.time_function('loss_computation'):
                # Get logits without internal loss computation
                logits, _ = model(X, None)
                
                # Ensure we have full sequence logits for both attention types
                if logits.size(1) == 1:  # Only last position (causal inference mode)
                    # Force full sequence by passing targets
                    logits, _ = model(X, Y)
                
                # Compute loss based on training type
                logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
                targets_flat = Y.view(-1)  # (batch_size * seq_len,)
                mask_flat = mask.view(-1)  # (batch_size * seq_len,)
                
                if training_type == 'unmasking':
                    # Unmasking: compute loss only on masked positions
                    if mask_flat.sum() > 0:
                        masked_logits = logits_flat[mask_flat]
                        masked_targets = targets_flat[mask_flat]
                        loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
                    else:
                        # Fallback if no masked tokens (shouldn't happen)
                        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                        
                elif training_type == 'remasking':
                    # Compute loss on all positions (always for remasking)
                    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                    
                elif training_type == 'remasking_binary':
                    # Binary classification: compute loss on all positions
                    # Symmetric task: predict remask_good_id or remask_wrong_id
                    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
                        
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
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# Cleanup prefetch thread
stop_prefetch()
