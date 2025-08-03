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

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# enhanced data augmentation
enhanced_data_probability = 0.0 # probability of using enhanced vs natural data (0.0 = disabled)
min_prefix_length = 50 # minimum prefix length for enhanced data generation
max_prefix_length = 950 # maximum prefix length for enhanced data generation
enhanced_generation_temperature = 0.8 # temperature for enhanced data generation
enhanced_generation_top_k = 200 # top_k for enhanced data generation
enhanced_buffer_size = 1000 # maximum number of pre-generated enhanced samples in rotating buffer
enhanced_generation_batch_size = 32 # batch size for parallel enhanced sample generation
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
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

# Enhanced data augmentation classes and functions
import threading
import queue
import random
from collections import deque

class EnhancedSampleBuffer:
    """Thread-safe rotating buffer for pre-generated enhanced samples"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_consumed = 0
        self.total_generated = 0
    
    def get_samples(self, n):
        """Get n random samples from buffer, returns list of (x, y) tensor pairs"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            # Sample with replacement to reuse samples if generation is slow
            available = len(self.buffer)
            samples = []
            
            for _ in range(n):
                idx = random.randint(0, available - 1)
                samples.append(self.buffer[idx])
            
            self.total_consumed += n
            return samples
    
    def add_samples(self, samples):
        """Add samples to buffer, oldest samples are automatically evicted if full"""
        with self.lock:
            for sample in samples:
                self.buffer.append(sample)
            self.total_generated += len(samples)
    
    def clear(self):
        """Clear all samples from buffer"""
        with self.lock:
            self.buffer.clear()
            # Reset counters when buffer is cleared (e.g., model update)
            self.total_consumed = 0
            self.total_generated = 0
    
    def is_full(self):
        """Check if buffer is at maximum capacity"""
        with self.lock:
            return len(self.buffer) >= self.max_size
    
    def size(self):
        """Get current number of samples in buffer"""
        with self.lock:
            return len(self.buffer)
    
    def get_consumption_stats(self):
        """Get generation and consumption statistics"""
        with self.lock:
            return self.total_generated, self.total_consumed

class EnhancedSampleGenerator:
    """Background generator for enhanced samples"""
    
    def __init__(self, buffer, inference_model, data, config, device, ctx):
        self.buffer = buffer
        self.inference_model = inference_model
        self.data = data
        self.config = config
        self.device = device
        self.ctx = ctx
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        
        # Adaptive generation timing
        self.last_generated = 0
        self.last_consumed = 0
        self.sleep_time = 0.1  # Start with 100ms
        self.min_sleep = 0.01  # Minimum 10ms
        self.max_sleep = 1.0   # Maximum 1s
    
    def start(self):
        """Start background generation thread"""
        if self.running:
            return
        
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background generation thread"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def update_model(self, new_inference_model):
        """Update inference model and clear buffer"""
        self.inference_model = new_inference_model
        self.buffer.clear()
        # Reset counters when model is updated
        self.last_generated = 0
        self.last_consumed = 0
    
    def is_running(self):
        """Check if generator is running"""
        return self.running
    
    def _generation_loop(self):
        """Main generation loop running in background thread"""
        import time
        last_log_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Get current stats
                current_generated, current_consumed = self.buffer.get_consumption_stats()
                
                # Calculate recent activity
                recent_generated = current_generated - self.last_generated
                recent_consumed = current_consumed - self.last_consumed
                
                # Adjust sleep time based on generation vs consumption balance
                # Target: generated = 2 * consumed
                if recent_consumed > 0:
                    target_generated = 2 * recent_consumed
                    if recent_generated < target_generated:
                        # Need to generate more, reduce sleep time
                        self.sleep_time = max(self.min_sleep, self.sleep_time * 0.8)
                    elif recent_generated > target_generated * 1.5:
                        # Generating too much, increase sleep time
                        self.sleep_time = min(self.max_sleep, self.sleep_time * 1.2)
                
                # Only generate if buffer is not full
                if not self.buffer.is_full():
                    # Generate a batch of enhanced samples
                    samples = generate_enhanced_samples_batch(
                        self.inference_model, 
                        self.data, 
                        self.config['enhanced_generation_batch_size'],
                        self.config['min_prefix_length'],
                        self.config['max_prefix_length'],
                        self.config['block_size'],
                        self.config['enhanced_generation_temperature'],
                        self.config['enhanced_generation_top_k'],
                        self.device,
                        self.ctx
                    )
                    
                    if samples:
                        self.buffer.add_samples(samples)
                        # Update global generation count
                        enhanced_stats['enhanced_samples_generated'] += len(samples)
                
                # Update tracking
                self.last_generated = current_generated
                self.last_consumed = current_consumed
                
                # Optionally log detailed stats every 30 seconds (less frequent)
                current_time = time.time()
                if current_time - last_log_time >= 30.0:
                    print(f"Enhanced data generator: sleep_time={self.sleep_time:.3f}s, buffer_size={self.buffer.size()}")
                    last_log_time = current_time
                
                # Adaptive sleep
                self.stop_event.wait(self.sleep_time)
                
            except Exception as e:
                print(f"Enhanced sample generation error: {e}")
                self.stop_event.wait(1.0)  # Wait longer on error

def determine_batch_composition(batch_size, probability):
    """Returns boolean mask indicating which batch elements should be enhanced"""
    if probability <= 0.0:
        return torch.zeros(batch_size, dtype=torch.bool)
    elif probability >= 1.0:
        return torch.ones(batch_size, dtype=torch.bool)
    else:
        return torch.rand(batch_size) < probability

def generate_enhanced_samples_batch(inference_model, data, batch_size, min_prefix_length, 
                                   max_prefix_length, block_size, temperature, top_k, device, ctx):
    """Generate batch of enhanced samples for buffer"""
    if inference_model is None or len(data) <= max_prefix_length:
        return []
    
    samples = []
    
    try:
        with torch.no_grad():
            with ctx:
                for _ in range(batch_size):
                    # Random prefix length
                    prefix_length = random.randint(min_prefix_length, max_prefix_length)
                    
                    # Random starting position for prefix
                    start_idx = random.randint(0, len(data) - prefix_length - 1)
                    
                    # Extract prefix
                    prefix_data = data[start_idx:start_idx + prefix_length]
                    prefix_tensor = torch.from_numpy(prefix_data.astype(np.int64)).unsqueeze(0).to(device)
                    
                    # Generate continuation
                    generated = inference_model.generate(
                        prefix_tensor, 
                        block_size, 
                        temperature=temperature, 
                        top_k=top_k
                    )
                    
                    # Extract random fragment of block_size
                    generated_seq = generated[0].cpu()
                    fragment = sample_random_fragments([generated_seq], block_size)
                    
                    if fragment:
                        x = fragment[0]
                        y = torch.cat([x[1:], torch.tensor([x[-1]])])  # Shift by 1 for target
                        samples.append((x.to(device), y.to(device)))
    
    except Exception as e:
        print(f"Error generating enhanced samples: {e}")
        return []
    
    return samples

def sample_random_fragments(sequences, block_size):
    """Extract random fragments of exact block_size from generated sequences"""
    fragments = []
    
    for seq in sequences:
        if len(seq) < block_size:
            continue
        
        # Random starting position
        max_start = len(seq) - block_size
        start_idx = random.randint(0, max_start)
        fragment = seq[start_idx:start_idx + block_size]
        fragments.append(fragment)
    
    return fragments

# Global enhanced data components (initialized later if needed)
enhanced_sample_buffer = None
enhanced_sample_generator = None

# Global statistics tracking
enhanced_stats = {
    'total_samples_in_batches': 0,
    'enhanced_samples_in_batches': 0,
    'enhanced_samples_generated': 0
}
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Enhanced data augmentation for training split
    if split == 'train' and enhanced_data_probability > 0.0 and enhanced_sample_buffer is not None:
        # Determine which batch elements should be enhanced vs natural
        enhanced_mask = determine_batch_composition(batch_size, enhanced_data_probability)
        n_enhanced = enhanced_mask.sum().item()
        n_natural = batch_size - n_enhanced
        
        # Update global statistics
        enhanced_stats['total_samples_in_batches'] += batch_size
        enhanced_stats['enhanced_samples_in_batches'] += n_enhanced
        
        # Get natural samples
        natural_x = []
        natural_y = []
        if n_natural > 0:
            ix_natural = torch.randint(len(data) - block_size, (n_natural,))
            natural_x = [torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix_natural]
            natural_y = [torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix_natural]
        
        # Get enhanced samples from buffer (reuse samples if not enough available)
        enhanced_samples = []
        if n_enhanced > 0:
            enhanced_samples = enhanced_sample_buffer.get_samples(n_enhanced)
        
        # Combine natural and enhanced samples according to mask
        x_batch = []
        y_batch = []
        natural_idx = 0
        enhanced_idx = 0
        
        for i in range(batch_size):
            if enhanced_mask[i] and enhanced_idx < len(enhanced_samples):
                # Use enhanced sample
                x_sample, y_sample = enhanced_samples[enhanced_idx]
                enhanced_idx += 1
            else:
                # Use natural sample (either mask[i] is False or no enhanced samples available)
                if natural_idx < len(natural_x):
                    x_sample = natural_x[natural_idx]
                    y_sample = natural_y[natural_idx]
                    natural_idx += 1
                else:
                    # Generate new natural sample if needed
                    rand_idx = torch.randint(len(data) - block_size, (1,)).item()
                    x_sample = torch.from_numpy((data[rand_idx:rand_idx+block_size]).astype(np.int64))
                    y_sample = torch.from_numpy((data[rand_idx+1:rand_idx+1+block_size]).astype(np.int64))
            
            x_batch.append(x_sample)
            y_batch.append(y_sample)
        
        x = torch.stack(x_batch)
        y = torch.stack(y_batch)
    else:
        # Original behavior for validation or when enhanced data is disabled
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

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
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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

# initialize enhanced data augmentation if enabled
inference_model = None
if enhanced_data_probability > 0.0:
    print("Initializing enhanced data augmentation...")
    
    # Create inference model copy
    if init_from == 'scratch':
        inference_gptconf = GPTConfig(**model_args)
        inference_model = GPT(inference_gptconf)
        inference_model.load_state_dict(model.state_dict())
    elif init_from == 'resume':
        inference_gptconf = GPTConfig(**model_args)
        inference_model = GPT(inference_gptconf)
        inference_model.load_state_dict(model.state_dict())
    elif init_from.startswith('gpt2'):
        override_args = dict(dropout=0.0)
        inference_model = GPT.from_pretrained(init_from, override_args)
    
    inference_model.eval()
    inference_model.to(device)
    
    # Initialize enhanced data components
    enhanced_sample_buffer = EnhancedSampleBuffer(enhanced_buffer_size)
    
    # Load training data for enhanced sample generation
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Create enhanced sample generator config
    enhanced_config = {
        'enhanced_generation_batch_size': enhanced_generation_batch_size,
        'min_prefix_length': min_prefix_length,
        'max_prefix_length': max_prefix_length,
        'block_size': block_size,
        'enhanced_generation_temperature': enhanced_generation_temperature,
        'enhanced_generation_top_k': enhanced_generation_top_k,
    }
    
    enhanced_sample_generator = EnhancedSampleGenerator(
        enhanced_sample_buffer,
        inference_model,
        train_data,
        enhanced_config,
        device,
        ctx
    )
    
    # Start background generation
    enhanced_sample_generator.start()
    print(f"Enhanced data augmentation initialized with probability {enhanced_data_probability}")

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
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
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

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Enhanced data statistics
        if enhanced_data_probability > 0.0:
            total_samples = enhanced_stats['total_samples_in_batches']
            enhanced_samples = enhanced_stats['enhanced_samples_in_batches']
            generated_samples = enhanced_stats['enhanced_samples_generated']
            enhanced_ratio = (enhanced_samples / total_samples * 100) if total_samples > 0 else 0
            buffer_size = enhanced_sample_buffer.size() if enhanced_sample_buffer else 0
            print(f"🔥 ENHANCED DATA: Total={total_samples}, Enhanced={enhanced_samples} ({enhanced_ratio:.1f}%), Generated={generated_samples}, Buffer={buffer_size}")
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
                
                # Update inference model with new checkpoint if enhanced data is enabled
                if enhanced_data_probability > 0.0 and enhanced_sample_generator is not None:
                    print("Updating inference model with new checkpoint...")
                    new_inference_model = GPT(GPTConfig(**model_args))
                    new_inference_model.load_state_dict(raw_model.state_dict())
                    new_inference_model.eval()
                    new_inference_model.to(device)
                    enhanced_sample_generator.update_model(new_inference_model)
                    print("Inference model updated and buffer cleared")
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
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
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
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Cleanup enhanced data augmentation
if enhanced_data_probability > 0.0 and enhanced_sample_generator is not None:
    print("Stopping enhanced sample generator...")
    enhanced_sample_generator.stop()

if ddp:
    destroy_process_group()
