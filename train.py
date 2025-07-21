import os
import time
import random
import math
import pickle
import json
import yaml
from contextlib import nullcontext
from datetime import datetime
import numpy as np
import torch
import torch._dynamo
try:
    torch._dynamo.config.recompile_limit = 1000000 # or a higher number
except AttributeError:
    # recompile_limit doesn't exist in this version of PyTorch
    pass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from logger import TrainingLogger
from analyzer import ModelAnalyzer

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
# file logging
log_dir = 'logs' # directory for log files
file_logging = True # enable file logging
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'fineweb10B'
train_shard_filenames = ['train.bin']
num_train_shards = 1

gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
n_hidden = None # feed forward hidden dimension, defaults to 4 * n_embd if None
# rotary embeddings
use_rotary_embeddings = False
rotary_base = 10000.0
rotary_max_position_embeddings = 2048
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
# LoRA architectural parameters. These will be overridden by config files.
embedding_mode = 'standard'
attn_lora_rank = 0 # rank for attention LoRA, 0 disables
embedding_rank = 0 # rank for embedding LoRA, 0 disables
lora_alpha = 1.0 # scaling factor for LoRA layers
# scaling schedule configuration
scaling_schedule_file = None # Path to scaling schedule config file (YAML/JSON)
scaling_schedule = [] # Will be loaded from file or set programmatically
target_architecture_config = None # Global state for target architecture
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging




if(dataset == 'fineweb10B'):
    num_train_shards = 103
    train_shard_filenames = [f"fineweb_train_{i:06d}.bin" for i in range(1, num_train_shards + 1)]
else:
    num_train_shards = len(train_shard_filenames)
      
def log_detailed_params(model_to_log):
    """Logs the detailed parameter count of the provided model."""
    if master_process:
        print("\nDetailed parameter count:")
        detailed_params = model_to_log.get_detailed_param_count()
        for component, counts in detailed_params.items():
            total_str = f"{counts['total']:,}"
            trainable_str = f"{counts['trainable']:,}"
            print(f"  {component:<22} | Total: {total_str:>12} | Trainable: {trainable_str:>12}")
        print("-" * 60) # Add a separator for clarity

def transfer_optimizer_state(new_optimizer, old_state_dict, old_param_dict, model):
    """
    Transfer optimizer state from old optimizer to new optimizer for parameters that still exist.
    Uses parameter names as the bridge instead of object identity to handle architectural changes.
    """
    if 'state' not in old_state_dict:
        return
    
    # Map old parameter names to their state from the old optimizer state_dict
    state_to_transfer = {}
    old_param_id_to_name = {id(p): name for name, p in old_param_dict.items()}
    
    for param_id, state in old_state_dict['state'].items():
        if param_id in old_param_id_to_name:
            param_name = old_param_id_to_name[param_id]
            state_to_transfer[param_name] = state
    
    # For each parameter in the new model, find its state by name and apply it
    transferred_count = 0
    new_param_name_map = {name: p for name, p in model.named_parameters()}
    
    for param_name, state in state_to_transfer.items():
        if param_name in new_param_name_map:
            param_tensor = new_param_name_map[param_name]
            # Directly set the state in the optimizer
            new_optimizer.state[param_tensor] = state
            transferred_count += 1
    
    total_params = len(list(model.parameters()))
    print(f"Transferred optimizer state for {transferred_count} / {total_params} parameters")

def transfer_optimizer_state_by_shape(new_optimizer, old_state_dict, model):
    """
    Fallback function to transfer optimizer state when parameter names are not available.
    Attempts to match parameters by shape and name similarity.
    """
    if 'state' not in old_state_dict:
        return 0

    # Get current model parameters
    current_params = {name: param for name, param in model.named_parameters()}

    # Try to match by parameter name (for parameters that haven't changed)
    transferred_count = 0

    # This is a simplified approach - we can't perfectly reconstruct the mapping
    # without the old parameter names, but we can try some heuristics
    print("Attempting basic optimizer state transfer by parameter name matching...")
    print("Note: This may not transfer all state due to architectural changes.")

    return transferred_count

def log_model_architecture(model, iter_num, is_initial=False, is_target=False):
    """Logs the model's current or target architecture to the console and W&B."""
    if not master_process:
        return

    # Determine the header based on the context
    if is_target:
        header = "TARGET MODEL ARCHITECTURE (at end of schedule)"
    elif is_initial:
        header = f"INITIAL MODEL ARCHITECTURE (at Iter {iter_num})"
    else:
        header = f"ARCHITECTURE CHANGE (at Iter {iter_num})"

    # Get the raw model config
    config = model.config if hasattr(model, 'config') else model

    print("\n" + "="*60)
    print(f"{header:^60}")
    print("="*60)

    arch_info = {
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'n_embd': config.n_embd,
        'n_hidden': config.n_hidden if config.n_hidden is not None else 4 * config.n_embd,
        'block_size': config.block_size,
        'vocab_size': config.vocab_size,
        'dropout': config.dropout,
        'bias': config.bias,
        'embedding_mode': config.embedding_mode,
        'attn_lora_rank': config.attn_lora_rank,
        'embedding_rank': config.embedding_rank,
        'lora_alpha': config.lora_alpha
    }

    # Print to console
    for key, value in arch_info.items():
        print(f"  {key:<22} | {value}")
    print("="*60 + "\n")

    # Log to Weights & Biases
    if wandb_log:
        # We prefix with 'arch/' to group these parameters in the W&B UI
        wandb_log_data = {f"{k}": v for k, v in arch_info.items()}
        wandb_log_data['iter'] = iter_num
        wandb.log(wandb_log_data)

def calculate_and_log_target_architecture(initial_config, schedule):
    """
    Simulates the schedule to determine the final target architecture,
    logs it, and returns it as a dictionary for later use.
    """
    if not master_process:
        return None

    # --- THIS SECTION IS REFINED TO BE LORA-AGNOSTIC ---
    # The target config reflects the final, deployed model, which has no LoRA.
    target_config = {
        'n_layer': initial_config['n_layer'],
        'n_hidden': initial_config['n_hidden'] if initial_config['n_hidden'] is not None else 4 * initial_config['n_embd'],
        'n_head': initial_config['n_head'],
        'n_embd': initial_config['n_embd'],
        'block_size': initial_config['block_size'],
        'vocab_size': initial_config['vocab_size'],
        'dropout': initial_config['dropout'],
        'bias': initial_config['bias'],
        # Hardcode final state for LoRA-related params
        'embedding_mode': 'standard',
        'attn_lora_rank': 0,
        'embedding_rank': 0,
        'lora_alpha': 0.0,
    }

    print("Calculating target architecture based on schedule...")
    for op in schedule:
        op_name = op['name']
        op_value = op['value']
        if op_name == 'stack_layers' and isinstance(op_value, list):
            target_config['n_layer'] = len(op_value)
        elif op_name == 'widen_mlp' and isinstance(op_value, (int, float)):
            target_config['n_hidden'] = int(op_value)

    # Log this calculated target architecture
    log_model_architecture(
        type('FakeConfig', (), target_config)(),
        iter_num=0,
        is_target=True
    )

    # Return the dictionary to be stored
    return target_config

def save_scaling_schedule(file_path, schedule_data):
    """Saves the updated schedule data back to its file (YAML or JSON).
    Automatically adds 'completed' field to operations that don't have it."""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        # Ensure all operations have a 'completed' field before saving
        for op in schedule_data:
            if 'completed' not in op:
                op['completed'] = False

        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(schedule_data, f, sort_keys=False)
            elif file_path.endswith('.json'):
                json.dump(schedule_data, f, indent=2)
        # print(f"Updated scaling schedule saved to {file_path}")
    except Exception as e:
        print(f"Error saving scaling schedule to {file_path}: {e}")

# Load scaling schedule if specified
def load_scaling_schedule(file_path, init_from):
    """Load scaling schedule and reset 'completed' status if starting from scratch."""
    if not file_path or not os.path.exists(file_path):
        return []
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                schedule = yaml.safe_load(f)
            elif file_path.endswith('.json'):
                schedule = json.load(f)
            else:
                print(f"Warning: Unknown file format for scaling schedule: {file_path}")
                return []
        if not isinstance(schedule, list):
            print(f"Warning: Scaling schedule must be a list, got {type(schedule)}")
            return []
        for i, op in enumerate(schedule):
            required_keys = ['name', 'value', 'trigger_loss', 'max_wait_iters', 'reevaluate']
            if not all(key in op for key in required_keys):
                print(f"Warning: Operation {i} missing required keys: {required_keys}")
                return []

        # Handle completion status based on init_from parameter
        # Note: Missing 'completed' fields are treated as False automatically
        if init_from == 'scratch':
            print("Starting from scratch, resetting schedule completion status.")
            for op in schedule:
                op['completed'] = False
        else:
            print("Resuming run, honoring existing schedule completion status.")
            # Ensure operations have 'completed' field for consistency (will be added on next save)
            for op in schedule:
                if 'completed' not in op:
                    op['completed'] = False

        print(f"Loaded scaling schedule with {len(schedule)} operations from {file_path}")
        return schedule
    except Exception as e:
        print(f"Error loading scaling schedule from {file_path}: {e}")
        return []

# Load scaling schedule
if scaling_schedule_file:
    scaling_schedule = load_scaling_schedule(scaling_schedule_file, init_from)

# Training Orchestrator State
iter_of_last_op = 0 # Iteration number when last operation was executed
lr_schedule_offset = 0 # Offset for learning rate schedule (for reset_lr_schedule)
# -----------------------------------------------------------------------------

# logging setup
training_logger = TrainingLogger(log_dir=log_dir, enabled=file_logging)

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
    training_logger.setup(config)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        shard_name = random.choice(train_shard_filenames)
        shard_path = os.path.join(data_dir, shard_name)
        data = np.memmap(shard_path, dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
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
                bias=bias, vocab_size=None, dropout=dropout, n_hidden=n_hidden,
                use_rotary_embeddings=use_rotary_embeddings,
                rotary_base=rotary_base,
                rotary_max_position_embeddings=rotary_max_position_embeddings,
                # --- ADD THESE LINES ---
                embedding_mode=embedding_mode,
                embedding_rank=embedding_rank,
                attn_lora_rank=attn_lora_rank,
                lora_alpha=lora_alpha
            )
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    # Force an update of the model args from the checkpoint for base architecture
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_hidden']:
        model_args[k] = checkpoint_model_args[k]

    # Create the model with the potentially new LoRA configuration from the config file
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']

    # --- SMART LOADER LOGIC ---
    print("Applying smart loader logic for model weights...")
    model_sd = model.state_dict()
    final_state_dict = {}

    for k, v in state_dict.items():
        # Case 1: The key from the checkpoint exists directly in the new model (e.g., non-LoRA to non-LoRA)
        if k in model_sd:
            final_state_dict[k] = v
        # Case 2: We are loading a standard weight into a LoRA layer's main_weight
        else:
            lora_key_equivalent = k.replace('.weight', '.main_weight.weight')
            if lora_key_equivalent in model_sd:
                print(f"  Remapping standard weight to LoRA: {k} -> {lora_key_equivalent}")
                final_state_dict[lora_key_equivalent] = v
            else:
                print(f"  Skipping unexpected key from checkpoint: {k}")

    # Remove the compilation wrapper prefix if it exists
    unwanted_prefix = '_orig_mod.'
    for k, v in list(final_state_dict.items()):
        if k.startswith(unwanted_prefix):
            final_state_dict[k[len(unwanted_prefix):]] = final_state_dict.pop(k)

    # Load the prepared state dict.
    # strict=False is essential, as LoRA A/B weights are expected to be missing.
    model.load_state_dict(final_state_dict, strict=False)

    # Load the rest of the training state
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'n_hidden']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    print("Attempting to load optimizer state...")
    try:
        # Try direct loading first
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Successfully loaded optimizer state directly from checkpoint")
    except (ValueError, RuntimeError) as e:
        print(f"Direct optimizer loading failed: {e}")
        print("This is expected when switching between LoRA and non-LoRA models.")

        # Attempt to transfer optimizer state using parameter names
        if 'param_names' in checkpoint:
            print("Attempting to transfer optimizer state using saved parameter names...")
            transfer_optimizer_state(optimizer, checkpoint['optimizer'], checkpoint['param_names'], model)
        else:
            print("No parameter names saved in checkpoint - this is an older checkpoint format.")
            transferred_count = transfer_optimizer_state_by_shape(optimizer, checkpoint['optimizer'], model)
            if transferred_count == 0:
                print("Could not transfer optimizer state. Optimizer will start fresh.")
                print("To enable full state transfer, re-save checkpoints with the updated format.")

checkpoint = None # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    actual_eval_iters = eval_iters
    for split in ['train', 'val']:
        losses = torch.zeros(actual_eval_iters)
        for k in range(actual_eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    effective_it = it - lr_schedule_offset
    actual_warmup_iters = warmup_iters
    actual_lr_decay_iters = lr_decay_iters

    if master_process and wandb_log:
        wandb.log({"iter": it, "effective_it": effective_it, "warmup_iters": actual_warmup_iters, "lr_decay_iters": actual_lr_decay_iters, "gradient_accumulation_steps":gradient_accumulation_steps, "batch_size":batch_size })
    if effective_it < actual_warmup_iters:
        return learning_rate * (effective_it + 1) / (actual_warmup_iters + 1)
    if effective_it > actual_lr_decay_iters:
        return min_lr
    decay_ratio = (effective_it - actual_warmup_iters) / (actual_lr_decay_iters - actual_warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (learning_rate - min_lr))





def execute_operation(op, trigger_reason, current_val_loss, iter_num, target_architecture_config):
    # Make globals mutable within this function
    global learning_rate, batch_size, gradient_accumulation_steps, warmup_iters, eval_iters, eval_interval
    global lr_schedule_offset, training_logger, master_process, model, optimizer, raw_model, unoptimized_model

    op_desc = op.get('desc', '')
    op_name = op['name']
    op_label = f"{op_name} {op_desc}"
    op_value = op['value']

    if master_process:
        print(f"\n--- EXECUTING OPERATION: {op_label} | Value: {op_value} ---")
        log_details = {
            'trigger_reason': trigger_reason,
            'current_val_loss': current_val_loss,
            'trigger_loss': op['trigger_loss'],
            'max_wait_iters': op['max_wait_iters']
        }
        training_logger.log_operation_start(iter_num, op_label, op_value, trigger_reason, current_val_loss,
                                          op['trigger_loss'], op['max_wait_iters'])

    try:
        # Check if this is an architectural operation
        architectural_ops = ['stack_layers', 'widen_mlp', 'set_attn_lora_rank',
                             'set_embedding_lora_rank', 'merge_lora_weights']

        if op_name in architectural_ops:
            if master_process:
                print(f"Performing architectural operation: {op_name}")

            unwrapped_model = unoptimized_model if compile else (model.module if ddp else model)
            old_optimizer_state = optimizer.state_dict()
            old_param_dict = {name: p for name, p in unwrapped_model.named_parameters()}

            # --- Perform the absolute architectural operation ---
            if op_name == 'stack_layers':
                unwrapped_model.stack_layers(op_value)
            elif op_name == 'widen_mlp':
                unwrapped_model.widen_mlp(op_value)
            elif op_name == 'set_attn_lora_rank':
                unwrapped_model.resize_lora_rank(op_value)
            elif op_name == 'set_embedding_lora_rank':
                unwrapped_model.resize_embedding_rank(op_value)
            elif op_name == 'merge_lora_weights':
                unwrapped_model.merge_lora_weights()

            # --- Re-create optimizer and wrappers (this logic remains the same) ---
            log_detailed_params(unwrapped_model)
            if master_process: print("Re-configuring optimizer...")
            optimizer = unwrapped_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
            if master_process: print("Transferring optimizer state...")
            transfer_optimizer_state(optimizer, old_optimizer_state, old_param_dict, unwrapped_model)

            model = unwrapped_model
            if compile:
                if master_process: print("Re-compiling the model...")
                model = torch.compile(model)
            if ddp:
                if master_process: print("Re-wrapping model in DDP...")
                model = DDP(model, device_ids=[ddp_local_rank])

            raw_model = model.module if ddp else model
            log_model_architecture(raw_model, iter_num)
            if master_process: training_logger.log_operation_success(iter_num, op_name, {'new_config': raw_model.config.__dict__})


        # --- Handle non-architectural (hyperparameter) operations ---
        else:
            if op_name == 'set_lr':
                if master_process: print(f"Learning rate: {learning_rate:.6f} -> {op_value:.6f}")
                learning_rate = op_value
            elif op_name == 'set_batch_size':
                if master_process: print(f"Batch size: {batch_size} -> {op_value}")
                batch_size = op_value
            elif op_name == 'set_grad_accum':
                if master_process: print(f"Grad accum steps: {gradient_accumulation_steps} -> {op_value}")
                gradient_accumulation_steps = op_value
            elif op_name == 'set_warmup_iters':
                if master_process: print(f"Warmup iters: {warmup_iters} -> {op_value}")
                warmup_iters = op_value
            elif op_name == 'set_eval_iters':
                if master_process: print(f"Eval iters: {eval_iters} -> {op_value}")
                eval_iters = op_value
            elif op_name == 'set_eval_interval':
                if master_process: print(f"Eval interval: {eval_interval} -> {op_value}")
                eval_interval = op_value
            elif op_name == 'reset_lr_schedule':
                if master_process: print(f"Resetting LR schedule at iter {iter_num}")
                lr_schedule_offset = iter_num
            else:
                raise ValueError(f"Unknown operation '{op_name}'")
            if master_process: training_logger.log_operation_success(iter_num, op_name, {'new_value': op_value})

        return True

    except ValueError as e:
        # Catch validation errors from the model methods (e.g., widening to smaller dim)
        error_msg = f"Operation '{op_name}' failed validation: {e}"
        if master_process:
            print(f"ERROR: {error_msg}")
            training_logger.log_operation_error(iter_num, op_name, error_msg)
        # We will not mark this operation as complete and let the program exit or continue
        # depending on your desired failure mode. For safety, we return False.
        # Consider adding `sys.exit(1)` if failure should be fatal.
        return False




if wandb_log and master_process:
    import wandb
    # Create a compact timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_wandb_run_name = f"{wandb_run_name}_{timestamp}"

    print(f"Initializing W&B run with name: {final_wandb_run_name}")
    wandb.init(project=wandb_project, name=final_wandb_run_name, config=config)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model

# Add logging for initial and target architecture
if master_process:
    # Calculate and store the target architecture
    target_architecture_config = calculate_and_log_target_architecture(model_args, scaling_schedule)
    # Log the initial model architecture
    log_model_architecture(raw_model, iter_num=0, is_initial=True)

log_detailed_params(raw_model)
running_mfu = -1.0
start_time = time.time() # Start the timer for elapsed time tracking
print(f"eval every: {eval_interval}")
while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # --- Model Analysis ---
            analyzer = ModelAnalyzer(raw_model)
            print (f"raw_model.config.n_layer={raw_model.config.n_layer}")

            for idx in range(raw_model.config.n_layer):
                print(f"doing something {idx}")
                attn_lora_layer = analyzer.model.transformer.h[idx].attn.c_attn
                attn_lora_res = analyzer.analyze_lora_update_rank(attn_lora_layer)
                eff_rank, full_rank, rank_util = analyzer.analyze_mlp_weight_rank(layer_idx=idx)
                if rank_util != -1.0:
                    print(f"  MLP Rank Utilization (L{idx}): {rank_util:.2%} ({eff_rank}/{full_rank})")

                if attn_lora_res[2] != -1.0:
                    print(f"{'Attn LoRA':<12} | {str(attn_lora_res[0])+'/'+str(attn_lora_res[1]):<15} | {attn_lora_res[2]:<12.2%}")

            # 5. Analyze Embedding Rank
            eff_rank, full_rank, rank_util = analyzer.analyze_embedding_rank()
            # 6. Display the results in a formatted block
            print("--- Model Analysis ---")
            if rank_util != -1.0:
                print(f"  Embedding Utilization: {rank_util:.2%} ({eff_rank}/{full_rank})")
            
            # 1. Analyze Embedding LoRA
            emb_lora_layer = analyzer.model.transformer.wte
            emb_lora_res = analyzer.analyze_lora_update_rank(emb_lora_layer)
            if emb_lora_res[2] != -1.0:
                 print(f"{'Embed LoRA':<12} | {str(emb_lora_res[0])+'/'+str(emb_lora_res[1]):<15} | {emb_lora_res[2]:<12.2%}")

            X_val, _ = get_batch('val')
            avg_entropy = analyzer.analyze_attention_entropy(X_val)

            if avg_entropy != -1.0:
                print(f"  Average Attention Entropy:  {avg_entropy:.4f}")
            print("----------------------")

            training_logger.log_step(iter_num, losses['train'], losses['val'])
            if wandb_log:
                elapsed_time_seconds = time.time() - start_time
                wandb_metrics = {
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,
                    "time/elapsed_seconds": elapsed_time_seconds, # Log elapsed time
                }
                # Add analysis metrics if they were computed successfully
                if rank_util != -1.0:
                    wandb_metrics["analysis/mlp_rank_utilization"] = rank_util
                if avg_entropy != -1.0:
                    wandb_metrics["analysis/attention_entropy"] = avg_entropy
                wandb.log(wandb_metrics)
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # --- MODIFICATION START ---
                    # Always save a universal, merged checkpoint
                    print("Creating universal checkpoint by merging LoRA weights...")
                    universal_state_dict = raw_model.get_merged_state_dict()

                    # Save parameter names to enable optimizer state transfer
                    param_names = {name: param for name, param in raw_model.named_parameters()}

                    checkpoint = {
                        'model': universal_state_dict, # Use the merged state dict
                        'optimizer': optimizer.state_dict(),
                        'param_names': param_names,  # Save parameter names for optimizer state transfer
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    # --- MODIFICATION END ---
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        # FIX: Wrapped orchestration logic in a while loop to handle consecutive non-blocking operations
        while True:
            op_to_run = [None]
            if master_process and scaling_schedule:
                # Find next uncompleted operation
                next_op = None
                for op in scaling_schedule:
                    if not op.get('completed', False):
                        next_op = op
                        break # Found the next operation to consider

                # If next_op is still None, all operations are complete
                if next_op is None:
                    if 'all_ops_done' not in globals(): # Log this message only once
                        print("All scheduled operations have been completed.")
                        globals()['all_ops_done'] = True
                else:
                    # We have an operation to consider, now check its trigger conditions
                    current_val_loss = losses['val']
                    loss_triggered = current_val_loss < next_op['trigger_loss']
                    timeout_triggered = (iter_num - iter_of_last_op) >= next_op['max_wait_iters']

                    if loss_triggered or timeout_triggered:
                        trigger_reason = 'Loss threshold' if loss_triggered else 'Timeout'
                        op_to_run[0] = {'op': next_op, 'reason': trigger_reason, 'loss': current_val_loss}
                    else:
                        print(f"{next_op['name']} {current_val_loss} {next_op['trigger_loss']} {next_op['max_wait_iters']}")

            if ddp:
                torch.distributed.broadcast_object_list(op_to_run, src=0)

            if op_to_run[0] is not None:
                op_data = op_to_run[0]
                next_op, trigger_reason, current_val_loss = op_data['op'], op_data['reason'], op_data['loss']
                if master_process:
                    print(f"\n=== SCALING OPERATION TRIGGERED (DDP SYNC) ===")
                    print(f"Operation: {next_op['name']}")
                    print(f"Trigger reason: {trigger_reason}")
                    print(f"Current val loss: {current_val_loss:.4f}, Trigger loss: {next_op['trigger_loss']:.4f}")
                    print(f"Iterations since last op: {iter_num - iter_of_last_op}, Max wait: {next_op['max_wait_iters']}")

                operation_succeeded = execute_operation(next_op, trigger_reason, current_val_loss, iter_num, target_architecture_config)
                if operation_succeeded:
                    # Instead of popping, we mark the operation as complete in our in-memory list
                    next_op['completed'] = True
                    iter_of_last_op = iter_num

                    # Save the updated schedule back to the file
                    if master_process:
                        save_scaling_schedule(scaling_schedule_file, scaling_schedule)
                    if next_op['reevaluate']:
                        if master_process: print("Re-evaluating validation loss after operation...")
                        losses = estimate_loss() # All processes re-evaluate to stay in sync
                        if master_process:
                            new_val_loss = losses['val']
                            print(f"New val loss after operation: {new_val_loss:.4f}")
                            training_logger.log_operation_reevaluation(iter_num, next_op['name'], current_val_loss, new_val_loss)
                            training_logger.log_step(iter_num, losses['train'], new_val_loss)
                            if wandb_log:
                                elapsed_time_seconds = time.time() - start_time
                                wandb.log({
                                    "iter": iter_num,
                                    "train/loss": losses['train'],
                                    "val/loss": new_val_loss,
                                    "lr": lr,
                                    "mfu": running_mfu*100,
                                    "time/elapsed_seconds": elapsed_time_seconds, # Log elapsed time
                                })
                        break # Exit the while loop after a blocking re-evaluation
                if master_process: print(f"=== SCALING OPERATION COMPLETE ===\n")
            else:
                break # No operation was triggered, exit the while loop

        if ddp:
            torch.distributed.barrier()

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.5f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if master_process:
    training_logger.close()
if ddp:
    destroy_process_group()