import os
import time
import random
import math
import pickle
import json
import yaml
import concurrent.futures
from contextlib import nullcontext
from datetime import datetime
import numpy as np
import torch
import psutil
import threading
import signal
import sys
from collections import deque, defaultdict
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
# Timing utilities
# -----------------------------------------------------------------------------
from collections import defaultdict
import time

class TimingProfiler:
    """
    A utility class for measuring training loop component timing.
    It supports a hierarchical EMA summary and provides a backward-compatible
    interface that also uses the smoothed EMA data.
    """

    def __init__(self, alpha=0.1):
        # --- Attributes for EMA calculation ---
        self.alpha = alpha
        self.ema_timings = {}
        self._context_stack = []
        self._current_raw_timings = {}

        # --- Attributes for full backward compatibility ---
        self.timings = defaultdict(list)
        self.current_timings = {} # Note: This will now be populated with EMA values
        self.total_iteration_time = 0.0

    def time_section(self, section_name):
        """Context manager for timing a specific section."""
        return TimingContext(self, section_name)

    def start_iteration(self):
        """Resets timers for the start of a new iteration."""
        self.iteration_start_time = time.perf_counter()
        self._current_raw_timings = {}
        self._context_stack = []

    def end_iteration(self):
        """Updates all timers at the end of an iteration."""
        self.total_iteration_time = time.perf_counter() - self.iteration_start_time
        self._update_ema(self.ema_timings, self._current_raw_timings)
        return self.total_iteration_time

    def _record_timing(self, section_name, duration):
        """Unified recording method called by TimingContext."""
        # 1. Record raw data for the current iteration in a nested way
        level = self._current_raw_timings
        for parent in self._context_stack[:-1]:
            if parent not in level: level[parent] = {'__duration__': 0.0}
            level = level[parent]
        if section_name not in level: level[section_name] = {'__duration__': 0.0}
        level[section_name]['__duration__'] += duration

        # 2. Record for the simple historical list (for get_summary_stats)
        self.timings[section_name].append(duration)

    def _update_ema(self, ema_level, raw_level):
        """Recursively updates the EMA timings at the end of an iteration."""
        all_keys = set(ema_level.keys()) | set(raw_level.keys())
        for key in all_keys:
            if key == '__duration__': continue
            raw_duration = raw_level.get(key, {}).get('__duration__', 0.0)
            if key not in ema_level:
                ema_level[key] = {'__ema_duration__': raw_duration, 'children': {}}
            else:
                old_ema = ema_level[key]['__ema_duration__']
                ema_level[key]['__ema_duration__'] = (self.alpha * raw_duration) + (1 - self.alpha) * old_ema
            self._update_ema(ema_level[key]['children'], raw_level.get(key, {}))

    # --- Public Interface Methods ---

    def get_current_percentages(self):
        """
        MODIFIED: Returns a flat dictionary of percentages based on the EMA-smoothed
        durations, maintaining the original method's interface.
        """
        flat_ema_durations = {}
        self._flatten_ema(self.ema_timings, flat_ema_durations)

        # The total time is the sum of all top-level sections' EMA durations
        total_ema_time = sum(v['__ema_duration__'] for v in self.ema_timings.values())
        if total_ema_time == 0:
            return {}

        percentages = {}
        for section, duration in flat_ema_durations.items():
            percentages[section] = (duration / total_ema_time) * 100

        return percentages

    def _flatten_ema(self, ema_level, flat_dict):
        """Internal helper to flatten the nested EMA dictionary."""
        for name, data in ema_level.items():
            flat_dict[name] = data['__ema_duration__']
            if data['children']:
                self._flatten_ema(data['children'], flat_dict)

    def get_summary(self):
        """Generates a clean, hierarchical string of the EMA timing breakdown."""
        total_time = sum(data['__ema_duration__'] for data in self.ema_timings.values())
        if total_time == 0: return "No timing data recorded yet."
        summary_lines = ["Timing Breakdown (EMA):"]
        self._format_level(summary_lines, self.ema_timings, total_time, indent=0)
        return "\n".join(summary_lines)

    def _format_level(self, summary_lines, level_data, parent_total, indent):
        """Recursively formats the summary string."""
        indent_str = "    " * indent
        for name, data in sorted(level_data.items(), key=lambda i: i[1]['__ema_duration__'], reverse=True):
            duration = data['__ema_duration__']
            percentage = (duration / parent_total) * 100 if parent_total > 0 else 0
            summary_lines.append(f"{indent_str} L {name:<22} | {percentage:5.1f}% | ({duration * 1000:.2f} ms)")
            if data['children']:
                self._format_level(summary_lines, data['children'], duration, indent + 1)

    # --- Other Backward-Compatibility Methods ---

    def get_average_percentages(self, last_n=10):
        if not self.timings: return {}
        avg_timings = {s: sum(t[-last_n:]) / len(t[-last_n:]) for s, t in self.timings.items()}
        total_avg = sum(avg_timings.values())
        if total_avg == 0: return {}
        return {s: (t / total_avg) * 100 for s, t in avg_timings.items()}

    def get_summary_stats(self, section_name, last_n=10):
        if section_name not in self.timings: return {}
        recent_times = self.timings[section_name][-last_n:]
        if not recent_times: return {}
        return {
            'avg_ms': (sum(recent_times) / len(recent_times)) * 1000,
            'min_ms': min(recent_times) * 1000,
            'max_ms': max(recent_times) * 1000,
            'count': len(recent_times)
        }

class TimingContext:
    """Context manager for timing a specific section."""
    def __init__(self, profiler, section_name):
        self.profiler = profiler
        self.section_name = section_name
        self.start_time = None

    def __enter__(self):
        self.profiler._context_stack.append(self.section_name)
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        self.profiler._record_timing(self.section_name, duration)
        self.profiler._context_stack.pop()
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
vocab_size = None # taken from meta_vocab_size
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
# embedding analysis configuration
ignored_outlayers_sum = 0.01 # Fraction of tokens to ignore as outliers in embedding analysis
# -----------------------------------------------------------------------------
# Shrunken vocabulary training parameters
shrunken_vocab_size = None # If set, enables training with a smaller vocab to save memory
vocab_remapping_file = None # Path to .pt file with the remapping tensor, required if shrunken_vocab_size is set
RARE_TOKEN_ID = None # The token ID in the shrunken vocab for all out-of-vocab tokens
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging




if dataset == 'fineweb10B':
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
        print("Scaling schedule not found")
        return []
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                schedule = yaml.safe_load(f)
                print("Scaling schedule loaded")
            elif file_path.endswith('.json'):
                schedule = json.load(f)
                print("Scaling schedule loaded")
            else:
                print(f"Warning: Unknown file format for scaling schedule: {file_path}")
                raise Exception(f"Warning: Unknown file format for scaling schedule: {file_path}")
                return []
        if not isinstance(schedule, list):
            print(f"Warning: Scaling schedule must be a list, got {type(schedule)}")
            raise Exception(f"Warning: Scaling schedule must be a list, got {type(schedule)}")
            return []
        for i, op in enumerate(schedule):
            required_keys = ['name', 'value', 'trigger_loss', 'max_wait_iters', 'reevaluate']
            if not all(key in op for key in required_keys):
                print(f"Warning: Operation {i} missing required keys: {required_keys}")
                raise Exception(f"Warning: Operation {i} missing required keys: {required_keys}")

        # Handle completion status based on init_from parameter
        # Note: Missing 'completed' fields are treated as False automatically
        if init_from == 'scratch':
            print("Starting from scratch, resetting schedule completion status.")
            for op in schedule:
                op['completed'] = False
            save_scaling_schedule(file_path, schedule)
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
        raise e

# Check for auto-resume configuration
auto_resume_marker = os.path.join(out_dir, '.auto_resume') 
if init_from == 'scratch' and os.path.exists(auto_resume_marker):
    print("Found auto-resume marker. Switching to 'resume' mode...")
    init_from = 'resume'
    # Read the marker file for additional settings
    try:
        with open(auto_resume_marker, 'r') as f:
            for line in f:
                if line.startswith('init_from='):
                    pass  # Already handled
                elif line.startswith('final_wandb_run_name='):
                    # This will be handled in wandb initialization
                    pass
    except Exception as e:
        print(f"Warning: Could not read auto-resume marker: {e}")

# Load scaling schedule (this will be overridden by checkpoint state if resuming)
if scaling_schedule_file:
    scaling_schedule = load_scaling_schedule(scaling_schedule_file, init_from)
else:
    scaling_schedule = []

# Training Orchestrator State
iter_of_last_op = 0 # Iteration number when last operation was executed
lr_schedule_offset = 0 # Offset for learning rate schedule (for reset_lr_schedule)
# -----------------------------------------------------------------------------

# logging setup
training_logger = TrainingLogger(log_dir=log_dir, file_enabled=file_logging)

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

# Vocabulary remapping setup
remapping_vector = None
remapping_active = False # Global flag to control remapping
if master_process and shrunken_vocab_size is not None:
    if not vocab_remapping_file or not os.path.exists(vocab_remapping_file):
        raise ValueError("`shrunken_vocab_size` is set, but `vocab_remapping_file` is missing or invalid.")
    if RARE_TOKEN_ID is None:
        raise ValueError("`shrunken_vocab_size` is set, but `RARE_TOKEN_ID` is not.")

    print(f"Loading vocabulary remapping from {vocab_remapping_file}")
    remapping_vector = torch.load(vocab_remapping_file)
    remapping_active = True # Initially active if configured

if ddp:
    # Broadcast the remapping_active flag and the tensor itself to all processes
    active_flag_tensor = torch.tensor([1.0 if remapping_active else 0.0], device=device)
    torch.distributed.broadcast(active_flag_tensor, src=0)
    remapping_active = active_flag_tensor.item() == 1.0

    if remapping_active:
        if ddp_rank != 0: # If not master, create a placeholder tensor
            # Use the meta_vocab_size or default if not available
            full_vocab_size = meta_vocab_size if meta_vocab_size is not None else 57664
            remapping_vector = torch.zeros(full_vocab_size, dtype=torch.long)
        remapping_vector = remapping_vector.to(device)
        torch.distributed.broadcast(remapping_vector, src=0)
elif remapping_vector is not None:
    remapping_vector = remapping_vector.to(device)

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    training_logger.setup(config)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# High-performance data loader with curriculum learning
data_dir = os.path.join('data', dataset)

class BatchManager:
    def __init__(self, data_dir, shard_filenames, vocab_size, batch_size, block_size, device, device_type,
                 starting_estimation_token_count=100_000_000, buffer_size=2000):
        print("Initializing High-Performance BatchManager (V2)...")
        self.data_dir = data_dir
        self.shard_filenames = shard_filenames
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        self.buffer_size = buffer_size

        # 1. Approximate or load the corpus token distribution
        self.corpus_distribution = self._get_corpus_distribution(starting_estimation_token_count)
        self.uniform_distribution = torch.ones(self.vocab_size, dtype=torch.float32) / self.vocab_size

        # 2. Initialize state for tracking served tokens
        self.served_token_counts = torch.zeros(self.vocab_size, dtype=torch.float64)
        self.total_tokens_served = 0

        # 3. Thread-safe candidate buffer and control variables for the worker
        self.candidate_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.rescore_event = threading.Event()
        self.shutdown_event = threading.Event()

        # 4. Initialize the curriculum and target distribution
        self.alpha = 1.0
        self.target_distribution = self.uniform_distribution.clone()

        # 5. Start the background worker thread
        self.worker_thread = threading.Thread(target=self._buffer_management_worker, daemon=True)
        self.worker_thread.start()
        print("BatchManager initialized and background worker started.")

    def _get_corpus_distribution(self, estimation_tokens):
        """Calculates an approximate token distribution from a sample of the dataset and caches it."""
        cache_path = os.path.join(self.data_dir, f'corpus_dist_approx_{estimation_tokens}.pt')
        if os.path.exists(cache_path):
            print(f"Loading cached approximate corpus distribution from {cache_path}")
            return torch.load(cache_path)

        print(f"Approximating corpus distribution from {estimation_tokens:,} tokens...")
        total_counts = torch.zeros(self.vocab_size, dtype=torch.int64)
        tokens_per_shard = estimation_tokens // len(self.shard_filenames)
        unknown_token_id = self.vocab_size - 1

        for shard_name in self.shard_filenames:
            shard_path = os.path.join(self.data_dir, shard_name)
            data = np.memmap(shard_path, dtype=np.uint16, mode='r')
            if len(data) > tokens_per_shard:
                sample = data[:tokens_per_shard]
                sample = np.clip(sample, 0, unknown_token_id)
                shard_counts = torch.from_numpy(np.bincount(sample, minlength=self.vocab_size))
                total_counts += shard_counts

        distribution = total_counts.float() / total_counts.sum()
        print(f"Saving approximate corpus distribution to {cache_path}")
        torch.save(distribution, cache_path)
        return distribution

    def _buffer_management_worker(self):
        """
        Runs in a background thread to continuously read, score, and manage the candidate buffer.
        """
        shard_cycle = iter(self.shard_filenames * 1000) # Loop over the dataset many times

        while not self.shutdown_event.is_set():
            # --- Phase 1: Refill the buffer if it has space ---
            if len(self.candidate_buffer) < self.buffer_size:
                try:
                    shard_name = next(shard_cycle)
                    shard_path = os.path.join(self.data_dir, shard_name)
                    data = np.memmap(shard_path, dtype=np.uint16, mode='r')

                    # Create multiple batches from a large sequential chunk for I/O efficiency
                    num_batches_to_create = 50
                    chunk_size = num_batches_to_create * self.batch_size * self.block_size
                    start_idx = random.randint(0, max(0, len(data) - chunk_size))
                    chunk = data[start_idx : start_idx + chunk_size]

                    new_candidates = []
                    # Create non-overlapping batches from the chunk
                    for i in range(0, len(chunk), self.batch_size * self.block_size):
                        if i + self.batch_size * self.block_size + 1 > len(chunk): continue
                        x = torch.from_numpy(chunk[i : i + self.batch_size * self.block_size].astype(np.int64)).view(self.batch_size, self.block_size)
                        y = torch.from_numpy(chunk[i+1 : i+1 + self.batch_size * self.block_size].astype(np.int64)).view(self.batch_size, self.block_size)
                        new_candidates.append({'x': x, 'y': y, 'score': -1.0})

                    with self.buffer_lock:
                        self.candidate_buffer.extend(new_candidates)

                except StopIteration:
                    print("Worker has finished all shard cycles.")
                    break
                except Exception as e:
                    print(f"Error in buffer refill worker: {e}")
                    time.sleep(1)

            # --- Phase 2: Re-score and sort the entire buffer if signaled ---
            if self.rescore_event.is_set():
                with self.buffer_lock:
                    print("(Async Worker) Re-scoring candidate buffer...")
                    served_dist = (self.served_token_counts / (self.total_tokens_served + 1e-9)).to(torch.float32)

                    temp_list = list(self.candidate_buffer)
                    for batch_data in temp_list:
                        tokens, counts = torch.unique(batch_data['x'], return_counts=True)
                        neglect_score = self.target_distribution[tokens] / (served_dist[tokens] + 1e-9)
                        batch_data['score'] = (neglect_score * counts).sum().item()

                    # Sort the buffer by score (highest first) and trim excess
                    temp_list.sort(key=lambda b: b['score'], reverse=True)
                    self.candidate_buffer = deque(temp_list[:self.buffer_size])

                    self.rescore_event.clear() # Mark re-scoring as done
                    print(f"(Async Worker) Buffer re-scored. Size: {len(self.candidate_buffer)}")

            time.sleep(0.1) # Prevent busy-looping, yield to other threads

    def update_target_distribution(self, alpha):
        """Updates the target distribution and signals the worker to re-score."""
        print(f"Updating batch manager alpha to {alpha:.3f}")
        self.alpha = alpha
        # Blend corpus and uniform distributions
        self.target_distribution = (1 - alpha) * self.corpus_distribution + alpha * self.uniform_distribution
        self.rescore_event.set() # Signal the worker thread to re-score all candidates

    def get_next_batch(self):
        """Waits for and retrieves the highest-scoring batch from the buffer."""
        # Wait for the buffer to be populated, especially at the start
        while not self.candidate_buffer:
            print("Main thread is waiting for the batch buffer to fill...")
            time.sleep(0.5)
            if self.shutdown_event.is_set(): return None, None

        with self.buffer_lock:
            # The buffer is kept sorted by the worker, so the best is always at the front
            best_batch_data = self.candidate_buffer.popleft()

        best_x, best_y = best_batch_data['x'], best_batch_data['y']

        # Update the state of served tokens
        unique_tokens, counts = torch.unique(best_x, return_counts=True)
        self.served_token_counts[unique_tokens] += counts.to(self.served_token_counts.dtype)
        self.total_tokens_served += best_x.numel()

        # Move the chosen batch to the correct GPU/CPU device
        if self.device_type == 'cuda':
            best_x = best_x.pin_memory().to(self.device, non_blocking=True)
            best_y = best_y.pin_memory().to(self.device, non_blocking=True)
        else:
            best_x, best_y = best_x.to(self.device), best_y.to(self.device)

        return best_x, best_y

    def get_non_outlier_tokens(self, ignored_outlayers_sum=0.01):
        """
        Extract token IDs that sum up to (1 - ignored_outlayers_sum) of total_tokens_served.
        Returns a list of token IDs, excluding the most and least frequent outliers.
        """
        if self.total_tokens_served == 0:
            return list(range(self.vocab_size))  # Return all tokens if no tokens served yet

        # Calculate the served distribution
        served_distribution = self.served_token_counts / self.total_tokens_served

        # Sort tokens by their served frequency
        sorted_indices = torch.argsort(served_distribution, descending=True)
        sorted_counts = served_distribution[sorted_indices]

        # Calculate cumulative sum
        cumulative_sum = torch.cumsum(sorted_counts, dim=0)

        # Find tokens that sum up to (1 - ignored_outlayers_sum) of total
        target_sum = 1.0 - ignored_outlayers_sum

        # Find the cutoff index where cumulative sum reaches target_sum
        cutoff_idx = torch.searchsorted(cumulative_sum, target_sum, right=True)

        # Ensure we don't exceed the vocabulary size
        cutoff_idx = min(cutoff_idx.item(), self.vocab_size - 1)

        # Get the non-outlier token IDs
        non_outlier_tokens = sorted_indices[:cutoff_idx + 1].tolist()

        print(f"Selected {len(non_outlier_tokens)} non-outlier tokens out of {self.vocab_size} total tokens")
        print(f"These tokens represent {cumulative_sum[cutoff_idx]:.4f} of total served tokens")

        return non_outlier_tokens

    def shutdown(self):
        """Signals the background worker to stop and waits for it to exit."""
        print("Shutting down BatchManager background worker...")
        self.shutdown_event.set()
        self.worker_thread.join(timeout=5)
        print("BatchManager shut down.")

# Simple validation batch function (unchanged from original)
def get_val_batch():
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
else:
    if vocab_size:
        meta_vocab_size = vocab_size
    else:
        raise ValueError("meta_vocab_size / vocab_size not set")

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
    # Determine the active vocabulary size for model instantiation
    if shrunken_vocab_size is not None:
        print(f"Using shrunken vocabulary of size: {shrunken_vocab_size}")
        active_vocab_size = shrunken_vocab_size
    else:
        active_vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304

    model_args['vocab_size'] = active_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    # Check for emergency checkpoints if main checkpoint is corrupted or missing
    if not os.path.exists(ckpt_path):
        # Look for emergency checkpoints
        emergency_files = [f for f in os.listdir(out_dir) if f.startswith('emergency_ckpt_iter_') and f.endswith('.pt')]
        if emergency_files:
            # Use the most recent emergency checkpoint
            emergency_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
            ckpt_path = os.path.join(out_dir, emergency_files[0])
            print(f"Main checkpoint not found. Using emergency checkpoint: {ckpt_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {out_dir}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if checkpoint.get('emergency_save', False):
            print("Loaded from emergency checkpoint - continuing training from emergency save point")
    except Exception as e:
        print(f"Error loading checkpoint {ckpt_path}: {e}")
        # Try emergency checkpoints as fallback
        emergency_files = [f for f in os.listdir(out_dir) if f.startswith('emergency_ckpt_iter_') and f.endswith('.pt')]
        if emergency_files:
            emergency_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]), reverse=True)
            ckpt_path = os.path.join(out_dir, emergency_files[0])
            print(f"Trying emergency checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            print("Successfully loaded from emergency checkpoint")
        else:
            raise e
    checkpoint_model_args = checkpoint['model_args']
    
    # Compare current config with checkpoint and handle overrides
    overrideable_params = ['n_hidden', 'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
                          'dropout', 'use_rotary_embeddings', 'rotary_base', 'rotary_max_position_embeddings',
                          'embedding_mode', 'embedding_rank', 'attn_lora_rank', 'lora_alpha']
    
    config_changes = []
    for param in overrideable_params:
        if param in checkpoint_model_args:
            checkpoint_value = checkpoint_model_args[param]
            current_value = model_args.get(param)
            
            if current_value != checkpoint_value:
                config_changes.append(f"{param}: {checkpoint_value} -> {current_value}")
                print(f"Overriding {param}: {checkpoint_value} -> {current_value}")
                model_args[param] = current_value  # Use current config value
            else:
                # Keep checkpoint value if no override specified
                model_args[param] = checkpoint_value
    
    if config_changes:
        print(f"Applied {len(config_changes)} hyperparameter overrides:")
        for change in config_changes:
            print(f"  - {change}")
        
        # Log the overrides for tracking
        if master_process:
            training_logger.log(f"Resume with hyperparameter overrides: {'; '.join(config_changes)}")
    else:
        print("No hyperparameter overrides detected.")
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
    
    # Load execution state variables
    if 'iter_of_last_op' in checkpoint:
        iter_of_last_op = checkpoint['iter_of_last_op']
        print(f"Restored iter_of_last_op: {iter_of_last_op}")
    if 'lr_schedule_offset' in checkpoint:
        lr_schedule_offset = checkpoint['lr_schedule_offset']
        print(f"Restored lr_schedule_offset: {lr_schedule_offset}")
    
    # Restore scaling schedule state from checkpoint if available
    if 'scaling_schedule' in checkpoint and checkpoint['scaling_schedule']:
        saved_schedule = checkpoint['scaling_schedule']
        saved_schedule_file = checkpoint.get('scaling_schedule_file')
        
        # If we have a current scaling schedule file and it matches the saved one,
        # use the saved state to preserve completion status
        if scaling_schedule_file and saved_schedule_file == scaling_schedule_file:
            print(f"Restoring scaling schedule state from checkpoint (preserving completion status)")
            scaling_schedule = saved_schedule
            # Also update the file to match the checkpoint state
            save_scaling_schedule(scaling_schedule_file, scaling_schedule)
        elif saved_schedule:
            print(f"Warning: Checkpoint has scaling schedule but current config doesn't match.")
            print(f"  Checkpoint file: {saved_schedule_file}")
            print(f"  Current file: {scaling_schedule_file}")
            print(f"  Using file-based schedule (some completion status may be lost).")
    
    # Override certain training parameters from current config if different
    training_overrides = ['learning_rate', 'max_iters', 'weight_decay', 'beta1', 'beta2', 'grad_clip',
                         'decay_lr', 'warmup_iters', 'lr_decay_iters', 'min_lr', 'batch_size', 
                         'gradient_accumulation_steps', 'eval_interval', 'eval_iters']
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        training_param_changes = []
        
        for param in training_overrides:
            if param in saved_config:
                saved_value = saved_config[param]
                current_value = globals().get(param)
                
                if current_value != saved_value:
                    training_param_changes.append(f"{param}: {saved_value} -> {current_value}")
                    print(f"Training override {param}: {saved_value} -> {current_value}")
                    # Keep current value (already in globals())
        
        if training_param_changes:
            print(f"Applied {len(training_param_changes)} training parameter overrides:")
            for change in training_param_changes:
                print(f"  - {change}")
            
            if master_process:
                training_logger.log(f"Resume with training parameter overrides: {'; '.join(training_param_changes)}")
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
    torch._dynamo.reset()
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        # Initialize core accuracy tracking
        core_token_correct = 0
        core_token_total = 0

        for k in range(eval_iters):
            if split == 'train':
                X, Y = batch_manager.get_next_batch()
            else:
                X, Y = get_val_batch()

            # If remapping is active, we need to handle metrics carefully
            if remapping_active:
                remapped_X = remapping_vector[X]
                remapped_Y = remapping_vector[Y]
                with ctx:
                    logits, loss = model(remapped_X, remapped_Y)

                # Calculate Core Accuracy on the validation set
                if split == 'val':
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)

                    # Mask to ignore the RARE_TOKEN_ID in accuracy calculation
                    is_core_token = (remapped_Y != RARE_TOKEN_ID)

                    core_token_correct += ((preds == remapped_Y) & is_core_token).sum().item()
                    core_token_total += is_core_token.sum().item()
            else:
                # Standard loss calculation
                with ctx:
                    logits, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()
        # Add core accuracy if remapping is active and we have core tokens
        if remapping_active and core_token_total > 0 and split == 'val':
            out['val_core_acc'] = core_token_correct / core_token_total

    model.train()
    return out

def get_lr(it):
    effective_it = it - lr_schedule_offset
    warmup_iters
    lr_decay_iters

    if master_process and wandb_log:
        wandb.log({"iter": it, "effective_it": effective_it, "warmup_iters": warmup_iters, "lr_decay_iters": lr_decay_iters, "gradient_accumulation_steps":gradient_accumulation_steps, "batch_size":batch_size })
    else:
        if it % log_interval == 0:
            print(f"iter: {it}, effective_it: {effective_it}, warmup_iters: {warmup_iters}, lr_decay_iters: {lr_decay_iters}, gradient_accumulation_steps:{gradient_accumulation_steps}, batch_size:{batch_size}")
    
    if effective_it < warmup_iters:
        return learning_rate * (effective_it + 1) / (warmup_iters + 1)
    if effective_it > lr_decay_iters:
        return min_lr
    decay_ratio = (effective_it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (min_lr + coeff * (learning_rate - min_lr))

def run_full_analysis_async(analyzer, current_snapshot, prev_snapshot, val_batch, iter_num, filtered_token_ids=None):
    """
    The new async task function. It calls the main analysis method of the analyzer.
    """
    print(f"(Async Analysis @ iter {iter_num}) Starting full model analysis job...")
    results = analyzer.run_full_analysis(current_snapshot, prev_snapshot, filtered_token_ids=filtered_token_ids)
    enthropy = analyzer.analyze_attention_entropy(val_batch)
    results['attention_entropy'] = enthropy
    results['iter_num'] = iter_num # Tag results with the iteration number
    print(f"(Async Analysis @ iter {iter_num}) Job finished.")
    return results

def analysis_done_callback(future):
    """
    The new callback, rewritten to handle the rich, nested results dictionary.
    """
    global training_logger, master_process  # Access global variables for logging

    try:
        results = future.result()
        iter_num = results['iter_num']
        attention_entropy = results['attention_entropy']

        print(f"\n--- ASYNC ANALYSIS RESULTS FOR ITERATION {iter_num} ---")

        # Prepare log messages for file logging
        log_messages = []
        log_messages.append(f"--- ASYNC ANALYSIS RESULTS FOR ITERATION {iter_num} ---")

        # --- Report Geometry & Rank Results ---
        if 'geometry' in results:
            geo = results['geometry']

            # Embedding Geometry Analysis
            if 'embeddings' in geo and geo['embeddings']:
                emb_geo = geo['embeddings']
                mean_sim = emb_geo['global_sparsity']['mean_similarity']
                std_sim = emb_geo['global_sparsity']['std_similarity']
                sim_10th = emb_geo['global_sparsity']['similarity_10th_percentile']
                sim_90th = emb_geo['global_sparsity']['similarity_90th_percentile']
                avg_neighbors = emb_geo['local_density']['average_neighborhood_size']
                nbhd_10th = emb_geo['local_density']['neighbor_10th_percentile']
                nbhd_90th = emb_geo['local_density']['neighbor_90th_percentile']
                nbhd_99th = emb_geo['local_density']['neighbor_99th_percentile']

                # Analysis info
                analysis_info = emb_geo.get('analysis_info', {})
                num_analyzed = analysis_info.get('num_embeddings_analyzed', 'N/A')
                total_embeddings = analysis_info.get('total_embeddings', 'N/A')
                is_filtered = analysis_info.get('filtered', False)

                geom_msg1 = f"  [Embeddings Geometry] Avg Neighbors: {avg_neighbors:.2f} 10th-90th-99th Percentile: {nbhd_10th:.4f} - {nbhd_90th:.4f} - {nbhd_99th:.4f} "
                geom_msg2 = f"  [Embeddings Geometry] Mean Similarity: {mean_sim:.4f} Std Similarity: {std_sim:.4f} | 10th-90th Percentile: {sim_10th:.4f} - {sim_90th:.4f}"
                geom_msg3 = f"  [Embeddings Analysis] Analyzed: {num_analyzed}/{total_embeddings} embeddings | Filtered: {is_filtered}"
                att_msg = f"  [Attention Entropy] Avg Entropy: {attention_entropy:.4f}"
                print(geom_msg1)
                print(geom_msg2)
                print(geom_msg3)
                log_messages.append(geom_msg1)
                log_messages.append(geom_msg2)
                log_messages.append(geom_msg3)

                if wandb_log:
                    wandb_data = {
                        "analysis/embed/geom_avg_neighbors": avg_neighbors,
                        "analysis/embed/geom_mean_sim": mean_sim,
                        "analysis/embed/geom_std_sim": std_sim,
                        "analysis/embed/geom_sim_10th": sim_10th,
                        "analysis/embed/geom_sim_90th": sim_90th
                    }
                    if isinstance(num_analyzed, (int, float)):
                        wandb_data["analysis/embed/num_embeddings_analyzed"] = num_analyzed
                    if isinstance(total_embeddings, (int, float)):
                        wandb_data["analysis/embed/total_embeddings"] = total_embeddings
                    wandb_data["analysis/embed/filtered"] = is_filtered
                    wandb.log(wandb_data, step=iter_num)

            # FFN Rank Analysis
            if 'ffn_ranks' in geo:
                num_layers = len(geo['ffn_ranks'])
                layers_to_log = {0, num_layers // 2, num_layers - 1}
                for i in layers_to_log:
                    rank_info = geo['ffn_ranks'].get(f'layer_{i}')
                    if rank_info:
                        util = rank_info['utilization']
                        eff_rank = rank_info['effective_rank']
                        full_rank = rank_info['full_rank']
                        ffn_msg = f"  [FFN Rank L{i}] Utilization: {util:.2%} ({eff_rank}/{full_rank})"
                        print(ffn_msg)
                        log_messages.append(ffn_msg)
                        if wandb_log: wandb.log({f"analysis/ffn/rank_util_L{i}": util}, step=iter_num)

            # Attention Rank Analysis
            if 'attn_ranks' in geo:
                num_layers = len(geo['attn_ranks'])
                layers_to_log = {0, num_layers // 2, num_layers - 1}
                for i in layers_to_log:
                    attn_info = geo['attn_ranks'].get(f'layer_{i}')
                    if attn_info:
                        for component in ['Q', 'K', 'V']:
                            comp_info = attn_info.get(component)
                            if comp_info:
                                util = comp_info['utilization']
                                eff_rank = comp_info['effective_rank']
                                full_rank = comp_info['full_rank']
                                attn_msg = f"  [Attn {component} L{i}] Utilization: {util:.2%} ({eff_rank}/{full_rank})"
                                print(attn_msg)
                                log_messages.append(attn_msg)
                                if wandb_log: wandb.log({f"analysis/attn/{component.lower()}_rank_util_L{i}": util}, step=iter_num)

        # --- Report Drift Results ---
        if 'drift' in results:
            drift = results['drift']

            # Embedding Drift
            if 'embeddings' in drift and drift['embeddings']:
                emb_drift_avg = drift['embeddings']['avg_cosine_similarity']
                emb_drift_10th = drift['embeddings']['cosine_sim_10th_percentile']
                emb_drift_90th = drift['embeddings']['cosine_sim_90th_percentile']
                drift_msg1 = f"  [Embeddings Drift] Avg Cosine Sim: {emb_drift_avg:.4f} | 10th Percentile: {emb_drift_10th:.4f} | 90th Percentile: {emb_drift_90th:.4f}"
                print(drift_msg1)
                log_messages.append(drift_msg1)
                if wandb_log: wandb.log({
                    "analysis/embed/drift_avg_sim": emb_drift_avg,
                    "analysis/embed/drift_10th_sim": emb_drift_10th,
                    "analysis/embed/drift_90th_sim": emb_drift_90th
                }, step=iter_num)

            # FFN Drift (report first layer for brevity)
            ffn0_drift = drift.get('ffn.0.c_fc.weight')
            if ffn0_drift:
                ffn_drift_avg = ffn0_drift['avg_cosine_similarity']
                ffn_drift_10th = ffn0_drift['cosine_sim_10th_percentile']
                ffn_drift_90th = ffn0_drift['cosine_sim_90th_percentile']
                drift_msg2 = f"  [FFN L0 Drift] Avg Cosine Sim: {ffn_drift_avg:.4f} | 10th Percentile: {ffn_drift_10th:.4f}"
                print(drift_msg2)
                log_messages.append(drift_msg2)
                if wandb_log: wandb.log({
                    "analysis/ffn/drift_avg_sim_L0": ffn_drift_avg,
                    "analysis/ffn/drift_10th_sim_L0": ffn_drift_10th,
                    "analysis/ffn/drift_90th_sim_L0": ffn_drift_90th
                }, step=iter_num)

        end_msg = "--- END OF ASYNC ANALYSIS RESULTS ---"
        print(end_msg + "\n")
        log_messages.append(end_msg)

        # Log to file if logging is enabled and we're the master process
        if master_process and training_logger.is_enabled:
            for msg in log_messages:
                training_logger.log(msg)

    except Exception as e:
        print(f"\n--- ERROR in analysis_done_callback: {e} ---\n")
        import traceback
        traceback.print_exc()
        # Also log errors to file if logging is enabled
        if master_process and training_logger.is_enabled:
            training_logger.log(f"ERROR DURING ASYNC ANALYSIS: {e}")


def print_timings(timing_profiler, training_logger):
    
    # Get timing breakdown
    timing_percentages = timing_profiler.get_current_percentages()
    evaluation = timing_percentages.get('evaluation', 0)
    gradient_accumulation =  timing_percentages.get('gradient_accumulation', 0)
    gradient_clipping =  timing_percentages.get('gradient_clipping', 0)
    forward_pass_pct = timing_percentages.get('forward_pass', 0)
    backward_pass_pct = timing_percentages.get('backward_pass', 0)
    data_loading_pct = timing_percentages.get('data_loading', 0)
    optimizer_step_pct = timing_percentages.get('optimizer_step', 0)

        # Enhanced logging with timing breakdown
    print(f"  timing breakdown: evaluation {evaluation:.1f}%")
    training_logger.log(f"  timing breakdown: evaluation {evaluation:.1f}%")
    print(f"  timing breakdown: gradient_accumulation {gradient_accumulation:.1f}%")
    training_logger.log(f"  timing breakdown: gradient_accumulation {gradient_accumulation:.1f}%")
    print(f"  timing breakdown:     forward {forward_pass_pct:.1f}%, backward {backward_pass_pct:.1f}%, data {data_loading_pct:.1f}%, optim {optimizer_step_pct:.1f}%")
    training_logger.log(f"  timing breakdown:     forward {forward_pass_pct:.1f}%, backward {backward_pass_pct:.1f}%, data {data_loading_pct:.1f}%, optim {optimizer_step_pct:.1f}%")
    print(f"  timing breakdown: gradient_clipping {gradient_clipping:.1f}%")
    training_logger.log(f"  timing breakdown: gradient_clipping {gradient_clipping:.1f}%")


def execute_operation(op, trigger_reason, current_val_loss, iter_num, target_architecture_config):
    # Make globals mutable within this function
    global learning_rate, batch_size, gradient_accumulation_steps, warmup_iters, eval_iters, eval_interval
    global lr_schedule_offset, training_logger, master_process, model, optimizer, raw_model, unoptimized_model
    global remapping_active # Add this global

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
                             'set_embedding_lora_rank', 'merge_lora_weights',
                             'resize_vocabulary', 'set_embedding_finetune_mode', 'set_embedding_freeze_mode',
                             'freeze_layer', 'unfreeze_layer', 'set_layer_lora_rank']

        if op_name in architectural_ops:
            if master_process:
                print(f"Performing architectural operation: {op_name}")

            unwrapped_model = unoptimized_model if compile else (model.module if ddp else model)
            old_optimizer_state = optimizer.state_dict()
            old_param_dict = {name: p for name, p in unwrapped_model.named_parameters()}

            # --- Perform the absolute architectural operation ---
            if op_name == 'stack_layers':
                start_time = time.time()
                unwrapped_model.stack_layers(op_value)
            elif op_name == 'widen_mlp':
                start_time = time.time()
                unwrapped_model.widen_mlp(op_value)
            elif op_name == 'set_attn_lora_rank':
                start_time = time.time()
                unwrapped_model.resize_lora_rank(op_value)
            elif op_name == 'set_embedding_lora_rank':
                start_time = time.time()
                unwrapped_model.resize_embedding_rank(op_value)
            elif op_name == 'merge_lora_weights':
                unwrapped_model.merge_lora_weights()
            elif op_name == 'resize_vocabulary':
                # op_value is expected to be [source_token_id, noise_std]
                source_token_id, noise_std = op_value
                # The target vocab size is derived from the meta file
                meta_path = os.path.join(data_dir, 'meta.pkl')
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                full_vocab_size = meta['vocab_size']
                unwrapped_model.resize_vocabulary(full_vocab_size, source_token_id, noise_std)
            elif op_name == 'set_embedding_finetune_mode':
                # op_value is expected to be True or False
                unwrapped_model.set_embedding_finetune_mode(op_value)
            elif op_name == 'set_embedding_freeze_mode':
                # op_value is expected to be True or False
                unwrapped_model.set_embedding_freeze_mode(op_value)
            elif op_name == 'freeze_layer':
                # op_value is expected to be a layer name string like "attn.2" or "wte"
                unwrapped_model.freeze_layer(op_value)
            elif op_name == 'unfreeze_layer':
                # op_value is expected to be a layer name string like "attn.2" or "wte"
                unwrapped_model.unfreeze_layer(op_value)
            elif op_name == 'set_layer_lora_rank':
                # op_value is expected to be [layer_name, rank] like ["attn.2", 16]
                layer_name, rank = op_value
                unwrapped_model.set_layer_lora_rank(layer_name, rank)

            # Clear CUDA cache after architectural changes to free up memory
            if device_type == 'cuda':
                if master_process: print("Clearing CUDA cache...")
                torch.cuda.empty_cache()

            # --- Re-create optimizer and wrappers (this logic remains the same) ---
            log_detailed_params(unwrapped_model)
            if master_process: print("Re-configuring optimizer...")
            optimizer = unwrapped_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
            if master_process: print("Transferring optimizer state...")
            transfer_optimizer_state(optimizer, old_optimizer_state, old_param_dict, unwrapped_model)

            model = unwrapped_model
            if compile:
                if master_process: print("Re-compiling the model...")
                torch._dynamo.reset()
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
                start_time = time.time()
                if master_process: print(f"Batch size: {batch_size} -> {op_value}")
                batch_size = op_value
            elif op_name == 'set_grad_accum':
                start_time = time.time()
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
            elif op_name == 'set_curriculum_alpha':
                if master_process: print(f"Setting curriculum alpha: {op_value}")
                # This calls the method in our new BatchManager to update the target and trigger a re-score
                batch_manager.update_target_distribution(op_value)
            elif op_name == 'disable_vocab_remapping':
                if master_process: print("Disabling vocabulary remapping.")
                remapping_active = False
            elif op_name == 'adjust_batch_size':
                # Get unwrapped model for VRAM calculation
                if compile:
                    unwrapped_model = unoptimized_model if hasattr(unoptimized_model, '_orig_mod') else unoptimized_model
                    if ddp:
                        unwrapped_model = unwrapped_model.module if hasattr(unwrapped_model, 'module') else unwrapped_model
                else:
                    unwrapped_model = model.module if ddp and hasattr(model, 'module') else model
                
                # Extract parameters from op_value
                if isinstance(op_value, dict):
                    batch_size_to_use = op_value.get('current_batch_size', batch_size)
                    max_batch_size = op_value.get('max_batch_size', 1024)
                    target_vram_percent = op_value.get('target_vram_percent', 82.0)
                else:
                    # Legacy support - use current batch_size or op_value
                    batch_size_to_use = batch_size or op_value or 32
                    max_batch_size = 1024
                    target_vram_percent = 82.0
                
                try:
                    optimal_batch_size = calculate_optimal_batch_size(
                        unwrapped_model, batch_size_to_use, max_batch_size, 
                        target_vram_percent, device_type, master_process
                    )
                    
                    if master_process:
                        print(f"Calculated optimal batch size: {optimal_batch_size}")
                        print(f"Batch size: {batch_size} -> {optimal_batch_size}")
                    
                    batch_size = optimal_batch_size
                    
                    if device_type == 'cuda':
                        if master_process: print("Clearing CUDA cache after batch size adjustment...")
                        torch.cuda.empty_cache()

                    if master_process: 
                        training_logger.log_operation_success(iter_num, op_name, 
                                                            {'calculated_batch_size': optimal_batch_size,
                                                             'original_batch_size': batch_size_to_use})
                except NameError:
                    # Fallback if calculate_optimal_batch_size is not available
                    if master_process:
                        print("Warning: calculate_optimal_batch_size not available, keeping current batch size")
                        training_logger.log_operation_success(iter_num, op_name, 
                                                            {'message': 'Operation skipped - function not available'})
            
            elif op_name == 'set_batch_size_relative':
                batch_size_to_use = batch_size
                scale_factor = op_value
                
                if not isinstance(scale_factor, (int, float)):
                    raise ValueError(f"set_batch_size_relative requires a numeric scale factor, got {type(scale_factor)}")
                
                try:
                    new_batch_size = calculate_relative_batch_size(
                        batch_size_to_use, scale_factor, master_process
                    )
                    
                    if master_process:
                        print(f"Calculated relative batch size: {new_batch_size}")
                        print(f"Batch size: {batch_size} -> {new_batch_size}")
                    
                    batch_size = new_batch_size
                    
                    if device_type == 'cuda':
                        if master_process: print("Clearing CUDA cache after batch size adjustment...")
                        torch.cuda.empty_cache()

                    if master_process:
                        training_logger.log_operation_success(iter_num, op_name, 
                                                            {'new_batch_size': new_batch_size,
                                                             'original_batch_size': batch_size_to_use,
                                                             'scale_factor': scale_factor})
                except NameError:
                    # Fallback if calculate_relative_batch_size is not available
                    new_batch_size_float = batch_size_to_use * scale_factor
                    new_batch_size = max(8, int(new_batch_size_float // 8) * 8)
                    
                    if master_process:
                        print(f"Fallback scaling batch size: {batch_size_to_use} × {scale_factor:.3f} = {new_batch_size_float:.1f}")
                        print(f"Rounded down to nearest multiple of 8: {new_batch_size}")
                        print(f"Batch size: {batch_size} -> {new_batch_size}")
                    
                    batch_size = new_batch_size
                    
                    if device_type == 'cuda':
                        if master_process: print("Clearing CUDA cache after batch size adjustment...")
                        torch.cuda.empty_cache()

                    if master_process:
                        training_logger.log_operation_success(iter_num, op_name, 
                                                            {'new_batch_size': new_batch_size,
                                                             'original_batch_size': batch_size_to_use,
                                                             'scale_factor': scale_factor})
            else:
                raise ValueError(f"Unknown operation '{op_name}'")
            
            # Log success for operations that don't have their own logging
            if op_name not in ['adjust_batch_size', 'set_batch_size_relative'] and master_process:
                training_logger.log_operation_success(iter_num, op_name, {'new_value': op_value})

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




# Initialize or restore wandb run name
final_wandb_run_name = None
if wandb_log and master_process:
    import wandb
    
    # Try to restore from checkpoint first
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            checkpoint_temp = torch.load(ckpt_path, map_location='cpu')
            if 'final_wandb_run_name' in checkpoint_temp:
                final_wandb_run_name = checkpoint_temp['final_wandb_run_name']
                print(f"Restored W&B run name: {final_wandb_run_name}")
            del checkpoint_temp  # Free memory
    
    # Create new run name if not restored
    if final_wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_wandb_run_name = f"{wandb_run_name}_{timestamp}"
        print(f"Created new W&B run name: {final_wandb_run_name}")
    
    wandb.init(project=wandb_project, name=final_wandb_run_name, config=config)

# Initialize the BatchManager for the training set
batch_manager = BatchManager(
    data_dir=data_dir,
    shard_filenames=train_shard_filenames,
    batch_size=batch_size,
    block_size=block_size,
    device=device,
    device_type=device_type,
    vocab_size=meta_vocab_size,
    starting_estimation_token_count=100_000_000  # ~100M tokens for approximation
)

X, Y = batch_manager.get_next_batch()
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

# Initialize thread pool executor for asynchronous analysis
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
prev_embeddings = None  # This will store the CPU snapshot for semantic drift

# Initialize timing profiler for granular performance measurements
timing_profiler = TimingProfiler()

# Add graceful shutdown handling
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nReceived signal {signum}. Requesting graceful shutdown...")
    shutdown_requested = True

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

t0 = time.time()

# MFU stats logging setup
if master_process and file_logging:
    mfu_log_path = os.path.join(out_dir, 'mfu_stats.txt')
    with open(mfu_log_path, 'w') as f:
        f.write("iter,loss,lr,time_ms,mfu_percent,vram_used_gb,vram_total_gb,vram_percent\n")

# VRAM monitoring
# Import shared VRAM utilities
try:
    from training.utils import get_vram_usage, calculate_relative_batch_size, calculate_optimal_batch_size
except ImportError:
    # Fallback implementation if training.utils is not available
    def get_vram_usage():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3     # GB  
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Use reserved memory as it's more accurate for actual GPU usage
            used_percent = (reserved / total) * 100
            
            return reserved, total, used_percent  # Return reserved instead of allocated
        return 0, 0, 0

def get_detailed_vram_usage():
    """Get detailed VRAM breakdown for debugging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GB  
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        free = total - reserved
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved, 
            'free_gb': free,
            'total_gb': total,
            'allocated_percent': (allocated / total) * 100,
            'reserved_percent': (reserved / total) * 100
        }
    return None

while True:
    # Start timing the training iteration
    timing_profiler.start_iteration()
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        with timing_profiler.time_section("evaluation"):
            losses = estimate_loss()

        if master_process:
            # The rest of the evaluation-related logic that should not be timed
            elapsed_time_seconds = time.time() - start_time
            tokens_per_second = batch_manager.total_tokens_served / elapsed_time_seconds if elapsed_time_seconds > 0 else 0
            wandb_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
                "time/elapsed_seconds": elapsed_time_seconds,
            }

            if 'val_core_acc' in losses:
                wandb_metrics["val/core_accuracy"] = losses['val_core_acc']
            if wandb_log:
                wandb.log(wandb_metrics)

            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr: {lr:.4f}, tokens/sec {tokens_per_second:.0f}")
            print_timings(timing_profiler, training_logger)

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # ... (checkpoint saving logic remains the same)
                    print("Creating universal checkpoint by merging LoRA weights...")
                    universal_state_dict = raw_model.get_merged_state_dict()
                    # ... (rest of checkpoint saving)

            # --- Model Analysis ---
            analyzer = ModelAnalyzer(raw_model)
            memory_info = psutil.virtual_memory()
            if memory_info.percent > 90.0:
                print(f"WARNING: Skipping async analysis due to high system memory usage ({memory_info.percent:.1f}%)")
            else:
                print(f"Dispatching async analysis for iter {iter_num}...")
                try:
                    current_snapshot = analyzer.get_model_state_snapshot()
                    filtered_tokens = batch_manager.get_non_outlier_tokens(ignored_outlayers_sum)
                    X_val, Y_val = get_val_batch()
                    """
                    # 3. Submit the new, generic analysis task to the executor.
                    future = executor.submit(
                        run_full_analysis_async,
                        analyzer,
                        current_snapshot,
                        prev_embeddings, # Will be None on the first run.
                        (X_val, Y_val),
                        iter_num,
                        filtered_tokens
                    )
                    """
                    future.add_done_callback(analysis_done_callback)

                        # 4. CRITICAL: Update state for the next analysis cycle.
                    prev_embeddings = current_snapshot
                    print("Async analysis job dispatched. Training continues.")

                except Exception as dispatch_error:
                    print(f"ERROR dispatching async analysis for iter {iter_num}: {dispatch_error}")

                # --- END OF NEW LOGIC ---

                training_logger.log_step(iter_num, losses['train'], losses['val'], tokens_per_second)

                # Log detailed timing breakdown to file
                avg_timing_percentages = timing_profiler.get_average_percentages(last_n=eval_interval)
                if avg_timing_percentages:
                    timing_breakdown = ", ".join([f"{section} {pct:.1f}%" for section, pct in avg_timing_percentages.items()])
                    training_logger.log(f"  timing breakdown (avg last 10): {timing_breakdown}")
                if wandb_log:
                    wandb_metrics = {
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100,
                        "time/elapsed_seconds": elapsed_time_seconds, # Log elapsed time
                        "throughput/tokens_per_second": tokens_per_second, # Log tokens per second
                    }

                    # Add timing breakdown metrics
                    avg_timing_percentages = timing_profiler.get_average_percentages(last_n=eval_interval)
                    if avg_timing_percentages:
                        for section, percentage in avg_timing_percentages.items():
                            wandb_metrics[f"timing/{section}_pct"] = percentage

                    # Add absolute timing metrics (in milliseconds)
                    for section in ['forward_pass', 'backward_pass', 'data_loading', 'optimizer_step']:
                        stats = timing_profiler.get_summary_stats(section, last_n=eval_interval)
                        if stats:
                            wandb_metrics[f"timing/{section}_avg_ms"] = stats['avg_ms']
                            wandb_metrics[f"timing/{section}_max_ms"] = stats['max_ms']

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
                            'final_wandb_run_name': final_wandb_run_name,  # Save wandb run name for continuity
                            'iter_of_last_op': iter_of_last_op,  # Save operation state
                            'lr_schedule_offset': lr_schedule_offset,  # Save LR schedule state
                            'scaling_schedule': scaling_schedule,  # Save current scaling schedule state
                            'scaling_schedule_file': scaling_schedule_file,  # Save schedule file path
                        }
                        # --- MODIFICATION END ---
                        print(f"saving checkpoint to {out_dir}")
                        
                        # Save to temporary file first, then atomically move to avoid corruption
                        temp_ckpt_path = os.path.join(out_dir, 'ckpt_temp.pt')
                        final_ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                        
                        try:
                            torch.save(checkpoint, temp_ckpt_path)
                            # Atomic move (prevents corruption from abrupt shutdown during save)
                            os.rename(temp_ckpt_path, final_ckpt_path)
                            print(f"Checkpoint saved successfully to {final_ckpt_path}")
                        except Exception as e:
                            print(f"Error saving checkpoint: {e}")
                            # Clean up temp file if it exists
                            if os.path.exists(temp_ckpt_path):
                                os.remove(temp_ckpt_path)
                            raise e
                        
                        # Auto-switch from 'scratch' to 'resume' after first save
                        if init_from == 'scratch' and iter_num > 0:
                            # Update the global config to resume mode for next run
                            import json
                            config_update = {'init_from': 'resume'}
                            config_file = os.path.join(out_dir, 'auto_resume_config.json')
                            with open(config_file, 'w') as f:
                                json.dump(config_update, f, indent=2)
                            print(f"Auto-created resume config at {config_file}. Use --config {config_file} for next run.")
                            
                            # Also create a marker file to indicate this run should auto-resume
                            marker_file = os.path.join(out_dir, '.auto_resume')
                            with open(marker_file, 'w') as f:
                                f.write(f"init_from=resume\nfinal_wandb_run_name={final_wandb_run_name}\n")
                            print(f"Created auto-resume marker at {marker_file}")
            
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
                else:
                    if master_process:
                        print(f"No scaling schedule")
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
                                # Calculate tokens per second for this re-evaluation
                                elapsed_time_seconds = time.time() - start_time
                                tokens_per_second_reeval = batch_manager.total_tokens_served / elapsed_time_seconds if elapsed_time_seconds > 0 else 0
                                print(f"New val loss after operation: {new_val_loss:.4f}")
                                training_logger.log_operation_reevaluation(iter_num, next_op['name'], current_val_loss, new_val_loss)
                                training_logger.log_step(iter_num, losses['train'], new_val_loss, tokens_per_second_reeval)
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

# Time the gradient accumulation loop
    with timing_profiler.time_section("gradient_accumulation"):
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    
            # Remap vocabulary on-the-fly if active
            if remapping_active:
                X, Y = remapping_vector[X], remapping_vector[Y]

            # Time the forward pass specifically
            with timing_profiler.time_section("forward_pass"):
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps

            # Time data loading for next batch
            with timing_profiler.time_section("data_loading"):
                X, Y = batch_manager.get_next_batch()

            # Time the backward pass
            with timing_profiler.time_section("backward_pass"):
                scaler.scale(loss).backward()

    # Time gradient clipping
    with timing_profiler.time_section("gradient_clipping"):
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Time optimizer step
    with timing_profiler.time_section("optimizer_step"):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # End timing the training iteration
    total_iter_time = timing_profiler.end_iteration()

    t1 = time.time()
    if iter_num % log_interval == 0 and master_process:
        # Calculate accumulated time over log_interval iterations
        dt = t1 - t0

        lossf = loss.item() * gradient_accumulation_steps
        # Get VRAM usage
        vram_used, vram_total, vram_percent = get_vram_usage()
        
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt/log_interval)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Print and log (with proper fallback values)
        avg_time_ms = dt/log_interval*1000
        mfu_percent = running_mfu*100 if running_mfu > 0 else 0
        
        print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.5f}, time {avg_time_ms:.2f}ms, mfu {mfu_percent:.2f}%, VRAM {vram_used:.1f}/{vram_total:.1f}GB ({vram_percent:.1f}%)")
        
        # Debug VRAM for first few iterations
        if iter_num <= 50 and iter_num % log_interval == 0:
            detailed_vram = get_detailed_vram_usage()
            if detailed_vram:
                print(f"  [VRAM Debug] Allocated: {detailed_vram['allocated_gb']:.3f}GB ({detailed_vram['allocated_percent']:.1f}%), Reserved: {detailed_vram['reserved_gb']:.3f}GB ({detailed_vram['reserved_percent']:.1f}%)")
        
        # MFU stats logging to dedicated file
        if file_logging:
            with open(mfu_log_path, 'a') as f:
                f.write(f"{iter_num},{lossf:.6f},{lr:.8f},{avg_time_ms:.2f},{mfu_percent:.2f},{vram_used:.3f},{vram_total:.3f},{vram_percent:.2f}\n")
        
        # Log train/loss and iter to wandb at log_interval
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "time/dt_ms": avg_time_ms,
                "mfu": mfu_percent,
                "vram/used_gb": vram_used,
                "vram/total_gb": vram_total,
                "vram/percent": vram_percent
            })
        
        # Reset timer after logging
        t0 = t1

    iter_num += 1
    local_iter_num += 1

    # Check for graceful shutdown request
    if shutdown_requested:
        print("\nGraceful shutdown requested. Saving final checkpoint...")
        if master_process and iter_num > 0:
            # Create emergency checkpoint
            print("Creating emergency checkpoint by merging LoRA weights...")
            universal_state_dict = raw_model.get_merged_state_dict()
            param_names = {name: param for name, param in raw_model.named_parameters()}
            
            emergency_checkpoint = {
                'model': universal_state_dict,
                'optimizer': optimizer.state_dict(),
                'param_names': param_names,
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
                'final_wandb_run_name': final_wandb_run_name,
                'iter_of_last_op': iter_of_last_op,
                'lr_schedule_offset': lr_schedule_offset,
                'scaling_schedule': scaling_schedule,
                'scaling_schedule_file': scaling_schedule_file,
                'emergency_save': True,  # Mark as emergency save
            }
            
            emergency_path = os.path.join(out_dir, f'emergency_ckpt_iter_{iter_num}.pt')
            torch.save(emergency_checkpoint, emergency_path)
            print(f"Emergency checkpoint saved to {emergency_path}")
        print("Graceful shutdown complete. Exiting...")
        break

    if iter_num > max_iters:
        break

if master_process:
    batch_manager.shutdown() # Add this line to stop the background worker
    print("Training finished. Shutting down analysis executor (waiting for any pending jobs)...")
    executor.shutdown(wait=True) # wait=True is important for a clean exit
    training_logger.close()

if ddp:
    destroy_process_group()