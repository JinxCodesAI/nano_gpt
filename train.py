"""
train.py
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

# Composable loss system imports
from loss import DiffusionLoss
from modifiers import TaskWeightingModifier, HardNegativeMiningModifier, StateDependentPenaltyModifier, EntropyPenaltyModifier

def get_curriculum_schedulers(it):
    """
    Controls the curriculum for penalties, data generation, dynamic task weights, and soft labels.
    Returns the current penalty values, re-masking task ratio, dynamic task weights, and soft label alpha.
    """
    corruption_distribution['replace'] = base_corruption_distribution['replace']+base_corruption_distribution['insert']*((proofreading_warmup_iters-it)/proofreading_warmup_iters)+base_corruption_distribution['delete']*(it/proofreading_warmup_iters)
    corruption_distribution['insert'] = base_corruption_distribution['insert']-base_corruption_distribution['insert']*((proofreading_warmup_iters-it)/proofreading_warmup_iters)
    corruption_distribution['delete'] = base_corruption_distribution['delete']-base_corruption_distribution['delete']*((proofreading_warmup_iters-it)/proofreading_warmup_iters)

    if it % 100 == 1:
        print(corruption_distribution)
    # --- Penalty Curriculum for "destructive editing" ---
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
    
    # --- Dynamic Task Weight Scheduling ---
    # Unmask task weight: Linear interpolation from min to max over masking_warmup_iters
    if it >= masking_warmup_iters:
        current_weight_unmask = weight_unmask_task_max
    else:
        ratio = it / masking_warmup_iters
        current_weight_unmask = weight_unmask_task_min + ratio * (weight_unmask_task_max - weight_unmask_task_min)
    
    # Remask task weight: Linear interpolation from min to max over proofreading_warmup_iters  
    if it >= proofreading_warmup_iters:
        current_weight_remask = weight_remask_task_max
    else:
        ratio = it / proofreading_warmup_iters
        current_weight_remask = weight_remask_task_min + ratio * (weight_remask_task_max - weight_remask_task_min)
    
    # --- Soft Label Curriculum (NEW) ---
    # This alpha controls the interpolation between a uniform distribution and a one-hot target.
    # It starts at 0.0 (fully uniform) and ramps to 1.0 (fully one-hot).
    if it >= soft_label_warmup_iters:
        soft_label_alpha = 1.0
    else:
        soft_label_alpha = it / soft_label_warmup_iters
        
    return current_penalty_mask_correct, remask_ratio, current_weight_unmask, current_weight_remask, soft_label_alpha

def log_diffusion_diagnostics(logits, targets, inputs, mask_token_id, replace_token_id, meta_vocab_size, 
                              iter_num, wandb_log=False):
    """
    A unified function to calculate and print key diagnostic metrics.
    This version includes Precision and Recall for the re-masking task.
    """
    with torch.no_grad():
        # --- 1. Logit Sanity Check (Unchanged) ---
        avg_mask_logit = logits[:, :, mask_token_id].mean().item()
        avg_wrong_logit = logits[:, :, replace_token_id].mean().item()
        avg_max_logit, predictions = logits.max(dim=-1)
        avg_max_logit = avg_max_logit.mean().item()

        print(
            f"[LOGITS] Avg MASK: {avg_mask_logit:7.2f} | "
            f"Avg WRONG: {avg_wrong_logit:7.2f} | "
            f"Avg MAX: {avg_max_logit:7.2f}"
        )

        # --- Setup for Behavioral Analysis ---
        epsilon = 1e-6
        flat_inputs = inputs.view(-1)
        flat_predictions = predictions.view(-1)
        
        # Convert targets to hard labels for metrics if they are soft
        if targets.dtype == torch.long:
            hard_targets = targets
        else:
            hard_targets = torch.argmax(targets, dim=-1)
        flat_hard_targets = hard_targets.view(-1)
        
        # --- 2. Un-masking Confidence and Accuracy (Unchanged) ---
        unmask_task_mask = (flat_inputs == mask_token_id) & (flat_hard_targets != replace_token_id)
        total_unmask_tasks = unmask_task_mask.sum().item()
        
        if total_unmask_tasks > 0:
            unmask_predictions = flat_predictions[unmask_task_mask]
            unmask_targets = flat_hard_targets[unmask_task_mask]
            correct_unmasks_mask = (unmask_predictions == unmask_targets)
            correct_unmasks_count = correct_unmasks_mask.sum().item()
            unmask_accuracy = correct_unmasks_count / total_unmask_tasks
            skill_vs_random = unmask_accuracy / (1.0 / meta_vocab_size if meta_vocab_size is not None else 1.0)
            incorrect_unmask_positions = unmask_task_mask.nonzero(as_tuple=True)[0][~correct_unmasks_mask]
            if len(incorrect_unmask_positions) > 0:
                incorrect_logits = logits.view(-1, logits.size(-1))[incorrect_unmask_positions]
                incorrect_probs = F.softmax(incorrect_logits, dim=-1)
                confidence_in_wrong_guess = incorrect_probs.max(dim=-1).values.mean().item()
            else:
                confidence_in_wrong_guess = 0.0
            kept_mask_count = (unmask_predictions == mask_token_id).sum().item()
        else:
            unmask_accuracy, skill_vs_random, confidence_in_wrong_guess, kept_mask_count = 0.0, 0.0, 0.0, 0

        print(
            f"[UNMASK] Accuracy: {unmask_accuracy:<6.2%} | "
            f"Skill: {skill_vs_random:<5.1f}x | "
            f"Conf. on Wrong: {confidence_in_wrong_guess:<6.2%} | "
            f"Kept Mask: {kept_mask_count}"
        )
        
        # --- 3. Re-masking Precision, Recall, and Error Analysis (NEW) ---
        remask_task_mask = (flat_inputs != mask_token_id) & (flat_hard_targets == replace_token_id)
        total_remask_tasks = remask_task_mask.sum().item()
        
        if total_remask_tasks > 0:
            # True Positives (TP): Correctly predicted [WRONG] where it should be.
            true_positives = (remask_task_mask & (flat_predictions == replace_token_id)).sum().item()
            
            # False Negatives (FN): Failed to predict [WRONG] where it should be.
            false_negatives = total_remask_tasks - true_positives
            
            # Recall = TP / (TP + FN) = TP / (Total Actual Positives)
            recall = true_positives / total_remask_tasks
            
            # False Positives (FP): Predicted [WRONG] where it shouldn't be.
            should_not_be_wrong_mask = (flat_hard_targets != replace_token_id)
            false_positives = (should_not_be_wrong_mask & (flat_predictions == replace_token_id)).sum().item()
            
            # Precision = TP / (TP + FP) = TP / (Total Predicted Positives)
            total_predicted_positives = true_positives + false_positives
            precision = true_positives / (total_predicted_positives + epsilon)
        else:
            recall, precision, false_positives, false_negatives = 0.0, 0.0, 0, 0

        print(
            f"[REMASK] Recall: {recall:<6.2%} | "
            f"Precision: {precision:<6.2%} | "
            f"FP: {false_positives:<5} | "
            f"FN: {false_negatives:<5} | "
            f"Tasks: {total_remask_tasks}"
        )

        # --- 4. Curriculum Status (Unchanged) ---
        current_penalty, remask_ratio, _, _, soft_label_alpha = get_curriculum_schedulers(iter_num)
        print(
            f"[CURRICULUM] Penalty: {current_penalty:.3f} | "
            f"Remask Ratio: {remask_ratio:.3f} | "
            f"Soft Label Alpha: {soft_label_alpha:.3f}"
        )
        
        # --- 5. Optional WandB Logging (Updated) ---
        if wandb_log:
            import wandb
            wandb.log({
                "diffusion/avg_mask_logit": avg_mask_logit,
                "diffusion/avg_wrong_logit": avg_wrong_logit,
                "diffusion/avg_max_logit": avg_max_logit,
                "diffusion/unmask_accuracy": unmask_accuracy,
                "diffusion/remask_recall": recall, # Updated metric
                "diffusion/remask_precision": precision, # New metric
                "diffusion/remask_false_positives": false_positives, # New metric
                "diffusion/remask_false_negatives": false_negatives, # New metric
                "diffusion/skill_vs_random": skill_vs_random,
                "diffusion/confidence_wrong_guess": confidence_in_wrong_guess,
                "diffusion/kept_mask_count": kept_mask_count,
                "curriculum/penalty_mask_correct": current_penalty,
                "curriculum/remask_ratio": remask_ratio,
                "curriculum/soft_label_alpha": soft_label_alpha,
            }, step=iter_num)
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
# Dynamic task-based loss weights (linearly interpolated)
weight_unmask_task_min = 1.5    # Initial weight for the "fill-in-the-blank" task
weight_unmask_task_max = 1.0    # Final weight for the "fill-in-the-blank" task
weight_remask_task_min = 0.5    # Initial weight for the "proofreading" task  
weight_remask_task_max = 1.0    # Final weight for the "proofreading" task

# Curriculum settings
penalty_mask_correct = 0.5    # Final discount for wrongly masking a correct token.
masking_warmup_iters = 1000   # Iterations to ramp up the penalty_mask_correct.
proofreading_warmup_iters = 2000 # Iterations to ramp up the "re-masking" task.
soft_label_warmup_iters = 5000 # NEW: Iterations to transition from soft to hard labels.

base_corruption_distribution = {
    'replace': 0.5,  # 50% of corruptions will be replacements
    'insert': 0.25,   # 25% will be insertions
    'delete': 0.25    # 25% will be deletions
}

corruption_distribution = {
    'replace': 0.5,  # 50% of corruptions will be replacements
    'insert': 0.25,   # 25% will be insertions
    'delete': 0.25    # 25% will be deletions
}

# Ensure they sum to 1 for safety
assert sum(corruption_distribution.values()) == 1.0, "Corruption distribution must sum to 1.0"

guaranteed_correct_factor = 0.01

# --- COMPOSABLE LOSS MODIFIERS CONFIG ---
# Enable/disable individual loss modifiers (easy to toggle on/off)
use_task_weighting = True           # Apply different weights to unmask/remask tasks
use_hard_negative_mining = True     # Apply higher weights to identity positions
use_state_dependent_penalty = True  # Apply dynamic penalties for destructive edits
entropy_penalty = 0.0

# Hard negative mining settings
weight_identity_task = 3.0  # Weight for identity task positions

# Diagnostic logging settings
log_diagnostics_interval = 100  # Log detailed diagnostics every N iterations
enable_diagnostics = True       # Master toggle for diagnostic logging

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
dry_run = False 
_DRY_RUN_SOURCES = []
itos = None

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
mask_token_id = None # Global variable to be set after model init
replace_token_id = None # Global variable to be set after model init
insert_token_id = None # Global variable to be set after model init
delete_token_id = None # Global variable to be set after model init

def get_safe_indices(x, y, special_token_ids):
    """
    Returns a tensor of indices that are safe to corrupt.
    A position is "safe" if neither it nor its left neighbor is a special token.
    """
    safe_mask = torch.ones(len(x), dtype=torch.bool)
    for token_id in special_token_ids:
        safe_mask &= (x != token_id)
    
    # Shift the mask right to check the left neighbor
    safe_mask_shifted = F.pad(safe_mask[:-1], (1, 0), 'constant', True)
    
    # A position is safe if both it AND its left neighbor are not special tokens
    final_safe_mask = safe_mask & safe_mask_shifted
    
    return torch.where(final_safe_mask)[0]

def apply_replacements(x, y, num_to_replace, special_token_ids):
    """Applies substitution corruption to safe indices."""
    if num_to_replace == 0: return x, y
    
    safe_indices = get_safe_indices(x, y, special_token_ids)
    if len(safe_indices) < num_to_replace: return x, y # Not enough safe places
    
    indices_to_replace = safe_indices[torch.randperm(len(safe_indices))[:num_to_replace]]
    
    random_tokens = torch.randint(1, meta_vocab_size, (num_to_replace,))
    x[indices_to_replace] = (x[indices_to_replace] + random_tokens) % meta_vocab_size
    y[indices_to_replace] = replace_token_id
    
    return x, y

def apply_insertions(x, y, num_to_insert, special_token_ids):
    """Applies insertion corruption at safe indices."""
    if num_to_insert == 0: return x, y
    
    safe_indices = get_safe_indices(x, y, special_token_ids)
    if len(safe_indices) < num_to_insert: return x, y
    
    indices_to_delete = safe_indices[torch.randperm(len(safe_indices))[:num_to_insert]]
    
    # The target for the token *before* the deletion is [INSERT]
    insert_target_indices = indices_to_delete - 1
    y[insert_target_indices] = insert_token_id
    
    keep_mask = torch.ones_like(x, dtype=torch.bool)
    keep_mask[indices_to_delete] = False
    
    return x[keep_mask], y[keep_mask]

def apply_deletions(x, y, num_to_delete, special_token_ids):
    """Applies deletion corruption at safe indices."""
    if num_to_delete == 0: return x, y
    
    safe_indices = get_safe_indices(x, y, special_token_ids)
    if len(safe_indices) < num_to_delete: return x, y
        
    indices_to_insert = safe_indices[torch.randperm(len(safe_indices))[:num_to_delete]]
    indices_to_insert, _ = torch.sort(indices_to_insert)
    
    random_tokens = torch.randint(1, meta_vocab_size, (num_to_delete,))
    
    for i in reversed(range(num_to_delete)):
        idx = indices_to_insert[i]
        x = torch.cat([x[:idx], random_tokens[i].unsqueeze(0), x[idx:]])
        y = torch.cat([y[:idx], torch.tensor([delete_token_id]), y[idx:]])
        
    return x, y

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    
    if model_type == 'diffusion':
        special_token_ids = [mask_token_id, replace_token_id, insert_token_id, delete_token_id]
        assert all(id is not None for id in special_token_ids), "Special tokens not initialized"
        
        # Get curriculum ratios for this iteration
        _, remask_ratio, _, _, soft_label_alpha = get_curriculum_schedulers(iter_num)

        batch_x, batch_y = [], []
        for _ in range(batch_size):
            # --- Step 1: Determine total corruption and its distribution ---
            max_corruption = 1 - guaranteed_correct_factor
            # The total number of corruptions is scaled by the remask_ratio curriculum
            total_corruption_rate = torch.rand(1) * max_corruption * remask_ratio
            total_corruptions = int(total_corruption_rate * block_size)
            
            # Distribute the total corruptions according to our defined proportions
            num_to_replace = int(total_corruptions * corruption_distribution['replace'])
            num_to_insert = int(total_corruptions * corruption_distribution['insert'])
            num_to_delete = int(total_corruptions * corruption_distribution['delete'])

            # --- Step 2: Determine source length and read data ---
            source_len = block_size + num_to_insert - num_to_delete
            if source_len <= 0: continue # Skip if corruptions are too extreme
            
            start_idx = torch.randint(len(data) - source_len, (1,)).item()
            x_clean = torch.from_numpy(data[start_idx : start_idx + source_len].astype(np.int64))
            y_clean = x_clean.clone()

            if dry_run:
                _DRY_RUN_SOURCES.append(x_clean.clone())

            # --- Step 3: Apply corruptions (order can matter, let's be consistent) ---
            # Deletions -> Insertions -> Replacements
            x_corrupted, y_targets = apply_deletions(x_clean, y_clean, num_to_delete, special_token_ids)
            x_corrupted, y_targets = apply_insertions(x_corrupted, y_targets, num_to_insert, special_token_ids)
            x_corrupted, y_targets = apply_replacements(x_corrupted, y_targets, num_to_replace, special_token_ids)
            
            # --- Step 4: Pad/truncate to final block_size ---
            if len(x_corrupted) > block_size:
                x_corrupted = x_corrupted[:block_size]
                y_targets = y_targets[:block_size]
            elif len(x_corrupted) < block_size:
                padding_len = block_size - len(x_corrupted)
                x_corrupted = F.pad(x_corrupted, (0, padding_len), 'constant', mask_token_id)
                y_targets = F.pad(y_targets, (0, padding_len), 'constant', -1)
            
            batch_x.append(x_corrupted)
            batch_y.append(y_targets)

        if not batch_x: return get_batch(split) # Retry if all samples were skipped

        x = torch.stack(batch_x)
        y_hard_targets = torch.stack(batch_y)

        # Apply soft labels if needed (your existing logic)
        y = y_hard_targets # Sticking to hard labels for now

    else: # Autoregressive path
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def run_diagnostic_dry_run():
    """
    Generates one batch of data and prints a detailed, human-readable
    visualization for debugging the corruption pipeline, then exits.
    """
    print("\n" + "="*80)
    print("--- RUNNING DIAGNOSTIC DRY RUN ---")
    print("="*80)

    # --- Set curriculum to final state for a worst-case test ---
    global iter_num
    iter_num = proofreading_warmup_iters 
    print(f"\nFetching one batch with curriculum schedulers set for iter_num = {iter_num}...")

    # --- Generate one batch using the exact, unmodified training function ---
    X, Y = get_batch('train')
    
    # The _DRY_RUN_SOURCES global variable has now been populated by the hook.
    
    if X is None or not _DRY_RUN_SOURCES:
        print("\nERROR: Failed to generate a batch. Check configurations.")
        return

    print(f"Successfully generated a batch of size {X.shape[0]}. Visualizing samples...")

    # --- Define a decoder for visualization ---
    if 'itos' not in globals():
        print("\nERROR: 'itos' mapping not found. Cannot decode tokens.")
        return
        
    special_token_map = {
        mask_token_id: "░",
        replace_token_id: "[R]",
        insert_token_id: "[I]",
        delete_token_id: "[D]",
        -1: "Ø" # For ignored padding in targets
    }
    def test_decode(tokens):
        return "".join([itos.get(t, special_token_map.get(t, f'<{t}>')) for t in tokens])

    # --- Print each sample ---
    num_samples_to_show = min(5, batch_size)
    for i in range(num_samples_to_show):
        print("\n" + "="*80)
        print(f"--- SAMPLE {i+1} ---")
        print("="*80)
        
        source_text = test_decode(_DRY_RUN_SOURCES[i].tolist())
        input_text = test_decode(X[i].tolist())
        target_text = test_decode(Y[i].tolist())
        
        print(f"\n[SOURCE] (Length: {len(_DRY_RUN_SOURCES[i])})\n{source_text}")
        print(f"\n[INPUT X] (Length: {len(X[i])})\n{input_text}")
        print(f"\n[TARGET Y] (Length: {len(Y[i])})\n{target_text}")

        counts = {
            'replace': (Y[i] == replace_token_id).sum().item(),
            'insert': (Y[i] == insert_token_id).sum().item(),
            'delete': (Y[i] == delete_token_id).sum().item(),
        }
        print(f"\n[STATS] Replacements: {counts['replace']} | Insertions: {counts['insert']} | Deletions: {counts['delete']}")
        print("="*80, flush=True)

    print("\n--- DRY RUN COMPLETE ---")
# =============================================================================

# =============================================================================

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
    itos = meta['itos']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, model_type=model_type) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # --- MODIFICATION ---
    if model_type == 'diffusion':
        # Vocab size is now +2 for [MASK] and [REPLACE] [INSERT] [DELETE]
        vocab_size = meta_vocab_size + 4 if meta_vocab_size is not None else 50306
        model_args['vocab_size'] = vocab_size
        # Assign the last two token IDs
        model_args['delete_token_id'] = vocab_size - 4
        model_args['insert_token_id'] = vocab_size - 3
        model_args['mask_token_id'] = vocab_size - 2
        model_args['replace_token_id'] = vocab_size - 1
        # Make them globally available to get_batch
        mask_token_id = model_args['mask_token_id']
        replace_token_id = model_args['replace_token_id']
        delete_token_id = model_args['delete_token_id']
        insert_token_id = model_args['insert_token_id']
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
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'model_type', 'mask_token_id', 'replace_token_id']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # Set global tokens if resuming diffusion model
    if model_args.get('model_type') == 'diffusion':
        if 'mask_token_id' in model_args:
            mask_token_id = model_args['mask_token_id']
        if 'replace_token_id' in model_args:
            replace_token_id = model_args['replace_token_id']
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


if dry_run:
    # Override config for a modest dry run if not specified otherwise
    guaranteed_correct_factor = 0.70
    corruption_distribution['replace'] = 0.333
    corruption_distribution['insert'] = 0.333
    corruption_distribution['delete'] = 0.333
    
    run_diagnostic_dry_run()
    exit() # Exit the script after the dry run is complete


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# Initialize composable loss system for diffusion model
loss_fn = None
if model_type == 'diffusion':
    print("Initializing composable diffusion loss system...")
    
    # Create the base loss function
    loss_fn = DiffusionLoss(mask_token_id, replace_token_id)
    
    # Add modifiers based on configuration
    if use_task_weighting:
        print("  - Adding task weighting modifier")
        # Get initial weights from scheduler
        _, _, initial_weight_unmask, initial_weight_remask, _ = get_curriculum_schedulers(iter_num)
        loss_fn.add_modifier(TaskWeightingModifier(
            weight_unmask=initial_weight_unmask,  # Will be updated dynamically
            weight_remask=initial_weight_remask   # Will be updated dynamically
        ))
    
    if use_hard_negative_mining:
        print("  - Adding hard negative mining modifier")
        loss_fn.add_modifier(HardNegativeMiningModifier(
            weight_identity=weight_identity_task
        ))
    
    if use_state_dependent_penalty:
        print("  - Adding state-dependent penalty modifier")
        loss_fn.add_modifier(StateDependentPenaltyModifier(
            penalty_strength=penalty_mask_correct  # Will be updated dynamically
        ))

    if entropy_penalty>0:
        print("  - Adding entropy modifier")
        loss_fn.add_modifier(EntropyPenaltyModifier(
            penalty_strength=entropy_penalty,
            vocab_size=vocab_size
            
        ))

    
    
    print(f"  - Loss system initialized with {len(loss_fn.modifiers)} modifiers")

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
    current_penalty_mask_correct, _, weight_unmask_task, weight_remask_task, _ = get_curriculum_schedulers(iter_num)
    
    # Update dynamic weights in composable loss system
    if model_type == 'diffusion' and loss_fn is not None:
        # Update task weighting modifier weights
        for modifier in loss_fn.modifiers:
            if isinstance(modifier, TaskWeightingModifier):
                modifier.weight_unmask = weight_unmask_task
                modifier.weight_remask = weight_remask_task
            elif isinstance(modifier, StateDependentPenaltyModifier):
                modifier.penalty_strength = current_penalty_mask_correct
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
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
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{str(t0)}_{iter_num}.pt'))
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
            # --- MODIFICATION ---
            if model_type == 'diffusion':
                # Enable diagnostics periodically during training
                should_log_diagnostics = (enable_diagnostics and 
                                        iter_num % log_diagnostics_interval == 0 and 
                                        micro_step == 0)  # Only log on first micro-step
                
                # Use the new composable loss system
                loss = loss_fn(logits, Y, X, log_diagnostics=should_log_diagnostics)
                if iter_num % log_diagnostics_interval == 0:
                    log_diffusion_diagnostics(logits, Y, X, mask_token_id, replace_token_id, vocab_size, iter_num)
            else:
                loss = loss_from_model
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # --- END MODIFICATION ---
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

if ddp:
    destroy_process_group()
