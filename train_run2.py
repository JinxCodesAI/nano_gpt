"""
Refactored training script for diffusion training.

This is a clean, readable version of train_run.py that mirrors the original train.py structure
while preserving all advanced features through modular design.

Key improvements:
- Clean training loop starting early in the file
- All complex logic moved to specialized modules
- Maintains all functionality from train_run.py
- Easy to understand and modify
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
    get_batch, estimate_loss, get_lr, load_synthetic_model,
    start_prefetch, stop_prefetch, TrainingContext, UnmaskingStage, update_stage_progress,
    create_unmasking_validation_set, create_sequence_scoring_validation_set, UnmaskingStageType,
    StickyStageConfig, RandomStageConfig, SpanStageConfig, create_stage_objects,
    calculate_wrong_answer_entropy, get_current_entropy_penalty, update_entropy_multiplier_ema,
    apply_label_smoothing,
    # Loss processing functions
    calculate_per_sample_losses, apply_per_sample_modifications,
    # New refactoring modules
    CheckpointManager, InstabilityDetector, TrainingLogger, ModelInitializer, SourceCodePrinter
)

torch._dynamo.config.suppress_errors = True

# Global timer instance
timer = Timer()

# -----------------------------------------------------------------------------
# Default config values 
# I/O
out_dir = 'out'
training_type = 'unmasking'  
eval_interval = 200
log_interval = 20
eval_iters = 20
eval_only = False
always_save_checkpoint = True
init_from = 'resume'
ckpt_filename = '34.5_58.4_UM.pt'

# Model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.01
bias = False
attention_type = 'bidirectional'
use_rope = True

# WandB logging
wandb_log = True
wandb_project = 'diffusion'
wandb_run_name = '13k_UN_noise_0.2'

# Data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 1024
use_paragraph_boundaries = False

# Training mode config
training_type = 'unmasking'
num_token_classes = 2

# Transfer learning config
init_from_checkpoint = ""
unfreeze_at_iteration = 80
unfreeze_lr_multiplier = 1

# Sequence scoring config
unmasking_model_checkpoint = ""

# Unmasking training config
unmasking_stages = []
validation_stages = []
use_all_stages_for_training = False
weight_loss_by_mask_ratio = False
enable_entropy_penalty = False
max_entropy_penalty = 0.5
entropy_penalty_start_iter = 6000
uncertainty_factor = 0.0

# AdamW optimizer
learning_rate = 1e-3
max_iters = 50000
warmup_iters = 2000
lr_decay_iters = 41000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-3

grad_clip = 1.0
decay_lr = True

# DDP settings
backend = 'nccl'

# System
device = 'cuda'
dtype = 'float16'
compile = False
start_iter_num = 0

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())

# Auto-configure freezing
freeze_transformer = (training_type == 'sequence_scoring' and unfreeze_at_iteration is not None)

if len(unmasking_stages) == 0 or unmasking_stages is None:
    print("No unmasking stages defined, exiting...")
    exit()

# Update wandb run name after configuration is loaded
if training_type == 'unmasking':
    wandb_run_name = f'{wandb_run_name}_unmasking'

config = {k: globals()[k] for k in config_keys}

# Print source code and global variables on startup (optional - disabled by default for cleaner output)
if os.environ.get('PRINT_SOURCE_CODE', '0') == '1':
    SourceCodePrinter.print_source_code_and_globals(globals())

# -----------------------------------------------------------------------------
# Setup and initialization

# DDP setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Data setup
data_dir = os.path.join('data', dataset)

# Initialize variables
iter_num = 0
best_val_loss = 1e9
checkpoint_training_context = None

# Derive vocab_size from dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    mask_token_id = meta_vocab_size
    extended_vocab_size = meta_vocab_size + 15
    print(f"mask_token_id = {mask_token_id}, extended_vocab_size = {extended_vocab_size}")
else:
    mask_token_id = 65
    extended_vocab_size = 65 + 15

# Create training context with all parameters
unmasking_stage_objects = create_stage_objects(unmasking_stages) if training_type in ['unmasking', 'sequence_scoring'] else None
validation_stage_objects = create_stage_objects(validation_stages) if training_type in ['unmasking', 'sequence_scoring'] and validation_stages else None

training_ctx = TrainingContext(
    training_type=training_type,
    num_token_classes=num_token_classes,
    batch_size=batch_size,
    block_size=block_size,
    max_iters=max_iters,
    device=device,
    device_type=device_type,
    seed_offset=seed_offset,
    data_dir=data_dir,
    meta_vocab_size=meta_vocab_size,
    mask_token_id=mask_token_id,
    extended_vocab_size=extended_vocab_size,
    iter_num=iter_num,
    unmasking_stages=unmasking_stage_objects,
    validation_stages=validation_stage_objects,
    eval_iters=eval_iters,
    warmup_iters=warmup_iters,
    lr_decay_iters=lr_decay_iters,
    learning_rate=learning_rate,
    min_lr=min_lr,
    use_paragraph_boundaries=use_paragraph_boundaries,
    use_all_stages_for_training=use_all_stages_for_training,
    weight_loss_by_mask_ratio=weight_loss_by_mask_ratio,
    enable_entropy_penalty=enable_entropy_penalty,
    max_entropy_penalty=max_entropy_penalty,
    entropy_penalty_start_iter=entropy_penalty_start_iter,
    uncertainty_factor=uncertainty_factor,
    freeze_transformer=freeze_transformer,
    init_from_checkpoint=init_from_checkpoint,
    unfreeze_at_iteration=unfreeze_at_iteration,
    unfreeze_lr_multiplier=unfreeze_lr_multiplier
)

# Initialize specialized modules
checkpoint_manager = CheckpointManager(out_dir, training_type, device)
instability_detector = InstabilityDetector(checkpoint_manager)
logger = TrainingLogger(wandb_log, master_process, log_interval, eval_interval)
model_initializer = ModelInitializer(device, device_type)

# Model mode validation and configuration
cls_token_id = None
if training_type == 'sequence_scoring':
    cls_token_id = meta_vocab_size + 5 if meta_vocab_size is not None else 70
    print(f"Setting cls_token_id = {cls_token_id} (using reserved special token slot)")
    training_ctx.cls_token_id = cls_token_id

    if unmasking_model_checkpoint is None:
        raise ValueError("Sequence scoring requires unmasking_model_checkpoint to be specified")

    if not os.path.exists(unmasking_model_checkpoint):
        raise FileNotFoundError(f"Unmasking model checkpoint not found: {unmasking_model_checkpoint}")

model_mode = model_initializer.validate_model_configuration(
    training_type, attention_type, num_token_classes, cls_token_id
)

# Model initialization
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout, attention_type=attention_type,
    use_rope=use_rope, mode=model_mode, num_token_classes=num_token_classes,
    cls_token_id=cls_token_id, freeze_transformer=freeze_transformer,
    init_from_checkpoint=init_from_checkpoint, unfreeze_at_iteration=unfreeze_at_iteration,
    unfreeze_lr_multiplier=unfreeze_lr_multiplier
)

if init_from == 'scratch':
    model = model_initializer.create_model_from_scratch(model_args, extended_vocab_size)
elif init_from == 'resume':
    model, iter_num, best_val_loss, checkpoint_training_context = model_initializer.resume_model_from_checkpoint(
        checkpoint_manager, model_args, ckpt_filename
    )
    start_iter_num = iter_num
    training_ctx.extended_vocab_size = model_args['vocab_size']

# Load unmasking model for sequence scoring if required
if training_type == 'sequence_scoring':
    unmasking_model = model_initializer.load_unmasking_model(
        unmasking_model_checkpoint, extended_vocab_size, block_size
    )
    training_ctx.unmasking_model = unmasking_model

# Apply restored training context state if resuming
if init_from == 'resume' and checkpoint_training_context is not None:
    print("Applying restored training context state...")
    training_ctx.current_stage = checkpoint_training_context.get('current_stage', 0)
    training_ctx.val_loss_stale_count = checkpoint_training_context.get('val_loss_stale_count', 0)
    training_ctx.best_val_loss_this_stage = checkpoint_training_context.get('best_val_loss_for_stage', float('inf'))
    training_ctx.entropy_multiplier_ema = checkpoint_training_context.get('entropy_multiplier_ema', 1.0)

# Transfer learning setup
if init_from_checkpoint and init_from_checkpoint != "":
    model = model_initializer.setup_transfer_learning(model, init_from_checkpoint)

# Crop model block size if needed
model_initializer.crop_model_block_size(model, block_size, model_args)

# Compile and wrap model
model = model_initializer.compile_and_wrap_model(model, compile, ddp, ddp_local_rank if ddp else None)

# Initialize optimizer and scaler
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint_manager.load_checkpoint(
        checkpoint_manager.find_latest_checkpoint(ckpt_filename)
    )['optimizer'])

# WandB logging setup
if wandb_log and master_process and not eval_only:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Print model summary
if master_process:
    raw_model = model.module if ddp else model
    print(f"Model: {raw_model.get_num_params()/1e6:.2f}M params, {training_type} training, vocab={extended_vocab_size}, block={block_size}")
    if os.environ.get('VERBOSE_MODEL_INFO', '0') == '1':
        SourceCodePrinter.print_model_architecture_summary(model)

# -----------------------------------------------------------------------------
# Helper functions (like original train.py)

def simple_estimate_loss():
    """Simple loss estimation wrapper."""
    with timer.time_function('validation'):
        training_ctx.iter_num = iter_num
        losses = estimate_loss(model, ctx, timer, training_ctx)
    return losses

def simple_save_checkpoint():
    """Simple checkpoint saving wrapper."""
    if losses['val'] < best_val_loss or always_save_checkpoint:
        training_context_state = {
            'current_stage': training_ctx.current_stage,
            'val_loss_stale_count': training_ctx.val_loss_stale_count,
            'best_val_loss_for_stage': training_ctx.best_val_loss_this_stage,
            'entropy_multiplier_ema': training_ctx.entropy_multiplier_ema
        }

        checkpoint_manager.save_checkpoint(
            model, optimizer, iter_num, best_val_loss, config,
            training_context_state, model_args
        )

# -----------------------------------------------------------------------------
# Training setup

# Initialize validation sets
if training_ctx.training_type == 'unmasking':
    print("Pre-creating validation set...")
    create_unmasking_validation_set(training_ctx)

    if training_ctx.use_all_stages_for_training:
        print("Training will generate fresh batches from all stages each iteration")

elif training_ctx.training_type == 'sequence_scoring':
    print(f"Sequence scoring training initialized with {len(training_ctx.validation_stages)} validation stages")
    create_sequence_scoring_validation_set(training_ctx)

# Get initial batch
X, Y, mask = get_batch('train', training_ctx)

# Training state
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
just_recovered = False

# Show initial frozen status for transfer learning
if freeze_transformer and hasattr(raw_model, 'get_frozen_status'):
    print(f"Initial frozen status: {raw_model.get_frozen_status()}")
    raw_model.print_parameter_status()

# Show initial stage configuration (simplified)
if training_ctx.training_type == 'unmasking':
    stage_config = training_ctx.get_current_stage_config()
    stage_type = stage_config.get_stage_type()
    print(f"Unmasking training: Stage {training_ctx.current_stage} ({stage_type.value}), {len(training_ctx.unmasking_stages)} total stages")

logger.print_and_flush("Starting training loop...")

# -----------------------------------------------------------------------------
# MAIN TRAINING LOOP (Clean and prominent like original train.py)

while True:

    # Set learning rate for this iteration
    lr = get_lr(iter_num, training_ctx) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    training_ctx.current_lr = lr  # Store for logging

    # Dynamic unfreezing logic
    if (unfreeze_at_iteration is not None and
        iter_num == unfreeze_at_iteration and
        hasattr(raw_model, 'get_frozen_status') and
        raw_model.get_frozen_status()):

        logger.print_and_flush(f"\n*** DYNAMIC UNFREEZING at iteration {iter_num} ***")
        logger.print_and_flush("Switching from feature extraction to fine-tuning mode")

        # Unfreeze transformer weights
        raw_model.unfreeze_transformer_weights()

        # Recreate optimizer to include newly unfrozen parameters
        logger.print_and_flush("Recreating optimizer to include unfrozen transformer parameters")
        current_lr = learning_rate * unfreeze_lr_multiplier if unfreeze_lr_multiplier < 1.0 else learning_rate
        optimizer = raw_model.configure_optimizers(weight_decay, current_lr, (beta1, beta2), device_type)

        if unfreeze_lr_multiplier < 1.0:
            logger.print_and_flush(f"Using reduced learning rate: {current_lr:.6f} (multiplier: {unfreeze_lr_multiplier})")
            learning_rate = current_lr

        # Verify optimizer now tracks all parameters
        optimizer_tensor_count = sum(len(group['params']) for group in optimizer.param_groups)
        total_trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        num_param_groups = len(optimizer.param_groups)
        logger.print_and_flush(f"Optimizer now tracking {num_param_groups} parameter groups with {optimizer_tensor_count} parameter tensors ({total_trainable_params:,} individual parameters)")

        raw_model.print_parameter_status()
        logger.print_and_flush("*** UNFREEZING COMPLETE ***\n")

    # Validation and checkpointing
    if (iter_num % eval_interval == 0 and master_process and not just_recovered) or eval_only:
        losses = simple_estimate_loss()

        # Check validation stability
        if not instability_detector.check_validation_stability(losses, iter_num):
            break

        # Log validation results
        logger.log_validation_results(iter_num, losses, lr, training_ctx)

        # Update stage progress
        stage_advanced = update_stage_progress(training_ctx, losses['val'])
        if stage_advanced:
            logger.print_and_flush(f"Advanced to stage {training_ctx.current_stage} - validation set remains consistent across all stages")

        # Log mask encounter statistics
        raw_model.log_mask_encounter_stats("  ")
        raw_model.reset_mask_encounter_stats()

        # Save checkpoint
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                simple_save_checkpoint()

        # WandB logging for validation
        if wandb_log and master_process and not eval_only:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "model vs random": losses.get('val_model_vs_random', 0.0),
                "signal to noise": losses.get('val_signal_to_noise', 0.0),
                "signal to noise median": losses.get('val_signal_to_noise_median', 0.0),
                "mfu": running_mfu*100,
                "masked_token_ratio": losses.get('train_masked_token_ratio', 0.0),
                "min_masked_token_ratio": losses.get('train_min_masked_token_ratio', 0.0),
                "max_masked_token_ratio": losses.get('train_max_masked_token_ratio', 0.0),
            }

            if training_ctx.enable_entropy_penalty:
                current_entropy_penalty = get_current_entropy_penalty(iter_num, training_ctx)
                log_dict["entropy_penalty"] = current_entropy_penalty
                log_dict["entropy_multiplier_ema"] = training_ctx.entropy_multiplier_ema

            # Add per-stage validation losses
            for stage_idx in range(len(training_ctx.validation_stages or [])):
                stage_loss_key = f'val_stage_{stage_idx}_loss'
                stage_samples_key = f'val_stage_{stage_idx}_samples'
                if stage_loss_key in losses:
                    log_dict[f'val/stage_{stage_idx}_loss'] = losses[stage_loss_key]
                    log_dict[f'val/stage_{stage_idx}_samples'] = losses[stage_samples_key]

            logger.log_to_wandb(log_dict)

    if eval_only:
        break

    # Forward backward update with gradient accumulation (like original train.py)
    with timer.time_function('gradient_accumulation_loop'):
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            with ctx:
                with timer.time_function('forward_pass'):
                    logits, loss = model(X, Y)

                # Instability detection
                with timer.time_function('instability_detection'):
                    if not instability_detector.check_logits_stability(logits, iter_num):
                        success, X, Y, mask = instability_detector.attempt_recovery(
                            model, optimizer, training_ctx, scaler, get_batch
                        )
                        if success:
                            local_iter_num = 0
                            running_mfu = -1.0
                            just_recovered = True
                            t0 = time.time()
                            continue
                        else:
                            break

                    if not instability_detector.check_loss_stability(loss, iter_num):
                        success, X, Y, mask = instability_detector.attempt_recovery(
                            model, optimizer, training_ctx, scaler, get_batch
                        )
                        if success:
                            local_iter_num = 0
                            running_mfu = -1.0
                            just_recovered = True
                            t0 = time.time()
                            continue
                        else:
                            break

                # Apply masking for unmasking training (per-sample processing)
                if training_ctx.training_type == 'unmasking' and mask.any():
                    # Step 1: Get per-sample losses without aggregation (replaces reduction='mean')
                    per_sample_losses, per_sample_mask_counts = calculate_per_sample_losses(logits, Y, mask)

                    # Step 2: Apply mask ratio weighting if enabled (per-sample)
                    if training_ctx.weight_loss_by_mask_ratio:
                        mask_ratios = mask.float().mean(dim=1)  # (batch_size,) - ratio per sample
                        valid_ratios = mask_ratios > 0
                        weights = torch.ones_like(mask_ratios)
                        weights[valid_ratios] = (1.0 / mask_ratios[valid_ratios]) ** 0.5
                        per_sample_losses = per_sample_losses * weights

                    # Step 3: Get external wrongness factor (replace with your actual implementation)
                    wrongness_factor = getattr(training_ctx, 'wrongness_factor', None)
                    if wrongness_factor is None:
                        wrongness_factor = torch.ones(training_ctx.batch_size, device=logits.device)

                    # Step 4: Apply per-sample modifications (entropy penalty + wrongness factor)
                    # This moves entropy penalty calculation BEFORE aggregation
                    with timer.time_function('loss_processing'):
                        modified_per_sample_losses = apply_per_sample_modifications(
                            per_sample_losses, logits, Y, mask, training_ctx, iter_num, wrongness_factor
                        )

                    # Step 5: Final aggregation to scalar loss (replaces the original reduction='mean')
                    valid_samples = per_sample_mask_counts > 0
                    if valid_samples.any():
                        loss = modified_per_sample_losses[valid_samples].mean()
                    else:
                        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

                else:
                    if training_ctx.training_type == 'unmasking':
                        loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

                # Final loss stability check
                if not instability_detector.check_loss_stability(loss, iter_num, "final"):
                    success, X, Y, mask = instability_detector.attempt_recovery(
                        model, optimizer, training_ctx, scaler, get_batch
                    )
                    if success:
                        local_iter_num = 0
                        running_mfu = -1.0
                        just_recovered = True
                        t0 = time.time()
                        continue
                    else:
                        break

                loss = loss / gradient_accumulation_steps

            # Get next batch while model is doing forward pass
            with timer.time_function('data_generation'):
                training_ctx.iter_num = iter_num
                X, Y, mask = get_batch('train', training_ctx)
                # Store for logging
                training_ctx.current_batch_X = X
                training_ctx.current_batch_mask = mask

            # Backward pass
            with timer.time_function('backward_pass'):
                scaler.scale(loss).backward()

    # Gradient processing and clipping
    with timer.time_function('gradient_processing'):
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if not instability_detector.check_gradient_stability(model, grad_norm, iter_num, grad_clip):
                success, X, Y, mask = instability_detector.attempt_recovery(
                    model, optimizer, training_ctx, scaler, get_batch
                )
                if success:
                    local_iter_num = 0
                    running_mfu = -1.0
                    just_recovered = True
                    t0 = time.time()
                    continue
                else:
                    break
        else:
            if not instability_detector.check_gradient_stability(model, None, iter_num, grad_clip):
                success, X, Y, mask = instability_detector.attempt_recovery(
                    model, optimizer, training_ctx, scaler, get_batch
                )
                if success:
                    local_iter_num = 0
                    running_mfu = -1.0
                    just_recovered = True
                    t0 = time.time()
                    continue
                else:
                    break

    # Optimizer operations
    with timer.time_function('optimizer_operations'):
        scaler.step(optimizer)
        scaler.update()

    # Parameter stability check
    with timer.time_function('parameter_stability_check'):
        if not instability_detector.check_parameter_stability(model, iter_num):
            success, X, Y, mask = instability_detector.attempt_recovery(
                model, optimizer, training_ctx, scaler, get_batch
            )
            if success:
                local_iter_num = 0
                running_mfu = -1.0
                just_recovered = True
                t0 = time.time()
                continue
            else:
                break

    # Cleanup operations
    with timer.time_function('cleanup_operations'):
        optimizer.zero_grad(set_to_none=True)

    # Timing and logging (like original train.py)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        with timer.time_function('gpu_synchronization'):
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

        # Log iteration stats
        logger.log_iteration_stats(iter_num, lossf, dt, running_mfu, timer, training_ctx, Y, model, eval_only)

        # Log validation timing
        logger.log_validation_timing(timer, iter_num)

        # Log mask encounter statistics
        if iter_num % log_interval == 0:
            raw_model.log_mask_encounter_stats("  ")
            raw_model.reset_mask_encounter_stats()

        # Log masking statistics
        logger.log_masking_statistics(iter_num, mask, training_ctx)

    iter_num += 1
    local_iter_num += 1
    just_recovered = False

    # Termination condition
    if iter_num > max_iters:
        break

# -----------------------------------------------------------------------------
# Cleanup (like original train.py)

if ddp:
    destroy_process_group()

stop_prefetch()

logger.print_and_flush("Training completed successfully!")
