#!/usr/bin/env python3
"""
Refactored nanoGPT training script with modular architecture.

This script trains a GPT model using the extracted training modules for better organization and maintainability.
"""
import os
import time
import math
import pickle
import torch
import numpy as np
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import model and core components
from model import GPTConfig, GPT
from logger import TrainingLogger
from analyzer import ModelAnalyzer

# Import training modules
from training.config import TrainingConfig, load_scaling_schedule
from training.utils import TimingProfiler, BatchManager, setup_signal_handlers, get_system_info, estimate_loss
from training.evaluation import get_val_batch, run_full_analysis_async, analysis_done_callback
from training.scheduler import TrainingScheduler, LearningRateScheduler, EarlyStoppingMonitor
from training.operations import log_detailed_params, log_model_architecture, calculate_and_log_target_architecture
from training.resume import (
    find_checkpoint_path, load_checkpoint_with_fallback, apply_model_parameter_overrides,
    apply_smart_state_dict_loading, transfer_optimizer_state, transfer_optimizer_state_by_shape,
    load_training_state, restore_scaling_schedule_state, apply_training_parameter_overrides
)

# Initialize global termination flag
should_terminate = False

def main():
    global should_terminate
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Load configuration
    config = TrainingConfig()
    
    # Override from configurator.py if it exists
    if os.path.exists('configurator.py'):
        config_globals = {}
        exec(open('configurator.py').read(), config_globals)
        config.update_from_dict({k: v for k, v in config_globals.items() 
                               if hasattr(config, k) and not k.startswith('_')})
    
    # Validate configuration
    config.validate()
    
    # Setup distributed training
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        print(f"DDP enabled: rank {ddp_rank}, local_rank {ddp_local_rank}, world_size {ddp_world_size}")
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = config.device
    
    # Set device and data type
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Initialize logging
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        training_logger = TrainingLogger(
            log_dir=config.log_dir,
            enabled=config.file_logging
        )
        training_logger.setup(config.to_dict())
        
        print("Configuration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        
        system_info = get_system_info()
        print(f"\nSystem Info:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
    else:
        training_logger = None
    
    # Load scaling schedule
    if config.scaling_schedule_file:
        config.scaling_schedule = load_scaling_schedule(config.scaling_schedule_file, config.init_from)
    
    # Setup data loading
    data_dir = os.path.join('data', config.dataset)
    vocab_remapping = None
    if config.shrunken_vocab_size is not None:
        vocab_remapping = torch.load(config.vocab_remapping_file)
    
    # Initialize batch manager
    batch_manager = BatchManager(
        train_shard_filenames=config.train_shard_filenames,
        num_train_shards=config.num_train_shards,
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=device,
        shrunken_vocab_size=config.shrunken_vocab_size,
        vocab_remapping=vocab_remapping,
        rare_token_id=config.RARE_TOKEN_ID
    )
    
    def get_val_batch_fn():
        return get_val_batch(
            data_dir=data_dir,
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=device,
            shrunken_vocab_size=config.shrunken_vocab_size,
            vocab_remapping=vocab_remapping,
            rare_token_id=config.RARE_TOKEN_ID
        )
    
    # Initialize model
    model_args = config.get_model_args()
    
    if config.init_from == 'scratch':
        print("Initializing a new model from scratch")
        # Set vocabulary size
        if config.shrunken_vocab_size is not None:
            model_args['vocab_size'] = config.shrunken_vocab_size
        else:
            meta_path = os.path.join(data_dir, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                model_args['vocab_size'] = meta['vocab_size']
            else:
                model_args['vocab_size'] = 50304  # Default
        
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = float('inf')
        
    elif config.init_from == 'resume':
        print(f"Resuming training from {config.out_dir}")

        # Find the best available checkpoint (including emergency checkpoints)
        ckpt_path = find_checkpoint_path(config.out_dir)

        # Load checkpoint with fallback to emergency checkpoints
        checkpoint = load_checkpoint_with_fallback(ckpt_path, device, config.out_dir)

        # Apply model parameter overrides from current config
        checkpoint_model_args = checkpoint['model_args']
        checkpoint_model_args, config_changes = apply_model_parameter_overrides(
            checkpoint_model_args, config
        )

        if config_changes:
            print(f"Applied {len(config_changes)} hyperparameter overrides:")
            for change in config_changes:
                print(f"  - {change}")

            # Log the overrides for tracking
            if master_process and training_logger:
                training_logger.log(f"Resume with hyperparameter overrides: {'; '.join(config_changes)}")
        else:
            print("No hyperparameter overrides detected.")

        # Create the model with the potentially new configuration
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)

        # Apply smart state dict loading with LoRA compatibility
        apply_smart_state_dict_loading(model, checkpoint['model'])

        # Load training state
        iter_num, best_val_loss, execution_state = load_training_state(checkpoint)

        # Restore scaling schedule state from checkpoint if available
        if config.scaling_schedule_file:
            restored_schedule = restore_scaling_schedule_state(checkpoint, config.scaling_schedule_file)
            if restored_schedule:
                config.scaling_schedule = restored_schedule

        # Apply training parameter overrides
        training_param_changes = apply_training_parameter_overrides(checkpoint, config)
        if training_param_changes and master_process and training_logger:
            training_logger.log(f"Resume with training parameter overrides: {'; '.join(training_param_changes)}")
        
    elif config.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {config.init_from}")
        model = GPT.from_pretrained(config.init_from, dict(dropout=config.dropout))
        iter_num = 0
        best_val_loss = float('inf')
    
    # Move model to device
    model.to(device)
    
    # Initialize schedulers and monitors
    training_scheduler = None
    if config.scaling_schedule:
        training_scheduler = TrainingScheduler(config.scaling_schedule, training_logger)
        if master_process:
            target_config = calculate_and_log_target_architecture(model.config, config.scaling_schedule)
            config.target_architecture_config = target_config
    
    lr_scheduler = LearningRateScheduler(
        learning_rate=config.learning_rate,
        warmup_iters=config.warmup_iters,
        lr_decay_iters=config.lr_decay_iters,
        min_lr=config.min_lr,
        decay_lr=config.decay_lr
    )
    
    # Setup optimizer
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate,
                                         (config.beta1, config.beta2), device_type)

    # Load optimizer state if resuming
    if config.init_from == 'resume':
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

        # Free up memory
        checkpoint = None
    
    # Initialize model analyzer
    analyzer = None
    if master_process:
        analyzer = ModelAnalyzer()
    
    # Compile model if requested
    unoptimized_model = model
    if config.compile:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Initialize timing profiler
    timing_profiler = TimingProfiler()
    
    # Log initial model architecture
    if master_process:
        log_model_architecture(model.module if ddp else model, iter_num, is_initial=True)
        log_detailed_params(model.module if ddp else model)
    
    # Training loop
    print(f"Starting training from iteration {iter_num}")
    
    X, Y = batch_manager.get_batch()  # Fetch the first batch
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while iter_num < config.max_iters and not should_terminate:
        
        # Determine current learning rate
        lr = lr_scheduler.get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate model periodically
        if iter_num % config.eval_interval == 0 and master_process:
            with timing_profiler.time_section("evaluation"):
                val_loss = estimate_loss(model, get_val_batch_fn, config.eval_iters, device_type, config.dtype)
                print(f"step {iter_num}: train loss {val_loss:.4f}")
                
                if training_logger:
                    training_logger.log_metrics(iter_num, {'val_loss': val_loss, 'lr': lr, 'mfu': running_mfu})
                
                # Save checkpoint if this is the best model
                if val_loss < best_val_loss or config.always_save_checkpoint:
                    best_val_loss = val_loss
                    if iter_num > 0:
                        checkpoint = {
                            'model': (model.module if ddp else model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': (model.module if ddp else model).config.__dict__,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config.to_dict()
                        }
                        print(f"saving checkpoint to {config.out_dir}")
                        torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        
        # Execute scheduled operations
        if training_scheduler and master_process:
            with timing_profiler.time_section("operations"):
                val_loss = estimate_loss(model, get_val_batch_fn, config.eval_iters, device_type, config.dtype)
                
                model, optimizer, hyperparameter_updates = training_scheduler.check_and_execute_operations(
                    iter_num=iter_num,
                    current_val_loss=val_loss,
                    model=model,
                    optimizer=optimizer,
                    compile_enabled=config.compile,
                    ddp_enabled=ddp,
                    ddp_local_rank=ddp_local_rank if ddp else 0,
                    master_process=master_process,
                    data_dir=data_dir,
                    weight_decay=config.weight_decay,
                    learning_rate=config.learning_rate,
                    beta1=config.beta1,
                    beta2=config.beta2,
                    device_type=device_type,
                    target_architecture_config=config.target_architecture_config
                )
                
                # Apply hyperparameter updates
                if hyperparameter_updates:
                    if 'set_lr' in hyperparameter_updates:
                        config.learning_rate = hyperparameter_updates['set_lr']
                        lr_scheduler.update_params(learning_rate=config.learning_rate)
                    if 'set_batch_size' in hyperparameter_updates:
                        config.batch_size = hyperparameter_updates['set_batch_size']
                        # Update batch manager
                        batch_manager.batch_size = config.batch_size
                    if 'reset_lr_schedule' in hyperparameter_updates:
                        lr_scheduler.reset_schedule(iter_num)
        
        # Forward and backward pass
        with timing_profiler.time_section("forward_backward"):
            for micro_step in range(config.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
                
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / config.gradient_accumulation_steps
                
                # Get next batch asynchronously
                X, Y = batch_manager.get_batch()
                
                # Backward pass
                loss.backward()
        
        # Gradient clipping
        if config.grad_clip != 0.0:
            with timing_profiler.time_section("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # Optimizer step
        with timing_profiler.time_section("optimizer_step"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:  # Let MFU calculation stabilize
                mfu = (model.module if ddp else model).estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
            if training_logger:
                metrics = {
                    'train_loss': lossf,
                    'lr': lr,
                    'dt': dt,
                    'mfu': running_mfu,
                    'iter_num': iter_num
                }
                training_logger.log_metrics(iter_num, metrics)
        
        iter_num += 1
        local_iter_num += 1
        
        # Print timing summary periodically
        if iter_num % (config.eval_interval * 5) == 0 and master_process:
            print(timing_profiler.get_summary())
    
    # Final cleanup
    if ddp:
        destroy_process_group()
    
    print("Training completed!")
    
    if master_process and training_scheduler:
        progress = training_scheduler.get_progress_summary()
        print(f"Operations completed: {progress['completed_operations']}/{progress['total_operations']}")


if __name__ == '__main__':
    main()