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
import concurrent.futures
import psutil
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Conditional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import model and core components
from model import GPTConfig, GPT
from logger import TrainingLogger
from analyzer import ModelAnalyzer

# Import training modules
from training.config import TrainingConfig, load_scaling_schedule
from training.utils import TimingProfiler, BatchManager, setup_signal_handlers, get_system_info, estimate_loss, get_vram_usage
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

        # File logging setup
        training_logger = TrainingLogger(
            log_dir=config.log_dir,
            file_enabled=config.file_logging
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

    # Initialize Wandb logging (separate from file logging)
    final_wandb_run_name = None
    if config.wandb_log and master_process:
        if not WANDB_AVAILABLE:
            print("Warning: wandb_log=True but wandb is not installed. Wandb logging disabled.")
            print("Install wandb with: pip install wandb")
        else:
            # Try to restore wandb run name from checkpoint first
            if config.init_from == 'resume':
                ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
                if os.path.exists(ckpt_path):
                    try:
                        checkpoint_temp = torch.load(ckpt_path, map_location='cpu')
                        if 'final_wandb_run_name' in checkpoint_temp:
                            final_wandb_run_name = checkpoint_temp['final_wandb_run_name']
                            print(f"Restored W&B run name: {final_wandb_run_name}")
                        del checkpoint_temp  # Free memory
                    except Exception as e:
                        print(f"Could not restore wandb run name: {e}")

            # Create new run name if not restored
            if final_wandb_run_name is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_wandb_run_name = f"{config.wandb_run_name}_{timestamp}"
                print(f"Created new W&B run name: {final_wandb_run_name}")

            wandb.init(project=config.wandb_project, name=final_wandb_run_name, config=config.to_dict())
    
    # Load scaling schedule
    if config.scaling_schedule_file:
        config.scaling_schedule = load_scaling_schedule(config.scaling_schedule_file, config.init_from)
    
    # Setup data loading
    data_dir = os.path.join('data', config.dataset)
    vocab_remapping = None
    if config.shrunken_vocab_size is not None:
        vocab_remapping = torch.load(config.vocab_remapping_file)
    
    # Initialize high-performance batch manager
    batch_manager = BatchManager(
        train_shard_filenames=config.train_shard_filenames,
        num_train_shards=config.num_train_shards,
        data_dir=data_dir,
        block_size=config.block_size,
        batch_size=config.batch_size,
        device=device,
        shrunken_vocab_size=config.shrunken_vocab_size,
        vocab_remapping=vocab_remapping,
        rare_token_id=config.RARE_TOKEN_ID,
        starting_estimation_token_count=100_000_000,
        buffer_size=2000
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
        analyzer = ModelAnalyzer(model)
    
    # Compile model if requested
    unoptimized_model = model
    if config.compile:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        config.gradient_accumulation_steps //= ddp_world_size
    
    # Initialize timing profiler
    timing_profiler = TimingProfiler()
    
    # Initialize thread pool executor for asynchronous analysis
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    prev_embeddings = None  # This will store the CPU snapshot for semantic drift
    active_futures = []  # Track active analysis futures for cleanup
    
    # Log initial model architecture
    if master_process:
        log_model_architecture(model.module if ddp else model, iter_num, is_initial=True)
        log_detailed_params(model.module if ddp else model)
    
        tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    # Training loop
    print(f"Starting training from iteration {iter_num}")
    
    X, Y = batch_manager.get_next_batch()  # Fetch the first batch
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    eval_start_time = time.time()
    
    # MFU stats logging setup
    if master_process and config.file_logging:
        mfu_log_path = os.path.join(config.log_dir, 'mfu_stats.txt')
        with open(mfu_log_path, 'w') as f:
            f.write("iter,loss,lr,time_ms,mfu_percent,vram_used_gb,vram_total_gb,vram_percent\n")
    
    # VRAM monitoring function already imported from training.utils
    
    while iter_num < config.max_iters and not should_terminate:
        
        # Determine current learning rate
        lr = lr_scheduler.get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate model periodically
        if iter_num % config.eval_interval == 0 and master_process:
            with timing_profiler.time_section("evaluation"):
                losses = estimate_loss(
                    model,
                    {'train': batch_manager.get_next_batch, 'val': get_val_batch_fn},
                    config.eval_iters, device_type, config.dtype
                )
                
                # Calculate tokens per second using batch_manager's served count
                elapsed_time_seconds = time.time() - eval_start_time if 'eval_start_time' in locals() else time.time() - t0
                tokens_per_second = batch_manager.total_tokens_served / elapsed_time_seconds if elapsed_time_seconds > 0 else 0
                eval_start_time = time.time()  # Reset for next interval
                
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr: {lr:.4f}, tokens/sec {tokens_per_second:.0f}")
                
                # Add timing breakdown output
                timing_breakdown = timing_profiler.get_breakdown_percentages()
                if timing_breakdown:
                    print(f"  timing breakdown: {', '.join([f'{k} {v:.1f}%' for k, v in timing_breakdown.items()])}")
                    # Duplicate print for exact match with original
                    print(f"  timing breakdown: {', '.join([f'{k} {v:.1f}%' for k, v in timing_breakdown.items()])}")

                # File logging  
                if training_logger:
                    training_logger.log_step(iter_num, losses['train'], losses['val'], tokens_per_second)
                    # Log timing breakdown
                    if timing_breakdown:
                        timing_str = ", ".join([f"{k} {v:.1f}%" for k, v in timing_breakdown.items()])
                        training_logger.log(f"  timing breakdown (avg last {config.eval_interval}): {timing_str}")

                # Wandb logging
                if config.wandb_log and WANDB_AVAILABLE:
                    wandb_metrics = {
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu * 100 if running_mfu > 0 else 0
                    }
                    wandb.log(wandb_metrics)
                
                # Asynchronous model analysis
                if analyzer:
                    # Check system memory before dispatching analysis to prevent OOM
                    memory_info = psutil.virtual_memory()
                    if memory_info.percent > 90.0:
                        print(f"WARNING: Skipping async analysis due to high system memory usage ({memory_info.percent:.1f}%)")
                    else:
                        print(f"Dispatching async analysis for iter {iter_num}...")
                        try:
                            # Create a full snapshot of the model state on CPU
                            current_snapshot = analyzer.get_model_state_snapshot()
                            # Get validation batch for analysis
                            X_val, Y_val = get_val_batch_fn()
                            # Get filtered tokens from BatchManager (non-outliers)
                            filtered_tokens = batch_manager.get_non_outlier_tokens(config.ignored_outlayers_sum)
                            print(f"Selected {len(filtered_tokens)} non-outlier tokens out of {len(filtered_tokens) + int(len(filtered_tokens) * config.ignored_outlayers_sum / (1 - config.ignored_outlayers_sum))} total tokens")
                            print(f"These tokens represent {1 - config.ignored_outlayers_sum:.4f} of total served tokens")
                            
                            # Define the analysis task function (matching original train.py approach)
                            def analysis_task():
                                print(f"(Async Analysis @ iter {iter_num}) Starting full model analysis job...")
                                results = analyzer.run_full_analysis(current_snapshot, prev_embeddings, filtered_token_ids=filtered_tokens)
                                entropy = analyzer.analyze_attention_entropy((X_val, Y_val))
                                results['attention_entropy'] = entropy
                                results['iter_num'] = iter_num
                                print(f"Analyzing {len(filtered_tokens)} filtered embeddings out of {len(filtered_tokens) + int(len(filtered_tokens) * config.ignored_outlayers_sum / (1 - config.ignored_outlayers_sum))} total")
                                print("No scaling schedule")  # Match original output
                                print(f"(Async Analysis @ iter {iter_num}) Job finished.")
                                return results
                            
                            # Submit the analysis task to the executor
                            future = executor.submit(analysis_task)
                            
                            # Use a simple callback that matches the original train.py approach
                            def simple_callback(fut):
                                try:
                                    results = fut.result()
                                    iter_num = results['iter_num']
                                    
                                    # Format output to match original train.py exactly
                                    log_messages = []
                                    log_messages.append(f"\n--- ASYNC ANALYSIS RESULTS FOR ITERATION {iter_num} ---")
                                    
                                    # Parse and format geometry results
                                    if 'geometry' in results:
                                        geom = results['geometry']
                                        if 'embeddings' in geom and 'global_sparsity' in geom['embeddings']:
                                            gs = geom['embeddings']['global_sparsity']
                                            log_messages.append(f"  [Embeddings Geometry] Mean Similarity: {gs['mean_similarity']:.4f} Std Similarity: {gs['std_similarity']:.4f} | 10th-90th Percentile: {gs['similarity_10th_percentile']:.4f} - {gs['similarity_90th_percentile']:.4f}")
                                    
                                    log_messages.append("--- END OF ASYNC ANALYSIS RESULTS ---")
                                    
                                    # Print all messages
                                    for msg in log_messages:
                                        print(msg)
                                    
                                    # Log to file if logging is enabled
                                    if master_process and training_logger:
                                        for msg in log_messages:
                                            training_logger.log(msg)
                                    
                                    print()  # Extra newline like original
                                    
                                except Exception as e:
                                    print(f"\n--- ERROR in analysis callback: {e} ---\n")
                                    import traceback
                                    traceback.print_exc()
                                    if master_process and training_logger:
                                        training_logger.log(f"ERROR DURING ASYNC ANALYSIS: {e}")
                                finally:
                                    # Remove completed future from active list
                                    if fut in active_futures:
                                        active_futures.remove(fut)
                            
                            future.add_done_callback(simple_callback)
                            active_futures.append(future)
                            # Update state for the next analysis cycle
                            prev_embeddings = current_snapshot
                            print("Async analysis job dispatched. Training continues.")
                        except Exception as dispatch_error:
                            print(f"ERROR dispatching async analysis for iter {iter_num}: {dispatch_error}")

                # Save checkpoint if this is the best model
                if losses['val'] < best_val_loss or config.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': (model.module if ddp else model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': (model.module if ddp else model).config.__dict__,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config.to_dict(),
                            'final_wandb_run_name': final_wandb_run_name,  # Save wandb run name for continuity
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
                    target_architecture_config=config.target_architecture_config,
                    current_batch_size=config.batch_size
                )
                
                # Apply hyperparameter updates
                if hyperparameter_updates:
                    if 'set_lr' in hyperparameter_updates:
                        config.learning_rate = hyperparameter_updates['set_lr']
                        lr_scheduler.update_params(learning_rate=config.learning_rate)
                        print(f"Updated learning rate to: {config.learning_rate}")
                    if 'set_batch_size' in hyperparameter_updates:
                        config.batch_size = hyperparameter_updates['set_batch_size']
                        # Update batch manager
                        batch_manager.batch_size = config.batch_size
                        print(f"Updated batch size to: {config.batch_size}")
                    if 'reset_lr_schedule' in hyperparameter_updates:
                        lr_scheduler.reset_schedule(iter_num)
                        print(f"Reset learning rate schedule at iteration: {iter_num}")

                    # Log hyperparameter updates to wandb
                    if config.wandb_log and WANDB_AVAILABLE:
                        wandb_updates = {}
                        for key, value in hyperparameter_updates.items():
                            wandb_updates[f"hyperparams/{key}"] = value
                        wandb_updates["iter"] = iter_num
                        wandb.log(wandb_updates)
        
        # Forward and backward pass
        with timing_profiler.time_section("forward_backward"):
            for micro_step in range(config.gradient_accumulation_steps):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)
                
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / config.gradient_accumulation_steps
                
                # Get next batch asynchronously
                X, Y = batch_manager.get_next_batch()
                
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
        
        if iter_num % config.log_interval == 0 and master_process:
            # Calculate accumulated time over log_interval iterations
            dt = t1 - t0
            lossf = loss.item() * config.gradient_accumulation_steps
            # Get VRAM usage
            vram_used, vram_total, vram_percent = get_vram_usage()
            
            if local_iter_num >= 5:  # Let MFU calculation stabilize
                # Fix MFU calculation: dt represents accumulated time over log_interval iterations
                # So dt/log_interval gives average time per iteration, which is what we need
                mfu = (model.module if ddp else model).estimate_mfu(config.batch_size * config.gradient_accumulation_steps, dt/config.log_interval)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu

            # Add the extra iteration info like original
            effective_it = iter_num
            print(f"iter: {iter_num}, effective_it: {effective_it}, warmup_iters: {config.warmup_iters}, lr_decay_iters: {config.lr_decay_iters}, gradient_accumulation_steps:{config.gradient_accumulation_steps}, batch_size:{config.batch_size}")
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.5f}, time {dt/config.log_interval*1000:.2f}ms, mfu {running_mfu*100:.2f}%, VRAM {vram_used:.1f}/{vram_total:.1f}GB ({vram_percent:.1f}%)")

            # File logging
            if training_logger:
                training_logger.log(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:.5f}, time {dt/config.log_interval*1000:.2f}ms, mfu {running_mfu*100:.2f}%, VRAM {vram_used:.1f}/{vram_total:.1f}GB ({vram_percent:.1f}%)")
            
            # MFU stats logging to dedicated file
            if config.file_logging:
                with open(mfu_log_path, 'a') as f:
                    f.write(f"{iter_num},{lossf:.6f},{lr:.8f},{dt/config.log_interval*1000:.2f},{running_mfu*100:.2f},{vram_used:.3f},{vram_total:.3f},{vram_percent:.2f}\n")

            # Wandb logging (separate from file logging)
            if config.wandb_log and WANDB_AVAILABLE:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "time/dt_ms": dt / config.log_interval * 1000,
                    "mfu": running_mfu * 100 if running_mfu > 0 else 0,
                    "vram/used_gb": vram_used,
                    "vram/total_gb": vram_total,
                    "vram/percent": vram_percent
                })
            
            # Reset timer after logging
            t0 = t1
        
        iter_num += 1
        local_iter_num += 1
        
        # Print timing summary periodically
        if iter_num % (config.eval_interval * 5) == 0 and master_process:
            print(timing_profiler.get_summary())
    
    # Final cleanup
    if ddp:
        destroy_process_group()
    
    # Shutdown batch manager
    batch_manager.shutdown()
    
    # Wait for any remaining async analysis jobs and shutdown executor
    if master_process:
        if active_futures:
            print(f"Waiting for {len(active_futures)} remaining analysis jobs to complete...")
            # Wait for all active futures to complete
            concurrent.futures.wait(active_futures, timeout=30)  # 30 second timeout
            print("All analysis jobs completed.")
        
        executor.shutdown(wait=True)
        print("Executor shutdown complete.")
    
    print("Training completed!")
    
    if master_process and training_scheduler:
        progress = training_scheduler.get_progress_summary()
        print(f"Operations completed: {progress['completed_operations']}/{progress['total_operations']}")


if __name__ == '__main__':
    main()