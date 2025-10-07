"""
GRPO Trainer: Orchestrates the GRPO training loop.

This module provides the GRPOTrainer class that manages:
- Main training loop
- Learning rate scheduling
- Logging and metrics
- Checkpointing
- Periodic sampling for quality monitoring
"""

import time
from typing import Optional, Dict, Any

import torch


class GRPOTrainer:
    """
    Orchestrates the GRPO training loop.
    
    Similar to core.trainer.Trainer but adapted for GRPO-specific needs:
    - No evaluation loop (or simplified)
    - GRPO-specific metrics logging
    - Periodic sampling to monitor generation quality
    """
    
    def __init__(
        self,
        *,
        generator_model,
        reference_model,
        judge_model,
        optimizer: torch.optim.Optimizer,
        scheduler,
        training_step,
        consumer,
        checkpoint_manager,
        logger,
        device: str,
        ddp: bool,
        master_process: bool,
        log_interval: int,
        save_interval: int,
        sample_interval: int,
        max_iters: int,
        batch_size: int,
        iter_num: int = 0,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            generator_model: Model being trained
            reference_model: Frozen reference model
            judge_model: Frozen judge model
            optimizer: Optimizer for generator
            scheduler: Learning rate scheduler
            training_step: GRPOTrainingStep instance
            consumer: DatasetConsumer for batches
            checkpoint_manager: CheckpointManager for saving
            logger: Logger instance
            device: Device string
            ddp: Whether using DDP
            master_process: Whether this is the master process
            log_interval: Log every N iterations
            save_interval: Save checkpoint every N iterations
            sample_interval: Sample generations every N iterations
            max_iters: Maximum training iterations
            batch_size: Batch size
            iter_num: Starting iteration number
        """
        self.generator = generator_model
        self.reference = reference_model
        self.judge = judge_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_step = training_step
        self.consumer = consumer
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.device = device
        self.ddp = bool(ddp)
        self.master_process = bool(master_process)
        self.log_interval = int(log_interval)
        self.save_interval = int(save_interval)
        self.sample_interval = int(sample_interval)
        self.max_iters = int(max_iters)
        self.batch_size = int(batch_size)
        self.iter_num = int(iter_num)
        
    def train(self) -> None:
        """Main GRPO training loop."""
        # Fetch the first batch
        batch = self.consumer.get_batch('train', self.device)
        t0 = time.time()
        local_iter_num = 0
        raw_model = self.generator.module if self.ddp else self.generator
        running_mfu = -1.0
        
        self.logger.log_info("Starting GRPO training loop")
        
        while True:
            # Set learning rate for this iteration
            lr = self.scheduler.get_lr(self.iter_num)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            
            # Execute GRPO training step
            loss, batch, metrics = self.training_step.execute_step(
                batch, self.consumer, self.device
            )
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            
            if self.iter_num % self.log_interval == 0 and self.master_process:
                # Estimate MFU
                if local_iter_num >= 5:  # Let training settle
                    mfu = raw_model.estimate_mfu(self.batch_size, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                
                # Prepare step metrics
                step_metrics = {
                    "iter": self.iter_num,
                    "lr": lr,
                    "time_ms": dt * 1000,
                    "mfu_pct": running_mfu * 100,
                    **metrics  # Include all GRPO metrics
                }
                
                # CUDA memory metrics (optional)
                try:
                    if torch.cuda.is_available():
                        alloc = torch.cuda.memory_allocated() / (1024**2)
                        reserved = torch.cuda.memory_reserved() / (1024**2)
                        max_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                        step_metrics['mem_alloc_mb'] = float(alloc)
                        step_metrics['mem_reserved_mb'] = float(reserved)
                        step_metrics['mem_max_alloc_mb'] = float(max_alloc)
                except Exception:
                    pass
                
                self.logger.log_step(step_metrics)
            
            # Checkpointing
            if self.iter_num % self.save_interval == 0 and self.iter_num > 0 and self.master_process:
                self.checkpoint_manager.update_progress(
                    iter_num=self.iter_num,
                    best_val_loss=0.0  # Not used in GRPO
                )
                ckpt_path = self.checkpoint_manager.save()
                self.logger.log_checkpoint(f"Saved checkpoint to {ckpt_path}")
            
            # Periodic sampling (optional, for monitoring quality)
            if self.sample_interval > 0 and self.iter_num % self.sample_interval == 0 and self.master_process:
                self._sample_and_log(self.iter_num)
            
            # Increment iteration
            self.iter_num += 1
            local_iter_num += 1
            
            # Termination condition
            if self.iter_num > self.max_iters:
                break
            
            t0 = time.time()
        
        self.logger.log_info(f"GRPO training completed after {self.iter_num} iterations")
    
    def _sample_and_log(self, iter_num: int) -> None:
        """
        Generate sample completions and log them for quality monitoring.

        Args:
            iter_num: Current iteration number
        """
        import torch
        from sample_utils import predict_and_sample_tokens, calculate_judge_scores
        from core.batch import unpack_batch

        self.generator.eval()

        try:
            # Get a sample batch
            sample_batch = self.consumer.get_batch('val', self.device)
            X, Y = unpack_batch(sample_batch)

            # Limit to first few samples for logging
            num_samples = min(4, X.shape[0])
            X_sample = X[:num_samples]

            # Generate completions
            with torch.no_grad():
                completions, _ = predict_and_sample_tokens(
                    model=self.generator,
                    tokens=X_sample,
                    mask_token_id=self.training_step.mask_token_id,
                    temperature=self.training_step.temperature,
                    top_p=self.training_step.top_p,
                    vocab_size=self.training_step.vocab_size,
                    device=self.device,
                    verbose=False,
                    return_logits=False,
                    pad_token_id=self.training_step.pad_token_id,
                    base_vocab_size=self.training_step.base_vocab_size
                )

                # Score with judge
                scores = calculate_judge_scores(
                    judge_model=self.judge,
                    tokens=completions,
                    device=self.device,
                    ctx=self.training_step.ctx
                )

            # Log samples
            self.logger.log_info(f"[Iter {iter_num}] Sample scores: {scores.tolist()}")

        except Exception as e:
            self.logger.log_info(f"[Iter {iter_num}] Sampling failed: {e}")

        finally:
            self.generator.train()

