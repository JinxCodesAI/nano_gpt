"""
Trainer orchestrator that coordinates the main training loop.

Extracts the control flow from train.py without changing functionality or
logging/timing behavior.
"""

import time
from typing import Optional

import torch


class Trainer:
    def __init__(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler,
        evaluator,
        logger,
        training_step,
        checkpoint_manager,
        consumer,
        device: str,
        ddp: bool,
        master_process: bool,
        eval_interval: int,
        log_interval: int,
        max_iters: int,
        always_save_checkpoint: bool,
        eval_only: bool,
        batch_size: int,
        gradient_accumulation_steps: int,
        iter_num: int = 0,
        best_val_loss: float = 1e9,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.logger = logger
        self.training_step = training_step
        self.checkpoint_manager = checkpoint_manager
        self.consumer = consumer
        self.device = device
        self.ddp = bool(ddp)
        self.master_process = bool(master_process)
        self.eval_interval = int(eval_interval)
        self.log_interval = int(log_interval)
        self.max_iters = int(max_iters)
        self.always_save_checkpoint = bool(always_save_checkpoint)
        self.eval_only = bool(eval_only)
        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(gradient_accumulation_steps)
        self.iter_num = int(iter_num)
        self.best_val_loss = float(best_val_loss)



    def train(self) -> None:
        # Initial progress update (parity with train.py)
        self.checkpoint_manager.update_progress(
            iter_num=self.iter_num, best_val_loss=self.best_val_loss
        )

        # fetch the very first batch
        batch = self.consumer.get_batch('train', self.device)
        t0 = time.time()
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        while True:
            # determine and set the learning rate for this iteration (via TrainingStep)
            lr = self.training_step.set_learning_rate(self.iter_num)

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.eval_interval == 0 and self.master_process:
                # expose current iteration to the model for critic alpha warmup during eval
                raw_model = self.model.module if self.ddp else self.model
                try:
                    setattr(raw_model, '_current_iter', self.iter_num)
                except Exception:
                    pass
                losses = self.evaluator.evaluate()
                # Reset timer after validation to exclude validation time from MFU calculation
                t0 = time.time()

                # Prepare evaluation metrics for logging
                eval_metrics = {
                    "iter": self.iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu_pct": running_mfu * 100,
                }
                # Pass-through zero-target validation stats if provided by evaluator
                if 'val/zero_mean' in losses:
                    eval_metrics['val/zero_mean'] = losses['val/zero_mean']
                if 'val/zero_p90' in losses:
                    eval_metrics['val/zero_p90'] = losses['val/zero_p90']
                # Include any additional scalar validation-only stats for console printing
                for k, v in losses.items():
                    if k not in ('train', 'val') and isinstance(v, (int, float)):
                        eval_metrics[k] = v
                # Log evaluation results
                self.logger.log_eval(eval_metrics)

                if losses['val'] < self.best_val_loss or self.always_save_checkpoint:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        self.checkpoint_manager.update_progress(
                            iter_num=self.iter_num, best_val_loss=self.best_val_loss
                        )
                        ckpt_path = self.checkpoint_manager.save()
                        self.logger.log_checkpoint(f"saving checkpoint to {ckpt_path}")

            if self.iter_num == 0 and self.eval_only:
                break



            # Dynamic unfreezing support (delegated to TrainingStep)
            self.training_step.maybe_unfreeze(self.iter_num)

            # forward/backward/update handled by TrainingStep
            # Expose current iteration to loss modifiers
            try:
                self.evaluator.loss_modifier_pipeline.current_iter = self.iter_num
            except Exception:
                pass
            # Also expose current iter to model for critic alpha warmup during training
            try:
                setattr(raw_model, '_current_iter', self.iter_num)
            except Exception:
                pass
            loss, batch = self.training_step.execute_step(
                batch, self.evaluator.loss_modifier_pipeline, self.consumer, self.device
            )

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            if self.iter_num % self.log_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.grad_accum_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.batch_size * self.grad_accum_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                # Prepare step metrics for logging
                step_metrics = {
                    "iter": self.iter_num,
                    "loss": lossf,
                    "time_ms": dt * 1000,
                    "mfu_pct": running_mfu * 100,
                }
                # If TrainingStep captured component losses, include them (already unscaled like lossf)
                try:
                    main_part = getattr(self.training_step, 'last_loss_main', None)
                    critic_part = getattr(self.training_step, 'last_loss_critic', None)
                    if isinstance(main_part, (int, float)):
                        step_metrics['loss_main'] = float(main_part)
                    if isinstance(critic_part, (int, float)):
                        step_metrics['loss_critic'] = float(critic_part)
                except Exception:
                    pass
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

            self.iter_num += 1
            self.checkpoint_manager.update_progress(iter_num=self.iter_num, best_val_loss=self.best_val_loss)
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break
            t0 = time.time()

