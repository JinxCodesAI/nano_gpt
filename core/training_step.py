"""
Training step handler encapsulating the micro-step loop, gradient scaling,
DDP sync control, gradient clipping, and optimizer stepping.

This class extracts the core forward/backward update logic from train.py
without changing functionality or timing semantics (including prefetching
next batches inside the micro-step loop).
"""

from typing import Tuple, Optional, Any
from core.batch import Batch, unpack_batch
from core.guidance import prepare_guidance
import torch
from torch.nn import functional as F
from model import ModelMode


class TrainingStep:
    """
    Encapsulates one training iteration consisting of multiple micro-steps for
    gradient accumulation, with identical behavior to the original train.py loop.
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        gradient_accumulation_steps: int,
        grad_clip: float,
        ddp: bool,
        ctx,
        *,
        scheduler=None,
        unfreeze_at_iteration: Optional[int] = None,
        unfreeze_lr_multiplier: float = 0.1,
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.grad_accum_steps = int(gradient_accumulation_steps)
        self.grad_clip = float(grad_clip)
        self.ddp = bool(ddp)
        self.ctx = ctx
        # extended responsibilities
        self.scheduler = scheduler
        self.unfreeze_at_iteration = unfreeze_at_iteration
        self.unfreeze_lr_multiplier = float(unfreeze_lr_multiplier)
        self.logger = logger
        # Track last loss components (unscaled, sum equals total loss before grad_accum scaling)
        self.last_loss_main: float = 0.0
        self.last_loss_critic: float = 0.0

    def set_learning_rate(self, iter_num: int) -> float:
        """
        Set optimizer learning rate for this iteration using the attached scheduler.
        Returns the LR that was set. No-op if scheduler is None.
        """
        if self.scheduler is None:
            return next((pg.get('lr', 0.0) for pg in self.optimizer.param_groups), 0.0)
        lr = self.scheduler.get_lr(iter_num)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def maybe_unfreeze(self, iter_num: int) -> None:
        """
        Perform dynamic unfreezing at the configured iteration, extend optimizer
        with newly-trainable params, and reduce LR by the configured multiplier.
        Preserves original ordering: this should be called AFTER evaluation at a given step.
        """
        if self.unfreeze_at_iteration is None:
            return
        if iter_num != self.unfreeze_at_iteration:
            return

        raw_model = self.model.module if self.ddp else self.model
        try:
            if hasattr(raw_model, 'get_frozen_status') and raw_model.get_frozen_status():
                if self.logger:
                    self.logger.log_info(f"Unfreezing transformer at iteration {iter_num}")
                # Unfreeze and extend optimizer
                raw_model.unfreeze_transformer_weights()
                raw_model.extend_optimizer_with_unfrozen(self.optimizer, weight_decay=None, learning_rate=None)
                # Adjust learning rate for stability across all param groups
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.unfreeze_lr_multiplier
                if self.logger:
                    # Use last param_group lr for message (same across groups after scaling)
                    self.logger.log_info(
                        f"Reduced learning rate by factor {self.unfreeze_lr_multiplier} current lr: {param_group['lr']}"
                    )
        except Exception as e:
            if self.logger:
                self.logger.log_info(f"Warning: dynamic unfreeze encountered an error: {e}")

    def execute_step(
        self,
        batch: Batch,
        loss_modifiers,
        consumer,
        device: str,
    ) -> Tuple[torch.Tensor, Batch]:
        """
        Execute one full optimizer step with gradient accumulation.

        Args:
            batch: Current batch dict. Supports keys x/y[/attention_mask] or input_ids/targets
            loss_modifiers: Loss modifier pipeline to pass to the model
            consumer: DatasetConsumer to prefetch the next batch
            device: Device string for consumer.get_batch

        Returns:
            (loss, next_batch):
                - loss: the (scaled by 1/grad_accum_steps) loss from the last micro-step
                - next_batch: next prefetched batch (dict), ready for the next iteration
        """
        last_loss = None


        for micro_step in range(self.grad_accum_steps):
            if self.ddp:
                # Only sync gradients at the last micro-step, identical to original code
                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )

            # Switch model mode based on batch metadata (if provided)
            raw_model = self.model.module if self.ddp else self.model
            if '_model_mode' in batch:
                mode_str = batch['_model_mode']
                # Convert string to ModelMode enum
                if mode_str == 'language_model' or mode_str == ModelMode.LANGUAGE_MODEL:
                    raw_model.set_mode(ModelMode.LANGUAGE_MODEL)
                elif mode_str == 'sequence_scorer' or mode_str == ModelMode.SEQUENCE_SCORER:
                    raw_model.set_mode(ModelMode.SEQUENCE_SCORER)
                # If mode_str is already a ModelMode enum, use it directly
                elif isinstance(mode_str, ModelMode):
                    raw_model.set_mode(mode_str)
            else:
                print(f"[DEBUG] No _model_mode in batch, current model mode: {raw_model.get_mode()}")

            X, Y = unpack_batch(batch)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(X.device)

            guidance_h, guidance_mask = prepare_guidance(
                raw_model,
                batch,
                X,
                attention_mask,
            )

            with self.ctx:
                # Delegate to model forward; model handles critic internally when enabled
                # Provide current iteration to model for effective critic alpha schedule
                try:
                    setattr(raw_model, "_current_iter", int(getattr(loss_modifiers, "current_iter", 0) if loss_modifiers is not None else 0))
                except Exception:
                    pass

                _logits, loss = self.model(
                    X,
                    Y,
                    attention_mask=attention_mask,
                    loss_modifiers=loss_modifiers,
                    guidance_h=guidance_h,
                    guidance_mask=guidance_mask,
                )

                # Read loss components exposed by model (if available)
                try:
                    self.last_loss_main = float(getattr(raw_model, "_last_lm_loss"))
                except Exception:
                    try:
                        self.last_loss_main = float(loss.detach().item())
                    except Exception:
                        self.last_loss_main = 0.0
                try:
                    self.last_loss_critic = float(getattr(raw_model, "_last_critic_loss"))
                except Exception:
                    self.last_loss_critic = 0.0

                # Scale the loss to account for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Immediately async prefetch next batch while GPU computes backward
            next_batch = consumer.get_batch('train', device)

            # Backward pass with gradient scaling for fp16
            self.scaler.scale(loss).backward()
            last_loss = loss
            batch = next_batch

        # Gradient clipping (after unscale), identical to original code
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step & scaler update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Flush the gradients as soon as we can
        self.optimizer.zero_grad(set_to_none=True)

        return last_loss, batch

