"""
Training step handler encapsulating the micro-step loop, gradient scaling,
DDP sync control, gradient clipping, and optimizer stepping.

This class extracts the core forward/backward update logic from train.py
without changing functionality or timing semantics (including prefetching
next batches inside the micro-step loop).
"""

from typing import Tuple
import torch


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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.grad_accum_steps = int(gradient_accumulation_steps)
        self.grad_clip = float(grad_clip)
        self.ddp = bool(ddp)
        self.ctx = ctx

    def execute_step(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        loss_modifiers,
        consumer,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute one full optimizer step with gradient accumulation.

        Args:
            X, Y: Current input and targets tensors for the first micro-step
            loss_modifiers: Loss modifier pipeline to pass to the model
            consumer: DatasetConsumer to prefetch the next batch
            device: Device string for consumer.get_batch

        Returns:
            (loss, X, Y):
                - loss: the (scaled by 1/grad_accum_steps) loss from the last micro-step
                - X, Y: next prefetched batch, ready for the next iteration
        """
        last_loss = None

        for micro_step in range(self.grad_accum_steps):
            if self.ddp:
                # Only sync gradients at the last micro-step, identical to original code
                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )
            with self.ctx:
                _, loss = self.model(X, Y, loss_modifiers=loss_modifiers)
                # Scale the loss to account for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Immediately async prefetch next batch while GPU computes backward
            X, Y = consumer.get_batch('train', device)

            # Backward pass with gradient scaling for fp16
            self.scaler.scale(loss).backward()
            last_loss = loss

        # Gradient clipping (after unscale), identical to original code
        if self.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step & scaler update
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # Flush the gradients as soon as we can
        self.optimizer.zero_grad(set_to_none=True)

        return last_loss, X, Y

