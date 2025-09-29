"""
Training step handler encapsulating the micro-step loop, gradient scaling,
DDP sync control, gradient clipping, and optimizer stepping.

This class extracts the core forward/backward update logic from train.py
without changing functionality or timing semantics (including prefetching
next batches inside the micro-step loop).
"""

from typing import Tuple, Optional, Any
from core.batch import Batch, unpack_batch
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
            X, Y = unpack_batch(batch)
            raw_model = self.model.module if self.ddp else self.model

            with self.ctx:
                # If critic head is enabled for LANGUAGE_MODEL, compute LM loss explicitly and add critic loss here
                if getattr(getattr(raw_model, 'config', object()), 'mode', None) == ModelMode.LANGUAGE_MODEL \
                   and getattr(getattr(raw_model, 'config', object()), 'add_critic_head', False) \
                   and hasattr(raw_model, 'critic_head'):
                    # 0) Compute effective critic alpha with linear warmup from config and current iter
                    base = float(getattr(raw_model.config, 'critic_alpha', 0.0) or 0.0)
                    start = int(getattr(raw_model.config, 'start_critic_iteration', 0) or 0)
                    end = int(getattr(raw_model.config, 'end_critic_iteration', 0) or 0)
                    it = int(getattr(loss_modifiers, 'current_iter', 0) if loss_modifiers is not None else 0)
                    if end <= start:
                        alpha_eff = base
                    elif it < start:
                        alpha_eff = 0.0
                    elif it >= end:
                        alpha_eff = base
                    else:
                        alpha_eff = max(0.0, min(base, base * float(it - start) / max(1.0, float(end - start))))

                    # 1) Forward LM to get full-sequence logits without triggering internal loss/critic
                    x_enc = raw_model._encode_tokens(X)
                    logits_gen = raw_model.lm_head(x_enc)

                    # 2) Compute LM base loss (per-position then aggregate) and apply loss_modifiers if any
                    flat_logits = logits_gen.view(-1, logits_gen.size(-1))
                    flat_targets = Y.view(-1)
                    per_pos = F.cross_entropy(
                        flat_logits, flat_targets,
                        ignore_index=raw_model.config.ignore_index,
                        reduction='none'
                    ).view_as(Y)
                    valid_mask = (Y != raw_model.config.ignore_index)
                    base_lm_loss = (per_pos * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
                    if loss_modifiers is not None and not loss_modifiers.is_empty():
                        lm_loss = loss_modifiers.modify_loss(
                            logits_gen, Y, base_lm_loss,
                            mask=valid_mask,
                            per_position_loss=per_pos,
                            ignore_index=raw_model.config.ignore_index,
                            model_mode=raw_model.config.mode
                        )
                    else:
                        lm_loss = base_lm_loss

                    if alpha_eff > 0.0:
                        # 3) Sample predictions using multinomial to reflect inference-time stochasticity
                        with torch.no_grad():
                            probs = torch.softmax(logits_gen.detach(), dim=-1)
                            sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view_as(Y)
                            pred_tokens = sampled

                        # 4) Build critic input by filling masked positions
                        if getattr(raw_model.config, 'mask_token_id', None) is not None:
                            masked_positions = (X == int(raw_model.config.mask_token_id))
                        else:
                            masked_positions = valid_mask
                        critic_input = X.clone()
                        critic_input[masked_positions] = pred_tokens[masked_positions]

                        # 5) Critic forward and loss
                        critic_logits = raw_model.critic_scores(critic_input)
                        critic_target = (critic_input != Y).float()
                        if getattr(raw_model.config, 'pad_token_id', None) is not None:
                            critic_valid = valid_mask & (X != int(raw_model.config.pad_token_id))
                        else:
                            critic_valid = valid_mask
                        critic_per_pos = F.binary_cross_entropy_with_logits(critic_logits, critic_target, reduction='none')
                        critic_loss = (critic_per_pos * critic_valid.float()).sum() / (critic_valid.float().sum() + 1e-8)

                        # 6) Combine losses with effective alpha
                        loss = lm_loss + float(alpha_eff) * critic_loss
                        # Track loss components before grad-accum scaling
                        self.last_loss_main = float(lm_loss.detach().item())
                        self.last_loss_critic = float(alpha_eff) * float(critic_loss.detach().item())
                    else:
                        # No critic contribution during warmup pre-start
                        loss = lm_loss
                        self.last_loss_main = float(lm_loss.detach().item())
                        self.last_loss_critic = 0.0
                else:
                    # Fallback: defer to model's standard loss path
                    _, loss = self.model(X, Y, loss_modifiers=loss_modifiers)
                    # Track only main loss component in fallback path
                    try:
                        self.last_loss_main = float(loss.detach().item())
                    except Exception:
                        self.last_loss_main = 0.0
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

