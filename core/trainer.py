"""
Trainer orchestrator that coordinates the main training loop.

Extracts the control flow from train.py without changing functionality or
logging/timing behavior.
"""

import time
from typing import Optional
from core.batch import Batch

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

        # Cumulative history of per-sample mask ratios (never cleared during run)
        self._mask_ratio_history = []

    def train(self) -> None:
        # Initial progress update (parity with train.py)
        self.checkpoint_manager.update_progress(
            iter_num=self.iter_num, best_val_loss=self.best_val_loss
        )

        # fetch the very first batch (dict)
        batch: Batch = self.consumer.get_batch('train', self.device)
        t0 = time.time()
        local_iter_num = 0
        raw_model = self.model.module if self.ddp else self.model
        running_mfu = -1.0

        while True:
            # determine and set the learning rate for this iteration (via TrainingStep)
            lr = self.training_step.set_learning_rate(self.iter_num)

            # evaluate the loss on train/val sets and write checkpoints
            if self.iter_num % self.eval_interval == 0 and self.master_process:
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

            # Accumulate per-sample mask ratios every iteration (independent of log_interval)
            try:
                raw_ignore = getattr(raw_model.config, 'ignore_index', -100)
                bx = batch.get('x', None)
                by = batch.get('y', None)
                battn = batch.get('attention_mask', None)
                if bx is not None and by is not None:
                    supervised_now = (by != raw_ignore)
                    if battn is not None:
                        valid_now = (battn != 0).sum(dim=1)
                        ratios_now = (supervised_now.float().sum(dim=1) / valid_now.clamp_min(1)).clamp(0, 1)
                    else:
                        Tnow = bx.shape[1]
                        ratios_now = (supervised_now.float().sum(dim=1) / float(Tnow)).clamp(0, 1)
                    self._mask_ratio_history.append(ratios_now.detach().float().cpu())
            except Exception:
                pass

            # Dynamic unfreezing support (delegated to TrainingStep)
            self.training_step.maybe_unfreeze(self.iter_num)

            # forward/backward/update handled by TrainingStep
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

                # Diagnostics: verify batch integrity and attention behavior
                try:
                    meta = self.consumer.meta
                    mask_id = int(meta.get('mask_token_id')) if 'mask_token_id' in meta else -1
                    sep_id = int(meta.get('sep_token_id')) if 'sep_token_id' in meta else -1
                    ignore_index = getattr(raw_model.config, 'ignore_index', -100)

                    x = batch.get('x', None)
                    y = batch.get('y', None)
                    attn = batch.get('attention_mask', None)

                    if x is not None and y is not None:
                        with torch.no_grad():
                            B, T = x.shape[0], x.shape[1]
                            supervised = (y != ignore_index)
                            total_supervised = int(supervised.sum().item())

                            if attn is not None:
                                attn_bool = attn != 0
                                invalid_supervised = int((supervised & ~attn_bool).sum().item())
                                valid_len = attn_bool.sum(dim=1)
                                valid_len_mean = float(valid_len.float().mean().item())
                                # per-sample masked ratio over attention-visible tokens
                                per_sample_mask_ratio = (supervised.float().sum(dim=1) / valid_len.clamp_min(1)).clamp(0, 1)
                            else:
                                invalid_supervised = 0
                                valid_len_mean = float(T)
                                # fallback: denominator is full T if no attention mask
                                per_sample_mask_ratio = (supervised.float().sum(dim=1) / float(T)).clamp(0, 1)

                            # Percentiles of per-sample mask ratio
                            try:
                                qs = torch.tensor([0.10, 0.50, 0.90], device=per_sample_mask_ratio.device, dtype=per_sample_mask_ratio.dtype)
                                p10, p50, p90 = torch.quantile(per_sample_mask_ratio, qs).tolist()
                            except Exception:
                                sorted_vals, _ = torch.sort(per_sample_mask_ratio)
                                n = sorted_vals.numel()
                                def idx(q):
                                    return min(n-1, max(0, int(q * (n-1))))
                                p10 = float(sorted_vals[idx(0.10)].item()) if n > 0 else 0.0


                                p50 = float(sorted_vals[idx(0.50)].item()) if n > 0 else 0.0
                                p90 = float(sorted_vals[idx(0.90)].item()) if n > 0 else 0.0

                            # Corruption type breakdown on supervised positions
                            if total_supervised > 0:
                                sup_x = x[supervised]
                                sup_y = y[supervised]
                                pct_unchanged = float((sup_x == sup_y).float().mean().item())
                                pct_mask = float((sup_x == mask_id).float().mean().item()) if mask_id >= 0 else 0.0
                                random_mask = (sup_x != sup_y)
                                if mask_id >= 0:
                                    random_mask &= (sup_x != mask_id)
                                pct_random = float(random_mask.float().mean().item())

                                # Accumulate per-sample mask ratios for cumulative stats (never cleared)
                                try:
                                    self._mask_ratio_history.append(per_sample_mask_ratio.detach().float().cpu())
                                    hist_tensor = torch.cat(self._mask_ratio_history)
                                    qs_total = torch.tensor([0.10, 0.50, 0.90], device=hist_tensor.device, dtype=hist_tensor.dtype)
                                    p10_c, p50_c, p90_c = torch.quantile(hist_tensor, qs_total).tolist()
                                    step_metrics["diag/mask_ratio_p10_cum"] = float(p10_c)
                                    step_metrics["diag/mask_ratio_p50_cum"] = float(p50_c)
                                    step_metrics["diag/mask_ratio_p90_cum"] = float(p90_c)
                                except Exception:
                                    # Best-effort: do not disrupt training if history aggregation fails
                                    pass
                            else:
                                pct_unchanged = 0.0
                                pct_mask = 0.0
                                pct_random = 0.0

                            step_metrics.update({
                                "diag/supervised_per_token": total_supervised / max(1, B*T),
                                "diag/valid_len_mean": valid_len_mean,
                                "diag/supervised_outside_attn": invalid_supervised,
                                "diag/pct_mask": pct_mask,
                                "diag/pct_random": pct_random,
                                "diag/pct_unchanged": pct_unchanged,
                                "diag/mask_ratio_p10": float(p10),
                                "diag/mask_ratio_p50": float(p50),
                                "diag/mask_ratio_p90": float(p90),
                            })

                            # Attention ablation probe on a tiny subsample (no grad)
                            try:
                                sub_n = min(4, x.size(0))
                                x_sub = x[:sub_n]
                                y_sub = y[:sub_n]
                                attn_sub = attn[:sub_n] if attn is not None else None
                                lm = self.evaluator.loss_modifier_pipeline
                                # With current modifiers
                                _, loss_attn = raw_model(x_sub, y_sub, attention_mask=attn_sub, loss_modifiers=lm)
                                _, loss_noattn = raw_model(x_sub, y_sub, attention_mask=None, loss_modifiers=lm)
                                if loss_attn is not None and loss_noattn is not None:
                                    step_metrics["diag/loss_attn_sub"] = float(loss_attn.item())
                                    step_metrics["diag/loss_noattn_sub"] = float(loss_noattn.item())
                                    step_metrics["diag/loss_noattn_minus_attn"] = float(loss_noattn.item() - loss_attn.item())
                                # Baseline CE (modifiers disabled) and supervised accuracy @ top1
                                try:
                                    with lm.temporarily_disabled():
                                        logits_ce, loss_ce_attn = raw_model(x_sub, y_sub, attention_mask=attn_sub, loss_modifiers=lm)
                                    if loss_ce_attn is not None:
                                        step_metrics["diag/loss_ce_attn_sub"] = float(loss_ce_attn.item())
                                    # supervised accuracy
                                    sup_mask_sub = (y_sub != ignore_index)
                                    if sup_mask_sub.any():
                                        preds = logits_ce.argmax(dim=-1)
                                        acc = (preds[sup_mask_sub] == y_sub[sup_mask_sub]).float().mean().item()
                                        step_metrics["diag/sup_acc_attn_sub"] = float(acc)
                                except Exception:
                                    pass
                            except Exception:
                                # Do not disrupt training if probe fails
                                pass
                except Exception:
                    # Non-fatal: diagnostics are best-effort only
                    pass

                self.logger.log_step(step_metrics)

            self.iter_num += 1
            self.checkpoint_manager.update_progress(iter_num=self.iter_num, best_val_loss=self.best_val_loss)
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break
            t0 = time.time()

