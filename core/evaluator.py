"""
Model evaluation functionality.

This module provides the Evaluator class that encapsulates loss estimation
logic extracted from the original estimate_loss() function in train.py.
"""

import torch
from model import ModelMode
from core.batch import Batch, unpack_batch

from typing import Dict


class Evaluator:
    """
    Handles model evaluation over train/val splits.

    Encapsulates the exact logic from the original estimate_loss() function
    to maintain identical behavior while improving modularity.
    """

    def __init__(
        self,
        model,
        consumer,
        loss_modifier_pipeline,
        eval_iters: int,
        ctx,
        device: str,
        min_zero_for_stats: int = 0,
        max_extra_batches_for_zero_stats: int = 0,
        reset_val_stream_each_eval: bool = False,
    ):
        """
        Initialize evaluator with required components.

        Args:
            model: GPT model to evaluate
            consumer: DatasetConsumer for getting batches
            loss_modifier_pipeline: Loss modifier pipeline (temporarily disabled during eval)
            eval_iters: Number of evaluation iterations per split
            ctx: Context manager for autocast (nullcontext or torch.amp.autocast)
            device: Device string for getting batches (e.g., 'cuda', 'cuda:0', 'cpu')
            min_zero_for_stats: Minimum zero-target samples required to compute zero-only stats; 0 disables top-up
            max_extra_batches_for_zero_stats: Upper bound on extra val batches drawn to top-up zero-only stats
            reset_val_stream_each_eval: If True, reset consumer val stream at evaluation start for determinism
        """
        self.model = model
        self.consumer = consumer
        self.loss_modifier_pipeline = loss_modifier_pipeline
        self.eval_iters = eval_iters
        self.ctx = ctx
        self.device = device
        self.min_zero_for_stats = int(min_zero_for_stats or 0)
        self.max_extra_batches_for_zero_stats = int(max_extra_batches_for_zero_stats or 0)
        self.reset_val_stream_each_eval = bool(reset_val_stream_each_eval)

    @torch.no_grad()
    def evaluate(self, splits=None) -> Dict[str, float]:
        """
        Estimate loss over specified splits using many batches.

        This function replicates the exact logic from the original estimate_loss()
        function to ensure identical behavior.

        Args:
            splits: List of splits to evaluate ['train', 'val']. If None, evaluates both.

        Returns:
            Dictionary mapping split names to average loss values
        """
        if splits is None:
            splits = ['train', 'val']

        out = {}
        self.model.eval()

        # Determine model mode (DDP-safe)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        is_sequence_scorer = getattr(raw_model.config, 'mode', None) == ModelMode.SEQUENCE_SCORER


        # Temporarily disable loss modifiers during evaluation to get comparable baseline metrics
        with self.loss_modifier_pipeline.temporarily_disabled():
            # Optionally reset validation stream to a fixed start for deterministic eval windows
            if self.reset_val_stream_each_eval and 'val' in splits:
                try:
                    self.consumer.reset_state('val')
                except Exception:
                    pass


            for split in splits:
                # For non-sequence scorer or for train split: original behavior
                if (not is_sequence_scorer) or (split != 'val'):
                    losses = torch.zeros(self.eval_iters)
                    # Validation-only accumulators for extra console stats
                    val_tokens_total = 0
                    val_masked_total = 0
                    critic_correct_total = 0
                    critic_sum_pred_t0 = 0.0
                    critic_sum_pred_t1 = 0.0
                    critic_cnt_t0 = 0
                    critic_cnt_t1 = 0
                    # For percentiles (store samples; memory OK at eval scale)
                    critic_pred_t0_list = []
                    critic_pred_t1_list = []

                    for k in range(self.eval_iters):
                        batch = self.consumer.get_batch(split, self.device)
                        X, Y = unpack_batch(batch)
                        with self.ctx:
                            logits, loss = self.model(X, Y, loss_modifiers=self.loss_modifier_pipeline)
                        # Apply blended critic scheme for LANGUAGE_MODEL with critic enabled
                        alpha_eff = 0.0
                        has_critic = False
                        try:
                            if getattr(raw_model.config, 'mode', None) == ModelMode.LANGUAGE_MODEL \
                               and getattr(raw_model.config, 'add_critic_head', False):
                                has_critic = True
                                alpha_eff = float(raw_model._effective_critic_alpha())
                        except Exception:
                            alpha_eff = 0.0
                        val = float(loss.item())
                        if alpha_eff > 0.0:
                            val = val / (1.0 + alpha_eff)
                        losses[k] = val

                        # Collect validation-only stats (do not log for train split)
                        if split == 'val':
                            val_tokens_total += int(Y.numel())
                            ignore_index = int(getattr(raw_model.config, 'ignore_index', -100))
                            # For display, keep "masked_total" as supervised positions (Y != ignore_index)
                            supervised_mask = (Y != ignore_index)
                            val_masked_total += int(supervised_mask.sum().item())

                            if has_critic:
                                # Sample predictions from LM logits
                                with torch.no_grad():
                                    probs = torch.softmax(logits.detach(), dim=-1)
                                    sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view_as(Y)
                                    pred_tokens = sampled
                                # Determine masked positions (require mask_token_id)
                                if getattr(raw_model.config, 'mask_token_id', None) is None:
                                    raise RuntimeError("Evaluator: mask_token_id is required for critic stats")
                                masked_positions = (X == int(raw_model.config.mask_token_id))

                                # Correct predictions among masked positions
                                critic_correct_total += int((pred_tokens[masked_positions] == Y[masked_positions]).sum().item())

                                # Build critic input and compute critic logits
                                critic_input = X.clone()
                                critic_input[masked_positions] = pred_tokens[masked_positions]
                                critic_logits = raw_model.critic_scores(critic_input)

                                # Critic targets and validity by scope
                                scope = getattr(raw_model.config, 'critic_target_scope', 'masked_and_ignore')
                                pad_token_id = getattr(raw_model.config, 'pad_token_id', None)
                                critic_target = (critic_input != Y).float()
                                if scope == 'masked_and_ignore':
                                    critic_valid = masked_positions | (Y == ignore_index)
                                    # For ignore_index positions, target is always 0
                                    critic_target = torch.where((Y == ignore_index), torch.zeros_like(critic_target), critic_target)
                                elif scope == 'masked_only':
                                    critic_valid = masked_positions
                                else:
                                    raise RuntimeError(f"Evaluator: unsupported critic_target_scope {scope}")
                                if pad_token_id is not None:
                                    critic_valid = critic_valid & (X != int(pad_token_id))

                                # Sigmoid probabilities
                                critic_prob = torch.sigmoid(critic_logits)
                                t0_mask = critic_valid & (critic_target == 0)
                                t1_mask = critic_valid & (critic_target == 1)
                                if t0_mask.any():
                                    vals0 = critic_prob[t0_mask]
                                    critic_sum_pred_t0 += float(vals0.sum().item())
                                    critic_cnt_t0 += int(t0_mask.sum().item())
                                    critic_pred_t0_list.append(vals0.detach().float().cpu())
                                if t1_mask.any():
                                    vals1 = critic_prob[t1_mask]
                                    critic_sum_pred_t1 += float(vals1.sum().item())
                                    critic_cnt_t1 += int(t1_mask.sum().item())
                                    critic_pred_t1_list.append(vals1.detach().float().cpu())

                    out[split] = float(losses.mean().item())

                    # Attach validation-only console stats
                    if split == 'val':
                        out['val/tokens_total'] = int(val_tokens_total)
                        out['val/masked_total'] = int(val_masked_total)
                        if has_critic:
                            out['val/critic_correct_total'] = int(critic_correct_total)
                            out['val/critic_target_zeros'] = int(critic_cnt_t0)
                            out['val/critic_target_ones'] = int(critic_cnt_t1)
                            if critic_cnt_t0 > 0:
                                out['val/critic_pred_mean_for_target0'] = float(critic_sum_pred_t0 / max(critic_cnt_t0, 1))
                                # percentiles for target 0
                                try:
                                    import torch as _t
                                    preds0 = _t.cat(critic_pred_t0_list, dim=0).view(-1)
                                    out['val/critic_pred_p10_for_target0'] = float(_t.quantile(preds0, _t.tensor(0.1)).item())
                                    out['val/critic_pred_p90_for_target0'] = float(_t.quantile(preds0, _t.tensor(0.9)).item())
                                except Exception:
                                    try:
                                        preds0 = _t.cat(critic_pred_t0_list, dim=0).view(-1)
                                        sorted0, _ = _t.sort(preds0)
                                        n0 = sorted0.numel()
                                        i10 = max(int(0.1 * (n0 - 1)), 0)
                                        i90 = max(int(0.9 * (n0 - 1)), 0)
                                        out['val/critic_pred_p10_for_target0'] = float(sorted0[i10].item())
                                        out['val/critic_pred_p90_for_target0'] = float(sorted0[i90].item())
                                    except Exception:
                                        pass
                            if critic_cnt_t1 > 0:
                                out['val/critic_pred_mean_for_target1'] = float(critic_sum_pred_t1 / max(critic_cnt_t1, 1))
                                # percentiles for target 1
                                try:
                                    import torch as _t
                                    preds1 = _t.cat(critic_pred_t1_list, dim=0).view(-1)
                                    out['val/critic_pred_p10_for_target1'] = float(_t.quantile(preds1, _t.tensor(0.1)).item())
                                    out['val/critic_pred_p90_for_target1'] = float(_t.quantile(preds1, _t.tensor(0.9)).item())
                                except Exception:
                                    try:
                                        preds1 = _t.cat(critic_pred_t1_list, dim=0).view(-1)
                                        sorted1, _ = _t.sort(preds1)
                                        n1 = sorted1.numel()
                                        j10 = max(int(0.1 * (n1 - 1)), 0)
                                        j90 = max(int(0.9 * (n1 - 1)), 0)
                                        out['val/critic_pred_p10_for_target1'] = float(sorted1[j10].item())
                                        out['val/critic_pred_p90_for_target1'] = float(sorted1[j90].item())
                                    except Exception:
                                        pass
                else:
                    # Sequence scorer, validation split: two-stage evaluation
                    nonzero_losses = []
                    zero_preds = []
                    zeros_collected = 0

                    # First pass: fixed eval_iters window (for val loss comparability)
                    for k in range(self.eval_iters):
                        batch = self.consumer.get_batch(split, self.device)
                        X, Y = unpack_batch(batch)
                        mask_nonzero = (Y > 0)
                        mask_zero = (Y == 0)
                        if mask_nonzero.any():
                            Xnz = X[mask_nonzero]
                            Ynz = Y[mask_nonzero]
                            with self.ctx:
                                _, loss_nz = self.model(Xnz, Ynz, attention_mask=None, loss_modifiers=self.loss_modifier_pipeline)
                            nonzero_losses.append(float(loss_nz.item()))
                        if mask_zero.any():
                            Xz = X[mask_zero]
                            with self.ctx:
                                logits_z, _ = self.model(Xz, targets=None, attention_mask=None, loss_modifiers=self.loss_modifier_pipeline)
                            zero_preds.append(logits_z.detach().float().cpu())
                            zeros_collected += int(mask_zero.sum().item())

                    # Optional top-up: draw extra val batches only to stabilize zero-only stats
                    if (self.min_zero_for_stats > 0 and zeros_collected < self.min_zero_for_stats
                        and self.max_extra_batches_for_zero_stats > 0):
                        extra = 0
                        while extra < self.max_extra_batches_for_zero_stats and zeros_collected < self.min_zero_for_stats:
                            batch = self.consumer.get_batch(split, self.device)
                            X, Y = unpack_batch(batch)
                            mask_zero = (Y == 0)
                            if mask_zero.any():
                                Xz = X[mask_zero]
                                with self.ctx:
                                    logits_z, _ = self.model(Xz, targets=None, loss_modifiers=self.loss_modifier_pipeline)
                                zero_preds.append(logits_z.detach().float().cpu())
                                zeros_collected += int(mask_zero.sum().item())
                            extra += 1

                    if len(nonzero_losses) > 0:
                        out['val'] = float(sum(nonzero_losses) / len(nonzero_losses))
                    else:
                        out['val'] = float('nan')

                    if len(zero_preds) > 0:
                        preds = torch.cat(zero_preds, dim=0).view(-1)
                        out['val/zero_mean'] = float(preds.mean().item())
                        try:
                            out['val/zero_p90'] = float(torch.quantile(preds, torch.tensor(0.9)).item())
                        except Exception:
                            sorted_preds, _ = torch.sort(preds)
                            n = sorted_preds.numel()
                            idx = max(int(0.9 * (n - 1)), 0)
                            out['val/zero_p90'] = float(sorted_preds[idx].item())

        self.model.train()
        return out