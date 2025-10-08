"""
Model evaluation functionality.

This module provides the Evaluator class that encapsulates loss estimation
logic extracted from the original estimate_loss() function in train.py.
"""

import torch
from model import ModelMode
from core.batch import Batch, unpack_batch

from typing import Dict

from sample_utils import build_critic_artifacts_from_logits


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

        # DDP-safe direct handle
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model


        # Temporarily disable loss modifiers during evaluation to get comparable baseline metrics
        with self.loss_modifier_pipeline.temporarily_disabled():
            # Optionally reset validation stream to a fixed start for deterministic eval windows
            if self.reset_val_stream_each_eval and 'val' in splits:
                try:
                    self.consumer.reset_state('val')
                except Exception:
                    pass


            for split in splits:
                # Unified evaluation: switch mode per-batch and compute loss
                losses = torch.zeros(self.eval_iters)
                # Validation-only accumulators for LM critic stats
                val_tokens_total = 0
                val_masked_total = 0
                critic_correct_total = 0
                critic_sum_pred_t0 = 0.0
                critic_sum_pred_t1 = 0.0
                critic_cnt_t0 = 0
                critic_cnt_t1 = 0
                critic_pred_t0_list = []
                critic_pred_t1_list = []

                for k in range(self.eval_iters):
                    batch = self.consumer.get_batch(split, self.device)

                    # Switch model mode based on batch metadata (same policy as training)
                    if '_model_mode' in batch:
                        mode_str = batch['_model_mode']
                        if mode_str == 'language_model' or mode_str == ModelMode.LANGUAGE_MODEL:
                            raw_model.set_mode(ModelMode.LANGUAGE_MODEL)
                        elif mode_str == 'sequence_scorer' or mode_str == ModelMode.SEQUENCE_SCORER:
                            raw_model.set_mode(ModelMode.SEQUENCE_SCORER)
                        elif isinstance(mode_str, ModelMode):
                            raw_model.set_mode(mode_str)

                    X, Y = unpack_batch(batch)
                    with self.ctx:
                        logits, loss = self.model(X, Y, loss_modifiers=self.loss_modifier_pipeline)

                    # If LM with critic enabled, deblend alpha for reporting
                    alpha_eff = 0.0
                    has_critic = False
                    try:
                        if raw_model.get_mode() == ModelMode.LANGUAGE_MODEL \
                           and getattr(raw_model.config, 'add_critic_head', False):
                            has_critic = True
                            alpha_eff = float(raw_model._effective_critic_alpha())
                    except Exception:
                        alpha_eff = 0.0
                    val = float(loss.item())
                    if alpha_eff > 0.0:
                        val = val / (1.0 + alpha_eff)
                    losses[k] = val

                    # Validation-only stats for LM batches
                    if split == 'val' and raw_model.get_mode() == ModelMode.LANGUAGE_MODEL:
                        val_tokens_total += int(Y.numel())
                        ignore_index = int(getattr(raw_model.config, 'ignore_index', -100))
                        supervised_mask = (Y != ignore_index)
                        val_masked_total += int(supervised_mask.sum().item())

                        if has_critic:
                            if getattr(raw_model.config, 'mask_token_id', None) is None:
                                raise RuntimeError("Evaluator: mask_token_id is required for critic stats")
                            artifacts = build_critic_artifacts_from_logits(
                                idx=X,
                                logits=logits,
                                targets=Y,
                                mask_token_id=int(raw_model.config.mask_token_id),
                                ignore_index=int(getattr(raw_model.config, 'ignore_index', -100)),
                                pad_token_id=getattr(raw_model.config, 'pad_token_id', None),
                                scope=getattr(raw_model.config, 'critic_target_scope', 'masked_and_ignore'),
                            )
                            pred_tokens = artifacts['pred_tokens']
                            critic_input = artifacts['critic_input']
                            critic_target = artifacts['critic_target']
                            critic_valid = artifacts['critic_valid']
                            masked_positions = (X == int(raw_model.config.mask_token_id))
                            critic_correct_total += int((pred_tokens[masked_positions] == Y[masked_positions]).sum().item())
                            critic_logits = raw_model.critic_scores(critic_input)
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

                if split == 'val' and (critic_cnt_t0 + critic_cnt_t1) > 0:
                    out['val/tokens_total'] = int(val_tokens_total)
                    out['val/masked_total'] = int(val_masked_total)
                    out['val/critic_correct_total'] = int(critic_correct_total)
                    out['val/critic_target_zeros'] = int(critic_cnt_t0)
                    out['val/critic_target_ones'] = int(critic_cnt_t1)
                    if critic_cnt_t0 > 0:
                        out['val/critic_pred_mean_for_target0'] = float(critic_sum_pred_t0 / max(critic_cnt_t0, 1))
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

        self.model.train()
        return out