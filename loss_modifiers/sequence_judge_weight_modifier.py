"""
SequenceScoringJudgeWeightModifier

- Supports LANGUAGE_MODEL only.
- Eager-loads a SEQUENCE_SCORER judge model from a training checkpoint.
- Samples a completion sequence from language-model logits (single-pass, per-position sampling).
- Computes masking_ratio from targets via mask = (targets != ignore_index).
- Computes per-sample factor = clamp((wrongness / clamp(masking_ratio, eps)) ** exponent,
  min_factor, max_factor), and scales the loss accordingly.
- Logs only: mean/min/max/std for wrongness and factor.

This modifier does not change base classes or the pipeline. It prefers to scale
per-position losses when provided (returning a replacement per_position_loss), so
the pipeline's final aggregation yields the intended scalar loss. If per-position
loss is not provided, it falls back to scaling the input scalar loss by mean factor.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from .base import BaseLossModifier
from model import ModelMode, GPTConfig, GPT


class SequenceScoringJudgeWeightModifier(BaseLossModifier):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Required config
        ckpt_path = config.get('judge_weight_checkpoint', None)
        if not self.enabled:
            # Still validate minimally to fail early if misconfigured while enabled is False? No.
            # Keep zero overhead when disabled.
            self._judge = None
            return
        if ckpt_path is None or str(ckpt_path).strip() == "":
            raise ValueError("judge_weight_checkpoint must be provided when judge_weight_modifier_enabled=True")

        # Hyperparameters
        self.exponent: float = float(config.get('judge_weight_exponent', 1.0))
        self.min_factor: float = float(config.get('judge_weight_min_factor', 0.1))
        self.max_factor: float = float(config.get('judge_weight_max_factor', 10.0))
        self.eps: float = float(config.get('judge_weight_eps', 1e-6))

        # Device/dtype from main training config
        self.device: str = str(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        dtype_str: str = str(config.get('dtype', 'bfloat16'))
        self.ptdtype = {
            'float32': torch.float32,
            'bfloat16': torch.bfloat16,
            'float16': torch.float16,
        }.get(dtype_str, torch.bfloat16)

        # EMA state for multiplayer std across steps
        self._mul_std_ema: Optional[float] = None

        # Eager-load judge to fail fast
        self._judge = self._load_judge(ckpt_path)
        self._cls_token_id: Optional[int] = getattr(self._judge.config, 'cls_token_id', None)
        self._judge_block_size: int = int(self._judge.config.block_size)

    # -------- internal helpers --------
    @staticmethod
    def _cleanup_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove known unwanted prefixes from keys (e.g., '_orig_mod.')."""
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        return state_dict

    def _load_judge(self, ckpt_path: str) -> GPT:
        # Torch 2.6 defaults weights_only=True which breaks pickled dict checkpoints.
        # Prefer explicit weights_only=False, with fallback for older torch versions.
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only kwarg
            ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model' not in ckpt or 'model_args' not in ckpt:
            raise ValueError("Judge checkpoint must contain 'model' and 'model_args'")
        model_args = dict(ckpt['model_args'])
        # Enforce SEQUENCE_SCORER
        mode = ModelMode(model_args.get('mode', 'sequence_scorer'))
        if mode != ModelMode.SEQUENCE_SCORER:
            raise ValueError(f"Judge checkpoint mode must be SEQUENCE_SCORER, got {mode}")
        # Do NOT chain-load judge's original pretrain; we have full weights here
        model_args['init_from_checkpoint'] = None
        # The judge is used only for inference inside the modifier: don't freeze
        model_args['freeze_transformer'] = False
        gptconf = GPTConfig(**model_args)
        judge = GPT(gptconf)
        state_dict = self._cleanup_state_dict_keys(ckpt['model'])
        judge.load_state_dict(state_dict)
        judge.to(self.device)
        # dtype cast of parameters (floating point only) and buffers in-place
        for p in judge.parameters():
            if torch.is_floating_point(p):
                p.data = p.data.to(self.ptdtype)
        cast_bufs = 0
        for _, buf in judge.named_buffers(recurse=True):
            if torch.is_floating_point(buf):
                buf.data = buf.data.to(self.ptdtype)
                cast_bufs += 1
        # minimal debug to aid diagnosis during bring-up
        print(f"[JudgeLoader] Loaded judge on {self.device} as {self.ptdtype}; cast {cast_bufs} floating buffers")
        judge.eval()
        return judge

    @staticmethod
    def _sample_sequences_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        Sample one token per position independently from LM logits.
        logits: (B, T, V) -> return LongTensor (B, T)
        """
        B, T, V = logits.shape
        probs = F.softmax(logits, dim=-1).to(torch.float32)
        # Flatten over positions for a single multinomial call
        sampled = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, T)
        return sampled

    def _build_judge_input(self, sampled_ids: torch.Tensor) -> torch.Tensor:
        """
        Prepend CLS if judge has cls_token_id; crop to judge block size.
        sampled_ids: (B, T)
        returns (B, T') where T'<=block_size
        """
        B, T = sampled_ids.shape
        if self._cls_token_id is not None:
            cls_col = torch.full((B, 1), int(self._cls_token_id), dtype=torch.long, device=sampled_ids.device)
            # Drop last token to preserve original length when prepending CLS
            core = sampled_ids[:, :-1] if T > 0 else sampled_ids
            seq = torch.cat([cls_col, core], dim=1)
        else:
            seq = sampled_ids
        # Crop if needed to respect judge block size
        if seq.shape[1] > self._judge_block_size:
            seq = seq[:, :self._judge_block_size]
        return seq

    def supports_mode(self, mode: ModelMode) -> bool:
        return mode == ModelMode.LANGUAGE_MODEL

    def modify_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: torch.Tensor,
        model_mode: ModelMode = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        if not self.enabled:
            return loss
        if model_mode is not None and not self.supports_mode(model_mode):
            return loss
        if self._judge is None:
            # Should not happen because we eager-load when enabled
            raise RuntimeError("Judge model not loaded")

        device = logits.device
        # Sampling from LM logits to get completion sequences
        with torch.no_grad():
            sampled_ids = self._sample_sequences_from_logits(logits)
            judge_input = self._build_judge_input(sampled_ids).to(device)
            wrong_logits, _ = self._judge(judge_input, targets=None, loss_modifiers=None)
            wrong = wrong_logits.view(-1).detach()
            wrong = torch.clamp(wrong, min=0.0, max=1.0)

        # Masking ratio from targets mask (prefer provided mask; else infer via ignore_index)
        mask: Optional[torch.Tensor] = kwargs.get('mask', None)
        if mask is None:
            ignore_index = kwargs.get('ignore_index', -100)
            mask = (targets != ignore_index)
        mask_f = mask.float()
        B, T = mask_f.shape
        ratio = mask_f.sum(dim=1) / (float(T) + self.eps)
        ratio = torch.clamp(ratio, min=self.eps)


        factor = wrong / ratio
        # Per-sample factor
        multiplier = torch.clamp((factor) ** self.exponent, min=self.min_factor, max=self.max_factor)

        # Metrics: only multiplayer_median and multiplayer_std_ema (EMA factor 0.99)
        with torch.no_grad():
            med = torch.median(multiplier.detach().float())
            cur_std = multiplier.detach().float().std(unbiased=False)
            cur_std_val = float(cur_std.cpu().item())
            if self._mul_med_ema is None:
                self._mul_med_ema = med
            else:
                alpha = 0.99
                self._mul_med_ema = float(alpha * self._mul_med_ema + (1.0 - alpha) * med)
            if self._mul_std_ema is None:
                self._mul_std_ema = cur_std_val
            else:
                alpha = 0.99
                self._mul_std_ema = float(alpha * self._mul_std_ema + (1.0 - alpha) * cur_std_val)
            self._metrics = {
                'multiplayer_median': self._mul_med_ema,
                'multiplayer_std_ema': self._mul_std_ema,
            }

        # Prefer scaling per-position loss if provided to avoid pipeline overwrite
        per_position_loss: Optional[torch.Tensor] = kwargs.get('per_position_loss', None)
        if per_position_loss is not None:
            # Broadcast per-sample multiplier across positions
            scaled_ppl = per_position_loss * multiplier.view(-1, 1)
            return {'per_position_loss': scaled_ppl}

        # Fallback: scale scalar loss by mean multiplier
        return loss * multiplier.mean()

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

