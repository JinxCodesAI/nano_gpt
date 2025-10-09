"""Utility helpers for preparing guidance representations during training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from core.batch import Batch


def _ensure_mask(mask: Optional[torch.Tensor], shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Create or reshape a key/value mask to match the expected shape."""

    if mask is None:
        return torch.ones(shape, dtype=torch.long, device=device)

    if mask.shape != shape:
        # Broadcast 1D masks if provided per-example
        if mask.dim() == 1 and mask.shape[0] == shape[0]:
            mask = mask[:, None].expand(shape)
        else:
            mask = mask.view(shape)
    return mask


def prepare_guidance(
    raw_model,
    batch: Batch,
    tokens: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare (guidance_h, guidance_mask) for model forward passes.

    This helper centralises the plan-encoder integration so both the training loop
    and evaluator can seamlessly enable guidance whenever the configuration
    requests it.
    """

    if raw_model is None:
        return None, None

    if not getattr(raw_model.config, "use_guidance", False):
        return None, None

    plan_encoder = getattr(raw_model, "plan_encoder", None)
    if plan_encoder is None:
        return None, None

    device = tokens.device

    # Prefer precomputed guidance states if the batch provides them
    if "guidance_states" in batch:
        guidance_h = batch["guidance_states"].to(device)
        guidance_mask = batch.get("guidance_mask")
        if guidance_mask is not None:
            guidance_mask = guidance_mask.to(device)
        else:
            guidance_mask = torch.ones(
                guidance_h.size(0), guidance_h.size(1), dtype=torch.long, device=device
            )
    else:
        guidance_input = batch.get("guidance_input")
        embedded_flag = batch.get("guidance_input_embedded", False)
        if isinstance(embedded_flag, torch.Tensor):
            guidance_input_embedded = bool(embedded_flag.item())
        else:
            guidance_input_embedded = bool(embedded_flag)
        guidance_mask = batch.get("guidance_mask")

        if guidance_input is None:
            guidance_input = tokens
            guidance_mask = attention_mask

        guidance_input = guidance_input.to(device)
        if guidance_mask is not None:
            guidance_mask = guidance_mask.to(device)

        guidance_h, guidance_mask = plan_encoder(
            guidance_input,
            src_mask=guidance_mask,
            already_embedded=guidance_input_embedded,
            return_mask=True,
        )

    # Apply classifier-free guidance dropout only while training
    cond_dropout_prob = float(getattr(raw_model.config, "cond_dropout_prob", 0.0) or 0.0)
    if cond_dropout_prob > 0.0 and raw_model.training and guidance_h is not None:
        drop_mask = torch.rand(guidance_h.size(0), device=device) < cond_dropout_prob
        if drop_mask.any():
            if drop_mask.all():
                return None, None
            guidance_h = guidance_h.clone()
            guidance_mask = _ensure_mask(guidance_mask, (guidance_h.size(0), guidance_h.size(1)), device)
            guidance_h[drop_mask] = 0.0
            guidance_mask = guidance_mask.clone()
            guidance_mask[drop_mask] = 0

    return guidance_h, guidance_mask

