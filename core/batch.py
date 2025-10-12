from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Global batch type used across training loop and evaluator (dict-only)
Batch = Dict[str, torch.Tensor]


def unpack_batch(b: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize batch dict to (X, Y) with unified naming first.

    Accepted keys:
    - unified: input, target
    - legacy LM: x, y
    - legacy SS: input_ids, targets
    """
    if not isinstance(b, dict):
        raise TypeError("Batch must be a dict[str, Tensor]")
    X = b.get('input', b.get('x', b.get('input_ids')))
    Y = b.get('target', b.get('y', b.get('targets')))
    return X, Y


def resolve_attention_mask(
    batch: Batch,
    X: torch.Tensor,
    Y: Optional[torch.Tensor],
    model_config,
) -> Optional[torch.Tensor]:
    """Resolve the attention mask for a batch if one is available or derivable.

    Priority order:
    1. Explicit ``attention_mask`` provided in the batch tensors.
    2. Token mask derived from ``pad_token_id`` and ``ignore_index`` metadata when
       available. Any positions flagged as padding or ignore-index are suppressed so
       decoder attention never incorporates tokens that should be excluded from the
       loss.

    Returns ``None`` when no reliable signal is available so callers can fall back to
    the model's default behaviour (typically fully visible attention).
    """

    attention_mask = batch.get('attention_mask')
    if attention_mask is not None:
        return attention_mask.to(device=X.device, dtype=torch.long)

    pad_token_id = getattr(model_config, 'pad_token_id', None)
    ignore_index = getattr(model_config, 'ignore_index', None)

    if pad_token_id is None and (ignore_index is None or Y is None):
        return None

    mask = torch.ones_like(X, dtype=torch.bool)

    if pad_token_id is not None:
        mask &= (X != int(pad_token_id))

    if ignore_index is not None and Y is not None:
        mask &= (Y != int(ignore_index))

    return mask.to(dtype=torch.long)

