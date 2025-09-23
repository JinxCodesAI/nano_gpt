from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Global batch type used across training loop and evaluator (dict-only)
Batch = Dict[str, torch.Tensor]


def unpack_batch(b: Batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize batch dict to (X, Y, attention_mask).

    Expects dict with keys {x|input_ids, y|targets, attention_mask?}.
    """
    if not isinstance(b, dict):
        raise TypeError("Batch must be a dict[str, Tensor]")
    X = b.get('x', b.get('input_ids'))
    Y = b.get('y', b.get('targets'))
    attn = b.get('attention_mask', None)
    return X, Y, attn

