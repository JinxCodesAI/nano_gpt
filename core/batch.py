from __future__ import annotations

from typing import Union, Dict, Tuple, Optional

import torch

# Global batch type used across training loop and evaluator
Batch = Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]


def unpack_batch(b: Batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize different batch shapes to (X, Y, attention_mask).

    Supports:
    - Tuple[Tensor, Tensor] -> (X, Y, None)
    - Dict with keys {x|input_ids, y|targets, attention_mask?}
    """
    if isinstance(b, tuple):
        X, Y = b
        return X, Y, None
    if isinstance(b, dict):
        X = b.get('x', b.get('input_ids'))
        Y = b.get('y', b.get('targets'))
        attn = b.get('attention_mask', None)
        return X, Y, attn
    raise TypeError("Unsupported batch type")

