from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Global batch type used across training loop and evaluator (dict-only)
Batch = Dict[str, torch.Tensor]


def unpack_batch(b: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize batch dict to (X, Y).

    Expects dict with keys {x|input_ids, y|targets}.
    """
    if not isinstance(b, dict):
        raise TypeError("Batch must be a dict[str, Tensor]")
    X = b.get('x', b.get('input_ids'))
    Y = b.get('y', b.get('targets'))
    return X, Y

