from __future__ import annotations

import torch


def sequence_scorer_target_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Non-linear transform used for sequence scorer targets and judge-weight modifier.
    Implements: y = 1 - (-x^4 + 2x^3 - 2x + 1)
    Assumes x in [0, 1]; returns y in [0, 1].
    """
    y = 1 - (-x**4 + 2 * x**3 - 2 * x + 1)
    return y

