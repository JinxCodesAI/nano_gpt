from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch

# Global batch type used across training loop and evaluator (dict-only)
Batch = Dict[str, torch.Tensor]


def unpack_batch(b: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize batch dict to (X, Y) using batch-level model_mode when available.

    Rules:
    - If b['_model_mode'] == 'sequence_scorer' (or ModelMode.SEQUENCE_SCORER): use ('input_ids', 'targets').
    - Else (default language_model): use ('x', 'y').
    - Fall back only if the expected keys are missing, raising clear errors on misconfiguration.
    """
    if not isinstance(b, dict):
        raise TypeError("Batch must be a dict[str, Tensor]")

    mode = b.get('_model_mode', 'language_model')
    # Support enum passed through
    try:
        from model import ModelMode
        if isinstance(mode, ModelMode):
            mode = mode.value
    except Exception:
        pass

    if mode == 'sequence_scorer':
        if 'input_ids' not in b or 'targets' not in b:
            raise KeyError("sequence_scorer batch must contain 'input_ids' and 'targets'")
        return b['input_ids'], b['targets']
    else:  # language_model (default)
        if 'x' not in b or 'y' not in b:
            # Provide a helpful message if mixed keys exist but mode is LM
            if 'input_ids' in b or 'targets' in b:
                raise KeyError("language_model batch must contain 'x' and 'y'; found sequence_scorer keys instead. Check model_mode metadata.")
            raise KeyError("language_model batch must contain 'x' and 'y'")
        return b['x'], b['y']

