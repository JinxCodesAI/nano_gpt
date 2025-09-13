import torch
from typing import Tuple, Dict, Any, Optional

from data.char_diffusion.masking_utils import (
    apply_span_masking_cpu,
    apply_target_driven_sticky_masking_cpu,
)


def apply_stage_masking_direct(
    x: torch.Tensor,
    stage_config: Dict[str, Any],
    mask_token_id: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply stage-specific masking with direct [MASK] token replacement (no BERT-style corruption).

    Returns:
        masked_x: Input with masked positions replaced by [MASK]
        mask: Boolean mask of masked positions
    """
    stage_type = stage_config['type']

    if stage_type == 'random':
        max_masked_ratio = stage_config['max_masked_ratio']
        batch_size = x.shape[0]
        # Different mask ratio per sample (vectorized)
        mask_ratios = torch.rand(batch_size, generator=rng, device=x.device) * max_masked_ratio
        # Generate mask probabilities for all positions once
        mask_probs = torch.rand(x.shape, generator=rng, device=x.device)
        # Broadcast per-sample ratios across sequence length
        thresholds = mask_ratios.view(-1, 1)
        mask = mask_probs < thresholds
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask

    elif stage_type == 'sticky':
        # Use existing sticky masking to compute mask, then directly replace with mask token
        target_masked_ratio = stage_config['target_masked_ratio']
        p1_probability = stage_config['p1_probability']
        p2_probability = stage_config['p2_probability']
        # We only need the mask; function returns (corrupted_x, mask)
        _, mask = apply_target_driven_sticky_masking_cpu(
            x, target_masked_ratio, p1_probability, p2_probability,
            mask_token_id, 0, rng  # vocab_size not used for direct masking
        )
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask

    elif stage_type == 'span':
        _, mask = apply_span_masking_cpu(
            x, stage_config['spans_count'], mask_token_id, 0, rng
        )
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask

    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def create_synthetic_text(
    original_text: torch.Tensor,
    mask_ratio: float,
    mlm_engine,
    mask_token_id: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Create synthetic text by masking and predicting with MLM model.

    Returns:
        synthetic_text: [B, T]
        actual_syntheticity: float in 0..1
    """
    batch_size, seq_len = original_text.shape

    # Random mask
    mask_probs = torch.rand(original_text.shape, generator=rng, device=original_text.device)
    mask = mask_probs < mask_ratio

    # Direct masking
    corrupted_input = original_text.clone()
    corrupted_input[mask] = mask_token_id

    # Predict masked tokens using MLM model
    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input,
        mask,
        temperature=sampling_temperature,
        top_k=top_k,
    )

    # Syntheticity equals fraction of positions we replaced
    synthetic_positions = mask
    total_positions = seq_len * batch_size
    actual_synthetic_count = synthetic_positions.sum().item()
    actual_syntheticity = actual_synthetic_count / max(total_positions, 1)

    return predicted_text, actual_syntheticity


def create_stage_synthetic_text(
    original_text: torch.Tensor,
    stage_config: Dict[str, Any],
    mlm_engine,
    mask_token_id: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Create synthetic text using stage-based masking configuration.
    """
    batch_size, seq_len = original_text.shape

    corrupted_input, mask = apply_stage_masking_direct(
        original_text, stage_config, mask_token_id, rng
    )

    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input,
        mask,
        temperature=sampling_temperature,
        top_k=top_k,
    )

    synthetic_positions = mask
    total_positions = seq_len * batch_size
    actual_synthetic_count = synthetic_positions.sum().item()
    actual_syntheticity = actual_synthetic_count / max(total_positions, 1)

    return predicted_text, actual_syntheticity


def add_cls_token(text: torch.Tensor, cls_token_id: int, block_size: int) -> torch.Tensor:
    """
    Add [CLS] token at the beginning of sequences, right-shifting and truncating as needed.
    """
    batch_size, seq_len = text.shape
    result = torch.zeros((batch_size, block_size), dtype=text.dtype, device=text.device)
    result[:, 0] = cls_token_id
    copy_len = min(seq_len, block_size - 1)
    result[:, 1:1+copy_len] = text[:, :copy_len]
    return result

