import random
import torch
from typing import Tuple, Dict, Any, Optional, List

from data.char_diffusion.masking_utils import (
    apply_span_masking_cpu,
    apply_target_driven_sticky_masking_cpu,
)


def apply_stage_masking_direct(
    x: torch.Tensor,
    stage_config: Dict[str, Any],
    mask_token_id: int,
    vocab_size: int,
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
        mask_ratios = torch.rand(batch_size, generator=rng).to(x.device) * max_masked_ratio
        # Generate mask probabilities for all positions once
        mask_probs = torch.rand(x.shape, generator=rng).to(x.device)
        # Broadcast per-sample ratios across sequence length
        thresholds = mask_ratios.view(-1, 1)
        mask = mask_probs < thresholds
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask

    elif stage_type == 'sticky':
        # Use existing sticky masking to compute mask, then directly replace with mask token
        target_masked_ratio = stage_config['target_masked_ratio']
        target_masked_ratio = max(random.uniform(0.0, target_masked_ratio), random.uniform(0.0, target_masked_ratio))
        p1_probability = stage_config['p1_probability']
        p2_probability = stage_config['p2_probability']
        # We only need the mask; function returns (corrupted_x, mask)
        _, mask = apply_target_driven_sticky_masking_cpu(
            x, target_masked_ratio, p1_probability, p2_probability,
            mask_token_id, vocab_size, rng
        )
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask

    elif stage_type == 'span':
        _, mask = apply_span_masking_cpu(
            x, stage_config['spans_count'], mask_token_id, vocab_size, rng
        )
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        return masked_x, mask



    else:
        raise ValueError(f"Unknown stage type: {stage_type}")


def create_synthetic_text(
    original_text: torch.Tensor,
    mask_ratio,  # float or Tensor[B]
    mlm_engine,
    mask_token_id: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic text by masking and predicting with MLM model.

    Returns:
        synthetic_text: [B, T]
        actual_syntheticity: float in 0..1
    """
    batch_size, seq_len = original_text.shape

    # Support per-sample mask ratios (vectorized)
    if not torch.is_tensor(mask_ratio):
        mask_ratio = torch.full((batch_size,), float(mask_ratio), device=original_text.device)
    else:
        mask_ratio = mask_ratio.to(original_text.device)

    # Random mask: different ratio per sample
    mask_probs = torch.rand(original_text.shape, generator=rng, device=original_text.device)
    thresholds = mask_ratio.view(-1, 1)
    mask = mask_probs < thresholds

    # Direct masking
    corrupted_input = original_text.clone()
    corrupted_input[mask] = mask_token_id

    # Predict masked tokens using MLM model (runs on mlm_engine.device)
    predicted_text = mlm_engine.predict_masked_tokens(
        corrupted_input,
        mask,
        temperature=sampling_temperature,
        top_k=top_k,
    )

    # Syntheticity equals fraction of positions we replaced, returned per-sample
    masked_counts = mask.sum(dim=1).to(torch.float32)
    actual_syntheticity = masked_counts / max(seq_len, 1)

    return predicted_text, actual_syntheticity


def create_stage_synthetic_text(
    original_text: torch.Tensor,
    stage_config: Dict[str, Any],
    mlm_engine,
    mask_token_id: int,
    vocab_size: int,
    rng: torch.Generator,
    sampling_temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic text using stage-based masking configuration.
    """
    batch_size, seq_len = original_text.shape

    corrupted_input, mask = apply_stage_masking_direct(
        original_text, stage_config, mask_token_id, vocab_size, rng
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


def apply_line_masking_direct(
    x: torch.Tensor,
    stage_config: Dict[str, Any],
    split_lines: List[List[int]],
    newline_token_id: int,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply line replacement masking: replace random FULL LINES with other FULL LINES from the split.

    Args:
        x: Input tensor [batch_size, seq_len]
        stage_config: Configuration with 'min_ratio', 'max_ratio'
        split_lines: All lines from the current split to sample replacements from
        newline_token_id: Token ID for newline character
        rng: Random number generator

    Returns:
        replaced_x: Input with replaced lines
        mask: Boolean mask indicating which positions were replaced
    """
    batch_size, seq_len = x.shape
    device = x.device

    # Get configuration parameters
    min_ratio = stage_config['min_ratio']
    max_ratio = stage_config['max_ratio']

    if not split_lines:
        # No replacement lines available, return original
        return x.clone(), torch.zeros_like(x, dtype=torch.bool, device=device)

    replaced_x = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool, device=device)

    # Process each sample in the batch
    for b in range(batch_size):
        sample = x[b]  # [seq_len]

        # Find newline positions to identify line boundaries
        newline_positions = torch.where(sample == newline_token_id)[0]

        if len(newline_positions) == 0:
            # No newlines found, treat entire sequence as one line
            line_starts = [0]
            line_ends = [seq_len]
        else:
            # Lines are separated by newlines
            line_starts = [0] + (newline_positions + 1).tolist()
            line_ends = newline_positions.tolist() + [seq_len]

            # Remove empty lines (where start >= end)
            valid_lines = [(s, e) for s, e in zip(line_starts, line_ends) if s < e]
            if not valid_lines:
                continue
            line_starts, line_ends = zip(*valid_lines)
            line_starts, line_ends = list(line_starts), list(line_ends)

        num_lines = len(line_starts)
        if num_lines == 0:
            continue

        # Calculate how many lines to replace based on ratio
        replacement_ratio = torch.rand(1, generator=rng).item()
        replacement_ratio = min_ratio + replacement_ratio * (max_ratio - min_ratio)
        num_lines_to_replace = max(1, int(replacement_ratio * num_lines))
        num_lines_to_replace = min(num_lines_to_replace, num_lines)  # Don't exceed available lines

        # Select which lines to replace
        line_indices = torch.randperm(num_lines, generator=rng)[:num_lines_to_replace].tolist()

        # Build new sequence by replacing selected lines
        new_sequence = []
        total_length = 0

        for line_idx in range(num_lines):
            if line_idx in line_indices:
                # Replace this line with a random line from split
                replacement_line = split_lines[torch.randint(0, len(split_lines), (1,), generator=rng).item()]
                new_sequence.extend(replacement_line)
                total_length += len(replacement_line)
            else:
                # Keep original line
                start_pos = line_starts[line_idx]
                end_pos = line_ends[line_idx]
                original_line = sample[start_pos:end_pos].tolist()
                new_sequence.extend(original_line)
                total_length += len(original_line)

            # Check if we exceed seq_len - if so, drop lines from the end
            if total_length > seq_len:
                # Truncate the sequence to fit
                new_sequence = new_sequence[:seq_len]
                break

        # Pad if necessary
        while len(new_sequence) < seq_len:
            new_sequence.append(0)  # Pad with zeros

        # Convert to tensor and create mask
        new_tensor = torch.tensor(new_sequence[:seq_len], dtype=torch.long, device=device)
        replaced_x[b] = new_tensor

        # Create mask by comparing original vs replaced
        mask[b] = (sample != new_tensor)

    return replaced_x, mask


def add_cls_token(text: torch.Tensor, cls_token_id: int, block_size: int, pad_token_id: int = 0) -> torch.Tensor:
    """
    Add [CLS] token at the beginning of sequences, right-shifting and truncating as needed.
    Fill remaining positions with PAD tokens.
    """
    batch_size, seq_len = text.shape
    result = torch.full((batch_size, block_size), pad_token_id, dtype=text.dtype, device=text.device)
    result[:, 0] = cls_token_id
    copy_len = min(seq_len, block_size - 1)
    result[:, 1:1+copy_len] = text[:, :copy_len]
    return result

