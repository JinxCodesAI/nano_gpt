#!/usr/bin/env python3
"""Model evaluator for diffusion checkpoints.

This script generates sticky-masked samples, evaluates multiple models with a
multinomial warmup followed by confidence-driven remasking, and reports token
match ratios. Results can optionally be inspected through a lightweight
interactive console viewer inspired by ``interactive_diffusion_explorer``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, ModelMode
from sample_utils import (
    linear_remasking_schedule,
    predict_and_sample_tokens,
)

# Utilities for masking
sys.path.append('data/char_diffusion')
from masking_utils import apply_target_driven_sticky_masking_cpu  # noqa: E402


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'float16' if DEVICE == 'cuda' else 'float32'


@dataclass(frozen=True)
class StickyConfig:
    target_ratio: float
    p1: float
    p2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'target_ratio': float(self.target_ratio),
            'p1': float(self.p1),
            'p2': float(self.p2),
        }


@dataclass
class EvaluationSample:
    sample_id: str
    base_index: int
    sticky: StickyConfig
    original_tokens: torch.Tensor
    masked_tokens: torch.Tensor
    protected_mask: torch.Tensor
    mask_positions: torch.Tensor


@dataclass
class SampleResult:
    model_name: str
    score: float
    correct: int
    total: int
    final_tokens: torch.Tensor
    stage1_tokens: torch.Tensor


def load_checkpoint_model(checkpoint_path: Path, *, device: str = DEVICE) -> Tuple[GPT, Dict]:
    """Load a GPT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_args' not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing 'model_args'")

    model_args = checkpoint['model_args'].copy()

    # Backward compatibility: default attention/position settings
    model_args.setdefault('attention_type', 'causal')
    model_args.setdefault('position_encoding', 'absolute')

    # Prevent cascading initializations during inference
    if model_args.get('init_from_checkpoint'):
        model_args['init_from_checkpoint'] = None

    deprecated = {'mode', 'num_token_classes', 'binary_classification'}
    filtered_args = {k: v for k, v in model_args.items() if k not in deprecated}

    config = GPTConfig(**filtered_args)
    model = GPT(config)

    # Restore original model mode when present
    old_mode = model_args.get('mode')
    if old_mode:
        if old_mode in (ModelMode.SEQUENCE_SCORER, 'sequence_scorer'):
            model.set_mode(ModelMode.SEQUENCE_SCORER)
        else:
            model.set_mode(ModelMode.LANGUAGE_MODEL)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    return model, checkpoint


def resolve_dataset_name(checkpoint: Dict, fallback: str = 'shakespeare_char') -> str:
    """Resolve dataset name from checkpoint metadata."""
    config = checkpoint.get('config') or {}
    dataset = config.get('dataset') or checkpoint.get('dataset')
    if not dataset:
        dataset = fallback
    return dataset


def load_meta(dataset_name: str) -> Dict:
    meta_path = Path('data') / dataset_name / 'meta.pkl'
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
    with meta_path.open('rb') as handle:
        meta = pickle.load(handle)
    return meta


def infer_field_names(meta: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Infer input/target tensor names from dataset metadata."""
    schema = meta.get('batch_schema') or []
    input_name: Optional[str] = None
    target_name: Optional[str] = None

    for field in schema:
        if not isinstance(field, dict):
            continue
        role = (field.get('role') or '').lower()
        name = field.get('name')
        if role == 'input' and input_name is None:
            input_name = name
        elif role == 'target' and target_name is None:
            target_name = name

    # Fallbacks by common naming conventions
    if input_name is None:
        for candidate in ['x', 'input', 'input_ids']:
            if candidate in meta.get('tensor_names', []):
                input_name = candidate
                break
    if target_name is None:
        for candidate in ['targets', 'y', 'target', 'labels']:
            if candidate in meta.get('tensor_names', []):
                target_name = candidate
                break

    return input_name, target_name


def decode_tokens(token_ids: Sequence[int], itos: Sequence[str], mask_token_id: int, pad_token_id: Optional[int]) -> str:
    pieces: List[str] = []
    for tid in token_ids:
        if tid == mask_token_id:
            pieces.append('[MASK]')
        elif pad_token_id is not None and tid == pad_token_id:
            pieces.append('[PAD]')
        elif 0 <= tid < len(itos):
            pieces.append(itos[tid])
        else:
            pieces.append('[UNK]')
    return ''.join(pieces)


def load_batch_samples(
    dataset_name: str,
    batch_file: Optional[str],
    *,
    input_name: Optional[str],
    target_name: Optional[str],
    limit: Optional[int] = None,
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    """Load original token sequences from a dataset batch file."""
    data_root = Path('data') / dataset_name / 'queue'
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset queue directory not found: {data_root}")

    if batch_file is None:
        candidates = sorted(data_root.rglob('*.pt'))
        if not candidates:
            raise FileNotFoundError(f"No .pt batch files found inside {data_root}")
        batch_path = candidates[0]
    else:
        batch_path = data_root / batch_file
        if not batch_path.exists():
            raise FileNotFoundError(f"Specified batch file not found: {batch_path}")

    container = torch.load(batch_path, map_location='cpu')
    if isinstance(container, dict) and 'batches' in container:
        batches = container.get('batches', [])
        if not batches:
            raise ValueError(f"Batch container {batch_path} contains no entries")
        entry = batches[0]
        tensors = entry.get('tensors', {})
    elif isinstance(container, dict):
        tensors = container.get('tensors', container)
    else:
        raise ValueError(f"Unsupported batch file format: {batch_path}")

    tensor_keys = list(tensors.keys())
    input_candidates = [input_name, 'x', 'input', 'input_ids']
    target_candidates = [target_name, 'targets', 'y', 'target', 'labels']

    input_key = next((cand for cand in input_candidates if cand and cand in tensors), None)
    if input_key is None:
        raise KeyError(f"Could not determine input tensor key from candidates {input_candidates} in {tensor_keys}")

    target_key = next((cand for cand in target_candidates if cand and cand in tensors), None)
    if target_key is None:
        raise KeyError(f"Could not determine target tensor key from candidates {target_candidates} in {tensor_keys}")

    x_tensor = tensors[input_key]
    y_tensor = tensors[target_key]
    if x_tensor.ndim != 2 or y_tensor.ndim != 2:
        raise ValueError('Evaluator expects MLM-style batches with 2D input/target tensors')
    if x_tensor.shape != y_tensor.shape:
        raise ValueError('Input and target tensors must share the same shape')

    ignore_index = -100
    originals: List[torch.Tensor] = []
    for idx in range(x_tensor.shape[0]):
        x_row = x_tensor[idx].clone()
        y_row = y_tensor[idx]
        mask = y_row != ignore_index
        x_row[mask] = y_row[mask]
        originals.append(x_row)
        if limit is not None and len(originals) >= limit:
            break

    return originals, {'input': x_tensor, 'target': y_tensor}


def build_sticky_samples(
    base_samples: Sequence[torch.Tensor],
    sticky_configs: Sequence[StickyConfig],
    *,
    mask_token_id: int,
    vocab_size: int,
    base_vocab_size: Optional[int],
    pad_token_id: Optional[int],
    max_per_base: Optional[int] = None,
    seed: int = 1234,
) -> List[EvaluationSample]:
    """Create sticky-masked evaluation samples from base sequences."""
    results: List[EvaluationSample] = []
    vocab_for_random = base_vocab_size if base_vocab_size is not None else vocab_size - 1
    rng_seed = seed

    for base_idx, original in enumerate(base_samples):
        seq_len = original.shape[0]
        if pad_token_id is not None:
            pad_positions = (original == pad_token_id).nonzero(as_tuple=True)[0]
            content_len = pad_positions[0].item() if pad_positions.numel() > 0 else seq_len
        else:
            content_len = seq_len

        if content_len == 0:
            continue

        content = original[:content_len].unsqueeze(0)
        for config_idx, sticky in enumerate(sticky_configs):
            if max_per_base is not None and config_idx >= max_per_base:
                break

            rng = torch.Generator(device='cpu')
            rng.manual_seed(rng_seed)
            rng_seed += 1

            masked_content, mask = apply_target_driven_sticky_masking_cpu(
                content,
                sticky.target_ratio,
                sticky.p1,
                sticky.p2,
                mask_token_id,
                vocab_for_random,
                rng,
            )

            mask_row = mask[0]
            if mask_row.sum().item() == 0:
                continue

            masked_full = original.clone()
            protected = torch.ones_like(original, dtype=torch.bool)
            mask_full = torch.zeros_like(original, dtype=torch.bool)

            masked_full[:content_len] = masked_content[0]
            mask_full[:content_len] = mask_row
            protected &= ~mask_full

            if pad_token_id is not None:
                pad_mask = original == pad_token_id
                protected |= pad_mask
                mask_full &= ~pad_mask

            sample_id = f"base{base_idx:03d}_cfg{config_idx:02d}"
            results.append(
                EvaluationSample(
                    sample_id=sample_id,
                    base_index=base_idx,
                    sticky=sticky,
                    original_tokens=original.clone(),
                    masked_tokens=masked_full,
                    protected_mask=protected,
                    mask_positions=mask_full,
                )
            )
    return results


def multinomial_warmup(
    model: GPT,
    tokens: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    iterations: int,
    temperature: float,
    mask_token_id: int,
    pad_token_id: Optional[int],
    base_vocab_size: Optional[int],
) -> torch.Tensor:
    """Iteratively refine tokens using multinomial sampling while preserving protected positions."""
    current = tokens.clone()
    for _ in range(iterations):
        dummy = torch.zeros_like(current)
        with torch.no_grad():
            logits, _ = model(current, targets=dummy)
        if temperature != 1.0:
            logits = logits / temperature
        logits[:, :, mask_token_id] = float('-inf')
        if pad_token_id is not None:
            logits[:, :, pad_token_id] = float('-inf')
        if base_vocab_size is not None and logits.shape[-1] > base_vocab_size:
            logits[:, :, base_vocab_size:] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        flat = probs.view(-1, probs.shape[-1])
        sampled = torch.multinomial(flat, num_samples=1).view_as(current)
        current = current.clone()
        current[~protected_mask] = sampled[~protected_mask]
    return current


def confidence_remasking(
    model: GPT,
    tokens: torch.Tensor,
    original_tokens: torch.Tensor,
    protected_mask: torch.Tensor,
    *,
    iterations: int,
    start_ratio: float,
    end_ratio: float,
    mask_token_id: int,
    pad_token_id: Optional[int],
    base_vocab_size: Optional[int],
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Run linear-schedule confidence-based remasking iterations."""
    current = tokens.clone()
    model_device = next(model.parameters()).device
    device_type = model_device.type
    for iteration in range(iterations):
        mask_ratio = linear_remasking_schedule(iteration, iterations, start_ratio, end_ratio)
        dummy = torch.zeros_like(current)
        with torch.no_grad():
            logits, _ = model(current, targets=dummy)
        probs = F.softmax(logits, dim=-1)
        taken = probs.gather(-1, current.unsqueeze(-1)).squeeze(-1)
        uncertainty = 1.0 - taken

        candidate_mask = ~protected_mask
        candidate_mask &= (current != mask_token_id)
        total_candidates = int(candidate_mask.sum().item())
        if total_candidates == 0:
            break

        k = max(1, int(math.ceil(total_candidates * mask_ratio)))
        flat_uncertainty = uncertainty.masked_fill(~candidate_mask, float('-inf')).view(-1)
        if (flat_uncertainty == float('-inf')).all():
            break
        top_values, top_indices = torch.topk(flat_uncertainty, k=min(k, total_candidates), largest=True)
        # Handle cases where additional -inf entries were selected
        valid_mask = top_values > float('-inf')
        if not valid_mask.any():
            break
        chosen_indices = top_indices[valid_mask]

        remask = torch.zeros_like(current, dtype=torch.bool)
        remask.view(-1)[chosen_indices] = True
        remask &= candidate_mask
        if not remask.any():
            continue

        remasked_tokens = current.clone()
        remasked_tokens[remask] = mask_token_id

        prediction_tokens, _ = predict_and_sample_tokens(
            model=model,
            tokens=remasked_tokens,
            mask_token_id=mask_token_id,
            temperature=temperature,
            top_p=top_p,
            vocab_size=model.config.vocab_size,
            device=device_type,
            return_logits=True,
            pad_token_id=pad_token_id,
            base_vocab_size=base_vocab_size,
        )
        current = prediction_tokens
        current[protected_mask] = original_tokens[protected_mask]

    return current


def evaluate_model(
    model: GPT,
    samples: Sequence[EvaluationSample],
    *,
    multinomial_iterations: int,
    multinomial_temperature: float,
    remask_iterations: int,
    remask_start: float,
    remask_end: float,
    temperature: float,
    top_p: float,
    mask_token_id: int,
    pad_token_id: Optional[int],
    base_vocab_size: Optional[int],
) -> Dict[str, SampleResult]:
    """Evaluate a model across samples and return per-sample results."""
    results: Dict[str, SampleResult] = {}
    model_device = next(model.parameters()).device

    for sample in samples:
        original = sample.original_tokens.unsqueeze(0).to(model_device)
        masked = sample.masked_tokens.unsqueeze(0).to(model_device)
        protected = sample.protected_mask.unsqueeze(0).to(model_device)

        stage1_input = masked.clone()
        stage1_input[protected] = original[protected]
        stage1_tokens = multinomial_warmup(
            model,
            stage1_input,
            protected,
            iterations=multinomial_iterations,
            temperature=multinomial_temperature,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            base_vocab_size=base_vocab_size,
        )
        stage1_tokens[protected] = original[protected]

        refined_tokens = confidence_remasking(
            model,
            stage1_tokens,
            original,
            protected,
            iterations=remask_iterations,
            start_ratio=remask_start,
            end_ratio=remask_end,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            base_vocab_size=base_vocab_size,
            temperature=temperature,
            top_p=top_p,
        )

        mask_positions = sample.mask_positions.unsqueeze(0).to(model_device)
        total = int(mask_positions.sum().item())
        if total == 0:
            score = float('nan')
            correct = 0
        else:
            correct = int((refined_tokens[mask_positions] == original[mask_positions]).sum().item())
            score = correct / total if total > 0 else float('nan')

        results[sample.sample_id] = SampleResult(
            model_name='unknown',
            score=score,
            correct=correct,
            total=total,
            final_tokens=refined_tokens.squeeze(0).cpu(),
            stage1_tokens=stage1_tokens.squeeze(0).cpu(),
        )
    return results


def summarize_results(
    all_results: Dict[str, Dict[str, SampleResult]],
    samples: Sequence[EvaluationSample],
    model_labels: Sequence[str],
) -> None:
    """Print a summary table of scores for each sample and model."""
    header = ['Sample', 'Target', 'p1', 'p2'] + list(model_labels)
    print('\n' + '-' * 80)
    print(' | '.join(f"{h:>12}" for h in header))
    print('-' * 80)
    for sample in samples:
        row = [
            sample.sample_id,
            f"{sample.sticky.target_ratio:.2f}",
            f"{sample.sticky.p1:.2f}",
            f"{sample.sticky.p2:.2f}",
        ]
        for label in model_labels:
            result = all_results.get(label, {}).get(sample.sample_id)
            if result is None or math.isnan(result.score):
                row.append('   n/a   ')
            else:
                row.append(f"{result.score:6.2%}")
        print(' | '.join(f"{cell:>12}" for cell in row))
    print('-' * 80)


def interactive_viewer(
    samples: Sequence[EvaluationSample],
    all_results: Dict[str, Dict[str, SampleResult]],
    *,
    model_labels: Sequence[str],
    itos: Sequence[str],
    mask_token_id: int,
    pad_token_id: Optional[int],
) -> None:
    """Simple interactive console viewer for per-sample outputs."""
    decode = lambda ids: decode_tokens(ids, itos, mask_token_id, pad_token_id)
    index_map = {sample.sample_id: idx for idx, sample in enumerate(samples)}
    sample_ids = [sample.sample_id for sample in samples]
    if not sample_ids:
        print('No samples available for interactive view.')
        return

    current = 0
    while True:
        sample = samples[current]
        print('\n' + '=' * 80)
        print(f"Sample {current + 1}/{len(samples)} :: {sample.sample_id}")
        print(f"  Sticky params -> target={sample.sticky.target_ratio:.3f}, p1={sample.sticky.p1:.3f}, p2={sample.sticky.p2:.3f}")
        print('-' * 80)
        print('Original :', repr(decode(sample.original_tokens.tolist())))
        print('Masked   :', repr(decode(sample.masked_tokens.tolist())))
        for label in model_labels:
            result = all_results.get(label, {}).get(sample.sample_id)
            if result is None:
                continue
            print('-' * 80)
            print(f"Model: {label}")
            if math.isnan(result.score):
                score_text = 'n/a'
            else:
                score_text = f"{result.score * 100:.2f}%"
            print(f"  Score : {score_text} ({result.correct}/{result.total})")
            print('  Stage1:', repr(decode(result.stage1_tokens.tolist())))
            print('  Final :', repr(decode(result.final_tokens.tolist())))
        print('=' * 80)
        cmd = input("[N]ext, [P]revious, [Q]uit, or enter sample id: ").strip().lower()
        if cmd == 'q':
            break
        elif cmd == 'n' or cmd == '':
            current = (current + 1) % len(samples)
        elif cmd == 'p':
            current = (current - 1) % len(samples)
        elif cmd in index_map:
            current = index_map[cmd]
        else:
            try:
                idx = int(cmd)
                if 1 <= idx <= len(samples):
                    current = idx - 1
            except ValueError:
                continue


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate multiple diffusion models on sticky-masked samples.')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model checkpoints (.pt/.pth)')
    parser.add_argument('--dataset-name', default=None, help='Dataset name (defaults to checkpoint metadata)')
    parser.add_argument('--batch-file', default=None, help='Relative path inside dataset queue directory')
    parser.add_argument('--num-base-samples', type=int, default=4, help='Number of base samples to draw from batch file')
    parser.add_argument('--p1-values', nargs='+', type=float, default=[0.1, 0.3], help='Sticky p1 probabilities')
    parser.add_argument('--p2-values', nargs='+', type=float, default=[0.5, 0.7], help='Sticky p2 probabilities')
    parser.add_argument('--target-ratios', nargs='+', type=float, default=[0.3, 0.5], help='Target mask ratios')
    parser.add_argument('--multinomial-iterations', type=int, default=3, help='Warmup multinomial iterations')
    parser.add_argument('--multinomial-temperature', type=float, default=0.8, help='Temperature for multinomial warmup')
    parser.add_argument('--remask-iterations', type=int, default=5, help='Number of confidence remasking iterations')
    parser.add_argument('--remask-start', type=float, default=0.6, help='Starting ratio for linear remasking schedule')
    parser.add_argument('--remask-end', type=float, default=0.1, help='Final ratio for linear remasking schedule')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature during remasking')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p for sampling during remasking')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive viewer after evaluation')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    model_paths = [Path(p) for p in args.models]
    if not model_paths:
        raise ValueError('No model checkpoints provided')

    loaded_models: Dict[str, GPT] = {}
    checkpoints: Dict[str, Dict] = {}

    dataset_name = args.dataset_name
    for path in model_paths:
        model, checkpoint = load_checkpoint_model(path)
        label = path.stem
        loaded_models[label] = model
        checkpoints[label] = checkpoint
        if dataset_name is None:
            dataset_name = resolve_dataset_name(checkpoint)

    if dataset_name is None:
        raise RuntimeError('Unable to determine dataset name; provide --dataset-name explicitly')

    meta = load_meta(dataset_name)
    stoi = meta.get('stoi')
    itos = meta.get('itos')
    if itos is None:
        raise ValueError('Vocabulary (itos) missing from dataset metadata; cannot decode tokens')
    vocab_size = meta.get('vocab_size', loaded_models[next(iter(loaded_models))].config.vocab_size)
    mask_token_id = meta.get('mask_token_id', vocab_size - 1)
    pad_token_id = meta.get('pad_token_id')
    base_vocab_size = meta.get('base_vocab_size')

    input_name, target_name = infer_field_names(meta)

    base_samples, _ = load_batch_samples(
        dataset_name,
        args.batch_file,
        input_name=input_name,
        target_name=target_name,
        limit=args.num_base_samples,
    )

    sticky_configs = [StickyConfig(tr, p1, p2) for tr, p1, p2 in itertools.product(args.target_ratios, args.p1_values, args.p2_values)]
    eval_samples = build_sticky_samples(
        base_samples,
        sticky_configs,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        base_vocab_size=base_vocab_size,
        pad_token_id=pad_token_id,
    )

    if not eval_samples:
        print('No evaluation samples generated; adjust sticky parameters or dataset.')
        return 1

    all_results: Dict[str, Dict[str, SampleResult]] = {}
    for label, model in loaded_models.items():
        sample_results = evaluate_model(
            model,
            eval_samples,
            multinomial_iterations=args.multinomial_iterations,
            multinomial_temperature=args.multinomial_temperature,
            remask_iterations=args.remask_iterations,
            remask_start=args.remask_start,
            remask_end=args.remask_end,
            temperature=args.temperature,
            top_p=args.top_p,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            base_vocab_size=base_vocab_size,
        )
        # Attach model label to results
        for res in sample_results.values():
            res.model_name = label
        all_results[label] = sample_results

    summarize_results(all_results, eval_samples, list(loaded_models.keys()))

    if args.interactive:
        interactive_viewer(
            eval_samples,
            all_results,
            model_labels=list(loaded_models.keys()),
            itos=itos,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
        )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
