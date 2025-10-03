"""
Simplified diffusion sampling script with critic-guided remasking and linear schedule.
- Supports LANGUAGE_MODEL checkpoints with critic head enabled (add_critic_head=True)
- Supports Judge and Node Quality metrics
- Supports two schedule modes:
  * 'ratio' (default): Uses linear schedule to decide how many tokens to re-mask; critic provides ordering
  * 'threshold': Masks all tokens with wrongness above threshold; finishes early if no tokens exceed threshold
- Supports seed_text placement as prefix or random
- Supports top-p sampling; no repetition penalty
- Supports batch generation and special tokens [MASK] and [PAD]
- Provides verbose logging similar to sample.py
- Command-line interface with reasonable defaults; running without args uses defaults; -h shows help
"""
import os
import time
import math
import pickle
import argparse
import json
from enum import Enum
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, ModelMode
from sample_utils import (
    linear_remasking_schedule,
    nucleus_sample,
    predict_and_sample_tokens,
    apply_remasking_step,
    calculate_judge_scores,
)


from core.common.timings import TimingAccumulator, print_global_summary
from timings_singleton import set_global_timer, get_global_timer

class SeedPlacement(Enum):
    PREFIX = 'prefix'
    RANDOM = 'random'


class QualityMetric(Enum):
    JUDGE = 'judge'
    NODE = 'node'  # node quality via critic head probabilities/statistics


def load_model_from_checkpoint(checkpoint_path: str, device: str, compile_model: bool = False):
    print(f"Loading model from {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_args' not in checkpoint:
        raise ValueError("Checkpoint missing 'model_args'. Please check checkpoint format.")
    model_args = checkpoint['model_args']

    if 'attention_type' not in model_args:
        model_args['attention_type'] = 'causal'
    if 'position_encoding' not in model_args:
        model_args['position_encoding'] = 'absolute'

    if 'init_from_checkpoint' in model_args and model_args.get('init_from_checkpoint'):
        print(f"Ignoring training-only init_from_checkpoint: {model_args['init_from_checkpoint']}")
        model_args['init_from_checkpoint'] = None

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    print("Model loaded successfully")
    print(f"  - Parameters: {model.get_num_params()/1e6:.1f}M")
    return model, checkpoint


def load_vocabulary(checkpoint, fallback_dataset='shakespeare_char'):
    dataset_name = None
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        if isinstance(cfg, dict):
            dataset_name = cfg.get('dataset')
        elif hasattr(cfg, 'get'):
            dataset_name = cfg.get('dataset')
    if not dataset_name:
        dataset_name = fallback_dataset
        print(f"Dataset name not found in checkpoint, using fallback: {dataset_name}")

    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Vocabulary file not found: {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    print(f"Vocabulary loaded from {dataset_name}: {vocab_size} tokens")
    return stoi, itos, vocab_size, dataset_name, meta


def diffusion_generate(
    model: GPT,
    batch_size: int,
    total_length: int,
    iterations: int,
    mask_token_id: int,
    vocab_size: int,
    base_vocab_size: int | None,
    pad_token_id: int | None,
    temperature: float,
    top_p: float,
    decode_fn,
    seed_ids=None,
    placement: SeedPlacement = SeedPlacement.PREFIX,
    start_ratio: float = 0.95,
    end_ratio: float = 0.05,
    masking_ratios: list[float] | None = None,
    randomness_strength: float = 0.0,
    verbose: bool = False,
    show_progress: bool = True,
    ctx=None,
    schedule_mode: str = 'ratio',
    save_iterations: bool = False,
):
    device = next(model.parameters()).device

    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), int(mask_token_id), dtype=torch.long, device=device)

    # Apply seed text (never to be masked) if provided
    protected_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    if seed_ids is not None and len(seed_ids) > 0:
        seed_len = min(len(seed_ids), total_length)
        if placement == SeedPlacement.RANDOM:
            max_start = total_length - seed_len
            start_idx = int(torch.randint(0, max(1, max_start + 1), (1,), device=device).item())
        else:
            start_idx = 0
        seed_tensor = torch.tensor(seed_ids[:seed_len], dtype=torch.long, device=device)
        tokens[:, start_idx:start_idx+seed_len] = seed_tensor.unsqueeze(0)
        protected_mask[:, start_idx:start_idx+seed_len] = True
        if verbose:
            print(f"DEBUG: Applied seed length: {seed_len} at index: {start_idx}")

    if verbose or show_progress:
        print("Starting diffusion generation:")
        print(f"  - Samples: {batch_size}")
        print(f"  - Length: {total_length}")
        print(f"  - Iterations: {iterations}")
        print(f"  - Temperature: {temperature}, Top-p: {top_p}")
        schedule_desc = "threshold-based" if schedule_mode == 'threshold' else "ratio-based"
        print(f"  - Using critic-guided remasking ({schedule_desc} schedule)")
        print("=" * 60)

    # Track min_wrongness from previous iteration for display
    prev_min_wrongness = None

    # Track iteration data for saving - organize by sample
    samples_data = [[] for _ in range(batch_size)] if save_iterations else None

    for iteration in range(iterations):
        if verbose or show_progress:
            masked_count = (tokens == mask_token_id).sum().item()
            total_tokens = tokens.numel()
            masked_ratio = masked_count / total_tokens

            # Calculate current ratio/threshold for display
            if masking_ratios is not None and iteration < len(masking_ratios):
                current_ratio = masking_ratios[iteration]
            else:
                from sample_utils import linear_remasking_schedule
                current_ratio = linear_remasking_schedule(iteration, iterations, start_ratio, end_ratio)

            if schedule_mode == 'threshold':
                # Threshold is inverted: ratio=0.95 means threshold=0.05
                current_threshold = 1.0 - current_ratio
                print(f"Iteration {iteration+1}/{iterations}: {masked_count}/{total_tokens} masked ({masked_ratio:.1%}), threshold={current_threshold:.3f}")
            else:
                # Ratio mode: show actual wrongness threshold from previous remasking
                if prev_min_wrongness is not None:
                    print(f"Iteration {iteration+1}/{iterations}: {masked_count}/{total_tokens} masked ({masked_ratio:.1%}), threshold={prev_min_wrongness:.3f}")
                else:
                    print(f"Iteration {iteration+1}/{iterations}: {masked_count}/{total_tokens} masked ({masked_ratio:.1%})")

            if iteration in (0, iterations - 1):
                sample_text = decode_fn(tokens[0])
                preview = sample_text[:100] + ('...' if len(sample_text) > 100 else '')
                print(f"  Sample: {preview}")

        # Step 1: Predict and sample tokens for masked positions
        _t = get_global_timer()
        _cm = _t.measure('predict_and_sample') if _t is not None else nullcontext()
        with _cm:
            pred_tokens, logits = predict_and_sample_tokens(
                model=model,
                tokens=tokens,
                mask_token_id=mask_token_id,
                temperature=temperature,
                top_p=top_p,
                vocab_size=vocab_size,
                device=device,
                verbose=verbose,
                return_logits=True,
                pad_token_id=pad_token_id,
                base_vocab_size=base_vocab_size,
            )

        # Step 2: Remask for next iteration (except last iteration)
        if iteration < iterations - 1:
            _t = get_global_timer()
            _cm = _t.measure('remask') if _t is not None else nullcontext()
            with _cm:
                remasked, min_wrongness, remasked_indices = apply_remasking_step(
                    tokens=tokens,
                    prediction_tokens=pred_tokens,
                    iteration=iteration,
                    iterations=iterations,
                    schedule_type='linear' if masking_ratios is None else 'custom',
                    masking_ratios=masking_ratios,
                    start_ratio=start_ratio,
                    end_ratio=end_ratio,
                    remasking_model=None,
                    randomness_strength=randomness_strength,
                    mask_token_id=mask_token_id,
                    device=device,
                    base_model=model,  # critic ordering
                    intelligent_remasking=False,
                    verbose=verbose,
                    logits_from_predict=logits,
                    protected_mask=protected_mask,
                    schedule_mode=schedule_mode,
                )

            # Check for early termination in threshold mode
            if remasked is None:
                if verbose or show_progress:
                    print(f"  Early termination: no tokens exceed threshold")
                if save_iterations:
                    # Save final iteration data for all samples
                    tokens_cpu = tokens.cpu().tolist()
                    pred_tokens_cpu = pred_tokens.cpu().tolist()
                    for sample_idx in range(batch_size):
                        samples_data[sample_idx].append({
                            'iteration': iteration + 1,
                            'input_masked': tokens_cpu[sample_idx],
                            'output_unmasked': pred_tokens_cpu[sample_idx],
                            'remasked_indices': []
                        })
                tokens = pred_tokens
                break

            if save_iterations:
                # Organize remasked_indices by sample
                tokens_cpu = tokens.cpu().tolist()
                pred_tokens_cpu = pred_tokens.cpu().tolist()
                remasked_by_sample = [[] for _ in range(batch_size)]
                for batch_idx, pos in remasked_indices:
                    remasked_by_sample[batch_idx].append(pos)

                for sample_idx in range(batch_size):
                    samples_data[sample_idx].append({
                        'iteration': iteration + 1,
                        'input_masked': tokens_cpu[sample_idx],
                        'output_unmasked': pred_tokens_cpu[sample_idx],
                        'remasked_indices': remasked_by_sample[sample_idx]
                    })

            tokens = remasked
            prev_min_wrongness = min_wrongness
        else:
            # Last iteration: no remasking
            if save_iterations:
                tokens_cpu = tokens.cpu().tolist()
                pred_tokens_cpu = pred_tokens.cpu().tolist()
                for sample_idx in range(batch_size):
                    samples_data[sample_idx].append({
                        'iteration': iteration + 1,
                        'input_masked': tokens_cpu[sample_idx],
                        'output_unmasked': pred_tokens_cpu[sample_idx],
                        'remasked_indices': []
                    })
            tokens = pred_tokens

    if save_iterations:
        return tokens, samples_data
    return tokens


def build_decode_functions(itos, mask_token_id: int, pad_token_id: int | None):
    def decode(token_ids):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        out = []
        for tid in token_ids:
            if tid == mask_token_id:
                out.append('[MASK]')
            elif pad_token_id is not None and tid == pad_token_id:
                out.append('[PAD]')
            elif 0 <= tid < len(itos):
                out.append(itos[tid])
            else:
                out.append('[UNK]')
        return ''.join(out)
    return decode


def main():
    parser = argparse.ArgumentParser(description='Simplified diffusion sampler (critic-guided, linear schedule)')
    # Model and paths
    parser.add_argument('--out-dir', type=str, default='out-char-diffusion', help='Output/checkpoints directory')
    parser.add_argument('--checkpoint-name', type=str, default='a40_3750_01_10.pt', help='Main model checkpoint file name')
    parser.add_argument('--judge-checkpoint-name', type=str, default='padded_judge_0.0155.pt', help='Judge model checkpoint (required when --quality-metric=judge)')

    # Generation params
    parser.add_argument('--num-samples', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--sequence-length', type=int, default=1024, help='Total generated sequence length')
    parser.add_argument('--iterations', type=int, default=15, help='Number of diffusion iterations')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=1.0, help='Nucleus top-p (1.0 to disable)')
    parser.add_argument('--schedule-mode', type=str, choices=['ratio', 'threshold'], default='ratio',
                        help='Schedule mode: ratio (mask fixed percentage, default) or threshold (mask all above threshold)')
    parser.add_argument('--start-ratio', type=float, default=0.95, help='Initial mask ratio/threshold for linear schedule')
    parser.add_argument('--end-ratio', type=float, default=0.05, help='Final mask ratio/threshold for linear schedule')
    parser.add_argument('--masking-ratios', type=str, default=None, help='Comma-separated custom ratios/thresholds (overrides linear schedule)')
    parser.add_argument('--randomness-strength', type=float, default=0.0, help='Blend randomization into critic ordering [0-1]')

    # Seed text and placement
    parser.add_argument('--seed-text', type=str, default='Be or not to be, that is the question.', help='Seed text to place into sequence')
    parser.add_argument('--seed-placement', type=str, choices=[m.value for m in SeedPlacement], default=SeedPlacement.RANDOM.value, help='Seed placement mode')

    # Device and misc
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'float16'], default='float16', help='AMP dtype')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed (-1 for random)')
    parser.add_argument('--quality-metric', type=str, choices=[m.value for m in QualityMetric], default=QualityMetric.JUDGE.value, help='Quality metric to compute')
    parser.add_argument('--verbose', action='store_true', help='Verbose iteration logging')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress logs')
    parser.add_argument('--save', type=str, default=None, help='Save iteration data to JSON file (e.g., output.json)')

    args = parser.parse_args()

    # Seed
    seed = args.seed if args.seed != -1 else int.from_bytes(os.urandom(4), byteorder='little')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Device/context
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    # Initialize and register global timing accumulator for this run
    _global_timer = TimingAccumulator()
    if device_type == 'cuda':
        _global_timer.set_cuda_sync(True)
    set_global_timer(_global_timer)

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load main model
    main_ckpt = os.path.join(args.out_dir, args.checkpoint_name)
    model, checkpoint = load_model_from_checkpoint(main_ckpt, args.device, compile_model=False)
    if getattr(model.config, 'mode', None) != ModelMode.LANGUAGE_MODEL:
        raise ValueError('This script supports LANGUAGE_MODEL checkpoints only')
    if not getattr(model.config, 'add_critic_head', False):
        raise ValueError('Model must have critic head enabled (add_critic_head=True)')

    # Load vocabulary/meta
    stoi, itos, meta_vocab_size, dataset_name, meta = load_vocabulary(checkpoint)
    meta_path = os.path.join('data', dataset_name, 'meta.pkl')
    vocab_size = int(model.config.vocab_size)
    mask_token_id = meta.get('mask_token_id', vocab_size - 1)
    pad_token_id = meta.get('pad_token_id', None)
    base_vocab_size = meta.get('base_vocab_size', None)

    print(f"Using mask_token_id: {mask_token_id} ({itos[mask_token_id] if mask_token_id < len(itos) else '[MASK]'})")
    if pad_token_id is not None and pad_token_id < len(itos):
        print(f"Using pad_token_id: {pad_token_id} ({itos[pad_token_id]})")
    print(f"Model vocab_size: {vocab_size} | Data vocab_size: {meta_vocab_size}")

    # Tokenize seed text
    seed_ids = [stoi[c] for c in args.seed_text if c in stoi] if args.seed_text else []
    if args.verbose and seed_ids:
        print(f"Seed text configured: length {len(seed_ids)} | placement: {args.seed_placement}")

    # Build decode function
    decode = build_decode_functions(itos, mask_token_id, pad_token_id)

    # Load judge if requested
    judge_model = None
    metric = QualityMetric(args.quality_metric)
    if metric == QualityMetric.JUDGE:
        judge_ckpt = os.path.join(args.out_dir, args.judge_checkpoint_name)
        if not os.path.exists(judge_ckpt):
            raise FileNotFoundError(f"Judge checkpoint not found: {judge_ckpt}")
        judge_model, _ = load_model_from_checkpoint(judge_ckpt, args.device, compile_model=False)
        if getattr(judge_model.config, 'mode', None) != ModelMode.SEQUENCE_SCORER:
            raise ValueError('Judge model must be configured with mode=SEQUENCE_SCORER')

    # Parse masking ratios if provided
    masking_ratios = None
    if args.masking_ratios:
        parts = [p.strip() for p in args.masking_ratios.split(',') if p.strip()]
        masking_ratios = [float(p) for p in parts]
        print(f"Using custom masking ratios with {len(masking_ratios)} steps (overrides linear schedule)")
        iterations = len(masking_ratios)
    else:
        iterations = args.iterations

    # Settings
    print("\n" + "="*60)
    print("GENERATION SETTINGS (simplified)")
    print("="*60)
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device} ({args.dtype})")
    print(f"Seed: {seed}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Iterations: {iterations}")
    schedule_type_str = 'custom' if masking_ratios is not None else 'linear'
    print(f"Schedule: {schedule_type_str} {args.schedule_mode} ({args.start_ratio:.1%} -> {args.end_ratio:.1%})")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Quality metric: {metric.value}")
    print("="*60)

    start_time = time.time()
    with torch.no_grad():
        with ctx:
            result = diffusion_generate(
                model=model,
                batch_size=args.num_samples,
                total_length=args.sequence_length,
                iterations=iterations,
                mask_token_id=int(mask_token_id),
                vocab_size=int(vocab_size),
                base_vocab_size=None if base_vocab_size is None else int(base_vocab_size),
                pad_token_id=None if pad_token_id is None else int(pad_token_id),
                temperature=args.temperature,
                top_p=args.top_p,
                decode_fn=decode,
                seed_ids=seed_ids,
                placement=SeedPlacement(args.seed_placement),
                start_ratio=args.start_ratio,
                end_ratio=args.end_ratio,
                masking_ratios=masking_ratios,
                randomness_strength=args.randomness_strength,
                verbose=args.verbose,
                show_progress=not args.no_progress,
                ctx=ctx,
                schedule_mode=args.schedule_mode,
                save_iterations=args.save is not None,
            )

            if args.save is not None:
                generated_tokens, iteration_data = result
            else:
                generated_tokens = result

    generation_time = time.time() - start_time

    # Save iteration data if requested
    if args.save is not None:
        print(f"\nSaving iteration data to {args.save}...")
        # Convert samples_data (list of lists) to proper structure
        samples_output = []
        for sample_idx, sample_iterations in enumerate(iteration_data):
            samples_output.append({
                'sample_id': sample_idx,
                'iterations': sample_iterations
            })

        output_data = {
            'generator': main_ckpt,
            'meta': meta_path,
            'samples': samples_output
        }
        with open(args.save, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(samples_output)} samples with {len(samples_output[0]['iterations']) if samples_output else 0} iterations each")

    # Quality metrics
    judge_scores = None
    node_stats = None
    critic_percentiles = None
    if metric == QualityMetric.JUDGE:
        print("\nEvaluating quality with Judge model...")
        _t = get_global_timer()
        _cm = _t.measure('judge_eval') if _t is not None else nullcontext()
        with _cm:
            judge_scores = calculate_judge_scores(judge_model=judge_model, tokens=generated_tokens, device=args.device, ctx=ctx)
        # Also compute critic percentiles to display, matching sample.py behavior
        if getattr(model.config, 'add_critic_head', False):
            with torch.no_grad():
                if ctx is not None:
                    with ctx:
                        _critic_scores = model.critic_scores(generated_tokens)
                else:
                    _critic_scores = model.critic_scores(generated_tokens)
            _valid = (generated_tokens != mask_token_id)
            if pad_token_id is not None:
                _valid = _valid & (generated_tokens != pad_token_id)
            _q = torch.tensor([0.1, 0.5, 0.9], device=generated_tokens.device, dtype=torch.float32)
            _per_list = []
            _prob = torch.sigmoid(_critic_scores.float())
            for b in range(generated_tokens.size(0)):
                _s = _prob[b][_valid[b]]
                if _s.numel() == 0:
                    _per_list.append(torch.tensor([float('nan'), float('nan'), float('nan')], device='cpu'))
                else:
                    _per_list.append(torch.quantile(_s, _q).cpu())
            critic_percentiles = torch.stack(_per_list, dim=0)
    elif metric == QualityMetric.NODE:
        print("\nComputing node (critic) quality statistics...")
        with torch.no_grad():
            if ctx is not None:
                with ctx:
                    critic_logits = model.critic_scores(generated_tokens)
            else:
                critic_logits = model.critic_scores(generated_tokens)
        valid = (generated_tokens != mask_token_id)
        if pad_token_id is not None:
            valid = valid & (generated_tokens != pad_token_id)
        probs = torch.sigmoid(critic_logits.float())
        # Compute mean and percentiles per sample
        q = torch.tensor([0.1, 0.5, 0.9], device=probs.device, dtype=torch.float32)
        per_list = []
        for b in range(probs.size(0)):
            s = probs[b][valid[b]]
            if s.numel() == 0:
                per_list.append(torch.tensor([float('nan'), float('nan'), float('nan'), float('nan')], device='cpu'))
            else:
                p10, p50, p90 = torch.quantile(s, q).cpu()
                mean_v = s.mean().cpu()
                per_list.append(torch.stack([mean_v, p10, p50, p90], dim=0))
        node_stats = torch.stack(per_list, dim=0)

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Generation completed in {generation_time:.2f} seconds")

    for i in range(args.num_samples):
        print(f"\n{'─'*40}")
        print(f"SAMPLE {i+1}/{args.num_samples}")
        if metric == QualityMetric.JUDGE and judge_scores is not None:
            score = float(judge_scores[i].item())

            print(f"Quality score (Judge): {score:.4f}")
            if 'critic_percentiles' in locals() and critic_percentiles is not None:
                p10, p50, p90 = [float(x) for x in critic_percentiles[i].tolist()]
                print(f"Critic probabilities p10/median/p90: {p10:.4f} / {p50:.4f} / {p90:.4f}")
        elif metric == QualityMetric.NODE and node_stats is not None:
            mean_v, p10, p50, p90 = [float(x) for x in node_stats[i].tolist()]
            print(f"Critic prob stats: mean={mean_v:.4f} p10/median/p90={p10:.4f}/{p50:.4f}/{p90:.4f}")
        print(f"{'─'*40}")
        sample_text = decode(generated_tokens[i])
        print(sample_text)

    total_tokens = args.num_samples * args.sequence_length
    tokens_per_sec = (total_tokens / generation_time) if generation_time > 0 else float('inf')
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print(f"Total wall time: {generation_time:.2f} s | Time per sample: {generation_time/max(1,args.num_samples):.2f} s")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s (tokens: {total_tokens})")

    if judge_scores is not None:
        print(f"Judge score mean: {judge_scores.mean().item():.4f} | min: {judge_scores.min().item():.4f} | max: {judge_scores.max().item():.4f}")
    # Detailed operation timing summary (counts and total time per category)
    from core.common.timings import print_global_hierarchical_summary as _print_h
    _print_h(title="OPERATION TIMING (hierarchical)", show_counts=True)


    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

