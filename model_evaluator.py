#!/usr/bin/env python3
"""Interactive sticky-mask model evaluator.

This script mirrors the console workflow of ``interactive_diffusion_explorer.py``.
Run ``python model_evaluator.py`` and follow the menu prompts to:

* Pick checkpoints interactively from ``out-char-diffusion`` (or another
  directory).
* Configure the sticky masking grid and generation parameters.
* Build a shared pool of sticky masked samples.
* Evaluate every selected model with a multinomial warm-up followed by
  confidence-guided remasking.
* Inspect per-model generations in a console viewer and compare scores.

The evaluation procedure is:
1. Generate sticky samples with varying ``p1``/``p2`` and mask ratios.
2. Mark every originally unmasked position as protected (never re-mask).
3. Perform multinomial warm-up iterations without altering protected tokens.
4. Run a linear schedule of confidence-based remasking iterations.
5. Compare each model's final output against the original targets across the
   initially masked positions to compute match ratios.
"""

from __future__ import annotations

import os
import sys
import textwrap
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from contextlib import nullcontext

from model import GPT, GPTConfig, ModelMode
from sample_utils import predict_and_sample_tokens, apply_remasking_step

# Sticky masking utilities live with the dataset helpers
sys.path.append('data/char_diffusion')
from masking_utils import apply_target_driven_sticky_masking_cpu  # noqa: E402


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StickyConfig:
    target_ratio: float
    p1: float
    p2: float

    def format_brief(self) -> str:
        return f"ratio={self.target_ratio:.2f} | p1={self.p1:.2f} | p2={self.p2:.2f}"


@dataclass
class EvaluationSample:
    sample_id: str
    base_index: int
    sticky: StickyConfig
    original_tokens: torch.Tensor  # (1, seq_len)
    masked_tokens: torch.Tensor  # (1, seq_len)
    protected_mask: torch.Tensor  # (1, seq_len) bool
    mask_positions: torch.Tensor  # (1, seq_len) bool


@dataclass
class SampleResult:
    model_name: str
    score: float
    correct: int
    total: int
    warmup_tokens: torch.Tensor  # (1, seq_len)
    final_tokens: torch.Tensor  # (1, seq_len)


@dataclass
class ModelInfo:
    path: Path
    name: str
    checkpoint: Optional[Dict] = None
    model: Optional[GPT] = None
    stoi: Optional[Dict[str, int]] = None
    itos: Optional[List[str]] = None
    vocab_size: Optional[int] = None
    mask_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    base_vocab_size: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_text: Optional[str] = None
    block_size: Optional[int] = None
    meta: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def clear_screen() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title: str) -> None:
    clear_screen()
    print("=" * 80)
    print(f" {title.center(76)} ")
    print("=" * 80)
    print()


def wait_for_key(prompt: str = "Press Enter to continue...") -> None:
    try:
        input(prompt)
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def list_checkpoints(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    paths = [p for p in directory.iterdir() if p.suffix in {'.pt', '.pth'} and p.is_file()]
    paths.sort()
    return paths


def resolve_dataset_name(checkpoint: Dict, fallback: str = 'shakespeare_char') -> str:
    config = checkpoint.get('config') or {}
    dataset = config.get('dataset') or checkpoint.get('dataset')
    return dataset or fallback


def load_checkpoint(path: Path, device: str) -> Tuple[GPT, Dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if 'model_args' not in checkpoint:
        raise ValueError(f"Checkpoint missing 'model_args': {path}")

    model_args = checkpoint['model_args'].copy()
    if model_args.get('init_from_checkpoint'):
        model_args['init_from_checkpoint'] = None

    deprecated = {'mode', 'num_token_classes', 'binary_classification', 'attention_type', 'position_encoding'}
    filtered_args = {k: v for k, v in model_args.items() if k not in deprecated}

    config = GPTConfig(**filtered_args)
    model = GPT(config)

    original_mode = model_args.get('mode')
    if original_mode:
        if original_mode in (ModelMode.SEQUENCE_SCORER, 'sequence_scorer'):
            model.set_mode(ModelMode.SEQUENCE_SCORER)
        else:
            model.set_mode(ModelMode.LANGUAGE_MODEL)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model, checkpoint


def load_meta(dataset_name: str, checkpoint: Dict) -> Dict:
    meta_path = Path('data') / dataset_name / 'meta.pkl'
    if meta_path.exists():
        with meta_path.open('rb') as handle:
            return pickle.load(handle)
    meta_from_checkpoint = checkpoint.get('meta')
    if meta_from_checkpoint:
        return meta_from_checkpoint
    raise FileNotFoundError(
        f"Could not locate meta.pkl for dataset '{dataset_name}'. "
        "Run prepare.py to generate dataset metadata."
    )


def decode_tokens(tokens: torch.Tensor, itos: Sequence[str], mask_token_id: int,
                  pad_token_id: Optional[int] = None, mask_char: str = '□') -> str:
    if tokens.ndim == 2 and tokens.size(0) == 1:
        tokens = tokens[0]
    pieces: List[str] = []
    for token in tokens.tolist():
        if token == mask_token_id:
            pieces.append(mask_char)
        elif pad_token_id is not None and token == pad_token_id:
            pieces.append('[PAD]')
        elif 0 <= token < len(itos):
            pieces.append(itos[token])
        else:
            pieces.append('[UNK]')
    return ''.join(pieces)


def encode_segment(segment: str, stoi: Dict[str, int]) -> torch.Tensor:
    ids: List[int] = []
    for ch in segment:
        if ch not in stoi:
            continue
        ids.append(int(stoi[ch]))
    if not ids:
        raise ValueError("Segment contains no encodable characters")
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def ensure_text(dataset_name: str, info: ModelInfo) -> str:
    if info.dataset_text:
        return info.dataset_text
    folder = Path('data') / dataset_name
    for candidate in ['input.txt', 'train.txt', 'text.txt']:
        file_path = folder / candidate
        if file_path.exists():
            info.dataset_text = file_path.read_text(encoding='utf-8')
            return info.dataset_text
    raise FileNotFoundError(
        f"Unable to locate text corpus for dataset '{dataset_name}'. Place an input.txt in data/{dataset_name}."
    )
# ---------------------------------------------------------------------------
# Core evaluator class
# ---------------------------------------------------------------------------


class ModelEvaluatorApp:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ptdtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(
            device_type=self.device, dtype=self.ptdtype
        )

        self.model_dir = Path('out-char-diffusion')
        self.model_cache: Dict[Path, ModelInfo] = {}
        self.selected_models: List[ModelInfo] = []

        self.sample_settings: Dict[str, float] = {
            'num_base_samples': 3,
            'sequence_length': 256,
            'iterations': 6,
            'temperature': 0.8,
            'top_p': 1.0,
            'start_ratio': 0.60,
            'end_ratio': 0.05,
            'randomness_strength': 0.10,
            'seed': float(torch.randint(0, 2**31 - 1, (1,)).item()),
        }

        self.sticky_grid: List[StickyConfig] = [
            StickyConfig(0.30, 0.05, 0.50),
            StickyConfig(0.40, 0.08, 0.60),
            StickyConfig(0.50, 0.10, 0.70),
        ]

        self.samples: List[EvaluationSample] = []
        self.results: Dict[str, Dict[str, SampleResult]] = {}

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def available_checkpoints(self) -> List[Path]:
        return list_checkpoints(self.model_dir)

    def ensure_checkpoint(self, info: ModelInfo) -> None:
        if info.checkpoint is None:
            _, checkpoint = load_checkpoint(info.path, device='cpu')
            info.checkpoint = checkpoint

    def ensure_metadata(self, info: ModelInfo) -> None:
        if info.stoi is not None and info.itos is not None:
            return
        self.ensure_checkpoint(info)
        assert info.checkpoint is not None
        dataset_name = resolve_dataset_name(info.checkpoint)
        meta = load_meta(dataset_name, info.checkpoint)
        stoi = meta.get('stoi')
        itos = meta.get('itos')
        if stoi is None or itos is None:
            raise ValueError(f"Metadata for dataset '{dataset_name}' is missing stoi/itos mappings")
        vocab_size = int(meta.get('vocab_size', len(itos)))
        mask_token_id = int(meta.get('mask_token_id', vocab_size - 1))
        pad_token_id = meta.get('pad_token_id')
        base_vocab_size = meta.get('base_vocab_size')
        block_size = meta.get('block_size') or info.checkpoint['model_args'].get('block_size')

        info.meta = meta
        info.dataset_name = dataset_name
        info.stoi = stoi
        info.itos = itos
        info.vocab_size = vocab_size
        info.mask_token_id = mask_token_id
        info.pad_token_id = int(pad_token_id) if pad_token_id is not None else None
        info.base_vocab_size = int(base_vocab_size) if base_vocab_size is not None else None
        info.block_size = int(block_size) if block_size is not None else None

    def ensure_model_loaded(self, info: ModelInfo) -> GPT:
        if info.model is not None:
            return info.model
        model, checkpoint = load_checkpoint(info.path, self.device)
        info.model = model
        info.checkpoint = checkpoint
        self.ensure_metadata(info)
        return model

    # ------------------------------------------------------------------
    # Sample configuration menus
    # ------------------------------------------------------------------

    def configure_sample_settings(self) -> None:
        while True:
            print_header("Configure Sample Settings")
            print("Current settings:\n")
            print(f"1. Base samples       : {int(self.sample_settings['num_base_samples'])}")
            print(f"2. Sequence length    : {int(self.sample_settings['sequence_length'])}")
            print(f"3. Iterations         : {int(self.sample_settings['iterations'])}")
            print(f"4. Temperature        : {self.sample_settings['temperature']:.2f}")
            print(f"5. Top-p              : {self.sample_settings['top_p']:.2f}")
            print(f"6. Start ratio        : {self.sample_settings['start_ratio']:.2f}")
            print(f"7. End ratio          : {self.sample_settings['end_ratio']:.2f}")
            print(f"8. Randomness         : {self.sample_settings['randomness_strength']:.2f}")
            print(f"9. RNG seed           : {int(self.sample_settings['seed'])}")
            print()
            print("Enter a number to edit, or press Enter to return.")
            choice = input("Selection: ").strip()
            if not choice:
                return
            if not choice.isdigit():
                wait_for_key("Please enter a valid option.")
                continue
            idx = int(choice)
            keys = [
                'num_base_samples',
                'sequence_length',
                'iterations',
                'temperature',
                'top_p',
                'start_ratio',
                'end_ratio',
                'randomness_strength',
                'seed',
            ]
            if idx < 1 or idx > len(keys):
                wait_for_key("Option out of range.")
                continue
            key = keys[idx - 1]
            current = self.sample_settings[key]
            new_value = input(f"New value for {key.replace('_', ' ')} (current {current}): ").strip()
            if not new_value:
                continue
            try:
                if key in {'num_base_samples', 'sequence_length', 'iterations', 'seed'}:
                    self.sample_settings[key] = max(0, int(new_value))
                else:
                    self.sample_settings[key] = float(new_value)
            except ValueError:
                wait_for_key("Invalid numeric input.")
                continue
            if key == 'start_ratio' and self.sample_settings['start_ratio'] < self.sample_settings['end_ratio']:
                self.sample_settings['start_ratio'] = self.sample_settings['end_ratio']
            wait_for_key("Updated.")

    def configure_sticky_grid(self) -> None:
        while True:
            print_header("Configure Sticky Masking Grid")
            if not self.sticky_grid:
                print("No sticky configurations defined yet.\n")
            else:
                print("Current configurations:\n")
                for idx, cfg in enumerate(self.sticky_grid, 1):
                    print(f" {idx}. {cfg.format_brief()}")
                print()
            print("Options:\n  1. Add configuration\n  2. Remove configuration\n  0. Return")
            choice = input("Selection: ").strip()
            if choice in {'0', ''}:
                return
            if choice == '1':
                try:
                    ratio = float(input("Target mask ratio (0-1): ").strip())
                    p1 = float(input("p1 probability (0-1): ").strip())
                    p2 = float(input("p2 probability (0-1): ").strip())
                except ValueError:
                    wait_for_key("Invalid numeric input.")
                    continue
                self.sticky_grid.append(StickyConfig(ratio, p1, p2))
                wait_for_key("Configuration added.")
            elif choice == '2':
                if not self.sticky_grid:
                    wait_for_key("Nothing to remove.")
                    continue
                idx_str = input("Enter the number to remove: ").strip()
                if not idx_str.isdigit():
                    wait_for_key("Please enter a valid index.")
                    continue
                idx = int(idx_str)
                if idx < 1 or idx > len(self.sticky_grid):
                    wait_for_key("Index out of range.")
                    continue
                del self.sticky_grid[idx - 1]
                wait_for_key("Removed.")
            else:
                wait_for_key("Unknown option.")

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def build_samples(self) -> None:
        if not self.selected_models:
            wait_for_key("Select at least one model first.")
            return

        primary = self.selected_models[0]
        self.ensure_metadata(primary)
        assert primary.stoi is not None
        assert primary.itos is not None
        assert primary.mask_token_id is not None
        assert primary.vocab_size is not None

        dataset_text = ensure_text(primary.dataset_name or 'dataset', primary)
        seq_len = int(self.sample_settings['sequence_length'])
        if primary.block_size is not None:
            seq_len = min(seq_len, primary.block_size)
        if seq_len <= 0:
            wait_for_key("Sequence length must be positive.")
            return
        if len(dataset_text) < seq_len:
            wait_for_key("Dataset text shorter than requested sequence length.")
            return
        if not self.sticky_grid:
            wait_for_key("Define at least one sticky configuration.")
            return

        num_base = max(1, int(self.sample_settings['num_base_samples']))
        rng = torch.Generator()
        rng.manual_seed(int(self.sample_settings['seed']))

        self.samples = []
        self.results.clear()

        for base_idx in range(num_base):
            start_idx = int(torch.randint(0, len(dataset_text) - seq_len, (1,), generator=rng).item())
            segment = dataset_text[start_idx:start_idx + seq_len]
            try:
                base_tokens = encode_segment(segment, primary.stoi)
            except ValueError:
                continue
            if base_tokens.size(1) < seq_len:
                pad_value = base_tokens[0, -1].item()
                pad_needed = seq_len - base_tokens.size(1)
                padding = torch.full((1, pad_needed), pad_value, dtype=torch.long)
                base_tokens = torch.cat([base_tokens, padding], dim=1)
            elif base_tokens.size(1) > seq_len:
                base_tokens = base_tokens[:, :seq_len]

            for cfg_idx, sticky in enumerate(self.sticky_grid):
                batch_tokens = base_tokens.clone()
                masked, mask = apply_target_driven_sticky_masking_cpu(
                    batch_tokens,
                    float(sticky.target_ratio),
                    float(sticky.p1),
                    float(sticky.p2),
                    int(primary.mask_token_id),
                    int(primary.vocab_size),
                    rng,
                )
                masked = masked.clone()
                masked[mask] = int(primary.mask_token_id)

                protected_mask = (~mask).to(torch.bool)

                sample_id = f"S{base_idx + 1:02d}-C{cfg_idx + 1:02d}"
                sample = EvaluationSample(
                    sample_id=sample_id,
                    base_index=base_idx,
                    sticky=sticky,
                    original_tokens=batch_tokens.clone(),
                    masked_tokens=masked.to(torch.long),
                    protected_mask=protected_mask.to(torch.bool),
                    mask_positions=mask.to(torch.bool),
                )
                self.samples.append(sample)

        wait_for_key(f"Built {len(self.samples)} evaluation samples.")
    # ------------------------------------------------------------------
    # Evaluation internals
    # ------------------------------------------------------------------

    def multinomial_warmup(self, info: ModelInfo, sample: EvaluationSample) -> torch.Tensor:
        assert info.model is not None
        assert info.mask_token_id is not None
        tokens = sample.masked_tokens.to(self.device)
        protected = sample.protected_mask.to(self.device)
        original = sample.original_tokens.to(self.device)

        iterations = max(0, int(self.sample_settings['warmup_iterations']))
        temperature = float(self.sample_settings['temperature'])
        top_p = float(self.sample_settings['top_p'])

        current = tokens.clone()
        for _ in range(iterations):
            prediction_tokens, _ = predict_and_sample_tokens(
                model=info.model,
                tokens=current,
                mask_token_id=int(info.mask_token_id),
                temperature=temperature,
                top_p=top_p,
                vocab_size=info.vocab_size,
                device=self.device,
                return_logits=True,
                pad_token_id=info.pad_token_id,
                base_vocab_size=info.base_vocab_size,
            )
            if protected.any():
                prediction_tokens = torch.where(protected, original, prediction_tokens)
            current = prediction_tokens
        return current

    def confidence_refine(self, info: ModelInfo, sample: EvaluationSample, initial_tokens: torch.Tensor) -> torch.Tensor:
        assert info.model is not None
        assert info.mask_token_id is not None
        iterations = max(0, int(self.sample_settings['remask_iterations']))
        if iterations == 0:
            return initial_tokens

        temperature = float(self.sample_settings['temperature'])
        top_p = float(self.sample_settings['top_p'])
        start_ratio = float(self.sample_settings['start_ratio'])
        end_ratio = float(self.sample_settings['end_ratio'])
        randomness = float(self.sample_settings['randomness_strength'])

        protected = sample.protected_mask.to(self.device)
        original = sample.original_tokens.to(self.device)

        current = initial_tokens.clone()
        for iteration in range(iterations):
            prediction_tokens, logits = predict_and_sample_tokens(
                model=info.model,
                tokens=current,
                mask_token_id=int(info.mask_token_id),
                temperature=temperature,
                top_p=top_p,
                vocab_size=info.vocab_size,
                device=self.device,
                return_logits=True,
                pad_token_id=info.pad_token_id,
                base_vocab_size=info.base_vocab_size,
            )
            if protected.any():
                prediction_tokens = torch.where(protected, original, prediction_tokens)

            current = prediction_tokens
            if iteration >= iterations - 1:
                break

            remask_result = apply_remasking_step(
                tokens=current,
                prediction_tokens=prediction_tokens,
                iteration=iteration,
                iterations=iterations,
                schedule_type='linear',
                masking_ratios=None,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                remasking_model=None,
                randomness_strength=randomness,
                mask_token_id=int(info.mask_token_id),
                device=self.device,
                base_model=info.model,
                intelligent_remasking=True,
                verbose=False,
                logits_from_predict=logits,
                protected_mask=protected,
            )
            if isinstance(remask_result, tuple):
                remasked_tokens = remask_result[0]
                if remasked_tokens is None:
                    break
                current = remasked_tokens
            elif isinstance(remask_result, torch.Tensor):
                current = remask_result
        return current

    def evaluate(self) -> None:
        if not self.samples:
            wait_for_key("Generate samples first.")
            return
        if not self.selected_models:
            wait_for_key("Select models before evaluating.")
            return

        print_header("Evaluating Models")
        print("Loading models...")
        for info in self.selected_models:
            self.ensure_model_loaded(info)
        print("Models loaded. Beginning evaluation...\n")
        wait_for_key("Press Enter to start.")

        self.results = {sample.sample_id: {} for sample in self.samples}

        for sample in self.samples:
            print_header(f"Evaluating {sample.sample_id}")
            for info in self.selected_models:
                assert info.model is not None
                assert info.mask_token_id is not None

                with torch.no_grad():
                    assert info.model is not None and info.mask_token_id is not None
                    current = sample.masked_tokens.to(self.device)
                    protected = sample.protected_mask.to(self.device)
                    original = sample.original_tokens.to(self.device)

                    iterations = max(1, int(self.sample_settings['iterations']))
                    temperature = float(self.sample_settings['temperature'])
                    top_p = float(self.sample_settings['top_p'])
                    start_ratio = float(self.sample_settings['start_ratio'])
                    end_ratio = float(self.sample_settings['end_ratio'])
                    randomness = float(self.sample_settings['randomness_strength'])

                    warmup_tokens = None
                    for it in range(iterations):
                        prediction_tokens, logits = predict_and_sample_tokens(
                            model=info.model,
                            tokens=current,
                            mask_token_id=int(info.mask_token_id),
                            temperature=temperature,
                            top_p=top_p,
                            vocab_size=info.vocab_size,
                            device=self.device,
                            return_logits=True,
                            pad_token_id=info.pad_token_id,
                            base_vocab_size=info.base_vocab_size,
                        )
                        if protected.any():
                            prediction_tokens = torch.where(protected, original, prediction_tokens)
                        if it == 0:
                            warmup_tokens = prediction_tokens.clone()
                        # if not last iteration, prepare next current by remasking
                        if it < iterations - 1:
                            remask_result = apply_remasking_step(
                                tokens=current,
                                prediction_tokens=prediction_tokens,
                                iteration=it,
                                iterations=iterations,
                                schedule_type='linear',
                                masking_ratios=None,
                                start_ratio=start_ratio,
                                end_ratio=end_ratio,
                                remasking_model=None,
                                randomness_strength=randomness,
                                mask_token_id=int(info.mask_token_id),
                                device=self.device,
                                base_model=info.model,
                                intelligent_remasking=True,
                                verbose=False,
                                logits_from_predict=logits,
                                protected_mask=protected,
                            )
                            if isinstance(remask_result, tuple):
                                nxt = remask_result[0]
                                current = prediction_tokens if nxt is None else nxt
                            elif isinstance(remask_result, torch.Tensor):
                                current = remask_result
                            else:
                                current = prediction_tokens
                        else:
                            current = prediction_tokens

                    refined_tokens = current

                final_cpu = refined_tokens.detach().cpu()
                warmup_cpu = warmup_tokens.detach().cpu() if warmup_tokens is not None else refined_tokens.detach().cpu()

                mask_positions = sample.mask_positions
                original = sample.original_tokens
                correct = int((final_cpu == original)[mask_positions].sum().item())
                total = int(mask_positions.sum().item())
                score = float(correct / total) if total > 0 else 0.0

                self.results[sample.sample_id][info.name] = SampleResult(
                    model_name=info.name,
                    score=score,
                    correct=correct,
                    total=total,
                    warmup_tokens=warmup_cpu,
                    final_tokens=final_cpu,
                )
            print(f"Completed {sample.sample_id}.")
            wait_for_key("Continue to next sample...")
        wait_for_key("Evaluation finished.")


    def view_samples(self) -> None:
        if not self.samples:
            wait_for_key("No samples available. Generate them first.")
            return
        primary = self.selected_models[0] if self.selected_models else None
        if primary is None or primary.itos is None or primary.mask_token_id is None:
            wait_for_key("Select at least one model to enable decoding.")
            return
        while True:
            print_header("Sample Browser")
            has_results = bool(self.results)
            primary_name = primary.name if primary is not None else None
            for idx, sample in enumerate(self.samples, 1):
                cfg = sample.sticky
                line = f" {idx}. {sample.sample_id} | {cfg.format_brief()}"
                if has_results and sample.sample_id in self.results and primary_name in self.results[sample.sample_id]:
                    res = self.results[sample.sample_id][primary_name]
                    # Compute Stage1 score from warmup_tokens
                    wt = res.warmup_tokens
                    ot = sample.original_tokens
                    mp = sample.mask_positions
                    try:
                        s1_correct = int((wt == ot)[mp].sum().item())
                        s1_total = int(mp.sum().item())
                        s1_score = float(s1_correct / s1_total) if s1_total > 0 else 0.0
                        line += f" | s1={s1_score:.2f} | final={res.score:.2f}"
                    except Exception:
                        # Be conservative; if shapes misalign, skip scores in browser list
                        pass
                print(line)
            print("\nEnter sample number to inspect, or press Enter to return.")
            choice = input("Selection: ").strip()
            if not choice:
                return
            if not choice.isdigit():
                wait_for_key("Please enter a valid index.")
                continue
            idx = int(choice)
            if idx < 1 or idx > len(self.samples):
                wait_for_key("Index out of range.")
                continue
            self.show_sample_detail(self.samples[idx - 1], primary)

    def show_sample_detail(self, sample: EvaluationSample, primary: ModelInfo) -> None:
        assert primary.itos is not None
        assert primary.mask_token_id is not None
        print_header(f"Sample {sample.sample_id}")
        print(f"Sticky config: {sample.sticky.format_brief()}")
        print(f"Masked tokens: {int(sample.mask_positions.sum().item())}\n")




        original_text = decode_tokens(
            sample.original_tokens, primary.itos, primary.mask_token_id, primary.pad_token_id, mask_char='□'
        )
        masked_text = decode_tokens(
            sample.masked_tokens, primary.itos, primary.mask_token_id, primary.pad_token_id, mask_char='■'
        )

        print("Original snippet:\n")
        print(textwrap.fill(original_text, width=78))
        print("\nSticky-masked snippet:\n")
        print(textwrap.fill(masked_text, width=78))

        if sample.sample_id in self.results:
            print("\nModel results:\n")
            per_sample = self.results[sample.sample_id]
            # Local helper to decode single token id
            def _decode_piece(tid: int) -> str:
                if tid == primary.mask_token_id:
                    return '□'
                if primary.pad_token_id is not None and tid == primary.pad_token_id:
                    return '[PAD]'
                if 0 <= tid < len(primary.itos):
                    return primary.itos[tid]
                return '[UNK]'
            orig_ids = sample.original_tokens.squeeze(0).tolist()
            mask_flags = sample.mask_positions.squeeze(0).tolist()
            for model_name, result in per_sample.items():
                stage1_ids = result.warmup_tokens.squeeze(0).tolist()
                final_ids = result.final_tokens.squeeze(0).tolist()
                # Build colored Stage1 string (masked-only): green correct, orange incorrect
                s1_parts = []
                for i, tid in enumerate(stage1_ids):
                    piece = _decode_piece(tid)
                    if i < len(mask_flags) and mask_flags[i]:
                        if i < len(orig_ids) and tid == orig_ids[i]:
                            s1_parts.append('\x1b[32m' + piece + '\x1b[0m')  # green
                        else:
                            s1_parts.append('\x1b[38;5;208m' + piece + '\x1b[0m')  # orange
                    else:
                        s1_parts.append(piece)
                stage1_colored = ''.join(s1_parts)
                # Build colored Final string with improvement/regression coding
                f_parts = []
                for i, tid in enumerate(final_ids):
                    piece = _decode_piece(tid)
                    if i < len(mask_flags) and mask_flags[i]:
                        is_final_correct = (i < len(orig_ids) and tid == orig_ids[i])
                        was_stage1_correct = (i < len(stage1_ids) and stage1_ids[i] == orig_ids[i])
                        if is_final_correct:
                            f_parts.append(('\x1b[32m' if was_stage1_correct else '\x1b[33m') + piece + '\x1b[0m')
                        else:
                            f_parts.append(('\x1b[31m' if was_stage1_correct else '\x1b[38;5;208m') + piece + '\x1b[0m')
                    else:
                        f_parts.append(piece)
                final_colored = ''.join(f_parts)
                # Stage1 score from warmup_tokens
                s1_correct = int((result.warmup_tokens == sample.original_tokens)[sample.mask_positions].sum().item())
                s1_total = int(sample.mask_positions.sum().item())
                s1_score = float(s1_correct / s1_total) if s1_total > 0 else 0.0
                print(f"{model_name} — score {result.score:.3f} ({result.correct}/{result.total}), s1 {s1_score:.3f}")
                print("  Stage1:")
                for _line in stage1_colored.splitlines():
                    print('    ' + _line)
                print("  Final :")
                for _line in final_colored.splitlines():
                    print('    ' + _line)
                print()
        wait_for_key()

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_models(self) -> None:
        while True:
            checkpoints = self.available_checkpoints()
            print_header("Select Models")
            print(f"Model directory: {self.model_dir}\n")
            if not checkpoints:
                print("No checkpoints found.\n")
                print("Options:\n  1. Change directory\n  0. Return")
                choice = input("Selection: ").strip()
                if choice == '1':
                    self.change_model_directory()
                    continue
                return

            for idx, path in enumerate(checkpoints, 1):
                print(f" {idx}. {path.name}")
            print("\nEnter numbers separated by commas to select models.")
            print("Type 'a' for all, 'd' to change directory, or press Enter to cancel.")
            raw = input("Selection: ").strip()
            if not raw:
                return
            if raw.lower() == 'd':
                self.change_model_directory()
                continue
            if raw.lower() == 'a':
                indices = list(range(1, len(checkpoints) + 1))
            else:
                parts = [p.strip() for p in raw.split(',') if p.strip()]
                indices = []
                valid = True
                for part in parts:
                    if part.isdigit():
                        indices.append(int(part))
                    else:
                        valid = False
                        break
                if not valid:
                    wait_for_key("Invalid selection format.")
                    continue
            selected: List[ModelInfo] = []
            for idx in indices:
                if idx < 1 or idx > len(checkpoints):
                    continue
                path = checkpoints[idx - 1]
                if path not in self.model_cache:
                    self.model_cache[path] = ModelInfo(path=path, name=path.name)
                selected.append(self.model_cache[path])
            if not selected:
                wait_for_key("No valid selections.")
                continue
            self.selected_models = selected
            wait_for_key(f"Selected {len(self.selected_models)} model(s).")
            return

    def change_model_directory(self) -> None:
        new_dir = input("Enter new model directory path: ").strip()
        if not new_dir:
            return
        candidate = Path(new_dir).expanduser()
        if not candidate.exists() or not candidate.is_dir():
            wait_for_key("Directory not found.")
            return
        self.model_dir = candidate
        wait_for_key(f"Model directory updated to {self.model_dir}.")

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------

    def main_menu(self) -> None:
        while True:
            print_header("Sticky Mask Model Evaluator")
            if self.selected_models:
                model_list = ', '.join(info.name for info in self.selected_models)
                print(f"Selected models: {model_list}\n")
            else:
                print("No models selected yet.\n")

            print("Main Menu:")
            print(" 1. Select models")
            print(" 2. Configure sample settings")
            print(" 3. Configure sticky masking grid")
            print(" 4. Generate samples")
            print(" 5. View samples")
            print(" 6. Run evaluation")
            print(" 7. Change model directory")
            print(" 0. Exit")

            choice = input("Selection: ").strip()
            if choice == '1':
                self.select_models()
            elif choice == '2':
                self.configure_sample_settings()
            elif choice == '3':
                self.configure_sticky_grid()
            elif choice == '4':
                self.build_samples()
            elif choice == '5':
                self.view_samples()
            elif choice == '6':
                self.evaluate()
            elif choice == '7':
                self.change_model_directory()
            elif choice == '0':
                print_header("Goodbye!")
                print("Thank you for using the Sticky Mask Model Evaluator.")
                wait_for_key()
                return
            else:
                wait_for_key("Unknown option.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = ModelEvaluatorApp()
    try:
        app.main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
    except Exception as exc:  # pragma: no cover - interactive tool
        print(f"\nUnexpected error: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
