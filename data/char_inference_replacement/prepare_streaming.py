"""Streaming provider that replaces random corruption with model predictions."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch

from data.char_diffusion.prepare_streaming import CharDiffusionProvider
from data.char_random_replacement.corruption_utils import (
    RandomReplacementCorruptor,
    apply_mixed_corruption,
    build_candidate_token_ids,
)

from .predictive_corruptor import CheckpointPredictionCorruptor


class CharInferenceReplacementProvider(CharDiffusionProvider):
    """Character dataset that fills masked tokens using checkpoint predictions."""

    def __init__(
        self,
        *args,
        checkpoint_dir: Optional[str] = None,
        inference_device: Optional[str] = None,
        inference_dtype: Optional[torch.dtype | str] = None,
        inference_refresh_seconds: float = 30.0,
        prediction_temperature: float = 1.0,
        fallback_to_random: bool = True,
        fallback_original_token_probability_multiplier: float = 1.0,
        fallback_extra_special_token_ids: Optional[Iterable[int]] = None,
        dataset_partial_targets: bool = True,
        train_corruption_mixture: Optional[Iterable[float]] = None,
        **kwargs,
    ) -> None:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided for inference-based corruption")

        self._train_corruption_mixture = self._coerce_train_corruption_mixture(train_corruption_mixture)
        self._dataset_partial_targets = bool(dataset_partial_targets)
        self._checkpoint_dir = os.path.abspath(checkpoint_dir)
        self._inference_device = inference_device
        self._inference_dtype = inference_dtype
        self._inference_refresh_seconds = float(inference_refresh_seconds)
        self._prediction_temperature = float(prediction_temperature)
        self._fallback_to_random = bool(fallback_to_random)
        self._fallback_original_multiplier = float(
            fallback_original_token_probability_multiplier
        )
        self._fallback_extra_special_token_ids = [int(token) for token in (fallback_extra_special_token_ids or [])]
        self._active_corruption_split: Optional[str] = None
        self._mask_ratio_tracker: Dict[str, Dict[str, list[float]]] = {
            "train": self._new_tracker(),
            "val": self._new_tracker(),
        }

        super().__init__(*args, **kwargs)
        self._initialize_corruptors()

    # ------------------------------------------------------------------ #
    # Corruption hooks

    def _initialize_corruptors(self) -> None:
        excluded_ids = set(self._fallback_extra_special_token_ids)
        excluded_ids.add(self.mask_token_id)
        candidate_ids = build_candidate_token_ids(self.vocab_size, excluded_token_ids=excluded_ids)
        self._random_corruptor = RandomReplacementCorruptor(
            candidate_ids,
            original_token_probability_multiplier=self._fallback_original_multiplier,
        )
        fallback: Optional[RandomReplacementCorruptor] = self._random_corruptor if self._fallback_to_random else None

        self._fragment_samplers = {
            "train": self._build_fragment_sampler("train"),
            "val": self._build_fragment_sampler("val"),
        }
        self._corruptor = CheckpointPredictionCorruptor(
            checkpoint_dir=self._checkpoint_dir,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            device=self._inference_device,
            dtype=self._inference_dtype,
            refresh_seconds=self._inference_refresh_seconds,
            temperature=self._prediction_temperature,
            fallback_corruptor=fallback,
            verbose=self.verbose,
        )

    def sample_batch(self, split: str, rng):
        self._active_corruption_split = split
        try:
            return super().sample_batch(split, rng)
        finally:
            self._active_corruption_split = None

    def _build_fragment_sampler(self, split: str) -> Callable[[int, torch.Generator], torch.Tensor]:
        if split == "train":
            line_indices = self._train_line_start_indices
            ids_tensor = self.train_ids_tensor
        elif split == "val":
            line_indices = self._val_line_start_indices
            ids_tensor = self.val_ids_tensor
        else:
            line_indices = self._train_line_start_indices
            ids_tensor = self.train_ids_tensor

        def sampler(batch_size: int, sampler_rng: torch.Generator) -> torch.Tensor:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive for fragment sampling")
            start_indices = self._sample_start_positions(
                line_indices, batch_size, sampler_rng
            )
            return self._gather_sequences(ids_tensor, start_indices)

        return sampler

    def _apply_prediction_corruption(
        self, x: torch.Tensor, mask: torch.Tensor, rng
    ) -> torch.Tensor:
        split = self._active_corruption_split or "train"
        fragment_sampler = self._fragment_samplers.get(split)
        if fragment_sampler is None:
            fragment_sampler = self._fragment_samplers["train"]
        mask_bool = mask.to(dtype=torch.bool)
        if not mask_bool.any():
            return x.clone()

        mixture = self._train_corruption_mixture
        if len(mixture) != 4:
            corrupted, random_mask, mask_token_mask, fragment_mask = apply_mixed_corruption(
                x,
                mask,
                rng,
                random_corruptor=self._random_corruptor,
                mask_token_id=self.mask_token_id,
                fragment_sampler=fragment_sampler,
                mixture_weights=mixture[:3],
                return_masks=True,
            )
            prediction_mask = mask_bool & torch.zeros_like(mask_bool)
        else:
            prediction_weight, random_weight, mask_weight, fragment_weight = mixture
            total_weight = float(prediction_weight + random_weight + mask_weight + fragment_weight)
            if total_weight <= 0:
                raise ValueError("train_corruption_mixture weights must sum to a positive value")
            norm_pred = prediction_weight / total_weight
            norm_random = random_weight / total_weight
            norm_mask = mask_weight / total_weight

            selection = torch.rand(mask.shape, generator=rng, device=x.device, dtype=torch.float32)

            prediction_mask = mask_bool & (selection < norm_pred)
            random_mask = mask_bool & (selection >= norm_pred) & (selection < norm_pred + norm_random)
            mask_token_mask = mask_bool & (
                selection >= norm_pred + norm_random
            ) & (selection < norm_pred + norm_random + norm_mask)
            fragment_mask = mask_bool & ~(prediction_mask | random_mask | mask_token_mask)

            corrupted = x.clone()
            if prediction_mask.any():
                prediction_corrupted = self._corruptor.corrupt(x, prediction_mask, rng)
                corrupted[prediction_mask] = prediction_corrupted[prediction_mask]
            if random_mask.any():
                random_corrupted = self._random_corruptor.corrupt(x, random_mask, rng)
                corrupted[random_mask] = random_corrupted[random_mask]
            if mask_token_mask.any():
                corrupted[mask_token_mask] = self.mask_token_id
            if fragment_mask.any():
                fragments = fragment_sampler(x.shape[0], rng)
                if fragments.shape != x.shape:
                    raise ValueError(
                        "Fragment sampler returned tensor with shape "
                        f"{fragments.shape}, expected {x.shape}."
                    )
                corrupted[fragment_mask] = fragments[fragment_mask]

        total_positions = x.numel()
        if total_positions > 0:
            mask_float = mask.to(dtype=torch.float32)
            mismatch_mask = torch.ne(corrupted, x)
            mismatch_float = mismatch_mask.to(dtype=torch.float32)
            initial_ratio = mask_float.sum().item() / float(total_positions)
            mismatch_ratio = mismatch_float.sum().item() / float(total_positions)
            tracker = self._mask_ratio_tracker.setdefault(split, self._new_tracker())
            tracker["initial_total"].append(initial_ratio)
            tracker["final_total"].append(mismatch_ratio)

            def _store(type_mask: torch.Tensor, total_key: str, frac_key: str) -> None:
                count = type_mask.to(dtype=torch.float32).sum().item()
                tracker[total_key].append(count / float(total_positions))
                if count > 0:
                    changed = mismatch_float[type_mask].sum().item()
                    tracker[frac_key].append(changed / count)

            _store(prediction_mask, "prediction_total", "prediction_changed_fraction")
            _store(random_mask, "random_total", "random_changed_fraction")
            _store(mask_token_mask, "mask_total", "mask_changed_fraction")
            _store(fragment_mask, "fragment_total", "fragment_changed_fraction")
        return corrupted

    def _apply_corruption(self, x: torch.Tensor, mask: torch.Tensor, rng):
        return self._apply_prediction_corruption(x, mask, rng)

    def _apply_stage_corruption(
        self,
        stage_corrupted_x: torch.Tensor,
        original_x: torch.Tensor,
        mask: torch.Tensor,
        rng,
    ) -> torch.Tensor:
        del stage_corrupted_x
        return self._apply_prediction_corruption(original_x, mask, rng)

    def _create_labels(
        self, original_x: torch.Tensor, mask: torch.Tensor, split: str
    ) -> torch.Tensor:
        if split == "train" and not self._dataset_partial_targets:
            return original_x.clone()
        return super()._create_labels(original_x, mask, split)

    # ------------------------------------------------------------------ #
    # Metadata

    @staticmethod
    def _coerce_train_corruption_mixture(
        mixture: Optional[Iterable[float]],
    ) -> Tuple[float, float, float, float]:
        default = (0.2, 0.2, 0.2, 0.4)
        if mixture is None:
            return default
        try:
            mixture_tuple = tuple(float(weight) for weight in mixture)
        except TypeError as exc:
            raise TypeError(
                "train_corruption_mixture must be an iterable of four numeric weights"
            ) from exc
        if len(mixture_tuple) != 4:
            raise ValueError("train_corruption_mixture must contain exactly four weights")
        if any(weight < 0 for weight in mixture_tuple):
            raise ValueError("train_corruption_mixture weights must be non-negative")
        if sum(mixture_tuple) == 0:
            raise ValueError("train_corruption_mixture weights must sum to a positive value")
        return mixture_tuple

    def build_meta(self) -> Dict[str, Any]:  # type: ignore[override]
        meta = super().build_meta()
        meta.update(
            {
                "dataset_name": "char_inference_replacement",
                "corruption": {
                    "type": "checkpoint_prediction",
                    "train_corruption_mixture": self._train_corruption_mixture,
                    "checkpoint_dir": self._checkpoint_dir,
                    "checkpoint_selection": "latest",
                    "prediction_temperature": self._prediction_temperature,
                    "fallback_to_random": self._fallback_to_random,
                },
                "train_partial_targets": self._dataset_partial_targets,
            }
        )
        return meta

    # ------------------------------------------------------------------ #
    # File-level reporting hooks

    @staticmethod
    def _new_tracker() -> Dict[str, list[float]]:
        return {
            "initial_total": [],
            "final_total": [],
            "prediction_total": [],
            "prediction_changed_fraction": [],
            "random_total": [],
            "random_changed_fraction": [],
            "mask_total": [],
            "mask_changed_fraction": [],
            "fragment_total": [],
            "fragment_changed_fraction": [],
        }

    def _on_file_start(self, split: str) -> None:
        self._mask_ratio_tracker[split] = self._new_tracker()

    def _file_report_lines(self, split: str):
        tracker = self._mask_ratio_tracker.get(split)
        if not tracker or not tracker["initial_total"]:
            self._mask_ratio_tracker[split] = self._new_tracker()
            return ()
        initial_values = tracker["initial_total"]
        final_values = tracker["final_total"] if tracker["final_total"] else [0.0]

        def _format_range(values: list[float]) -> str:
            if not values:
                return "n/a"
            return f"{min(values):.4f}/{max(values):.4f}"

        def _format_fraction(values: list[float]) -> str:
            if not values:
                return "n/a"
            return f"{min(values):.4f}/{max(values):.4f}"

        prediction_totals = tracker["prediction_total"]
        random_totals = tracker["random_total"]
        mask_totals = tracker["mask_total"]
        fragment_totals = tracker["fragment_total"]

        line = (
            "[provider] mask ratios: "
            f"initial min/max={_format_range(initial_values)}, "
            f"final min/max={_format_range(final_values)}, "
            f"prediction total={_format_range(prediction_totals)}, "
            f"random total={_format_range(random_totals)}, "
            f"mask total={_format_range(mask_totals)}, "
            f"fragment total={_format_range(fragment_totals)}, "
            f"prediction change frac={_format_fraction(tracker['prediction_changed_fraction'])}, "
            f"random change frac={_format_fraction(tracker['random_changed_fraction'])}, "
            f"mask change frac={_format_fraction(tracker['mask_changed_fraction'])}, "
            f"fragment change frac={_format_fraction(tracker['fragment_changed_fraction'])}"
        )

        self._mask_ratio_tracker[split] = self._new_tracker()
        return (line,)


def _load_stage_kwargs(composition_config: Optional[str]) -> Dict[str, Any]:
    if not composition_config:
        return {}

    this_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(this_dir, "config", f"{composition_config}.py"),
        os.path.join(
            os.path.dirname(this_dir),
            "char_random_replacement",
            "config",
            f"{composition_config}.py",
        ),
    ]

    config_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if config_path is None:
        print(
            "Warning: composition config file not found for "
            f"{composition_config} in {candidate_paths}"
        )
        return {}

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"{composition_config}_config", config_path
    )
    config_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(config_module)

    stage_kwargs: Dict[str, Any] = {}
    for attr_name in [
        "use_all_stages_for_training",
        "unmasking_stages",
        "validation_stages",
    ]:
        if hasattr(config_module, attr_name):
            stage_kwargs[attr_name] = getattr(config_module, attr_name)
    print(f"Loaded composition config from {config_path}")
    return stage_kwargs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming char dataset with checkpoint-driven corruption"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--mask_probability", type=float, default=0.15)
    parser.add_argument("--batches_per_file", type=int, default=100)
    parser.add_argument("--max_backlog_files", type=int, default=2)
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--composition_config",
        type=str,
        default=None,
        help='Name of composition config to load (e.g., "complex")',
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoints to use for corruption.",
    )
    parser.add_argument(
        "--inference_device",
        type=str,
        default=None,
        help="Device string for inference (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--inference_dtype",
        type=str,
        default=None,
        help="Optional dtype for inference (float32, float16, bfloat16).",
    )
    parser.add_argument(
        "--inference_refresh_seconds",
        type=float,
        default=30.0,
        help="How often to check for new checkpoints.",
    )
    parser.add_argument(
        "--prediction_temperature",
        type=float,
        default=1.0,
        help="Temperature applied to logits before argmax prediction.",
    )
    parser.add_argument(
        "--fallback_to_random",
        action="store_true",
        help="Enable random replacement fallback when checkpoints are unavailable.",
    )
    parser.add_argument(
        "--no-fallback_to_random",
        dest="fallback_to_random",
        action="store_false",
        help="Disable random replacement fallback.",
    )
    parser.set_defaults(fallback_to_random=True)
    parser.add_argument(
        "--fallback_original_token_probability_multiplier",
        type=float,
        default=1.0,
        help="Original token bias for the fallback random corruptor.",
    )
    parser.add_argument(
        "--fallback_extra_special_token_ids",
        type=int,
        nargs="*",
        default=None,
        help="Token IDs to exclude from fallback random replacement.",
    )
    parser.add_argument(
        "--train_corruption_mixture",
        type=float,
        nargs=3,
        default=(0.8, 0.2, 0.0),
        metavar=("PRED", "MASK", "FRAGMENT"),
        help=(
            "Mixture weights for (prediction replacement, mask token, fragment copy) "
            "when corrupting training batches."
        ),
    )
    parser.add_argument(
        "--dataset_partial_targets",
        action="store_true",
        dest="dataset_partial_targets",
        help="Keep partial targets (mask positions only) for the training split.",
    )
    parser.add_argument(
        "--no-dataset_partial_targets",
        action="store_false",
        dest="dataset_partial_targets",
        help="Use full identity targets for the training split.",
    )
    parser.set_defaults(dataset_partial_targets=True)

    args = parser.parse_args()

    stage_kwargs = _load_stage_kwargs(args.composition_config)

    data_dir = os.path.dirname(__file__)
    input_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(input_path):
        fallback_dir = os.path.join(os.path.dirname(__file__), "..", "char_diffusion")
        data_dir = os.path.abspath(fallback_dir)

    provider = CharInferenceReplacementProvider(
        data_dir=data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        mask_probability=args.mask_probability,
        batches_per_file=args.batches_per_file,
        max_backlog_files=args.max_backlog_files,
        sleep_seconds=args.sleep_seconds,
        seed=args.seed,
        verbose=args.verbose,
        checkpoint_dir=args.checkpoint_dir,
        inference_device=args.inference_device,
        inference_dtype=args.inference_dtype,
        inference_refresh_seconds=args.inference_refresh_seconds,
        prediction_temperature=args.prediction_temperature,
        fallback_to_random=args.fallback_to_random,
        fallback_original_token_probability_multiplier=(
            args.fallback_original_token_probability_multiplier
        ),
        fallback_extra_special_token_ids=args.fallback_extra_special_token_ids,
        train_corruption_mixture=tuple(args.train_corruption_mixture),
        dataset_partial_targets=args.dataset_partial_targets,
        **stage_kwargs,
    )
    provider.run()


# Explicit provider alias for prepare.py discovery
Provider = CharInferenceReplacementProvider


if __name__ == "__main__":
    main()
