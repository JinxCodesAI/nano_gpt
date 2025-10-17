"""Streaming provider with random replacement corruption."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, Optional

import torch

from data.char_diffusion.prepare_streaming import CharDiffusionProvider

from .corruption_utils import (
    RandomReplacementCorruptor,
    apply_mixed_corruption,
    build_candidate_token_ids,
)


class CharRandomReplacementProvider(CharDiffusionProvider):
    """Character dataset that replaces masked tokens with random characters."""

    def __init__(
        self,
        *args,
        original_token_probability_multiplier: float = 1.0,
        extra_special_token_ids: Optional[Iterable[int]] = None,
        dataset_partial_targets: bool = True,
        **kwargs,
    ) -> None:
        self._train_corruption_mixture = (0.8, 0.2, 0.0"""Problems to make it work""")
        self._original_multiplier = original_token_probability_multiplier
        self._extra_special_token_ids = [int(token) for token in (extra_special_token_ids or [])]
        self._dataset_partial_targets = bool(dataset_partial_targets)
        self._active_corruption_split: Optional[str] = None
        super().__init__(*args, **kwargs)
        self._initialize_corruptor()

    # ------------------------------------------------------------------
    # Corruption hooks
    # ------------------------------------------------------------------
    def _initialize_corruptor(self) -> None:
        excluded_ids = set(self._extra_special_token_ids)
        excluded_ids.add(self.mask_token_id)
        candidate_ids = build_candidate_token_ids(
            self.vocab_size, excluded_token_ids=excluded_ids
        )
        self._corruptor = RandomReplacementCorruptor(
            candidate_ids,
            original_token_probability_multiplier=self._original_multiplier,
        )
        self._fragment_sampler = self._build_fragment_sampler()

    def sample_batch(self, split: str, rng):
        self._active_corruption_split = split
        try:
            return super().sample_batch(split, rng)
        finally:
            self._active_corruption_split = None

    def _build_fragment_sampler(self) -> Callable[[int, torch.Generator], torch.Tensor]:
        def sampler(batch_size: int, sampler_rng: torch.Generator) -> torch.Tensor:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive for fragment sampling")
            start_indices = self._sample_start_positions(
                self._train_line_start_indices, batch_size, sampler_rng
            )
            return self._gather_sequences(self.train_ids_tensor, start_indices)

        return sampler

    def _apply_train_corruption(
        self, x: torch.Tensor, mask: torch.Tensor, rng
    ) -> torch.Tensor:
        return apply_mixed_corruption(
            x,
            mask,
            rng,
            random_corruptor=self._corruptor,
            mask_token_id=self.mask_token_id,
            fragment_sampler=self._fragment_sampler,
            mixture_weights=self._train_corruption_mixture,
        )

    def _apply_corruption(self, x: torch.Tensor, mask: torch.Tensor, rng):
        if self._active_corruption_split == "train":
            return self._apply_train_corruption(x, mask, rng)
        return self._corruptor.corrupt(x, mask, rng)

    def _apply_stage_corruption(
        self,
        stage_corrupted_x: torch.Tensor,
        original_x: torch.Tensor,
        mask: torch.Tensor,
        rng,
    ) -> torch.Tensor:
        del stage_corrupted_x  # already encoded; we only need the mask
        if self._active_corruption_split == "train":
            return self._apply_train_corruption(original_x, mask, rng)
        return self._corruptor.corrupt(original_x, mask, rng)

    def _create_labels(
        self, original_x: torch.Tensor, mask: torch.Tensor, split: str
    ) -> torch.Tensor:
        if split == "train" and not self._dataset_partial_targets:
            return original_x.clone()
        return super()._create_labels(original_x, mask, split)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def build_meta(self) -> Dict[str, Any]:  # type: ignore[override]
        meta = super().build_meta()
        meta.update(
            {
                "dataset_name": "char_random_replacement",
                "corruption": {
                    "type": "random_replacement",
                    "original_token_probability_multiplier": self._original_multiplier,
                    "extra_special_token_ids": self._extra_special_token_ids,
                },
                "train_partial_targets": self._dataset_partial_targets,
            }
        )
        return meta


def _load_stage_kwargs(composition_config: Optional[str]) -> Dict[str, Any]:
    if not composition_config:
        return {}

    config_path = os.path.join(
        os.path.dirname(__file__), "config", f"{composition_config}.py"
    )
    if not os.path.exists(config_path):
        print(f"Warning: composition config file not found at {config_path}")
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
        description="Streaming char dataset with random replacement corruption"
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
        "--original_token_probability_multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--extra_special_token_ids",
        type=int,
        nargs="*",
        default=None,
        help="Token IDs to exclude from random replacement",
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
    provider = CharRandomReplacementProvider(
        data_dir=data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        mask_probability=args.mask_probability,
        batches_per_file=args.batches_per_file,
        max_backlog_files=args.max_backlog_files,
        sleep_seconds=args.sleep_seconds,
        seed=args.seed,
        verbose=args.verbose,
        original_token_probability_multiplier=
        args.original_token_probability_multiplier,
        extra_special_token_ids=args.extra_special_token_ids,
        dataset_partial_targets=args.dataset_partial_targets,
        **stage_kwargs,
    )
    provider.run()


# Explicit provider alias for prepare.py discovery
Provider = CharRandomReplacementProvider


if __name__ == "__main__":
    main()
