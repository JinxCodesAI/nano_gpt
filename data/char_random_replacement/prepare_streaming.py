"""Streaming provider with random replacement corruption."""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import torch

from data.char_diffusion.prepare_streaming import CharDiffusionProvider

from .corruption_utils import RandomReplacementCorruptor, build_candidate_token_ids


class CharRandomReplacementProvider(CharDiffusionProvider):
    """Character dataset that replaces masked tokens with random characters."""

    def __init__(
        self,
        *args,
        original_token_probability_multiplier: float = 1.0,
        extra_special_token_ids: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        self._original_multiplier = original_token_probability_multiplier
        self._extra_special_token_ids = [int(token) for token in (extra_special_token_ids or [])]
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

    def _apply_corruption(self, x: torch.Tensor, mask: torch.Tensor, rng):
        return self._corruptor.corrupt(x, mask, rng)

    def _apply_stage_corruption(
        self,
        stage_corrupted_x: torch.Tensor,
        original_x: torch.Tensor,
        mask: torch.Tensor,
        rng,
    ) -> torch.Tensor:
        del stage_corrupted_x  # already encoded; we only need the mask
        return self._corruptor.corrupt(original_x, mask, rng)

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
        **stage_kwargs,
    )
    provider.run()


# Explicit provider alias for prepare.py discovery
Provider = CharRandomReplacementProvider


if __name__ == "__main__":
    main()
