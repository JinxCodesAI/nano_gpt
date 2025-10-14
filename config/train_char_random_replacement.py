"""Training configuration for the random replacement character dataset.

This module mirrors :mod:`config.train_char_diffusion` but swaps in the
``char_random_replacement`` dataset and reloads stage composition configs from
``data/char_random_replacement/config``. Importing the diffusion config first
lets us inherit sensible defaults without copying every hyperparameter.
"""

from __future__ import annotations

import importlib.util
import os

from config.train_char_diffusion import *  # noqa: F401,F403 - re-export base defaults


dataset = "char_random_replacement"
wandb_run_name = "random-replacement-char"

# ``composition_config`` refers to ``data/char_random_replacement/config/<name>.py``.
composition_config = "example"


def _load_stage_config(dataset_name: str, config_name: str | None) -> None:
    """Populate globals with stage configuration overrides.

    This replicates the logic used in ``train_char_diffusion`` but resolves the
    composition module relative to ``dataset_name`` so that both ``train.py``
    and ``prepare.py`` can reuse the same config file.
    """

    global use_all_stages_for_training
    global unmasking_stages
    global validation_stages

    if config_name is None:
        use_all_stages_for_training = None
        unmasking_stages = None
        validation_stages = None
        return

    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        dataset_name,
        "config",
        f"{config_name}.py",
    )

    if not os.path.exists(config_path):
        print(f"Warning: composition config file not found at {config_path}")
        use_all_stages_for_training = None
        unmasking_stages = None
        validation_stages = None
        return

    spec = importlib.util.spec_from_file_location(f"{config_name}_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load composition config at {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for attr_name, value in module.__dict__.items():
        if attr_name.startswith("_"):
            continue
        globals()[attr_name] = value

    print(f"Loaded composition config from {config_path}")


_load_stage_config(dataset, composition_config)

