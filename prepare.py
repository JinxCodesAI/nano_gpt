"""
Unified data preparation entrypoint.

Usage:
    python prepare.py config/<your_config>.py

Loads the same training config (via configurator.py), validates it, and starts
streaming data generation using the appropriate dataset provider under data/.
"""
from __future__ import annotations

import os
import sys
from typing import Type

from importlib import import_module

# lightweight configurator reuse: execute same as train.py does
# We'll mimic train.py's behavior: treat first arg as config file path

def _load_config() -> dict:
    # This mirrors configurator.py but returns a dict of globals after exec
    import builtins
    cfg_globals = {}
    # seed with minimal expected defaults so exec can override if needed
    # not strictly necessary; the config file should set values itself
    args = sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python prepare.py <config_file.py>")
    config_file = args[0]
    if '=' in config_file or config_file.startswith('--'):
        raise SystemExit("First argument must be a config file path, no key=value here")
    print(f"Overriding config with {config_file}:")
    with open(config_file) as f:
        print(f.read())
    exec(open(config_file).read(), cfg_globals)
    return cfg_globals


def _provider_for_dataset(dataset: str) -> Type:
    """Discover a provider class by convention.

    Convention:
    - Module: data.<dataset>.prepare_streaming must exist
    - Within the module, exactly one subclass of DataProviderBase must be defined
      OR the module must export a symbol named 'Provider' pointing to the class.
    """
    module_name = f"data.{dataset}.prepare_streaming"
    try:
        mod = import_module(module_name)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Could not find provider module '{module_name}'. Ensure your dataset folder has prepare_streaming.py"
        ) from e

    # If explicit Provider symbol is present, use it
    if hasattr(mod, 'Provider'):
        return getattr(mod, 'Provider')

    # Otherwise, search for exactly one subclass of DataProviderBase
    try:
        from data.common.provider_base import DataProviderBase
    except Exception as e:
        raise RuntimeError("Failed to import DataProviderBase for provider discovery") from e

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, DataProviderBase) and obj is not DataProviderBase:
            candidates.append(obj)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(
            f"No DataProviderBase subclass found in {module_name}. Define one subclass or export 'Provider'."
        )
    # Multiple candidates found; ask user to disambiguate via Provider symbol
    raise ValueError(
        f"Multiple provider classes found in {module_name}: {[c.__name__ for c in candidates]}. "
        f"Export a single 'Provider' symbol in the module to disambiguate."
    )


def main() -> None:
    from config.validator import validate_config

    cfg = _load_config()
    # Validate config sufficiency for both training and data gen
    validate_config(cfg)

    dataset = cfg['dataset']
    data_dir = os.path.join('data', dataset)

    ProviderCls = _provider_for_dataset(dataset)

    provider = ProviderCls(
        data_dir=data_dir,
        batch_size=cfg['batch_size'],
        block_size=cfg['block_size'],
        target_size=cfg.get('target_size', None),
        batches_per_file=cfg.get('batches_per_file', 100),
        max_backlog_files=cfg.get('max_backlog_files', 2),
        sleep_seconds=cfg.get('sleep_seconds', 2.0),
        seed=cfg.get('seed', 1337),
        verbose=cfg.get('data_stream_verbose', True),
    )
    provider.run()


if __name__ == '__main__':
    main()

