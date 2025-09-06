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
    # Map dataset string to provider class path
    registry = {
        'shakespeare_char': ('data.shakespeare_char.prepare_streaming', 'ShakespeareCharProvider'),
        # add other datasets here as they are implemented
    }
    if dataset not in registry:
        raise ValueError(f"No streaming provider registered for dataset '{dataset}'")
    module_name, class_name = registry[dataset]
    mod = import_module(module_name)
    cls = getattr(mod, class_name)
    return cls


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
        verbose=cfg.get('data_stream_verbose', False),
    )
    provider.run()


if __name__ == '__main__':
    main()

