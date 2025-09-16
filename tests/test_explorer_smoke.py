#!/usr/bin/env python3
"""
Non-interactive smoke tests for interactive_diffusion_explorer dataset loading.

- Skips automatically if dataset directories or batch files are not present.
- Does not invoke interactive UI or model inference.
- Validates that meta is read and tensors are extracted using schema-aware logic.
"""
import os
import glob
import pytest

# Make sure we import from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive_diffusion_explorer import DiffusionExplorer


class DummyConfig:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size


class DummyModel:
    def __init__(self, vocab_size=4096):
        self.config = DummyConfig(vocab_size=vocab_size)


def _find_first_batch_file(dataset_root: str):
    # look under queue/train and queue/val for first .pt file
    for split in ("train", "val"):
        pattern = os.path.join(dataset_root, "queue", split, "*.pt")
        files = sorted(glob.glob(pattern))
        if files:
            # Return relative path from queue dir, i.e., 'train/<filename>'
            return os.path.join(split, os.path.basename(files[0]))
    return None


@pytest.mark.parametrize("dataset_name", ["char_diffusion", "sequence_scorer"])  # run for both if available
def test_explorer_loads_dataset_if_present(dataset_name):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", dataset_name)
    if not os.path.isdir(data_dir) or not os.path.exists(os.path.join(data_dir, "meta.pkl")):
        pytest.skip(f"Dataset {dataset_name} not present locally; skipping")

    rel_batch = _find_first_batch_file(data_dir)
    if rel_batch is None:
        pytest.skip(f"No batch files found for {dataset_name}; skipping")

    exp = DiffusionExplorer(interactive=False)
    # Inject dummy model for vocab_size (load_vocabulary requires it)
    exp.model = DummyModel(vocab_size=4096)
    exp.dataset_name = dataset_name

    assert exp.load_vocabulary() is True

    # load_batch_file expects a path relative to the queue dir (e.g., 'train/<file>')
    ok = exp.load_batch_file(rel_batch)
    assert ok is True, "Batch file failed to load"

    # Validate current_batch contents
    assert isinstance(exp.current_batch, dict)
    assert 'input' in exp.current_batch and 'target' in exp.current_batch
    x = exp.current_batch['input']
    y = exp.current_batch['target']
    # basic shape sanity
    assert x.dim() == 2  # [batch, seq]
    assert x.shape[0] > 0 and x.shape[1] > 0
    # Targets can be token-wise [B, L] (MLM) or scalar [B]
    assert y.shape[0] == x.shape[0]

