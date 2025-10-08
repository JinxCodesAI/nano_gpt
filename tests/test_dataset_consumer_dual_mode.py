import os
import pickle
import time
import shutil
import torch

from dataset_consumer import DatasetConsumer


def setup_dual_mode_test_dir(root: str):
    data_dir = os.path.join('data', root)
    train_dir = os.path.join(data_dir, 'queue', 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Write minimal meta.pkl
    meta = {
        'training_type': 'DUAL_MODE',
        'batch_schema': [
            {'name': 'x', 'dtype': 'int64', 'shape': [4], 'role': 'input'},
            {'name': 'y', 'dtype': 'int64', 'shape': [4], 'role': 'target'},
        ],
        'vocab_size': 10,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    # Create one file with 3 batches, batch_size=2
    batch_size = 2
    num_batches = 3
    seq = 0
    produced_at = int(time.time() * 1000)

    x = torch.zeros(batch_size * num_batches, 4, dtype=torch.long)
    y = torch.zeros(batch_size * num_batches, 4, dtype=torch.long)

    # Provide one SS batch worth of tensors to match the sequence_scorer slot
    input_ids = torch.full((batch_size, 4), 7, dtype=torch.long)
    targets = torch.linspace(0.0, 1.0, batch_size, dtype=torch.float32)

    tensors = {'x': x, 'y': y, 'input_ids': input_ids, 'targets': targets}
    metadata = {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'file_idx': seq,
        'split': 'train',
        'produced_at': produced_at,
        'model_mode': ['language_model', 'sequence_scorer', 'language_model'],
    }

    tmp_name = f".tmp-{produced_at}-{seq:06d}.pt"
    final_name = f"{produced_at}-{seq:06d}-{num_batches}.pt"
    tmp_path = os.path.join(train_dir, tmp_name)
    final_path = os.path.join(train_dir, final_name)
    torch.save({'tensors': tensors, 'metadata': metadata}, tmp_path)
    os.replace(tmp_path, final_path)

    return data_dir


def teardown_dual_mode_test_dir(data_dir: str):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)


def test_consumer_sets_model_mode_per_batch():
    dataset_root = 'test_dual_mode_unit'
    data_dir = setup_dual_mode_test_dir(dataset_root)
    try:
        consumer = DatasetConsumer(
            data_dir=data_dir,
            batch_size=2,
            block_size=4,
            target_size=4,
            device_type='cpu',
            prefer_queue=True,
            cache_files=1,
            wait_sleep_seconds=0.01,
            wait_timeout_seconds=1.0,
            verbose=False,
        )

        # First batch
        b1 = consumer.get_batch('train', device='cpu')
        assert b1['_model_mode'] == 'language_model'
        assert b1['x'].shape == (2, 4)
        assert b1['y'].shape == (2, 4)

        # Second batch
        b2 = consumer.get_batch('train', device='cpu')
        assert b2['_model_mode'] == 'sequence_scorer'
        # For sequence_scorer mode, x/y are not present; keys are input_ids/targets
        assert set(b2.keys()) == {'input_ids', 'targets', '_model_mode'}

        # Third batch
        b3 = consumer.get_batch('train', device='cpu')
        assert b3['_model_mode'] == 'language_model'
        assert b3['x'].shape == (2, 4)
        assert b3['y'].shape == (2, 4)
    finally:
        teardown_dual_mode_test_dir(data_dir)

