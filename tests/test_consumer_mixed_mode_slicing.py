import os
import pickle
import time
import shutil
import torch

from dataset_consumer import DatasetConsumer


def setup_mixed_file(root: str, modes):
    data_dir = os.path.join('data', root)
    train_dir = os.path.join(data_dir, 'queue', 'train')
    os.makedirs(train_dir, exist_ok=True)

    meta = {
        'training_type': 'DUAL_MODE',
        'batch_schema': [
            {'name': 'x', 'dtype': 'int64', 'shape': [4], 'role': 'input'},
            {'name': 'y', 'dtype': 'int64', 'shape': [4], 'role': 'target'},
            {'name': 'input_ids', 'dtype': 'int64', 'shape': [4], 'role': 'input'},
            {'name': 'targets', 'dtype': 'float32', 'shape': [], 'role': 'target'},
        ],
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    batch_size = 2
    produced_at = int(time.time() * 1000)

    lm_batches = sum(1 for m in modes if m == 'language_model')
    ss_batches = sum(1 for m in modes if m == 'sequence_scorer')

    tensors = {}
    if lm_batches:
        tensors['x'] = torch.arange(lm_batches * batch_size * 4, dtype=torch.long).view(lm_batches * batch_size, 4)
        tensors['y'] = tensors['x'].clone()
    if ss_batches:
        tensors['input_ids'] = torch.full((ss_batches * batch_size, 4), 7, dtype=torch.long)
        tensors['targets'] = torch.linspace(0.0, 1.0, ss_batches * batch_size, dtype=torch.float32)

    metadata = {
        'batch_size': batch_size,
        'num_batches': len(modes),
        'file_idx': 0,
        'split': 'train',
        'produced_at': produced_at,
        'model_mode': modes,
    }

    tmp_name = f".tmp-{produced_at}-000000.pt"
    final_name = f"{produced_at}-000000-{len(modes)}.pt"
    tmp_path = os.path.join(train_dir, tmp_name)
    final_path = os.path.join(train_dir, final_name)
    torch.save({'tensors': tensors, 'metadata': metadata}, tmp_path)
    os.replace(tmp_path, final_path)

    return data_dir


def teardown_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)


def test_mixed_mode_slicing_and_keys():
    modes = ['language_model', 'sequence_scorer', 'language_model']
    data_dir = setup_mixed_file('test_mixed_dual', modes)
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
        b0 = consumer.get_batch('train', device='cpu')
        assert b0['_model_mode'] == 'language_model'
        assert set(b0.keys()) == {'x', 'y', '_model_mode'} or set(b0.keys()) == {'x', 'y', 'attention_mask', '_model_mode'}
        b1 = consumer.get_batch('train', device='cpu')
        assert b1['_model_mode'] == 'sequence_scorer'
        assert set(b1.keys()) == {'input_ids', 'targets', '_model_mode'}
        b2 = consumer.get_batch('train', device='cpu')
        assert b2['_model_mode'] == 'language_model'
        assert 'x' in b2 and 'y' in b2 and 'input_ids' not in b2
    finally:
        teardown_dir(data_dir)

