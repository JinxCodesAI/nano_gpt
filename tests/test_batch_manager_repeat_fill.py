import os
import sys
import tempfile
import torch
import shutil

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from batch_manager import BatchManager


def create_synthetic_batch_file(tmpdir, split, total_rows, block_size, target_size, gen_batch_size, batches_per_file=1):
    os.makedirs(tmpdir, exist_ok=True)
    x = torch.arange(total_rows * block_size, dtype=torch.int64).reshape(total_rows, block_size)
    y = torch.arange(total_rows * target_size, dtype=torch.int64).reshape(total_rows, target_size)
    metadata = {
        'batch_size': gen_batch_size,
        'block_size': block_size,
        'target_size': target_size,
        'num_batches': batches_per_file,
        'file_idx': 0,
        'split': split,
    }
    torch.save({'x': x, 'y': y, 'metadata': metadata}, os.path.join(tmpdir, f'{split}_batches_0000.pt'))


def test_repeat_to_fill_end_of_file():
    tmpdir = tempfile.mkdtemp()
    try:
        # total_rows not divisible by train_batch_size ensures last slice underfilled
        total_rows = 10  # e.g., gen_batch_size * num_batches
        gen_batch_size = 2
        block_size = 4
        target_size = 4
        # need both train and val files to satisfy initializer
        create_synthetic_batch_file(tmpdir, 'train', total_rows, block_size, target_size, gen_batch_size)
        create_synthetic_batch_file(tmpdir, 'val', max(2, total_rows // 5), block_size, target_size, gen_batch_size)

        train_batch_size = 6  # larger; will request 6 rows at a time
        bm = BatchManager(tmpdir, train_batch_size, block_size, target_size, device_type='cpu')

        # First call should return 6 rows (indices 0..5)
        x, y = bm.get_batch('train', device='cpu')
        assert x.shape == (train_batch_size, block_size)
        assert y.shape == (train_batch_size, target_size)
        assert torch.equal(x[0], torch.arange(0, block_size))
        assert torch.equal(x[5], torch.arange(5*block_size, 6*block_size))

        # Second call: start_idx = 6, only 4 rows remain (6..9) -> should repeat 2 rows to fill 6
        x2, y2 = bm.get_batch('train', device='cpu')
        assert x2.shape == (train_batch_size, block_size)
        assert y2.shape == (train_batch_size, target_size)

        # Check the first 4 rows are 6..9 and last 2 are repeats of 6..7 again
        for i in range(4):
            start = (6 + i) * block_size
            assert torch.equal(x2[i], torch.arange(start, start + block_size))
        # Repeats
        assert torch.equal(x2[4], torch.arange(6 * block_size, 7 * block_size))
        assert torch.equal(x2[5], torch.arange(7 * block_size, 8 * block_size))

    finally:
        shutil.rmtree(tmpdir)

