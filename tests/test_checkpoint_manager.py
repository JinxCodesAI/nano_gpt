import os
import json
import tempfile
import unittest

import torch

from checkpoint_manager import CheckpointManager


class TestCheckpointManagerResolution(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_dir = self.tmpdir.name
        self.cm = CheckpointManager(self.out_dir)
        # minimal config with training_type
        self.training_type = 'SEQUENCE_SCORING'
        config = {'meta': {'training_type': self.training_type}}
        self.cm.set_metadata(model_args={}, config=config)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_ckpt(self, iter_num: int):
        path = os.path.join(self.out_dir, f"ckpt_{self.training_type}_{iter_num}.pt")
        torch.save({'iter_num': iter_num, 'model': {}, 'optimizer': {}, 'model_args': {}, 'best_val_loss': 0.0, 'config': {}}, path)
        return path

    def test_scan_latest_versioned(self):
        self._write_ckpt(5000)
        latest = self._write_ckpt(5250)
        ckpt = self.cm.load(device='cpu')
        # Should pick the latest versioned file
        self.assertEqual(ckpt['iter_num'], 5250)

    def test_metadata_preferred(self):
        p5000 = self._write_ckpt(5000)
        self._write_ckpt(5250)
        # Write metadata to point to 5000
        meta_path = os.path.join(self.out_dir, 'ckpt_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({self.training_type: {'last_path': os.path.basename(p5000), 'iter': 5000}}, f)
        ckpt = self.cm.load(device='cpu')
        # Should use metadata, even if not the latest by scan
        self.assertEqual(ckpt['iter_num'], 5000)


if __name__ == '__main__':
    unittest.main()

