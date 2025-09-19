import os
import pickle
import tempfile
import unittest
from contextlib import nullcontext

import torch

from dataset_consumer import DatasetConsumer
from core.evaluator import Evaluator
from model import ModelMode


def write_meta(path):
    with open(path, 'wb') as f:
        pickle.dump({'vocab_size': 100}, f)


def write_val_file(path, num_samples=12, zeros_every=5):
    # Build simple tensors with controllable zeros density
    input_ids = torch.randint(0, 50, (num_samples, 4), dtype=torch.long)
    targets = torch.ones(num_samples, dtype=torch.float32)  # start as non-zero
    if zeros_every > 0:
        targets[::zeros_every] = 0.0
    payload = {
        'tensors': {
            'input_ids': input_ids,
            'targets': targets,
        },
        'metadata': {
            'num_samples': num_samples
        }
    }
    torch.save(payload, path)


class DummyLossPipeline:
    def temporarily_disabled(self):
        return nullcontext()


class DummySeqScorer:
    class Cfg:
        def __init__(self):
            self.mode = ModelMode.SEQUENCE_SCORER

    def __init__(self):
        self.config = DummySeqScorer.Cfg()

    def __call__(self, X, targets=None, loss_modifiers=None):
        # For non-zero loss: return synthetic loss (mean of inputs)
        if targets is not None:
            loss = torch.tensor(0.123, dtype=torch.float32)
            return None, loss
        # For zero-only preds: return logits ~0.2
        b = X.shape[0]
        logits = torch.full((b,), 0.2, dtype=torch.float32)
        return logits, None


class TestValConsumerAndEvaluator(unittest.TestCase):
    def test_val_files_not_deleted_and_circular(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = td
            os.makedirs(os.path.join(data_dir, 'queue', 'val'), exist_ok=True)
            # meta
            write_meta(os.path.join(data_dir, 'meta.pkl'))
            # two val files
            f1 = os.path.join(data_dir, 'queue', 'val', '000001.pt')
            f2 = os.path.join(data_dir, 'queue', 'val', '000002.pt')
            write_val_file(f1, num_samples=8, zeros_every=4)
            write_val_file(f2, num_samples=8, zeros_every=4)

            cons = DatasetConsumer(
                data_dir=data_dir,
                batch_size=4,
                device_type='cpu',
                prefer_queue=True,
                verbose=False,
            )

            # Consume more than total samples across both files
            # Expect no deletion for val files
            for _ in range(6):  # 6 batches * 4 = 24 > 16 samples
                X, Y = cons.get_batch('val', device='cpu')
                self.assertEqual(X.shape[0], 4)
                self.assertEqual(Y.shape[0], 4)

            self.assertTrue(os.path.exists(f1), 'val file 1 should not be deleted')
            self.assertTrue(os.path.exists(f2), 'val file 2 should not be deleted')

    def test_evaluator_topup_zero_only_stats(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = td
            os.makedirs(os.path.join(data_dir, 'queue', 'val'), exist_ok=True)
            write_meta(os.path.join(data_dir, 'meta.pkl'))
            # Create first file with NO zeros so the main eval window may miss zeros
            f1 = os.path.join(data_dir, 'queue', 'val', '000001.pt')
            write_val_file(f1, num_samples=12, zeros_every=0)  # no zeros
            # Create second file with some zeros to be picked up by top-up
            f2 = os.path.join(data_dir, 'queue', 'val', '000002.pt')
            write_val_file(f2, num_samples=12, zeros_every=3)  # zeros present

            cons = DatasetConsumer(
                data_dir=data_dir,
                batch_size=4,
                device_type='cpu',
                prefer_queue=True,
                verbose=False,
            )

            model = DummySeqScorer()
            pipeline = DummyLossPipeline()
            evaluator = Evaluator(
                model=model,
                consumer=cons,
                loss_modifier_pipeline=pipeline,
                eval_iters=3,  # 3 batches -> will draw only from f1 initially
                ctx=nullcontext(),
                device='cpu',
                min_zero_for_stats=2,
                max_extra_batches_for_zero_stats=10,
                reset_val_stream_each_eval=True,
            )

            out = evaluator.evaluate(splits=['val'])
            # Expect zero stats present thanks to top-up batches
            self.assertIn('val', out)
            self.assertIn('val/zero_mean', out)
            self.assertIn('val/zero_p90', out)


if __name__ == '__main__':
    unittest.main()

