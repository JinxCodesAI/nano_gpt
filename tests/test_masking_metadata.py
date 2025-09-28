import os
import tempfile
import unittest

import torch

from data.common.provider_base import DataProviderBase


class FakeProvider(DataProviderBase):
    def build_meta(self):
        # Minimal required meta
        return {
            "training_type": "TESTING",
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [self.block_size], "role": "target"},
            ],
        }

    def sample_batch(self, split: str, rng):
        x = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
        y = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
        # Return per-sample metadata list
        masking_strategy = ["random"] * self.batch_size
        masking_ratio = [0.0] * self.batch_size
        return {"x": x, "y": y, "masking_strategy": masking_strategy, "masking_ratio": masking_ratio}


class TestMaskingMetadataAggregation(unittest.TestCase):
    def test_per_sample_masking_strategy_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = FakeProvider(
                data_dir=tmpdir,
                batch_size=4,
                block_size=8,
                batches_per_file=3,
                max_backlog_files=1,
                verbose=False,
            )
            # Ensure output directories exist
            provider.ensure_dirs()
            # Produce a single train file
            provider.produce_one_file("train", seq=0)

            # Locate produced file
            out_dir = os.path.join(tmpdir, "queue", "train")
            files = [fn for fn in os.listdir(out_dir) if fn.endswith(".pt")]
            self.assertTrue(files, "No output .pt file produced")
            path = os.path.join(out_dir, files[0])

            data = torch.load(path, map_location="cpu")
            self.assertIn("metadata", data)
            meta = data["metadata"]
            # Strategies
            self.assertIn("masking_strategy", meta)
            strategies = meta["masking_strategy"]
            self.assertEqual(len(strategies), 3 * 4)
            self.assertTrue(all(s == "random" for s in strategies))
            # Ratios
            self.assertIn("masking_ratio", meta)
            ratios = meta["masking_ratio"]
            self.assertEqual(len(ratios), 3 * 4)
            self.assertTrue(all(float(r) == 0.0 for r in ratios))


if __name__ == "__main__":
    unittest.main()

