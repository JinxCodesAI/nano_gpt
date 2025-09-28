import unittest
import torch

from data.sequence_scorer.prepare_streaming import SequenceScorerProvider


class TestTargetTransform(unittest.TestCase):
    def test_linear_inversion(self):
        # Bypass heavy __init__ by constructing without initialization
        provider = SequenceScorerProvider.__new__(SequenceScorerProvider)
        x = torch.tensor([0.0, 0.85, 1.0], dtype=torch.float32)
        y = provider._transform_ratio_to_target(x)
        self.assertTrue(torch.allclose(y, torch.tensor([1.0, 0.15, 0.0], dtype=torch.float32), atol=1e-6))


if __name__ == "__main__":
    unittest.main()

