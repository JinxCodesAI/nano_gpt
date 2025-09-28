import unittest
import torch

from data.sequence_scorer.prepare_streaming import SequenceScorerProvider


class TestTargetTransform(unittest.TestCase):
    def test_non_linear_inverted_polynomial(self):
        provider = SequenceScorerProvider.__new__(SequenceScorerProvider)
        x = torch.tensor([0.0, 0.85, 1.0], dtype=torch.float32)
        y = provider._transform_ratio_to_target(x)
        # Expected: 1 - (-x^4 + 2x^3 - 2x + 1)
        expected = 1 - (-x**4 + 2 * x**3 - 2 * x + 1)
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))


if __name__ == '__main__':
    unittest.main()

