import torch
import unittest

from data.common.line_aligned_utils import prepare_line_data
from data.sequence_scorer.synthetic_generation import apply_line_masking_direct


class TestLineMasking(unittest.TestCase):
    def test_prepare_line_data_uses_external_stoi(self):
        data = "a\nb\n"
        custom_stoi = {'a': 10, 'b': 11, '\n': 12}
        lines, lines_ids, used_stoi = prepare_line_data(data, stoi=custom_stoi)
        self.assertEqual(used_stoi, custom_stoi)
        # Expect two lines: 'a\n' and 'b\n'
        self.assertEqual(lines, ['a\n', 'b\n'])
        self.assertEqual(lines_ids, [[10, 12], [11, 12]])

    def test_apply_line_masking_direct_masks_only_replaced_lines(self):
        # Build a batch with 10 short lines of equal length (2 chars + newline)
        # seq_len = 10 * 3 = 30
        stoi = {ch: i for i, ch in enumerate(list("abcdefghijklmnopqrstuvwxyz\n"))}
        a, b, c, d, e, f, g, h, i_, j = [stoi[ch] for ch in "abcdefghij"]
        nl = stoi['\n']

        def line_pair(x):
            return [x, x, nl]

        # Sample tokens: aa\n bb\n ... jj\n
        tokens = []
        for ch in [a, b, c, d, e, f, g, h, i_, j]:
            tokens.extend(line_pair(ch))
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, 30]

        # Split lines pool: use the same ten lines
        split_lines = [line_pair(ch) for ch in [a, b, c, d, e, f, g, h, i_, j]]

        # Fix RNG for reproducibility; set ratio = exactly 0.2 (2 lines)
        stage_config = {"min_ratio": 0.2, "max_ratio": 0.2}
        g = torch.Generator().manual_seed(42)
        pad_token_id = 0

        replaced_x, mask = apply_line_masking_direct(
            x, stage_config, split_lines, nl, pad_token_id, g
        )

        # Expect exactly 2 full lines (each length 3) masked
        self.assertEqual(x.shape, replaced_x.shape)
        self.assertEqual(mask.shape, x.shape)
        self.assertEqual(int(mask.sum().item()), 2 * 3)

        # Ensure mask is False on padding (none in this sample)
        self.assertTrue(not mask[0, -1].item() or replaced_x[0, -1].item() != pad_token_id)


if __name__ == '__main__':
    unittest.main()

