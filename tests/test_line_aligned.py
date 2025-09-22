import unittest
import torch
from data.char_diffusion.prepare_streaming import CharDiffusionProvider
from model import GPTConfig, GPT, ModelMode


class TestLineAlignedSequences(unittest.TestCase):
    def setUp(self):
        self.block_size = 48
        self.batch_size = 3
        self.provider = CharDiffusionProvider(
            data_dir='data/char_diffusion',
            batch_size=self.batch_size,
            block_size=self.block_size,
            config={'enable_line_aligned_sequences': True},
            verbose=False,
        )
        self.rng = torch.Generator()
        self.rng.manual_seed(123)

    def test_batch_shapes_and_masks(self):
        b = self.provider._sample_default_batch('train', self.rng)
        x, y, attn = b['x'], b['y'], b['attention_mask']
        self.assertEqual(tuple(x.shape), (self.batch_size, self.block_size))
        self.assertEqual(tuple(y.shape), (self.batch_size, self.block_size))
        self.assertEqual(tuple(attn.shape), (self.batch_size, self.block_size))
        # each row should have at least one valid token (SEP itself)
        self.assertTrue(torch.all(attn.sum(dim=1) >= 1).item())
        self.assertTrue(torch.all(attn.sum(dim=1) <= self.block_size).item())
        # labels should be ignore_index wherever attention_mask==0
        mask_violation = ((y != self.provider.ignore_index) & (attn == 0)).any().item()
        self.assertFalse(mask_violation)
        # SEP should be at the last attended position
        for i in range(self.batch_size):
            last_pos = int((attn[i] == 1).nonzero(as_tuple=False)[-1].item())
            self.assertEqual(int(x[i, last_pos].item()), int(self.provider.sep_token_id))

    def test_model_forward_with_attention_mask(self):
        b = self.provider._sample_default_batch('train', self.rng)
        cfg = GPTConfig(
            n_layer=2, n_head=2, n_embd=32, block_size=self.block_size,
            bias=False, vocab_size=self.provider.vocab_size, dropout=0.0,
            attention_type='bidirectional', position_encoding='absolute',
            mode=ModelMode.LANGUAGE_MODEL,
        )
        model = GPT(cfg)
        logits, loss = model(b['x'], b['y'], attention_mask=b['attention_mask'])
        self.assertEqual(tuple(logits.shape), (self.batch_size, self.block_size, self.provider.vocab_size))
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == '__main__':
    unittest.main()

