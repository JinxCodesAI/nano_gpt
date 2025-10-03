import types
import torch
import pytest

from model import GPT, GPTConfig, ModelMode

def make_small_lm_with_critic():
    cfg = GPTConfig(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        add_critic_head=True,
        critic_alpha=0.5,
        mask_token_id=63,
        pad_token_id=0,
    )
    model = GPT(cfg)
    model.eval()
    return model

@pytest.mark.parametrize("disable", [False, True])
@torch.no_grad()
def test_forward_uses_single_or_double_encode_when_critic_toggled(disable):
    model = make_small_lm_with_critic()

    # Count calls to _encode_tokens on this instance
    calls = {"n": 0}
    orig_encode = model._encode_tokens

    def counted_encode(idx, attention_mask=None):
        calls["n"] += 1
        return orig_encode(idx, attention_mask=attention_mask)

    # Bind the wrapper back to the instance
    model._encode_tokens = types.MethodType(lambda self, idx, attention_mask=None: counted_encode(idx, attention_mask), model)

    if disable:
        model.disable_critic()
    else:
        model.enable_critic()

    B, T = 2, 8
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    # Ensure at least one masked token exists to exercise critic artifacts path
    idx[0, 0] = int(model.config.mask_token_id)

    targets = torch.randint(0, model.config.vocab_size, (B, T))

    logits, loss = model(idx, targets=targets)
    assert logits.shape[:2] == (B, T)

    # When critic enabled -> 2 passes (LM + critic); when disabled -> 1 pass
    expected = 1 if disable else 2
    assert calls["n"] == expected, f"_encode_tokens calls={calls['n']} expected={expected} (disable={disable})"

@torch.no_grad()
def test_disable_enable_idempotent_and_noop_without_critic_head():
    # Model without critic head should tolerate toggles (no-op)
    cfg = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        add_critic_head=False,
    )
    m = GPT(cfg)
    # Should not raise
    m.disable_critic()
    m.enable_critic()

