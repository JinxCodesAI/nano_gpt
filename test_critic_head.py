import torch
import pytest

from model import GPT, GPTConfig, ModelMode


def make_small_config(**overrides):
    cfg = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
        attention_type='causal',
        position_encoding='absolute',
        mode=ModelMode.LANGUAGE_MODEL,
        add_critic_head=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_critic_head_disabled_no_side_effect():
    cfg = make_small_config(add_critic_head=False)
    model = GPT(cfg)
    B, T = 2, 6
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)
    assert logits.shape[:2] == (B, T)
    assert loss is not None and torch.is_tensor(loss)
    assert not hasattr(model, 'critic_head')


def test_critic_head_enabled_combined_loss_and_scores():
    mask_id = 1
    pad_id = 0
    cfg = make_small_config(add_critic_head=True, critic_alpha=0.5, mask_token_id=mask_id, pad_token_id=pad_id)
    model = GPT(cfg)
    B, T = 2, 6
    idx = torch.randint(2, cfg.vocab_size, (B, T))  # avoid pad/mask ids initially
    # insert some masks and pads
    idx[0, 0] = pad_id
    idx[0, 1] = mask_id
    idx[1, 2] = mask_id
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    # mark some ignore positions in targets to simulate standard LM shifting
    targets[targets == pad_id] = cfg.ignore_index

    logits, loss = model(idx, targets=targets)
    assert hasattr(model, 'critic_head')
    assert logits.shape[:2] == (B, T)
    assert loss is not None and torch.is_tensor(loss)

    # critic_scores API
    scores = model.critic_scores(idx)
    assert scores.shape == (B, T)

