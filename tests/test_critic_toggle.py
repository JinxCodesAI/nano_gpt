import types
import os
import sys
import torch
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model import GPT, GPTConfig, ModelMode, CriticMode

def make_small_lm_with_critic():
    cfg = GPTConfig(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        critic_mode=CriticMode.TARGETLESS,
        critic_alpha=0.5,
    )
    model = GPT(cfg)
    model.eval()
    return model

@torch.no_grad()
def test_critic_modes():
    """Test that different critic modes produce different loss values"""
    model = make_small_lm_with_critic()

    B, T = 2, 8
    idx = torch.randint(0, model.config.vocab_size, (B, T))
    targets = torch.randint(0, model.config.vocab_size, (B, T))

    # Test TARGETLESS mode
    model.config.critic_mode = CriticMode.TARGETLESS
    logits1, loss1 = model(idx, targets=targets)
    assert logits1.shape[:2] == (B, T)
    assert loss1 is not None

    # Test TARGETED mode
    model.config.critic_mode = CriticMode.TARGETED
    logits2, loss2 = model(idx, targets=targets)
    assert logits2.shape[:2] == (B, T)
    assert loss2 is not None

    # Test NONE mode
    model.config.critic_mode = CriticMode.NONE
    logits3, loss3 = model(idx, targets=targets)
    assert logits3.shape[:2] == (B, T)
    assert loss3 is not None

@torch.no_grad()
def test_critic_head_not_created_when_mode_none():
    # Model with critic_mode=NONE should not have critic_head
    cfg = GPTConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        critic_mode=CriticMode.NONE,
    )
    m = GPT(cfg)
    assert not hasattr(m, 'critic_head'), "critic_head should not exist when critic_mode=NONE"

