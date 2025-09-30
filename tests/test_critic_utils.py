import os, sys
import torch
# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sample_utils import build_critic_artifacts_from_logits

def make_logits_for_tokens(batch_size, seq_len, vocab_size, picks):
    """
    Build logits tensor with argmax at picks[b][t] for each position.
    picks: list of lists of length seq_len with integer token ids (or None for default 0)
    """
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for t in range(seq_len):
            tok = picks[b][t]
            if tok is None:
                tok = 0
            logits[b, t, tok] = 10.0  # strong preference
    return logits


def test_build_critic_artifacts_masked_only():
    B, T, V = 1, 6, 100
    MASK = 99
    IGN = -100
    # token ids
    BROWN, FOX, JUMPED, OVER, BALL, TALL, FENCE, RUN, IN = 1, 2, 3, 4, 5, 6, 7, 8, 9

    idx = torch.tensor([[MASK, FOX, MASK, BALL, TALL, FENCE]])
    targets = torch.tensor([[BROWN, IGN, JUMPED, OVER, IGN, IGN]])

    # logits prefer BROWN at pos0, RUN at pos2 (others arbitrary)
    picks = [[BROWN, 0, RUN, 0, 0, 0]]
    logits = make_logits_for_tokens(B, T, V, picks)

    out = build_critic_artifacts_from_logits(
        idx=idx,
        logits=logits,
        targets=targets,
        mask_token_id=MASK,
        ignore_index=IGN,
        pad_token_id=None,
        scope='masked_only',
    )

    critic_input = out['critic_input'][0]
    critic_target = out['critic_target'][0]
    critic_valid = out['critic_valid'][0]

    # Valid only at masked positions 0 and 2
    assert critic_valid.tolist() == [True, False, True, False, False, False]
    # Targets: pos0 correct -> 0, pos2 incorrect -> 1
    assert int(critic_target[0].item()) == 0
    assert int(critic_target[2].item()) == 1


def test_build_critic_artifacts_masked_and_ignore():
    B, T, V = 1, 6, 100
    MASK = 99
    IGN = -100
    BROWN, FOX, JUMPED, OVER, BALL, TALL, FENCE, RUN, IN = 1, 2, 3, 4, 5, 6, 7, 8, 9

    idx = torch.tensor([[MASK, FOX, MASK, BALL, TALL, FENCE]])
    targets = torch.tensor([[BROWN, IGN, JUMPED, OVER, IGN, IGN]])

    picks = [[BROWN, 0, RUN, 0, 0, 0]]
    logits = make_logits_for_tokens(B, T, V, picks)

    out = build_critic_artifacts_from_logits(
        idx=idx,
        logits=logits,
        targets=targets,
        mask_token_id=MASK,
        ignore_index=IGN,
        pad_token_id=None,
        scope='masked_and_ignore',
    )

    critic_target = out['critic_target'][0]
    critic_valid = out['critic_valid'][0]

    # Valid at masked (0,2) and ignore positions (1,4,5)
    assert critic_valid.tolist() == [True, True, True, False, True, True]

    # Ignore positions forced to 0 target
    assert int(critic_target[1].item()) == 0
    assert int(critic_target[4].item()) == 0
    assert int(critic_target[5].item()) == 0

    # Masked positions: pos0 correct -> 0, pos2 incorrect -> 1
    assert int(critic_target[0].item()) == 0
    assert int(critic_target[2].item()) == 1

