import torch
import types

from sample_utils import apply_remasking_step


class DummyConfig:
    def __init__(self, add_critic_head=True):
        self.add_critic_head = add_critic_head


class DummyModel:
    def __init__(self, scores):
        self.config = DummyConfig(add_critic_head=True)
        self._scores = scores

    @torch.no_grad()
    def critic_scores(self, idx, attention_mask=None):
        # Return preset critic logits
        return self._scores


def test_apply_remasking_step_critic_path_masks_high_scores():
    # Setup tokens
    B, T = 2, 6
    mask_token_id = 99
    tokens = torch.tensor([
        [10, 11, 12, 13, 14, 15],
        [20, 21, 22, 23, 24, 25],
    ])
    prediction_tokens = tokens.clone()

    # Critic logits: choose top-2 per row (mask_ratio=2/T)
    scores = torch.tensor([
        [0.1, 5.0, 0.2, 4.0, 0.3, 0.0],  # expect mask positions 1 and 3
        [0.0, 0.1, 0.2, 0.3, 9.0, 8.0],  # expect mask positions 4 and 5
    ], dtype=torch.float32)

    base_model = DummyModel(scores)

    # mask_ratio to mask exactly k=2 per row
    iteration = 0
    iterations = 10
    start_ratio = 2 / T
    end_ratio = 2 / T

    remasked = apply_remasking_step(
        tokens=tokens,
        prediction_tokens=prediction_tokens,
        iteration=iteration,
        iterations=iterations,
        schedule_type='linear',
        masking_ratios=None,
        start_ratio=start_ratio,
        end_ratio=end_ratio,
        remasking_model=None,  # ensure critic branch is used
        randomness_strength=0.0,  # deterministic
        mask_token_id=mask_token_id,
        device='cpu',
        base_model=base_model,
        intelligent_remasking=False,
        verbose=False,
        logits_from_predict=None,
        protected_mask=None,
    )

    # Validate chosen indices
    assert remasked[0, 1].item() == mask_token_id
    assert remasked[0, 3].item() == mask_token_id
    assert remasked[1, 4].item() == mask_token_id
    assert remasked[1, 5].item() == mask_token_id

    # Others remain unchanged
    for i, j in [(0,0), (0,2), (0,4), (0,5), (1,0), (1,1), (1,2), (1,3)]:
        assert remasked[i, j].item() != mask_token_id

