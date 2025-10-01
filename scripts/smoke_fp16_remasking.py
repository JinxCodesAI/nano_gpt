import os
import sys
import torch
# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sample_utils import apply_remasking_step


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T = 2, 32
    mask_token_id = 65

    # Create a token matrix with values not equal to mask_token_id
    prediction_tokens = torch.randint(low=0, high=100, size=(B, T), dtype=torch.long, device=device)
    prediction_tokens[:, 0] = mask_token_id  # ensure at least one masked token to exclude from unmaskable

    class DummyCfg:
        add_critic_head = True

    class DummyModel:
        def __init__(self):
            self.config = DummyCfg()
        def critic_scores(self, x):
            # Return half-precision scores to emulate fp16 path
            return torch.randn(x.size(0), x.size(1), dtype=torch.float16, device=x.device)

    base_model = DummyModel()

    out = apply_remasking_step(
        tokens=prediction_tokens,
        prediction_tokens=prediction_tokens,
        iteration=0,
        iterations=2,
        schedule_type='linear',
        start_ratio=0.9,
        end_ratio=0.1,
        remasking_model=None,
        randomness_strength=0.0,  # deterministic
        mask_token_id=mask_token_id,
        device=device,
        base_model=base_model,
        intelligent_remasking=False,
        verbose=False,
        logits_from_predict=None,
        protected_mask=None,
    )

    assert out.shape == prediction_tokens.shape
    print('OK: apply_remasking_step executed without fp16 overflow')


if __name__ == '__main__':
    main()

