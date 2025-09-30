import os, sys
import torch
from contextlib import nullcontext

# Ensure project root on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.training_step import TrainingStep
from model import GPT, GPTConfig, ModelMode


class DummyConsumer:
    def __init__(self, batch):
        self._batch = batch
    def get_batch(self, split, device):
        return self._batch


class DummyLossModifiers:
    def __init__(self, current_iter=5):
        self.current_iter = current_iter
    def is_empty(self):
        return True


def make_dummy_batch(B=2, T=8, V=32, MASK=30, PAD=31, IGN=-100):
    # Build inputs with a couple of masked positions per sample
    X = torch.randint(low=0, high=V-2, size=(B, T))  # avoid MASK/PAD
    Y = torch.full((B, T), IGN, dtype=torch.long)
    # For each row, choose two positions to supervise via masked tokens
    for b in range(B):
        pos = torch.randperm(T)[:2]
        # ground truth tokens for these positions
        gt = torch.randint(low=0, high=V-2, size=(2,))
        Y[b, pos] = gt
        X[b, pos] = MASK
    return {"x": X, "y": Y}


def test_training_step_with_critic_end_to_end_cpu():
    device = "cpu"
    V = 32
    cfg = GPTConfig(
        block_size=16,
        vocab_size=V,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        mode=ModelMode.LANGUAGE_MODEL,
        add_critic_head=True,
        critic_alpha=0.5,
        start_critic_iteration=0,
        end_critic_iteration=0,
        mask_token_id=30,
        pad_token_id=31,
    )
    model = GPT(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    batch = make_dummy_batch(V=V, MASK=cfg.mask_token_id, PAD=cfg.pad_token_id)
    consumer = DummyConsumer(batch)

    step = TrainingStep(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_accumulation_steps=1,
        grad_clip=0.0,
        ddp=False,
        ctx=nullcontext(),
    )

    loss, next_batch = step.execute_step(batch=batch, loss_modifiers=DummyLossModifiers(), consumer=consumer, device=device)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert step.last_loss_main >= 0.0
    assert step.last_loss_critic >= 0.0

