# GRPO Fine-Tuning Pipeline

This guide walks through the production Group-Relative Policy Optimisation
(GRPO) training stack that lives under [`grpo/`](../../grpo/README.md). It is a
reinforcement-style fine-tuning loop built on top of the diffusion-ready GPT
in this repository. Use it when you want judge-guided refinement instead of
pure diffusion unmasking.

## When to Use GRPO

GRPO complements the supervised diffusion training loop:

- **Reinforcement fine-tuning:** Optimise a pretrained generator with a frozen
  judge for preference-style rewards.
- **Self-generated data:** Generate completions on the fly instead of relying on
  static target sequences.
- **Policy regularisation:** Keep the generator close to its starting point via
  KL control against a frozen reference copy.

If you only need supervised loss modifiers or critic-guided sampling, stick to
`train.py`. Switch to GRPO when judge rewards should shape the policy directly.

## File Layout

```text
grpo/
├── train_grpo.py         # CLI entry point and config loading
├── grpo_config.py        # Default hyperparameters and CLI plumbing
├── grpo_trainer.py       # Training orchestration and logging
├── grpo_training_step.py # Core GRPO algorithm (generation + updates)
└── README.md             # High-level overview and usage notes
```

### `train_grpo.py`

- Parses CLI flags or config files produced by `grpo_config.py`.
- Boots the generator, reference, and judge models, wiring them into a
  `GRPOTrainer` instance.
- Handles checkpoint directory setup and optional Weights & Biases logging.

### `grpo_trainer.py`

- Wraps the shared `Trainer` utilities from `core/` with GRPO-specific
  bookkeeping.
- Coordinates gradient scaling, logging hooks, checkpoint cadence, and optional
  sample dumps during training.

### `grpo_training_step.py`

- Implements the full GRPO algorithm: batched generation via
  `predict_and_sample_tokens`, reward scoring with the frozen judge, group
  advantage computation, KL-regularised policy gradient loss, and optimiser
  steps.
- Integrates with the existing dataset streaming pipeline by consuming masked
  inputs from `DatasetConsumer`.

### `grpo_config.py`

- Provides a `dataclass` that mirrors the command-line arguments and exposes a
  `.from_file()` helper so configs can be stored alongside diffusion presets.
- Houses defaults for group size, KL coefficient, reward normalisation, and
  sampling temperature.

## Running GRPO Training

```bash
python grpo/train_grpo.py config/grpo_config.py \
  --generator_checkpoint=out-char-diffusion/checkpoint.pt \
  --judge_checkpoint=out-char-diffusion/judge.pt \
  --dataset=char_diffusion
```

Key runtime expectations:

- **Generator checkpoint** – start from a diffusion-trained GPT.
- **Judge checkpoint** – a frozen scorer that outputs scalar rewards.
- **Dataset** – masked prompts served by the streaming pipeline.
- **Reference model** – automatically cloned from the starting generator to
  anchor KL control.

## Monitoring and Debugging

- `grpo_trainer.py` emits policy gradient loss, KL penalty, reward statistics,
  and advantage moments each step.
- Enable WandB in the config for remote dashboards; otherwise logs print to
  stdout.
- Periodic sample dumps help verify qualitative progress—use `sample_utils.py`
  helpers to inspect completions.

## Historical Notes

Earlier planning documents now live in the archive:

- [`docs/archive/grpo_design_notes.md`](../archive/grpo_design_notes.md)
- [`docs/archive/grpo_implementation_plan_v2.md`](../archive/grpo_implementation_plan_v2.md)

Consult them for background rationale, but rely on the files above for the
maintained implementation.
