# GRPO Training

Group-Relative Policy Optimization (GRPO) training for single-step fill-in tasks.

## Overview

GRPO is a reinforcement learning approach that trains a generator model to produce high-quality completions by:
1. Generating multiple completions for each masked input
2. Scoring completions with a frozen judge model
3. Computing group-relative advantages to reduce variance
4. Optimizing with a KL divergence penalty to prevent policy collapse

## Architecture

```
grpo/
├── train_grpo.py           # Main training script
├── grpo_trainer.py         # Training loop orchestrator
├── grpo_training_step.py   # Core GRPO algorithm
├── grpo_config.py          # Default configuration
└── README.md               # This file
```

## Key Components

### 1. GRPOTrainingStep (`grpo_training_step.py`)

Implements the core GRPO algorithm:
- Fetches masked inputs from dataset
- Generates k completions per input
- Scores with frozen judge model
- Computes group-relative advantages
- Calculates GRPO loss with KL penalty
- Updates generator weights

### 2. GRPOTrainer (`grpo_trainer.py`)

Orchestrates the training loop:
- Learning rate scheduling
- Logging and metrics
- Checkpointing
- Periodic sampling for quality monitoring

### 3. Configuration (`grpo_config.py`)

Key hyperparameters:
- `group_size`: Number of completions per input (default: 8)
- `kl_beta`: KL divergence penalty coefficient (default: 0.1)
- `learning_rate`: Learning rate for fine-tuning (default: 1e-5)
- `temperature`: Sampling temperature (default: 0.8)
- `top_p`: Nucleus sampling threshold (default: 0.95)

## Usage

### Basic Usage

```bash
python grpo/train_grpo.py config/grpo_config.py
```

### With Custom Parameters

```bash
python grpo/train_grpo.py --group_size=16 --kl_beta=0.2 --learning_rate=5e-6
```

### Required Setup

Before running GRPO training, you need:

1. **Generator checkpoint**: A pretrained model to fine-tune
2. **Judge checkpoint**: A frozen sequence scorer model for rewards
3. **Dataset**: Masked inputs (e.g., from `char_diffusion` dataset)

Example configuration:

```python
# In grpo_config.py or command line
generator_checkpoint = 'out-char-diffusion/checkpoint.pt'
judge_checkpoint = 'out-char-diffusion/judge.pt'
dataset = 'char_diffusion'
```

## How It Works

### Training Loop

For each iteration:

1. **Fetch batch**: Get masked inputs from dataset (B samples)
2. **Generate**: Create k completions for each input (B×k samples)
3. **Score**: Evaluate all completions with judge model
4. **Advantages**: Compute group-relative advantages (per-input baseline)
5. **Loss**: Calculate policy gradient loss + KL penalty
6. **Update**: Backpropagate and update generator weights

### Loss Function

```
Loss = -E[log π(a) × Advantage] + β × KL(π || π_ref)
```

Where:
- `π(a)`: Current policy (generator)
- `Advantage`: Reward - group baseline
- `π_ref`: Reference policy (frozen initial generator)
- `β`: KL penalty coefficient

### Key Features

1. **Group-relative advantages**: Reduces variance by using per-input baselines
2. **KL divergence penalty**: Prevents model from deviating too far from reference
3. **Reuses existing infrastructure**: Leverages `DatasetConsumer`, `predict_and_sample_tokens`, `calculate_judge_scores`
4. **No model modifications**: Computes log-probabilities from existing logits

## Monitoring

### Logged Metrics

- `loss`: Total GRPO loss
- `pg_loss`: Policy gradient component
- `kl_penalty`: KL divergence penalty
- `mean_reward`: Average judge score
- `std_reward`: Reward standard deviation
- `mean_advantage`: Average advantage value
- `mean_kl`: Average KL divergence
- `mean_log_prob`: Average log-probability

### Checkpoints

Checkpoints are saved to `out_dir` (default: `out-grpo/`) every `save_interval` iterations.

### WandB Integration

Enable WandB logging:

```python
wandb_log = True
wandb_project = 'grpo'
wandb_run_name = 'my_grpo_run'
```

## Hyperparameter Tuning

### group_size

- **Larger** (16-32): More stable, slower training
- **Smaller** (4-8): Faster, higher variance
- **Recommended**: Start with 8

### kl_beta

- **Higher** (0.2-0.5): More conservative, stays close to reference
- **Lower** (0.01-0.1): More exploration, may diverge
- **Recommended**: Start with 0.1

### learning_rate

- **Higher** (1e-4): Faster adaptation, may be unstable
- **Lower** (1e-6): Slower, more stable
- **Recommended**: 10-100x lower than pretraining (e.g., 1e-5)

### temperature

- **Higher** (1.0-1.5): More diverse generations
- **Lower** (0.5-0.8): More focused generations
- **Recommended**: 0.8

## Troubleshooting

### Loss not decreasing

- Check that judge model is loaded correctly
- Verify rewards are varying (not all same)
- Try lower learning rate
- Increase group_size for more stable gradients

### KL divergence exploding

- Increase `kl_beta`
- Lower learning rate
- Check that reference model is frozen

### Out of memory

- Reduce `batch_size`
- Reduce `group_size`
- Use gradient accumulation
- Enable gradient checkpointing (requires model modification)

## Differences from Standard Training

| Aspect | Standard Training | GRPO Training |
|--------|------------------|---------------|
| Data | Fixed targets | Generated on-the-fly |
| Loss | Cross-entropy | Policy gradient + KL |
| Objective | Match targets | Maximize judge rewards |
| Models | 1 (generator) | 3 (generator, reference, judge) |
| Batch size | B | B × k (group_size) |

## Code Reuse

GRPO maximally reuses existing codebase:

- `DatasetConsumer`: Fetches masked inputs
- `predict_and_sample_tokens`: Generates completions
- `calculate_judge_scores`: Scores completions
- `CheckpointManager`: Saves checkpoints
- `CosineLRScheduler`: Learning rate scheduling
- `create_logger`: Logging infrastructure

No modifications to existing files required!

## References

- GRPO paper: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- PPO paper: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- RLHF: [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)

