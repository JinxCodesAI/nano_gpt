# Documentation Index

This directory consolidates reference material for the diffusion-first
reimagining of nanoGPT. The documents are grouped so it is easier to find the
right level of detail—whether you are wiring up a training job or digging into
long-term research plans.

## Guides

Practical, task-focused walk-throughs for day-to-day development.

- [Datasets](./guides/datasets.md) – streaming data providers, metadata
  contracts, and how `prepare.py` coordinates with `DatasetConsumer`.
- [Loss Modifiers](./guides/loss_modifiers.md) – entropy weighting, label
  smoothing, mask-ratio compensation, and judge-assisted weighting.
- [GRPO Training](../grpo/README.md) – production reinforcement loop for
  judge-guided refinement with group-relative policy optimisation.
- [Multi-mode Usage](./guides/multi_mode_usage.md) – configuring language
  modeling, token classification, and sequence scoring within a single GPT
  implementation.
- [Sampler Head](./guides/sampler_head.md) – optional bidirectional sampler for
  coherent diffusion decoding.

## Research & Design Notes

Exploratory architectures, critique summaries, and algorithm proposals that
drive the diffusion roadmap.

- [Critic Overview](./research/critic.md) and
  [Implementation Plan](./research/critic_implementation_plan.md) – learned
  error detection for iterative re-masking.
- [Critic Review](./research/critic_review.md) – external feedback and risk
  assessment.
- [GRPO Design Notes](./research/grpo.md) &
  [Implementation Plan v2](./research/grpo_v2.md) – historical planning memos
  for the reinforcement fine-tuning stack (see the guide above for the
  maintained implementation).
- [Hierarchical U-Net Transformer](./research/unet_transformer.md) – discrete
  diffusion architecture for long-form coherence.
- [Variable Length Line Study](./research/sep_varlen_lines_report.md) – dataset
  analysis supporting sequence scoring work.

## Archive

Historical scratch files, early experiments, and other artefacts that may be
useful for context but are not actively maintained.

- [Raw notes and prototypes](./archive/raw_notes/) – includes exploratory error
  logs, early improvement proposals, and prototype scripts.

If you add new documents, please drop them into one of these categories and
update this index so future contributors can navigate the repository quickly.
