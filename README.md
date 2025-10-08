# nanoGPT Diffusion Stack

This repository evolves the original nanoGPT codebase into a
diffusion-oriented research stack for long-form text generation. The
project keeps the small, hackable ethos of nanoGPT while layering on the
infrastructure required to explore discrete diffusion, multi-head models,
and streaming data pipelines.

## Highlights

- **Diffusion-first training loop** – `train.py` and the configs under
  `config/train_char_diffusion*.py` run iterative unmasking experiments on
  top of the GPT backbone, including critic-guided refinement and optional
  sampler heads.【F:train.py†L1-L29】【F:config/train_char_diffusion.py†L1-L65】
- **Streaming data pipeline** – `prepare.py`, `dataset_consumer.py`, and
  the dataset providers in `data/` coordinate background batch generation
  with strong metadata contracts so training can begin before the full
  dataset is materialised.【F:prepare.py†L1-L57】【F:dataset_consumer.py†L1-L80】
- **Multi-mode GPT** – `model.py` supports language modeling, token
  classification, sequence scoring, critic heads, and optional sampler
  modules from a single configuration surface.【F:model.py†L1-L118】
- **Loss modifier toolbox** – Configurable entropy weighting, label
  smoothing, mask-ratio compensation, and judge-based weighting live in
  `loss_modifiers/` and are documented in `docs/guides/loss_modifiers.md`.
- **Interactive tooling** – `interactive_diffusion_explorer.py` allows
  live inspection of diffusion checkpoints, masking strategies, and critic
  diagnostics from the terminal.【F:interactive_diffusion_explorer.py†L1-L64】

## Getting Started

1. **Install dependencies**

   ```bash
   pip install torch numpy transformers datasets tiktoken wandb tqdm keyboard
   ```

   The additional `keyboard` package enables optional interactive controls
   in the explorer utility.

2. **Prepare streaming batches**

   Use the same config that you plan to train with so the provider and
   consumer share metadata:

   ```bash
   python prepare.py config/train_char_diffusion.py
   ```

   This will start a background producer that writes batch files into the
   dataset directory defined by the config (e.g. `data/char_diffusion`).

3. **Launch training**

   ```bash
   python train.py config/train_char_diffusion.py
   ```

   The configuration toggles the critic head, sampler head, loss modifiers,
   and other diffusion-specific features. See `config/train_char_diffusion_*`
   variants for curated experiments, including critic-guided weighting and
   ablations.

4. **Sample or inspect**

   Generate sequences with the scripted sampler or explore them
   interactively:

   ```bash
   python sample.py --out_dir=out-char-diffusion
   python interactive_diffusion_explorer.py
   ```

   The explorer can reload checkpoints on the fly, visualise masking
   schedules, and surface critic scores for debugging.

## Documentation

All documentation has been reorganised. Start with
[docs/README.md](docs/README.md) for a full index. Highlights include the
datasets guide, loss modifier handbook, sampler head walkthrough, and the
research notes covering critic design, GRPO experiments, and the
Hierarchical U-Net Transformer proposal.

## Repository Layout

- `config/` – experiment presets for diffusion, critic heads, multi-mode
  inference, and evaluation.
- `core/` – shared training orchestration (trainer, scheduler, logging,
  timing utilities).
- `data/` – dataset-specific streaming providers and masking utilities.
- `docs/` – structured documentation (guides, research notes, archives).
- `loss_modifiers/` – modular loss transformation pipeline.
- `sample_utils.py`, `interactive_diffusion_explorer.py` – inference tools
  for iterative unmasking and critique-driven sampling.

## Contributing

Pull requests are welcome! Keep configs reproducible, update the docs index
when adding new material, and prefer the streaming pipeline for new
datasets so the training loop stays consistent.
