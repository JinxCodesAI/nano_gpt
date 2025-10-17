# Discrete Diffusion Playground (nanoGPT fork)

![nanoGPT](assets/nanogpt.jpg)

This repository turns Andrej Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT) into an experimentation bed for **discrete diffusion** on character-level language models. The original nanoGPT license and spirit of minimal, hackable code carry over; this fork rebuilds the data pipeline, adds diffusion-flavoured datasets, and ships utilities for iterative unmasking-based inference.

Takes ideas and inspirations from:

 [BERT](https://arxiv.org/pdf/1810.04805), [LLADA](https://arxiv.org/abs/2502.09992), [D3PM](https://arxiv.org/pdf/2107.03006), with also some ideas and research directions inspired by [Julia Turc channel](https://www.youtube.com/@juliaturc1/videos)

 with a bit of a twist. 

## Highlights

- Attribution: forked from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) under the MIT license.
- Streaming data pipeline with a strict producer/consumer split for long-running experiments.
- Two discrete diffusion recipes:
  - **Char Diffusion** – stage-aware BERT masking with partial targets.
  - **Char Random Replacement** – stochastic replacement objective tuned for editing-style diffusion.
- Ready-to-train configs, inspection scripts, and an inference harness (`sample.py`) that iteratively refines text with configurable insertion, deletion, and re-noising schedules.
- Documentation geared toward tinkering—follow the quick-starts, then dive into [`docs/diffusion_strategies.md`](docs/diffusion_strategies.md) for deeper comparisons and inference tips.

## Installation

> Requires Python 3.10+ and PyTorch 2.0 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install torch numpy wandb tqdm
```

Optional extras for data prep or experimentation:

- [`transformers`](https://github.com/huggingface/transformers) for tokenizer experiments.
- [`tiktoken`](https://github.com/openai/tiktoken) for fast byte-pair encoding utilities.
- [`wandb`](https://wandb.ai/site) logging is enabled by default; configure credentials or disable via CLI.

## Training Quick Start

All experiments rely on the streaming producer/consumer workflow:

1. **Launch the producer.** Generates masked batches and metadata into a filesystem queue.

   ```bash
   python prepare.py config/train_char_diffusion.py
   # swap with config/train_char_random_replacement_targets_full.py for the random-replacement objective
   ```

   The producer streams `.pt` files into `data/<dataset>/queue/{train,val}` and stores schema information in `meta.pkl`.

2. **Train the consumer model.**

   ```bash
   python train.py config/train_char_diffusion.py
   # or python train.py config/train_char_random_replacement_targets_full.py
   ```

   The shared config file is executed first; additional `--key=value` overrides are applied afterwards through the lightweight [`configurator.py`](configurator.py).

3. **Tailor to your hardware.**

   ```bash
   # CPU-only sanity check
   python train.py config/train_char_diffusion.py \
       --device=cpu --compile=False --batch_size=4 --block_size=128

   # Enable verbose streaming logs
   python train.py config/train_char_diffusion.py --data_stream_verbose=True
   ```

   All config variables are overrideable in this style (e.g., `--mask_probability=0.35`, `--wandb_log=False`).

### Choosing a Recipe

| Aspect | Char Diffusion (`config/train_char_diffusion.py`) | Char Random Replacement (`config/train_char_random_replacement_targets_full.py`) |
| --- | --- | --- |
| Corruption | BERT 80/10/10 masking with optional multi-stage unmasking schedules. | Mixture of random token replacement, `[MASK]` inserts, and fragment borrowing; tuned for iterative editing. |
| Targets | Partial labels (`ignore_index` outside masked spans). | Full-token supervision during training to stabilise replacements (toggle via `dataset_partial_targets`). |
| Use cases | Masked-model pretraining, diffusion curricula exploration. | Inference experimentation with insertion/deletion loops and editing-style diffusion. |
| Stage configs | `data/char_diffusion/config/*.py` | Reuses stage configs; adds corruption knobs. |

Switch between them by passing the corresponding config file to both `prepare.py` and `train.py`.

## Inference Quick Start (Random Replacement)

`sample.py` loads a trained checkpoint, iteratively applies insertions/deletions, optionally re-noises tokens, and resamples until convergence.

```bash
python sample.py \
    --out_dir=out-char-random-replacement \
    --ckpt_name=your_checkpoint.pt \
    --start="FILE:prompt.txt" \
    --max_iterations=30 \
    --max_new_tokens=400 \
    --temperature=1.1 \
    --edit_schedule=cosine \
    --insert_ratio_start=0.05 --insert_ratio_end=0.00 \
    --delete_ratio_start=0.04 --delete_ratio_end=0.00 \
    --noise_schedule=cosine --noise_start=0.10 --noise_end=0.00
```

Key parameters are documented in [`docs/diffusion_strategies.md`](docs/diffusion_strategies.md#random-replacement-inference-knobs). Highlights:

- `insert_ratio_*` / `delete_ratio_*`: fraction of the editable span considered for structural edits as diffusion progresses.
- `noise_*`: percentage of editable tokens to re-randomise each iteration; enables stochastic exploration.
- `fix_prompt_during_diffusion`: keeps the conditioning prefix immutable.

Prompts may be supplied inline, as files (`"FILE:path.txt"`), or using the special `<|endoftext|>` token to start from scratch.

## Data Pipeline Architecture

- **Preparation (`prepare.py`)**: Executes the dataset provider defined in your config. Providers (e.g., [`CharDiffusionProvider`](data/char_diffusion/prepare_streaming.py), [`CharRandomReplacementProvider`](data/char_random_replacement/prepare_streaming.py)) stream batches to disk, emit metadata (`meta.pkl`), and respect queue backpressure.
- **Consumption (`train.py`)**: Loads batches through [`DatasetConsumer`](dataset_consumer.py) which blocks on empty queues, prefetches to CPU, and deletes files once they are fully consumed. Metadata is accessible via `consumer.meta` for model code that needs vocabulary IDs or schema details.
- The handshake lets you swap in custom datasets: implement a new provider under `data/<name>/`, point a config’s `dataset` variable at it, and re-use the same training pipeline. See [`docs/datasets_guide.md`](docs/datasets_guide.md) for a full walkthrough.

## Repository Tour

- [`config/`](config) – shared experiment configs consumed by both producer and trainer.
- [`data/`](data) – dataset providers (`char_diffusion`, `char_random_replacement`) plus their configs.
- [`train.py`](train.py) / [`prepare.py`](prepare.py) – core training loop and streaming producer harness.
- [`sample.py`](sample.py) – random-replacement inference driver with iterative editing controls.
- [`sampling_utils.py`](sampling_utils.py) – insertion/deletion primitives shared by inference routines.
- [`docs/`](docs) – supplementary documentation: dataset guide, diffusion strategy notes, design documents.
- [`tests/`](tests) – smoke tests covering queue handling and consumer behaviour.

## Further Reading & Next Steps

- [`docs/diffusion_strategies.md`](docs/diffusion_strategies.md) – diffusion objectives, similarities/differences, inference knobs.
- [`docs/datasets_guide.md`](docs/datasets_guide.md) – how to add a new streaming dataset.
- [`design_document.md`](design_document.md) – original rationale for the streaming rewrite.

Questions or ideas for new experiments? Open an issue or start hacking—this fork is intentionally compact so you can iterate quickly on new discrete diffusion tricks.

## License

Distributed under the MIT license, consistent with the upstream [nanoGPT](https://github.com/karpathy/nanoGPT) project.
