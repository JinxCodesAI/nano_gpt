# Char Random Replacement Streaming Dataset

This dataset mirrors the streaming infrastructure of `data/char_diffusion` but
swaps BERT-style corruption for a configurable random replacement strategy. The
provider inherits the sampling loop, stage masking (`random`, `sticky`,
`span`), and queue handling from `CharDiffusionProvider` while overriding the
corruption step.

## Key differences from `char_diffusion`

- **Mask selection** – Masked positions are generated exactly as in
  `char_diffusion` so existing stage compositions remain valid.
- **Corruption** - Masked characters in the training split now follow a
  three-way mix: 60% random token replacement (respecting the configurable
  `original_token_probability_multiplier`), 20% literal `[MASK]` insertion, and
  20% borrowing characters from a randomly sampled fragment of the training
  corpus at the same positions. Validation batches retain the original pure
  random-replacement behaviour.
- **Extensibility** – The corruption logic lives in
  `corruption_utils.RandomReplacementCorruptor`, making it easy to add new
  replacement rules in future revisions.

## Stage composition configs

Stage schedules live under `config/`. They follow the same structure as the
`char_diffusion` configs, meaning you can either reuse the shared
`config/complex.py` proxy or start from the included
[`config/example.py`](config/example.py) file and adjust the stage dictionaries
for your experiment. A ready-to-run training/prepare configuration is available
at [`config/train_char_random_replacement.py`](../../config/train_char_random_replacement.py);
use it as-is or copy it into your own experiment directory as a template.

## Usage

Run `prepare.py` with a config that sets `dataset='char_random_replacement'`
and either copies `input.txt` into this directory or points `data_dir` at
`data/char_diffusion` to reuse the existing Shakespeare corpus. All other knobs
(`batch_size`, `block_size`, stages, etc.) work identically to the original
dataset.
