# Char Random Replacement Streaming Dataset

`CharRandomReplacementProvider` extends `CharDiffusionProvider` with a corruption pipeline that replaces masked characters using a configurable random-replacement mixture. All queue handling, stage scheduling, and sampling utilities are inherited from the base provider.

## Corruption Strategy

- **Mixture** – Training batches use `(0.8, 0.2, 0.0)` weights for random token replacement, direct `[MASK]` insertion, and fragment borrowing from another slice of the training corpus. Override this trio via `--train_corruption_mixture` or the `train_corruption_mixture` entry in your config.
- **Corruptor** – `_initialize_corruptor` builds `RandomReplacementCorruptor`, excluding the `[MASK]` token and any IDs passed via `--extra_special_token_ids`. The original-token multiplier (`--original_token_probability_multiplier`) can bias the random branch toward leaving the source token unchanged; tune it on the CLI or through `original_token_probability_multiplier` in configs.
- **Fragments** – `_build_fragment_sampler` draws aligned samples from the train split so fragment corruption can drop realistic context into the masked positions.
- **Split awareness** – Validation batches skip the fragment pipeline and fall back to plain random replacement, keeping evaluation deterministic with respect to the corruption distribution.

## Target Options

By default the provider mirrors the base class and emits partial targets (only masked positions contribute to the loss). Passing `--no-dataset_partial_targets` switches the training split to full-identity labels while validation stays partial, which can simplify debugging. The same toggle is available when launching through `prepare.py`; set `dataset_partial_targets = False` in your config (see `config/train_char_random_replacement_targets_full.py`) to request full targets without touching the CLI flags.

## CLI Additions

Run `python data/char_random_replacement/prepare_streaming.py --help` to view the extended arguments:

- `--original_token_probability_multiplier`
- `--train_corruption_mixture RANDOM MASK FRAGMENT`
- `--extra_special_token_ids`
- `--dataset_partial_targets / --no-dataset_partial_targets`

When `input.txt` is missing from this directory, the CLI automatically falls back to `data/char_diffusion/input.txt`, letting both providers share the same corpus.

> **Config note:** `prepare.py` now forwards `dataset_partial_targets`, `original_token_probability_multiplier`, `train_corruption_mixture`, and `extra_special_token_ids` to the provider. See `config/train_char_random_replacement_targets_full.py` for an example wiring that toggles full targets and documents how to adjust the corruption knobs from a config file.

## Usage

- Via the shared harness: `python prepare.py config/train_char_random_replacement.py`
- Standalone for experimentation: `python data/char_random_replacement/prepare_streaming.py --batch_size 32 --block_size 256`

All other knobs (`batch_size`, `block_size`, `mask_probability`, stage configs, queue sizing) match the base dataset. Stage composition configs under `config/` can be reused as-is because mask selection occurs upstream of the corruption logic. When `use_all_stages_for_training=True`, the provider follows the same mixing strategy as the base implementation: it pre-builds batches for every configured stage, merges and shuffles them, and re-splits the pool so each emitted batch blends samples drawn from all stages rather than sticking to a single-stage slice.
