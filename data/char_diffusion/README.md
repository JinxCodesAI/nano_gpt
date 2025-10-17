# Char Diffusion Streaming Dataset

`CharDiffusionProvider` streams character-level masked-language-modeling (MLM) batches to the filesystem queue consumed by the training pipeline. It builds a vocabulary from `input.txt`, applies BERT-style corruption, and can mix multiple masking stages for diffusion-style curricula.

## Key Files

- `prepare_streaming.py` – provider implementation and standalone CLI.
- `masking_utils.py` – masking primitives (`random`, `sticky`, `span`) used by stage configurations.
- `file_utils.py` – atomic queue helpers reused by the CLI.
- `config/*.py` – optional stage presets (see `config/complex.py`).
- `input.txt` – source corpus; edit or replace to build a new vocabulary.

## Workflow

1. **Corpus bootstrap** – On construction the provider reads `input.txt`, sorts the unique characters, and reserves an extra `[MASK]` token. It then materializes 90/10 train/validation splits on device-backed tensors.
2. **Batch sampling** – Each batch samples `batch_size` line-aligned spans of length `block_size` from the requested split and draws a boolean mask using `mask_probability`.
3. **Corruption** – Masked positions follow the classic BERT 80/10/10 rule: 80% `[MASK]`, 10% random replacement, and 10% left unchanged.
4. **Targets** – Labels copy the original token where the mask is true and fill `ignore_index` otherwise, matching the PyTorch MLM loss contract.
5. **Queue output** – `DataProviderBase` batches `batches_per_file` samples, writes them to `data/char_diffusion/queue/{train,val}` with deterministic filenames, and pauses when the backlog exceeds `max_backlog_files`.

Metadata saved next to the queue captures the vocabulary, mask settings, and `(x, y)` schema so consumers can validate compatibility.

## Stage-Based Masking (Optional)

Supplying `use_all_stages_for_training=True` together with `unmasking_stages` and `validation_stages` enables stage mixing. Each stage dictionary points to the masking type plus parameters consumed by `masking_utils.apply_stage_masking`. The provider cycles through the configured stages, pre-builds batches for each, shuffles the results into a single pool, and re-splits them so every emitted batch contains a mixture of samples from all configured stages rather than a single-stage slice.

Stage definitions can be passed directly via the experiment config or loaded from a module under `config/` using the `--composition_config` CLI flag.

## Running the Producer

- Through the shared harness: `python prepare.py config/train_char_diffusion.py`
- Standalone for debugging: `python data/char_diffusion/prepare_streaming.py --help`

The CLI exposes the same knobs as the harness (`batch_size`, `block_size`, `mask_probability`, `batches_per_file`, queue sizing arguments, and `--composition_config`). Enable `--verbose` to watch queue progress; production restarts from the recorded sequence numbers when the process is restarted.

## Practical Tips

- Replace or edit `input.txt` to train on a different corpus; the provider will rebuild the vocabulary automatically.
- Clear `data/char_diffusion/queue` when changing the vocabulary, `block_size`, or masking regime to avoid mixing incompatible batches with older runs.
- Inspect a generated file with `torch.load(path)` if you need to confirm tensor shapes or mask density before starting a long training job.
