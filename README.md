# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

nanoGPT is now a focused playground for **discrete diffusion** experiments built on top of the original minimal GPT training loop. Instead of chasing GPT-2 reproductions, the repository ships with a streaming data pipeline and training recipe for a character-level masked language model that can be used as the backbone for diffusion/unmasking research.

The core pieces you will interact with are:

- `config/train_char_diffusion.py` – shared configuration used by both the data producer (`prepare.py`) and the trainer (`train.py`).
- `data/char_diffusion/` – self-contained dataset provider that performs BERT-style masking over Shakespeare text and streams batches to disk.
- `train.py` – the generic training loop that now defaults to the discrete diffusion configuration but is still hackable for custom experiments.

## Install

```sh
pip install torch numpy wandb tqdm
```

Optional extras:

- [`transformers`](https://github.com/huggingface/transformers) and [`tiktoken`](https://github.com/openai/tiktoken) are only required if you plan to use the legacy sampling utilities in `sample.py`.

## Quick start: character diffusion workflow

1. **Launch the streaming data producer.** This reads `data/char_diffusion/input.txt`, applies stage-aware BERT corruption, and writes queue files under `data/char_diffusion/queue/{train,val}`.

    ```sh
    python prepare.py config/train_char_diffusion.py
    ```

    The producer prints progress as it generates `.pt` files that contain masked inputs (`x`) and reconstruction targets (`y`). You can stop the process at any time—the consumer will keep reusing the existing backlog.

2. **Train the diffusion-friendly masked model.** In a new shell (or after the producer has filled a backlog) run:

    ```sh
    python train.py config/train_char_diffusion.py
    ```

    The config sets up an 8-layer bidirectional Transformer with rotary position encodings, half precision training, and learning-rate warmup tailored for masked language modeling. Checkpoints land in `out-char-diffusion/`.

3. **Customize to your hardware.** Example overrides:

    ```sh
    # CPU-only quick smoke test
    python train.py config/train_char_diffusion.py --device=cpu --compile=False --batch_size=4 --block_size=128

    # Enable more aggressive logging from the streaming pipeline
    python train.py config/train_char_diffusion.py --data_stream_verbose=True
    ```

    Any flag accepted by `train.py` can be provided on the command line; config values are overridden in the same way as the classic nanoGPT workflow.

## Understanding the dataset provider

`data/char_diffusion/prepare_streaming.py` implements `CharDiffusionProvider`, a subclass of `DataProviderBase` that:

- Splits Shakespeare text into train/val character streams and adds a dedicated `[MASK]` token.
- Applies 80/10/10 BERT-style corruption with optional multi-stage unmasking schedules (see `data/char_diffusion/config/`).
- Emits batch files that include rich metadata (`meta.pkl`) describing the vocabulary, schema, and masking hyperparameters for the consumer.

The producer/consumer handshake allows you to swap in your own dataset by creating a new folder under `data/` with a provider module that exports another `DataProviderBase` subclass and pointing a config file at it.

## Streaming workflow tips

- The consumer blocks until files exist; always start `prepare.py` before `train.py` or pre-populate the queue.
- Each batch file is deleted after all ranks consume it. If you change masking settings, clear `data/char_diffusion/queue/` to avoid mixing schemas.
- Metadata such as `mask_token_id` and `vocab_size` is automatically propagated into the training run (`consumer.meta`). Access it from your model code when experimenting with new objectives.

## Efficiency notes

`train.py` remains identical in spirit to the original nanoGPT script. It supports PyTorch 2.0 compilation, gradient accumulation, and distributed training. The default configuration now targets masked modeling, but causal language modeling remains possible by swapping in your own config.

`bench.py` can still be used for micro-benchmarks. By default it operates on synthetic data so you can measure kernel throughput without preparing a dataset. Set `real_data=True` and adapt the loader if you build a new provider with a memmap-compatible format.

## Troubleshooting

- `ValueError: Could not find provider module ...` – ensure your config's `dataset` matches a folder under `data/` with a `prepare_streaming.py` file.
- `Queue directory not found` – run the producer first so that `data/<dataset>/queue/{train,val}` exists.
- PyTorch compile issues – add `--compile=False` when running on unsupported platforms (e.g., older CUDA stacks or some CPU builds).

For additional context on the design of the streaming pipeline, see `design_document.md`.
