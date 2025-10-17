# Char Random Replacement Provider – Key Differences

`CharRandomReplacementProvider` subclasses `CharDiffusionProvider` and overrides the corruption path. The points below capture the behaviours that diverge from the base implementation.

- **Configurable corruption mixture** – The constructor now accepts `train_corruption_mixture`, coercing it to a validated `(random, mask, fragment)` triple that defaults to `(0.8, 0.2, 0.0)` before `_initialize_corruptor` reuses it when calling `apply_mixed_corruption` (`data/char_random_replacement/prepare_streaming.py:21`, `data/char_random_replacement/prepare_streaming.py:30`, `data/char_random_replacement/prepare_streaming.py:77`).
- **Fragment resampling** – `_build_fragment_sampler` draws aligned slices from the training corpus to support the fragment branch of `apply_mixed_corruption`, something the base provider never attempts (`data/char_random_replacement/prepare_streaming.py:59`).
- **Train/validation split awareness** – The subclass tracks `_active_corruption_split`, reapplying the mixture only for the training split while validation batches fall back to straight random replacement (`data/char_random_replacement/prepare_streaming.py:52`, `data/char_random_replacement/prepare_streaming.py:70`, `data/char_random_replacement/prepare_streaming.py:88`).
- **Optional full targets** – `_create_labels` can emit full-identity targets for the training split when `--no-dataset_partial_targets` is passed, extending beyond the base class’ partial-target contract (`data/char_random_replacement/prepare_streaming.py:100`).
- **Metadata extensions** – `build_meta` annotates the dataset with the corruption type, mixture weights, mixture multiplier, extra special tokens, and whether partial targets are active, information absent from the parent provider (`data/char_random_replacement/prepare_streaming.py:112`).
- **Reusable stage config loader** – `_load_stage_kwargs` encapsulates composition-config discovery so both the CLI wrapper and other entrypoints can share it (`data/char_random_replacement/prepare_streaming.py:151`).
- **Expanded CLI** – The `main` entrypoint now exposes knobs for the corruption multiplier, the train mixture, extra special token IDs, and dataset target policy, plus a fallback that reuses the `char_diffusion` corpus if a local `input.txt` is missing (`data/char_random_replacement/prepare_streaming.py:186`).

All other sampling and queue mechanics are inherited unchanged from the base provider.
