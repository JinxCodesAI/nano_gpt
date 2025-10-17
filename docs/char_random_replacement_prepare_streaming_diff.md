# Char Random Replacement Provider – Key Differences

`CharRandomReplacementProvider` subclasses `CharDiffusionProvider` and overrides the corruption path. The points below capture the behaviours that diverge from the base implementation.

- **Corruptor construction** – `_initialize_corruptor` builds a `RandomReplacementCorruptor` with explicit exclusion lists and trainside mixture weights `(0.6, 0.2, 0.2)` for random tokens, mask tokens, and fragment borrowing respectively (`data/char_random_replacement/prepare_streaming.py:29`, `data/char_random_replacement/prepare_streaming.py:40`).
- **Fragment resampling** – `_build_fragment_sampler` draws aligned slices from the training corpus to support the fragment branch of `apply_mixed_corruption`, something the base provider never attempts (`data/char_random_replacement/prepare_streaming.py:59`).
- **Train/validation split awareness** – The subclass tracks `_active_corruption_split`, reapplying the mixture only for the training split while validation batches fall back to straight random replacement (`data/char_random_replacement/prepare_streaming.py:52`, `data/char_random_replacement/prepare_streaming.py:70`, `data/char_random_replacement/prepare_streaming.py:88`).
- **Optional full targets** – `_create_labels` can emit full-identity targets for the training split when `--no-dataset_partial_targets` is passed, extending beyond the base class’ partial-target contract (`data/char_random_replacement/prepare_streaming.py:100`).
- **Metadata extensions** – `build_meta` annotates the dataset with the corruption type, mixture multiplier, extra special tokens, and whether partial targets are active, information absent from the parent provider (`data/char_random_replacement/prepare_streaming.py:110`).
- **Reusable stage config loader** – `_load_stage_kwargs` encapsulates composition-config discovery so both the CLI wrapper and other entrypoints can share it (`data/char_random_replacement/prepare_streaming.py:126`).
- **Expanded CLI** – The `main` entrypoint adds knobs for the corruption multiplier, extra special token IDs, and dataset target policy, plus a fallback that reuses the `char_diffusion` corpus if a local `input.txt` is missing (`data/char_random_replacement/prepare_streaming.py:158`).

All other sampling and queue mechanics are inherited unchanged from the base provider.
