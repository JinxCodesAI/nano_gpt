# Char Diffusion Provider – Key Differences

This note outlines the behaviours that remain unique to the base `CharDiffusionProvider` when compared with `CharRandomReplacementProvider`.

- **BERT-style corruption** – `_apply_corruption` simply applies the 80/10/10 BERT masking rule implemented in `apply_bert_style_corruption_cpu`, without any fragment resampling or corruption mixtures beyond the classic mask/random/identity split (`data/char_diffusion/prepare_streaming.py:18`, `data/char_diffusion/prepare_streaming.py:153`).
- **Stage masking passthrough** – `_apply_stage_corruption` trusts the tensors returned by `apply_stage_masking` and does not revisit the mask to add extra corruptions; the random-replacement subclass reprocesses stage batches for the train split (`data/char_diffusion/prepare_streaming.py:163`).
- **Label generation contract** – Training and validation always use partial targets, emitting `ignore_index` where the mask is false; there is no toggle for full-identity labels like the random-replacement variant offers (`data/char_diffusion/prepare_streaming.py:177`).
- **Metadata payload** – Metadata focuses on generic MLM fields (dataset name, vocab, mask stats, batch schema) and omits corruption-specific knobs that the subclass surfaces (`data/char_diffusion/prepare_streaming.py:315`).
- **Stage scheduling bootstrap** – Stage configurations are expanded in-place via `_ensure_stage_cycle` and related helpers; configuration loading happens inline in `main` without the reusable loader that the random-replacement script adds (`data/char_diffusion/prepare_streaming.py:218`, `data/char_diffusion/prepare_streaming.py:335`).
- **CLI surface area** – The standalone entrypoint only exposes the core streaming controls plus an optional composition config; there are no options for corruption mixtures, extra special tokens, or dataset target policies (`data/char_diffusion/prepare_streaming.py:335`).

Shared components—vocabulary construction, line-aligned sampling, stage mixing buffers, and queue management—are inherited wholesale by the random-replacement provider and therefore are not reiterated here.
