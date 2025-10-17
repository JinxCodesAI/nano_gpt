# Discrete Diffusion Strategies

This note summarises the two diffusion-style objectives provided in the repo, how they reuse the streaming pipeline, and how to control the iterative inference loop. Treat it as a launch pad for designing your own corruption schemes or extending the sampling logic.

## 1. Approaches at a Glance

| Dimension | Char Diffusion (`data/char_diffusion`) | Char Random Replacement (`data/char_random_replacement`) |
| --- | --- | --- |
| Core idea | Mask tokens BERT-style and train the model to reconstruct them. Optional stage schedules gradually reduce mask density to mimic diffusion. | Replace tokens using a configurable mixture of random IDs, explicit `[MASK]`, and fragments sampled from other spans. Targets default to full supervision to stabilise edits. |
| Training config | `config/train_char_diffusion.py` | `config/train_char_random_replacement_targets_full.py` (toggle `dataset_partial_targets` as needed). |
| Use cases | Curriculum research, analysis of denoising schedules, masked-model pretraining. | Editing-friendly diffusion, insertion/deletion experiments, inference benchmarking. |
| Queue format | `(x, y)` tensors with `ignore_index` applied outside masked spans. | `(x, y)` tensors with full targets by default; metadata includes corruption mixture and fragment settings. |

### Shared DNA

- Both providers subclass [`DataProviderBase`](../data/char_diffusion/prepare_streaming.py) and rely on the same queue semantics.
- Stage configurations (`data/<dataset>/config/*.py`) define masking density, mask types (`span`, `sticky`, `random`), and stage weights. Char Random Replacement consumes the same stage definitions before applying its corruption mixture.
- Metadata (`meta.pkl`) carries vocabulary details, `ignore_index`, stage summaries, and corruption knobs so the training loop can validate compatibility.

### Key Differences

- **Corruption target**: Char Diffusion keeps targets sparse; Char Random Replacement can emit full labels so the loss is defined everywhere, which helps the model learn when to maintain or alter tokens.
- **Fragment sampling**: The random replacement provider can inject contiguous snippets from the train split, encouraging coherent insertions. This path is disabled for validation to keep evaluation deterministic.
- **Inference support**: All sampling utilities live in `sampling_utils.py`, but only the random replacement recipe has a ready-made iterative inference harness in [`sample.py`](../sample.py).

## 2. Producer vs. Consumer Responsibilities

The pipeline deliberately separates dataset authoring from training so experiments can iterate on either side independently:

- **Producers** (`prepare.py` + provider module) own *what data looks like*. They stream batches into `data/<dataset>/queue/{split}`, enforce metadata consistency, and honour backpressure when the queue fills up. Providers control corruption schedules, masking ratios, and label semantics.
- **Consumers** (`train.py` + [`DatasetConsumer`](../dataset_consumer.py)) own *how data is used*. The consumer blocks until files exist, prefetches to host memory, deletes batches after use, and exposes `consumer.meta` to the model/trainer. Because producers describe their schema, the training loop can adapt to richer batch dictionaries without changing the queue format.

When you design a new experiment:

1. Clone an existing provider, adjust the corruption/label logic, and drop it under `data/<your_dataset>/`.
2. Point a config’s `dataset` value at your new folder.
3. Launch the familiar `prepare.py` / `train.py` pair—no training code changes required unless your schema deviates from the default `x`/`y`.

Refer to [`datasets_guide.md`](datasets_guide.md) for the boilerplate steps.

## 3. Char Diffusion Deep Dive

The char diffusion provider follows classic masked language modelling but augments it with stage-based masking:

- **Masking utilities**: `masking_utils.py` implements random, span, and sticky masking. Stage configs choose one or mix several.
- **Stages**: Define a sequence of dictionaries with `mask_probability`, `mask_type`, and optional arguments (e.g. `span_length`). When `use_all_stages_for_training=True`, the provider samples from every stage per batch to approximate diffusion’s gradual denoising.
- **Targets**: Labels are `ignore_index` outside masked spans. This keeps the loss focused on denoising performance and mimics masked LM training.
- **Best for**: Curriculum comparisons (how fast to anneal mask density), architecture tweaks (e.g., positional encodings), or prepping checkpoints for downstream diffusion tasks.

## 4. Char Random Replacement Deep Dive

The random replacement provider keeps the stage scheduling machinery but swaps the corruption step:

- **Mixture weights** (`train_corruption_mixture`): Tuple `(random, mask, fragment)` controlling how masked tokens are replaced. Values default to `(0.8, 0.2, 0.0)`; increase the fragment weight to inject longer edits.
- **Original token multiplier**: Bias the random branch toward keeping the original character. Raising this above `1.0` produces gentler corruption that favours high-confidence substitutions.
- **Fragment sampler**: Draws spans from the train split so fragment replacements are plausible; automatically disabled for validation to keep metrics deterministic.
- **Targets**: Full labels (no `ignore_index`) when `dataset_partial_targets=False`. This matches the needs of the inference loop, which benefits from the model understanding when to preserve vs. change tokens.
- **Best for**: Iterative editing, ablation of insertion/deletion policies, experimenting with hybrid discrete diffusion objectives.

## 5. Random Replacement Inference Knobs

[`sample.py`](../sample.py) orchestrates iterative editing rounds. Three parameters often raised while experimenting are summarised here:

- **`insert_ratio_start` / `insert_ratio_end` & `delete_ratio_start` / `delete_ratio_end`**  
  Control how aggressively structural edits are attempted across the diffusion schedule. Ratios are interpreted as fractions of the editable span (tokens beyond the fixed prompt). At each iteration the code converts the ratio into `k_ins` / `k_del`, i.e., the number of insertions or deletions to try.  
  - High starting ratios (e.g. `0.08`) encourage large edits early in the process when the sample is rough.  
  - Decay toward zero ensures the sequence stabilises in later iterations.  
  - Cooling masks prevent back-to-back edits in neighbouring positions; keep ratios modest (<0.10) to avoid thrashing.

- **`noise_start` / `noise_end`**  
  Determine the fraction of editable tokens to re-randomise (`apply_re_noise`) each round. This acts like temperature annealing for diffusion:  
  - A non-zero `noise_start` (e.g. `0.15`) injects stochasticity at the beginning, helping the model escape local optima after insertions/deletions.  
  - Gradually decaying to zero lets the sequence settle while still allowing the model to overwrite low-confidence choices when noise is present.

Combine these knobs to match your editing goals. For example:

- **Creative rewrite**: `insert_ratio_start=0.08`, `delete_ratio_start=0.06`, `noise_start=0.12`, cosine schedules throughout. Encourages bold edits early, calmer refinement later.
- **Conservative polish**: `insert_ratio_start=0.02`, `delete_ratio_start=0.01`, `noise_start=0.05`, short run (`max_iterations=10`). Keeps most of the prompt intact while fixing glitches.

Remember that all parameters can be overridden from the command line thanks to `configurator.py`, so you can launch batches of experiments without editing `sample.py`.

## 6. Experiment Ideas

- Swap the Shakespeare corpus with a domain-specific dataset by replacing `input.txt` and rebuilding the queue—observe how fragment corruption behaves with technical language.
- Prototype a third provider that mixes span masking with auxiliary labels (e.g., POS tags) and extend the consumer to return richer batch dictionaries.
- Modify `sampling_utils.py` to add substitution scoring based on entropy thresholds instead of top-k selection, then measure its impact on coherence.

Keep notes as you explore; this repository is meant to stay hackable and concise so new diffusion tricks are easy to prototype.
