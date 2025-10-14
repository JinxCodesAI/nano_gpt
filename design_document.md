## Parallel Data Pipeline Design: DatasetConsumer & DataProvider

### Overview
This document proposes a producer–consumer data pipeline that allows preparing data in parallel to training. A dataset-specific DataProvider produces batch files lazily with file-system backpressure. A generalized DatasetConsumer replaces BatchManager, consuming batches on demand, exposing dataset metadata, and coordinating with the provider via atomic file operations.

The design preserves backward compatibility with precomputed batch files and extends to diverse tasks (LM, BERT-style MLM, classification, token classification), with explicit, strict metadata and schema.

### Goals
- Enable prepare.py to run concurrently with train.py
- Backpressure: provider sleeps when backlog is high; consumer deletes consumed files to signal demand
- Dataset-agnostic base provider class; dataset scripts override only sample logic
- Strict metadata contract (no silent fallbacks); fail fast on misconfiguration
- Minimal logging by default; opt-in verbosity
- Efficient batch loading: pinned memory, non-blocking transfers, circular iteration
- Extensible batch schema beyond (X, Y) (e.g., input_ids, attention_mask, labels)
- Compatible with multi-GPU/DDP (single consumer process per rank)

### Non-goals (phase 1)
- Multi-consumer coordination (multiple trainers) reading the same queue
- Network/distributed queues; we stick to local filesystem semantics
- Cross-process locks beyond atomic rename/delete

### Current State (baseline)
- data/*/prepare.py pre-generates all train/val batch files: `{split}_batches_####.pt` containing x, y tensors and minimal metadata; meta.pkl stores vocab, stoi/itos, shapes.
- BatchManager scans directory, loads a file, slices per-batch, circularly advances, repeats to fill last batch.
- train.py depends on BatchManager.get_batch(split, device) and meta.pkl for vocab_size.

### Proposed Architecture
- DatasetConsumer (new): drop-in replacement for BatchManager. Adds:
  - Waiting for availability; blocking semantics when no batches exist
  - Cache window of loaded files (configurable) for reuse
  - Deletion of files after consumption to drive provider backpressure
  - Expose meta (lazy-load meta.pkl) and batch schema
  - Backward compatibility mode: support existing `{split}_batches_*.pt` files with {x,y} only
- DataProvider Base (new): abstract class in data/common that:
  - Standardizes directory layout, file naming, atomic writes, backlog monitoring
  - Provides a production loop (train/val) with sleep/backoff policy
  - Persists meta.pkl (required) and batch files with a typed schema
  - Delegates dataset-specific sampling to overrides

### Directory Layout & Files
- Dataset root: data/<dataset_name>/
  - meta.pkl (required)
  - queue/
    - train/
    - val/
  - archives/ (optional, rotated or compressed old files)
  - input.* (dataset raw files as needed)

Back-compat: DatasetConsumer also reads legacy files at root matching `{split}_batches_*.pt` if queue/* is absent.

### File Naming & Atomicity
- Provider writes to a temp path and atomically renames on completion:
  - Temp: `{split}/.tmp-{timestamp}-{seq}.pt`
  - Final: `{split}/{timestamp}-{seq}-{nb}{ext}` where:
    - timestamp: UTC millis since epoch (monotonic enough)
    - seq: zero-padded sequential counter (per split)
    - nb: number of batches in file
    - ext: `.pt`
- Only files without the `.tmp-` prefix are visible to consumers.
- On Windows, use os.replace for atomic rename on same filesystem.

### Batch File Format (generalized)
Each batch file is a torch.save of a dict:
- metadata: {
  - batch_size, num_batches, file_idx (seq), split, produced_at
  - schema: list[FieldSpec] where FieldSpec = {
    - name: str (e.g., "x", "y", "input_ids", "attention_mask", "labels")
    - dtype: torch dtype name (e.g., "int64", "float32")
    - shape: [sample_dims...] excluding batch axis
    - role: optional hint (e.g., "input", "target", "mask")
  }
}
- tensors: a dict[name -> Tensor] of shape [num_batches * batch_size] + shape from schema

This mirrors current concatenation strategy and still slices contiguous chunks per batch at read time.

### meta.pkl (strict, required)
- Required keys:
  - dataset_name (str)
  - training_type (enum: LM, MLM, CLASSIFICATION, TOKEN_CLASSIFICATION, SEQUENCE_SCORING)
  - vocab_size (int) if applicable
  - batch_size (int) default batch size used by provider (consumer may warn if different)
  - block_size (int) if applicable
  - target_size (int) if applicable
  - batch_schema (same structure as in files; describes default training split schema)
  - split_info: {train: {...}, val: {...}} such as sampling windows, counts, etc.
- Optional keys (dataset-specific): stoi/itos, label_map, cls_token_id, pad_token_id, mask_token_id, etc.
- Misconfiguration: throw if missing required keys when consumer starts.

### DatasetConsumer API
- init(data_dir: str, device_type: str = "cuda", prefer_queue: bool = True, cache_files: int = 1, high_watermark: int = 2, low_watermark: int = 0)
  - prefer_queue: if True, use queue/train & queue/val; else fallback to legacy pattern
  - high_watermark/low_watermark are advisory for monitoring/logging; actual backpressure is provider-driven
- properties
  - meta: dict (lazy-loaded from meta.pkl)
  - schema(split) -> list[FieldSpec]
  - stats() -> dict of counters and current file/batch indices
- methods
  - get_batch(split: "train"|"val", device) ->
    - default: returns (X, Y) for LM-style if schema only includes x/y
    - otherwise: returns dict[str, Tensor] adhering to schema
  - wait_for_data(split, timeout=None) -> bool
  - reset_state(split=None)
  - set_batch_size(new_bs) (affects slicing and warning if mismatch)

Behavior
- On first call, scan queue/{split} for files sorted by (timestamp, seq); ignore temp files.
- If no available file: block with exponential backoff (sleep), optionally timeout.
- When a file is selected:
  - Load tensors to CPU once, keep in memory until all batches are consumed (cache window)
  - Slice by [start:end] with current batch index
  - When all batches from a file are consumed, delete file (os.remove) to signal provider
- Pin to device_type; use .pin_memory().to(device, non_blocking=True) for CUDA
- Repeat-last padding behavior maintained for compatibility (repeat samples to fill short last batch)

### DataProvider Base API
Location: data/common/provider_base.py

- class DataProviderBase:
  - __init__(self, data_dir, batch_size, block_size=None, target_size=None, batches_per_file=100, max_backlog_files=2, sleep_seconds=2.0, seed=1337, verbose=False)
  - abstract hooks (dataset-specific override minimal set):
    - build_meta(self) -> dict: fill required keys; base will complete/validate
    - sample_batch(self, split: str, rng: torch.Generator) -> dict[str, Tensor]: returns one batch according to meta.batch_schema
      - Must create tensors with shapes [batch_size, ...] per field schema
    - optional: on_start(self), on_stop(self)
  - provided methods:
    - run(self, splits=("train","val")): create dirs, write meta if absent, then per-split production loop
    - produce_one_file(self, split, seq):
      - calls sample_batch split batches_per_file times
      - concatenates along batch dimension, builds metadata, writes temp file, os.replace to final
    - backlog_size(self, split) -> int: counts non-temp files present
    - ensure_dirs(), write_meta(meta), load_meta_if_exists()
    - validate_meta(meta): strict checks; raise on missing/mismatch

Backpressure
- For each split loop: while True:
  - if backlog_size >= max_backlog_files: sleep(sleep_seconds) and continue
  - else produce_one_file and increment seq

Extensibility points for advanced datasets
- Provide helper utilities in base for:
  - tokenizer interface abstraction (encode/decode), optional caching
  - common sampling helpers: next-span for LM; dynamic masking for MLM; stratified sampling for classification
  - deterministic RNG per file via base seed + file seq for reproducibility

### Example: Char Diffusion Provider (sketch)
Dataset-specific code focuses on masked token generation while relying on the base class for queue management.

- `build_meta`: sets `training_type='MLM'`, adds vocabulary/mask metadata, and defines `batch_schema` with masked inputs and reconstruction targets.
- `sample_batch(split, rng)`: chooses stage configuration, applies BERT-style masking, and returns `{"x": corrupted_tokens, "y": targets}`.

### Example: BERT MLM Provider (sketch)
- training_type=MLM, batch_schema includes input_ids:int64[block_size], attention_mask:int8[block_size], labels:int64[block_size] with -100 for unmasked
- sample_batch applies 15% masking with standard BERT rules and returns dict

### Configuration & CLI
- Consumer: constructed from training config (mirrors `config/train_char_diffusion.py`), auto-discovers meta and schema
- Provider: prepare.py scripts become thin wrappers around subclass of DataProviderBase with argparse options. Minimal flags:
  - --batch_size, --block_size, --target_size, --batches_per_file, --max_backlog_files, --sleep_seconds, --force_regenerate (optional: purge existing queue/*)
- Misconfiguration handling:
  - No defaults for optional features; missing required config -> immediate error

### Logging
- Provider: minimal periodic logs (one line per produced file, backlog changes); verbose flag for debug
- Consumer: minimal per-file logs, warnings on schema/batch_size mismatch; no noisy per-iteration logs

### Migration Plan
- Phase 1: Implement DatasetConsumer with backward-compat reading of legacy `{split}_batches_*.pt`. No provider changes needed; training switches to DatasetConsumer with same get_batch API.
- Phase 2: Introduce DataProviderBase and refactor data/shakespeare_char/prepare.py to subclass it. Keep legacy prepare.py behavior behind a flag.
- Phase 3: Update train.py to optionally accept dict batches (schema-driven). Keep tuple (X,Y) path for LM.
- Phase 4: Extend to BERT/classification providers and add tests.

### Testing Strategy
- Unit tests (no GPU required):
  - DatasetConsumer: file discovery ordering, blocking behavior with timeout (use temp dir), batch slicing across file boundaries, repeat-to-fill correctness, schema enforcement, deletion signaling
  - DataProviderBase: atomic temp-to-final rename, backlog backpressure, metadata validation, deterministic per-file RNG
- Integration tests:
  - Producer–consumer end-to-end with small synthetic dataset: provider running in background thread/process producing two files while consumer trains for N steps; assert no deadlocks; assert counts match
  - Backward compatibility: consume legacy files created by current prepare.py
  - Windows path/rename behavior (CI step on Windows runner if available)
- Performance smoke: measure get_batch latency vs. BatchManager on CPU

### Error Handling & Robustness
- Any missing meta.pkl -> fail with explicit message
- Mismatched schema: consumer raises descriptive error
- Incomplete temp files are ignored by consumer
- Corrupted final file: attempt load; on failure, move to quarantine directory and continue
- Safe deletion: consumer deletes only files it has fully consumed; if delete fails (permission), retry/backoff

### Train Loop Integration
- Minimal code change required in train.py:
  - Replace BatchManager with DatasetConsumer
  - Keep existing calls model(X, Y) for LM; later allow dict-based forward adapters when training_type != LM
- Estimate_loss reuses same consumer

### Open Questions
- Should we persist a persistent sequence counter for files to survive restarts without scanning? Initial approach: compute next seq as 1 + max(seq from filenames). Good enough.
- Multiple providers for same dataset? Out of scope.
- Cross-rank coordination in DDP (avoid deleting files others might still need). For now: each rank uses its own data_dir (or separate split subdir e.g., queue/train_rank{r}). Document this requirement.

### Summary
This design introduces a clean producer–consumer abstraction around filesystem queues, allowing flexible, parallel data preparation with strict metadata and schema, minimal logging, and safe defaults. It maintains compatibility with existing workflows while opening the door to richer datasets (BERT/classification) and more efficient training loops.

