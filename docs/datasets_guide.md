## Creating and Using Datasets (Streaming Pipeline)

This guide explains how to add a new dataset to the streaming data pipeline and how to use it with `train.py` and `prepare.py` using a shared config.

### Concepts
- `DataProviderBase`: base class that writes batch files into a filesystem queue at `data/<dataset>/queue/{train,val}`. You subclass it to implement sampling logic and metadata.
- `DatasetConsumer`: the training-side loader that blocks until data is available and consumes batches, deleting fully-consumed files to signal demand.
- Shared config: a single config `.py` file used by both `prepare.py` and `train.py`. The global validator checks that the config is minimally sufficient.

### Directory layout
- Put your dataset under `data/<your_dataset>/`
  - `data/<your_dataset>/prepare_streaming.py` (provider subclass)
  - `data/<your_dataset>/input.*` (raw files), optional
  - `data/<your_dataset>/config/*.py` (dataset-specific settings), optional
  - `data/<your_dataset>/queue/train` and `data/<your_dataset>/queue/val` are created by the provider

### Step 1: Implement a provider
Create `data/<your_dataset>/prepare_streaming.py` by subclassing `DataProviderBase`.

Example skeleton inspired by the shipped `CharDiffusionProvider`:

````python
class CharDiffusionProvider(DataProviderBase):
    def build_meta(self) -> Dict:
        return {
            "training_type": "MLM",
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [self.block_size], "role": "target"},
            ],
        }

    def sample_batch(self, split, rng):
        # Generate tensors and return a dict matching batch_schema
        return {"x": masked_inputs, "y": reconstruction_targets}
````

### Step 2: Register your provider for `prepare.py`
`prepare.py` uses a simple convention to map dataset name to the provider class.

Ensure the folder layout is `data/<dataset>/prepare_streaming.py` and export your subclass. `prepare.py` will automatically find the only `DataProviderBase` subclass or the symbol named `Provider`.

The dataset name must match your config's `dataset` variable.

### Step 3: Create or adapt a training config
Create `config/your_config.py` (e.g., `config/train_char_diffusion.py`). At minimum, set:
- `dataset = '<your_dataset>'`
- `batch_size` (int)
- `block_size` (int)
- Diffusion-specific knobs such as `mask_probability`, `ignore_index`, or stage definitions can be passed through the config and accessed via `globals()` in the provider.

The same config is consumed by both `prepare.py` and `train.py`. The global validator (`config/validator.py`) checks basic sufficiency.

### Step 4: Run producer and trainer (from repo root)
- Producer (streaming writer):
  - `python prepare.py config/train_char_diffusion.py`
- Trainer (consumer):
  - `python train.py config/train_char_diffusion.py`

`DatasetConsumer` will fail loudly if `data/<dataset>/queue` is missing; run `prepare.py` first. Do not run provider modules directly with a config file; use `prepare.py` to load the shared config.

### Step 5: Accessing meta and schema in training (optional)
`train.py` automatically reads metadata from `consumer.meta`. You can also inspect the schema:

````python
def schema(self, split: str) -> Optional[List[Dict]]:
    return self.meta.get("batch_schema")
````

If your `batch_schema` is not the default `x`/`y` pair (e.g., additional masks or class labels), adapt the training step to accept a dict instead of `(X, Y)` when you extend the model/trainer; `DatasetConsumer` already returns a dict in those cases.

### Design rules and tips
- Fail loudly on misconfiguration: do not provide silent fallbacks for missing directories or metadata.
- Keep provider code focused on sample logic; leave file naming, atomic writes, backlog control to the base class.
- Ensure `sample_batch` returns tensors with correct dtypes and shapes that match `batch_schema`.
- For DDP, prefer separate datasets/queues per rank or document your strategy; shared queues across ranks are not yet coordinated.

### Troubleshooting
- Consumer error: "Queue directory not found" -> run `prepare.py` with the same config.
- Device moves: `DatasetConsumer` uses pinned memory and non-blocking transfers on CUDA.
- Meta changes: If you change schema or vocab, purge queue files to avoid loading stale batches.
