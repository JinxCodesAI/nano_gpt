## Creating and Using Datasets (Streaming Pipeline)

This guide explains how to add a new dataset to the streaming data pipeline and how to use it with train.py and prepare.py using a shared config.

### Concepts
- DataProviderBase: base class that writes batch files into a filesystem queue at data/<dataset>/queue/{train,val}. You subclass it to implement only sampling logic and meta.
- DatasetConsumer: the training-side loader that blocks until data is available and consumes batches, deleting fully-consumed files to signal demand.
- Shared config: a single config .py file used by both prepare.py and train.py. The global validator checks that the config is minimally sufficient.

### Directory layout
- Put your dataset under data/<your_dataset>/
  - data/<your_dataset>/prepare_streaming.py (provider subclass)
  - data/<your_dataset>/input.* (raw files), optional
  - data/<your_dataset>/queue/train and data/<your_dataset>/queue/val are created by provider

### Step 1: Implement a Provider
Create data/<your_dataset>/prepare_streaming.py by subclassing DataProviderBase.

Example skeleton:

````python
class DataProviderBase:
    def build_meta(self) -> Dict: ...
    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, Tensor]: ...
````

Your provider implements:
- build_meta(): returns a dict with at least:
  - training_type: e.g., "LM", "MLM", "CLASSIFICATION", ...
  - batch_schema: list of fields, each: {name, dtype, shape[, role]}
  - dataset-specific keys (e.g., vocab_size, stoi/itos)
- sample_batch(split, rng): returns one batch as a dict[name -> tensor] with shapes [batch_size, ...]. The base class handles concatenation and atomic saving.

Minimal example (character-level LM):

````python
class ShakespeareCharProvider(DataProviderBase):
    def build_meta(self) -> Dict:
        return {
            "training_type": "LM",
            "vocab_size": self.vocab_size,
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size]},
                {"name": "y", "dtype": "int64", "shape": [self.target_size or self.block_size]},
            ],
        }
    def sample_batch(self, split, rng):
        # create dict of tensors shaped [batch_size, ...]
        return {"x": x, "y": y}
````

### Step 2: Provider discovery and config passing
prepare.py discovers your provider by convention and injects the full training config.

- Convention: create data/<dataset>/prepare_streaming.py and define a subclass of DataProviderBase. Optionally export `Provider = YourProvider`.
- Discovery: prepare.py imports `data.<dataset>.prepare_streaming` and uses `Provider` symbol if present, otherwise the single DataProviderBase subclass found.
- Config injection: prepare.py calls `Provider(data_dir=..., batch_size=..., block_size=..., ..., config=cfg)` where `cfg` is the entire config dict loaded from your config .py.

Your provider should parse dataset-specific settings from `config` in its __init__, e.g.:

````python
class YourProvider(DataProviderBase):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.pop('config', {}) or {}
        self.some_setting = cfg.get('some_setting', default)
        super().__init__(*args, **kwargs)
````

Do not modify prepare.py to add dataset-specific kwargs. Keep prepare.py dataset-agnostic.

Ensure the dataset name in your config matches your folder name under data/.

### Step 3: Create or adapt a training config
Create config/your_config.py (e.g., config/train_shakespeare_char.py). At minimum, set:
- dataset = 'your_dataset'
- batch_size (int)
- block_size (int)
- Optionally: target_size, batches_per_file, max_backlog_files, sleep_seconds, seed

Example excerpt (from config/train_shakespeare_char.py):

````python
dataset = 'shakespeare_char'
batch_size = 8
block_size = 1024
````

The same config is consumed by both prepare.py and train.py. The global validator (config/validator.py) checks basic sufficiency.

### Step 4: Run producer and trainer (from repo root)
- Producer (streaming writer):
  - python prepare.py config/train_shakespeare_char.py
- Trainer (consumer):
  - python train.py config/train_shakespeare_char.py

DatasetConsumer will fail loudly if data/<dataset>/queue is missing; run prepare.py first. Do not run provider modules directly with a config file; use prepare.py to load the shared config.

### Step 5: Accessing meta and schema in training (optional)
train.py automatically reads vocab_size from consumer.meta. You can also inspect schema:

````python
def schema(self, split: str) -> Optional[List[Dict]]:
    return self.meta.get("batch_schema")
````

If your batch_schema is not x/y (e.g., MLM), adapt the training step to accept a dict instead of (X, Y) when you extend the model/trainer; DatasetConsumer already returns a dict in those cases.

### Design rules and tips
- Fail loudly on misconfiguration: do not provide silent fallbacks for missing directories or meta.
- Keep provider code focused on sample logic; leave file naming, atomic writes, backlog control to the base class.
- Ensure sample_batch returns tensors with correct dtypes and shapes that match batch_schema.
- For DDP, prefer separate datasets/queues per rank or document your strategy; shared queues across ranks are not yet coordinated.

### Troubleshooting
- Consumer error: "Queue directory not found" -> run prepare.py with the same config.
- Device moves: DatasetConsumer uses pinned memory and non_blocking transfers on CUDA.
- Meta changes: If you change meta shape/schema, purge queue files to avoid loading stale batches.

