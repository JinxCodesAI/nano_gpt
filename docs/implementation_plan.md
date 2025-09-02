###  Development Plan: Refactoring to a Multi-Mode Architecture

The goal is to replace the boolean flags in your implementation with a more robust `Enum` to manage different model behaviors, including the new sequence-level classifier.

-----

###  Step 1: Define the Model Operating Modes

  * **WHY:** An `Enum` provides a type-safe and explicit way to define the model's operating mode, preventing invalid states and making the code more readable than multiple boolean flags.
  * **WHAT:** Define a `ModelMode` enum with three possible states: `LANGUAGE_MODEL`, `TOKEN_CLASSIFIER`, and `SEQUENCE_CLASSIFIER`.
  * **WHERE:** At the top of your file, after the imports.

<!-- end list -->

```python
#
# Put this after your imports
#
from enum import Enum, auto

class ModelMode(Enum):
    LANGUAGE_MODEL = auto()
    TOKEN_CLASSIFIER = auto()
    SEQUENCE_CLASSIFIER = auto()
```

-----

###  Step 2: Update the Model Configuration

  * **WHY:** The model's configuration should use the new `Enum` to determine its architecture and behavior, replacing the less clear `binary_classification` flag.
  * **WHAT:** In `GPTConfig`, remove the `binary_classification` flag and add a new `mode` field that uses the `ModelMode` enum. We'll also ensure that sequence classification defaults to and requires bidirectional attention.
  * **WHERE:** In the `GPTConfig` dataclass.

<!-- end list -->

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    attention_type: str = 'causal'
    use_rope: bool = True
    # --- REMOVE OLD FLAG and ADD THE NEW MODE ---
    mode: ModelMode = ModelMode.LANGUAGE_MODEL # Default to language modeling

    def __post_init__(self):
        # Automatically set attention to bidirectional for sequence classification
        if self.mode == ModelMode.SEQUENCE_CLASSIFIER:
            if self.attention_type != 'bidirectional':
                print(f"Warning: mode is {self.mode}, forcing attention_type to 'bidirectional'.")
                self.attention_type = 'bidirectional'
```

-----

###  Step 3: Adapt the Model Architecture

  * **WHY:** The model needs to build the correct output "head" based on the selected `mode`. This requires separate heads for language modeling, token classification, and sequence classification.
  * **WHAT:** Modify the `GPT.__init__` method to create the appropriate head (`lm_head` for token-level tasks or `sequence_head` for sequence-level tasks) based on `config.mode`.
  * **WHERE:** In the `GPT.__init__` method.

<!-- end list -->

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # (keep the self.transformer definition as is)
        self.transformer = nn.ModuleDict(...)

        # --- REPLACE THE HEAD INITIALIZATION LOGIC WITH THIS ---
        self.lm_head = None
        self.sequence_head = None

        if self.config.mode == ModelMode.LANGUAGE_MODEL:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight # Weight tying
        elif self.config.mode == ModelMode.TOKEN_CLASSIFIER:
            # Head for per-token binary classification
            self.lm_head = nn.Linear(config.n_embd, 2, bias=False)
        elif self.config.mode == ModelMode.SEQUENCE_CLASSIFIER:
            # Head for sequence-level regression (outputs a single value)
            self.sequence_head = nn.Linear(config.n_embd, 1, bias=False)
        
        # (keep the weight initialization logic as is)
        self.apply(self._init_weights)
        # ... rest of __init__
```

-----

###  Step 4: Implement the Multi-Mode Forward Pass

  * **WHY:** The core logic of the model must change depending on the mode. We need to route the transformer's output to the correct head and compute the appropriate loss function for each task.
  * **WHAT:** Restructure the `GPT.forward` method to handle all three modes. For `SEQUENCE_CLASSIFIER`, it will select the first token's output (`[CLS]` token), pass it to the `sequence_head`, and use `MSELoss` for the regression task.
  * **WHERE:** In the `GPT.forward` method.

<!-- end list -->

```python
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # (keep the forward pass through the transformer blocks as is)
        # ...
        x = self.transformer.ln_f(x) # Final hidden states, shape (b, t, n_embd)

        logits, loss = None, None

        if self.config.mode == ModelMode.SEQUENCE_CLASSIFIER:
            # Sequence classification/regression: use the [CLS] token's output
            # We assume the [CLS] token is at the first position
            cls_output = x[:, 0, :] # Shape: (b, n_embd)
            logits = self.sequence_head(cls_output).squeeze(-1) # Shape: (b,)

            if targets is not None:
                # For regression, targets must be a FloatTensor
                loss = F.mse_loss(logits, targets.float())

        elif self.config.mode in [ModelMode.LANGUAGE_MODEL, ModelMode.TOKEN_CLASSIFIER]:
            # Token-level predictions
            if self.lm_head is not None:
                logits = self.lm_head(x) # Shape: (b, t, vocab_size) or (b, t, 2)
            
            if targets is not None and logits is not None:
                if self.config.mode == ModelMode.TOKEN_CLASSIFIER:
                    # Your existing per-token binary classification loss
                    loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1), ignore_index=-1)
                else: # Language Modeling
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        # Handle inference case where targets is None
        if targets is None:
            # For LM/Token inference, logits are already computed above
            # For Sequence inference, logits are also computed above
            pass

        return logits, loss
```

-----

###  Step 5: Update Your Training Script Logic

  * **WHY:** To use the new `SEQUENCE_CLASSIFIER` mode, your training script must be updated to initialize the model correctly and provide the data in the expected format.
  * **WHAT:**
    1.  Initialize `GPTConfig` with `mode=ModelMode.SEQUENCE_CLASSIFIER`.
    2.  In your data pipeline, **prepend a `[CLS]` token ID** to every input sequence.
    3.  Ensure your `targets` tensor is a `torch.FloatTensor` containing the regression values (e.g., the synthetic percentage).
  * **WHERE:** In your external training and data loading scripts.

<!-- end list -->

```python
# --- Example of usage in your training script ---

# 1. Import the new Enum
# from model import ModelMode, GPTConfig, GPT

# 2. Configure the model for sequence classification
config = GPTConfig(
    # ... other parameters ...
    vocab_size=80, # Your vocab size
    mode=ModelMode.SEQUENCE_CLASSIFIER
    # The config will automatically set attention_type to 'bidirectional'
)
model = GPT(config)

# 3. In your data preparation / training loop
CLS_TOKEN_ID = 65 # Your chosen ID for the [CLS] token

# Assume original_idx is shape (batch_size, seq_len)
# Assume regression_targets is shape (batch_size,) -> e.g., torch.tensor([0.1, 0.5, 0.0])

# Prepend the CLS token
cls_token_tensor = torch.full((batch_size, 1), CLS_TOKEN_ID, device=device)
batch_idx = torch.cat([cls_token_tensor, original_idx], dim=1)

# Forward pass with float targets
predictions, loss = model(batch_idx, regression_targets)

# ... proceed with backward pass and optimization ...
```