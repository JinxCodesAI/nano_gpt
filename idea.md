# Extending a GPT Model for Diffusion-Based Text Generation

## 1. Description of the Idea

The core idea is to transform a standard, autoregressive Generative Pre-trained Transformer (GPT) into a non-autoregressive, iterative text generator that operates on principles inspired by diffusion models.

Unlike a traditional GPT which generates text sequentially from left to right, this new model will work on an entire sequence of text in parallel. The generation process begins with a sequence composed almost entirely of a special `[MASK]` token, representing maximum uncertainty. Through a series of iterative refinement steps, the model repeatedly "denoises" this sequence, replacing `[MASK]` tokens with probable words and correcting other tokens until a coherent text fragment emerges.

This is achieved by modifying the model's architecture to be fully bidirectional and retraining it on a novel objective: to repair artificially corrupted text. The model learns to simultaneously fill in blanks and identify and flag incorrect tokens, effectively learning a distribution over entire sequences of text rather than just the next token.

## 2. Functional Specification

This section details the precise functional differences between the original autoregressive solution and the target diffusion-style solution.

### 2.1. Model Architecture
| Feature | Original (Autoregressive) | Target (Diffusion) |
| :--- | :--- | :--- |
| **Attention Mechanism** | **Causal Self-Attention:** Unidirectional. Tokens can only attend to previous tokens. | **Bidirectional Self-Attention:** Unmasked. Tokens can attend to all other tokens in the sequence (past and future). |
| **Vocabulary** | Standard vocabulary (e.g., GPT-2's 50257 tokens). | Standard vocabulary **plus one additional special token: `[MASK]`**. |
| **Inference Logic**| Returns logits for the **final token position** only. | Returns logits for **all token positions** in the sequence. |

### 2.2. Training Process
| Feature | Original (Autoregressive) | Target (Diffusion) |
| :--- | :--- | :--- |
| **Input Data (`X`)**| A clean text sequence, e.g., `tokens[0...T-1]`. | A **corrupted** version of the clean text sequence. |
| **Target Data (`Y`)**| The shifted input sequence, e.g., `tokens[1...T]`. | A **reconstruction target** derived from the clean sequence and the corruption method. |
| **Data Corruption** | N/A | For each sequence in a batch, a two-step corruption is applied with dynamic rates:<br> 1. `rate_mask = random(0, 0.99)`: This percentage of tokens is replaced with `[MASK]`<br> 2. `rate_random = random(0, 0.99 - rate_mask)`: This percentage of the *remaining* tokens is replaced with a random token from the vocabulary. |
| **Target Generation**| N/A | The target tensor is constructed based on the corruption:<br> 1. For positions that were **masked**, the target is the **original token**.<br> 2. For positions that were **corrupted** with a random token, the target is the **`[MASK]` token**.<br> 3. For all **untouched** positions, the target is the **original token**. |
| **Loss Function** | Standard `CrossEntropyLoss`. | A **custom weighted CrossEntropyLoss** that penalizes different types of errors differently to encourage desired behavior. The default penalty weight is `1.0`, with the following specific discounts:<br> 1. **"Honest I Don't Know"**: If `Input=[MASK]`, `Target=word`, but `Model Predicts=[MASK]`, the penalty is multiplied by a low factor `penalty_keep_mask` (e.g., 0.25).<br> 2. **"Nervous Masking"**: If `Input=word`, `Target=word`, but `Model Predicts=[MASK]`, the penalty is multiplied by a low factor `penalty_mask_correct` (e.g., 0.5).<br> 3. All other errors (e.g., predicting the wrong word) receive the full `1.0` penalty. |

### 2.3. Inference Process
| Feature | Original (Autoregressive) | Target (Diffusion) |
| :--- | :--- | :--- |
| **Initialization** | A short prompt of starting tokens. | A sequence of a fixed length composed entirely of `[MASK]` tokens. |
| **Generation Loop** | For N new tokens:<br> 1. Forward pass.<br> 2. Sample from logits of the last token.<br> 3. Append sampled token to the sequence. | For `max_steps` iterations:<br> 1. Forward pass on the entire sequence.<br> 2. Get logits for all positions.<br> 3. Sample a new sequence.<br> 4. **Update only the tokens that were `[MASK]`** in the previous step's sequence. |
| **Termination** | Loop finishes after N tokens are generated. | Loop finishes after `max_steps` or when no `[MASK]` tokens remain. A final pass resamples any remaining `[MASK]` tokens, forbidding `[MASK]` as an output. |

## 3. Implementation Plan

This is a step-by-step guide to implement the diffusion model by extending the existing codebase. The changes are designed to be controlled by a `model_type` configuration flag.

### Step 1: Modify `model.py` — Architectural Changes

#### 1.1. Extend `GPTConfig`
Add `model_type` and `mask_token_id` to the dataclass to control behavior.

```python
# In model.py
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # --- NEW ADDITIONS ---
    model_type: str = 'gpt2' # Can be 'gpt2' or 'diffusion'
    mask_token_id: int = None # To be set during model initialization
```

#### 1.2. Create `BidirectionalSelfAttention`
This new class is a copy of `CausalSelfAttention` but with the causal masking logic removed.

```python
# In model.py
class BidirectionalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Key change: is_causal=False for bidirectional attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # The causal mask fill line is removed
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

#### 1.3. Make `Block` Class Polymorphic
Modify the `Block`'s `__init__` to select the correct attention mechanism.

```python
# In model.py
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # --- MODIFICATION ---
        if config.model_type == 'diffusion':
            self.attn = BidirectionalSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        # --- END MODIFICATION ---
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### 1.4. Update `GPT.forward()`
Modify the inference path to return full logits for the diffusion model.

```python
# In model.py, inside the GPT class
    def forward(self, idx, targets=None):
        # ... (code for embeddings and transformer blocks) ...
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # --- MODIFICATION ---
            if self.config.model_type == 'diffusion':
                logits = self.lm_head(x) # Return logits for all positions
            else:
                logits = self.lm_head(x[:, [-1], :]) # Original autoregressive behavior
            loss = None
            # --- END MODIFICATION ---
        return logits, loss
```

#### 1.5. Add `generate_diffusion()` Method
Add the new iterative generation method to the `GPT` class.

```python
# In model.py, inside the GPT class
    @torch.no_grad()
    def generate_diffusion(self, idx, max_steps, temperature=1.0, top_k=None):
        assert self.config.model_type == 'diffusion', "This generation method is only for diffusion models"
        assert self.config.mask_token_id is not None, "mask_token_id must be configured."
        self.eval()

        for _ in range(max_steps):
            # Check if we are done
            mask_token_id = self.config.mask_token_id
            if not (idx == mask_token_id).any():
                break

            logits, _ = self(idx)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, :, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(idx.shape)

            # Only update the tokens that were previously [MASK]
            mask = (idx == mask_token_id)
            idx = torch.where(mask, idx_next, idx)
        
        # Finalization: force any remaining [MASK] tokens to be something else
        if (idx == self.config.mask_token_id).any():
            logits, _ = self(idx)
            logits[:, :, self.config.mask_token_id] = -float('Inf') # Forbid predicting MASK
            probs = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(idx.shape)
            mask = (idx == self.config.mask_token_id)
            idx = torch.where(mask, idx_next, idx)

        self.train()
        return idx
```

### Step 2: Modify `train.py` — Training Logic

#### 2.1. Add New Configuration Parameters
Add these to the global configuration section.

```python
# In train.py
model_type = 'gpt2' # 'gpt2' or 'diffusion'
# Diffusion loss specific penalties
penalty_keep_mask = 0.25      # Discount for failing to unmask, but keeping [MASK].
penalty_mask_correct = 0.5    # Discount for wrongly masking a correct token.
```

#### 2.2. Implement the Nuanced Loss Function
Add this function to the top of `train.py` or a separate `utils.py`.

```python
# In train.py
def diffusion_loss_function(logits, targets, inputs, mask_token_id, penalty_keep_mask, penalty_mask_correct):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    flat_inputs = inputs.view(-1)
    flat_predictions = torch.argmax(flat_logits, dim=1)

    per_token_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='none')
    weights = torch.ones_like(per_token_loss)

    # Case 1: Input=[MASK], Target=word, Predicted=[MASK]
    case1_positions = (flat_inputs == mask_token_id) & \
                      (flat_targets != mask_token_id) & \
                      (flat_predictions == mask_token_id)
    weights[case1_positions] = penalty_keep_mask

    # Case 3a: Input=word, Target=word, Predicted=[MASK]
    case3a_positions = (flat_inputs == flat_targets) & \
                       (flat_inputs != mask_token_id) & \
                       (flat_predictions == mask_token_id)
    weights[case3a_positions] = penalty_mask_correct
    
    return (per_token_loss * weights).mean()
```

#### 2.3. Rewrite `get_batch()` for Diffusion Training
This function will now create corrupted data on the fly.

```python
# In train.py
mask_token_id = None # Global variable to be set after model init

def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_clean = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    if model_type == 'diffusion':
        assert mask_token_id is not None, "mask_token_id must be set globally"
        x_corrupted = x_clean.clone()
        y_target = x_clean.clone() # Start with the ground truth

        b, t = x_corrupted.shape
        for i in range(b):
            rate_mask = torch.rand(1) * 0.99
            rate_random = torch.rand(1) * (0.99 - rate_mask)
            num_to_mask = int(t * rate_mask)
            num_to_random = int(t * rate_random)
            
            rand_pos = torch.randperm(t)
            pos_mask = rand_pos[:num_to_mask]
            pos_random = rand_pos[num_to_mask : num_to_mask + num_to_random]

            # Apply mask corruption and set target
            x_corrupted[i, pos_mask] = mask_token_id
            # y_target already has the correct original token here

            # Apply random corruption and set target
            random_tokens = torch.randint(1, meta_vocab_size, (num_to_random,))
            x_corrupted[i, pos_random] = (x_clean[i, pos_random] + random_tokens) % meta_vocab_size
            y_target[i, pos_random] = mask_token_id # Target for these is [MASK]

        x, y = x_corrupted, y_target
    else:
        y_autoregressive = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x_clean, y_autoregressive

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

#### 2.4. Update Model Initialization Logic
Handle the new vocabulary size and `mask_token_id` when initializing a model.

```python
# In train.py
# ...
model_args = dict(..., model_type=model_type) # Add model_type to args dict

if init_from == 'scratch':
    # ...
    # --- MODIFICATION ---
    if model_type == 'diffusion':
        vocab_size = meta_vocab_size + 1 if meta_vocab_size is not None else 50305
        model_args['vocab_size'] = vocab_size
        model_args['mask_token_id'] = vocab_size - 1
        global mask_token_id
        mask_token_id = model_args['mask_token_id']
    else:
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    # --- END MODIFICATION ---
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
# ...
# Also ensure the model_type and mask_token_id are handled correctly when resuming from a checkpoint.
```

#### 2.5. Update the Training Loop to Use Custom Loss
Replace the standard loss calculation with a call to our new function.

```python
# In train.py, inside the `while True:` training loop
# ...
    for micro_step in range(gradient_accumulation_steps):
        # ...
        with ctx:
            logits, loss_from_model = model(X, Y)
            # --- MODIFICATION ---
            if model_type == 'diffusion':
                loss = diffusion_loss_function(logits, Y, X, mask_token_id, penalty_keep_mask, penalty_mask_correct)
            else:
                loss = loss_from_model
            loss = loss / gradient_accumulation_steps
            # --- END MODIFICATION ---
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
# ...
```

### Step 3: Create `generate_diffusion.py` — New Inference Script

This standalone script is for generating text using a trained diffusion model checkpoint.

```python
# Create new file: generate_diffusion.py
import os
import torch
from model import GPTConfig, GPT
# import tiktoken # If you have a tokenizer

# --- Parameters ---
out_dir = 'out-diffusion'
device = 'cuda'
max_new_tokens = 100
num_samples = 1
temperature = 0.8
top_k = 200
seed = 1337
max_steps = 25 # Number of refinement steps

# --- Load Model ---
torch.manual_seed(seed)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
# (Add logic to remove unwanted prefix `_orig_mod.` from state_dict keys if necessary)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# --- Prepare Initial Sequence ---
mask_token_id = gptconf.mask_token_id
assert mask_token_id is not None, "Mask token not found in model config"
start_ids = torch.full((num_samples, max_new_tokens), mask_token_id, dtype=torch.long, device=device)

# --- Generate ---
print("Generating with diffusion model...")
with torch.no_grad():
    y = model.generate_diffusion(start_ids, max_steps, temperature=temperature, top_k=top_k)
    # --- Decode and Print ---
    # enc = tiktoken.get_encoding("gpt2")
    # print(enc.decode(y[0].tolist()))
    print("Generated token IDs:")
    print(y[0].tolist())
```