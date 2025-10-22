Awesome—let’s lock in a **clean, parameter-efficient Encoder-Guidance** that:

* builds `enc_x` **on the fly** from `(idx, targets)` (no dataset change),
* uses a **tiny bidirectional encoder** (with its **own LoRA**, not shared with decoder),
* produces a single **global vector `g`** (via **[CLS]**),
* and **FiLM-modulates** each decoder block with **low-rank (rank-r) adapters**.

Below is a **from-scratch, copy-pasteable plan** that matches your repo’s current code paths (where the model is a bidirectional decoder with RoPE/flash attention and LoRA) and training/eval loops that call `model(X, Y)` (i.e., no third tensor passed today). Citations point to the relevant parts of your current files.

---

# WHY / WHAT (short intro for the next dev)

**WHY:** In discrete diffusion, the decoder must both infer global intent and denoise locally. A small encoder that sees a “cleaner” reference sequence gives the decoder a **global plan** (`g`) so it can focus on local fixes.

**WHAT:**

1. A **tiny encoder** (bidirectional, BERT-style) reads a reference sequence (derived at runtime from `idx` & `targets`) and produces one vector **`g`** via a learned **[CLS]** token.
2. Every **decoder block** applies **FiLM** (feature-wise scale-and-shift) to the pre-norm activations before **self-attention** and **MLP** using **low-rank adapters** of `g → (γ, β)`.
3. **Training:** same CE loss; just pass `g` internally.
4. **Inference:** compute `g` **once** from the prompt; reuse at all diffusion steps.

This preserves your bidirectional decoder & LoRA, adds minimal params, and keeps encoder LoRA **independent** from decoder LoRA.

---

# Step-by-step changes

> File names in bold; paste the code blocks in the indicated places.
> I’ll keep variables aligned to your existing code (e.g., `GPTConfig`, `Block`, `GPT.forward`) so diffs are minimal.
> Current decoder callsites: `for block in self.transformer.h: x = block(x)` in **model.py** , and training loops call `model(X, Y)` in **train.py** ; sampling calls `model(x)` in **sample.py** .

---

## 1) **model.py** — add encoder + FiLM and thread `g`

### 1.1) Extend `GPTConfig` (near its definition)

Add encoder & FiLM knobs (independent LoRA for encoder):

```python
@dataclass
class GPTConfig:
    # --- existing fields ---
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    ignore_index: int = -100
    use_lora_attn: bool = False
    use_lora_mlp: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # --- NEW: encoder and FiLM ---
    use_encoder_guidance: bool = False
    enc_n_layer: int = 1
    enc_n_head: int = 4
    enc_n_embd: int = 256
    enc_dropout: float = 0.0
    # LoRA for encoder (independent from decoder)
    enc_use_lora_attn: bool = False
    enc_use_lora_mlp: bool = False
    enc_lora_rank: int = 8
    enc_lora_alpha: float = 16.0
    enc_lora_dropout: float = 0.0

    # FiLM low-rank
    film_rank: int = 8
    guidance_scale: float = 1.0  # scalar multiplier on (gamma, beta)
```

### 1.2) Add **FiLM** adapter (low-rank) and **Encoder** (CLS + Blocks)

Paste **above** `class Block`:

```python
class FiLMAdapterLR(nn.Module):
    """Low-rank FiLM: g -> (gamma, beta) in R^{C}."""
    def __init__(self, g_dim: int, c_dim: int, rank: int = 8, bias: bool = False, scale: float = 1.0):
        super().__init__()
        r = max(1, int(rank))
        self.scale = float(scale)
        self.down = nn.Linear(g_dim, r, bias=bias)
        self.up   = nn.Linear(r, 2 * c_dim, bias=bias)

    def forward(self, g):  # g: [B, g_dim]
        gb = self.up(self.down(g)) * self.scale
        gamma, beta = gb.chunk(2, dim=-1)  # each [B, C]
        return gamma, beta


class Encoder(nn.Module):
    """
    Tiny bidirectional encoder producing one global vector g via [CLS].
    Independent LoRA switches from decoder.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        Cdec = config.n_embd
        Cenc = config.enc_n_embd

        # Independent token embedding (NOT tied to decoder)
        self.wte = nn.Embedding(config.vocab_size, Cenc)
        self.drop = nn.Dropout(config.enc_dropout)

        # Use your Block for a mini encoder stack (bidirectional)
        enc_cfg = GPTConfig(**{**config.__dict__})
        enc_cfg.n_embd = Cenc
        enc_cfg.n_head = config.enc_n_head
        enc_cfg.n_layer = 0  # we'll build list manually
        # Encoder uses its own LoRA knobs
        enc_cfg.use_lora_attn = config.enc_use_lora_attn
        enc_cfg.use_lora_mlp  = config.enc_use_lora_mlp
        enc_cfg.lora_rank     = config.enc_lora_rank
        enc_cfg.lora_alpha    = config.enc_lora_alpha
        enc_cfg.lora_dropout  = config.enc_lora_dropout

        self.h = nn.ModuleList([Block(enc_cfg) for _ in range(config.enc_n_layer)])
        self.ln_f = LayerNorm(Cenc, bias=config.bias)

        # Learned [CLS] (first position)
        self.cls = nn.Parameter(torch.randn(1, 1, Cenc) * 0.02)

        # Project to decoder width for FiLM
        self.to_dec = nn.Linear(Cenc, Cdec, bias=config.bias) if Cenc != Cdec else nn.Identity()

    def forward(self, enc_tokens: torch.Tensor) -> torch.Tensor:
        # enc_tokens: [B, T_enc], Long
        B = enc_tokens.size(0)
        x = self.drop(self.wte(enc_tokens))
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS
        for blk in self.h:
            x = blk(x)  # Block is already bidirectional in your code
        x = self.ln_f(x)
        g = x[:, 0, :]          # CLS
        g = self.to_dec(g)      # [B, n_embd] for FiLM
        return g
```

> Your `Block` is bidirectional and uses RoPE+flash attention already, so reusing it for the encoder needs no changes to attention masking. 

### 1.3) Make `Block` accept optional `g` and FiLM its pre-norms

Find `class Block` and change it like this (keep everything else intact):

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # FiLM only when decoder is configured to use encoder guidance
        self.use_film = getattr(config, 'use_encoder_guidance', False)
        if self.use_film:
            g_dim = getattr(config, 'n_embd')  # encoder projects to decoder width
            rank = getattr(config, 'film_rank', 8)
            scale = float(getattr(config, 'guidance_scale', 1.0))
            # two small adapters: pre-attn and pre-mlp
            self.film_attn = FiLMAdapterLR(g_dim, config.n_embd, rank=rank, bias=config.bias, scale=scale)
            self.film_mlp  = FiLMAdapterLR(g_dim, config.n_embd, rank=rank, bias=config.bias, scale=scale)

    def _apply_film(self, h, gamma, beta):
        # h: [B, T, C]; gamma/beta: [B, C]
        return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, x, g=None):
        if self.use_film and g is not None:
            h = self.ln_1(x)
            gam, bet = self.film_attn(g)   # [B,C] each
            h = self._apply_film(h, gam, bet)
            x = x + self.attn(h)

            h2 = self.ln_2(x)
            gam2, bet2 = self.film_mlp(g)
            h2 = self._apply_film(h2, gam2, bet2)
            x = x + self.mlp(h2)
            return x

        # fallback (no guidance)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

Your attention & MLP internals (with independent decoder LoRA) are untouched. 

### 1.4) Wire encoder into `GPT`

In `GPT.__init__`, after creating `self.transformer` and `self.lm_head`, instantiate encoder if requested:

```python
self.encoder = None
if getattr(config, 'use_encoder_guidance', False):
    self.encoder = Encoder(config)
```

### 1.5) Extend `GPT.forward` so callers still do `model(X, Y)`

We’ll **derive `enc_x` from (idx, targets)** inside forward and compute `g` there.

Find `def forward(self, idx, targets=None):` and change to:

```python
def forward(self, idx, targets=None, *, g: torch.Tensor = None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f"..."

    # --- NEW: compute g if guidance is on and g not provided ---
    g_vec = None
    if self.encoder is not None:
        if g is not None:
            g_vec = g
        else:
            # Build enc_x on the fly: prefer targets; fallback to idx where target is ignore_index
            if targets is None:
                enc_x = idx
            else:
                ignore = getattr(self.config, 'ignore_index', -100)
                # Note: targets is Long with ignore_index in masked positions
                enc_x = torch.where(targets != ignore, targets, idx)
            g_vec = self.encoder(enc_x)

    # decoder forward
    tok_emb = self.transformer.wte(idx)
    x = self.transformer.drop(tok_emb)
    for block in self.transformer.h:
        x = block(x, g=g_vec)  # pass g_vec (or None)
    x = self.transformer.ln_f(x)

    logits = self.lm_head(x)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=self.config.ignore_index
    ) if targets is not None else None
    return logits, loss
```

> Notes:
>
> * If your dataset uses **partial targets** by default, masked positions have `ignore_index`, so `enc_x = where(target != ignore, target, idx)` mixes clean labels where known with the current corrupted tokens elsewhere—good enough to teach the encoder to extract globals. (See how partial vs. full labels are produced in **prepare_streaming.py**. When `dataset_partial_targets` is False, labels are full identity, i.e., perfect `enc_x=targets` during train. )

Add a tiny helper for inference:

```python
@torch.no_grad()
def encode(self, enc_x: torch.Tensor) -> torch.Tensor:
    assert self.encoder is not None, "Encoder guidance is disabled in config."
    return self.encoder(enc_x)
```

That’s it for the model. Existing bidirectional attention, RoPE, flash, and LoRA in your decoder remain unchanged. 

---

## 2) **train.py** — no schema change; just pass `(X, Y)` as before

Your loops currently do `X, Y = consumer.get_batch(...); logits, loss = model(X, Y)` both in eval and train. Keep this; the model now **internally** builds `enc_x` from `(X, Y)` and computes `g`. 

Add the new config knobs near the existing model hyper-params and push them into `model_args`:

```python
# --- Encoder guidance knobs ---
use_encoder_guidance = True
enc_n_layer = 1
enc_n_head = 4
enc_n_embd = 256
enc_dropout = 0.0
# independent LoRA for encoder
enc_use_lora_attn = False
enc_use_lora_mlp  = False
enc_lora_rank     = 8
enc_lora_alpha    = 16.0
enc_lora_dropout  = 0.0
# FiLM
film_rank = 8
guidance_scale = 1.0
```

When you build `model_args` (already done before init), include:

```python
model_args.update(dict(
    use_encoder_guidance=use_encoder_guidance,
    enc_n_layer=enc_n_layer,
    enc_n_head=enc_n_head,
    enc_n_embd=enc_n_embd,
    enc_dropout=enc_dropout,
    enc_use_lora_attn=enc_use_lora_attn,
    enc_use_lora_mlp=enc_use_lora_mlp,
    enc_lora_rank=enc_lora_rank,
    enc_lora_alpha=enc_lora_alpha,
    enc_lora_dropout=enc_lora_dropout,
    film_rank=film_rank,
    guidance_scale=guidance_scale,
))
```

If you resume from checkpoint, add these keys to the **resume consistency list** so shape-critical things match:

```python
resume_keys += [
    'use_encoder_guidance',
    'enc_n_layer', 'enc_n_head', 'enc_n_embd', 'enc_dropout',
    'enc_use_lora_attn', 'enc_use_lora_mlp', 'enc_lora_rank', 'enc_lora_alpha', 'enc_lora_dropout',
    'film_rank', 'guidance_scale',
]
```

No other train-loop code changes are needed; keep calling `model(X, Y)` as is. (Your loops and `estimate_loss()` expect exactly that signature. )

---

## 3) **sample.py** — compute `g` **once** from the prompt and reuse

Where you prepare `prompt`, you call `model(x)` multiple times during diffusion. Change those callsites to pass `g` computed once.

Right after you build `prompt` & set `x`:

```python
g_ctx = None
if getattr(model.config, "use_encoder_guidance", False):
    enc_x = prompt.unsqueeze(0)  # [1, T_prompt]
    g_ctx = model.encode(enc_x)   # [1, n_embd]
```

Then replace both places that do `logits, _ = model(x)` to:

```python
logits, _ = model(x, g=g_ctx)
```

You have two such calls inside the diffusion loop (one before re-noise, one after edits). Just patch them both. (See the current call sites in **sample.py**. )

---

## 4) **dataset / consumer** — nothing to change

We **don’t add fields** to batches. The model derives `enc_x` internally from the usual `(x, y)` that the consumer returns today. (Consumer returns `(X, Y)` tuples for schemas that include `x` and `y`; that’s your current path. )

If later you turn off partial targets (so train labels are full identity), the encoder will automatically get a perfect `enc_x = targets` during training (see label creation in **prepare_streaming.py**). 

---

# Sizing & independence notes

* **Param overhead (typical):**
  FiLM per block ≈ `2 * r * (g_dim + 2*C)`; with `C=384`, `g_dim=384`, `r=8` → ~18.4k/block → ~110k for 6 blocks.
  Encoder (1 layer, `Cenc=256`) ≈ ~1.6M if you use a full Transformer layer; or keep **1 layer** and **LoRA** on to cut trainable params dramatically while keeping capacity.
* **LoRA independence:** encoder Blocks use **`enc_use_lora_*`** flags; decoder uses existing `use_lora_*`. No sharing of weights nor LoRA routes.
* **No weight tying:** encoder has its **own `wte`**; your decoder retains `lm_head` tied to decoder `wte` as before. (See your current tie in **model.py**. )

---

# How it runs (mental model)

* **Training:** `model(X, Y)` → internally `enc_x = where(Y != ignore, Y, X)` → `g = Encoder(enc_x)` → each decoder block FiLM-modulates its pre-norm activations with `(γ, β)=Adapter(g)` → logits/CE same as before.
* **Inference:** `g = model.encode(prompt)` once → reuse `g` for every diffusion iteration: `logits, _ = model(x, g=g)`.
