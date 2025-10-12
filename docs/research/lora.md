# Parameter Sharing + Per-Layer LoRA Adapters

### Developer implementation guide (why & how)

---

## What we want to do (and why)

We will (1) **share the large projection matrices across all transformer blocks** (ALBERT-style), and (2) add **tiny, per-layer LoRA adapters** on top of those shared matrices for both the **BidirectionalSelfAttention** and the **MLP**. This keeps the heavy compute/params in a single set of weights while giving each layer its own small “adapter” to specialize. In your codebase, the big matrices live in `BidirectionalSelfAttention.c_attn`, `BidirectionalSelfAttention.c_proj`, `MLP.c_fc`, and `MLP.c_proj` in `model.py`. 

Effect: dramatic parameter reduction with minimal code surgery; layer individuality is preserved via the LoRA adapters (and via the already per-layer LayerNorms), and the rest of your model shape (forward APIs, RoPE, SDPA, critic head, etc.) stays intact.

---

## Step-by-step: code changes

Below are **drop-in edits to `model.py`**. Paths and class names match your current file.

### 0) Add new config knobs

Extend `GPTConfig` with toggles for LoRA and sharing. (Keep your existing fields.) Put these near the other config attrs. 

```python
# --- in GPTConfig ---
# LoRA
use_lora_attn: bool = False
use_lora_mlp: bool = False
lora_rank: int = 8
lora_alpha: float = 16.0
lora_dropout: float = 0.0

# Cross-layer sharing
share_main_matrices: bool = False  # share attn.c_attn/.c_proj and mlp.c_fc/.c_proj across blocks
```

---

### 1) Add a minimal LoRA helper

Place this small module near your other helpers (e.g., above `LayerNorm`). It returns **0** when disabled, so call sites can just **add** it.

```python
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16.0, dropout=0.0, enabled=False):
        super().__init__()
        self.enabled = bool(enabled and r > 0)
        if not self.enabled:
            self.register_buffer("_dummy", torch.tensor(0.), persistent=False)
            return
        self.r = r
        self.scaling = float(alpha) / float(r)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        # init: A ~ kaiming, B ~ 0
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        if not self.enabled:
            return 0.0
        return self.B(self.A(self.drop(x))) * self.scaling
```

---

### 2) Wire LoRA into **BidirectionalSelfAttention**

**Where:** `class BidirectionalSelfAttention` in `model.py`. This class already defines `c_attn` (QKV) and `c_proj` (output) and applies RoPE + SDPA. We’ll add two LoRA members and sum their deltas into the linear outputs. None of the RoPE/SDPA logic changes.

**Add in `__init__`:**

```python
self.lora_qkv  = LoRALinear(
    config.n_embd, 3 * config.n_embd,
    r=getattr(config, 'lora_rank', 0),
    alpha=getattr(config, 'lora_alpha', 1.0),
    dropout=getattr(config, 'lora_dropout', 0.0),
    enabled=getattr(config, 'use_lora_attn', False),
)
self.lora_out  = LoRALinear(
    config.n_embd, config.n_embd,
    r=getattr(config, 'lora_rank', 0),
    alpha=getattr(config, 'lora_alpha', 1.0),
    dropout=getattr(config, 'lora_dropout', 0.0),
    enabled=getattr(config, 'use_lora_attn', False),
)
```

**Edit the start of `forward`** where QKV are computed; replace:

```python
q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
```

with:

```python
qkv = self.c_attn(x) + (self.lora_qkv(x) if self.lora_qkv.enabled else 0.0)
q, k, v = qkv.split(self.n_embd, dim=2)
```

**Edit the output projection** (just before returning); replace:

```python
y = self.resid_dropout(self.c_proj(y))
```

with:

```python
y = self.c_proj(y) + (self.lora_out(y) if self.lora_out.enabled else 0.0)
y = self.resid_dropout(y)
```

That’s it for attention—RoPE and SDPA masks remain as in your code. 

---

### 3) Wire LoRA into **MLP**

**Where:** `class MLP` in `model.py`. The class currently does `c_fc → GELU → c_proj → Dropout`. We add LoRA on `c_fc` and `c_proj` and sum deltas. 

**Add in `__init__`:**

```python
self.lora_fc   = LoRALinear(
    config.n_embd, 4 * config.n_embd,
    r=getattr(config, 'lora_rank', 0),
    alpha=getattr(config, 'lora_alpha', 1.0),
    dropout=getattr(config, 'lora_dropout', 0.0),
    enabled=getattr(config, 'use_lora_mlp', False),
)
self.lora_proj = LoRALinear(
    4 * config.n_embd, config.n_embd,
    r=getattr(config, 'lora_rank', 0),
    alpha=getattr(config, 'lora_alpha', 1.0),
    dropout=getattr(config, 'lora_dropout', 0.0),
    enabled=getattr(config, 'use_lora_mlp', False),
)
```

**Edit `forward`:**

```python
x = self.c_fc(x) + (self.lora_fc(x) if self.lora_fc.enabled else 0.0)
x = self.gelu(x)
x = self.c_proj(x) + (self.lora_proj(x) if self.lora_proj.enabled else 0.0)
x = self.dropout(x)
return x
```

---

### 4) Share the big matrices across layers

**Where:** `class GPT.__init__`, **after** you build the ModuleList of blocks (`self.transformer.h`). Alias the Parameters from block 0 into all other blocks. This is standard PyTorch parameter sharing (all grads accumulate into the same tensor). Do **not** share LayerNorms. 

```python
# after: self.transformer = nn.ModuleDict({... 'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ...})
if getattr(config, 'share_main_matrices', False) and len(self.transformer.h) > 1:
    base = self.transformer.h[0]
    for blk in self.transformer.h[1:]:
        # Attention projections
        blk.attn.c_attn.weight = base.attn.c_attn.weight
        blk.attn.c_attn.bias   = base.attn.c_attn.bias
        blk.attn.c_proj.weight = base.attn.c_proj.weight
        blk.attn.c_proj.bias   = base.attn.c_proj.bias
        # MLP projections
        blk.mlp.c_fc.weight    = base.mlp.c_fc.weight
        blk.mlp.c_fc.bias      = base.mlp.c_fc.bias
        blk.mlp.c_proj.weight  = base.mlp.c_proj.weight
        blk.mlp.c_proj.bias    = base.mlp.c_proj.bias
```

Your optimizer grouping continues to work normally; shared `nn.Parameter`s appear once and receive grads from all layers. (See your `configure_optimizers` for context.) 

---


## Training & usage notes

* **Config to start with:**

  ```python
  gptconf = GPTConfig(
      # ... your dims ...
      use_lora_attn=True,
      use_lora_mlp=True,
      lora_rank=8,          # try 4–16
      lora_alpha=16.0,
      lora_dropout=0.05,    # 0–0.1
      share_main_matrices=True,
  )
  ```
* **Freeze strategy (optional):** You can train with **shared base + LoRA** from scratch. If memory is extremely tight, consider freezing the shared base after warm-up and continue with **LoRA + LayerNorms** only; your optimizer setup makes this straightforward. 
* **Inference:** Keep LoRA active; compute overhead is small and preserves per-layer specialization. No API changes to `forward`, `generate`, or critic pathways.

---

## Parameter/compute intuition

With `n_embd=384` and `lora_rank=r=8`, rough per-layer LoRA overhead is ~**40–45k params** (QKV, attn-out, MLP fc, MLP out). That’s tiny compared to a full per-layer set of projections. Because the projections are shared once across all layers, the total parameter count scales mostly with the number of layers × LoRA size + (one copy of the big matrices), not layers × full matrices.

---

## Quick checklist (for PR review)

* [ ] New config fields exist and have sensible defaults. 
* [ ] `LoRALinear` added and imported in the file.
* [ ] `BidirectionalSelfAttention` uses `lora_qkv` and `lora_out` as additive deltas. RoPE/SDPA untouched. 
* [ ] `MLP` uses `lora_fc` and `lora_proj` as additive deltas. 
* [ ] GPT constructor aliases `c_attn`, `c_proj`, `c_fc`, `c_proj` weights/biases across blocks when `share_main_matrices=True`. 
* [ ] `_build_default_attention_mask` returns 1=keep / 0=mask to match SDPA padding logic.
* [ ] Optimizer still builds groups; no param duplication. 

---

If you like, I can turn this into a ready-to-apply unified diff against your current `model.py`.
