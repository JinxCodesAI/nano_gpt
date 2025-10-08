# Hierarchical Guidance for Long-Range Coherence (Developer Guide)

## What we’re doing (goal)

We’ll add a **small, bidirectional “plan encoder”** that produces a handful of **global plan tokens** and let the decoder **cross-attend** to them while denoising/generating. This gives the model stable, document-level context (topic, outline, entities, tense), which improves **long-range coherence** without supervision.

**Why this helps:** Your current stack learns local grammar and style well, but long arcs drift because every step reasons mostly from local context. A tiny global latent (the plan tokens) acts like an outline the decoder can consult at every layer. This pattern is common in long-form systems (hierarchical LMs, RAG, diffusion guidance) and is easy to bolt onto your codebase.

---

## Does this affect training?

**Yes (a bit), but still unsupervised.**

* We introduce an **encoder forward pass** per diffusion step (cheap: only K global tokens).
* The decoder’s loss remains your **unsupervised diffusion/LM objective**.
* Optionally use **classifier-free guidance (CFG)** at inference; to make CFG effective, we *train* with **condition dropout** (`p_drop`) so the model learns both conditional and unconditional paths.
* Everything stays self-supervised: no labels, no extra datasets.

If you skip CFG at train time, the encoder+cross-attention still helps; CFG just gives you a stronger inference control knob.

---

## Implementation plan (in the context of your code)

Below are focused patches and new modules that match your style and naming. You can paste them with minimal surgery.

### 1) Add a minimal Cross-Attention module

Create a new file (e.g., `cross_attention.py`) or place near `Block`:

```python
class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, kv, kv_mask=None):
        B, T, C = x.shape
        H = self.n_head; D = C // H
        q = self.q(x).view(B, T, H, D).transpose(1, 2)         # (B, H, T, D)
        k = self.k(kv).view(B, kv.size(1), H, D).transpose(1, 2)
        v = self.v(kv).view(B, kv.size(1), H, D).transpose(1, 2)

        if self.flash:
            attn_mask = None
            if kv_mask is not None:
                attn_mask = (kv_mask == 0)[:, None, None, :]    # (B,1,1,K)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            if kv_mask is not None:
                att = att.masked_fill((kv_mask == 0)[:, None, None, :], float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v                                              # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(self.dropout(y))
```

### 2) Extend your `Block` to **optionally** use cross-attention

Modify `Block.__init__` and `Block.forward` (safe no-op when guidance is absent):

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        attention_type = getattr(config, 'attention_type', 'causal')
        self.attn = BidirectionalSelfAttention(config) if attention_type == 'bidirectional' else CausalSelfAttention(config)

        self.cross = CrossAttention(config)            # NEW
        self.ln_cross = LayerNorm(config.n_embd, bias=config.bias)  # NEW

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, guidance_h=None, guidance_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        if guidance_h is not None:
            x = x + self.cross(self.ln_cross(x), guidance_h, kv_mask=guidance_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
```

> Backward compatibility: existing code that doesn’t pass `guidance_h` behaves exactly the same as before.

### 3) Add the **PlanEncoder** (bidirectional, shallow)

Place this near your `GPT` class:

```python
import dataclasses

class PlanEncoder(nn.Module):
    def __init__(self, config, K=16, token_embedding=None, position_embedding=None):
        super().__init__()
        self.K = K
        # K learnable plan slots
        self.plan_emb = nn.Parameter(torch.randn(K, config.n_embd) * 0.02)

        # Optionally tie embeddings with decoder to align token space
        self.wte = token_embedding if token_embedding is not None else nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = position_embedding if position_embedding is not None else (nn.Embedding(config.block_size, config.n_embd) if config.position_encoding == 'absolute' else None)

        # A shallow bidirectional stack (half depth or fixed small number)
        enc_cfg = dataclasses.replace(config, attention_type='bidirectional')
        depth = max(1, enc_cfg.n_layer // 2)
        self.ln_in  = LayerNorm(enc_cfg.n_embd, bias=enc_cfg.bias)
        self.layers = nn.ModuleList([Block(enc_cfg) for _ in range(depth)])
        self.ln_out = LayerNorm(enc_cfg.n_embd, bias=enc_cfg.bias)

    def embed(self, idx):
        tok = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(idx.size(1), device=idx.device, dtype=torch.long)
            tok = tok + self.wpe(pos)  # broadcast over batch
        return tok

    def forward(self, src_emb_or_idx, src_mask=None, already_embedded=False):
        # src may be token ids (LongTensor) or precomputed embeddings
        if already_embedded:
            src = src_emb_or_idx
        else:
            src = self.embed(src_emb_or_idx)

        B = src.size(0)
        plan = self.plan_emb.unsqueeze(0).expand(B, -1, -1)   # (B, K, d)
        x = torch.cat([plan, src], dim=1)                     # (B, K+T, d)
        x = self.ln_in(x)
        for blk in self.layers:
            x = blk(x, attention_mask=None)                   # bidirectional
        x = self.ln_out(x)
        P = x[:, :self.K, :]                                  # (B, K, d)
        return P
```

**Parameter sharing:** We recommend **tying token embeddings** with the decoder:

```python
plan_encoder = PlanEncoder(
    config,
    K=16,
    token_embedding=model.transformer.wte,
    position_embedding=getattr(model.transformer, 'wpe', None),
)
```

Blocks are **independent** from the decoder (best stability). You can make the encoder even cheaper (2–4 layers).

### 4) Thread guidance through the decoder

Add optional arguments to the methods that walk blocks:

* In `GPT._encode_tokens(...)`, pass `guidance_h` and `guidance_mask` down:

```python
def _encode_tokens(self, idx, attention_mask=None, guidance_h=None, guidance_mask=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size

    tok_emb = self.transformer.wte(idx)
    if hasattr(self.transformer, 'wpe'):
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
    else:
        x = self.transformer.drop(tok_emb)

    for block in self.transformer.h:
        x = block(x, attention_mask=attention_mask, guidance_h=guidance_h, guidance_mask=guidance_mask)

    x = self.transformer.ln_f(x)
    return x
```

* In `GPT.forward(...)`, accept and forward `guidance_h`/`guidance_mask`:

```python
def forward(self, idx, targets=None, attention_mask=None, loss_modifiers=None,
            guidance_h=None, guidance_mask=None):
    ...
    x = self._encode_tokens(idx, attention_mask=attention_mask, guidance_h=guidance_h, guidance_mask=guidance_mask)
    ...
```

> If you don’t pass guidance, nothing changes.

### 5) Using the plan encoder during **diffusion training**

At each reverse step `t` (conceptual trainer pseudocode):

```python
# x_t : current noised discrete tokens (B, T)
# Build encoder source: simplest is the *current noised tokens themselves*
with torch.no_grad():  # you can start with no grad for stability; unfreeze later
    src_emb = model.transformer.wte(x_t)  # (B, T, d)

P = plan_encoder(src_emb, already_embedded=True)  # (B, K, d)

# Optional: classifier-free guidance training (condition dropout)
use_cond = (torch.rand(B, device=x_t.device) > p_drop)  # e.g., p_drop = 0.1~0.2
guidance_h = P
guidance_h_uncond = None

# Compute logits with/without guidance depending on use_cond mask
logits_cond = model(idx=x_t, guidance_h=guidance_h)[0]         # (B, T, V)
logits_uncond = model(idx=x_t, guidance_h=guidance_h_uncond)[0]

# Supervise both paths where they’re active.
# Easiest: compute your diffusion loss on a mixed batch built from the mask.
```

**Notes:**

* Start by **detaching** `P` (no grad through encoder) for a few thousand steps, then allow gradients to flow if you want the plan to specialize. Alternatively, always keep the encoder no-grad (it still learns if you periodically unfreeze and update).
* Keep `K` small (8–32). Cross-attention cost is `O(B·T·K·H)`.

### 6) **Inference with CFG** (optional but recommended)

At sampling step `t`:

```python
# Build P once per step (or cache if source doesn’t change)
src_emb = model.transformer.wte(x_t)
P = plan_encoder(src_emb, already_embedded=True)

# Two decoder passes: unconditional and conditional
logits_uncond, _ = model(idx=x_t, guidance_h=None)
logits_cond, _   = model(idx=x_t, guidance_h=P)

# Mix with a late-rising guidance scale w(t) (near 0 early, 1.5–3.0 late)
logits_cfg = logits_cond + w_t * (logits_cond - logits_uncond)

# Continue your discrete diffusion step with logits_cfg
```

Implement `w_t` as a schedule function of timestep; e.g., linear from 0 → 2.0 over the last 40% of steps.

### 7) Config additions (lightweight)

Extend `GPTConfig` with:

```python
@dataclass
class GPTConfig:
    ...
    use_guidance: bool = True
    plan_tokens: int = 16
    plan_encoder_depth_factor: float = 0.5  # or int layers
    cond_dropout_prob: float = 0.1          # for CFG training
```

Wire these into your model/trainer construction.

### 8) Integration checklist

* [ ] CrossAttention module added.
* [ ] `Block` accepts `guidance_h`/`guidance_mask` and performs cross-attention when provided.
* [ ] `GPT._encode_tokens` and `GPT.forward` forward `guidance_*` args.
* [ ] `PlanEncoder` created; embeddings **tied** to decoder `wte`/`wpe` (recommended).
* [ ] Trainer builds `P` each step and passes it to the decoder.
* [ ] (Optional) CFG: add conditional/unconditional passes at inference; train with `cond_dropout_prob`.

---

## Practical tips & defaults

* **K (plan tokens)**: start with **16**.
* **Where to cross-attend**: every layer works; if tight on compute, **upper half** of layers only.
* **Training schedule**:

  * First 1–5k steps: **detach** plan (`with torch.no_grad()`).
  * Then allow gradients into the plan encoder, or train it with a **smaller LR**.
* **Corruption**: span masking / absorbing noise helps the plan learn structure.
* **RoPE vs absolute**: both fine. If you use RoPE in decoder, the plan encoder can still use absolute (simpler), since cross-attention doesn’t require positional alignment.

---

## FAQ

**Q: Are encoder and decoder weights shared?**
A: **No.** Blocks are independent by default (best stability). We *do* recommend sharing **token embeddings** (`wte`) (and `wpe` if using absolute), to align spaces and save params.

**Q: Do we need labels or summaries?**
A: No. The plan tokens are learned **unsupervised**—the decoder depends on them to reduce denoising loss.

**Q: What if I don’t want to change training right now?**
A: You can add the encoder and cross-attention and keep training identical; just pass `guidance_h=None` during training and **enable it at inference**. You’ll get some gains from CFG-style conditioning, but the best results come when the model has seen **cond-drop** during training.

---

## Minimal diff summary

* **New:** `CrossAttention`, `PlanEncoder`.
* **Changed:** `Block.forward(x, attention_mask, guidance_h=None, guidance_mask=None)`.
* **Changed:** `GPT._encode_tokens(..., guidance_h=None, guidance_mask=None)` and `GPT.forward(..., guidance_h=None, guidance_mask=None)`.
* **Trainer:** compute `P = plan_encoder(...)` per step; (optional) cond-drop for CFG training; at inference, do cond/uncond passes and mix logits.

That’s it. This is a small, safe set of changes that gives you a strong lever on long-range coherence while staying fully unsupervised.
