# Attention-Mask Specification for BERT-style Unmasking (MLM & Iterative Decoding)

This spec defines **what tokens are visible as keys/values** under different circumstances, so you get stable training and fast inference (FlashAttention/SDPA fast path). It assumes **bidirectional self-attention** and **key-padding masks** (not pairwise masks).

---

## 1) Mask semantics (what “1/0” means)

* We use a **key-padding mask** `attn_mask ∈ {0,1}^{B×T}`.

  * `1` → this position **is visible** as a key/value.
  * `0` → this position **is hidden** as a key/value (no query can attend to it).
* Inside the model, expand to `(B,1,1,T)` if needed. **Do not** build a per-pair `(B,T,T)` mask—this usually disables FlashAttention/SDPA fast kernels.

---

## 2) Token visibility policy

Let the vocabulary include at least:

* `[PAD]`: padding
* `[MASK]`: masked token placeholder
* `[CLS]` / `[BOS]` (optional but recommended): anchor token(s)
* Regular tokens (“content”)

### Always

* **Hide `[PAD]` keys**: `attn_mask[pos_PAD] = 0`.

### Optional (switchable) policy for `[MASK]` keys

* **Block**: `attn_mask[pos_MASK] = 0` → nobody attends to masked slots.
* **Allow**: `attn_mask[pos_MASK] = 1` → masked slots are part of context.

> You will select **Block** or **Allow** depending on the scenario below.

### Anchors / safety

* Ensure at least one stable key is always visible (e.g., keep `[CLS]` visible).
* Do **not** mask out all positions in any sequence (guard against empty-key edge cases).

---

## 3) Scenarios & recommended settings

### A) Classic BERT MLM training (single forward pass)

* Goal: match the original MLM regime.
* **Visibility**:

  * `[PAD]`: **hide**
  * `[MASK]`: **allow** (keep visible)
  * Content tokens: **visible**
  * Anchors (`[CLS]`/`[BOS]`): **visible**
* Rationale: In classic MLM, the learned `[MASK]` embedding can carry useful signal; allowing it is standard.

### B) Diffusion/Iterative Unmasking **training** (to match inference)

* Goal: train under the same visibility pattern used at decode time.
* **Visibility**:

  * `[PAD]`: **hide**
  * `[MASK]`: **hide** (recommended)
  * Content tokens: **visible**
  * Anchors: **visible**
* Rationale: When most tokens are masked, letting queries look at dozens of `[MASK]` vectors is noisy and slower to converge. Hiding them focuses attention on revealed context.

> If you mix classic MLM and iterative training, you can **probabilistically** choose between “allow” and “block” per batch (e.g., 70% block, 30% allow).

### C) Diffusion/Iterative Unmasking **inference/decoding**

Choose one of two stable policies:

1. **Fixed “Block-MASK”** (simple & robust)

   * `[PAD]`: **hide**, `[MASK]`: **hide** at every step.
   * Best default for stability and speed.

2. **Ratio-aware schedule** (slightly more expressive)

   * Let `r = mean(tokens == MASK)` (fraction masked).
   * While `r ≥ 0.5`: **hide** `[MASK]`.
   * Once `r < 0.5`: **allow** `[MASK]` (optional).
   * Benefit: after sufficient context is revealed, allowing `[MASK]` can transmit whatever global bias the model stored in the `[MASK]` embedding.

> In both cases, **content tokens attend to content tokens (and anchors) only**; they never need to attend to `[PAD]`, and—under the fixed policy—never to `[MASK]`.

### D) Critic / Auxiliary scorers (per-step quality estimates)

* Use **the same mask as the generator** for that step.
* Minimal variant: always **hide** `[PAD]` and `[MASK]`.

### E) Evaluation / Perplexity-like scoring on masked inputs

* For MLM-style eval that emulates BERT papers: **allow** `[MASK]`.
* For diffusion-style eval: **hide** `[MASK]`.

---

## 4) API & helper

```python
def make_attn_mask(tokens, pad_token_id=None, mask_token_id=None,
                   block_mask_keys: bool = True,
                   always_keep_ids: tuple | None = None):
    """
    Returns (B,T) mask with 1=keep-as-key, 0=hide-as-key.
    """
    attn = torch.ones_like(tokens, dtype=torch.long, device=tokens.device)
    if pad_token_id is not None:
        attn &= (tokens != pad_token_id).long()
    if block_mask_keys and (mask_token_id is not None):
        attn &= (tokens != mask_token_id).long()
    if always_keep_ids:
        keep = None
        for _id in always_keep_ids:
            k = (tokens == _id).long()
            keep = k if keep is None else (keep | k)
        attn = torch.where(keep.bool(), torch.ones_like(attn), attn)
    return attn
```

**How to call:**

* **Classic MLM train/eval:** `block_mask_keys=False`
* **Iterative train/infer:** `block_mask_keys=True` (or ratio-aware toggle)
* Always pass `pad_token_id`, and (if used) `always_keep_ids=(cls_id, bos_id)`.

---

## 5) Efficiency constraints (to keep fast kernels)

* Use **key-padding masks only** (shape `(B,T)` → expand to `(B,1,1,T)` internally).
* Avoid per-pair `(B,T,T)` masks (e.g., “let masked queries attend only unmasked keys”)—they often disable FlashAttention/SDPA fast paths.
* Keep attention **bidirectional** for unmasking tasks (causal is the wrong bias).

---

## 6) Special cases & batching

* **Padding:** left or right pad is fine—just ensure `[PAD]` keys are hidden.
* **Fully masked sequences:** guarantee at least one visible anchor key (`[CLS]` kept visible and never replaced by `[MASK]`), or you’ll have an empty key set.
* **Packed samples (multiple documents concatenated):**

  * Preferred: **do not** hard-pack unrelated samples into one sequence unless you also provide a block-diagonal `(B,T,T)` mask (which is slower).
  * If you must pack, accept the perf hit and build a block-diagonal pairwise mask—or insert strong delimiters and rely on the model to ignore cross-doc attention (not guaranteed).

---

## 7) Worked examples

### Example 1 — Classic MLM training

* Input: `"[CLS] it [MASK] raining [PAD] [PAD]"`
* Mask:

  * keep keys: `[CLS]`, `it`, `[MASK]`, `raining`
  * hide keys: `[PAD]`, `[PAD]`
* Call: `block_mask_keys=False`

### Example 2 — Iterative step (early, 80% masked)

* Input contains many `[MASK]`.
* Mask:

  * keep keys: anchors + any **revealed** content
  * hide keys: `[MASK]` + `[PAD]`
* Call: `block_mask_keys=True`

### Example 3 — Iterative step (late, 20% masked; ratio-aware)

* If `r < 0.5`: set `block_mask_keys=False` to allow `[MASK]` keys; else keep blocking.

---

## 8) Testing checklist

* ✅ FlashAttention/SDPA stays enabled (verify runtime and kernel logs).
* ✅ No batch has an all-zero key set.
* ✅ Loss computed **only** on masked targets (ignore index elsewhere).
* ✅ Switching `block_mask_keys` between train and infer changes results as expected (sanity plots).
* ✅ PAD never influences attention maps.

---

## 9) TL;DR table

| Scenario                                  | `[PAD]` keys | `[MASK]` keys               | Anchors (`[CLS]/[BOS]`) |
| ----------------------------------------- | ------------ | --------------------------- | ----------------------- |
| Classic BERT MLM (train/eval)             | Hide         | **Allow**                   | Keep visible            |
| Iterative unmasking **training**          | Hide         | **Hide**                    | Keep visible            |
| Iterative unmasking **inference** (fixed) | Hide         | **Hide**                    | Keep visible            |
| Iterative unmasking (ratio-aware)         | Hide         | Hide if `r≥0.5`, else Allow | Keep visible            |
| Critic / quality scoring                  | Hide         | Hide (recommended)          | Keep visible            |

Use this as the authoritative reference when wiring `attention_mask` in training loops and decoders.
