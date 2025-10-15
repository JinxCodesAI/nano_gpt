# Developer Spec: Training-Free Variable-Length Refinement (Insert/Delete with Spaces)

## WHAT (scope & deliverables)

Implement **structural edits** inside the sampler to allow **variable-length** refinement during discrete diffusion **without retraining**:

1. **Insertions**: At each iteration, **insert literal space `' '` characters** at the **k_ins** most uncertain **gaps** (between positions *i* and *i+1*). Implementation: **right-shift** the suffix and write `' '` at the gap index.

2. **Deletions**: At each iteration, **delete** the **k_del** most “misfit” characters—positions where the model **prefers the right neighbor** to occupy the current slot. Implementation: **left-shift** the suffix, effectively removing the character at *i*.

3. **Independent budgets**: `k_ins` and `k_del` are **independent** per iteration (do **not** force equality). Sequence **length breathes** naturally.

4. **Length limits**: Respect `block_size`. If an insertion would overflow, **trim the tail**. When deleting, decrease the active window (`max_token_pos`) accordingly.

5. **No special placeholder token**: **Use the existing space character** as the gap/filler token by design (this is a product requirement). We explicitly **do not introduce a new PAD/BLANK token**.

6. **Schedules & knobs**: Provide **cosine or linear** schedules for per-iteration **insert**/**delete** ratios. (Re-noise schedule already exists and remains separate.) Defaults are included below.

Deliverables:

* Updated `sampling_utils.py`: scoring helpers + in-place shift utilities.
* Updated `sample.py`: per-iteration structural edits (insert/delete), budgeting, and length handling.
* Minimal configuration knobs with sensible defaults.
* Guardrails to avoid corrupting the fixed prompt.

---

## WHY (rationale & requirements)

* **Problem**: Fixed-length resampling struggles with edits that **change word length** (e.g., “rock” → “fox” or vice-versa). Without structure, the sampler must rewrite everything to the right.
* **Goal**: Enable **insertions** and **deletions** during refinement so the model can make **local** length changes (training-free).
* **Choice**: Use **space** as a gap token (explicit product choice). We will **only** remove spaces we inserted (via logic), never arbitrary spaces from the source text.
* **Design**: Keep **insert**/**delete** **budgets independent** (do not force them to match). Manage length softly by trimming at the block boundary and adjusting `max_token_pos` on deletes.

---

## HOW (implementation plan)

### Files to modify

* `sampling_utils.py` — add new utilities (pure functions, torch-native).
* `sample.py` — wire structural edits into the main sampling loop.

### New configuration knobs (in `sample.py`, near existing sampler/re-noise configs)

```python
# --- Structural edit knobs ---
edit_schedule        = 'cosine'   # 'linear' or 'cosine'
insert_ratio_start   = 0.04       # fraction of active length inserted early (e.g., 4%)
insert_ratio_end     = 0.00       # decays to zero
delete_ratio_start   = 0.04       # fraction of active length deleted early
delete_ratio_end     = 0.00
delete_margin        = 0.02       # ignore tiny advantages when scoring deletions
delete_lambda        = 0.30       # weight for the lookahead term Δ2
length_target_mode   = 'none'     # 'none' | 'to_max_new' (optional soft controller)
cooldown_distance    = 1          # optional: avoid re-editing same neighborhood
# Uses existing: block_size, initial_length, max_new_tokens, max_iterations, space_token_id
```

### Utilities to add to `sampling_utils.py`

Add the following (names chosen to be explicit):

```python
def uncertainty_from_logprobs(log_probs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    log_probs: (1, T, V) log softmax
    tokens:    (1, T)
    Returns u[i] = 1 - p_self[i] in [0,1], as (T,)
    """
    with torch.no_grad():
        T = tokens.size(1)
        idx = torch.arange(T, device=log_probs.device)
        p_self = log_probs[0, idx, tokens[0]].exp()
        return (1.0 - p_self).detach()

def score_gaps_for_insertion(u: torch.Tensor) -> torch.Tensor:
    """gap_score[i] = max(u[i], u[i+1]) for i in [0..T-2], shape (T-1,)"""
    if u.numel() < 2: return torch.empty(0, device=u.device)
    return torch.maximum(u[:-1], u[1:])

def right_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """Shift x[ start : end ) right by 1. Fill x[start] with fill_id. Drop old x[end-1]."""
    if end - start <= 0: return
    x[:, start+1:end] = x[:, start:end-1]
    x[:, start] = fill_id

def left_shift_(x: torch.Tensor, start: int, end: int, fill_id: int) -> None:
    """Shift x[ start : end ) left by 1. Fill x[end-1] with fill_id (space)."""
    if end - start <= 0: return
    x[:, start:end-1] = x[:, start+1:end]
    x[:, end-1] = fill_id

def select_topk_mask(scores: torch.Tensor, k: int, forbid_lo: int, forbid_hi: int) -> torch.Tensor:
    """
    scores: 1D tensor length N (gaps or positions).
    Returns boolean mask (N,) with up to k True.
    Forbids indices in [forbid_lo, forbid_hi) (half-open).
    """
    if scores.numel() == 0 or k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    mask = torch.ones_like(scores, dtype=torch.bool)
    lo = max(0, forbid_lo); hi = min(scores.numel(), forbid_hi)
    if hi > lo: mask[lo:hi] = False
    masked_scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
    k = min(k, int(mask.sum().item()))
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    topk_idx = torch.topk(masked_scores, k=k).indices
    sel = torch.zeros_like(scores, dtype=torch.bool)
    sel[topk_idx] = True
    return sel

def deletion_scores_from_probs(probs: torch.Tensor, tokens: torch.Tensor, margin: float = 0.0, lam: float = 0.3) -> torch.Tensor:
    """
    probs:  (1, T, V) softmax probabilities
    tokens: (1, T)
    Score desire to delete token at i (shift left). We use:
      Δ1 = p(i, right_neighbor) - p(i, current)
      Δ2 = p(i+1, next_after_right) - p(i+1, right_neighbor)
    del_score[i] = max(0, Δ1 - margin) + lam * max(0, Δ2 - margin)
    Returns shape (T-1,); last position cannot be deleted by left-shift heuristic.
    """
    T = tokens.size(1)
    if T < 2:
        return torch.zeros(1, device=tokens.device)
    c = tokens[0]                          # (T,)
    r = tokens[0, 1:]                      # (T-1,)
    p_self = probs[0, torch.arange(T), c]  # (T,)
    p_here_right = probs[0, torch.arange(T-1), r]   # (T-1,)
    d1 = p_here_right - p_self[:-1]                # (T-1,)

    if T >= 3:
        n = tokens[0, 2:]                              # (T-2,)
        p_next_next = probs[0, 1:-1, n]                # (T-2,)
        p_next_right = probs[0, 1:-1, r[:-1]]          # (T-2,)
        d2_core = p_next_next - p_next_right
        d2 = torch.cat([d2_core, torch.zeros(1, device=tokens.device)], dim=0)
    else:
        d2 = torch.zeros(T-1, device=tokens.device)

    d1 = torch.clamp_min(d1 - margin, 0.0)
    d2 = torch.clamp_min(d2 - margin, 0.0)
    return (d1 + lam * d2)
```

> These utilities rely only on `torch` and assume `x` is a `(1, T)` `LongTensor`.

### Wiring into `sample.py`

**A) Warmup pass** (seed stats for first iteration). After initializing `x` and before the loop:

```python
with torch.no_grad(), ctx:
    logits, _ = model(x)
    last_log_probs = torch.log_softmax(logits, dim=-1)  # (1, T, V)
```

**B) Per-iteration at the very top (before re-noise and forward)**

1. **Compute budgets** (`k_ins_s`, `k_del_s`) from ratios using the chosen schedule:

```python
act_len = max_token_pos - initial_length
r_ins = compute_noise_ratio(iteration, max_iterations, edit_schedule, insert_ratio_start, insert_ratio_end)
r_del = compute_noise_ratio(iteration, max_iterations, edit_schedule, delete_ratio_start, delete_ratio_end)
k_ins = max(0, int(r_ins * act_len))
k_del = max(0, int(r_del * act_len))

# Optional soft controller towards target length (initial_length + max_new_tokens)
if length_target_mode == 'to_max_new':
    target_len = min(block_size, initial_length + max_new_tokens)
    e = (max_token_pos - target_len)   # positive => too long
    if e > 0:
        k_del = min(k_del + min(3, e), act_len)
    elif e < 0:
        k_ins = min(k_ins + min(3, -e), max(0, block_size - max_token_pos))
```

2. **INSERT at top-k uncertain gaps (using previous stats)**

```python
# Uncertainty & gap scores on active window
u = uncertainty_from_logprobs(last_log_probs[:, :max_token_pos, :], x[:, :max_token_pos])
gap_scores = score_gaps_for_insertion(u)  # len = max_token_pos - 1

# Never touch fixed prompt: forbid gaps that start before initial_length
sel_gaps = select_topk_mask(gap_scores, k_ins, forbid_lo=0, forbid_hi=max(0, initial_length))

# Apply inserts from right->left to keep indices stable
gap_indices = torch.nonzero(sel_gaps, as_tuple=False).flatten().tolist()
gap_indices.sort(reverse=True)
for g in gap_indices:
    # Insert a space between positions g and g+1 by shifting [g : max_token_pos) right
    right_shift_(x, start=g, end=min(block_size, max_token_pos), fill_id=space_token_id)
    max_token_pos = min(block_size, max_token_pos + 1)  # trim at block boundary
```

3. **DELETE at top-k misfit positions (using previous stats)**

```python
last_probs = last_log_probs.exp()
del_scores = deletion_scores_from_probs(last_probs[:, :max_token_pos, :], x[:, :max_token_pos],
                                        margin=delete_margin, lam=delete_lambda)
# Forbid positions < initial_length
sel_del = select_topk_mask(del_scores, k_del, forbid_lo=0, forbid_hi=max(0, initial_length))

# Apply deletes from left->right
del_indices = torch.nonzero(sel_del, as_tuple=False).flatten().tolist()
del_indices.sort()
for i in del_indices:
    left_shift_(x, start=i, end=max_token_pos, fill_id=space_token_id)
    max_token_pos = max(initial_length, max_token_pos - 1)
```

**C) Existing steps (unchanged order)**

* **Re-noise** (if enabled), **forward pass**, **temperature**, **sample**, **clamp prompt**, **zero beyond `max_token_pos`**.
* At the **end** of the iteration: set `last_log_probs = log_probs` to drive the next iteration’s structural decisions.

> Placement is important: **structural edits → re-noise → forward → sample**.

### Guardrails / invariants

* **Never** edit indices `< initial_length` (prompt is frozen).
* When **inserting**, always process **right→left**; when **deleting**, process **left→right**—this keeps index math correct.
* If an insertion would exceed `block_size`, the right-shift drops the tail; we then clamp `max_token_pos` to `block_size`.
* If we over-delete, clamp `max_token_pos ≥ initial_length`.
* Optional: implement a **cooldown** (do not select gaps/positions within ±`cooldown_distance` of any index edited in the previous iteration) to reduce oscillations.

### Recommended defaults

* `edit_schedule = 'cosine'`
* `insert_ratio_start = delete_ratio_start = 0.04` (4% of active length), `*_end = 0.0`
* `delete_margin = 0.02`, `delete_lambda = 0.30`
* Keep your existing **re-noise** schedule independent (e.g., start 0.10–0.15 → 0)

---

## Scoring definitions (for reference)

* **Uncertainty per position**:
  ( u_i = 1 - p(i, x_i) ), where ( p(i, a) = \mathrm{softmax}(\mathrm{logits}_i)[a] ).

* **Gap score for insertion** between positions *i* and *i+1*:
  ( \text{gap_score}[i] = \max(u_i, u_{i+1}) ).

* **Deletion score** at position *i* (shift left desirable):
  Let ( c = x_i ), ( r = x_{i+1} ), ( n = x_{i+2} ). Define
  ( \Delta_1 = p(i, r) - p(i, c) ) and ( \Delta_2 = p(i{+}1, n) - p(i{+}1, r) ).
  Then
  ( \text{del_score}[i] = \max(0, \Delta_1 - \text{margin}) + \lambda \cdot \max(0, \Delta_2 - \text{margin}) ).

---

## Testing checklist

1. **Unit tests on toy strings**:

   * Insertion only: confirm right-shifts and space inserts at target gaps; length increases and clamps at `block_size`.
   * Deletion only: confirm left-shifts at target positions; length decreases but never below `initial_length`.
   * Mixed edits: ensure index stability (no off-by-one), correct `max_token_pos` updates.

2. **Invariants**:

   * Prompt substring `x[:initial_length]` remains unchanged across all iterations.
   * No index edits occur outside `[initial_length, max_token_pos)`.

3. **A/B sanity runs**:

   * With edits OFF vs ON, verify qualitative improvements on variable-length fixes (e.g., “rock↔fox”).
   * With `length_target_mode='to_max_new'`, check the active length tracks the target without oscillation.

---
