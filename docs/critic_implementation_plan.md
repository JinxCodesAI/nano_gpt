## Goal and Scope

Add an optional, non-breaking Critic head to the existing GPT model that learns to predict token-level error likelihoods and can guide re-masking during diffusion-style generation. This is an extension (default-off) that must not change existing behavior unless explicitly enabled via configuration.

Out of scope for this iteration:
- No changes to training loop structure in train.py or core/training_step.py beyond using the existing call paths
- No pushing branches or implementing a separate standalone “remasking model”

## Backward Compatibility Requirements
- New config flag add_critic_head: default False
- All current training modes must keep working unchanged when add_critic_head is False: LANGUAGE_MODEL, TOKEN_CLASSIFIER, SEQUENCE_SCORER
- Checkpoint load/resume must work for older checkpoints without this flag
- Sampling code should continue to work without critic unless explicitly enabled

## Audience and quick background

This document targets a Python-proficient developer who is not deeply familiar with ML or this codebase. It explains what to change, where, and why.

- Language modeling (LM): the model predicts tokens; training minimizes cross-entropy between predictions and ground-truth tokens.
- Diffusion-style demasking (in this repo): we iteratively fill masked tokens, then re-mask a subset and repeat. Today, the subset is chosen by heuristics (confidence or randomness).
- Critic head (this proposal): an auxiliary per-token classifier producing a logit that estimates “this token is likely wrong.” During training it learns from differences between the model’s own predictions and the ground truth. During inference it can guide which tokens to re-mask.

## Where things live in this repo (key references)

- model.py
  - GPTConfig (dataclass): holds model hyperparameters and mode (ModelMode)
  - GPT: transformer body plus heads
    - forward() dispatches to _forward_language_model/_forward_token_classifier/_forward_sequence_scorer
    - _forward_language_model(): current LM loss implementation
    - freeze_transformer_weights()/unfreeze_transformer_weights(): transfer learning support
- sample_utils.py
  - predict_and_sample_tokens(): fills masked positions using LM logits
  - apply_remasking_step(): decides what to re-mask next; supports external remasking_model, “intelligent_remasking” (self-confidence), or random
- sample.py
  - diffusion_generate(): generation loop calling predict_and_sample_tokens() then apply_remasking_step()
  - Loads mask_token_id/pad_token_id from dataset meta.pkl
- train.py
  - Assembles config, constructs GPTConfig and GPT, training orchestrated by Trainer
  - Uses DatasetConsumer (data/common) and CheckpointManager
  - validator: config/validator.py validates training configs
- core/training_step.py
  - TrainingStep.execute_step(): calls model(X, Y, loss_modifiers=...) inside AMP, handles grad scaling/accumulation/clip/step

These are the only places we need to touch or reference for the critic extension.

## High-Level Design
- Multi-task extension of LANGUAGE_MODEL only. The Critic head shares the transformer trunk and predicts a single logit per token representing “error likelihood”.
- Training (when enabled): LM loss is computed as today. Additionally, a second forward pass runs on a “filled” sequence (masked positions replaced by sampled/argmax tokens from the LM output) to compute a critic loss (BCEWithLogits). Final loss = LM loss (with existing loss modifiers applied) + alpha * critic_loss.
- Inference: When enabled and no external remasking model is provided, re-masking selection can use critic scores instead of 1 - confidence. Randomness blending and protection masks remain as they are today.

## Model Changes (model.py)
1) GPTConfig additions
- add_critic_head: bool = False (default)
- critic_alpha: float = 0.5 (weight for critic loss in training)
- critic_target_scope: str = 'masked_only' (compute critic targets/loss only on positions that were masked this iteration; future options: 'all')

These config fields are saved in checkpoints via existing model_args flow and default to safe values for older checkpoints.

2) Parameters
- Conditionally create a critic_head when add_critic_head is True:
  - critic_head = nn.Linear(n_embd, 1, bias=False)  # raw logits for BCEWithLogits
  - Keep separate from existing heads; does not interfere with weight tying
  - Ensure freeze/unfreeze methods keep heads trainable if transformer is frozen

3) Public helper for inference
- Add method: critic_scores(self, idx, attention_mask=None) -> Tensor[(B,T)]
  - Runs the same transformer encoding path up to ln_f and returns per-token critic logits (squeezed)
  - Requires add_critic_head True; else raise a clear error

4) Training integration inside LANGUAGE_MODEL path only
- In _forward_language_model, keep current logic to compute logits and base LM loss
- Apply existing loss modifiers to LM only (unchanged behavior and signatures)
- If add_critic_head and targets is not None:
  - Build critic_input by filling masked positions in the current batch with predicted tokens from the LM logits
    - Sampling policy: start with argmax for stability; later we can add multinomial sampling under a flag
    - Use the same ignore_index mask; identify masked positions as those where targets != ignore_index AND current input had mask token (requires access to the original idx). To avoid changing forward signature externally, compute mask_positions inside forward from idx vs. mask_token_id if available in consumer meta via config; if not available, default to "positions where targets != ignore_index" (documented behavior). See Risks below.
  - Forward pass 2 (with grad): logits_critic = critic_head(ENCODE(critic_input))
  - Build critic_target: float tensor shape (B,T,1) where 1.0 for error and 0.0 otherwise; scope controlled by critic_target_scope
    - masked_only: only consider positions that were masked in this iteration; ignore others
## Configuration wiring and persistence

- New GPTConfig fields (model.py): add_critic_head, critic_alpha, critic_target_scope (defaults safe)
- Persistence: these flow through train.py via model_args into checkpoints (CheckpointManager). Older checkpoints load as-is (defaults take effect).
- Validator: update config/validator.py to allow these keys if present. Keep defaults disabled to preserve existing behavior.

### Enabling the critic during training (two options)

1) Preferred minimal-change approach consistent with repo policy (avoid changing train.py):
   - Train a baseline LM checkpoint without critic (as today), then create a new config that sets init_from_checkpoint to that checkpoint and toggles add_critic_head=True when resuming; implement model-side logic that respects add_critic_head during resume. Note: because train.py constructs model_args from a fixed dict, enabling critic on a fresh-from-scratch run will require adding the keys to model_args (see option 2).

2) Small, explicit train.py wiring (optional, convenience):
   - In the block where model_args is constructed, include:
     - add_critic_head=globals().get('add_critic_head', False)
     - critic_alpha=globals().get('critic_alpha', 0.5)
     - critic_target_scope=globals().get('critic_target_scope', 'masked_only')
   - This keeps behavior identical by default and allows enabling via command-line/config overrides.

### Inference-time enabling

- No changes required to sample.py CLI: if model.config.add_critic_head is True and no external remasking_model is provided, apply_remasking_step can automatically use the critic path.

## Code references for each change

- model.py
  - class GPTConfig: add fields add_critic_head, critic_alpha, critic_target_scope
  - class GPT.__init__: if config.add_critic_head: self.critic_head = nn.Linear(n_embd, 1, bias=False)
  - GPT.freeze_transformer_weights(): ensure critic_head remains trainable (like lm_head)
  - GPT.critic_scores(idx, attention_mask=None): new helper that returns per-token logits (B,T)
  - GPT._forward_language_model(...): compute LM loss (with loss_modifiers), then, if add_critic_head and targets present, compute critic loss on filled sequences and add to total

- sample_utils.py
  - apply_remasking_step(...): add critic branch between remasking_model and intelligent_remasking

- sample.py
  - diffusion_generate(...): unchanged logic; rely on apply_remasking_step precedence

- config/validator.py
  - Allow/validate the three new config keys (no default enabling)

## How to enable & run (examples)

Training from scratch with critic (if option 2 wiring is added):

- In your training config (e.g., config/my_training/with_critic.py), set:
  - add_critic_head = True
  - critic_alpha = 0.5  # tune 0.1–1.0
  - critic_target_scope = 'masked_only'
- Run:
  - python train.py config/my_training/with_critic.py

Training by resuming from a baseline LM checkpoint (option 1):

- Keep train.py unchanged; use an LM checkpoint produced previously
- In your resume config:
  - init_from = 'resume'; init_from_checkpoint points to that checkpoint
  - add_critic_head = True (picked up via checkpoint model_args if present; else requires option 2 wiring)

Sampling with critic guidance:

- sample.py will detect critic availability via model.config.add_critic_head
- Ensure no external remasking_model is passed to favor the critic branch


Notes:
- We do not modify the signature of forward nor loss_modifiers interfaces
- We add a new helper critic_scores() for inference-time utilities

## Sampling/Inference Changes (sample_utils.py and sample.py)
1) sample_utils.apply_remasking_step
- Add an optional path to use critic scores when remasking_model is None and intelligent_remasking is False and the base model has add_critic_head=True
- Implementation sketch:
  - With torch.no_grad(), compute scores = model.critic_scores(prediction_tokens)
  - scores meaning: higher logit = higher error probability
  - Mask scores at positions that are not unmaskable or are protected
  - Blend with randomness_strength as today
  - Pick top-k (per row) indices to re-mask
  - Set those positions to mask_token_id
- Precedence order becomes: remasking_model > critic_head > intelligent_remasking > random

2) sample.py integration
- Detect availability of critic via getattr(model.config, 'add_critic_head', False)
- Pass this intent implicitly through to apply_remasking_step; no CLI change is required initially (optional future flag enable_critic_remasking)
- Do not alter any other generation flow

## Train/Eval Code (train.py, core/training_step.py)
- No structural changes. The entire feature is gated inside model forward and the sampling utilities
- DDP/AMP/GradScaler behavior stays identical
- Unfreezing logic remains identical (transformer may be frozen/unfrozen; the critic head is always trainable when present)

## Detailed Steps
1) Add config fields to GPTConfig with defaults and post-init no-op for BC
2) In GPT.__init__:
   - Instantiate critic_head if config.add_critic_head
   - Extend freeze_transformer_weights() to leave critic_head trainable
3) Implement GPT.critic_scores(idx, attention_mask=None)
   - Run the standard embedding/blocks/ln_f path and apply critic_head, returning logits.squeeze(-1)
4) Extend _forward_language_model to compute combined loss when add_critic_head and targets is not None
   - Compute LM logits and base LM loss (unchanged)
   - Apply loss_modifiers to LM loss only (unchanged)
   - Build critic_input by filling masked positions (see Design) and compute logits_critic
   - Build critic_target tensor and valid mask; compute loss_critic with BCEWithLogitsLoss
   - loss = loss_after_modifiers + critic_alpha * loss_critic
   - Return LM logits and combined loss
5) In sample_utils.apply_remasking_step add critic path as described; preserve existing API and default behavior
6) In sample.py, do a minimal detection of model.config.add_critic_head and let apply_remasking_step use the critic automatically when enabled

## Data/Targets for Critic
- Mask identification: prefer using (tokens == mask_token_id) captured before filling to identify positions originally masked in the iteration
- Targets: 1.0 when the filled token differs from ground truth Y; 0.0 otherwise
- Valid positions: exclude padding/ignored tokens; if meta supplies pad_token_id, mask those out
- Scope: start with masked_only to avoid trivial negatives on unmasked positions

## Logging
- At model init: log that critic head is enabled and critic_alpha
- Optionally (future): expose per-iteration scalar for critic loss via logger; for now, keep the training loop untouched
## Migration / Config Integration
- Add config flag add_critic_head and critic_alpha to training configs; default False/0.5
- Update config.validator to allow/validate the new keys (no enabling by default)
- No changes required to CheckpointManager; new fields flow through model_args as today

## Risks and Mitigations
- Mask identification inside model.forward requires the original mask positions; if not available, we will infer masked positions using (targets != ignore_index) as an approximation. Plan to refactor later to explicitly pass mask information through the batch (preferred) or provide mask_token_id in config/meta to reconstruct.
- Training cost increases ~2x for transformer passes; mitigate by enabling only where beneficial or lowering iterations
- If critic overwhelms LM gradients, tune critic_alpha down
- Numerical stability: use BCEWithLogitsLoss with reduction by valid count; clamp valid count to avoid divide-by-zero

## Future Extensions (Not in this iteration)
- Add multinomial sampling for critic_input under a flag
- Train a standalone dedicated remasking model and unify the interface
- Make critic usable for TOKEN_CLASSIFIER mode as auxiliary supervision if desired
- Richer critic targets (e.g., soft targets based on LM uncertainty)

