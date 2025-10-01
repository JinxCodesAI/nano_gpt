## Functional Specification: sample.py — Diffusion-based Text Generation

### 1) Purpose and Scope
- sample.py is a standalone text generation script supporting two modes:
  - Diffusion-based iterative demasking (default)
  - Standard autoregressive sampling
- It loads a trained model checkpoint, generates multiple samples in batch, optionally evaluates quality (Self or Judge model), and prints results and performance metrics.

### 2) High-level Capabilities
- Load main model checkpoint with safe backward-compat handling
- Load dataset vocabulary and special token metadata
- Diffusion generation with iterative remasking and multiple strategies:
  - Random remasking
  - Model-guided remasking via external remasking_model (if provided)
  - Critic-guided remasking using the main model’s critic head (if enabled)
  - "Intelligent" remasking using self-uncertainty from logits
- Optional seed text injection (prefix or random placement) that is protected from remasking
- Standard autoregressive generation via GPT.generate
- Optional quality evaluation of generated sequences using:
  - Self-confidence (average log probability)
  - A separate Judge (sequence scorer) model
- Config acquisition via configurator.py overrides and runtime flags
- Detailed console logging and throughput reporting

### 3) Configuration Parameters (established at top of script)
- Model loading
  - init_from: 'resume' (present for compatibility; model loads from explicit checkpoint path)
  - out_dir: directory with checkpoints
  - checkpoint_name: main LM checkpoint filename
- Generation
  - num_samples: batch size of samples to generate
  - sequence_length: length for diffusion generation (seq_len)
  - max_new_tokens: new tokens for standard sampling
  - seed: RNG seed (auto-randomized if -1)
  - device: 'cuda' or 'cpu'
  - dtype: 'float16'|'float32'|'bfloat16' (autocast type)
  - compile: whether to torch.compile the model
- Sampling method
  - sampling_method: 'diffusion'|'standard'
- Seed placement
  - seed_text: optional text to inject
  - SeedPlacement: PREFIX | RANDOM_PLACEMENT
  - seed_placement: strategy for placing seed_text tokens
- Diffusion parameters
  - temperature, top_p
  - repetition_penalty, repetition_window (used in token sampling for masked positions)
  - diffusion_iterations: total demasking iterations
  - start_ratio, end_ratio: schedule endpoints for remasking ratios
- Remasking parameters
  - randomness_strength: blending random vs guided selection
  - intelligent_remasking: enable remasking using self-estimated uncertainty
- Quality metric
  - QualityMetric: NONE | SELF | JUDGE
  - judge_checkpoint_name: required when QualityMetric == JUDGE
- Schedule parameters
  - schedule_type: 'linear' | 'custom'
  - masking_ratios: per-iteration ratios if custom
- Logging
  - use_verbose_logging, log_debug, show_progress
- Standard sampling parameters
  - start_text, remasking_model (optional; used only in diffusion path), std_temperature, top_k
- Config override
  - exec(open('configurator.py').read()) applies external overrides

### 4) Runtime and Device Setup
- Seeds torch and torch.cuda (if available)
- Enables TF32 and sets torch.amp autocast context per device and dtype

### 5) Model Loading (load_model_from_checkpoint)
- Validates checkpoint path, loads pickle via torch.load(weights_only=False)
- Extracts 'model_args' (required); backfills defaults for attention_type and position_encoding
- Clears any training-only init_from_checkpoint contained in model_args
- Constructs GPTConfig and GPT model, strips '_orig_mod.' prefix from state dict keys (torch.compile artifacts), loads weights, sets eval(), moves to device
- Optional torch.compile(model)
- Returns (model, checkpoint)

### 6) Vocabulary Loading (load_vocabulary)
- Attempts to infer dataset name from checkpoint['config'] (dict-like) under 'dataset'
- Fallback dataset is 'shakespeare_char'
- Loads data/<dataset>/meta.pkl and returns:
  - stoi, itos, vocab_size, dataset_name, meta
- meta may carry special token ids: 'mask_token_id', 'pad_token_id', 'base_vocab_size'

### 7) Special Tokens and Metadata Handling
- Uses model.config.vocab_size as the active vocab_size
- mask_token_id: from meta if present, otherwise fallback to vocab_size - 1 (warns)
- pad_token_id, base_vocab_size: read if present
- decode(token_ids): renders [MASK], [PAD], [UNK] where applicable
- decode_with_mask_char(token_ids, mask_char): same but masks with a single character

### 8) Diffusion Generation (diffusion_generate)
- Inputs: model, batch_size, total_length, iterations, mask_token_id, vocab_size, decode_fn, remasking_model, verbose, seed_ids, placement
- Initialization
  - tokens: (B, T) all set to mask_token_id
  - protected_mask: (B, T) False initially; if seed_ids provided, inject them either at PREFIX (start_idx=0) or RANDOM index; mark injected span True in protected_mask to prevent remasking
- Iterative loop for iteration in [0..iterations-1]
  1) Predict masked positions and sample tokens
     - Calls sample_utils.predict_and_sample_tokens with:
       - model, tokens, mask_token_id, temperature, top_p, repetition_penalty, repetition_window
       - vocab_size, device, return_logits=True, pad_token_id, base_vocab_size
     - Returns updated "tokens" (filled at masked positions), prediction_tokens (same), and logits
  2) Compute next-iteration remasking (skip after last iteration)
     - Calls sample_utils.apply_remasking_step with:
       - prediction_tokens, schedule params (schedule_type, masking_ratios, start_ratio, end_ratio), iteration & iterations
       - remasking_model (if provided)
       - randomness_strength
       - mask_token_id
       - base_model=model (for critic/intelligent remasking paths)
       - intelligent_remasking flag (auto-disabled when using critic-head remasking)
       - logits_from_predict=logits (to avoid extra forward pass for intelligent remasking)
       - protected_mask to avoid masking seed span
     - Strategy precedence in apply_remasking_step:
       1. External remasking_model (batched confidence)
       2. Critic-guided remasking (if model.config.add_critic_head True)
       3. Intelligent remasking (uncertainty from logits)
       4. Random remasking fallback
- Logging:
  - log_iteration_progress prints masked ratio and a short decoded preview on first and last iterations
  - Start banner shows remasking mode selected
- Returns: final tokens (B, T)

### 9) Standard Generation (standard_generate)
- Batches num_samples execution of model.generate
- If start_ids provided (from start_text), repeats across batch; otherwise picks a random first token per sample
- Calls model.generate(idx, max_new_tokens, temperature=std_temperature, top_k=top_k)
- Returns generated token sequences (B, T')

### 10) Quality Metrics (post-generation)
- Controlled by QualityMetric
- SELF
  - Calls calculate_selfconfidence_ratio(model, generated_tokens, mask_token_id, device, ctx)
  - Computes average log probability over non-mask tokens per sample
- JUDGE
  - Requires judge_checkpoint_name; loads another checkpoint as judge_model
  - Enforces judge_model.config.mode == SEQUENCE_SCORER
  - Calls calculate_judge_scores(judge_model, tokens, device, ctx) returning scores in [0,1] as 1 - evaluation
  - Measures judge inference time and throughput and reports per-sample score
- NONE
  - Skips evaluation

### 11) Main Execution Flow
1. Load main model from out_dir/checkpoint_name
2. Load vocabulary and meta; resolve special tokens
3. Prepare seed_ids from seed_text per stoi
4. Print generation settings
5. with torch.no_grad() and autocast ctx:
   - If sampling_method == 'diffusion': run diffusion_generate (B=num_samples, T=sequence_length)
     - Optionally compute quality metric (Self/Judge)
   - Else: run standard_generate(start_ids, max_new_tokens, std_temperature, top_k)
6. Print RESULTS section: each sample, optional quality score, and short character statistics
7. Compute timing summary from the first "Starting diffusion generation" print (or start_time for standard) to end of generation; compute tokens/s throughput
8. If Judge ran: print average/min/max judge scores and judge tokens/s
9. Print "GENERATION COMPLETE"

### 12) Error Handling and Validations
- sampling_method validated against {'diffusion','standard'}
- If schedule_type == 'custom' and masking_ratios provided: diffusion_iterations set to len(masking_ratios)
- load_model_from_checkpoint validates file existence and model_args presence; strips compile prefix keys
- load_vocabulary validates meta.pkl presence; falls back to default dataset when missing in checkpoint
- Judge mode enforces presence of judge checkpoint and correct model mode
- Intelligent remasking auto-disabled if critic-guided path is used (to avoid conflict)

### 13) External Dependencies and Integration Points
- model.py
  - GPTConfig, GPT, ModelMode
  - Critic guidance requires GPT with add_critic_head=True and critic_scores method
- sample_utils.py
  - linear_remasking_schedule (used within apply_remasking_step)
  - nucleus_sample (used within predict_and_sample_tokens)
  - predict_and_sample_tokens (masked-position logits extraction and sampling)
  - apply_remasking (per-sample random/intelligent remasking fallback)
  - apply_remasking_step (vectorized, prioritized remasking strategy)
  - calculate_selfconfidence_ratio (SELF metric)
  - calculate_judge_scores (JUDGE metric; handles optional CLS prepend and batching)
- configurator.py
  - Allows overriding script-level configuration variables at runtime

### 14) Inputs and Outputs
- Inputs
  - out_dir/checkpoint_name: main LM checkpoint (required)
  - data/<dataset>/meta.pkl: vocabulary + special tokens (required)
  - out_dir/judge_checkpoint_name: judge model (required only for JUDGE)
  - Optional: remasking_model instance for remasking guidance
- Outputs
  - Console prints: settings, per-iteration progress (if enabled), generated samples, quality scores, statistics, performance summary
  - No files are written by sample.py itself

### 15) Operational Notes and Constraints
- Device autocast is used when device == 'cuda' and dtype is set (float16/bfloat16)
- sequence_length must fit model.config.block_size during diffusion
- For Judge: optionally prepends a CLS token (if configured) and truncates to judge block_size
- When meta is missing special tokens, mask_token_id falls back to vocab_size-1; pad may be absent
- Repetition penalty is applied only for masked-position sampling within diffusion
- Seed text placement protects those tokens from remasking across iterations

### 16) Extensibility Considerations
- New remasking strategies can be integrated by extending apply_remasking_step priority branches
- Additional quality metrics can be integrated post-generation similar to SELF/JUDGE
- Schedule customization supported via masking_ratios per-iteration values
- Critic-guided remasking depends only on model configuration and artifacts provided by model.py



### 17) Mode configuration: exact switches and knobs
- Standard autoregressive mode (single forward per new token)
  - Required: sampling_method='standard'
  - Effective settings:
    - Generation: max_new_tokens, std_temperature, top_k
    - Prompt: start_text (string). If empty, each sample starts with a random single token.
    - Batch: num_samples (same prompt replicated across batch)
    - Unused/ignored in this mode: sequence_length, diffusion_iterations, start_ratio, end_ratio, randomness_strength, intelligent_remasking, remasking_model, temperature, top_p, repetition_penalty, repetition_window, schedule_type, masking_ratios.
  - Implementation path: standard_generate() -> model.generate(...)

- Diffusion iterative demasking mode (masked infill across fixed length)
  - Required: sampling_method='diffusion'
  - Required constraints: sequence_length <= model.config.block_size
  - Core knobs:
    - Iterations/schedule: diffusion_iterations, schedule_type ('linear'|'custom'), start_ratio, end_ratio, masking_ratios (when schedule_type='custom')
    - Token sampling: temperature, top_p; repetition_penalty, repetition_window (applies to masked-position sampling only)
    - Remasking control: randomness_strength (0..1), intelligent_remasking (bool), remasking_model (optional model)
    - Seeding: seed_text, seed_placement in {PREFIX, RANDOM_PLACEMENT}
  - Implementation path: diffusion_generate() -> predict_and_sample_tokens() -> apply_remasking_step() per iteration

### 18) Low-level dataflow and tensor shapes (diffusion)
- Initial state
  - tokens: LongTensor (B=num_samples, T=sequence_length) filled with mask_token_id
  - protected_mask: BoolTensor (B, T) all False; seed injection sets a contiguous True region
- Iteration i = 0..N-1
  1) Predict-and-sample masked positions
     - dummy_targets = zeros_like(tokens); logits, _ = model(tokens, targets=dummy_targets)
     - mask_positions = (tokens == mask_token_id)
     - For each batch b:
       - mask_indices = nonzero(mask_positions[b])
       - masked_logits = logits[b, mask_indices, :]
       - masked_logits adjustments:
         - masked_logits[:, mask_token_id] = -inf
         - if pad_token_id is not None: masked_logits[:, pad_token_id] = -inf
         - if base_vocab_size is not None and vocab_size > base_vocab_size: masked_logits[:, base_vocab_size:] = -inf
         - if masked_logits.shape[-1] > vocab_size: slice to [:, :vocab_size]
       - repetition penalty (if repetition_penalty != 1.0): for each mask position, call apply_repetition_penalty(single_logits[1,1,V], tokens[b:b+1, :], penalty, window) and write back
       - new_tokens = nucleus_sample(masked_logits, top_p=top_p, temperature=temperature)
       - prediction_tokens[b, mask_indices] = new_tokens
     - Return value used by caller: tokens := prediction_tokens; logits returned for remasking heuristics
  2) Remasking for next iteration (skip after last)
     - Compute mask_ratio for next iteration:
       - If schedule_type == 'custom' and masking_ratios provided: mask_ratio = masking_ratios[i+1] or end_ratio if OOB
       - Else linear_remasking_schedule(i+1, iterations, start_ratio, end_ratio)
     - k = int(T * mask_ratio) tokens per sample will be masked for next iteration
     - Build unmaskable = (prediction_tokens != mask_token_id) & ~protected_mask
     - Strategy precedence in apply_remasking_step():
       a) External remasking_model:
          - logits_r, _ = remasking_model(prediction_tokens)
          - confidence = logits_r[:,:,1] if last dim > 1 else logits_r.squeeze(-1)
          - scores = -confidence; mask non-unmaskable to -1e9; blend with randomness via scores = (1-randStrength)*scores + randStrength*rand()
          - top-k scores per row -> select mask positions; set to mask_token_id
       b) Critic-guided remasking (requires model.config.add_critic_head True):
          - critic_logits = model.critic_scores(prediction_tokens)  # shape (B,T)
          - scores = critic_logits; same masking/blending/top-k as above
       c) Intelligent remasking (requires intelligent_remasking True):
          - If logits_from_predict available: probs = softmax(logits_from_predict, -1); p_taken = probs.gather(-1, prediction_tokens.unsqueeze(-1)).squeeze(-1); scores = 1 - p_taken
          - Else fallback per-sample using base_model forward to compute uncertainty 1 - p(token)
          - Same masking/blending/top-k
       d) Random remasking: uniformly sample among unmaskable positions for k per row

### 19) Seeding behavior
- Tokenization: seed_ids = [stoi[c] for c in seed_text if c in stoi]
- Placement:
  - PREFIX: start_idx=0
  - RANDOM_PLACEMENT: start_idx ~ Uniform{0..(T - len(seed_ids))}, clipped to 0 if seed longer than T (then truncated)
- Protection: protected_mask[:, start_idx:start_idx+seed_len] = True to prevent remasking of seed span for all iterations

### 20) Quality metrics configuration and mechanics
- QualityMetric.NONE: skip evaluation
- QualityMetric.SELF:
  - calculate_selfconfidence_ratio(model, tokens, mask_token_id, device, ctx)
  - Internals: logits over full sequence via dummy targets; probs=softmax; for each sample, compute avg log prob of non-mask tokens (exclude [MASK])
- QualityMetric.JUDGE:
  - judge_checkpoint_name must be provided; loaded via load_model_from_checkpoint(judge_path, device)
  - Validation: judge_model.config.mode must equal ModelMode.SEQUENCE_SCORER
  - Scoring: calculate_judge_scores(judge_model, tokens, device, ctx)
    - If judge_model.config.cls_token_id is set, prepend CLS and truncate to judge block_size
    - judge output is in [0,1]; score = 1 - evaluation
  - Throughput accounting: judge_tokens_per_sample = min(sequence_len (+1 if CLS), judge_model.config.block_size)

### 21) Inter-module dependencies (must be preserved)
- sample.py -> model.py
  - Expects GPTConfig, GPT, ModelMode
  - Optional critic path requires GPT(config.add_critic_head=True) and GPT.critic_scores(idx) returning (B,T) logits
  - load_model_from_checkpoint builds GPTConfig from checkpoint['model_args'] and loads state dict after stripping '_orig_mod.'
- sample.py -> sample_utils.py
  - Uses linear_remasking_schedule, nucleus_sample, predict_and_sample_tokens, apply_remasking_step, calculate_selfconfidence_ratio, calculate_judge_scores
- sample_utils.py -> model contract
  - model(tokens, targets=dummy) must return (logits, loss) where logits shape is (B,T,V)
  - remasking_model(tokens) must return logits shaped (B,T,2) or (B,T,1)

### 22) Error conditions and invariants to keep during refactor
- If sampling_method not in {'diffusion','standard'}: raise ValueError
- If judge metric selected but judge_checkpoint_name missing or judge file not found: raise
- If judge model mode != SEQUENCE_SCORER: raise
- If sequence_length > model.config.block_size: assert/raise in model forward; doc requires caller to set appropriately
- If checkpoint missing or lacks 'model_args': raise
- During critic-guided remasking inference: do not require pad/mask ids; critic_scores works solely from idx
- In apply_remasking_step: never remask protected positions; never remask positions already equal to mask_token_id
- In predict_and_sample_tokens: never sample [MASK] or [PAD] or special tokens beyond base_vocab_size
- Scheduling: next-iteration mask ratio computed from iteration+1 (first remask happens after first prediction)

### 23) Configuration checklists per mode
- Standard
  - sampling_method='standard'
  - max_new_tokens set
  - Optional: start_text, std_temperature, top_k
  - Irrelevant: remasking config, diffusion-specific params
- Diffusion
  - sampling_method='diffusion'
  - sequence_length <= model.config.block_size
  - diffusion_iterations >= 1
  - schedule_type in {'linear','custom'}; if 'custom', masking_ratios length must be diffusion_iterations
  - temperature/top_p set; repetition_penalty/window as needed
  - Choose ONE remasking strategy (external remasking_model takes precedence, then critic, then intelligent, else random)
  - If using Judge metric: judge_checkpoint_name provided and judge model is SEQUENCE_SCORER

### 24) End-to-end algorithm summaries
- Standard: for step in 1..max_new_tokens: idx_cond = crop(idx); logits = model(idx_cond); sample next token with temperature/top_k; append; repeat.
- Diffusion: initialize all [MASK] (+ seed); for i in 0..N-1: fill masked tokens via predict_and_sample_tokens; if i<N-1, compute next mask set via apply_remasking_step; after final iteration decode.

### 25) Notes for maintainers
- Do not change priority order of remasking strategies; it is relied upon by current behavior.
- Maintain the protected seed mask semantics across refactors.
- Keep predict_and_sample_tokens behavior of restricting sampling to base vocab and excluding [MASK]/[PAD].
- Keep judge score definition as 1 - evaluation and batch everything.
