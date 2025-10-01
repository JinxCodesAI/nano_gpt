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

