# Diffusion-based LLM Development Plan

## Overview
Transform the current nanoGPT implementation into a diffusion-based language model through incremental changes, adding bidirectional attention, multi-phase training, and iterative demasking inference without major rewrites.

## Target System Description
Build a diffusion-based LLM with the following key features:

### Training Approach
- **Bidirectional attention**: Model can attend to all positions in sequence (no causal masking)
- **Masked token prediction**: Instead of next-token prediction, predict randomly masked tokens
- **Multi-phase training**: 
  - Phase 1 (N1 iterations): Identity task - predict same token for unmasked positions, uniform distribution for masked
  - Phase 2 (N2 iterations): Gradually increase probability of correct token from uniform to 100%
  - Phase 3 (N3 iterations): Standard cross-entropy loss on masked positions
  - Phase 4 (N4 iterations, parallel to phases 2-3): Entropy penalty with adaptive multiplier

### Data Generation
- **Masking strategies**:
  - Independent: Random masking with probability p
  - Sticky: Neighbor-aware masking (p1 if neighbors not masked, p2 if neighbors masked, over k rounds)
- **Asynchronous generation**: Background CPU-based data preparation with GPU-based training
- **Fixed validation**: Pre-generated validation set for consistent comparison

### Inference
- **Iterative demasking with remasking**: Start with all masks → unmask all tokens → re-mask X% → repeat for I iterations
- **Multiple remasking algorithms**: Different strategies for choosing X% based on iteration and total iterations
- **Configurable**: Number of iterations I and remasking percentage schedules

## Current Codebase Analysis
- **model.py**: 331 lines - GPT model with causal self-attention
- **train.py**: 337 lines - Training loop with standard next-token prediction  
- **sample.py**: 90 lines - Autoregressive text generation
- Architecture: Clean and readable, minimal dependencies

## Development Philosophy
- **Incremental changes**: Build on existing code without major rewrites
- **Diffusion-focused**: Transform to support only diffusion-based training/inference
- **Early validation**: Get inference working quickly to verify training quality
- **Detailed logging**: Track performance at every step for optimization

---

## Milestone 1: Basic Masking Data Generation ✅ COMPLETED
**Duration**: 1-2 days  
**Goal**: Convert train.py from autoregressive to masked token prediction

### Detailed Tasks:

#### 1.1 Add Masking Configuration to train.py
```python
# Add to config defaults in train.py
dataset = 'shakespeare_char'     # Use character-level Shakespeare dataset
guaranteed_unmasked = 0.15       # Guaranteed fraction of tokens to keep unmasked

# Gradual transition from independent to sticky masking
sticky_transition_start = 20000  # When to start introducing sticky masking
sticky_transition_end = 100000   # When to reach full sticky masking
sticky_rounds = 3                # Number of sticky masking rounds
sticky_p1_p2_multiplier = 3.0    # Multiplier for sticky_p2 = sticky_p1 * multiplier
```

#### 1.2 Modify get_batch() Function
**Current behavior**: Returns (x, y) where y = x shifted by 1 position  
**New behavior**: Returns (masked_x, original_y, mask) where:
- `masked_x`: Input with random tokens replaced by mask_token_id
- `original_y`: Original tokens (targets for masked positions only)
- `mask`: Boolean mask indicating which positions were masked

**Implementation**:
```python
def get_batch(split):
    # Get original data (keep existing logic)
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    # Dynamically sample masking probability each batch
    masking_prob = torch.rand(1).item() * (1.0 - guaranteed_unmasked)
    
    # Create masks: 1 where we mask, 0 where we keep original
    mask = torch.rand(x.shape) < masking_prob
    
    # Create input with masked tokens
    masked_x = x.clone()
    masked_x[mask] = mask_token_id
    
    # Target is original x, loss computed only on masked positions
    y = x.clone()
    
    # Move to device
    if device_type == 'cuda':
        masked_x = masked_x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        masked_x, y, mask = masked_x.to(device), y.to(device), mask.to(device)
    
    return masked_x, y, mask
```

#### 1.3 Update Model Forward Pass and Loss
**Current**: `logits, loss = model(X, Y)` computes autoregressive loss  
**New**: Compute loss only on masked positions

```python
# In training loop, replace:
# X, Y = get_batch('train')
# logits, loss = model(X, Y)

# With:
X, Y, mask = get_batch('train')
logits, _ = model(X, None)  # Don't compute loss in model

# Compute masked language model loss
logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
targets = Y.view(-1)  # (batch_size * seq_len,)
mask = mask.view(-1)   # (batch_size * seq_len,)

# Loss only on masked positions
loss = F.cross_entropy(logits[mask], targets[mask])
```

#### 1.4 Update Vocabulary Size from meta.pkl
**Current**: `vocab_size = 50304` (GPT-2 padded)  
**New**: Load vocab_size from meta.pkl and add 1 for mask token

**Implementation**:
```python
# In train.py, after the existing meta.pkl loading logic (line 137-144)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # Set mask_token_id to be vocab_size (next available ID)
    mask_token_id = meta_vocab_size
    extended_vocab_size = meta_vocab_size + 1  # Add 1 for mask token
    print(f"mask_token_id = {mask_token_id}, extended_vocab_size = {extended_vocab_size}")
else:
    print("No meta.pkl found, using default GPT-2 vocab")
    mask_token_id = 50304
    extended_vocab_size = 50305

# Update model initialization (line 155)
model_args['vocab_size'] = extended_vocab_size if meta_vocab_size is not None else 50305
```

#### 1.5 Update estimate_loss() Function
```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)  # Updated to return mask
            with ctx:
                logits, _ = model(X)
                # Compute masked loss
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = Y.view(-1)
                mask_flat = mask.view(-1)
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

### Deliverables:
- Modified get_batch() returning (masked_input, targets, mask)
- Masked language model loss computation
- Extended vocabulary for mask token
- Working basic diffusion training
- Updated validation loss computation

### Validation:
- Training loss decreases over iterations ✅
- Model learns to predict masked tokens correctly ✅
- Masking pattern shows ~15% of tokens masked per batch ✅
- Validation loss computed consistently on masked positions ✅

### Implementation Notes:
- **Identity Task**: Model now trains on ALL positions - masked tokens learn original token prediction, unmasked tokens learn identity task (input = output)
- **CPU Default**: Set device='cpu' by default for compatibility
- **Dynamic Masking**: masking_prob sampled uniformly from (0, 1-guaranteed_unmasked) each batch
- **Extended Vocabulary**: 65 chars + 1 mask token = 66 total vocab size from shakespeare_char dataset
- **Corrected Loss**: Removed conditional masking check to align with multi-phase training requirements

---

## Milestone 2: Basic Demasking Inference ✅ COMPLETED
**Duration**: 1-2 days
**Goal**: Implement simple iterative demasking in sample.py for training quality verification

**Note**: This works with existing model.py without changes. The model's forward method `model(X, None)` returns logits without computing loss, which is exactly what we need for inference.

### Detailed Tasks:

#### 2.1 Convert sample.py from Autoregressive to Demasking ✅
**Current behavior**: Autoregressively generate tokens one by one
**New behavior**: Start with all masks, iteratively unmask tokens using existing model architecture

#### 2.2 Add Demasking Configuration ✅
```python
# Add to sample.py config defaults
diffusion_iterations = 10        # Number of demasking rounds
remasking_schedule = 'linear'    # 'linear' or 'exponential'
sequence_length = 1024           # Total length of generated sequence
```

#### 2.3 Implement Remasking Algorithms ✅
**Correct inference process**: Start with all masks → unmask all → re-mask X% → repeat I times

```python
import math

def linear_remasking_schedule(total_iterations, current_iteration):
    """Linear decrease in remask ratio from high to low"""
    # Start with high remasking (e.g., 80%), end with low (e.g., 10%)
    start_ratio = 0.8
    end_ratio = 0.1
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    return start_ratio - progress * (start_ratio - end_ratio)

def exponential_remasking_schedule(total_iterations, current_iteration):
    """Exponential decrease in remask ratio"""
    start_ratio = 0.8
    end_ratio = 0.1
    progress = current_iteration / (total_iterations - 1) if total_iterations > 1 else 1.0
    # Exponential decay
    decay_factor = -2.0  # Controls steepness
    exp_progress = (math.exp(decay_factor * progress) - 1) / (math.exp(decay_factor) - 1)
    return start_ratio - exp_progress * (start_ratio - end_ratio)

def diffusion_generate(model, total_length, iterations, schedule='linear', mask_token_id=None, decode_fn=None, decode_mask_fn=None, verbose=True):
    """
    Generate text using diffusion-based iterative demasking

    Args:
        model: Trained diffusion model
        total_length: Total length of sequence to generate
        iterations: Number of demasking/remasking iterations
        schedule: Remasking schedule ('linear' or 'exponential')
        mask_token_id: ID of the mask token
        decode_fn: Function to decode tokens to text (handles mask tokens)
        decode_mask_fn: Function to decode tokens with mask character
        verbose: Whether to print iteration results
    """

    # Start with ALL positions masked (pure diffusion approach)
    tokens = torch.full((1, total_length), mask_token_id, dtype=torch.long, device=device)
    
    for iteration in range(iterations):
        # Step 1: Unmask ALL tokens (predict for all masked positions)
        current_mask_positions = (x[0] == mask_token_id)
        if current_mask_positions.sum() > 0:
            with torch.no_grad():
                logits, _ = model(x)
            
            # Sample tokens for all masked positions
            mask_positions = torch.where(current_mask_positions)[0]
            mask_logits = logits[0, mask_positions]  # (num_masked, vocab_size)
            probs = torch.softmax(mask_logits, dim=-1)
            new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Unmask all positions
            x[0, mask_positions] = new_tokens
        
        # Step 2: Re-mask X% of tokens (excluding context)
        if iteration < iterations - 1:  # Don't remask on final iteration
            # Calculate remask percentage for this iteration
            if algorithm == 'linear':
                remask_ratio = linear_remasking_schedule(iterations, iteration + 1)
            elif algorithm == 'exponential':
                remask_ratio = exponential_remasking_schedule(iterations, iteration + 1)
            else:
                remask_ratio = 0.5  # Default
            
            # Only remask generated tokens (not context)
            generatable_positions = torch.arange(context_length, sequence_length, device=device)
            num_to_remask = int(len(generatable_positions) * remask_ratio)
            
            if num_to_remask > 0:
                # Randomly select positions to remask
                remask_indices = torch.randperm(len(generatable_positions))[:num_to_remask]
                positions_to_remask = generatable_positions[remask_indices]
                x[0, positions_to_remask] = mask_token_id
    
    return x[0]
```

#### 2.4 Replace Autoregressive Generation ✅
Replace the autoregressive generation loop in sample.py with demasking generation:

```python
# Replace autoregressive generation with pure diffusion generation:
for k in range(num_samples):
    generated_tokens = diffusion_generate(
        model,
        sequence_length,
        diffusion_iterations,
        remasking_schedule,
        mask_token_id,
        decode,
        decode_with_mask_char,
        verbose=True
    )
    print(f"\nFINAL RESULT:")
    print(decode(generated_tokens.tolist()))
```

### Deliverables:
- Modified sample.py with demasking-only inference ✅
- Linear and exponential remasking algorithm implementation ✅
- Configurable demasking parameters ✅
- Working text generation for training verification ✅
- Enhanced logging with iteration-by-iteration output ✅
- Proper mask token handling in decode functions ✅

### Validation:
- Generates coherent text sequences ✅
- Demasking progresses from all masks to complete text ✅
- Can verify training quality immediately after Milestone 1 ✅
- Generated text quality improves as training progresses ✅

### Implementation Notes:
- **Pure Diffusion Approach**: Implemented true diffusion generation starting with ALL tokens masked (no prompt/context)
- **Iterative Demasking**: Successfully implemented the core diffusion process: unmask all → re-mask X% → repeat
- **Remasking Schedules**: Both linear and exponential schedules working correctly
- **Model Integration**: Fixed inference issue by passing dummy targets to get full sequence logits
- **Enhanced Logging**: Added detailed iteration-by-iteration logging showing masked/unmasked states
- **Mask Character Display**: Uses '#' character to visualize masked positions during generation
- **Vocabulary Handling**: Properly handles mask tokens (ID=65) in decode functions
- **Working Generation**: Successfully generates coherent Shakespeare-style text from trained checkpoint

---

## Milestone 3: Basic Logging and Performance Monitoring
**Duration**: 1 day  
**Goal**: Add essential timing and performance tracking before advanced features

### Detailed Tasks:

#### 3.1 Add Timing Infrastructure to train.py
```python
import time
from collections import defaultdict

# Add timing tracking
class Timer:
    def __init__(self):
        self.times = defaultdict(list)
    
    def time_function(self, name):
        """Context manager for timing function calls"""
        class TimerContext:
            def __init__(self, timer, name):
                self.timer = timer
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.timer.times[self.name].append(elapsed)
        
        return TimerContext(self, name)
    
    def get_average(self, name, last_n=100):
        """Get average time for last N calls"""
        if name not in self.times or not self.times[name]:
            return 0.0
        return sum(self.times[name][-last_n:]) / min(len(self.times[name]), last_n)

# Global timer
timer = Timer()
```

#### 3.2 Add Timing to Critical Sections
```python
# Time data generation
with timer.time_function('data_generation'):
    X, Y, mask = get_batch('train')

# Time forward pass
with timer.time_function('forward_pass'):
    logits, _ = model(X)

# Time loss computation  
with timer.time_function('loss_computation'):
    loss = compute_masked_loss(logits, Y, mask)

# Time backward pass
with timer.time_function('backward_pass'):
    scaler.scale(loss).backward()

# Time validation
if iter_num % eval_interval == 0:
    with timer.time_function('validation'):
        losses = estimate_loss()
```

#### 3.3 Enhanced Logging Output
```python
# Add to logging output every log_interval iterations
if iter_num % log_interval == 0 and master_process:
    # Existing logging
    lossf = loss.item() * gradient_accumulation_steps
    if local_iter_num >= 5:
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    
    # New detailed timing
    data_time = timer.get_average('data_generation') * 1000
    forward_time = timer.get_average('forward_pass') * 1000  
    loss_time = timer.get_average('loss_computation') * 1000
    backward_time = timer.get_average('backward_pass') * 1000
    
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    print(f"  data: {data_time:.2f}ms, forward: {forward_time:.2f}ms, loss: {loss_time:.2f}ms, backward: {backward_time:.2f}ms")
    
    # Validation timing (when applicable)
    if iter_num % eval_interval == 0:
        val_time = timer.get_average('validation') * 1000
        print(f"  validation: {val_time:.2f}ms")
```

#### 3.4 Masking Statistics Logging
```python
# Add masking statistics to logging
def log_masking_stats(mask, iter_num):
    """Log statistics about masking patterns"""
    mask_ratio = mask.float().mean().item()
    batch_size, seq_len = mask.shape
    
    # Count consecutive masked regions
    mask_np = mask.cpu().numpy()
    consecutive_regions = []
    for batch_idx in range(batch_size):
        regions = []
        current_length = 0
        for pos in range(seq_len):
            if mask_np[batch_idx, pos]:
                current_length += 1
            else:
                if current_length > 0:
                    regions.append(current_length)
                    current_length = 0
        if current_length > 0:
            regions.append(current_length)
        consecutive_regions.extend(regions)
    
    avg_region_length = sum(consecutive_regions) / len(consecutive_regions) if consecutive_regions else 0
    
    if iter_num % (log_interval * 10) == 0:  # Less frequent detailed stats
        print(f"Masking stats: {mask_ratio:.3f} ratio, {avg_region_length:.1f} avg region length")

# Add to training loop
if iter_num % log_interval == 0:
    log_masking_stats(mask, iter_num)
```

### Deliverables:
- Timer infrastructure for performance tracking
- Detailed timing logs for all training components
- Masking pattern statistics
- Performance baseline before advanced features

### Validation:
- Clear timing breakdown of training bottlenecks
- Masking statistics match expected patterns
- Easy identification of performance issues
- Baseline metrics for optimization

---

## Milestone 4: Bidirectional Attention Conversion
**Duration**: 1-2 days  
**Goal**: Convert model to use bidirectional attention for improved masked language modeling

### Detailed Tasks:

#### 4.1 Convert to Bidirectional Attention in model.py
**Current**: `CausalSelfAttention` with causal masking prevents attending to future tokens  
**Target**: `BidirectionalSelfAttention` allows attending to all sequence positions

**Implementation**:
```python
# In model.py, rename class
class BidirectionalSelfAttention(nn.Module):  # was CausalSelfAttention

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # NO CAUSAL MASK REGISTRATION - this is the key change

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # bidirectional self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # KEY CHANGE: is_causal=False for bidirectional attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # KEY CHANGE: NO causal masking applied
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
```

#### 4.2 Update All References to Use Bidirectional Attention
```python
# In Block class, update the attention reference
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)  # Changed from CausalSelfAttention
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### 4.3 Extend Vocabulary Size for Mask Token
```python
# Update GPTConfig to include mask token
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 + 1  # Add 1 for mask token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

#### 4.4 Test Bidirectional Functionality
```python
# Add simple test to verify bidirectional attention
def test_bidirectional_attention():
    """Test that model can attend to future tokens"""
    config = GPTConfig(block_size=10, vocab_size=100, n_layer=1, n_head=1, n_embd=32)
    model = GPT(config)
    model.eval()
    
    # Create test input with known pattern
    x = torch.randint(0, 100, (1, 10))  # Random tokens
    
    with torch.no_grad():
        logits, _ = model(x)
    
    print(f"Bidirectional model output shape: {logits.shape}")  # Should be (1, 10, vocab_size)
    print("Bidirectional attention test passed")

# Run test after implementing changes
test_bidirectional_attention()
```

### Deliverables:
- Converted model.py with bidirectional-only attention
- Extended vocabulary handling for mask token
- Working bidirectional training
- Attention pattern verification

### Validation:
- Model can attend to all sequence positions (no causal masking)
- Bidirectional attention improves masked token prediction accuracy
- Training converges better than causal version
- Attention patterns show full bidirectional connectivity

---

## Milestone 5: Advanced Masking Strategies (Sticky Masking)
**Duration**: 1-2 days  
**Goal**: Implement neighbor-aware sticky masking with detailed algorithm specification

### Detailed Tasks:

#### 5.1 Sticky Masking Algorithm Specification
**Sticky masking** creates clusters of masked tokens by making masking probability depend on neighbors:
- `p1`: Probability of masking if neighbors are NOT masked
- `p2`: Probability of masking if at least one neighbor IS masked  
- `k rounds`: Apply masking iteratively k times to build up clusters

**Implementation**:
```python
def apply_sticky_masking(tokens, rounds, mask_token_id, sticky_p1_p2_multiplier):
    \"\"\"
    Apply sticky masking algorithm
    
    Args:
        tokens: Original token sequence (batch_size, seq_len)
        rounds: Number of masking rounds
        mask_token_id: ID of mask token
        sticky_p1_p2_multiplier: Multiplier for p2 = p1 * multiplier
    
    Returns:
        masked_tokens: Tokens with sticky masking applied
        mask: Boolean mask showing which positions were masked
    \"\"\"
    batch_size, seq_len = tokens.shape
    device = tokens.device
    
    # Start with no masks
    masked_tokens = tokens.clone()
    
    for round_idx in range(rounds):
        # Dynamically sample sticky probabilities each round
        p1 = torch.rand(1).item() / (rounds * 2)  # Sample from (0, 1/(rounds*2))
        p2 = min(1.0, p1 * sticky_p1_p2_multiplier)  # p2 = p1 * multiplier, capped at 1
        
        # Current mask state
        current_mask = (masked_tokens == mask_token_id)
        
        # For each position, check if neighbors are masked
        neighbor_masked = torch.zeros_like(current_mask, dtype=torch.bool)
        
        # Check left neighbor
        neighbor_masked[:, 1:] |= current_mask[:, :-1]
        # Check right neighbor  
        neighbor_masked[:, :-1] |= current_mask[:, 1:]
        
        # Generate random values for masking decision
        rand_vals = torch.rand(batch_size, seq_len, device=device)
        
        # Apply p1 where neighbors not masked, p2 where neighbors masked
        mask_probs = torch.where(neighbor_masked, p2, p1)
        new_masks = rand_vals < mask_probs
        
        # Don't mask positions that are already masked
        new_masks = new_masks & ~current_mask
        
        # Apply new masks
        masked_tokens[new_masks] = mask_token_id
    
    # Final mask state
    final_mask = (masked_tokens == mask_token_id)
    return masked_tokens, final_mask
```

#### 5.2 Add Sticky Masking Configuration
```python
# Add to train.py config defaults
masking_strategy = 'independent'  # 'independent' or 'sticky'
sticky_p1 = 0.1                  # Probability if neighbors not masked  
sticky_p2 = 0.3                  # Probability if neighbors masked
sticky_rounds = 3                # Number of sticky masking rounds
```

#### 5.3 Update get_batch() Function with Gradual Transition
```python
def get_batch(split):
    # Existing data loading logic...
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    
    # Calculate sticky masking ratio based on current iteration
    global iter_num  # Access current training iteration
    
    if iter_num < sticky_transition_start:
        # Pure independent masking
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        # Pure sticky masking  
        sticky_ratio = 1.0
    else:
        # Gradual transition from independent to sticky
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress
    
    # Apply mixed masking strategy
    if sticky_ratio == 0.0:
        # Pure independent masking
        mask = torch.rand(x.shape) < masking_prob
        masked_x = x.clone()
        masked_x[mask] = mask_token_id
        
    elif sticky_ratio == 1.0:
        # Pure sticky masking
        masked_x, mask = apply_sticky_masking(x, sticky_p1, sticky_p2, sticky_rounds, mask_token_id)
        
    else:
        # Mixed strategy: some batches independent, some sticky
        batch_size = x.shape[0]
        num_sticky_batches = int(batch_size * sticky_ratio)
        
        masked_x = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        
        # Apply independent masking to first part of batch
        if num_sticky_batches < batch_size:
            indep_mask = torch.rand(x[:batch_size-num_sticky_batches].shape) < masking_prob
            masked_x[:batch_size-num_sticky_batches][indep_mask] = mask_token_id
            mask[:batch_size-num_sticky_batches] = indep_mask
        
        # Apply sticky masking to remaining part of batch
        if num_sticky_batches > 0:
            sticky_masked_x, sticky_mask = apply_sticky_masking(
                x[-num_sticky_batches:], sticky_rounds, mask_token_id, sticky_p1_p2_multiplier
            )
            masked_x[-num_sticky_batches:] = sticky_masked_x
            mask[-num_sticky_batches:] = sticky_mask
    
    y = x.clone()
    
    # Device transfer logic...
    return masked_x, y, mask
```

#### 5.4 Enhanced Masking Analysis with Transition Tracking
```python
def analyze_masking_patterns_with_transition(mask, iter_num):
    """Analyze masking patterns during independent->sticky transition"""
    mask_ratio = mask.float().mean().item()
    
    # Calculate current transition state
    if iter_num < sticky_transition_start:
        transition_state = "independent"
        sticky_ratio = 0.0
    elif iter_num >= sticky_transition_end:
        transition_state = "sticky"
        sticky_ratio = 1.0
    else:
        progress = (iter_num - sticky_transition_start) / (sticky_transition_end - sticky_transition_start)
        sticky_ratio = progress
        transition_state = f"transition ({sticky_ratio:.2f})"
    
    # Analyze clustering (more relevant during sticky phase)
    if sticky_ratio > 0.1:  # Only analyze clusters when some sticky masking present
        cluster_stats = analyze_clustering(mask)
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio,
            **cluster_stats
        }
    else:
        return {
            'mask_ratio': mask_ratio,
            'transition_state': transition_state,
            'sticky_ratio': sticky_ratio
        }

def analyze_clustering(mask):
    """Analyze clustering properties of mask patterns"""
    batch_size, seq_len = mask.shape
    cluster_sizes = []
    
    for batch_idx in range(batch_size):
        mask_seq = mask[batch_idx].cpu().numpy()
        
        # Find connected components (clusters)
        in_cluster = False
        current_cluster_size = 0
        
        for pos in range(seq_len):
            if mask_seq[pos]:  # Masked position
                if not in_cluster:
                    in_cluster = True
                    current_cluster_size = 1
                else:
                    current_cluster_size += 1
            else:  # Unmasked position
                if in_cluster:
                    cluster_sizes.append(current_cluster_size)
                    in_cluster = False
                    current_cluster_size = 0
        
        # Handle cluster at end of sequence
        if in_cluster:
            cluster_sizes.append(current_cluster_size)
    
    if cluster_sizes:
        avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        num_clusters = len(cluster_sizes)
        return {
            'avg_cluster_size': avg_cluster_size,
            'max_cluster_size': max_cluster_size,
            'num_clusters_per_batch': num_clusters / batch_size
        }
    else:
        return {
            'avg_cluster_size': 0,
            'max_cluster_size': 0,
            'num_clusters_per_batch': 0
        }

# Update logging call
if iter_num % (log_interval * 10) == 0:
    masking_stats = analyze_masking_patterns_with_transition(mask, iter_num)
    print(f"Masking: {masking_stats}")
```

### Deliverables:
- Complete sticky masking algorithm implementation
- Configurable masking strategies (independent vs sticky)
- Detailed masking pattern analysis
- Performance comparison between strategies

### Validation:
- Sticky masking creates larger clustered regions than independent
- p2 > p1 results in more clustering than p2 = p1
- Multiple rounds increase cluster sizes
- Training convergence comparison between strategies

---

## Milestone 6: Multi-Phase Training Integration
**Duration**: 3-4 days  
**Goal**: Implement sophisticated multi-phase training with detailed loss function specifications

### Detailed Tasks:

#### 6.1 Phase Management System
**Phase definitions**:
- **Phase 1 (N1 iterations)**: Identity task
  - Unmasked positions: predict same token (loss = 0 if correct)
  - Masked positions: uniform probability over all vocab except mask token
- **Phase 2 (N2 iterations)**: Gradual target increase
  - Gradually increase probability of correct token from uniform to 100%
  - Apply label smoothing 
- **Phase 3 (N3 iterations)**: Standard training  
  - Standard cross-entropy loss on masked positions
- **Phase 4 (N4 iterations)**: Entropy penalty (parallel to phases 2-3)
  - Add entropy penalty to encourage confidence in predictions
  - Adaptive multiplier decreases from max to min over N4 iterations

```python
# Add to train.py config
n1_iterations = 10000    # Phase 1: Identity task
n2_iterations = 50000    # Phase 2: Gradual target increase  
n3_iterations = 140000   # Phase 3: Standard training
n4_iterations = 100000   # Phase 4: Entropy penalty (overlaps 2-3)

entropy_multiplier_max = 5.0    # Maximum entropy penalty multiplier
entropy_multiplier_min = 1.0    # Minimum entropy penalty multiplier

def get_current_phase(iter_num):
    \"\"\"Determine current training phase\"\"\"
    if iter_num < n1_iterations:
        return 1
    elif iter_num < n1_iterations + n2_iterations:
        return 2  
    elif iter_num < n1_iterations + n2_iterations + n3_iterations:
        return 3
    else:
        return 3  # Continue phase 3 beyond n3_iterations
        
def in_entropy_penalty_phase(iter_num):
    \"\"\"Check if entropy penalty should be applied\"\"\"
    phase_2_3_start = n1_iterations
    phase_4_end = n1_iterations + max(n2_iterations + n3_iterations, n4_iterations)
    return phase_2_3_start <= iter_num < phase_4_end
```

### Deliverables:
- Multi-phase training integrated in train.py
- Configurable phase proportions (N1, N2, N3, N4)
- Advanced loss functions with entropy penalty
- Phase transition monitoring

### Validation:
- Identity task converges in phase 1
- Target probability increases smoothly in phase 2
- Entropy penalty effectively regularizes in phases 3-4
- Standard training works well in phase 3

---

## Milestone 7: Async Data Generation Pipeline
**Duration**: 1-2 days  
**Goal**: Implement background CPU-based data generation with performance monitoring

### Tasks:
1. **Implement async data pipeline**:
   - Background thread for data preparation
   - Queue-based producer-consumer pattern
   - CPU-based masking generation
   - GPU-based training consumption

2. **Performance monitoring**:
   - Track data generation speed
   - Monitor consumption vs generation rates
   - Memory usage tracking
   - Queue size optimization

3. **Validation set improvements**:
   - Pre-generate fixed validation set at start
   - Store in memory for consistent evaluation
   - Support all masking strategies for validation

### Deliverables:
- Async data generation system
- Performance monitoring and reporting
- Fixed validation dataset
- Speed comparison metrics

### Validation:
- Data generation keeps up with training
- No memory leaks in async pipeline
- Consistent validation across runs
- Performance metrics accurately reported

---

## Milestone 8: Polish and Optimization
**Duration**: 1-2 days  
**Goal**: Final optimization and documentation

### Tasks:
1. **Performance optimization**:
   - Profile and optimize critical paths
   - Memory usage improvements
   - Training speed optimizations

2. **Configuration improvements**:
   - Validate configuration parameters
   - Better error messages and debugging
   - Example configurations for different use cases

3. **Documentation and testing**:
   - Usage examples and tutorials
   - Basic unit tests for new functionality
   - Performance benchmarks

### Deliverables:
- Optimized performance
- Robust configuration system
- Complete documentation
- Working examples

### Validation:
- Performance within acceptable bounds
- System handles errors gracefully
- Documentation enables easy usage

## Configuration Examples

### Diffusion Training Configuration
```python
# Add to train.py config defaults  
dataset = 'shakespeare_char'     # Use character-level dataset
guaranteed_unmasked = 0.15       # Guaranteed fraction of tokens to keep unmasked

# Masking configuration - gradual transition from independent to sticky
sticky_transition_start = 20000  # When to start introducing sticky masking
sticky_transition_end = 100000   # When to reach full sticky masking
sticky_rounds = 3                # Number of sticky masking rounds
sticky_p1_p2_multiplier = 3.0    # Multiplier for sticky_p2 = sticky_p1 * multiplier

# Vocab size and mask token determined from meta.pkl
# mask_token_id will be set to meta['vocab_size'] if meta.pkl exists

# Training phases (iterations)
n1_iterations = 10000           # Phase 1: Identity task
n2_iterations = 50000           # Phase 2: Gradual target increase
n3_iterations = 140000          # Phase 3: Standard training
n4_iterations = 100000          # Phase 4: Entropy penalty fade

# Loss configuration  
entropy_multiplier_max = 5.0    # Maximum entropy penalty multiplier
entropy_multiplier_min = 1.0    # Minimum entropy penalty multiplier

# Async data generation
async_data_generation = True    # Enable background data generation
data_queue_size = 100           # Size of data queue
```

### Inference Configuration
```python
# Add to sample.py config defaults
demasking_algorithm = 'linear'  # 'linear', 'exponential'  
demasking_iterations = 10       # Number of demasking/remasking iterations
```

## Implementation Strategy

### Development Principles
1. **One feature at a time**: Complete each milestone fully before moving to next
2. **Test as you go**: Verify functionality after each change
3. **Diffusion-only focus**: Convert from autoregressive to diffusion-based approach
4. **Incremental complexity**: Start simple, add sophistication gradually
5. **Performance monitoring**: Track speed/memory impact of each addition

### File Modification Strategy
- **Milestone 1**: Only modify train.py (convert to masking + diffusion training)
- **Milestone 2**: Only modify model.py (convert to bidirectional attention)
- **Milestone 3**: Continue with train.py (add sticky masking strategies)  
- **Milestone 4**: Continue with train.py (add multi-phase training)
- **Milestone 5**: Continue with train.py (add async data generation)
- **Milestone 6**: Continue with train.py (add comprehensive logging)
- **Milestone 7**: Only modify sample.py (convert to demasking inference)
- **Milestone 8**: Polish existing files, add docs/tests

### Success Metrics
- [ ] Bidirectional attention: Model can attend to all sequence positions
- [ ] Multi-phase training: Identity → gradual → standard phases work smoothly
- [x] **Masking strategies: Basic independent masking implemented and working**
- [x] **Demasking inference: Multiple algorithms generate coherent text**
- [x] **Performance: Training/inference speed reasonable for experimentation**

## Risk Mitigation
- **Start small**: Begin with simplest versions of each feature
- **Regular testing**: Test diffusion functionality after each change
- **Rollback plan**: Keep git history clean for easy reverting
- **Performance monitoring**: Profile memory/speed impact continuously

## Next Steps
1. ~~**Begin Milestone 1**: Convert train.py to masking + diffusion training~~ ✅ **COMPLETED**
2. ~~**Test immediately**: Verify masking and basic training works~~ ✅ **COMPLETED**
3. ~~**Move to Milestone 2**: Get demasking inference working for quality verification~~ ✅ **COMPLETED**
4. **Add Milestone 3**: Basic logging to track performance
5. **Move to Milestone 4**: Convert to bidirectional attention
6. **Commit frequently**: Small, focused commits for each feature
7. **Document changes**: Add comments explaining new diffusion features

## Current Status Summary
**Milestones Completed**: 2/8
- ✅ **Milestone 1**: Basic masking data generation with identity task training
- ✅ **Milestone 2**: Pure diffusion inference with iterative demasking/remasking

**Key Achievements**:
- Successfully converted from autoregressive to masked language modeling
- Implemented working diffusion-based text generation
- Model generates coherent Shakespeare-style text using iterative demasking
- Enhanced logging shows detailed generation process
- Proper vocabulary handling with mask tokens

**Ready for Next Phase**: The foundation is solid for implementing bidirectional attention and advanced masking strategies.

This incremental approach transforms nanoGPT into a diffusion-based LLM through small, focused changes while maintaining the clean, readable structure of the original codebase.