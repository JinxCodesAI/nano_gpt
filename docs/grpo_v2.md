# GRPO Training Implementation Plan v2
## Maximizing Code Reuse from Existing Codebase

---

## **Phase 0: Conceptual Overview & Goal**

**What:** Implement Group-Relative Policy Optimization (GRPO) training for single-step fill-in tasks by extending the existing codebase with a new `grpo/` folder.

**Why:** Train the model to produce better single-step completions of masked inputs by using reinforcement learning with a frozen judge model as the reward signal.

**Key Approach:**
1. **Reuse existing dataset infrastructure** - fetch masked inputs from `DatasetConsumer`
2. **Reuse existing sampling infrastructure** - use `predict_and_sample_tokens` from `sample_utils.py`
3. **Reuse existing judge scoring** - use `calculate_judge_scores` from `sample_utils.py`
4. **Compute log-probabilities from existing logits** - no new model methods needed
5. **Implement KL divergence penalty** using a frozen reference policy

**Core Training Loop:**
1. **Fetch masked input** from dataset (via `DatasetConsumer`)
2. **Generate k completions** for each masked input using the generator
3. **Score completions** with frozen judge model
4. **Calculate advantages** (reward - baseline)
5. **Compute GRPO loss** with KL penalty to reference policy
6. **Update generator** weights

---

## **Phase 1: Architecture & File Structure**

Create a new `grpo/` folder with the following structure:

```
grpo/
├── train_grpo.py           # Main training script (orchestrator)
├── grpo_trainer.py         # GRPO-specific Trainer class
├── grpo_training_step.py   # GRPO training step logic
├── grpo_config.py          # GRPO-specific configuration
└── README.md               # Documentation
```

**Why this structure:**
- Keeps GRPO code isolated from existing codebase
- Follows existing patterns (`core/trainer.py`, `core/training_step.py`)
- Easy to maintain and extend

---

## **Phase 2: GRPO Training Step (`grpo/grpo_training_step.py`)**

This is the core GRPO logic. It replaces the supervised training step.

### **Key Design Decisions:**

1. **Reuse `predict_and_sample_tokens`** from `sample_utils.py`:
   - Already returns both `prediction_tokens` AND `logits`
   - No need for new model methods
   - Efficient batched sampling

2. **Compute log-probabilities from logits**:
   ```python
   # From existing logits (B, T, vocab_size)
   log_probs = F.log_softmax(logits, dim=-1)
   # Gather log-probs for sampled tokens
   token_log_probs = torch.gather(log_probs, -1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
   # Sum over sequence (only masked positions)
   sequence_log_probs = (token_log_probs * mask).sum(dim=1)
   ```

3. **KL Divergence to Reference Policy**:
   - Load a frozen copy of the initial generator as `reference_model`
   - Compute KL divergence: `KL = (log_prob_current - log_prob_reference)`
   - Add penalty: `loss = -advantages * log_probs + beta * KL`

### **Pseudocode:**

```python
class GRPOTrainingStep:
    def __init__(self, generator_model, reference_model, judge_model, 
                 optimizer, scaler, config):
        self.generator = generator_model
        self.reference = reference_model  # Frozen copy of initial generator
        self.judge = judge_model          # Frozen judge
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        
        # Freeze reference and judge
        for param in self.reference.parameters():
            param.requires_grad = False
        for param in self.judge.parameters():
            param.requires_grad = False
    
    def execute_step(self, batch, consumer, device):
        """
        Execute one GRPO training step.
        
        Args:
            batch: Dict with 'x' (masked input) and 'y' (targets)
            consumer: DatasetConsumer for fetching next batch
            device: Device string
        
        Returns:
            loss, next_batch
        """
        X = batch['x']  # (B, T) - masked input
        Y = batch['y']  # (B, T) - targets (ignore_index where not masked)
        
        # Identify masked positions
        mask = (X == self.config['mask_token_id'])  # (B, T)
        
        # STEP 1: Generate k completions per input
        # Repeat each input k times: (B*k, T)
        X_repeated = X.repeat_interleave(self.config['group_size'], dim=0)
        mask_repeated = mask.repeat_interleave(self.config['group_size'], dim=0)
        
        # Sample completions using existing infrastructure
        with torch.no_grad():
            completions, logits_gen = predict_and_sample_tokens(
                model=self.generator,
                tokens=X_repeated,
                mask_token_id=self.config['mask_token_id'],
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                vocab_size=self.config['vocab_size'],
                device=device,
                return_logits=True,
                pad_token_id=self.config['pad_token_id'],
                base_vocab_size=self.config['base_vocab_size']
            )
        
        # STEP 2: Score completions with judge
        with torch.no_grad():
            rewards = calculate_judge_scores(
                judge_model=self.judge,
                tokens=completions,
                device=device,
                ctx=self.ctx
            )  # (B*k,)
        
        # STEP 3: Calculate advantages (group-relative)
        # Reshape to (B, k) to compute per-group baseline
        rewards_grouped = rewards.view(-1, self.config['group_size'])
        baseline = rewards_grouped.mean(dim=1, keepdim=True)  # (B, 1)
        advantages = rewards_grouped - baseline  # (B, k)
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.view(-1)  # Flatten back to (B*k,)
        
        # STEP 4: Compute log-probabilities (with gradients)
        # Forward pass through generator to get logits
        logits_current, _ = self.generator(X_repeated, targets=None)
        log_probs_current = F.log_softmax(logits_current, dim=-1)
        
        # Gather log-probs for sampled tokens
        token_log_probs = torch.gather(
            log_probs_current, -1, 
            completions.unsqueeze(-1)
        ).squeeze(-1)  # (B*k, T)
        
        # Sum only over masked positions
        sequence_log_probs = (token_log_probs * mask_repeated.float()).sum(dim=1)
        
        # STEP 5: Compute KL divergence to reference policy
        with torch.no_grad():
            logits_ref, _ = self.reference(X_repeated, targets=None)
            log_probs_ref = F.log_softmax(logits_ref, dim=-1)
            token_log_probs_ref = torch.gather(
                log_probs_ref, -1,
                completions.unsqueeze(-1)
            ).squeeze(-1)
            sequence_log_probs_ref = (token_log_probs_ref * mask_repeated.float()).sum(dim=1)
        
        kl_divergence = sequence_log_probs - sequence_log_probs_ref
        
        # STEP 6: Compute GRPO loss
        # Policy gradient term
        pg_loss = -(sequence_log_probs * advantages.detach()).mean()
        
        # KL penalty term
        kl_penalty = self.config['kl_beta'] * kl_divergence.mean()
        
        # Total loss
        loss = pg_loss + kl_penalty
        
        # STEP 7: Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.config['grad_clip'] > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.config['grad_clip']
            )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Fetch next batch
        next_batch = consumer.get_batch('train', device)
        
        # Return metrics
        metrics = {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        return loss, next_batch, metrics
```

---

## **Phase 3: GRPO Trainer (`grpo/grpo_trainer.py`)**

Extends the existing `Trainer` class with GRPO-specific logic.

**Key differences from standard training:**
1. No evaluation loop (or simplified evaluation)
2. Periodic sampling to monitor quality
3. Logging of GRPO-specific metrics (rewards, advantages, KL)

```python
class GRPOTrainer:
    def __init__(self, generator_model, reference_model, judge_model,
                 optimizer, scheduler, training_step, consumer,
                 checkpoint_manager, logger, config):
        self.generator = generator_model
        self.reference = reference_model
        self.judge = judge_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_step = training_step
        self.consumer = consumer
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.config = config
        
    def train(self):
        """Main GRPO training loop."""
        batch = self.consumer.get_batch('train', self.config['device'])
        iter_num = 0
        
        while iter_num < self.config['max_iters']:
            # Set learning rate
            lr = self.scheduler.get_lr(iter_num)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            
            # Execute GRPO step
            loss, batch, metrics = self.training_step.execute_step(
                batch, self.consumer, self.config['device']
            )
            
            # Logging
            if iter_num % self.config['log_interval'] == 0:
                self.logger.log_step({
                    'iter': iter_num,
                    'lr': lr,
                    **metrics
                })
            
            # Checkpointing
            if iter_num % self.config['save_interval'] == 0:
                self.checkpoint_manager.save()
            
            
            iter_num += 1
```

---

## **Phase 4: Main Training Script (`grpo/train_grpo.py`)**

Orchestrates the entire GRPO training process.

**Key components:**
1. Load generator, reference (copy of generator), and judge models
2. Initialize optimizer (only for generator)
3. Initialize `DatasetConsumer` (reuse existing)
4. Create `GRPOTrainingStep` and `GRPOTrainer`
5. Start training

```python
# Load generator model (to be trained)
generator_model, _ = load_model_from_checkpoint(
    generator_checkpoint_path, device, compile=False
)
generator_model.train()

# Load reference model (frozen copy of initial generator)
reference_model, _ = load_model_from_checkpoint(
    generator_checkpoint_path, device, compile=False
)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False

# Load judge model (frozen)
judge_model, _ = load_model_from_checkpoint(
    judge_checkpoint_path, device, compile=False
)
judge_model.eval()
for param in judge_model.parameters():
    param.requires_grad = False

# Initialize optimizer (only for generator)
optimizer = generator_model.configure_optimizers(...)

# Initialize data consumer (reuse existing)
consumer = DatasetConsumer(...)

# Create GRPO training step
grpo_step = GRPOTrainingStep(
    generator_model=generator_model,
    reference_model=reference_model,
    judge_model=judge_model,
    optimizer=optimizer,
    scaler=scaler,
    config=grpo_config
)

# Create GRPO trainer
trainer = GRPOTrainer(...)

# Start training
trainer.train()
```

---

## **Phase 5: Configuration (`grpo/grpo_config.py`)**

GRPO-specific hyperparameters:

```python
# GRPO hyperparameters
group_size = 8              # Number of completions per input (k)
kl_beta = 0.1               # KL divergence penalty coefficient
temperature = 0.8           # Sampling temperature
top_p = 0.95                # Nucleus sampling

# Model checkpoints
generator_checkpoint = 'out-char-diffusion/checkpoint.pt'
judge_checkpoint = 'out-char-diffusion/judge.pt'

# Training
learning_rate = 1e-5        # Lower LR for fine-tuning
max_iters = 10000
grad_clip = 1.0
log_interval = 10
save_interval = 1000

# Dataset (reuse existing)
dataset = 'char_diffusion'
batch_size = 16             # Effective batch size is batch_size * group_size
```

---

## **Phase 6: Proposed Modifications to Existing Codebase**

To dramatically increase code reuse, consider these **optional** modifications:

### **1. Expose `ctx` in `sample_utils.py` functions**

**Current:** `calculate_judge_scores` doesn't accept `ctx` parameter  
**Proposed:** Add `ctx` parameter for consistency with training

```python
def calculate_judge_scores(judge_model, tokens, device, ctx=None):
    if ctx is None:
        ctx = nullcontext()
    with ctx:
        # ... existing logic
```

**Benefit:** Consistent autocast behavior between training and sampling

### **2. Make `predict_and_sample_tokens` return logits by default**

**Current:** `return_logits` is optional  
**Proposed:** Always return logits (or make it default True)

**Benefit:** GRPO always needs logits, so this avoids redundant forward passes

### **3. Add `repeat_interleave` utility to `sample_utils.py`**

**Proposed:** Helper function for repeating inputs k times

```python
def repeat_inputs_for_group_sampling(inputs, group_size):
    """Repeat each input k times for group sampling."""
    return {k: v.repeat_interleave(group_size, dim=0) 
            for k, v in inputs.items()}
```

**Benefit:** Cleaner GRPO code

---

## **Phase 7: Implementation Checklist**

- [ ] Create `grpo/` folder structure
- [ ] Implement `grpo_training_step.py` with full GRPO logic
- [ ] Implement `grpo_trainer.py` extending existing patterns
- [ ] Implement `train_grpo.py` main script
- [ ] Create `grpo_config.py` with hyperparameters
- [ ] Test with small batch size and few iterations
- [ ] Verify log-probability computation is correct
- [ ] Verify KL divergence computation is correct
- [ ] Verify advantages are normalized properly
- [ ] Monitor training: loss should decrease, rewards should increase
- [ ] Sample periodically to verify quality improvements
- [ ] Add comprehensive logging (WandB integration)
- [ ] Document usage in `grpo/README.md`

---

## **Key Advantages of This Approach**

1. **Maximum code reuse**: Uses existing `DatasetConsumer`, `predict_and_sample_tokens`, `calculate_judge_scores`
2. **No model modifications**: Computes log-probs from existing logits
3. **Clean separation**: All GRPO code in `grpo/` folder
4. **Follows existing patterns**: Mirrors `core/trainer.py` and `core/training_step.py`
5. **Easy to debug**: Each component is isolated and testable
6. **Minimal dependencies**: Only adds GRPO-specific logic

---

## **Expected Training Behavior**

- **Initial iterations**: High variance in rewards, large KL divergence
- **After warmup**: Rewards increase, KL stabilizes around target
- **Convergence**: Model produces higher-quality completions as judged by the frozen judge
- **Monitoring**: Sample generations periodically to verify quality improvements

---

## **Next Steps**

1. Review this plan and confirm approach
2. Implement `grpo_training_step.py` first (core logic)
3. Test in isolation with dummy data
4. Implement `grpo_trainer.py` and `train_grpo.py`
5. Run small-scale experiment
6. Scale up and monitor

