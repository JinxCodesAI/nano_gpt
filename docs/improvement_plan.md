# Training Pipeline Core Classes Extraction - Refactoring Plan

## Overview

This document outlines a step-by-step refactoring plan to extract core classes for the training pipeline from the monolithic `train.py` script (~370 lines). The goal is to make the code more modular, testable, and maintainable while preserving all existing functionality and configuration compatibility.

**Current State Analysis:**
- `train.py`: Large monolithic script handling initialization, DDP setup, data loading, training loop, evaluation, logging, and checkpointing
- `model.py`: Well-designed GPT model with good separation of concerns 
- `loss_modifiers/`: Already well-modularized pipeline system for loss modifications
- Configuration: Uses global variables with `configurator.py` execution

**Target**: Extract 4-5 core classes following single responsibility principle without changing any functionality.

---

## Milestone 1: Extract Learning Rate Scheduler (Easiest - Low Risk)

**Rationale**: The `get_lr()` function is pure and stateless, making it the safest extraction target.

### Step 1.1: Create LR Scheduler Classes
- **File**: `core/scheduler.py` (new)
- **Classes**: 
  - `LRScheduler` (abstract base class)
  - `CosineLRScheduler` (implements current `get_lr` logic)
- **Testing**: Verify identical LR values for various iteration numbers

### Step 1.2: Update train.py to Use Scheduler
- Replace direct `get_lr(iter_num)` calls with `scheduler.get_lr(iter_num)`
- Zero functional changes, just abstraction

**Validation**: 
- Training curves should be identical
- LR logging values should match exactly
- All existing configurations should work unchanged

---

## Milestone 2: Extract Evaluator Class (Low Risk)

**Rationale**: The `estimate_loss()` function has clear boundaries and minimal state dependencies.

### Step 2.1: Create Evaluator Class
- **File**: `core/evaluator.py` (new)
- **Class**: `Evaluator`
- **Methods**:
  - `__init__(model, consumer, loss_modifier_pipeline, eval_iters, ctx)`
  - `evaluate(split='val') -> float`
- **Features**:
  - Encapsulates current `estimate_loss()` logic exactly
  - Handles loss modifier temporary disabling
  - Supports both 'train' and 'val' splits

### Step 2.2: Update train.py Integration
- Replace `estimate_loss()` calls with `evaluator.evaluate('train')` and `evaluator.evaluate('val')`
- Maintain all existing evaluation behavior

**Validation**:
- Evaluation loss values should be identical
- Loss modifier metrics collection should work unchanged
- Evaluation timing should remain the same

---

## Milestone 3: Extract Logger Class (Medium Risk)

**Rationale**: Logging logic is scattered but has clear input/output interfaces.

### Step 3.1: Create Logger Abstraction
- **File**: `core/logger.py` (new)
- **Classes**:
  - `Logger` (abstract base class)
  - `ConsoleLogger` (handles print statements)
  - `WandBLogger` (handles wandb.log calls)
  - `CompositeLogger` (combines multiple loggers)

### Step 3.2: Implement Logger Classes
- **ConsoleLogger**: Handles all `print()` calls from train.py
- **WandBLogger**: Handles wandb initialization and logging with identical logic
- **CompositeLogger**: Orchestrates multiple loggers

### Step 3.3: Update train.py Integration  
- Replace direct print/wandb calls with logger methods
- Maintain exact same logging output and timing

**Validation**:
- Console output should be identical
- WandB logs should match exactly (same keys, values, timing)
- All logging configurations should work unchanged

---

## Milestone 4: Extract Training Step Handler (Medium-High Risk)

**Rationale**: The training loop has complex state but clear forward/backward logic boundaries.

### Step 4.1: Create TrainingStep Class
- **File**: `core/training_step.py` (new)
- **Class**: `TrainingStep`
- **Responsibilities**:
  - Gradient accumulation loop
  - Forward/backward pass coordination
  - Gradient clipping
  - Optimizer stepping
  - DDP synchronization

### Step 4.2: Implement Training Step Logic
- **Methods**:
  - `__init__(model, optimizer, scaler, gradient_accumulation_steps, grad_clip, ddp, ctx)`
  - `execute_step(X, Y, loss_modifier_pipeline, consumer, device) -> float`
- **Features**:
  - Encapsulates micro-step loop exactly as in current code
  - Handles DDP grad sync logic
  - Returns loss value for logging

### Step 4.3: Update train.py Integration
- Replace training loop inner logic with `training_step.execute_step()`
- Maintain all existing training behavior and performance

**Validation**:
- Training loss trajectories should be identical
- Gradient norms should match
- MFU calculations should be unchanged  
- DDP behavior should be identical

---

## Milestone 5: Extract Main Trainer Orchestrator (Highest Risk)

**Rationale**: Final extraction to create clean main training controller.

### Step 5.1: Create Trainer Class
- **File**: `core/trainer.py` (new)
- **Class**: `Trainer`
- **Responsibilities**:
  - Coordinate all other classes
  - Main training loop control
  - Checkpoint management integration
  - Termination condition checking

### Step 5.2: Implement Trainer Orchestration
- **Methods**:
  - `__init__(model, optimizer, scheduler, evaluator, logger, training_step, checkpoint_manager, config)`
  - `train() -> None`
  - `should_evaluate(iter_num) -> bool`
  - `should_log(iter_num) -> bool`
  - `should_terminate(iter_num) -> bool`

### Step 5.3: Minimize train.py to Orchestration
- **New train.py structure** (~50-80 lines):
  ```python
  # Configuration and initialization (existing logic)
  config = load_config()
  
  # Component creation (existing factory logic)  
  model = create_model(config)
  optimizer = create_optimizer(model, config)
  # ... other components
  
  # Core class instantiation
  scheduler = CosineLRScheduler(config)
  evaluator = Evaluator(model, consumer, loss_modifier_pipeline, eval_iters, ctx)
  logger = create_logger(config)
  training_step = TrainingStep(model, optimizer, scaler, config, ddp, ctx)
  
  trainer = Trainer(
      model=model,
      optimizer=optimizer, 
      scheduler=scheduler,
      evaluator=evaluator,
      logger=logger,
      training_step=training_step,
      checkpoint_manager=checkpoint_manager,
      config=config
  )
  
  # Execute training
  trainer.train()
  ```

**Validation**:
- Complete training runs should produce identical results
- All checkpointing behavior should be preserved
- All logging and evaluation should match exactly

---

## Implementation Guidelines

### File Structure
```
core/
├── __init__.py
├── scheduler.py      # Milestone 1
├── evaluator.py      # Milestone 2  
├── logger.py         # Milestone 3
├── training_step.py  # Milestone 4
└── trainer.py        # Milestone 5
```

### Testing Strategy
1. **Unit Tests**: Each extracted class should have comprehensive unit tests
2. **Integration Tests**: After each milestone, run short training to verify functionality
3. **Regression Tests**: Compare training curves, loss values, and checkpoints
4. **Configuration Tests**: Verify all existing config combinations work

### Backwards Compatibility
- All existing command-line arguments must work unchanged
- All existing configuration files must work unchanged  
- All existing checkpoint formats must be loadable
- All existing metrics and logging must be preserved

### Error Handling
- Each class should handle errors gracefully
- Error messages should be clear and actionable
- No new failure modes should be introduced

### Performance Considerations
- No performance regression in training speed
- Memory usage should remain the same
- MFU calculations should be unchanged
- GPU utilization should be identical

---

## Success Criteria

After completing all milestones:

1. **Functionality**: Complete training run produces identical results to original
2. **Modularity**: Each class has single, clear responsibility
3. **Testability**: Each class can be unit tested in isolation
4. **Readability**: Code intent is clearer with well-named classes and methods
5. **Maintainability**: Adding new features (schedulers, loggers, etc.) is straightforward
6. **Configuration**: All existing configurations work without modification
7. **Performance**: No regression in training speed or GPU utilization

---

## Risk Mitigation

### High-Risk Areas
1. **DDP Logic**: Gradient synchronization timing is critical
2. **Loss Modifier Integration**: Pipeline state management must be preserved
3. **Checkpoint State**: All state must be properly restored
4. **Autocast Context**: Mixed precision timing is sensitive

### Mitigation Strategies
1. **Incremental Testing**: Test after each milestone
2. **Comparison Scripts**: Automated comparison of training curves
3. **Checkpoint Validation**: Verify checkpoint loading/saving works identically
4. **Rollback Plan**: Each milestone can be reverted independently

---

## Timeline Estimate

- **Milestone 1**: 1-2 days (low risk, straightforward)
- **Milestone 2**: 1-2 days (low risk, clear boundaries)  
- **Milestone 3**: 2-3 days (medium risk, multiple loggers)
- **Milestone 4**: 3-4 days (high risk, complex training logic)
- **Milestone 5**: 2-3 days (orchestration and integration)

**Total**: 9-14 days for complete refactoring with thorough testing

This plan prioritizes safety and functionality preservation while achieving the modularity goals outlined in the improvement proposal.