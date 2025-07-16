# Requirements Document

## Introduction

This feature implements a dual-mode GPT architecture that can operate as either a language model generator or a reward model discriminator. The core innovation is sharing the majority of model weights (the "trunk") between both modes while having task-specific "heads" for their respective outputs. This enables efficient training of reward models for GRPO (Generative Reward Policy Optimization) without maintaining two separate large models.

## Requirements

### Requirement 1

**User Story:** As a researcher implementing GRPO training, I want a GPT model that can switch between generator and reward modes, so that I can efficiently train both components without duplicating the large transformer weights.

#### Acceptance Criteria

1. WHEN the model is configured with mode='generator' THEN the system SHALL create a standard language modeling head for next-token prediction
2. WHEN the model is configured with mode='reward' THEN the system SHALL create a reward head that outputs probability distributions for natural vs synthetic text classification
3. WHEN either mode is selected THEN the system SHALL share the same transformer trunk (embeddings, blocks, layer norm) between both modes

### Requirement 2

**User Story:** As a developer training reward models, I want the reward head to process sequence-level information, so that it can classify entire sequences rather than individual tokens.

#### Acceptance Criteria

1. WHEN the reward model processes a sequence THEN the system SHALL pool the final token's hidden state as a sequence representation
2. WHEN the pooled representation is processed THEN the system SHALL pass it through an MLP with ReLU activation
3. WHEN the MLP output is generated THEN the system SHALL apply softmax to produce a 2-element probability distribution [P(natural), P(synthetic)]

### Requirement 3

**User Story:** As a machine learning engineer, I want the model configuration to specify the operating mode, so that I can instantiate the appropriate model variant for my training pipeline.

#### Acceptance Criteria

1. WHEN GPTConfig is instantiated THEN the system SHALL accept a 'mode' parameter with values 'generator' or 'reward'
2. WHEN an invalid mode is provided THEN the system SHALL raise an assertion error with a clear message
3. WHEN no mode is specified THEN the system SHALL default to 'generator' mode for backward compatibility

### Requirement 4

**User Story:** As a researcher, I want the forward pass to handle both modes appropriately, so that I can use the same model class for different training objectives.

#### Acceptance Criteria

1. WHEN the model is in 'generator' mode THEN the forward pass SHALL compute standard language modeling logits and cross-entropy loss
2. WHEN the model is in 'reward' mode THEN the forward pass SHALL compute probability distributions and optionally MSE loss against targets
3. WHEN targets are provided in reward mode THEN the system SHALL calculate Mean Squared Error loss between predicted and target probabilities
4. WHEN no targets are provided THEN the system SHALL return None for the loss component

### Requirement 5

**User Story:** As a developer, I want proper weight initialization for the reward head, so that training starts from a reasonable state.

#### Acceptance Criteria

1. WHEN the reward head is created THEN the system SHALL initialize the MLP weights using the existing _init_weights method
2. WHEN weight tying is applied THEN the system SHALL only tie weights for the generator mode's lm_head
3. WHEN the reward mode is used THEN the system SHALL NOT apply weight tying to the reward head