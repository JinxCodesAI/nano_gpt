# Implementation Plan

- [x] 1. Update GPTConfig class to support mode parameter
  - Add mode parameter to GPTConfig dataclass with default 'generator'
  - Add validation to ensure mode is either 'generator' or 'reward'
  - **Commit after completion:** "Add mode parameter to GPTConfig for dual-mode support"
  - _Requirements: 1.3, 3.1, 3.2, 3.3_

- [x] 2. Modify GPT class constructor for dual-mode support
  - Add mode validation assertion in __init__ method
  - Restructure transformer component creation to be mode-agnostic
  - Implement conditional head creation based on mode parameter
  - **Commit after completion:** "Implement dual-mode GPT constructor with conditional heads"
  - _Requirements: 1.1, 1.2, 1.3, 5.2, 5.3_

- [x] 3. Implement reward head architecture
  - Create reward head as nn.Sequential with Linear-ReLU-Linear-Softmax layers
  - Use 256 hidden units and 2 output units as specified in design
  - Ensure proper weight initialization through existing _init_weights method
  - **Commit after completion:** "Add reward head architecture with MLP and softmax"
  - _Requirements: 2.2, 2.3, 5.1_

- [x] 4. Update forward method for dual-mode operation
  - Modify forward method to branch based on mode after shared trunk processing
  - Implement generator mode forward pass (existing logic)
  - Implement reward mode forward pass with sequence pooling
  - **Commit after completion:** "Implement dual-mode forward pass with sequence pooling for reward mode"
  - _Requirements: 4.1, 4.2, 2.1_

- [x] 5. Implement reward mode loss calculation
  - Add MSE loss calculation for reward mode when targets are provided
  - Return None for loss when no targets provided in reward mode
  - Maintain existing cross-entropy loss for generator mode
  - **Commit after completion:** "Add MSE loss calculation for reward mode training"
  - _Requirements: 4.3, 4.4_

- [x] 6. Update weight tying logic for mode compatibility
  - Apply weight tying only for generator mode between wte and lm_head
  - Skip weight tying for reward mode to avoid conflicts with reward head
  - **Commit after completion:** "Update weight tying to be mode-specific for compatibility"
  - _Requirements: 5.2, 5.3_

- [x] 7. Create basic functionality tests

  - Write test for GPTConfig mode parameter validation
  - Write test for correct head creation in both modes
  - Write test for forward pass output shapes in both modes
  - Test basic loss calculation for both modes (CPU-compatible tests)
  - **Commit after completion:** "Add comprehensive tests for dual-mode GPT functionality"
  - _Requirements: All requirements validation_

- [ ] 8. Test backward compatibility and integration



  - Verify existing generator functionality remains unchanged
  - Test model parameter counting works correctly for both modes
  - Verify checkpoint loading compatibility with existing models
  - Test that default mode maintains backward compatibility
  - **Commit after completion:** "Verify backward compatibility and integration with existing codebase"
  - _Requirements: 3.3, interface compatibility_