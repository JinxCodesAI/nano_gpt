# Requirements Document

## Introduction

This feature enhances the reward model data preparation system to support configurable tokenization methods (both BPE and character-level) and provides an alternative path to reuse existing train.bin and val.bin files from base models instead of creating new splits from raw text. This ensures consistency with existing model training data and supports different tokenization approaches used across the codebase.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to configure the tokenization method for reward data preparation, so that I can work with both BPE-tokenized and character-level tokenized models.

#### Acceptance Criteria

1. WHEN preparing reward data THEN the system SHALL support both tiktoken BPE encoding and character-level encoding
2. WHEN using character-level encoding THEN the system SHALL load vocabulary mappings from meta.pkl files
3. WHEN using BPE encoding THEN the system SHALL use tiktoken GPT-2 encoding as currently implemented
4. IF a meta.pkl file exists in the data directory THEN the system SHALL automatically detect character-level tokenization
5. WHEN tokenization method is detected THEN the system SHALL log which method is being used

### Requirement 2

**User Story:** As a developer, I want to reuse existing train.bin and val.bin files from base models, so that I maintain perfect data consistency without recreating splits.

#### Acceptance Criteria

1. WHEN specifying existing binary files THEN the system SHALL load tokens directly from train.bin and val.bin
2. WHEN using existing binary files THEN the system SHALL skip raw text processing and splitting
3. WHEN loading binary files THEN the system SHALL validate file format and token ranges
4. IF binary files are incompatible with the model THEN the system SHALL provide clear error messages
5. WHEN using binary files THEN the system SHALL respect the original tokenization method used to create them

### Requirement 3

**User Story:** As a developer, I want configurable input modes for data preparation, so that I can choose between raw text processing and binary file reuse based on my workflow needs.

#### Acceptance Criteria

1. WHEN running the preparation script THEN the system SHALL accept either raw text files or binary file paths
2. WHEN both raw text and binary files are specified THEN the system SHALL prioritize binary files and warn about the conflict
3. WHEN using binary file mode THEN the system SHALL require both train.bin and val.bin to be specified
4. IF only one binary file is provided THEN the system SHALL raise an error requesting both files
5. WHEN switching between modes THEN the system SHALL validate that all required parameters are provided

### Requirement 4

**User Story:** As a developer, I want automatic tokenization detection and validation, so that I can avoid configuration errors and ensure data compatibility.

#### Acceptance Criteria

1. WHEN loading binary files THEN the system SHALL detect the tokenization method from associated metadata
2. WHEN metadata is missing THEN the system SHALL attempt to infer tokenization from file structure and vocab size
3. WHEN tokenization methods mismatch THEN the system SHALL provide clear error messages with resolution steps
4. IF vocab sizes are incompatible THEN the system SHALL prevent processing and suggest corrections
5. WHEN validation passes THEN the system SHALL log detected configuration for user confirmation

### Requirement 5

**User Story:** As a developer, I want backward compatibility with existing workflows, so that current scripts and processes continue to work without modification.

#### Acceptance Criteria

1. WHEN using existing command-line parameters THEN the system SHALL function exactly as before
2. WHEN no new parameters are specified THEN the system SHALL default to current BPE behavior
3. WHEN existing data paths are used THEN the system SHALL process them using the original logic
4. IF new features are not used THEN the system SHALL have identical output to the current implementation
5. WHEN upgrading THEN existing reward datasets SHALL remain compatible and loadable