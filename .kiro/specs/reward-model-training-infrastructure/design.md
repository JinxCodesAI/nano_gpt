# Design Document

## Overview

The reward model training infrastructure provides a complete pipeline for generating, loading, and testing mixed natural/synthetic datasets used in GRPO training. The system consists of three main components: data preparation scripts, dataset utilities, and testing frameworks. This infrastructure integrates with the dual-mode GPT architecture to enable efficient reward model training.

## Architecture

### High-Level Architecture

```
Base Model (Generator) → prepare_reward_data.py → Binary Dataset Files
                                ↓
                         reward_dataset_loader.py → PyTorch DataLoader
                                ↓
                         Reward Model Training Loop
```

### Component Design

#### 1. Data Preparation Pipeline (prepare_reward_data.py)

**Purpose**: Generate mixed natural/synthetic training data for reward models

**Key Components**:
- Base model loader with generator mode enforcement
- Text generation with configurable sampling parameters
- Crossover point sampling and sequence mixing
- Binary data serialization with metadata

**Algorithm Flow**:
1. Load pre-trained base model in generator mode
2. Load and tokenize raw text data using tiktoken GPT-2 encoding
3. Split data maintaining alignment with base model's train/val splits
4. For each data block:
   - Sample random crossover point K ∈ [1, block_size-1]
   - Take first K tokens from natural data
   - Generate remaining (block_size - K) tokens using base model
   - Create target labels [K/block_size, (block_size-K)/block_size]
5. Save sequences and labels to binary files with metadata

#### 2. Dataset Loading Infrastructure (reward_dataset_loader.py)

**Purpose**: Provide PyTorch-compatible dataset classes for reward model training

**Key Components**:
- RewardDataset class extending torch.utils.data.Dataset
- Binary file loading with automatic reshaping
- Metadata parsing and validation
- Dataset statistics and quality analysis
- DataLoader factory functions

**Data Format**:
- Input sequences: uint16 arrays shaped (num_samples, block_size)
- Target probabilities: float32 arrays shaped (num_samples, 2)
- Metadata: Text files with shape and type information

#### 3. Testing Framework (test_reward_data_prep.py)

**Purpose**: Validate data preparation and reward model functionality

**Test Categories**:
- Binary data loading and format validation
- Reward model forward pass verification
- Probability distribution correctness
- Dataset statistics analysis

## Components and Interfaces

### prepare_reward_data.py Interface

```python
def main():
    # Command-line arguments
    --model_path: str          # Path to base model checkpoint
    --data_path: str           # Path to raw text data
    --output_dir: str          # Output directory for datasets
    --train_split: float       # Train/validation split ratio
    --samples_per_chunk: int   # Samples per data chunk
    --temperature: float       # Generation temperature
    --top_k: int              # Top-k sampling parameter
    --device: str             # Computation device
    --seed: int               # Random seed

def load_base_model(model_path, device) -> (GPT, GPTConfig):
    # Load checkpoint and create generator model

def create_reward_samples(tokens, model, config, device, ...) -> (List, List):
    # Generate mixed sequences with crossover points

def save_reward_dataset(samples_x, samples_y, output_dir, split_name):
    # Serialize to binary files with metadata
```

### RewardDataset Class Interface

```python
class RewardDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train')
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]
    def get_stats(self) -> Dict[str, Any]

def create_reward_dataloaders(data_dir: str, batch_size: int = 32, 
                            num_workers: int = 0) -> Tuple[DataLoader, DataLoader]

def print_dataset_info(data_dir: str):
    # Display comprehensive dataset statistics
```

### GPTConfig Extension

```python
@dataclass
class GPTConfig:
    # ... existing parameters ...
    reward_head_hidden_dim: int = 256  # Configurable reward head size
```

## Data Models

### Binary File Format

**Input Sequences (X files)**:
- Format: Binary uint16 array
- Shape: (num_samples, block_size)
- Content: Token IDs from mixed natural/synthetic sequences

**Target Labels (Y files)**:
- Format: Binary float32 array  
- Shape: (num_samples, 2)
- Content: [P(natural), P(synthetic)] probability pairs

**Metadata Files**:
- Format: Plain text key-value pairs
- Content: Dataset dimensions, types, and statistics

### Memory Layout

```
Dataset Directory Structure:
├── train_x.bin          # Training input sequences
├── train_y.bin          # Training target probabilities  
├── train_metadata.txt   # Training set metadata
├── val_x.bin           # Validation input sequences
├── val_y.bin           # Validation target probabilities
└── val_metadata.txt    # Validation set metadata
```

## Error Handling

### Data Preparation Errors
- Model loading failures with clear error messages
- Generation failures with graceful continuation
- File I/O errors with proper cleanup
- Invalid crossover points with bounds checking

### Dataset Loading Errors
- Missing file detection with helpful suggestions
- Shape mismatch validation with diagnostic info
- Metadata parsing errors with fallback behavior
- Memory allocation failures with size reporting

### Configuration Validation
- Parameter range checking for generation settings
- Device availability verification
- Seed validation for reproducibility
- Path existence verification

## Testing Strategy

### Unit Tests

1. **Data Preparation Tests**
   - Binary file format validation
   - Metadata consistency checking
   - Probability distribution correctness
   - Crossover point sampling verification

2. **Dataset Loading Tests**
   - PyTorch Dataset interface compliance
   - DataLoader integration functionality
   - Statistics calculation accuracy
   - Memory efficiency validation

3. **Configuration Tests**
   - Reward head hidden dimension configuration
   - Parameter validation and defaults
   - Backward compatibility verification

### Integration Tests

1. **End-to-End Pipeline**
   - Complete data preparation workflow
   - Dataset loading and training integration
   - Multi-device compatibility testing

2. **Performance Tests**
   - Large dataset handling capability
   - Memory usage optimization
   - Generation speed benchmarking

### Validation Tests

1. **Mathematical Correctness**
   - Probability sum validation (should equal 1.0)
   - Crossover point distribution analysis
   - Target label accuracy verification

2. **Data Quality**
   - Natural/synthetic text mixing verification
   - Generation consistency across runs
   - Dataset balance analysis