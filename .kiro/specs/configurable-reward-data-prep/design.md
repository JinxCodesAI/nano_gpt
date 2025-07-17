# Design Document

## Overview

This design enhances the reward model data preparation system with configurable tokenization support and binary file reuse capabilities. The solution maintains backward compatibility while adding flexible input modes that support both BPE and character-level tokenization, and allows direct reuse of existing train.bin/val.bin files.

## Architecture

### Core Components

1. **TokenizationManager**: Handles detection and switching between tokenization methods
2. **DataLoader**: Unified interface for loading from raw text or binary files  
3. **ConfigurationValidator**: Ensures parameter compatibility and provides clear error messages
4. **Enhanced prepare_reward_data.py**: Updated main script with new command-line options

### Data Flow

```
Input Sources → TokenizationManager → DataLoader → RewardSampleGenerator → Binary Output
     ↓                    ↓              ↓              ↓                    ↓
Raw Text Files      Detect Method   Load Tokens    Generate Mixed      Save Dataset
Binary Files        Load Vocab      Validate       Sequences           
```

## Components and Interfaces

### TokenizationManager Class

```python
class TokenizationManager:
    def __init__(self, data_path: str = None, meta_path: str = None):
        self.tokenization_type: str  # 'bpe' or 'char'
        self.vocab_size: int
        self.encoder: callable
        self.decoder: callable
    
    def detect_tokenization_method(self, data_path: str) -> str:
        """Auto-detect tokenization from meta.pkl or infer from context"""
    
    def load_char_tokenization(self, meta_path: str):
        """Load character-level vocab from meta.pkl"""
    
    def load_bpe_tokenization(self):
        """Initialize tiktoken BPE encoding"""
    
    def encode(self, text: str) -> List[int]:
        """Encode text using detected method"""
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens using detected method"""
```

### DataLoader Class

```python
class DataLoader:
    def __init__(self, tokenization_manager: TokenizationManager):
        self.tokenizer = tokenization_manager
    
    def load_from_text(self, text_path: str, train_split: float) -> Tuple[List[int], List[int]]:
        """Load and split raw text file"""
    
    def load_from_binary(self, train_bin: str, val_bin: str) -> Tuple[List[int], List[int]]:
        """Load existing binary files"""
    
    def validate_binary_files(self, train_bin: str, val_bin: str):
        """Validate binary file format and compatibility"""
```

### Enhanced Command-Line Interface

New parameters added to prepare_reward_data.py:

```python
# New input mode parameters
parser.add_argument('--input_mode', choices=['text', 'binary'], default='text',
                    help='Input mode: text (raw text file) or binary (existing .bin files)')
parser.add_argument('--train_bin', type=str, 
                    help='Path to existing train.bin file (binary mode only)')
parser.add_argument('--val_bin', type=str,
                    help='Path to existing val.bin file (binary mode only)')

# Tokenization configuration
parser.add_argument('--tokenization', choices=['auto', 'bpe', 'char'], default='auto',
                    help='Tokenization method (auto-detect, bpe, or char)')
parser.add_argument('--meta_path', type=str,
                    help='Path to meta.pkl file for character tokenization')
```

## Data Models

### Configuration Structure

```python
@dataclass
class RewardDataConfig:
    # Input configuration
    input_mode: str = 'text'  # 'text' or 'binary'
    data_path: str = None     # Raw text file path
    train_bin: str = None     # Existing train.bin path
    val_bin: str = None       # Existing val.bin path
    
    # Tokenization configuration
    tokenization: str = 'auto'  # 'auto', 'bpe', 'char'
    meta_path: str = None       # meta.pkl path for char tokenization
    
    # Generation parameters (existing)
    model_path: str
    output_dir: str = 'data/reward_dataset'
    samples_per_chunk: int = 10
    temperature: float = 1.0
    # ... other existing parameters
```

### Tokenization Metadata

```python
@dataclass
class TokenizationInfo:
    method: str           # 'bpe' or 'char'
    vocab_size: int      # Size of vocabulary
    meta_path: str = None # Path to meta.pkl if char tokenization
    
    def to_dict(self) -> dict:
        """Serialize for saving with dataset"""
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TokenizationInfo':
        """Deserialize from saved dataset"""
```

## Error Handling

### Validation Rules

1. **Input Mode Validation**:
   - Binary mode requires both train_bin and val_bin
   - Text mode requires data_path
   - Conflicting parameters trigger warnings

2. **Tokenization Validation**:
   - Character tokenization requires meta.pkl file
   - Binary files must match expected tokenization method
   - Vocab size compatibility checks

3. **File Validation**:
   - Binary files must exist and be readable
   - File sizes must be reasonable (not empty, not too large)
   - Token values must be within vocab range

### Error Messages

```python
class RewardDataPrepError(Exception):
    """Base exception for reward data preparation errors"""

class TokenizationMismatchError(RewardDataPrepError):
    """Raised when tokenization methods are incompatible"""

class BinaryFileError(RewardDataPrepError):
    """Raised when binary files are invalid or incompatible"""

class ConfigurationError(RewardDataPrepError):
    """Raised when command-line parameters are invalid"""
```

## Testing Strategy

### Unit Tests

1. **TokenizationManager Tests**:
   - Test BPE encoding/decoding
   - Test character encoding/decoding  
   - Test auto-detection logic
   - Test error handling for missing files

2. **DataLoader Tests**:
   - Test text file loading and splitting
   - Test binary file loading and validation
   - Test error cases (missing files, corrupted data)

3. **Configuration Tests**:
   - Test parameter validation
   - Test backward compatibility
   - Test error message clarity

### Integration Tests

1. **End-to-End Workflows**:
   - Text mode with BPE tokenization (existing workflow)
   - Text mode with character tokenization
   - Binary mode with existing train/val files
   - Mixed scenarios and error cases

2. **Compatibility Tests**:
   - Existing datasets remain loadable
   - Generated datasets work with existing training code
   - Command-line backward compatibility

### Test Data

Create test fixtures:
- Small shakespeare text file
- Character-level meta.pkl file
- Sample train.bin/val.bin files (both BPE and char)
- Corrupted/invalid files for error testing

## Implementation Approach

### Phase 1: Core Infrastructure

1. Implement TokenizationManager class
2. Implement DataLoader class  
3. Add configuration validation
4. Create comprehensive unit tests

### Phase 2: Integration

1. Update prepare_reward_data.py main function
2. Add new command-line parameters
3. Integrate new classes with existing logic
4. Maintain backward compatibility

### Phase 3: Testing and Documentation

1. Create integration tests
2. Test with real shakespeare_char data
3. Update documentation and examples
4. Validate performance impact

## Backward Compatibility

### Preserved Behavior

- All existing command-line parameters work unchanged
- Default behavior identical to current implementation
- Existing output format and file structure maintained
- No changes to reward_dataset_loader.py required

### Migration Path

Users can gradually adopt new features:
1. Continue using existing workflows (no changes needed)
2. Experiment with character tokenization on new projects
3. Migrate to binary file reuse when beneficial
4. Full adoption of new features as needed

## Performance Considerations

### Memory Usage

- Binary file loading avoids text processing overhead
- Character tokenization uses less memory than BPE
- Validation steps add minimal overhead

### Processing Speed

- Binary mode significantly faster (skips text processing)
- Character tokenization faster than BPE encoding
- Auto-detection adds minimal startup cost

### Storage Impact

- No changes to output format or size
- Metadata files remain small
- Binary input files already exist (no additional storage)

## Security Considerations

### File Access

- Validate all file paths to prevent directory traversal
- Check file permissions before processing
- Handle file access errors gracefully

### Input Validation

- Validate token ranges against vocab size
- Check file sizes to prevent memory exhaustion
- Sanitize command-line inputs

## Configuration Examples

### Text Mode with Character Tokenization

```bash
python prepare_reward_data.py \
    --model_path checkpoints/shakespeare_char_model.pt \
    --input_mode text \
    --data_path data/shakespeare_char/input.txt \
    --tokenization char \
    --meta_path data/shakespeare_char/meta.pkl \
    --output_dir data/reward_dataset_char
```

### Binary Mode (Reuse Existing Files)

```bash
python prepare_reward_data.py \
    --model_path checkpoints/base_model.pt \
    --input_mode binary \
    --train_bin data/shakespeare/train.bin \
    --val_bin data/shakespeare/val.bin \
    --output_dir data/reward_dataset_reuse
```

### Auto-Detection Mode

```bash
python prepare_reward_data.py \
    --model_path checkpoints/model.pt \
    --data_path data/shakespeare_char/input.txt \
    --tokenization auto \
    --output_dir data/reward_dataset_auto
```

This design provides a flexible, backward-compatible solution that addresses all requirements while maintaining the existing system's reliability and performance.