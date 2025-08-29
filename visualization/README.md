# Diffusion Inference Visualizer

A modern PyQt6-based application for visualizing and interacting with the step-by-step diffusion inference process using intelligent remasking.

## Features

### 🎯 **Inference Visualization**
- Step-by-step visualization of diffusion inference process
- Real-time display of masked and unmasked tokens at each step
- Interactive step navigation with precise slider control
- Visual distinction between masked (█) and regular tokens

### 🎮 **Interactive Controls**
- **MainPanel**: Display and edit content at any generation step
- **LeftPanel**: Complete generation settings (temperature, iterations, ratios)
- **BottomPanel**: Step slider and navigation controls for precise step selection
- Edit tokens at any step and re-run generation from that point

### 🔧 **Generation Settings**
- Temperature control for sampling randomness
- Configurable diffusion iterations (1-200 steps)
- Adjustable sequence length (50-1024 tokens)
- Start and end masking ratios
- Randomness strength for intelligent remasking balance

### 🧠 **Intelligent Remasking**
- Uses base model probabilities for smart token remasking
- Combines model predictions with randomness for natural generation
- Real-time probability assessment and visualization
- No separate remasking model required

### ✏️ **Step Editing & Branching**
- Edit content at any step during generation
- Mask/unmask tokens manually
- Replace selected text portions
- Restart generation from edited step
- Explore different generation paths

### 🎨 **Modern UI**
- Dark theme optimized for text visualization
- Three-panel layout: Settings | Main Display | Step Controls
- Real-time progress tracking
- Intuitive navigation controls

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- PyQt6
- Trained diffusion model (base model only, no remasking model needed)

### Quick Start
```bash
# Navigate to visualization folder
cd visualization

# Install dependencies and launch (automatic)
python launch.py
```

### Manual Installation
```bash
# Install PyQt6
pip install PyQt6

# Install other requirements if needed
pip install -r requirements.txt

# Run the new inference visualizer
python inference_visualizer.py

# Or run the old model explorer
python model_explorer.py
```

## Usage

### 1. **Launch Application**
```bash
python launch.py
```
or
```bash
python inference_visualizer.py
```

### 2. **Load Base Model**
- Select a trained diffusion model from the dropdown
- Click "Load Model" to load it (background loading with progress)
- Only base models are needed - intelligent remasking uses the base model
- Models should be in `../out/` directory

### 3. **Configure Generation**
- **Temperature**: Controls randomness (0.1-2.0)
- **Iterations**: Number of diffusion steps (1-200)
- **Length**: Sequence length to generate (50-1024)
- **Start/End Ratio**: Initial and final masking ratios
- **Randomness**: Balance between model-guided and random remasking

### 4. **Start Generation**
- Click "Start Generation" to begin inference
- Watch real-time progress in the main panel
- Each step shows current token states with masked tokens as █

### 5. **Navigate Steps**
- Use the bottom slider to jump to any completed step
- Use Previous/Next buttons for precise navigation
- Step display updates immediately

### 6. **Edit and Branch**
- At any step, edit the text content manually
- Use "Mask Selected", "Unmask Selected", or "Replace Selected"
- Click "Restart from Current" to continue generation from edited state
- Explore different generation paths by editing different steps

## File Structure

```
visualization/
├── inference_visualizer.py  # NEW: Main inference visualization app
├── model_explorer.py        # OLD: Original model testing app
├── enhanced_explorer.py     # OLD: Enhanced version with batch processing
├── launch.py               # Launcher (now uses inference_visualizer.py)
├── requirements.txt        # Python dependencies  
└── README.md              # This file
```

## Features in Detail

### Inference Visualization System
- **Step-by-Step Display**: See exactly how diffusion inference progresses
- **Real-time Updates**: Watch tokens unmask and remask in real-time
- **Visual Token States**: Clear distinction between masked (█) and unmasked tokens
- **Progress Tracking**: Visual progress bar and status updates

### Interactive Step Navigation
- **Precise Slider Control**: Jump to any step with pixel-perfect accuracy
- **Keyboard Navigation**: Previous/Next buttons for fine control
- **Instant Updates**: Main display updates immediately on step change
- **Step Memory**: All completed steps are stored and can be revisited

### Intelligent Remasking Integration
- **Base Model Only**: No separate remasking model required
- **Probability-based**: Uses model confidence to guide remasking decisions
- **Randomness Control**: Balance between smart and random remasking
- **Real-time Application**: Remasking happens during generation

### Step Editing & Branching
- **In-place Editing**: Edit text content directly in the main panel
- **Token Manipulation**: Mask, unmask, or replace selected tokens
- **Branch Generation**: Restart from any edited step to explore alternatives
- **State Preservation**: Original steps are preserved when branching

### Generation Control
- **Flexible Parameters**: Full control over all diffusion parameters
- **Start/Stop Control**: Can stop generation at any point
- **Resume Capability**: Continue from current step with new settings
- **Background Processing**: Generation runs in separate thread

## Technical Details

### Architecture
- **PyQt6 Framework**: Modern, cross-platform GUI framework
- **Three-Panel Layout**: MainPanel, LeftPanel, BottomPanel design
- **Threading**: Background inference prevents UI blocking
- **Memory Management**: Efficient step storage and token handling
- **GPU Support**: Automatic CUDA detection and utilization

### Intelligent Remasking Implementation
- **Base Model Integration**: Uses loaded model for probability assessment
- **Token Probability Calculation**: 1 - current_token_probability
- **Weighted Combination**: Balances model probabilities with randomness
- **Real-time Application**: Applies during each inference step

### Step Management
- **Step Storage**: Each step's tokens stored in memory
- **Fast Navigation**: Instant switching between completed steps
- **Edit Tracking**: Maintains original and edited states
- **Branching Support**: Multiple generation paths from single base

### Performance Optimizations
- **Background Processing**: Inference runs in separate QThread
- **Progressive Updates**: UI updates as steps complete
- **Memory Efficient**: Only stores necessary step data
- **Responsive UI**: Never blocks during generation

## Troubleshooting

### Common Issues

**1. PyQt6 Installation Issues**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install PyQt6
```

**2. Model Loading Errors**
- Ensure model files exist in `../out/` directory
- Check that models were trained with diffusion architecture
- Verify model files are not corrupted

**3. Generation Errors**
- Check that vocabulary files exist in `../data/shakespeare_char/`
- Ensure `meta.pkl` contains proper vocabulary mapping
- Verify model architecture matches expectations

**4. Performance Issues**
- Use GPU (CUDA) for faster inference
- Reduce sequence length for faster generation
- Lower iteration count for quicker results

**5. Step Navigation Issues**
- Generation must complete steps before they can be navigated
- Edited steps create new branches - original steps are preserved
- Use "Stop Generation" before editing steps

### Performance Tips
- Use GPU (CUDA) for significantly better performance
- Start with shorter sequences (50-128 tokens) for testing
- Use fewer iterations (10-20) for rapid experimentation
- Intelligent remasking works best with randomness 0.3-0.8

## Key Differences from Old Explorer

### New Inference Visualizer
- **Focus**: Step-by-step inference visualization
- **Interaction**: Edit any step and restart generation
- **Model Support**: Base models only with intelligent remasking
- **UI**: Three-panel layout optimized for inference
- **Navigation**: Precise step-by-step control

### Old Model Explorer
- **Focus**: Model testing and corruption analysis
- **Interaction**: Batch processing and comparison
- **Model Support**: Multiple model types simultaneously
- **UI**: Tab-based layout for different views
- **Navigation**: Tab switching between analysis views

## License

This application is part of the diffusion training project and inherits its license terms.