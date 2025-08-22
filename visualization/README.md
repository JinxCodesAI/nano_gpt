# Diffusion Model Explorer

A modern PyQt6-based application for testing and visualizing diffusion models with different corruption strategies and model predictions.

## Features

### 🎯 **Model Loading**
- Load multiple pre-trained models simultaneously
- Support for unmasking, remasking, and remasking binary models
- Background loading with progress indication
- Automatic model detection and configuration

### 📊 **Text Visualization** 
- Color-coded text display showing:
  - **Original text** in default color
  - **Masked/corrupted positions** highlighted in blue
  - **Correct predictions** highlighted in green
  - **Incorrect predictions** highlighted in pink
- Three separate tabs for original, corrupted, and predicted text
- Monospace font for clear character alignment

### 🛠 **Corruption Strategies**
- **Random corruption**: Random token replacement
- **Sticky corruption**: Spatially correlated masking
- **Fragment corruption**: Real text segment replacement
- **Mixed strategy**: Combination of all strategies
- Real-time preview of corruption effects
- Adjustable masking ratio (10% - 80%)
- Configurable sticky rounds for spatial correlation

### 🤖 **Model Predictions**
- Generate predictions for any loaded model
- Side-by-side comparison with original text
- Accuracy calculation on corrupted positions
- Visual highlighting of correct/incorrect predictions
- Support for all training types (unmasking, remasking, binary)

### 🎨 **Modern UI**
- Dark theme with professional styling
- Intuitive layout with control panel and visualization area
- Responsive design with adjustable panels
- Real-time parameter updates
- Status bar with detailed feedback

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA support recommended)
- PyQt6

### Quick Start
```bash
# Navigate to visualization folder
cd visualization

# Install dependencies (automatic)
python launch.py
```

### Manual Installation
```bash
# Install PyQt6
pip install PyQt6

# Install other requirements if needed
pip install -r requirements.txt

# Run the application
python model_explorer.py
```

## Usage

### 1. **Launch Application**
```bash
python launch.py
```
or
```bash
python model_explorer.py
```

### 2. **Load Models**
- Click the model loading buttons to load pre-trained models:
  - **Unmasking Model**: `../out/14.6_unmasking_no_noise.pt`
  - **Remasking Binary**: `../out/1.23_remasking_bin.pt` 
  - **Remasking**: `../out/1.35_remask.pt`
- Models load in background with progress indication
- Multiple models can be loaded simultaneously

### 3. **Generate Text Samples**
- Adjust sample length (50-1024 characters)
- Click "Generate New Sample" to load text from Shakespeare dataset
- Original text appears in the "Original Text" tab

### 4. **Configure Corruption**
- Select corruption strategy (random/sticky/fragment/mixed)
- Adjust masking ratio with slider (10%-80%)
- Set sticky rounds for spatial correlation
- Preview updates automatically

### 5. **Apply Corruption**
- Click "Apply Corruption" to process the sample
- View results in "Corrupted Text" tab
- Corrupted positions highlighted in blue

### 6. **Generate Predictions**
- Select a loaded model from dropdown
- Click "Generate Predictions" 
- View results in "Model Predictions" tab:
  - Green highlight: Correct predictions
  - Pink highlight: Incorrect predictions
- Accuracy statistics shown in status bar

## File Structure

```
visualization/
├── model_explorer.py    # Main application
├── launch.py           # Launcher with dependency checking
├── requirements.txt    # Python dependencies  
└── README.md          # This file
```

## Features in Detail

### Model Loading System
- **Background Loading**: Models load in separate threads to prevent UI freezing
- **Progress Tracking**: Visual progress bar shows loading status
- **Error Handling**: Graceful handling of missing or corrupted model files
- **Multi-model Support**: Load and compare different model types simultaneously

### Text Visualization System
- **Color Coding**: Intuitive color scheme for different text states
- **Character-level Display**: Precise visualization of token-level changes
- **HTML Rendering**: Rich text display with proper formatting
- **Scrollable Views**: Handle long text samples efficiently

### Corruption System Integration
- **Direct Integration**: Uses `train_utils.py` functions for consistency
- **Real-time Preview**: See corruption effects immediately
- **Multiple Strategies**: All training corruption strategies supported
- **Parameter Control**: Fine-tune corruption parameters interactively

### Prediction Analysis
- **Accuracy Metrics**: Detailed accuracy calculation on masked positions
- **Visual Comparison**: Side-by-side original vs predicted text
- **Model Comparison**: Easy switching between different loaded models
- **Performance Tracking**: Real-time feedback on prediction quality

## Technical Details

### Architecture
- **PyQt6 Framework**: Modern, cross-platform GUI framework
- **Threading**: Background model loading prevents UI blocking
- **Memory Management**: Efficient handling of large models and datasets
- **GPU Support**: Automatic CUDA detection and utilization

### Performance
- **Lazy Loading**: Data and models loaded on demand
- **Efficient Visualization**: Optimized text rendering for large samples
- **Memory Mapping**: Direct access to dataset files without full loading
- **Cached Computations**: Avoid redundant calculations

### Extensibility
- **Modular Design**: Easy to add new corruption strategies
- **Plugin Architecture**: Simple to integrate new model types
- **Configurable Parameters**: All training parameters exposed in UI
- **Data Format Support**: Easy to adapt to different datasets

## Troubleshooting

### Common Issues

**1. PyQt6 Installation Issues**
```bash
# Try upgrading pip first
pip install --upgrade pip
pip install PyQt6
```

**2. CUDA Not Available**
- Application works on CPU but slower
- Install PyTorch with CUDA support for better performance

**3. Model Loading Errors**
- Ensure model files exist in correct paths
- Check that models were trained with compatible architecture

**4. Data Loading Issues**
- Verify `../data/shakespeare_char/` directory exists
- Ensure `train.bin` and `meta.pkl` files are present

### Performance Tips
- Use GPU (CUDA) for better performance
- Start with smaller samples (50-256 chars) for faster processing
- Load models one at a time if memory is limited

## License

This application is part of the diffusion training project and inherits its license terms.