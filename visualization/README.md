# Diffusion Inference Visualizer

A PyQt6-based application for visualizing and interacting with the step-by-step diffusion inference process.

## Quick Start

```bash
python launch.py
```

## Features

- **Step-by-step visualization** of diffusion inference process
- **Interactive navigation** through prediction and remasking substeps
- **Real-time token statistics** and visualization
- **Editable remasking** - modify tokens at any step and restart from there
- **Intelligent remasking** using base model when no dedicated remasking model available
- **Enhanced token display** with colors and extended character mappings

## Requirements

- Python 3.7+
- PyQt6
- PyTorch
- NumPy

The launcher will automatically install PyQt6 and NumPy if missing. PyTorch must be installed manually from https://pytorch.org/

## File Structure

- `launch.py` - Main entry point with dependency checking
- `inference_visualizer.py` - Core PyQt6 application
- `diffusion_utils.py` - Shared utility functions for diffusion operations

## Usage

1. Ensure model files exist in `../out/` directory
2. Ensure data files exist in `../data/shakespeare_char/` directory
3. Run `python launch.py`

The visualizer will load the configured model and allow you to:
- View each diffusion step in detail
- See actual model predictions vs remasked tokens
- Edit tokens at any remasking step
- Restart generation from any point

## Troubleshooting

- **Missing PyQt6**: Launcher will auto-install
- **Missing PyTorch**: Install from https://pytorch.org/get-started/locally/
- **Model not found**: Check `../out/` directory for checkpoint files
- **Data not found**: Check `../data/shakespeare_char/` directory for vocab files