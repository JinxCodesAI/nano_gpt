"""
Modern Model Explorer for Diffusion Training Models
A comprehensive tool for testing and visualizing different corruption strategies and model predictions.
"""

import sys
import os
import pickle
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path to import train_utils
sys.path.append(str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QTextEdit, QPushButton, QLabel, QComboBox, QSlider, QSpinBox,
    QGroupBox, QGridLayout, QSplitter, QScrollArea, QFrame,
    QProgressBar, QMessageBox, QTabWidget, QListWidget, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor

# Import our training utilities
from train_utils import (
    TrainingContext, apply_random_corruption_gpu, apply_sticky_corruption_gpu,
    apply_fragment_corruption_gpu, apply_synthetic_corruption,
    apply_gpu_masking_training, find_double_newline_indices, load_synthetic_model
)
from model import GPT, GPTConfig


class ModelLoader(QThread):
    """Background thread for loading models"""
    progress = pyqtSignal(int)
    finished_signal = pyqtSignal(str, object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, model_path, device):
        super().__init__()
        self.model_path = model_path
        self.device = device
        
    def run(self):
        try:
            self.progress.emit(25)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.progress.emit(50)
            
            # Create model from checkpoint
            model_args = checkpoint['model_args']
            config = GPTConfig(**model_args)
            model = GPT(config)
            
            self.progress.emit(75)
            
            # Load state dict
            state_dict = checkpoint['model']
            # Fix keys if needed
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.progress.emit(100)
            model_name = Path(self.model_path).stem
            self.finished_signal.emit(model_name, model)
            
        except Exception as e:
            self.error_signal.emit(f"Error loading model: {str(e)}")


class TextVisualizer(QWidget):
    """Widget for visualizing text with color-coded tokens"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Text display area
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setFont(QFont("Consolas", 12))
        self.text_display.setMinimumHeight(200)
        
        layout.addWidget(self.text_display)
        self.setLayout(layout)
        
    def display_text(self, tokens, vocab, mask=None, predictions=None, title="Text", probabilities=None, min_prob=None, max_prob=None):
        """Display text with optional mask and prediction highlighting"""
        html = f"<h3>{title}</h3><div style='font-family: Consolas, monospace; font-size: 14px;'>"
        
        for i, token_id in enumerate(tokens):
            char = vocab.get(token_id, f"[UNK{token_id}]")
            
            # Escape HTML characters
            if char == '<':
                char = '&lt;'
            elif char == '>':
                char = '&gt;'
            elif char == '&':
                char = '&amp;'
            elif char == ' ':
                char = '&nbsp;'
            elif char == '\n':
                char = '<br>'
                
            # Apply styling based on mask/predictions/probabilities
            style = ""
            
            if probabilities is not None and min_prob is not None and max_prob is not None:
                # Probability-based coloring (for remasking assessment)
                prob = probabilities[i]
                if max_prob > min_prob:
                    # Normalize probability to 0-1 range
                    normalized_prob = (prob - min_prob) / (max_prob - min_prob)
                    # Interpolate between gray (low prob) and red (high prob)
                    red_intensity = int(255 * normalized_prob)
                    gray_intensity = int(128 * (1 - normalized_prob))
                    style = f"background-color: rgb({red_intensity + gray_intensity}, {gray_intensity}, {gray_intensity}); color: black;"
                else:
                    style = "background-color: #808080; color: black;"  # All same probability - gray
            elif mask is not None and mask[i]:
                if predictions is not None:
                    # Show prediction vs original
                    pred_char = vocab.get(predictions[i], f"[UNK{predictions[i]}]")
                    if predictions[i] == token_id:
                        style = "background-color: #90EE90; color: black;"  # Correct prediction - green
                    else:
                        style = "background-color: #FFB6C1; color: black;"  # Wrong prediction - pink
                        char = pred_char  # Show only the wrong predicted character
                else:
                    style = "background-color: #87CEEB; color: black;"  # Masked - blue
            elif predictions is not None:
                # Unmasked but showing predictions
                if predictions[i] == token_id:
                    style = "background-color: #F0F8FF; color: black;"  # Correct - light blue
                else:
                    style = "background-color: #FFFFE0; color: black;"  # Different - light yellow
                    
            html += f"<span style='{style}'>{char}</span>"
            
        html += "</div>"
        self.text_display.setHtml(html)


class ModelExplorerApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.vocab = None
        self.current_sample = None
        self.current_corrupted = None
        self.current_predictions = None
        self.token_probabilities = None
        self.min_probability = None
        self.max_probability = None
        self.data = None
        self.current_sample = None
        self.training_ctx = None
        
        # Load data and vocabulary
        self.load_data()
        self.init_ui()
        self.setup_training_context()
        
    def load_data(self):
        """Load Shakespeare data and vocabulary"""
        try:
            # Load vocabulary
            meta_path = Path("../data/shakespeare_char/meta.pkl")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            # Create vocab mapping from the actual format
            self.vocab = meta['itos']  # itos is already a mapping from int to char
            self.vocab_size = meta['vocab_size']
            
            # Load training data
            data_path = Path("../data/shakespeare_char/train.bin")
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            
            # Find valid starting positions
            self.valid_indices = find_double_newline_indices(self.data, self.vocab_size, 1024)
            
            print(f"Loaded data: vocab_size={self.vocab_size}, data_length={len(self.data)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            print(f"Data loading error: {e}")
            # Set defaults to prevent crashes
            self.vocab = {}
            self.vocab_size = 65
            self.data = None
            self.valid_indices = []
            
    def setup_training_context(self):
        """Setup training context with default values matching current training configuration"""
        self.training_ctx = TrainingContext(
            training_type='unmasking',
            batch_size=1,
            block_size=512,
            max_iters=13000,  # Match train_run.py default
            device=str(self.device),
            device_type='cuda' if 'cuda' in str(self.device) else 'cpu',
            seed_offset=0,
            data_dir='../data/shakespeare_char',
            meta_vocab_size=self.vocab_size,
            mask_token_id=self.vocab_size,
            wrong_token_id=self.vocab_size + 1,
            remask_good_id=self.vocab_size + 2,
            remask_wrong_id=self.vocab_size + 3,
            extended_vocab_size=self.vocab_size + 4,
            iter_num=5000,  # Middle of training
            guaranteed_unmasked_max=0.8,  # Match train_run.py
            guaranteed_unmasked_min=0.2,  # Match train_run.py
            random_mask_warmup=8000,  # Match train_run.py
            noise_max_ratio=0.2,  # Match train_run.py
            sticky_rounds=30,  # Match train_run.py
            sticky_p1_p2_multiplier=20.0,  # Match train_run.py
            sticky_p1_divisor=3.0,  # Match train_run.py
            sticky_transition_start=5000,  # Match train_run.py
            sticky_transition_end=20000,  # Match train_run.py
            remasking_corruption_strategy='mixed',
            remasking_strategy_weights=[0.25, 0.4, 0.25, 0.1],
            eval_iters=20,  # Match train_run.py
            warmup_iters=2000,
            lr_decay_iters=11000,
            learning_rate=1e-3,
            min_lr=1e-4
        )
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Diffusion Model Explorer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0084ff;
                border: none;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
            QComboBox, QSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #3c3c3c;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0084ff;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0084ff;
                border-radius: 3px;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create splitter for main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - controls
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - text visualization
        right_panel = self.create_visualization_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 1000])
        
        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        main_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Load models and generate samples to begin")
        
    def create_control_panel(self):
        """Create the control panel with all settings"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        
        # Model Loading Section
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout()
        
        # Dynamic model list
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(100)
        self.refresh_model_list()
        model_layout.addWidget(QLabel("Available Models:"))
        model_layout.addWidget(self.model_list)
        
        self.load_model_btn = QPushButton("Load Selected Model")
        self.load_model_btn.clicked.connect(self.load_selected_model)
        model_layout.addWidget(self.load_model_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        model_layout.addWidget(self.progress_bar)
        
        # Model status
        self.model_status = QLabel("No models loaded")
        self.model_status.setStyleSheet("color: #888888;")
        model_layout.addWidget(self.model_status)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Sample Generation Section
        sample_group = QGroupBox("Sample Generation")
        sample_layout = QVBoxLayout()
        
        self.generate_btn = QPushButton("Generate New Sample")
        self.generate_btn.clicked.connect(self.generate_sample)
        sample_layout.addWidget(self.generate_btn)
        
        # Sample length
        length_layout = QHBoxLayout()
        length_layout.addWidget(QLabel("Sample Length:"))
        self.length_spinbox = QSpinBox()
        self.length_spinbox.setRange(50, 1024)
        self.length_spinbox.setValue(256)
        length_layout.addWidget(self.length_spinbox)
        sample_layout.addLayout(length_layout)
        
        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)
        
        # Corruption Settings Section
        corruption_group = QGroupBox("Corruption Settings")
        corruption_layout = QVBoxLayout()
        
        # Corruption strategy
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["random", "sticky", "fragment", "synthetic", "mixed"])
        self.strategy_combo.setCurrentText("mixed")
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(self.strategy_combo)
        corruption_layout.addLayout(strategy_layout)
        
        # Create tab widget for strategy-specific controls
        self.corruption_tabs = QTabWidget()
        
        # Random Corruption Tab
        self.create_random_corruption_tab()
        
        # Sticky Corruption Tab  
        self.create_sticky_corruption_tab()
        
        # Fragment Corruption Tab
        self.create_fragment_corruption_tab()
        
        # Synthetic Corruption Tab
        self.create_synthetic_corruption_tab()
        
        # Mixed Strategy Tab
        self.create_mixed_corruption_tab()
        
        corruption_layout.addWidget(self.corruption_tabs)
        corruption_group.setLayout(corruption_layout)
        layout.addWidget(corruption_group)
        
        # Actions Section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.apply_corruption_btn = QPushButton("Apply Corruption")
        self.apply_corruption_btn.clicked.connect(self.apply_corruption)
        self.apply_corruption_btn.setEnabled(False)
        actions_layout.addWidget(self.apply_corruption_btn)
        
        # Model selection for prediction
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Select model...")
        model_layout.addWidget(self.model_combo)
        actions_layout.addLayout(model_layout)
        
        self.predict_btn = QPushButton("Generate Predictions")
        self.predict_btn.clicked.connect(self.generate_predictions)
        self.predict_btn.setEnabled(False)
        actions_layout.addWidget(self.predict_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Remasking Section
        remasking_group = QGroupBox("Remasking")
        remasking_layout = QVBoxLayout()
        
        # Remasking model selection
        remasking_model_layout = QHBoxLayout()
        remasking_model_layout.addWidget(QLabel("Remasking Model:"))
        self.remasking_model_combo = QComboBox()
        self.remasking_model_combo.addItem("Select remasking model...")
        self.remasking_model_combo.currentTextChanged.connect(self.update_remasking_buttons)
        remasking_model_layout.addWidget(self.remasking_model_combo)
        remasking_layout.addLayout(remasking_model_layout)
        
        # Remasking buttons
        remasking_buttons_layout = QHBoxLayout()
        self.assess_btn = QPushButton("Assess")
        self.assess_btn.clicked.connect(self.assess_tokens)
        self.assess_btn.setEnabled(False)
        remasking_buttons_layout.addWidget(self.assess_btn)
        
        self.apply_remasking_btn = QPushButton("Apply")
        self.apply_remasking_btn.clicked.connect(self.apply_remasking)
        self.apply_remasking_btn.setEnabled(False)
        remasking_buttons_layout.addWidget(self.apply_remasking_btn)
        
        remasking_layout.addLayout(remasking_buttons_layout)
        
        # Probability info display
        self.prob_info_label = QLabel("Min/Max Probability: -")
        self.prob_info_label.setStyleSheet("font-size: 11px; color: #666666;")
        remasking_layout.addWidget(self.prob_info_label)
        
        remasking_group.setLayout(remasking_layout)
        layout.addWidget(remasking_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
    
    def create_random_corruption_tab(self):
        """Create tab for random corruption parameters"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Guaranteed unmasked range
        unmasked_group = QGroupBox("Dynamic Guaranteed Unmasked")
        unmasked_layout = QGridLayout()
        
        # Max guaranteed unmasked (start of training)
        unmasked_layout.addWidget(QLabel("Max (start):"), 0, 0)
        self.guaranteed_unmasked_max = QDoubleSpinBox()
        self.guaranteed_unmasked_max.setRange(0.0, 1.0)
        self.guaranteed_unmasked_max.setSingleStep(0.1)
        self.guaranteed_unmasked_max.setValue(0.8)  # Match train_run.py
        self.guaranteed_unmasked_max.valueChanged.connect(self.update_corruption_preview)
        unmasked_layout.addWidget(self.guaranteed_unmasked_max, 0, 1)
        
        # Min guaranteed unmasked (end of training)
        unmasked_layout.addWidget(QLabel("Min (end):"), 1, 0)
        self.guaranteed_unmasked_min = QDoubleSpinBox()
        self.guaranteed_unmasked_min.setRange(0.0, 1.0)
        self.guaranteed_unmasked_min.setSingleStep(0.1)
        self.guaranteed_unmasked_min.setValue(0.2)  # Match train_run.py
        self.guaranteed_unmasked_min.valueChanged.connect(self.update_corruption_preview)
        unmasked_layout.addWidget(self.guaranteed_unmasked_min, 1, 1)
        
        unmasked_group.setLayout(unmasked_layout)
        layout.addWidget(unmasked_group)
        
        # Random mask warmup settings
        warmup_group = QGroupBox("Random Mask Warmup")
        warmup_layout = QGridLayout()
        
        warmup_layout.addWidget(QLabel("Warmup iterations:"), 0, 0)
        self.random_mask_warmup = QSpinBox()
        self.random_mask_warmup.setRange(1, 50000)
        self.random_mask_warmup.setValue(8000)  # Match train_run.py
        self.random_mask_warmup.valueChanged.connect(self.update_corruption_preview)
        warmup_layout.addWidget(self.random_mask_warmup, 0, 1)
        
        # Add info label
        info_label = QLabel("After this many iterations, guaranteed_unmasked stays at minimum value")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 10px;")
        warmup_layout.addWidget(info_label, 1, 0, 1, 2)
        
        warmup_group.setLayout(warmup_layout)
        layout.addWidget(warmup_group)
        
        # Transition settings
        transition_group = QGroupBox("Sticky Transition")
        transition_layout = QGridLayout()
        
        transition_layout.addWidget(QLabel("Start iteration:"), 0, 0)
        self.sticky_transition_start = QSpinBox()
        self.sticky_transition_start.setRange(0, 50000)
        self.sticky_transition_start.setValue(5000)  # Match train_run.py
        self.sticky_transition_start.valueChanged.connect(self.update_corruption_preview)
        transition_layout.addWidget(self.sticky_transition_start, 0, 1)
        
        transition_layout.addWidget(QLabel("End iteration:"), 1, 0)
        self.sticky_transition_end = QSpinBox()
        self.sticky_transition_end.setRange(0, 50000)
        self.sticky_transition_end.setValue(20000)  # Match train_run.py
        self.sticky_transition_end.valueChanged.connect(self.update_corruption_preview)
        transition_layout.addWidget(self.sticky_transition_end, 1, 1)
        
        transition_group.setLayout(transition_layout)
        layout.addWidget(transition_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.corruption_tabs.addTab(tab, "Random")
    
    def create_sticky_corruption_tab(self):
        """Create tab for sticky corruption parameters"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Sticky masking parameters
        sticky_group = QGroupBox("Sticky Masking Parameters")
        sticky_layout = QGridLayout()
        
        sticky_layout.addWidget(QLabel("Rounds:"), 0, 0)
        self.sticky_rounds = QSpinBox()
        self.sticky_rounds.setRange(1, 50)
        self.sticky_rounds.setValue(30)  # Match train_run.py
        self.sticky_rounds.valueChanged.connect(self.update_corruption_preview)
        sticky_layout.addWidget(self.sticky_rounds, 0, 1)
        
        sticky_layout.addWidget(QLabel("P1-P2 Multiplier:"), 1, 0)
        self.sticky_p1_p2_multiplier = QDoubleSpinBox()
        self.sticky_p1_p2_multiplier.setRange(1.0, 50.0)
        self.sticky_p1_p2_multiplier.setValue(20.0)  # Match train_run.py
        self.sticky_p1_p2_multiplier.valueChanged.connect(self.update_corruption_preview)
        sticky_layout.addWidget(self.sticky_p1_p2_multiplier, 1, 1)
        
        sticky_layout.addWidget(QLabel("P1 Divisor:"), 2, 0)
        self.sticky_p1_divisor = QDoubleSpinBox()
        self.sticky_p1_divisor.setRange(1.0, 20.0)
        self.sticky_p1_divisor.setValue(3.0)  # Match train_run.py
        self.sticky_p1_divisor.valueChanged.connect(self.update_corruption_preview)
        sticky_layout.addWidget(self.sticky_p1_divisor, 2, 1)
        
        sticky_group.setLayout(sticky_layout)
        layout.addWidget(sticky_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.corruption_tabs.addTab(tab, "Sticky")
    
    def create_fragment_corruption_tab(self):
        """Create tab for fragment corruption parameters"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Info about fragment corruption
        info_label = QLabel("Fragment corruption uses real text segments from training data.\nIt uses sticky masking parameters to determine corruption patterns.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(info_label)
        
        # Reuse sticky parameters for fragment
        fragment_group = QGroupBox("Fragment + Sticky Parameters")
        fragment_layout = QGridLayout()
        
        fragment_layout.addWidget(QLabel("Sticky Rounds:"), 0, 0)
        self.fragment_sticky_rounds = QSpinBox()
        self.fragment_sticky_rounds.setRange(1, 50)
        self.fragment_sticky_rounds.setValue(30)  # Match train_run.py
        self.fragment_sticky_rounds.valueChanged.connect(self.update_corruption_preview)
        fragment_layout.addWidget(self.fragment_sticky_rounds, 0, 1)
        
        fragment_layout.addWidget(QLabel("P1-P2 Multiplier:"), 1, 0)
        self.fragment_p1_p2_multiplier = QDoubleSpinBox()
        self.fragment_p1_p2_multiplier.setRange(1.0, 50.0)
        self.fragment_p1_p2_multiplier.setValue(20.0)  # Match train_run.py
        self.fragment_p1_p2_multiplier.valueChanged.connect(self.update_corruption_preview)
        fragment_layout.addWidget(self.fragment_p1_p2_multiplier, 1, 1)
        
        fragment_layout.addWidget(QLabel("P1 Divisor:"), 2, 0)
        self.fragment_p1_divisor = QDoubleSpinBox()
        self.fragment_p1_divisor.setRange(1.0, 20.0)
        self.fragment_p1_divisor.setValue(3.0)  # Match train_run.py
        self.fragment_p1_divisor.valueChanged.connect(self.update_corruption_preview)
        fragment_layout.addWidget(self.fragment_p1_divisor, 2, 1)
        
        fragment_group.setLayout(fragment_layout)
        layout.addWidget(fragment_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.corruption_tabs.addTab(tab, "Fragment")
    
    def create_synthetic_corruption_tab(self):
        """Create tab for synthetic corruption parameters"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Info about synthetic corruption
        info_label = QLabel("Synthetic corruption uses a trained unmasking model to generate realistic corrupted text.\nRequires loading a synthetic model first.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(info_label)
        
        # Synthetic model loading
        synthetic_group = QGroupBox("Synthetic Model")
        synthetic_layout = QVBoxLayout()
        
        self.synthetic_model_combo = QComboBox()
        self.synthetic_model_combo.addItem("No synthetic model loaded...")
        synthetic_layout.addWidget(QLabel("Synthetic Model:"))
        synthetic_layout.addWidget(self.synthetic_model_combo)
        
        synthetic_group.setLayout(synthetic_layout)
        layout.addWidget(synthetic_group)
        
        # Synthetic + sticky parameters
        synth_sticky_group = QGroupBox("Synthetic + Sticky Parameters")
        synth_sticky_layout = QGridLayout()
        
        synth_sticky_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.synthetic_temperature = QDoubleSpinBox()
        self.synthetic_temperature.setRange(0.1, 2.0)
        self.synthetic_temperature.setSingleStep(0.1)
        self.synthetic_temperature.setValue(0.8)
        self.synthetic_temperature.valueChanged.connect(self.update_corruption_preview)
        synth_sticky_layout.addWidget(self.synthetic_temperature, 0, 1)
        
        synth_sticky_layout.addWidget(QLabel("Sticky Rounds:"), 1, 0)
        self.synthetic_sticky_rounds = QSpinBox()
        self.synthetic_sticky_rounds.setRange(1, 50)
        self.synthetic_sticky_rounds.setValue(30)  # Match train_run.py
        self.synthetic_sticky_rounds.valueChanged.connect(self.update_corruption_preview)
        synth_sticky_layout.addWidget(self.synthetic_sticky_rounds, 1, 1)
        
        synth_sticky_layout.addWidget(QLabel("P1-P2 Multiplier:"), 2, 0)
        self.synthetic_p1_p2_multiplier = QDoubleSpinBox()
        self.synthetic_p1_p2_multiplier.setRange(1.0, 50.0)
        self.synthetic_p1_p2_multiplier.setValue(20.0)  # Match train_run.py
        self.synthetic_p1_p2_multiplier.valueChanged.connect(self.update_corruption_preview)
        synth_sticky_layout.addWidget(self.synthetic_p1_p2_multiplier, 2, 1)
        
        synth_sticky_layout.addWidget(QLabel("P1 Divisor:"), 3, 0)
        self.synthetic_p1_divisor = QDoubleSpinBox()
        self.synthetic_p1_divisor.setRange(1.0, 20.0)
        self.synthetic_p1_divisor.setValue(3.0)  # Match train_run.py
        self.synthetic_p1_divisor.valueChanged.connect(self.update_corruption_preview)
        synth_sticky_layout.addWidget(self.synthetic_p1_divisor, 3, 1)
        
        synth_sticky_group.setLayout(synth_sticky_layout)
        layout.addWidget(synth_sticky_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.corruption_tabs.addTab(tab, "Synthetic")
    
    def create_mixed_corruption_tab(self):
        """Create tab for mixed corruption strategy weights"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Info about mixed strategy
        info_label = QLabel("Mixed strategy randomly selects from different corruption methods based on weights.\nWeights should sum to 1.0.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(info_label)
        
        # Strategy weights
        weights_group = QGroupBox("Strategy Weights")
        weights_layout = QGridLayout()
        
        weights_layout.addWidget(QLabel("Random:"), 0, 0)
        self.weight_random = QDoubleSpinBox()
        self.weight_random.setRange(0.0, 1.0)
        self.weight_random.setSingleStep(0.05)
        self.weight_random.setValue(0.25)
        self.weight_random.valueChanged.connect(self.update_corruption_preview)
        weights_layout.addWidget(self.weight_random, 0, 1)
        
        weights_layout.addWidget(QLabel("Sticky:"), 1, 0)
        self.weight_sticky = QDoubleSpinBox()
        self.weight_sticky.setRange(0.0, 1.0)
        self.weight_sticky.setSingleStep(0.05)
        self.weight_sticky.setValue(0.4)
        self.weight_sticky.valueChanged.connect(self.update_corruption_preview)
        weights_layout.addWidget(self.weight_sticky, 1, 1)
        
        weights_layout.addWidget(QLabel("Fragment:"), 2, 0)
        self.weight_fragment = QDoubleSpinBox()
        self.weight_fragment.setRange(0.0, 1.0)
        self.weight_fragment.setSingleStep(0.05)
        self.weight_fragment.setValue(0.25)
        self.weight_fragment.valueChanged.connect(self.update_corruption_preview)
        weights_layout.addWidget(self.weight_fragment, 2, 1)
        
        weights_layout.addWidget(QLabel("Synthetic:"), 3, 0)
        self.weight_synthetic = QDoubleSpinBox()
        self.weight_synthetic.setRange(0.0, 1.0)
        self.weight_synthetic.setSingleStep(0.05)
        self.weight_synthetic.setValue(0.1)
        self.weight_synthetic.valueChanged.connect(self.update_corruption_preview)
        weights_layout.addWidget(self.weight_synthetic, 3, 1)
        
        # Normalize button
        normalize_btn = QPushButton("Normalize Weights")
        normalize_btn.clicked.connect(self.normalize_weights)
        weights_layout.addWidget(normalize_btn, 4, 0, 1, 2)
        
        weights_group.setLayout(weights_layout)
        layout.addWidget(weights_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.corruption_tabs.addTab(tab, "Mixed")
        
    def create_visualization_panel(self):
        """Create the visualization panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Original text tab
        self.original_viz = TextVisualizer()
        self.tab_widget.addTab(self.original_viz, "Original Text")
        
        # Corrupted text tab
        self.corrupted_viz = TextVisualizer()
        self.tab_widget.addTab(self.corrupted_viz, "Corrupted Text")
        
        # Predictions tab
        self.predictions_viz = TextVisualizer()
        self.tab_widget.addTab(self.predictions_viz, "Model Predictions")
        
        # Remasking tab
        self.remasking_viz = TextVisualizer()
        self.tab_widget.addTab(self.remasking_viz, "Remasking Assessment")
        
        layout.addWidget(self.tab_widget)
        panel.setLayout(layout)
        return panel
        
    def load_model(self, model_path):
        """Load a model in background thread"""
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found: {model_path}")
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.model_loader = ModelLoader(model_path, self.device)
        self.model_loader.progress.connect(self.progress_bar.setValue)
        self.model_loader.finished_signal.connect(self.on_model_loaded)
        self.model_loader.error_signal.connect(self.on_model_error)
        self.model_loader.start()
    
    def refresh_model_list(self):
        """Refresh the list of available models"""
        self.model_list.clear()
        out_dir = Path("../out")
        if out_dir.exists():
            model_files = list(out_dir.glob("*.pt"))
            for model_file in sorted(model_files):
                self.model_list.addItem(model_file.name)
    
    def load_selected_model(self):
        """Load the currently selected model"""
        current_item = self.model_list.currentItem()
        if current_item is None:
            QMessageBox.warning(self, "Warning", "Please select a model to load")
            return
        
        model_name = current_item.text()
        model_path = f"../out/{model_name}"
        self.load_model(model_path)
        
    def on_model_loaded(self, model_name, model):
        """Handle successful model loading"""
        self.models[model_name] = model
        self.progress_bar.setVisible(False)
        
        # Update model combo box
        if model_name not in [self.model_combo.itemText(i) for i in range(1, self.model_combo.count())]:
            self.model_combo.addItem(model_name)
        
        # Update remasking model combo box 
        if model_name not in [self.remasking_model_combo.itemText(i) for i in range(1, self.remasking_model_combo.count())]:
            self.remasking_model_combo.addItem(model_name)
        
        # Update synthetic model combo box
        self.update_synthetic_model_combo()
        
        # Update status
        model_names = list(self.models.keys())
        self.model_status.setText(f"Loaded: {', '.join(model_names)}")
        self.model_status.setStyleSheet("color: #90EE90;")
        
        # Update button states
        self.update_remasking_buttons()
        
        self.statusBar().showMessage(f"Successfully loaded {model_name}")
        
    def on_model_error(self, error_msg):
        """Handle model loading error"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Model Loading Error", error_msg)
        
    def generate_sample(self):
        """Generate a new text sample"""
        if self.data is None:
            return
            
        try:
            # Get random starting position
            if len(self.valid_indices) > 0:
                start_idx = np.random.choice(self.valid_indices)
            else:
                start_idx = np.random.randint(0, len(self.data) - self.length_spinbox.value())
                
            # Extract sample
            sample_length = self.length_spinbox.value()
            self.current_sample = self.data[start_idx:start_idx + sample_length].astype(np.int64)
            
            # Display original text
            self.original_viz.display_text(self.current_sample, self.vocab, title="Original Text")
            
            # Enable corruption button
            self.apply_corruption_btn.setEnabled(True)
            self.predict_btn.setEnabled(len(self.models) > 0)
            
            # Update preview
            self.update_corruption_preview()
            
            self.statusBar().showMessage(f"Generated sample with {len(self.current_sample)} characters")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate sample: {str(e)}")
    
    def on_strategy_changed(self):
        """Handle strategy selection change"""
        strategy = self.strategy_combo.currentText()
        
        # Switch to appropriate tab
        if strategy == "random":
            self.corruption_tabs.setCurrentIndex(0)
        elif strategy == "sticky":
            self.corruption_tabs.setCurrentIndex(1)
        elif strategy == "fragment":
            self.corruption_tabs.setCurrentIndex(2)
        elif strategy == "synthetic":
            self.corruption_tabs.setCurrentIndex(3)
        elif strategy == "mixed":
            self.corruption_tabs.setCurrentIndex(4)
            
        # Update synthetic model combo
        if strategy == "synthetic":
            self.update_synthetic_model_combo()
            
        self.update_corruption_preview()
    
    def update_synthetic_model_combo(self):
        """Update synthetic model combo with available models"""
        self.synthetic_model_combo.clear()
        self.synthetic_model_combo.addItem("No synthetic model loaded...")
        for model_name in self.models.keys():
            self.synthetic_model_combo.addItem(model_name)
    
    def normalize_weights(self):
        """Normalize strategy weights to sum to 1.0"""
        total = (self.weight_random.value() + 
                self.weight_sticky.value() + 
                self.weight_fragment.value() + 
                self.weight_synthetic.value())
        
        if total > 0:
            self.weight_random.setValue(self.weight_random.value() / total)
            self.weight_sticky.setValue(self.weight_sticky.value() / total)
            self.weight_fragment.setValue(self.weight_fragment.value() / total)
            self.weight_synthetic.setValue(self.weight_synthetic.value() / total)
            
    def update_corruption_preview(self):
        """Update corruption preview when settings change"""
        if self.current_sample is None:
            return
            
        try:
            # Apply corruption with current settings
            corrupted_sample, mask = self.get_corrupted_sample()
            
            # Display corrupted text
            self.corrupted_viz.display_text(corrupted_sample, self.vocab, mask, title="Corrupted Text (Preview)")
            
        except Exception as e:
            print(f"Preview update error: {e}")
            
    def get_corrupted_sample(self):
        """Get corrupted sample based on current settings"""
        if self.current_sample is None:
            return None, None
            
        # Convert to tensor
        x = torch.from_numpy(self.current_sample).unsqueeze(0).to(self.device)
        
        strategy = self.strategy_combo.currentText()
        
        try:
            if strategy == "random":
                # Use random corruption with UI parameters
                corrupted_x, mask = apply_random_corruption_gpu(
                    x, 
                    iter_num=self.training_ctx.iter_num,
                    guaranteed_unmasked_max=self.guaranteed_unmasked_max.value(),
                    guaranteed_unmasked_min=self.guaranteed_unmasked_min.value(),
                    sticky_transition_start=self.sticky_transition_start.value(),
                    sticky_transition_end=self.sticky_transition_end.value(),
                    meta_vocab_size=self.vocab_size,
                    random_mask_warmup=self.random_mask_warmup.value()
                )
                
            elif strategy == "sticky":
                # Use sticky corruption with UI parameters
                corrupted_x, mask = apply_sticky_corruption_gpu(
                    x,
                    sticky_rounds=self.sticky_rounds.value(),
                    sticky_p1_p2_multiplier=self.sticky_p1_p2_multiplier.value(),
                    mask_token_id=self.training_ctx.mask_token_id,
                    meta_vocab_size=self.vocab_size,
                    sticky_p1_divisor=self.sticky_p1_divisor.value()
                )
                
            elif strategy == "fragment":
                # Use fragment corruption with UI parameters
                if self.data is not None:
                    corrupted_x, mask = apply_fragment_corruption_gpu(
                        x,
                        data=self.data,
                        block_size=len(self.current_sample),
                        sticky_rounds=self.fragment_sticky_rounds.value(),
                        sticky_p1_p2_multiplier=self.fragment_p1_p2_multiplier.value(),
                        mask_token_id=self.training_ctx.mask_token_id,
                        sticky_p1_divisor=self.fragment_p1_divisor.value()
                    )
                else:
                    # Fallback to sticky if no data available
                    corrupted_x, mask = apply_sticky_corruption_gpu(
                        x,
                        sticky_rounds=self.fragment_sticky_rounds.value(),
                        sticky_p1_p2_multiplier=self.fragment_p1_p2_multiplier.value(),
                        mask_token_id=self.training_ctx.mask_token_id,
                        meta_vocab_size=self.vocab_size,
                        sticky_p1_divisor=self.fragment_p1_divisor.value()
                    )
                    
            elif strategy == "synthetic":
                # Use synthetic corruption with UI parameters
                # For now, fallback to sticky since synthetic requires loaded model
                corrupted_x, mask = apply_sticky_corruption_gpu(
                    x,
                    sticky_rounds=self.synthetic_sticky_rounds.value(),
                    sticky_p1_p2_multiplier=self.synthetic_p1_p2_multiplier.value(),
                    mask_token_id=self.training_ctx.mask_token_id,
                    meta_vocab_size=self.vocab_size,
                    sticky_p1_divisor=self.synthetic_p1_divisor.value()
                )
                
            elif strategy == "mixed":
                # Use mixed strategy - randomly select based on weights
                import random
                weights = [
                    self.weight_random.value(),
                    self.weight_sticky.value(), 
                    self.weight_fragment.value(),
                    self.weight_synthetic.value()
                ]
                strategies = ["random", "sticky", "fragment", "synthetic"]
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [0.25, 0.25, 0.25, 0.25]
                    
                chosen_strategy = random.choices(strategies, weights=weights)[0]
                
                # Call specific corruption method directly without changing UI
                if chosen_strategy == "random":
                    corrupted_x, mask = apply_random_corruption_gpu(
                        x, 
                        iter_num=self.training_ctx.iter_num,
                        guaranteed_unmasked_max=self.guaranteed_unmasked_max.value(),
                        guaranteed_unmasked_min=self.guaranteed_unmasked_min.value(),
                        sticky_transition_start=self.sticky_transition_start.value(),
                        sticky_transition_end=self.sticky_transition_end.value(),
                        meta_vocab_size=self.vocab_size,
                        random_mask_warmup=self.random_mask_warmup.value()
                    )
                elif chosen_strategy == "sticky":
                    corrupted_x, mask = apply_sticky_corruption_gpu(
                        x,
                        sticky_rounds=self.sticky_rounds.value(),
                        sticky_p1_p2_multiplier=self.sticky_p1_p2_multiplier.value(),
                        mask_token_id=self.training_ctx.mask_token_id,
                        meta_vocab_size=self.vocab_size,
                        sticky_p1_divisor=self.sticky_p1_divisor.value()
                    )
                elif chosen_strategy == "fragment":
                    if self.data is not None:
                        corrupted_x, mask = apply_fragment_corruption_gpu(
                            x,
                            data=self.data,
                            block_size=len(self.current_sample),
                            sticky_rounds=self.fragment_sticky_rounds.value(),
                            sticky_p1_p2_multiplier=self.fragment_p1_p2_multiplier.value(),
                            mask_token_id=self.training_ctx.mask_token_id,
                            sticky_p1_divisor=self.fragment_p1_divisor.value()
                        )
                    else:
                        # Fallback to sticky
                        corrupted_x, mask = apply_sticky_corruption_gpu(
                            x,
                            sticky_rounds=self.fragment_sticky_rounds.value(),
                            sticky_p1_p2_multiplier=self.fragment_p1_p2_multiplier.value(),
                            mask_token_id=self.training_ctx.mask_token_id,
                            meta_vocab_size=self.vocab_size,
                            sticky_p1_divisor=self.fragment_p1_divisor.value()
                        )
                else:  # synthetic
                    corrupted_x, mask = apply_sticky_corruption_gpu(
                        x,
                        sticky_rounds=self.synthetic_sticky_rounds.value(),
                        sticky_p1_p2_multiplier=self.synthetic_p1_p2_multiplier.value(),
                        mask_token_id=self.training_ctx.mask_token_id,
                        meta_vocab_size=self.vocab_size,
                        sticky_p1_divisor=self.synthetic_p1_divisor.value()
                    )
                
            else:
                # Default fallback
                corrupted_x, mask = apply_random_corruption_gpu(
                    x, 0, 0.5, 0.0, 500, 15000, self.vocab_size, 8000
                )
                
            return corrupted_x.squeeze(0).cpu().numpy(), mask.squeeze(0).cpu().numpy()
            
        except Exception as e:
            print(f"Corruption error: {e}")
            # Return uncorrupted as fallback
            return self.current_sample, np.zeros(len(self.current_sample), dtype=bool)
        
    def apply_corruption(self):
        """Apply corruption to the current sample"""
        if self.current_sample is None:
            return
            
        try:
            corrupted_sample, mask = self.get_corrupted_sample()
            
            # Store for predictions
            self.current_corrupted = corrupted_sample
            self.corruption_mask = mask
            
            # Display
            self.corrupted_viz.display_text(corrupted_sample, self.vocab, mask, title="Corrupted Text")
            
            # Switch to corrupted tab
            self.tab_widget.setCurrentIndex(1)
            
            num_corrupted = np.sum(mask)
            total_tokens = len(mask)
            self.statusBar().showMessage(f"Corrupted {num_corrupted}/{total_tokens} tokens ({num_corrupted/total_tokens*100:.1f}%)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply corruption: {str(e)}")
            
    def generate_predictions(self):
        """Generate model predictions for corrupted text"""
        if self.current_corrupted is None or self.model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Warning", "Please apply corruption and select a model first")
            return
            
        try:
            model_name = self.model_combo.currentText()
            model = self.models[model_name]
            
            # Prepare input
            x = torch.from_numpy(self.current_corrupted).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits, _ = model(x, None)
                predictions = torch.argmax(logits, dim=-1)
                
            predictions_np = predictions.squeeze(0).cpu().numpy()
            
            # Store predictions
            self.current_predictions = predictions_np
            
            # Display predictions vs original
            self.predictions_viz.display_text(
                self.current_sample, self.vocab, self.corruption_mask, 
                predictions_np, f"Predictions from {model_name}"
            )
            
            # Calculate accuracy on corrupted positions
            if hasattr(self, 'corruption_mask'):
                mask = self.corruption_mask
                correct_predictions = predictions_np[mask] == self.current_sample[mask]
                accuracy = np.mean(correct_predictions) * 100
                
                self.statusBar().showMessage(f"Accuracy on corrupted tokens: {accuracy:.1f}%")
            
            # Switch to predictions tab
            self.tab_widget.setCurrentIndex(2)
            
            # Enable remasking assess button if remasking model is selected
            self.update_remasking_buttons()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate predictions: {str(e)}")
    
    def update_remasking_buttons(self):
        """Update remasking button states based on current conditions"""
        has_predictions = self.current_predictions is not None
        has_remasking_model = self.remasking_model_combo.currentIndex() > 0
        
        # Assess button: need both predictions and remasking model
        self.assess_btn.setEnabled(has_predictions and has_remasking_model)
        
        # Apply button: enabled after successful assessment
        # (this will be enabled in assess_tokens method)
    
    def assess_tokens(self):
        """Assess token probabilities using remasking model"""
        if self.current_predictions is None or self.remasking_model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Warning", "Please generate predictions and select a remasking model first")
            return
            
        try:
            remasking_model_name = self.remasking_model_combo.currentText()
            remasking_model = self.models[remasking_model_name]
            
            # Use predictions from third tab as input to assess their correctness
            x = torch.from_numpy(self.current_predictions).unsqueeze(0).to(self.device)
            
            # Get token probabilities from remasking model
            with torch.no_grad():
                logits, _ = remasking_model(x, None)
                # Debug info
                print(f"Logits shape: {logits.shape}")
                print(f"Vocab size: {self.vocab_size}")
                print(f"Model vocab size: {logits.shape[-1]}")
                
                # For remasking models, we want probabilities of tokens being wrong
                # Assuming the model outputs probabilities for each token being correct/incorrect
                probs = torch.softmax(logits, dim=-1)
                
                # Get probability of each token being wrong (assume last class is "wrong" indicator)
                if logits.shape[-1] > self.vocab_size:
                    # If model has extended vocab (with wrong indicators), use that
                    print("Using extended vocab for wrong probability")
                    wrong_probs = probs[:, :, -1]  # Last class is typically wrong indicator
                else:
                    # Otherwise, use 1 - max probability as wrongness indicator
                    print("Using 1 - max_prob as wrong probability")
                    max_probs = torch.max(probs, dim=-1)[0]
                    wrong_probs = 1.0 - max_probs
                    
            probabilities = wrong_probs.squeeze(0).cpu().numpy()
            
            # Store probability data
            self.token_probabilities = probabilities
            self.min_probability = float(np.min(probabilities))
            self.max_probability = float(np.max(probabilities))
            
            # Update probability info display
            self.prob_info_label.setText(f"Min/Max Probability: {self.min_probability:.3f} / {self.max_probability:.3f}")
            
            # Display in remasking tab with probability coloring
            self.remasking_viz.display_text(
                self.current_predictions, self.vocab, 
                title=f"Remasking Assessment from {remasking_model_name}",
                probabilities=probabilities, 
                min_prob=self.min_probability, 
                max_prob=self.max_probability
            )
            
            # Switch to remasking tab
            self.tab_widget.setCurrentIndex(3)
            
            # Enable apply button
            self.apply_remasking_btn.setEnabled(True)
            
            self.statusBar().showMessage(f"Assessment complete - Prob range: {self.min_probability:.3f} to {self.max_probability:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to assess tokens: {str(e)}")
    
    def apply_remasking(self):
        """Apply remasking based on probability assessment"""
        if self.token_probabilities is None:
            QMessageBox.warning(self, "Warning", "Please run assessment first")
            return
            
        try:
            # For now, this could apply threshold-based remasking
            # This is a placeholder for future functionality
            QMessageBox.information(self, "Apply Remasking", 
                                  "Apply remasking functionality would be implemented here.\n"
                                  f"Current assessment shows {len(self.token_probabilities)} tokens\n"
                                  f"with probability range {self.min_probability:.3f} to {self.max_probability:.3f}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply remasking: {str(e)}")


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Diffusion Model Explorer")
    
    # Check if PyQt6 is available
    try:
        window = ModelExplorerApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Make sure PyQt6 is installed: pip install PyQt6")
        sys.exit(1)


if __name__ == "__main__":
    main()