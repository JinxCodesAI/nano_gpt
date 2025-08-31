"""
Inference Process Visualizer for Diffusion Models
Visualize and edit the step-by-step diffusion inference process with intelligent remasking.
"""

import sys
import os
import pickle
import math
import numpy as np
import torch
from pathlib import Path
from contextlib import nullcontext

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QTextEdit, QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QGridLayout, QSplitter, QMessageBox, QComboBox,
    QScrollArea, QFrame, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCharFormat, QTextCursor, QColor

from model import GPT, GPTConfig
import diffusion_utils
from diffusion_utils import linear_remasking_schedule, apply_remasking


class InferenceRunner(QThread):
    """Background thread for running diffusion inference"""
    substep_complete = pyqtSignal(int, str, object, bool)  # step_num, substep_type, tokens, is_editable
    progress_update = pyqtSignal(int, str)  # progress, status
    error_signal = pyqtSignal(str)
    generation_complete = pyqtSignal()
    
    def __init__(self, model, settings, vocab, device, start_step=0, start_tokens=None, protected_positions=None):
        super().__init__()
        self.model = model
        self.settings = settings
        self.vocab = vocab
        self.device = device
        self.start_step = start_step
        self.start_tokens = start_tokens
        self.should_stop = False
        self.protected_positions = protected_positions or set()
        
    def run(self):
        try:
            # Initialize tokens
            if self.start_tokens is not None:
                tokens = self.start_tokens.clone()
            else:
                # Start with all positions masked
                mask_token_id = len(self.vocab)  # Use vocab_size as mask token
                tokens = torch.full(
                    (1, self.settings['sequence_length']), 
                    mask_token_id, 
                    dtype=torch.long, 
                    device=self.device
                )
            
            # Store step 0 (initial state)
            if self.start_step == 0:
                self.substep_complete.emit(0, "initial", tokens.clone(), True)
            
            # Run diffusion iterations starting from start_step
            for iteration in range(self.start_step, self.settings['diffusion_iterations']):
                if self.should_stop:
                    break
                    
                self.progress_update.emit(
                    int((iteration - self.start_step) / (self.settings['diffusion_iterations'] - self.start_step) * 100),
                    f"Step {iteration + 1}/{self.settings['diffusion_iterations']}"
                )
                
                # Step 1: Predict tokens for all masked positions
                mask_token_id = len(self.vocab)
                masked_positions = (tokens == mask_token_id)
                total_masked = masked_positions.sum().item()
                
                if total_masked > 0:
                    with torch.no_grad():
                        dummy_targets = torch.zeros_like(tokens)
                        logits, _ = self.model(tokens, dummy_targets)
                        
                        # Sample new tokens for masked positions
                        sample_masked = masked_positions[0]
                        if sample_masked.sum() > 0:
                            mask_indices = torch.where(sample_masked)[0]
                            masked_logits = logits[0, mask_indices]
                            
                            # Apply temperature
                            temperature = self.settings['temperature']
                            if temperature != 1.0:
                                masked_logits = masked_logits / temperature
                            
                            probs = torch.softmax(masked_logits, dim=-1)
                            new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                            tokens[0, mask_indices] = new_tokens
                
                # Emit ACTUAL prediction results (after prediction, before remasking)
                self.substep_complete.emit(iteration + 1, "prediction", tokens.clone(), False)
                
                # Step 2: Remask tokens for next iteration (except final iteration)
                if iteration < self.settings['diffusion_iterations'] - 1:
                    # Set global variables for the function
                    diffusion_utils.start_ratio = self.settings['start_ratio']
                    diffusion_utils.end_ratio = self.settings['end_ratio']
                    
                    remask_ratio = linear_remasking_schedule(
                        self.settings['diffusion_iterations'], 
                        iteration + 1
                    )
                    tokens = apply_remasking(
                        tokens, remask_ratio, None, self.settings['randomness_strength'],
                        mask_token_id, self.device, self.model, True, True, self.protected_positions  # intelligent_remasking=True, verbose=True, protected_positions
                    )
                    
                    # Emit remasking substep (editable)
                    self.substep_complete.emit(iteration + 1, "remasking", tokens.clone(), True)
                else:
                    # Final iteration - no remasking, just emit final result as editable
                    self.substep_complete.emit(iteration + 1, "final", tokens.clone(), True)
            
            self.generation_complete.emit()
            
        except Exception as e:
            self.error_signal.emit(f"Generation error: {str(e)}")
    
    def stop(self):
        self.should_stop = True
    
    def update_protected_positions(self, protected_positions):
        """Update the protected positions during generation"""
        self.protected_positions = protected_positions or set()
        print(f"DEBUG: InferenceRunner - updated protected positions: {sorted(self.protected_positions)}")


class EditableTextWidget(QTextEdit):
    """Text widget that allows editing with mask/unmask functionality"""
    text_edited = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setFont(QFont("Consolas", 12))
        self.vocab = None
        self.mask_token_id = None
        self.current_tokens = None
        self.is_editable = True
        
        # Note: No longer connecting to textChanged - we save only on navigation
        
    def set_vocab(self, vocab, mask_token_id):
        self.vocab = vocab
        self.mask_token_id = mask_token_id
        
    def display_tokens(self, tokens, show_mask_char=True, is_editable=True, protected_positions=None):
        """Display tokens with proper formatting and coloring"""
        self.current_tokens = tokens.clone() if isinstance(tokens, torch.Tensor) else torch.tensor(tokens)
        self.is_editable = is_editable
        self.setReadOnly(not is_editable)
        
        if protected_positions is None:
            protected_positions = set()
        
        html = "<div style='font-family: Consolas, monospace; font-size: 12px;'>"
        
        for i, token_id in enumerate(tokens.flatten()):
            is_protected = i in protected_positions
            
            if token_id == self.mask_token_id:
                if is_protected:
                    # Protected mask (should not happen normally, but handle it)
                    char = "█" if show_mask_char else "_"
                    style = "background-color: #90EE90; color: black; border: 2px solid #228B22;"
                elif show_mask_char:
                    char = "█"  # Block character for mask
                    style = "background-color: #87CEEB; color: black;"
                else:
                    char = "_"
                    style = "background-color: #FFE4E1; color: black;"
            else:
                char = self.vocab.get(int(token_id), f"[UNK{token_id}]")
                if is_protected:
                    # Protected text - green background with border
                    style = "background-color: #90EE90; color: black; border: 2px solid #228B22;"
                else:
                    style = ""
                
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
                
            html += f"<span style='{style}'>{char}</span>"
        
        html += "</div>"
        self.setHtml(html)
    
    def get_edited_tokens(self):
        """Convert current displayed text back to tokens"""
        print(f"DEBUG: get_edited_tokens - vocab: {type(self.vocab)}, mask_token_id: {self.mask_token_id}")
        
        if not self.vocab or not self.mask_token_id:
            print("DEBUG: get_edited_tokens - falling back to current_tokens")
            return self.current_tokens
        
        # Get the plain text content (not HTML)
        plain_text = self.toPlainText()
        
        # Convert string/char vocab to integer mapping if needed
        if isinstance(list(self.vocab.keys())[0], str):
            # This is a char-level vocab like {'a': 0, 'b': 1, ...}
            char_to_id = self.vocab
        else:
            # This might be id-to-char, flip it
            char_to_id = {v: k for k, v in self.vocab.items() if isinstance(v, str)}
        
        tokens = []
        for char in plain_text:
            if char == '█':  # Mask character
                tokens.append(self.mask_token_id)
            elif char in char_to_id:
                tokens.append(char_to_id[char])
            else:
                # Unknown character - mask it
                tokens.append(self.mask_token_id)
        
        # Ensure correct sequence length - pad with masks or trim
        expected_length = len(self.current_tokens)
        
        if len(tokens) < expected_length:
            # Pad with mask tokens at the end
            tokens.extend([self.mask_token_id] * (expected_length - len(tokens)))
        elif len(tokens) > expected_length:
            # Trim to expected length
            tokens = tokens[:expected_length]
            
        result = torch.tensor(tokens, dtype=torch.long)
        return result
        
    def mask_selected_text(self):
        """Replace selected text with mask characters"""
        cursor = self.textCursor()
        if cursor.hasSelection():
            # Replace selected text with mask characters
            selected_text = cursor.selectedText()
            mask_text = '█' * len(selected_text)
            cursor.insertText(mask_text)
    
    def update_display_for_protection(self, protected_positions):
        """Update visual display to show protected positions with special styling"""
        if not self.vocab or self.current_tokens is None or len(self.current_tokens) == 0:
            return
            
        # Re-display tokens with protection indicators
        tokens = self.current_tokens
        html = "<div style='font-family: Consolas, monospace; font-size: 12px;'>"
        
        for i, token_id in enumerate(tokens.flatten()):
            is_protected = i in protected_positions
            
            if token_id == self.mask_token_id:
                if is_protected:
                    # Protected mask (should not happen normally, but handle it)
                    char = "█"
                    style = "background-color: #90EE90; color: black; border: 2px solid #228B22;"
                else:
                    char = "█"  # Block character for mask
                    style = "background-color: #87CEEB; color: black;"
            else:
                char = self.vocab.get(int(token_id), f"[UNK{token_id}]")
                if is_protected:
                    # Protected text - green background with border
                    style = "background-color: #90EE90; color: black; border: 2px solid #228B22;"
                else:
                    style = ""
                
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
                
            html += f"<span style='{style}'>{char}</span>"
        
        html += "</div>"
        self.setHtml(html)
        


class MainPanel(QWidget):
    """Main panel showing the current step content with editing capabilities"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header with step info
        self.header_label = QLabel("Step 0 / 100")
        self.header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.header_label)
        
        # Substep info
        self.substep_label = QLabel("Initial State")
        self.substep_label.setFont(QFont("Arial", 12))
        self.substep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.substep_label.setStyleSheet("color: #888888;")
        layout.addWidget(self.substep_label)
        
        # Token statistics
        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Arial", 10))
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet("color: #CCCCCC; background-color: #2b2b2b; padding: 5px;")
        layout.addWidget(self.stats_label)
        
        # Text display/edit area
        self.text_widget = EditableTextWidget()
        layout.addWidget(self.text_widget)
        
        
        self.setLayout(layout)
        
    def update_substep(self, step_num, total_steps, substep_type, tokens, vocab, mask_token_id, is_editable, protected_positions=None):
        """Update display for a specific substep"""
        self.header_label.setText(f"Step {step_num} / {total_steps}")
        
        # Update substep label with appropriate styling
        substep_names = {
            "initial": "Initial State",
            "prediction": "After Prediction", 
            "remasking": "After Remasking",
            "final": "Final Result"
        }
        substep_text = substep_names.get(substep_type, substep_type)
        
        if is_editable:
            self.substep_label.setText(f"{substep_text} (Editable)")
            self.substep_label.setStyleSheet("color: #90EE90; font-weight: bold;")
        else:
            self.substep_label.setText(f"{substep_text} (Read-only)")
            self.substep_label.setStyleSheet("color: #FFB366; font-weight: bold;")
        
        self.text_widget.set_vocab(vocab, mask_token_id)
        self.text_widget.display_tokens(tokens, True, is_editable, protected_positions)
        
        # Calculate token statistics
        token_stats = self.calculate_token_stats(tokens, vocab, mask_token_id)
        self.stats_label.setText(token_stats)
        
        # Debug output to console with clear context - ONLY for new substeps
        if not hasattr(self, '_last_logged_substep') or self._last_logged_substep != f"{step_num}_{substep_type}":
            substep_display = substep_names.get(substep_type, substep_type)
            print(f"\n=== Step {step_num}/{total_steps} - {substep_display} ===")
            print(token_stats)
            if substep_type == "prediction":
                tokens_np = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else tokens
                mask_count = np.sum(tokens_np == mask_token_id)
                if mask_count > 0:
                    print(f"⚠️  Model predicted {mask_count} mask tokens during prediction!")
                    # Show first few positions where masks were predicted
                    mask_positions = np.where(tokens_np == mask_token_id)[0][:10]
                    print(f"Mask positions (first 10): {mask_positions}")
            print("=" * 50)
            self._last_logged_substep = f"{step_num}_{substep_type}"
        
    def calculate_token_stats(self, tokens, vocab, mask_token_id):
        """Calculate and format token statistics"""
        # Debug print to see what we're getting
        print(f"DEBUG: calculate_token_stats received tokens type: {type(tokens)}")
        print(f"DEBUG: tokens value: {tokens}")
        
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        elif isinstance(tokens, (list, tuple)):
            tokens = np.array(tokens)
        elif isinstance(tokens, np.ndarray):
            pass  # Already numpy array
        else:
            print(f"ERROR: Invalid tokens type: {type(tokens)}")
            return "Error: Invalid token data"
        
        total_tokens = len(tokens)
        mask_count = np.sum(tokens == mask_token_id)
        wrong_token_count = np.sum(tokens == mask_token_id + 1)
        remask_good_count = np.sum(tokens == mask_token_id + 2)
        remask_wrong_count = np.sum(tokens == mask_token_id + 3)
        
        # Count regular vocab tokens
        vocab_count = np.sum(tokens < len(vocab))
        unknown_count = np.sum(tokens >= mask_token_id + 4)
        
        stats = [
            f"Total: {total_tokens}",
            f"Vocab: {vocab_count}",
            f"Masks: {mask_count}",
        ]
        
        if wrong_token_count > 0:
            stats.append(f"Wrong: {wrong_token_count}")
        if remask_good_count > 0:
            stats.append(f"RemaskGood: {remask_good_count}")
        if remask_wrong_count > 0:
            stats.append(f"RemaskWrong: {remask_wrong_count}")
        if unknown_count > 0:
            stats.append(f"Unknown: {unknown_count}")
            
        # Add percentages for masks if significant
        if mask_count > 0:
            mask_pct = (mask_count / total_tokens) * 100
            stats.append(f"({mask_pct:.1f}% masks)")
            
        return " | ".join(stats)


class LeftPanel(QWidget):
    """Left panel containing all generation settings"""
    settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setMaximumWidth(300)
        layout = QVBoxLayout()
        
        # Model loading
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("Select model...")
        self.refresh_models()
        model_layout.addWidget(QLabel("Base Model:"))
        model_layout.addWidget(self.model_combo)
        
        self.load_btn = QPushButton("Load Model")
        model_layout.addWidget(self.load_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #888888;")
        model_layout.addWidget(self.model_status)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Generation settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QGridLayout()
        
        # Temperature
        gen_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(1.0)
        self.temperature_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.temperature_spin, 0, 1)
        
        # Diffusion iterations
        gen_layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 200)
        self.iterations_spin.setValue(20)
        self.iterations_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.iterations_spin, 1, 1)
        
        # Sequence length
        gen_layout.addWidget(QLabel("Length:"), 2, 0)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(50, 1024)
        self.length_spin.setValue(1024)
        self.length_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.length_spin, 2, 1)
        
        # Start ratio
        gen_layout.addWidget(QLabel("Start Ratio:"), 3, 0)
        self.start_ratio_spin = QDoubleSpinBox()
        self.start_ratio_spin.setRange(0.1, 1.0)
        self.start_ratio_spin.setSingleStep(0.05)
        self.start_ratio_spin.setValue(0.99)
        self.start_ratio_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.start_ratio_spin, 3, 1)
        
        # End ratio
        gen_layout.addWidget(QLabel("End Ratio:"), 4, 0)
        self.end_ratio_spin = QDoubleSpinBox()
        self.end_ratio_spin.setRange(0.0, 0.5)
        self.end_ratio_spin.setSingleStep(0.05)
        self.end_ratio_spin.setValue(0.15)
        self.end_ratio_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.end_ratio_spin, 4, 1)
        
        # Randomness strength
        gen_layout.addWidget(QLabel("Randomness:"), 5, 0)
        self.randomness_spin = QDoubleSpinBox()
        self.randomness_spin.setRange(0.0, 1.0)
        self.randomness_spin.setSingleStep(0.1)
        self.randomness_spin.setValue(0.8)
        self.randomness_spin.valueChanged.connect(self.settings_changed)
        gen_layout.addWidget(self.randomness_spin, 5, 1)
        
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Control buttons
        control_group = QGroupBox("Control")
        control_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Generation")
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Generation")
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.restart_btn = QPushButton("Restart from Current")
        self.restart_btn.setEnabled(False)
        control_layout.addWidget(self.restart_btn)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Edit controls
        edit_group = QGroupBox("Edit Controls")
        edit_layout = QVBoxLayout()
        
        self.mask_btn = QPushButton("Mask Selected")
        self.mask_btn.setEnabled(False)
        edit_layout.addWidget(self.mask_btn)
        
        # Protect edits checkbox
        self.protect_edits_checkbox = QCheckBox("Protect Edits")
        self.protect_edits_checkbox.setToolTip("When checked, manually edited characters will never be masked again")
        self.protect_edits_checkbox.setChecked(False)
        edit_layout.addWidget(self.protect_edits_checkbox)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def refresh_models(self):
        """Refresh the list of available models"""
        self.model_combo.clear()
        self.model_combo.addItem("Select model...")
        
        # Use relative path from current working directory (should be parent of visualization)
        out_dir = Path("out")
        if out_dir.exists():
            model_files = list(out_dir.glob("*.pt"))
            for model_file in sorted(model_files):
                self.model_combo.addItem(model_file.name)
                
    def get_settings(self):
        """Get current generation settings"""
        return {
            'temperature': self.temperature_spin.value(),
            'diffusion_iterations': self.iterations_spin.value(),
            'sequence_length': self.length_spin.value(),
            'start_ratio': self.start_ratio_spin.value(),
            'end_ratio': self.end_ratio_spin.value(),
            'randomness_strength': self.randomness_spin.value()
        }


class BottomPanel(QWidget):
    """Bottom panel with substep navigation controls"""
    substep_changed = pyqtSignal(str)  # substep_key (e.g., "1_prediction", "1_remasking")
    
    def __init__(self):
        super().__init__()
        self.total_steps = 100
        self.substep_keys = []  # List of available substep keys
        self.current_index = 0
        self.init_ui()
        
    def init_ui(self):
        self.setMaximumHeight(100)
        layout = QVBoxLayout()
        
        # Step slider
        slider_layout = QHBoxLayout()
        
        self.substep_slider = QSlider(Qt.Orientation.Horizontal)
        self.substep_slider.setRange(0, 0)
        self.substep_slider.setValue(0)
        self.substep_slider.valueChanged.connect(self.on_slider_changed)
        
        self.substep_label = QLabel("No steps")
        self.substep_label.setMinimumWidth(120)
        
        slider_layout.addWidget(QLabel("Substep:"))
        slider_layout.addWidget(self.substep_slider)
        slider_layout.addWidget(self.substep_label)
        
        layout.addLayout(slider_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.prev_step)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_step)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        self.setLayout(layout)
        
    def set_substep_keys(self, substep_keys):
        """Update the available substep keys"""
        self.substep_keys = substep_keys
        self.substep_slider.setRange(0, len(substep_keys) - 1 if substep_keys else 0)
        self.current_index = 0
        self.substep_slider.setValue(0)
        self.update_label()
        
    def set_current_substep(self, substep_key):
        """Set the current substep without emitting signal"""
        if substep_key in self.substep_keys:
            index = self.substep_keys.index(substep_key)
            self.current_index = index
            self.substep_slider.blockSignals(True)
            self.substep_slider.setValue(index)
            self.substep_slider.blockSignals(False)
            self.update_label()
        
    def on_slider_changed(self, value):
        """Handle slider value change"""
        self.current_index = value
        self.update_label()
        if 0 <= value < len(self.substep_keys):
            self.substep_changed.emit(self.substep_keys[value])
        
    def update_label(self):
        """Update the substep label"""
        if self.substep_keys:
            current_key = self.substep_keys[self.current_index] if 0 <= self.current_index < len(self.substep_keys) else "?"
            self.substep_label.setText(f"{current_key} ({self.current_index + 1}/{len(self.substep_keys)})")
        else:
            self.substep_label.setText("No substeps")
        
    def prev_step(self):
        """Go to previous substep"""
        if self.current_index > 0:
            self.substep_slider.setValue(self.current_index - 1)
            
    def next_step(self):
        """Go to next substep"""
        if self.current_index < len(self.substep_keys) - 1:
            self.substep_slider.setValue(self.current_index + 1)


class InferenceVisualizerApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.vocab_size = None
        self.generation_substeps = {}  # substep_key -> {tokens, substep_type, is_editable}
        self.current_substep_key = None
        self.inference_runner = None
        
        # Protected positions tracking for "protect edits" feature
        self.protected_positions = set()  # Set of sequence positions that are protected from remasking
        self.protect_edits_enabled = False  # Whether protection is currently enabled
        self.baseline_tokens = {}  # substep_key -> original tokens before user edits (for detecting changes)
        
        self.load_vocab()
        self.init_ui()
        
    def load_vocab(self):
        """Load vocabulary"""
        try:
            # Use relative path from current working directory (should be parent of visualization)
            meta_path = Path("data/shakespeare_char/meta.pkl")
            
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            
            self.vocab = meta['itos']
            self.vocab_size = meta['vocab_size']
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load vocabulary: {str(e)}\nLooked for: {meta_path}\nCurrent dir: {Path.cwd()}")
            
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Diffusion Inference Visualizer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
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
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #004499;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                min-height: 15px;
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
        
        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Top section with left panel and main panel
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel
        self.left_panel = LeftPanel()
        self.left_panel.load_btn.clicked.connect(self.load_model)
        self.left_panel.start_btn.clicked.connect(self.start_generation)
        self.left_panel.stop_btn.clicked.connect(self.stop_generation)
        self.left_panel.restart_btn.clicked.connect(self.restart_from_current)
        self.left_panel.mask_btn.clicked.connect(self.mask_selected)
        self.left_panel.protect_edits_checkbox.toggled.connect(self.on_protect_edits_toggled)
        top_splitter.addWidget(self.left_panel)
        
        # Main panel
        self.main_panel = MainPanel()
        # Note: No longer connecting text_edited - we save only on navigation
        top_splitter.addWidget(self.main_panel)
        
        # Set splitter proportions
        top_splitter.setSizes([300, 1100])
        
        main_layout.addWidget(top_splitter)
        
        # Bottom panel
        self.bottom_panel = BottomPanel()
        self.bottom_panel.substep_changed.connect(self.on_substep_changed)
        main_layout.addWidget(self.bottom_panel)
        
        main_widget.setLayout(main_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Load a model to begin")
        
        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
    def load_model(self):
        """Load the selected model"""
        if self.left_panel.model_combo.currentIndex() == 0:
            QMessageBox.warning(self, "Warning", "Please select a model to load")
            return
            
        model_name = self.left_panel.model_combo.currentText()
        model_path = Path("out") / model_name
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model
            model_args = checkpoint['model_args']
            if 'attention_type' not in model_args:
                model_args['attention_type'] = 'causal'
                
            model_config = GPTConfig(**model_args)
            self.model = GPT(model_config)
            
            # Load weights
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)
            
            # Update UI
            self.left_panel.model_status.setText(f"Loaded: {model_name}")
            self.left_panel.model_status.setStyleSheet("color: #90EE90;")
            self.left_panel.start_btn.setEnabled(True)
            
            self.statusBar().showMessage(f"Model loaded: {model_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
    def start_generation(self):
        """Start diffusion generation from scratch"""
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return
            
        # Clear previous substeps
        self.generation_substeps.clear()
        self.current_substep_key = None
        
        # Clear protected positions and baselines when starting new generation
        self.protected_positions.clear()
        self.baseline_tokens.clear()
        
        # Get settings
        settings = self.left_panel.get_settings()
        
        # Start inference runner
        self.inference_runner = InferenceRunner(
            self.model, settings, self.vocab, self.device, protected_positions=self.protected_positions
        )
        self.inference_runner.substep_complete.connect(self.on_substep_complete)
        self.inference_runner.progress_update.connect(self.on_progress_update)
        self.inference_runner.error_signal.connect(self.on_inference_error)
        self.inference_runner.generation_complete.connect(self.on_generation_complete)
        
        # Update UI state
        self.left_panel.start_btn.setEnabled(False)
        self.left_panel.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        
        self.inference_runner.start()
        
    def stop_generation(self):
        """Stop ongoing generation"""
        if self.inference_runner:
            self.inference_runner.stop()
            self.inference_runner.wait()  # Wait for thread to finish
            
        self.on_generation_complete()
        
    def restart_from_current(self):
        """Restart generation from current step"""
        if self.model is None or not self.current_substep_key or self.current_substep_key not in self.generation_substeps:
            QMessageBox.warning(self, "Warning", "No valid substep to restart from")
            return
            
        # Save any edits from current step if protection enabled
        if self.protect_edits_enabled:
            self.save_current_edits()
            
        # Get current substep tokens (potentially edited)
        current_tokens = self.main_panel.text_widget.get_edited_tokens()
        
        # Debug: Print what we got from edited tokens
        print(f"DEBUG: restart_from_current - edited tokens type: {type(current_tokens)}")
        print(f"DEBUG: restart_from_current - edited tokens shape: {current_tokens.shape if hasattr(current_tokens, 'shape') else 'no shape'}")
        print(f"DEBUG: restart_from_current - first 20 tokens: {current_tokens[:20] if hasattr(current_tokens, '__getitem__') else 'not indexable'}")
        
        # Extract step number from substep key (e.g., "3_remasking" -> 3)
        try:
            step_num = int(self.current_substep_key.split('_')[0])
            print(f"DEBUG: restart_from_current - starting from step: {step_num}")
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Warning", "Invalid substep key")
            return
        
        # Get settings
        settings = self.left_panel.get_settings()
        
        # Start inference runner from current step
        self.inference_runner = InferenceRunner(
            self.model, settings, self.vocab, self.device, 
            step_num, current_tokens.unsqueeze(0).to(self.device), self.protected_positions
        )
        self.inference_runner.substep_complete.connect(self.on_substep_complete)
        self.inference_runner.progress_update.connect(self.on_progress_update)
        self.inference_runner.error_signal.connect(self.on_inference_error)
        self.inference_runner.generation_complete.connect(self.on_generation_complete)
        
        # Update UI state
        self.left_panel.start_btn.setEnabled(False)
        self.left_panel.stop_btn.setEnabled(True)
        self.left_panel.restart_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.inference_runner.start()
        
    def on_substep_complete(self, step_num, substep_type, tokens, is_editable):
        """Handle completion of a generation substep"""
        substep_key = f"{step_num}_{substep_type}"
        self.generation_substeps[substep_key] = {
            'tokens': tokens.clone(),
            'substep_type': substep_type,
            'is_editable': is_editable,
            'step_num': step_num
        }
        
        # Store baseline tokens (original model output before user edits)
        # This is crucial for detecting user edits later
        self.baseline_tokens[substep_key] = tokens.clone()
        
        # Update bottom panel with new substep keys
        substep_keys = sorted(self.generation_substeps.keys(), key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
        self.bottom_panel.set_substep_keys(substep_keys)
        
        # If this is the latest substep, switch to it
        if not self.current_substep_key or self.should_show_latest_substep(substep_key):
            self.current_substep_key = substep_key
            self.bottom_panel.set_current_substep(substep_key)
            self.update_main_panel()
            
    def should_show_latest_substep(self, new_substep_key):
        """Determine if we should automatically switch to the new substep"""
        if not self.current_substep_key:
            return True
        
        # Extract step numbers
        try:
            new_step = int(new_substep_key.split('_')[0])
            current_step = int(self.current_substep_key.split('_')[0])
            return new_step >= current_step
        except (ValueError, IndexError):
            return True
            
    def on_substep_changed(self, substep_key):
        """Handle substep change from bottom panel"""
        # Before switching, save any edits from current step (ALWAYS, not just when protection enabled)
        if self.current_substep_key:
            self.save_current_edits()
        
        if substep_key in self.generation_substeps:
            self.current_substep_key = substep_key
            self.update_main_panel()
            
    def update_main_panel(self):
        """Update the main panel display"""
        if self.current_substep_key and self.current_substep_key in self.generation_substeps:
            substep_data = self.generation_substeps[self.current_substep_key]
            tokens = substep_data['tokens']
            total_steps = self.left_panel.get_settings()['diffusion_iterations']
            mask_token_id = len(self.vocab)
            
            self.main_panel.update_substep(
                substep_data['step_num'], total_steps, substep_data['substep_type'],
                tokens[0], self.vocab, mask_token_id, substep_data['is_editable'], self.protected_positions
            )
            
            # Update edit controls based on editability
            self.left_panel.mask_btn.setEnabled(substep_data['is_editable'])
            
    def on_progress_update(self, progress, status):
        """Handle progress update"""
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(status)
        
    def on_inference_error(self, error_msg):
        """Handle inference error"""
        QMessageBox.critical(self, "Generation Error", error_msg)
        self.on_generation_complete()
        
    def on_generation_complete(self):
        """Handle generation completion"""
        # Update UI state
        self.left_panel.start_btn.setEnabled(True)
        self.left_panel.stop_btn.setEnabled(False)
        self.left_panel.restart_btn.setEnabled(len(self.generation_substeps) > 0 and self.current_substep_key is not None)
        self.progress_bar.setVisible(False)
        
        self.statusBar().showMessage("Generation complete")
        
    def mask_selected(self):
        """Mask selected tokens in current substep"""
        if not self.current_substep_key or self.current_substep_key not in self.generation_substeps:
            return
            
        # Get selected text range from main panel
        selection = self.main_panel.text_widget.textCursor().selectedText()
        if not selection:
            QMessageBox.information(self, "Info", "Please select text to mask")
            return
            
        # Apply masking to selected text
        self.main_panel.text_widget.mask_selected_text()
        
        # Update the stored substep data with modified tokens
        substep_data = self.generation_substeps[self.current_substep_key]
        modified_tokens = self.main_panel.text_widget.get_edited_tokens()
        substep_data['tokens'] = modified_tokens.clone()
        self.generation_substeps[self.current_substep_key] = substep_data
        
    def on_protect_edits_toggled(self, checked):
        """Handle protect edits checkbox toggle"""
        self.protect_edits_enabled = checked
        if checked:
            print("Protect edits enabled - manually edited characters will be protected from remasking")
        else:
            print("Protect edits disabled - all positions can be remasked normally")
            # Clear current protections when disabled
            self.protected_positions.clear()
        
        # Update visual display to show/hide protection indicators
        if hasattr(self.main_panel, 'text_widget'):
            self.main_panel.text_widget.update_display_for_protection(self.protected_positions)
        
    def save_current_edits(self):
        """Save edits from current step and detect protected positions"""
        if not self.current_substep_key or self.current_substep_key not in self.generation_substeps:
            return
            
        # Get current substep data
        substep_data = self.generation_substeps[self.current_substep_key]
        
        # Only save edits for editable substeps
        if not substep_data['is_editable']:
            return
            
        print(f"DEBUG: save_current_edits - saving edits for {self.current_substep_key}")
            
        try:
            # Get edited tokens and ensure they match the original device/format
            modified_tokens = self.main_panel.text_widget.get_edited_tokens()
            original_tokens = substep_data['tokens']
            
            # Ensure same shape and device as original
            if len(modified_tokens.shape) == 1 and len(original_tokens.shape) == 2:
                modified_tokens = modified_tokens.unsqueeze(0)
            
            # Move to same device as original for saving
            modified_tokens = modified_tokens.to(original_tokens.device)
            
            # Save the edited tokens immediately (basic functionality)
            substep_data['tokens'] = modified_tokens
            self.generation_substeps[self.current_substep_key] = substep_data
            print(f"DEBUG: save_current_edits - basic save completed, shape: {modified_tokens.shape}")
            
            # Do protection detection if enabled (with simpler device handling)
            if self.protect_edits_enabled and self.current_substep_key in self.baseline_tokens:
                self.detect_and_protect_edits(modified_tokens)
                
        except Exception as e:
            print(f"DEBUG: save_current_edits - error: {e}")
            import traceback
            traceback.print_exc()
    
    def detect_and_protect_edits(self, modified_tokens):
        """Separate method for protection detection with robust device handling"""
        try:
            baseline_tokens = self.baseline_tokens[self.current_substep_key]
            
            # Move both to CPU for comparison
            baseline_cpu = baseline_tokens.cpu()
            modified_cpu = modified_tokens.cpu()
            
            # Ensure same shape
            if len(baseline_cpu.shape) != len(modified_cpu.shape):
                if len(baseline_cpu.shape) == 1:
                    baseline_cpu = baseline_cpu.unsqueeze(0)
                if len(modified_cpu.shape) == 1:
                    modified_cpu = modified_cpu.unsqueeze(0)
            
            mask_token_id = len(self.vocab) if self.vocab else 0
            
            # Find changed positions
            changed_positions = torch.where(
                (baseline_cpu[0] != modified_cpu[0]) &
                (modified_cpu[0] != mask_token_id)
            )[0]
            
            # Add to protected set
            new_protections = []
            for pos in changed_positions:
                pos_int = pos.item()
                if pos_int not in self.protected_positions:
                    self.protected_positions.add(pos_int)
                    new_protections.append(pos_int)
            
            if len(new_protections) > 0:
                print(f"DEBUG: Protected {len(new_protections)} positions: {new_protections}")
                
                # Update InferenceRunner
                if self.inference_runner and hasattr(self.inference_runner, 'update_protected_positions'):
                    self.inference_runner.update_protected_positions(self.protected_positions)
                
                # Update visual display
                self.main_panel.text_widget.update_display_for_protection(self.protected_positions)
                
        except Exception as e:
            print(f"DEBUG: detect_and_protect_edits - error: {e}")
    
    def on_text_edited(self):
        """Legacy method - no longer used since we save only on navigation"""
        pass


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Diffusion Inference Visualizer")
    
    try:
        window = InferenceVisualizerApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
