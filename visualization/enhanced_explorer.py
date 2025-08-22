"""
Enhanced Model Explorer with Advanced Features
- Model comparison
- Batch processing
- Export capabilities
- Advanced visualization options
"""

import sys
import os
import pickle
import csv
import json
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QTextEdit, QPushButton, QLabel, QComboBox, QSlider, QSpinBox,
    QGroupBox, QGridLayout, QSplitter, QScrollArea, QFrame,
    QProgressBar, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QCheckBox, QFileDialog, QTextBrowser, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QTextCharFormat, QTextCursor

# Import training utilities
from train_utils import (
    TrainingContext, apply_random_corruption_gpu, apply_sticky_corruption_gpu,
    apply_fragment_corruption_gpu, apply_synthetic_corruption,
    apply_gpu_masking_validation, apply_gpu_masking_training,
    find_double_newline_indices, load_synthetic_model
)
from model import GPT, GPTConfig


class BatchProcessor(QThread):
    """Background thread for batch processing multiple samples"""
    progress = pyqtSignal(int, str)
    results_ready = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def __init__(self, samples, models, corruption_settings, device):
        super().__init__()
        self.samples = samples
        self.models = models
        self.corruption_settings = corruption_settings
        self.device = device
        
    def run(self):
        try:
            results = []
            total_operations = len(self.samples) * len(self.models)
            current_op = 0
            
            for i, sample in enumerate(self.samples):
                self.progress.emit(int(current_op / total_operations * 100), f"Processing sample {i+1}/{len(self.samples)}")
                
                # Apply corruption
                x = torch.from_numpy(sample).unsqueeze(0).to(self.device)
                corrupted_x, mask = self.apply_corruption(x)
                
                sample_result = {
                    'sample_id': i,
                    'original': sample,
                    'corrupted': corrupted_x.squeeze(0).cpu().numpy(),
                    'mask': mask.squeeze(0).cpu().numpy(),
                    'predictions': {}
                }
                
                # Get predictions from each model
                for model_name, model in self.models.items():
                    with torch.no_grad():
                        logits, _ = model(corrupted_x, None)
                        predictions = torch.argmax(logits, dim=-1)
                        
                    predictions_np = predictions.squeeze(0).cpu().numpy()
                    
                    # Calculate accuracy on corrupted positions
                    mask_np = mask.squeeze(0).cpu().numpy()
                    if mask_np.any():
                        correct = predictions_np[mask_np] == sample[mask_np]
                        accuracy = np.mean(correct)
                    else:
                        accuracy = 0.0
                        
                    sample_result['predictions'][model_name] = {
                        'tokens': predictions_np,
                        'accuracy': accuracy
                    }
                    
                    current_op += 1
                    self.progress.emit(int(current_op / total_operations * 100), 
                                     f"Model {model_name} on sample {i+1}")
                
                results.append(sample_result)
                
            self.results_ready.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f"Batch processing error: {str(e)}")
            
    def apply_corruption(self, x):
        """Apply corruption based on settings"""
        strategy = self.corruption_settings['strategy']
        ratio = self.corruption_settings['ratio']
        
        if strategy == 'random':
            return apply_random_corruption_gpu(x, ratio, 0.0, self.corruption_settings['vocab_size'])
        else:
            # Default to random for batch processing
            return apply_random_corruption_gpu(x, ratio, 0.0, self.corruption_settings['vocab_size'])


class ComparisonTable(QWidget):
    """Widget for displaying model comparison results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Model Comparison Results")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Results table
        self.table = QTableWidget()
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #3c3c3c;
                gridline-color: #555555;
            }
            QTableWidget::item {
                padding: 8px;
                border: 1px solid #555555;
            }
            QTableWidget::item:selected {
                background-color: #0084ff;
            }
        """)
        layout.addWidget(self.table)
        
        # Export button
        export_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export Results to CSV")
        self.export_btn.clicked.connect(self.export_results)
        export_layout.addWidget(self.export_btn)
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
        self.setLayout(layout)
        self.results_data = None
        
    def display_results(self, results, vocab):
        """Display batch processing results"""
        if not results:
            return
            
        self.results_data = results
        self.vocab = vocab
        
        # Get model names
        model_names = list(results[0]['predictions'].keys())
        
        # Setup table
        columns = ['Sample ID', 'Corrupted Tokens', 'Total Tokens', 'Corruption %'] + \
                 [f'{name} Accuracy' for name in model_names]
        
        self.table.setRowCount(len(results))
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        
        # Fill table
        for i, result in enumerate(results):
            mask = result['mask']
            corrupted_count = np.sum(mask)
            total_count = len(mask)
            corruption_pct = (corrupted_count / total_count) * 100 if total_count > 0 else 0
            
            # Basic info
            self.table.setItem(i, 0, QTableWidgetItem(str(result['sample_id'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(corrupted_count)))
            self.table.setItem(i, 2, QTableWidgetItem(str(total_count)))
            self.table.setItem(i, 3, QTableWidgetItem(f"{corruption_pct:.1f}%"))
            
            # Model accuracies
            for j, model_name in enumerate(model_names):
                accuracy = result['predictions'][model_name]['accuracy']
                item = QTableWidgetItem(f"{accuracy:.3f}")
                
                # Color code based on accuracy
                if accuracy > 0.8:
                    item.setBackground(QColor(144, 238, 144))  # Light green
                elif accuracy > 0.6:
                    item.setBackground(QColor(255, 255, 224))  # Light yellow
                else:
                    item.setBackground(QColor(255, 182, 193))  # Light pink
                    
                self.table.setItem(i, 4 + j, item)
                
        # Resize columns to content
        self.table.resizeColumnsToContents()
        
    def export_results(self):
        """Export results to CSV file"""
        if not self.results_data:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    # Get model names
                    model_names = list(self.results_data[0]['predictions'].keys())
                    
                    # CSV header
                    fieldnames = ['sample_id', 'corrupted_tokens', 'total_tokens', 'corruption_pct'] + \
                               [f'{name}_accuracy' for name in model_names]
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write data
                    for result in self.results_data:
                        mask = result['mask']
                        row = {
                            'sample_id': result['sample_id'],
                            'corrupted_tokens': np.sum(mask),
                            'total_tokens': len(mask),
                            'corruption_pct': (np.sum(mask) / len(mask)) * 100
                        }
                        
                        for model_name in model_names:
                            row[f'{model_name}_accuracy'] = result['predictions'][model_name]['accuracy']
                            
                        writer.writerow(row)
                        
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")


class EnhancedModelExplorer(QMainWindow):
    """Enhanced version with batch processing and comparison features"""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.vocab = None
        self.data = None
        self.current_sample = None
        self.training_ctx = None
        self.batch_results = []
        
        # Load data and setup
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
            
            # Create vocab mapping
            self.vocab = meta['itos']  # itos is already a mapping from int to char
            self.vocab_size = meta['vocab_size']
            
            # Load training data
            data_path = Path("../data/shakespeare_char/train.bin")
            self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
            
            # Find valid starting positions
            self.valid_indices = find_double_newline_indices(self.data, self.vocab_size, 1024)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            
    def setup_training_context(self):
        """Setup training context with default values"""
        self.training_ctx = TrainingContext(
            training_type='unmasking',
            batch_size=1,
            block_size=512,
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
            iter_num=5000,
            guaranteed_unmasked=0.0,
            noise_max_ratio=0.05,
            sticky_rounds=10,
            sticky_p1_p2_multiplier=10.0,
            sticky_transition_start=500,
            sticky_transition_end=12000,
            remasking_corruption_strategy='mixed',
            remasking_strategy_weights=[0.25, 0.4, 0.25, 0.1]
        )
        
    def init_ui(self):
        """Initialize the enhanced user interface"""
        self.setWindowTitle("Enhanced Diffusion Model Explorer")
        self.setGeometry(100, 100, 1600, 1000)
        
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
            QTextEdit, QTextBrowser {
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
            QTabWidget::pane {
                border: 1px solid #555555;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: white;
                padding: 8px 16px;
                margin: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0084ff;
            }
        """)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create main tab widget
        self.main_tabs = QTabWidget()
        
        # Single Sample Tab
        single_tab = self.create_single_sample_tab()
        self.main_tabs.addTab(single_tab, "Single Sample")
        
        # Batch Processing Tab
        batch_tab = self.create_batch_processing_tab()
        self.main_tabs.addTab(batch_tab, "Batch Processing")
        
        # Results/Comparison Tab
        self.comparison_tab = ComparisonTable()
        self.main_tabs.addTab(self.comparison_tab, "Model Comparison")
        
        layout = QVBoxLayout()
        layout.addWidget(self.main_tabs)
        main_widget.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready - Enhanced Model Explorer")
        
    def create_single_sample_tab(self):
        """Create the single sample processing tab"""
        widget = QWidget()
        
        # Create splitter for main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - controls (reuse from original)
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # Right panel - text visualization
        right_panel = self.create_visualization_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([400, 1200])
        
        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        widget.setLayout(layout)
        
        return widget
        
    def create_batch_processing_tab(self):
        """Create the batch processing tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Batch Settings Group
        batch_group = QGroupBox("Batch Processing Settings")
        batch_layout = QGridLayout()
        
        # Number of samples
        batch_layout.addWidget(QLabel("Number of Samples:"), 0, 0)
        self.batch_samples_spinbox = QSpinBox()
        self.batch_samples_spinbox.setRange(1, 100)
        self.batch_samples_spinbox.setValue(10)
        batch_layout.addWidget(self.batch_samples_spinbox, 0, 1)
        
        # Sample length
        batch_layout.addWidget(QLabel("Sample Length:"), 1, 0)
        self.batch_length_spinbox = QSpinBox()
        self.batch_length_spinbox.setRange(50, 512)
        self.batch_length_spinbox.setValue(128)
        batch_layout.addWidget(self.batch_length_spinbox, 1, 1)
        
        # Corruption ratio
        batch_layout.addWidget(QLabel("Corruption Ratio:"), 2, 0)
        self.batch_ratio_spinbox = QDoubleSpinBox()
        self.batch_ratio_spinbox.setRange(0.1, 0.8)
        self.batch_ratio_spinbox.setSingleStep(0.1)
        self.batch_ratio_spinbox.setValue(0.3)
        batch_layout.addWidget(self.batch_ratio_spinbox, 2, 1)
        
        # Model selection for batch processing
        batch_layout.addWidget(QLabel("Models to Test:"), 3, 0)
        self.batch_models_layout = QVBoxLayout()
        self.batch_model_checkboxes = {}
        batch_layout.addLayout(self.batch_models_layout, 3, 1)
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_batch_btn = QPushButton("Start Batch Processing")
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        self.start_batch_btn.setEnabled(False)
        button_layout.addWidget(self.start_batch_btn)
        
        self.stop_batch_btn = QPushButton("Stop Processing")
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        self.stop_batch_btn.setEnabled(False)
        button_layout.addWidget(self.stop_batch_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Progress
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)
        
        self.batch_status = QLabel("Ready for batch processing")
        layout.addWidget(self.batch_status)
        
        # Results preview
        results_group = QGroupBox("Quick Results Summary")
        results_layout = QVBoxLayout()
        
        self.batch_summary = QTextBrowser()
        self.batch_summary.setMaximumHeight(200)
        results_layout.addWidget(self.batch_summary)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        
        return widget
        
    # Include the original methods from model_explorer.py
    def create_control_panel(self):
        """Create the control panel (from original)"""
        # [Copy the create_control_panel method from model_explorer.py]
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        
        # Model Loading Section
        model_group = QGroupBox("Model Loading")
        model_layout = QVBoxLayout()
        
        self.load_unmasking_btn = QPushButton("Load Unmasking Model")
        self.load_unmasking_btn.clicked.connect(lambda: self.load_model("../out/14.6_unmasking_no_noise.pt"))
        model_layout.addWidget(self.load_unmasking_btn)
        
        self.load_remasking_bin_btn = QPushButton("Load Remasking Binary Model")
        self.load_remasking_bin_btn.clicked.connect(lambda: self.load_model("../out/1.23_remasking_bin.pt"))
        model_layout.addWidget(self.load_remasking_bin_btn)
        
        self.load_remasking_btn = QPushButton("Load Remasking Model")
        self.load_remasking_btn.clicked.connect(lambda: self.load_model("../out/1.35_remask.pt"))
        model_layout.addWidget(self.load_remasking_btn)
        
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
        
        # [Continue with rest of control panel...]
        # Sample Generation, Corruption Settings, Actions sections
        # (Copy remaining sections from original create_control_panel method)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
        
    def create_visualization_panel(self):
        """Create visualization panel (from original)"""
        # [Copy from model_explorer.py]
        pass
        
    def load_model(self, model_path):
        """Load model and update batch processing options"""
        # [Include original load_model logic]
        # Plus update batch processing model checkboxes
        pass
        
    def start_batch_processing(self):
        """Start batch processing of multiple samples"""
        if not self.models:
            QMessageBox.warning(self, "Warning", "No models loaded")
            return
            
        # Get selected models
        selected_models = {}
        for name, checkbox in self.batch_model_checkboxes.items():
            if checkbox.isChecked():
                selected_models[name] = self.models[name]
                
        if not selected_models:
            QMessageBox.warning(self, "Warning", "No models selected for batch processing")
            return
            
        # Generate samples
        num_samples = self.batch_samples_spinbox.value()
        sample_length = self.batch_length_spinbox.value()
        samples = []
        
        for i in range(num_samples):
            if len(self.valid_indices) > 0:
                start_idx = np.random.choice(self.valid_indices)
            else:
                start_idx = np.random.randint(0, len(self.data) - sample_length)
                
            sample = self.data[start_idx:start_idx + sample_length].astype(np.int64)
            samples.append(sample)
            
        # Setup corruption settings
        corruption_settings = {
            'strategy': 'random',
            'ratio': self.batch_ratio_spinbox.value(),
            'vocab_size': self.vocab_size
        }
        
        # Start batch processing
        self.batch_processor = BatchProcessor(samples, selected_models, corruption_settings, self.device)
        self.batch_processor.progress.connect(self.update_batch_progress)
        self.batch_processor.results_ready.connect(self.on_batch_results)
        self.batch_processor.error_signal.connect(self.on_batch_error)
        
        self.batch_progress.setVisible(True)
        self.start_batch_btn.setEnabled(False)
        self.stop_batch_btn.setEnabled(True)
        
        self.batch_processor.start()
        
    def stop_batch_processing(self):
        """Stop batch processing"""
        if hasattr(self, 'batch_processor'):
            self.batch_processor.terminate()
            
        self.batch_progress.setVisible(False)
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_status.setText("Batch processing stopped")
        
    def update_batch_progress(self, value, status):
        """Update batch processing progress"""
        self.batch_progress.setValue(value)
        self.batch_status.setText(status)
        
    def on_batch_results(self, results):
        """Handle batch processing results"""
        self.batch_results = results
        
        # Update summary
        if results:
            model_names = list(results[0]['predictions'].keys())
            summary_text = f"<h3>Batch Processing Complete</h3>"
            summary_text += f"<p>Processed {len(results)} samples</p>"
            
            # Calculate average accuracies
            summary_text += "<h4>Average Accuracies:</h4><ul>"
            for model_name in model_names:
                accuracies = [r['predictions'][model_name]['accuracy'] for r in results]
                avg_accuracy = np.mean(accuracies)
                summary_text += f"<li><b>{model_name}</b>: {avg_accuracy:.3f} ± {np.std(accuracies):.3f}</li>"
            summary_text += "</ul>"
            
            self.batch_summary.setHtml(summary_text)
            
            # Update comparison table
            self.comparison_tab.display_results(results, self.vocab)
            
            # Switch to results tab
            self.main_tabs.setCurrentIndex(2)
            
        self.batch_progress.setVisible(False)
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.batch_status.setText("Batch processing complete")
        
    def on_batch_error(self, error_msg):
        """Handle batch processing error"""
        QMessageBox.critical(self, "Batch Processing Error", error_msg)
        self.stop_batch_processing()


def main():
    """Main entry point for enhanced explorer"""
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Diffusion Model Explorer")
    
    try:
        window = EnhancedModelExplorer()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting enhanced application: {e}")
        print("Falling back to basic model explorer...")
        
        # Fallback to basic version
        from model_explorer import ModelExplorerApp
        window = ModelExplorerApp()
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()