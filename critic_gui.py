#!/usr/bin/env python3
"""
Critic-Guided Text Refinement GUI

Interactive GUI for loading models and iteratively refining text using critic scores.
Workflow: Generate -> Unmask -> Score -> Remask -> Unmask -> Score ...
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import re
import json

import torch
import numpy as np

from inference_utils import (
    load_model_checkpoint,
    load_vocabulary,
    encode_text,
    decode_tokens,
    unmask_tokens,
    score_tokens_with_critic,
    remask_worst_tokens
)


class CriticGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Critic-Guided Text Refinement")
        self.root.geometry("1200x800")
        
        # State
        self.model = None
        self.metadata = None
        self.itos = None
        self.stoi = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = 'float16' if self.device == 'cuda' else 'float32'
        self.current_tokens = None
        self.current_scores = None
        self.workflow_state = 'generate'  # generate, unmask, score, remask

        # JSON save mode state
        self.json_mode = False
        self.json_data = None
        self.current_sample_idx = 0
        self.current_iteration_idx = 0

        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Top frame: Model loading
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_label = ttk.Label(top_frame, text="No model loaded", foreground="red")
        self.model_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(top_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Load JSON Save", command=self.load_json_save).pack(side=tk.LEFT, padx=5)

        # Navigation frame (hidden by default, shown in JSON mode)
        self.nav_frame = ttk.Frame(self.root, padding="10")

        ttk.Label(self.nav_frame, text="Sample:").pack(side=tk.LEFT, padx=5)
        self.sample_prev_btn = ttk.Button(self.nav_frame, text="◀", command=self.prev_sample, width=3)
        self.sample_prev_btn.pack(side=tk.LEFT, padx=2)

        self.sample_label = ttk.Label(self.nav_frame, text="0 / 0", width=10)
        self.sample_label.pack(side=tk.LEFT, padx=5)

        self.sample_next_btn = ttk.Button(self.nav_frame, text="▶", command=self.next_sample, width=3)
        self.sample_next_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(self.nav_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Label(self.nav_frame, text="Iteration:").pack(side=tk.LEFT, padx=5)
        self.iter_prev_btn = ttk.Button(self.nav_frame, text="◀", command=self.prev_iteration, width=3)
        self.iter_prev_btn.pack(side=tk.LEFT, padx=2)

        self.iter_label = ttk.Label(self.nav_frame, text="0 / 0", width=10)
        self.iter_label.pack(side=tk.LEFT, padx=5)

        self.iter_next_btn = ttk.Button(self.nav_frame, text="▶", command=self.next_iteration, width=3)
        self.iter_next_btn.pack(side=tk.LEFT, padx=2)
        
        # Middle frame: Text editor with scrollbar
        middle_frame = ttk.Frame(self.root, padding="10")
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(middle_frame, text="Text (editable when Unmask is active):").pack(anchor=tk.W)
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(middle_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                   font=("Courier", 11), height=20)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        # Configure text tags for coloring
        self.text_widget.tag_configure("correct", background="#90EE90")  # Light green
        self.text_widget.tag_configure("wrong", background="#FFB6C1")    # Light red
        
        # Bottom frame: Controls
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)
        
        # Workflow buttons
        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.unmask_btn = ttk.Button(button_frame, text="Unmask", command=self.unmask, state=tk.DISABLED)
        self.unmask_btn.pack(side=tk.LEFT, padx=5)
        
        self.score_btn = ttk.Button(button_frame, text="Score", command=self.score, state=tk.DISABLED)
        self.score_btn.pack(side=tk.LEFT, padx=5)
        
        self.remask_btn = ttk.Button(button_frame, text="Remask", command=self.remask, state=tk.DISABLED)
        self.remask_btn.pack(side=tk.LEFT, padx=5)
        
        # Threshold controls
        threshold_frame = ttk.Frame(bottom_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Remask Threshold (%):").pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.DoubleVar(value=30.0)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0, to=100,
                                         variable=self.threshold_var, orient=tk.HORIZONTAL,
                                         length=300, command=self.on_threshold_change)
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        
        self.threshold_label = ttk.Label(threshold_frame, text="30.0%")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(threshold_frame, text="Wrongness cutoff:").pack(side=tk.LEFT, padx=5)
        self.wrongness_label = ttk.Label(threshold_frame, text="N/A", foreground="blue")
        self.wrongness_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(threshold_frame, text="Tokens to remask:").pack(side=tk.LEFT, padx=5)
        self.tokens_count_label = ttk.Label(threshold_frame, text="N/A", foreground="blue")
        self.tokens_count_label.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Load a model or JSON save to begin")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def load_model(self):
        """Load a model checkpoint"""
        checkpoint_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt *.pth"), ("All Files", "*.*")]
        )
        
        if not checkpoint_path:
            return
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            # Load model
            self.model, self.metadata = load_model_checkpoint(checkpoint_path, self.device, self.dtype)
            
            # Check for critic head
            if not self.metadata['has_critic']:
                messagebox.showwarning("No Critic Head", 
                                      "This model does not have a critic head. Scoring will not be available.")
            
            # Load vocabulary - try multiple locations
            checkpoint_dir = Path(checkpoint_path).parent
            meta_path = None

            # Try 1: Same directory as checkpoint
            candidate = checkpoint_dir / 'meta.pkl'
            if candidate.exists():
                meta_path = candidate

            # Try 2: data/ subdirectory relative to checkpoint
            if meta_path is None:
                candidate = checkpoint_dir / 'data' / 'meta.pkl'
                if candidate.exists():
                    meta_path = candidate

            # Try 3: data/ directory relative to script
            if meta_path is None:
                candidate = Path('data') / 'meta.pkl'
                if candidate.exists():
                    meta_path = candidate

            # Try 4: Ask user to locate it
            if meta_path is None:
                response = messagebox.askyesno(
                    "meta.pkl not found",
                    f"Could not find meta.pkl automatically.\n\n"
                    f"Searched:\n"
                    f"  - {checkpoint_dir / 'meta.pkl'}\n"
                    f"  - {checkpoint_dir / 'data' / 'meta.pkl'}\n"
                    f"  - {Path('data') / 'meta.pkl'}\n\n"
                    f"Would you like to select meta.pkl manually?"
                )

                if response:
                    meta_file = filedialog.askopenfilename(
                        title="Select meta.pkl",
                        filetypes=[("Pickle files", "*.pkl"), ("All Files", "*.*")],
                        initialdir=checkpoint_dir
                    )
                    if meta_file:
                        meta_path = Path(meta_file)
                    else:
                        return
                else:
                    return

            # Load vocabulary
            self.itos, self.stoi, _ = load_vocabulary(str(meta_path))
            
            # Update UI
            model_name = Path(checkpoint_path).name
            self.model_label.config(text=f"{model_name} ✓", foreground="green")
            self.status_var.set(f"Model loaded: {model_name} | Device: {self.device}")
            
            # Enable unmask button to start workflow
            self.unmask_btn.config(state=tk.NORMAL)
            self.text_widget.config(state=tk.NORMAL)
            self.workflow_state = 'unmask'
            
            # Insert sample text with masks
            sample_text = "The [MASK][MASK][MASK][MASK][MASK] brown fox [MASK][MASK][MASK][MASK][MASK] over the lazy dog."
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", sample_text)
            
        except Exception as e:
            messagebox.showerror("Error Loading Model", str(e))
            self.status_var.set("Error loading model")

    def load_json_save(self):
        """Load a JSON save file with precomputed iterations"""
        json_path = filedialog.askopenfilename(
            title="Select JSON Save File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )

        if not json_path:
            return

        try:
            self.status_var.set("Loading JSON save...")
            self.root.update()

            # Load JSON data
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)

            # Extract paths
            generator_path = self.json_data['generator']
            meta_path = self.json_data['meta']

            # Load model
            if not os.path.exists(generator_path):
                messagebox.showerror("Error", f"Generator model not found: {generator_path}")
                return

            self.model, self.metadata = load_model_checkpoint(generator_path, self.device, self.dtype)

            # Load vocabulary
            if not os.path.exists(meta_path):
                messagebox.showerror("Error", f"Meta file not found: {meta_path}")
                return

            self.itos, self.stoi, _ = load_vocabulary(meta_path)

            # Enter JSON mode
            self.json_mode = True
            self.current_sample_idx = 0
            self.current_iteration_idx = 0

            # Update UI
            model_name = Path(generator_path).name
            self.model_label.config(text=f"{model_name} (JSON) ✓", foreground="green")

            # Show navigation frame
            self.nav_frame.pack(fill=tk.X, after=self.root.winfo_children()[0])

            # Update navigation labels
            num_samples = len(self.json_data['samples'])
            num_iterations = len(self.json_data['samples'][0]['iterations']) if num_samples > 0 else 0
            self.sample_label.config(text=f"1 / {num_samples}")
            self.iter_label.config(text=f"1 / {num_iterations}")

            # Load first iteration
            self.load_iteration_from_json()

            self.status_var.set(f"JSON loaded: {num_samples} samples, {num_iterations} iterations each")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error Loading JSON", str(e))
            self.status_var.set("Error loading JSON")

    def load_iteration_from_json(self):
        """Load and display the current sample/iteration from JSON data"""
        if not self.json_mode or self.json_data is None:
            return

        try:
            sample_data = self.json_data['samples'][self.current_sample_idx]
            iteration_data = sample_data['iterations'][self.current_iteration_idx]

            # Get tokens
            input_masked = torch.tensor(iteration_data['input_masked'], dtype=torch.long)
            output_unmasked = torch.tensor(iteration_data['output_unmasked'], dtype=torch.long)
            remasked_indices = iteration_data['remasked_indices']

            # Store current state
            self.current_tokens = output_unmasked
            self.current_scores = None

            # Display input (masked)
            input_text = self.tokens_to_text(input_masked)
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", input_text)
            self.text_widget.config(state=tk.DISABLED)

            # Update button states for JSON mode
            self.unmask_btn.config(state=tk.NORMAL)
            self.score_btn.config(state=tk.DISABLED)
            self.remask_btn.config(state=tk.DISABLED)

            self.workflow_state = 'unmask'

            iteration_num = iteration_data['iteration']
            self.status_var.set(f"Sample {self.current_sample_idx + 1}, Iteration {iteration_num}: {len(remasked_indices)} tokens will be remasked")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to load iteration: {e}")

    def prev_sample(self):
        """Navigate to previous sample"""
        if not self.json_mode:
            return

        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.current_iteration_idx = 0
            self.update_navigation_labels()
            self.load_iteration_from_json()

    def next_sample(self):
        """Navigate to next sample"""
        if not self.json_mode:
            return

        num_samples = len(self.json_data['samples'])
        if self.current_sample_idx < num_samples - 1:
            self.current_sample_idx += 1
            self.current_iteration_idx = 0
            self.update_navigation_labels()
            self.load_iteration_from_json()

    def prev_iteration(self):
        """Navigate to previous iteration"""
        if not self.json_mode:
            return

        if self.current_iteration_idx > 0:
            self.current_iteration_idx -= 1
            self.update_navigation_labels()
            self.load_iteration_from_json()

    def next_iteration(self):
        """Navigate to next iteration"""
        if not self.json_mode:
            return

        sample_data = self.json_data['samples'][self.current_sample_idx]
        num_iterations = len(sample_data['iterations'])
        if self.current_iteration_idx < num_iterations - 1:
            self.current_iteration_idx += 1
            self.update_navigation_labels()
            self.load_iteration_from_json()

    def update_navigation_labels(self):
        """Update navigation label text"""
        num_samples = len(self.json_data['samples'])
        sample_data = self.json_data['samples'][self.current_sample_idx]
        num_iterations = len(sample_data['iterations'])

        self.sample_label.config(text=f"{self.current_sample_idx + 1} / {num_samples}")
        self.iter_label.config(text=f"{self.current_iteration_idx + 1} / {num_iterations}")

    def text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to tokens, handling [MASK] as special token"""
        # Replace [MASK] with a placeholder character temporarily
        mask_placeholder = '\x00'  # Null character unlikely to be in text
        text_processed = text.replace('[MASK]', mask_placeholder)
        
        # Encode character by character
        tokens = []
        for char in text_processed:
            if char == mask_placeholder:
                tokens.append(self.metadata['mask_token_id'])
            else:
                tokens.append(self.stoi.get(char, 0))  # 0 for unknown
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def tokens_to_text(self, tokens: torch.Tensor) -> str:
        """Convert tokens to text, rendering special tokens"""
        return decode_tokens(tokens, self.itos,
                           mask_token_id=self.metadata['mask_token_id'],
                           pad_token_id=self.metadata['pad_token_id'],
                           cls_token_id=self.metadata['cls_token_id'],
                           sep_token_id=self.metadata['sep_token_id'])
    
    def unmask(self):
        """Unmask all [MASK] tokens in the text"""
        if self.model is None:
            messagebox.showerror("Error", "No model loaded")
            return

        try:
            self.status_var.set("Unmasking...")
            self.root.update()

            if self.json_mode:
                # In JSON mode, load precomputed unmasked output
                sample_data = self.json_data['samples'][self.current_sample_idx]
                iteration_data = sample_data['iterations'][self.current_iteration_idx]
                unmasked_tokens = torch.tensor(iteration_data['output_unmasked'], dtype=torch.long)
                self.current_tokens = unmasked_tokens
                self.current_scores = None
            else:
                # Normal mode: compute unmasking
                text = self.text_widget.get("1.0", tk.END).strip()
                tokens = self.text_to_tokens(text)

                unmasked_tokens = unmask_tokens(
                    self.model, tokens,
                    self.metadata['mask_token_id'],
                    self.metadata['vocab_size'],
                    temperature=0.8,
                    device=self.device,
                    dtype=self.dtype
                )

                self.current_tokens = unmasked_tokens
                self.current_scores = None

            # Update text widget
            unmasked_text = self.tokens_to_text(unmasked_tokens)
            self.text_widget.config(state=tk.NORMAL)
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", unmasked_text)
            self.text_widget.config(state=tk.DISABLED)

            # Update workflow state
            self.workflow_state = 'score'
            self.unmask_btn.config(state=tk.DISABLED)

            if self.metadata['has_critic']:
                self.score_btn.config(state=tk.NORMAL)
            else:
                # Skip to remask if no critic
                self.remask_btn.config(state=tk.NORMAL)

            self.status_var.set("Unmasking complete. Click Score to evaluate.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Unmasking failed: {e}")
            self.status_var.set("Unmasking failed")
    
    def score(self):
        """Score tokens using critic head"""
        if self.model is None or self.current_tokens is None:
            messagebox.showerror("Error", "No tokens to score")
            return
        
        if not self.metadata['has_critic']:
            messagebox.showerror("Error", "Model does not have critic head")
            return
        
        try:
            self.status_var.set("Scoring with critic...")
            self.root.update()
            
            # Score tokens
            scores = score_tokens_with_critic(
                self.model, self.current_tokens,
                device=self.device, dtype=self.dtype
            )
            
            self.current_scores = scores
            
            # Apply gradient coloring to text
            self.apply_gradient_coloring(scores)
            
            # Update wrongness display
            self.update_wrongness_display()
            
            # Update workflow state
            self.workflow_state = 'remask'
            self.score_btn.config(state=tk.DISABLED)
            self.remask_btn.config(state=tk.NORMAL)
            
            self.status_var.set("Scoring complete. Adjust threshold and click Remask.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Scoring failed: {e}")
            self.status_var.set("Scoring failed")
    
    def apply_gradient_coloring(self, scores: torch.Tensor):
        """Apply continuous gradient coloring from green (0) to red (1)"""
        # Clear existing tags
        self.text_widget.tag_remove("correct", "1.0", tk.END)
        self.text_widget.tag_remove("wrong", "1.0", tk.END)
        
        # Get text
        text = self.text_widget.get("1.0", tk.END).strip()
        
        # Apply color to each character
        scores_np = scores.cpu().numpy()
        
        char_idx = 0
        for i, char in enumerate(text):
            if char_idx >= len(scores_np):
                break
            
            score = float(scores_np[char_idx])
            
            # Create gradient color from green (0,255,0) to red (255,0,0)
            red = int(255 * score)
            green = int(255 * (1 - score))
            color = f'#{red:02x}{green:02x}00'
            
            # Create unique tag for this character
            tag_name = f"char_{i}"
            start_idx = f"1.0+{i}c"
            end_idx = f"1.0+{i+1}c"
            
            self.text_widget.tag_add(tag_name, start_idx, end_idx)
            self.text_widget.tag_configure(tag_name, background=color)
            
            char_idx += 1
    
    def remask(self):
        """Remask worst tokens based on threshold"""
        if self.model is None or self.current_tokens is None or self.current_scores is None:
            messagebox.showerror("Error", "No scores available for remasking")
            return

        try:
            if self.json_mode:
                # In JSON mode, use precomputed remasked indices
                sample_data = self.json_data['samples'][self.current_sample_idx]
                iteration_data = sample_data['iterations'][self.current_iteration_idx]
                remasked_indices = iteration_data['remasked_indices']

                # Apply remasking using precomputed indices
                remasked_tokens = self.current_tokens.clone()
                for idx in remasked_indices:
                    remasked_tokens[idx] = self.metadata['mask_token_id']

                num_remasked = len(remasked_indices)
            else:
                # Normal mode: compute remasking based on threshold
                threshold_pct = self.threshold_var.get()

                remasked_tokens, worst_indices = remask_worst_tokens(
                    self.current_tokens,
                    self.current_scores,
                    self.metadata['mask_token_id'],
                    threshold_pct,
                    content_len=len(self.current_tokens) if self.current_tokens.dim() == 1 else self.current_tokens.shape[1]
                )

                num_remasked = len(worst_indices)

            # Update workflow state FIRST to prevent preview from running
            self.workflow_state = 'unmask'

            # Store remasked tokens
            self.current_tokens = remasked_tokens
            self.current_scores = None

            # Update text widget
            remasked_text = self.tokens_to_text(remasked_tokens)

            # Clear ALL tags first
            for tag in self.text_widget.tag_names():
                if tag.startswith("char_") or tag.startswith("mask_preview_"):
                    self.text_widget.tag_delete(tag)

            # Enable editing before updating text
            self.text_widget.config(state=tk.NORMAL)

            # Update text
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.insert("1.0", remasked_text)

            # Update button states
            self.remask_btn.config(state=tk.DISABLED)
            self.score_btn.config(state=tk.DISABLED)

            if self.json_mode:
                # In JSON mode, disable unmask and enable navigation
                self.unmask_btn.config(state=tk.DISABLED)
                self.text_widget.config(state=tk.DISABLED)
            else:
                # In normal mode, enable unmask for next iteration
                self.unmask_btn.config(state=tk.NORMAL)

            self.status_var.set(f"Remasked {num_remasked} tokens. " +
                              ("Navigate to next iteration." if self.json_mode else "Edit text or click Unmask to continue."))

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Remasking failed: {e}")
            self.status_var.set("Remasking failed")
    
    def on_threshold_change(self, *args):
        """Handle threshold slider change - update display and preview"""
        self.update_wrongness_display()

        # ONLY preview if in remask state - do NOT preview in other states
        if self.workflow_state == 'remask' and self.current_scores is not None:
            self.preview_remask()
        else:
            # In other states, just update the wrongness display
            pass

    def preview_remask(self):
        """Preview which tokens will be masked at current threshold"""
        try:
            threshold_pct = self.threshold_var.get()

            # Calculate which tokens would be remasked
            scores_np = self.current_scores.cpu().numpy()
            num_to_remask = int((threshold_pct / 100.0) * len(scores_np))
            num_to_remask = max(1, min(num_to_remask, len(scores_np)))

            # Get indices of worst tokens
            worst_indices = np.argsort(scores_np)[::-1][:num_to_remask]
            worst_set = set(worst_indices)

            # Clear existing coloring
            for tag in self.text_widget.tag_names():
                if tag.startswith("char_") or tag.startswith("mask_preview_"):
                    self.text_widget.tag_delete(tag)

            # Get current text
            text = self.text_widget.get("1.0", tk.END).strip()

            # Build new display with [MASK] preview
            new_text_parts = []
            for i, char in enumerate(text):
                if i >= len(scores_np):
                    new_text_parts.append(char)
                elif i in worst_set:
                    new_text_parts.append('[MASK]')
                else:
                    new_text_parts.append(char)

            # Update text widget
            self.text_widget.delete("1.0", tk.END)

            # Insert character by character with coloring
            char_pos = 0
            text_pos = 0
            for i, char in enumerate(text):
                if i >= len(scores_np):
                    self.text_widget.insert(tk.END, char)
                    continue

                if i in worst_set:
                    # Insert [MASK] with red background
                    start_idx = self.text_widget.index(tk.END + "-1c")
                    self.text_widget.insert(tk.END, '[MASK]')
                    end_idx = self.text_widget.index(tk.END + "-1c")
                    tag_name = f"mask_preview_{i}"
                    self.text_widget.tag_add(tag_name, start_idx, end_idx)
                    self.text_widget.tag_configure(tag_name, background="#FFB6C1")  # Light red
                else:
                    # Insert original character with gradient color
                    score = float(scores_np[i])
                    red = int(255 * score)
                    green = int(255 * (1 - score))
                    color = f'#{red:02x}{green:02x}00'

                    start_idx = self.text_widget.index(tk.END + "-1c")
                    self.text_widget.insert(tk.END, char)
                    end_idx = self.text_widget.index(tk.END + "-1c")
                    tag_name = f"char_{i}"
                    self.text_widget.tag_add(tag_name, start_idx, end_idx)
                    self.text_widget.tag_configure(tag_name, background=color)

        except Exception as e:
            # Silently fail for preview - don't interrupt user
            pass

    def update_wrongness_display(self, *args):
        """Update the wrongness cutoff display based on threshold"""
        threshold_pct = self.threshold_var.get()
        self.threshold_label.config(text=f"{threshold_pct:.1f}%")

        if self.current_scores is not None:
            # Calculate wrongness cutoff (score value at threshold percentile)
            scores_np = self.current_scores.cpu().numpy()
            num_to_remask = int((threshold_pct / 100.0) * len(scores_np))
            num_to_remask = max(1, min(num_to_remask, len(scores_np)))

            # Update token count
            self.tokens_count_label.config(text=f"{num_to_remask} / {len(scores_np)}")

            # Get the score at the cutoff
            sorted_scores = np.sort(scores_np)[::-1]  # Descending
            if num_to_remask <= len(sorted_scores):
                cutoff_score = sorted_scores[num_to_remask - 1]
                self.wrongness_label.config(text=f"{cutoff_score:.3f}")
            else:
                self.wrongness_label.config(text="N/A")
        else:
            self.wrongness_label.config(text="N/A")
            self.tokens_count_label.config(text="N/A")


def main():
    root = tk.Tk()
    app = CriticGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

