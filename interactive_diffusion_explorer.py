#!/usr/bin/env python3
"""
Interactive Diffusion Model Explorer
A console-based application for exploring diffusion models, datasets, and sampling strategies.
"""

import os
import sys
import pickle
import math
import time
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, List, Tuple, Optional

import torch
from model import GPTConfig, GPT

# Import utilities from existing files
sys.path.append('data/char_diffusion')
from masking_utils import apply_stage_masking, apply_random_masking_cpu, apply_target_driven_sticky_masking_cpu, apply_span_masking_cpu

# Terminal control
try:
    import keyboard
    import msvcrt  # Windows
    HAS_KEYBOARD = True
except ImportError:
    try:
        import tty
        import termios  # Unix
        HAS_KEYBOARD = True
    except ImportError:
        HAS_KEYBOARD = False

# Configuration
MODEL_PATH = 'out/big_boy2.pt'  # Hardcoded model path - change this as needed
DATA_DIR = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'float16' if DEVICE == 'cuda' else 'float32'

class DiffusionExplorer:
    def __init__(self):
        self.model = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None
        self.mask_token_id = None
        self.dataset_name = None
        self.current_batch = None
        self.current_sample_idx = 0
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
        self.ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=self.ptdtype)
        
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self, title: str):
        """Print formatted header"""
        self.clear_screen()
        print("=" * 80)
        print(f" {title.center(76)} ")
        print("=" * 80)
        print()
        
    def wait_for_key(self, prompt: str = "Press any key to continue...") -> str:
        """Wait for user input"""
        if HAS_KEYBOARD:
            print(prompt)
            if os.name == 'nt':  # Windows
                return msvcrt.getch().decode('utf-8', errors='ignore')
            else:  # Unix
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.cbreak(fd)
                    return sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        else:
            return input(prompt + " (Press Enter)")
            
    def get_menu_choice(self, options: List[str], title: str = "Select an option:") -> int:
        """Display menu and get user choice"""
        while True:
            self.clear_screen()
            print(f"{title}\n")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            print(f"\n0. Exit")
            
            try:
                choice = input(f"\nEnter choice (0-{len(options)}): ").strip()
                if choice == '0':
                    return 0
                choice_int = int(choice)
                if 1 <= choice_int <= len(options):
                    return choice_int
                print("Invalid choice. Please try again.")
                self.wait_for_key()
            except ValueError:
                print("Please enter a valid number.")
                self.wait_for_key()
    
    def load_model(self) -> bool:
        """Load the diffusion model"""
        self.print_header("Loading Model")
        
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            print("Please update MODEL_PATH in the script configuration.")
            self.wait_for_key()
            return False
            
        try:
            print(f"📂 Loading model from: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            
            if 'model_args' not in checkpoint:
                print("❌ Invalid checkpoint: missing 'model_args'")
                self.wait_for_key()
                return False
                
            model_args = checkpoint['model_args']
            
            # Ensure backward compatibility
            if 'attention_type' not in model_args:
                model_args['attention_type'] = 'causal'
            if 'position_encoding' not in model_args:
                model_args['position_encoding'] = 'absolute'
                
            print(f"🔧 Model configuration:")
            print(f"   • vocab_size: {model_args.get('vocab_size')}")
            print(f"   • block_size: {model_args.get('block_size')}")
            print(f"   • attention_type: {model_args.get('attention_type')}")
            print(f"   • position_encoding: {model_args.get('position_encoding')}")
            
            # Create model
            gptconf = GPTConfig(**model_args)
            self.model = GPT(gptconf)
            
            # Load weights
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                    
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(DEVICE)
            
            print(f"✅ Model loaded successfully")
            print(f"   • Parameters: {self.model.get_num_params()/1e6:.1f}M")
            print(f"   • Device: {DEVICE}")
            
            self.wait_for_key()
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.wait_for_key()
            return False
    
    def select_dataset(self) -> bool:
        """Select dataset from available datasets"""
        self.print_header("Dataset Selection")
        
        if not os.path.exists(DATA_DIR):
            print(f"❌ Data directory not found: {DATA_DIR}")
            self.wait_for_key()
            return False
            
        # Find available datasets
        datasets = []
        for item in os.listdir(DATA_DIR):
            dataset_path = os.path.join(DATA_DIR, item)
            if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, 'meta.pkl')):
                datasets.append(item)
                
        if not datasets:
            print(f"❌ No datasets found in {DATA_DIR}")
            print("Datasets should contain a meta.pkl file")
            self.wait_for_key()
            return False
            
        datasets.sort()
        choice = self.get_menu_choice(datasets, "Available datasets:")
        
        if choice == 0:
            return False
            
        self.dataset_name = datasets[choice - 1]
        return self.load_vocabulary()
    
    def load_vocabulary(self) -> bool:
        """Load vocabulary for selected dataset"""
        self.print_header(f"Loading Vocabulary: {self.dataset_name}")
        
        try:
            meta_path = os.path.join(DATA_DIR, self.dataset_name, 'meta.pkl')
            print(f"📂 Loading vocabulary from: {meta_path}")
            
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                
            self.stoi = meta['stoi']
            self.itos = meta['itos']
            vocab_size_meta = meta['vocab_size']
            
            # Get model's vocabulary size
            model_vocab_size = self.model.config.vocab_size
            self.vocab_size = model_vocab_size
            self.mask_token_id = self.vocab_size - 1
            
            print(f"✅ Vocabulary loaded:")
            print(f"   • Dataset vocab size: {vocab_size_meta}")
            print(f"   • Model vocab size: {model_vocab_size}")
            print(f"   • Using vocab size: {self.vocab_size}")
            print(f"   • Mask token ID: {self.mask_token_id}")
            if self.mask_token_id < len(self.itos):
                print(f"   • Mask token: '{self.itos[self.mask_token_id]}'")
            
            self.wait_for_key()
            return True
            
        except Exception as e:
            print(f"❌ Error loading vocabulary: {e}")
            self.wait_for_key()
            return False
    
    def select_data_file(self) -> bool:
        """Select data file from dataset"""
        self.print_header(f"Data File Selection: {self.dataset_name}")
        
        queue_dir = os.path.join(DATA_DIR, self.dataset_name, 'queue')
        if not os.path.exists(queue_dir):
            print(f"❌ Queue directory not found: {queue_dir}")
            self.wait_for_key()
            return False
            
        # Find batch files
        batch_files = []
        for root, dirs, files in os.walk(queue_dir):
            for file in files:
                if file.endswith('.pt'):
                    rel_path = os.path.relpath(os.path.join(root, file), queue_dir)
                    batch_files.append(rel_path)
                    
        if not batch_files:
            print(f"❌ No .pt files found in {queue_dir}")
            self.wait_for_key()
            return False
            
        batch_files.sort()
        
        # Show first 20 files for selection
        display_files = batch_files[:20]
        if len(batch_files) > 20:
            display_files.append(f"... and {len(batch_files) - 20} more files")
            
        choice = self.get_menu_choice(display_files, f"Available batch files (showing first 20):")
        
        if choice == 0:
            return False
            
        if choice > len(batch_files):
            print("❌ Invalid selection")
            self.wait_for_key()
            return False
            
        selected_file = batch_files[choice - 1]
        return self.load_batch_file(selected_file)
    
    def load_batch_file(self, batch_file: str) -> bool:
        """Load selected batch file"""
        self.print_header(f"Loading Batch File")
        
        try:
            full_path = os.path.join(DATA_DIR, self.dataset_name, 'queue', batch_file)
            print(f"📂 Loading: {batch_file}")
            print(f"📂 Full path: {full_path}")
            
            batch_data = torch.load(full_path, map_location='cpu')
            
            # Log additional batch information if available
            if 'metadata' in batch_data:
                metadata = batch_data['metadata']
                print(f"📋 Batch metadata found:")
                for key, value in metadata.items():
                    if key not in ['tensors', 'stage_info']:  # Skip tensor data and verbose stage info
                        print(f"     {key}: {value}")
                # Show stage info summary if present
                if 'stage_info' in metadata:
                    stage_info = metadata['stage_info']
                    if isinstance(stage_info, list):
                        print(f"     stage_info: {len(stage_info)} samples with stage configurations")
                    else:
                        print(f"     stage_info: {stage_info}")
            
            # Check for generation info
            if 'generation_info' in batch_data:
                gen_info = batch_data['generation_info']
                print(f"⚙️ Generation info: {gen_info}")
                
            # List all keys in batch data for debugging
            print(f"🔍 Available keys in batch: {list(batch_data.keys())}")
            
            # Extract tensors
            if 'tensors' in batch_data:
                tensors = batch_data['tensors']
                x_tensor = tensors.get('x', None)
                y_tensor = tensors.get('y', None)
            else:
                x_tensor = batch_data.get('x', None)
                y_tensor = batch_data.get('y', None)
                
            if x_tensor is None or y_tensor is None:
                print("❌ Could not find 'x' or 'y' tensors in batch file")
                print("Available keys:", list(batch_data.keys()))
                if 'tensors' in batch_data:
                    print("Tensors keys:", list(batch_data['tensors'].keys()))
                self.wait_for_key()
                return False
                
            self.current_batch = {'x': x_tensor, 'y': y_tensor}
            self.current_sample_idx = 0
            
            batch_size, seq_len = x_tensor.shape
            print(f"✅ Batch file loaded:")
            print(f"   • Batch size: {batch_size}")
            print(f"   • Sequence length: {seq_len}")
            
            # Show some statistics
            total_tokens = batch_size * seq_len
            ignore_index = -100
            masked_tokens = (y_tensor != ignore_index).sum().item()
            mask_percentage = (masked_tokens / total_tokens) * 100
            
            # Calculate per-sample masking statistics
            per_sample_masked = (y_tensor != ignore_index).sum(dim=1).float()  # (batch_size,)
            per_sample_percentages = (per_sample_masked / seq_len) * 100  # Convert to percentages
            
            # Sort percentages for percentile calculations
            sorted_percentages, _ = torch.sort(per_sample_percentages)
            
            min_masked_pct = sorted_percentages[0].item()
            max_masked_pct = sorted_percentages[-1].item()
            
            # Calculate 10th and 90th percentiles
            p10_idx = int(0.1 * batch_size)
            p90_idx = int(0.9 * batch_size)
            p10_masked_pct = sorted_percentages[p10_idx].item()
            p90_masked_pct = sorted_percentages[p90_idx].item()
            
            print(f"   • Total tokens: {total_tokens}")
            print(f"   • Masked tokens: {masked_tokens}")
            print(f"   • Overall mask percentage: {mask_percentage:.2f}%")
            print(f"   • Per-sample mask distribution:")
            print(f"     - Min: {min_masked_pct:.2f}%")
            print(f"     - 10th percentile: {p10_masked_pct:.2f}%")
            print(f"     - 90th percentile: {p90_masked_pct:.2f}%")
            print(f"     - Max: {max_masked_pct:.2f}%")
            
            self.wait_for_key()
            return True
            
        except Exception as e:
            print(f"❌ Error loading batch file: {e}")
            self.wait_for_key()
            return False
    
    def decode_tokens(self, tokens) -> str:
        """Decode token IDs to text"""
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
            
        result = []
        for token_id in tokens:
            if token_id == self.mask_token_id:
                result.append('[MASK]')
            elif token_id < len(self.itos):
                result.append(self.itos[token_id])
            else:
                result.append(f'[UNK:{token_id}]')
        return ''.join(result)
    
    def create_colored_result(self, original_tokens, predicted_tokens, ground_truth, masked_positions, ignore_index):
        """Create colored result text showing correct/incorrect predictions"""
        if hasattr(original_tokens, 'tolist'):
            original_tokens = original_tokens.tolist()
        if hasattr(predicted_tokens, 'tolist'):
            predicted_tokens = predicted_tokens.tolist()
        if hasattr(ground_truth, 'tolist'):
            ground_truth = ground_truth.tolist()
        
        masked_pos_set = set(pos.item() if hasattr(pos, 'item') else pos for pos in masked_positions)
        
        result = []
        for i, (orig_token, pred_token, gt_token) in enumerate(zip(original_tokens, predicted_tokens, ground_truth)):
            char = self.itos.get(pred_token, f'[UNK:{pred_token}]') if pred_token < len(self.itos) else f'[UNK:{pred_token}]'
            
            if i in masked_pos_set:
                # This was a masked position - check if prediction is correct
                if gt_token != ignore_index and pred_token == gt_token:
                    # Correct prediction - green text
                    result.append(f'\033[32m{char}\033[0m')  # Green text
                elif gt_token != ignore_index:
                    # Incorrect prediction - red text
                    result.append(f'\033[31m{char}\033[0m')  # Red text
                else:
                    # No ground truth available - yellow text
                    result.append(f'\033[33m{char}\033[0m')  # Yellow text
            else:
                # Original character - no color
                result.append(char)
        
        result_text = ''.join(result)
        # Ensure color is reset at the end
        return result_text + '\033[0m'
    
    def navigate_samples(self):
        """Navigate through samples in the batch"""
        if self.current_batch is None:
            print("❌ No batch loaded")
            self.wait_for_key()
            return
            
        x_tensor = self.current_batch['x']
        y_tensor = self.current_batch['y']
        batch_size = x_tensor.shape[0]
        ignore_index = -100
        
        while True:
            self.print_header(f"Sample Navigation: {self.dataset_name}")
            
            seq_len = x_tensor.shape[1]
            print(f"📊 Current sample: {self.current_sample_idx + 1}/{batch_size} (sequence length: {seq_len})")
            print(f"🔧 Navigation: ← Previous | → Next | U Unmask | G Generate | Q Quit")
            print("-" * 80)
            
            # Show current sample
            x_tokens = x_tensor[self.current_sample_idx]
            y_tokens = y_tensor[self.current_sample_idx]
            
            x_decoded = self.decode_tokens(x_tokens)
            
            # Show target tokens for masked positions
            y_decoded_parts = []
            masked_positions = []
            for j, (x_tok, y_tok) in enumerate(zip(x_tokens.tolist(), y_tokens.tolist())):
                if y_tok != ignore_index:
                    y_decoded_parts.append(self.itos.get(y_tok, f'[UNK:{y_tok}]'))
                    masked_positions.append(j)
                else:
                    y_decoded_parts.append('_')
            
            print(f"📝 Input (x):  {repr(x_decoded)}")
            print(f"🎯 Target (y): {''.join(y_decoded_parts)}")
            print(f"🎭 Masked positions: {len(masked_positions)} positions")
            
            print(f"\nCommands:")
            print(f"  ← / A - Previous sample")
            print(f"  → / D - Next sample") 
            print(f"  U - Run model unmasking on this sample")
            print(f"  G - Generate test sample")
            print(f"  Q - Quit to main menu")
            
            key = self.wait_for_key("Enter command: ").lower()
            
            if key in ['q']:
                break
            elif key in ['a'] or (HAS_KEYBOARD and key == '\x1b'):  # Left arrow or A
                self.current_sample_idx = (self.current_sample_idx - 1) % batch_size
            elif key in ['d'] or (HAS_KEYBOARD and key == '\x1b'):  # Right arrow or D
                self.current_sample_idx = (self.current_sample_idx + 1) % batch_size
            elif key in ['u']:
                self.run_model_unmasking()
            elif key in ['g']:
                self.generate_test_sample()
    
    def run_model_unmasking(self):
        """Run model unmasking on current sample"""
        if self.current_batch is None:
            print("❌ No batch loaded")
            self.wait_for_key()
            return
            
        self.print_header("Model Unmasking")
        
        x_tensor = self.current_batch['x']
        sample_tokens = x_tensor[self.current_sample_idx:self.current_sample_idx+1].to(DEVICE)  # Keep batch dim
        
        print(f"🔄 Running model unmasking...")
        original_tokens = sample_tokens[0]
        original_text = self.decode_tokens(original_tokens)
        total_tokens = len(original_tokens)
        print(f"📝 Original input (all {total_tokens} tokens):")
        print(f"    {repr(original_text)}")
        print()
        
        try:
            with torch.no_grad():
                with self.ctx:
                    # Get model predictions - pass dummy targets to get full sequence logits
                    dummy_targets = torch.zeros_like(sample_tokens)
                    model_output = self.model(sample_tokens, targets=dummy_targets)
                    
                    # Handle different model output formats
                    if isinstance(model_output, tuple):
                        logits = model_output[0]  # First element is usually logits
                    else:
                        logits = model_output
                    
                    # Apply temperature and get probabilities
                    temperature = 0.8
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Find masked positions
                    masked_positions = (sample_tokens[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(masked_positions) == 0:
                        print("ℹ️  No masked positions found in this sample")
                        self.wait_for_key()
                        return
                    
                    print(f"🎭 Found {len(masked_positions)} masked positions")
                    
                    # Get ground truth for comparison if available
                    y_tensor = self.current_batch['y']
                    ground_truth = y_tensor[self.current_sample_idx]
                    ignore_index = -100
                    
                    # Sample from predictions for masked positions
                    result_tokens = sample_tokens[0].clone()
                    predicted_tokens = []
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for pos in masked_positions:
                        pos_probs = probs[0, pos, :self.vocab_size-1]  # Exclude mask token
                        predicted_token = torch.multinomial(pos_probs, 1).item()
                        result_tokens[pos] = predicted_token
                        predicted_tokens.append((pos.item(), predicted_token))
                        
                        # Check if prediction matches ground truth
                        if ground_truth[pos] != ignore_index:
                            total_predictions += 1
                            if predicted_token == ground_truth[pos].item():
                                correct_predictions += 1
                    
                    predicted_text = self.decode_tokens(result_tokens)
                    
                    print(f"✅ Unmasking complete!")
                    if total_predictions > 0:
                        accuracy = (correct_predictions / total_predictions) * 100
                        print(f"🎯 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
                    print()
                    
                    # Create colored result text
                    colored_result = self.create_colored_result(
                        sample_tokens[0], result_tokens, ground_truth, masked_positions, ignore_index
                    )
                    print(f"📄 Result with color coding:")
                    total_tokens = len(result_tokens)
                    print(f"    {colored_result}")
                    print(f"    Total tokens: {total_tokens}")
                    print()
                    print(f"    \033[32m🟢 = Correct prediction\033[0m  \033[31m🔴 = Incorrect prediction\033[0m  ⚪ = Original text")
                    
        except Exception as e:
            print(f"❌ Error during unmasking: {e}")
            
        self.wait_for_key()
    
    def reconstruct_original_text(self, x_tokens: torch.Tensor, y_tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct original text from masked input (x) and targets (y)"""
        original_tokens = x_tokens.clone()
        ignore_index = -100
        
        # Replace masked tokens with their original values from y
        mask_positions = (y_tokens != ignore_index)
        original_tokens[mask_positions] = y_tokens[mask_positions]
        
        return original_tokens

    def generate_test_sample(self):
        """Generate a test sample using masking strategies"""
        self.print_header("Generate Test Sample")
        
        if self.current_batch is None:
            print("❌ No batch loaded")
            self.wait_for_key()
            return
        
        # Get sample text for masking - reconstruct original unmasked text
        x_tensor = self.current_batch['x']
        y_tensor = self.current_batch['y']
        
        # Reconstruct the original unmasked text
        original_sample_tokens = self.reconstruct_original_text(
            x_tensor[self.current_sample_idx], 
            y_tensor[self.current_sample_idx]
        )
        base_sample = original_sample_tokens.unsqueeze(0)  # Add batch dimension
        original_text = self.decode_tokens(original_sample_tokens)
        
        print(f"📝 Base sample: {repr(original_text)}")
        
        # Select masking strategy
        strategies = [
            "Random masking",
            "Sticky masking", 
            "Span masking"
        ]
        
        strategy_choice = self.get_menu_choice(strategies, "Select masking strategy:")
        if strategy_choice == 0:
            return
            
        print(f"\n🎯 Selected: {strategies[strategy_choice - 1]}")
        
        try:
            # Create random number generator for consistent results
            rng = torch.Generator()
            rng.manual_seed(42)  # Fixed seed for reproducible results
            
            if strategy_choice == 1:  # Random masking
                print("\nRandom Masking Parameters:")
                mask_prob = float(input("Mask probability (0.0-1.0, default 0.3): ") or "0.3")
                
                masked_x, mask = apply_random_masking_cpu(
                    base_sample, mask_prob, self.mask_token_id, self.vocab_size - 1, rng
                )
                
            elif strategy_choice == 2:  # Sticky masking
                print("\nSticky Masking Parameters:")
                target_ratio = float(input("Target masked ratio (0.0-1.0, default 0.3): ") or "0.3")
                p1_prob = float(input("P1 probability (default 0.1): ") or "0.1")
                p2_prob = float(input("P2 probability (default 0.7): ") or "0.7")
                
                masked_x, mask = apply_target_driven_sticky_masking_cpu(
                    base_sample, target_ratio, p1_prob, p2_prob, 
                    self.mask_token_id, self.vocab_size - 1, rng
                )
                
            elif strategy_choice == 3:  # Span masking
                print("\nSpan Masking Parameters:")
                spans_count = int(input("Number of spans (default 3): ") or "3")
                
                masked_x, mask = apply_span_masking_cpu(
                    base_sample, spans_count, self.mask_token_id, self.vocab_size - 1, rng
                )
            
            # Show results
            self.print_header("Generated Test Sample")
            
            masked_text = self.decode_tokens(masked_x[0])
            mask_count = mask.sum().item()
            total_tokens = mask.numel()
            mask_percentage = (mask_count / total_tokens) * 100
            
            print(f"📊 Masking Statistics:")
            print(f"   • Strategy: {strategies[strategy_choice - 1]}")
            print(f"   • Masked tokens: {mask_count}/{total_tokens} ({mask_percentage:.1f}%)")
            
            print(f"\n📝 Original: {repr(original_text)}")
            print(f"🎭 Masked:   {repr(masked_text)}")
            
            # Show masked positions
            masked_positions = mask[0].nonzero(as_tuple=True)[0].tolist()
            print(f"\n🎯 Masked positions: {masked_positions}")
            
            # Option to unmask with model
            print(f"\nOptions:")
            print(f"  U - Run model unmasking on this sample")
            print(f"  Any other key - Return to navigation")
            
            key = self.wait_for_key("Enter choice: ").lower()
            
            if key == 'u':
                self.unmask_generated_sample(masked_x, mask, original_sample_tokens)
                
        except Exception as e:
            print(f"❌ Error generating test sample: {e}")
            self.wait_for_key()
    
    def unmask_generated_sample(self, masked_tokens: torch.Tensor, mask: torch.Tensor, original_tokens: torch.Tensor):
        """Unmask a generated test sample with color coding"""
        self.print_header("Unmasking Generated Sample")
        
        try:
            masked_tokens = masked_tokens.to(DEVICE)
            
            print(f"🔄 Running model unmasking...")
            input_tokens = masked_tokens[0]
            input_text = self.decode_tokens(input_tokens)
            total_tokens = len(input_tokens)
            print(f"📝 Input (all {total_tokens} tokens):")
            print(f"    {repr(input_text)}")
            print()
            
            with torch.no_grad():
                with self.ctx:
                    # Get model predictions - pass dummy targets to get full sequence logits
                    dummy_targets = torch.zeros_like(masked_tokens)
                    model_output = self.model(masked_tokens, targets=dummy_targets)
                    
                    # Handle different model output formats
                    if isinstance(model_output, tuple):
                        logits = model_output[0]  # First element is usually logits
                    else:
                        logits = model_output
                    
                    temperature = 0.8
                    scaled_logits = logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Sample predictions for masked positions
                    result_tokens = masked_tokens[0].clone()
                    masked_positions = mask[0].nonzero(as_tuple=True)[0]
                    
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for pos in masked_positions:
                        pos_probs = probs[0, pos, :self.vocab_size-1]  # Exclude mask token
                        predicted_token = torch.multinomial(pos_probs, 1).item()
                        result_tokens[pos] = predicted_token
                        
                        # Check if prediction matches original
                        total_predictions += 1
                        if predicted_token == original_tokens[pos].item():
                            correct_predictions += 1
                    
                    print(f"✅ Unmasking complete!")
                    if total_predictions > 0:
                        accuracy = (correct_predictions / total_predictions) * 100
                        print(f"🎯 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
                    print()
                    
                    # Create colored result text
                    ignore_index = -100  # Not used here but needed for consistency with create_colored_result
                    colored_result = self.create_colored_result(
                        masked_tokens[0], result_tokens, original_tokens, masked_positions, ignore_index
                    )
                    print(f"📄 Result with color coding:")
                    total_result_tokens = len(result_tokens)
                    print(f"    {colored_result}")
                    print(f"    Total tokens: {total_result_tokens}")
                    print()
                    print(f"    \033[32m🟢 = Correct prediction\033[0m  \033[31m🔴 = Incorrect prediction\033[0m  ⚪ = Original text")
                    
        except Exception as e:
            print(f"❌ Error during unmasking: {e}")
            
        self.wait_for_key()
    
    def main_menu(self):
        """Main application loop"""
        while True:
            options = []
            
            if self.model is None:
                options.append("🔧 Load Model")
            else:
                options.append(f"✅ Model loaded ({self.model.get_num_params()/1e6:.1f}M params)")
                
            if self.model is not None:
                if self.dataset_name is None:
                    options.append("📁 Select Dataset")
                else:
                    options.append(f"📁 Change Dataset (Current: {self.dataset_name})")
                    
                if self.dataset_name is not None:
                    if self.current_batch is None:
                        options.append("📂 Select Data File")
                    else:
                        batch_info = f"📂 Change Data File (Current: {self.current_batch['x'].shape[0]} samples)"
                        options.append(batch_info)
                        
                    if self.current_batch is not None:
                        options.append("🔍 Navigate Samples")
                        
            if self.model is not None and self.dataset_name is not None:
                options.append("🧪 Generate Test Sample")
            
            choice = self.get_menu_choice(options, "🔬 Diffusion Model Explorer - Main Menu")
            
            if choice == 0:
                break
                
            if "Load Model" in options[choice - 1]:
                if self.load_model():
                    continue
            elif "Select Dataset" in options[choice - 1] or "Change Dataset" in options[choice - 1]:
                if self.select_dataset():
                    continue
            elif "Select Data File" in options[choice - 1] or "Change Data File" in options[choice - 1]:
                if self.select_data_file():
                    continue
            elif "Navigate Samples" in options[choice - 1]:
                self.navigate_samples()
            elif "Generate Test Sample" in options[choice - 1]:
                self.generate_test_sample()
        
        self.print_header("Goodbye!")
        print("Thank you for using Diffusion Model Explorer!")


def main():
    """Main entry point"""
    if not HAS_KEYBOARD:
        print("⚠️  Warning: Keyboard module not available. Navigation will be limited.")
        print("Install 'keyboard' package for better navigation experience.")
        print()
    
    explorer = DiffusionExplorer()
    
    try:
        explorer.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()