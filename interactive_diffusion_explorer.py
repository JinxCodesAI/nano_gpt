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
from sample_utils import build_critic_artifacts_from_logits


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
MODEL_PATH = 'out-char-diffusion/a40_3750_01_10.pt'  # Hardcoded model path - change this as needed
DATA_DIR = 'data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'float16' if DEVICE == 'cuda' else 'float32'

class DiffusionExplorer:
    def __init__(self, interactive: bool = True):
        self.interactive = bool(interactive)
        self.model = None
        self.stoi = None
        self.itos = None
        self.vocab_size = None  # model vocab size
        self.mask_token_id = None
        self.dataset_name = None
        self.current_batch = None
        self.current_sample_idx = 0
        # dataset/meta-related fields
        self.meta = None
        self.training_type = None
        self.schema = None
        self.input_field = None
        self.target_field = None
        self.file_metadata = None
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
        """Wait for user input. In non-interactive mode, no-op."""
        if not self.interactive:
            # Non-interactive mode: skip waiting
            return ""
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
        """Load dataset meta/schema and reconcile vocabulary with model"""
        self.print_header(f"Loading Dataset Meta: {self.dataset_name}")

        try:
            meta_path = os.path.join(DATA_DIR, self.dataset_name, 'meta.pkl')
            print(f"📂 Loading meta from: {meta_path}")

            with open(meta_path, 'rb') as f:
                self.meta = pickle.load(f)

            # Basic meta
            self.training_type = self.meta.get('training_type')
            self.schema = self.meta.get('batch_schema')
            self.stoi = self.meta.get('stoi')
            self.itos = self.meta.get('itos')
            dataset_vocab_size = self.meta.get('vocab_size')

            # Determine special tokens from meta if available
            self.mask_token_id = self.meta.get('mask_token_id', None)
            self.sep_token_id = self.meta.get('sep_token_id', None)
            self.cls_token_id = self.meta.get('cls_token_id', None)
            self.pad_token_id = self.meta.get('pad_token_id', None)

            # Assert that all token IDs are unique
            token_ids = []
            token_names = []
            if self.mask_token_id is not None:
                token_ids.append(self.mask_token_id)
                token_names.append('mask_token_id')
            if self.cls_token_id is not None:
                token_ids.append(self.cls_token_id)
                token_names.append('cls_token_id')
            if self.pad_token_id is not None:
                token_ids.append(self.pad_token_id)
                token_names.append('pad_token_id')
            if self.sep_token_id is not None:
                token_ids.append(self.sep_token_id)
                token_names.append('sep_token_id')

            # Check for duplicates
            if len(token_ids) != len(set(token_ids)):
                duplicates = []
                for i, token_id in enumerate(token_ids):
                    for j, other_id in enumerate(token_ids):
                        if i != j and token_id == other_id:
                            duplicates.append(f"{token_names[i]}={token_id} == {token_names[j]}={other_id}")
                raise ValueError(f"Duplicate token IDs found: {duplicates}")

            cls_token_id = self.cls_token_id

            # Model vocab
            model_vocab_size = getattr(self.model.config, 'vocab_size', None)
            if model_vocab_size is None:
                raise ValueError("Model missing vocab_size in config")
            self.vocab_size = model_vocab_size

            # Report
            print(f"✅ Meta loaded:")
            print(f"   • training_type: {self.training_type}")
            print(f"   • Dataset vocab size: {dataset_vocab_size}")
            print(f"   • Model vocab size: {model_vocab_size}")
            if self.mask_token_id is not None:
                print(f"   • mask_token_id (from meta): {self.mask_token_id}")
            if self.cls_token_id is not None:
                print(f"   • cls_token_id (from meta): {self.cls_token_id}")
            if self.pad_token_id is not None:
                print(f"   • pad_token_id (from meta): {self.pad_token_id}")

            # If mask_token_id not in meta, assume last id in model vocab for display/decoding purposes
            if self.mask_token_id is None:
                self.mask_token_id = self.vocab_size - 1

            # Minimal schema introspection for input/target field names
            self.input_field, self.target_field = None, None
            if isinstance(self.schema, list):
                for field in self.schema:
                    role = field.get('role') or ''
                    if role.lower() == 'input' and self.input_field is None:
                        self.input_field = field.get('name')
                    if role.lower() == 'target' and self.target_field is None:
                        self.target_field = field.get('name')
            # Fallbacks by common names
            if self.input_field is None:
                for candidate in ['x', 'input_ids']:
                    if candidate in (f.get('name') for f in self.schema):
                        self.input_field = candidate
                        break
            if self.target_field is None:
                for candidate in ['y', 'targets']:
                    if candidate in (f.get('name') for f in self.schema):
                        self.target_field = candidate
                        break

            print(f"   • Detected input field: {self.input_field}")
            print(f"   • Detected target field: {self.target_field}")

            self.wait_for_key()
            return True

        except Exception as e:
            print(f"❌ Error loading meta: {e}")
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
                self.file_metadata = metadata
                print(f"📋 Batch metadata found:")
                for key, value in metadata.items():
                    if key not in ['tensors', 'stage_info', 'masking_strategy', 'masking_ratio']:  # Skip tensor data and verbose/long lists
                        print(f"     {key}: {value}")
                # Show stage info summary if present
                if 'stage_info' in metadata:
                    stage_info = metadata['stage_info']
                    if isinstance(stage_info, list):
                        print(f"     stage_info: {len(stage_info)} samples with stage configurations")
                    else:
                        print(f"     stage_info: {stage_info}")
                # Show masking strategy summary if present
                if 'masking_strategy' in metadata and isinstance(metadata['masking_strategy'], list):
                    print(f"     masking_strategy: {len(metadata['masking_strategy'])} per-sample labels")
                # Show masking ratio summary if present
                if 'masking_ratio' in metadata and isinstance(metadata['masking_ratio'], list):
                    print(f"     masking_ratio: {len(metadata['masking_ratio'])} per-sample floats")

            # Check for generation info
            if 'generation_info' in batch_data:
                gen_info = batch_data['generation_info']
                print(f"⚙️ Generation info: {gen_info}")

            # List all keys in batch data for debugging
            print(f"🔍 Available keys in batch: {list(batch_data.keys())}")

            # Extract tensors using schema/meta to support multiple dataset types
            input_name = self.input_field or 'x'
            target_name = self.target_field or 'y'

            def _get_tensor(container, name):
                if isinstance(container, dict):
                    return container.get(name)
                return None

            def _pick_from(container, names: List[str]):
                for n in names:
                    v = _get_tensor(container, n)
                    if v is not None:
                        return v
                return None

            tensors = batch_data.get('tensors', batch_data)
            x_tensor = _pick_from(tensors, [input_name, 'x', 'input_ids'])
            y_tensor = _pick_from(tensors, [target_name, 'y', 'targets'])
            attn_tensor = _pick_from(tensors, ['attention_mask', 'attn', 'mask'])

            if x_tensor is None or y_tensor is None:
                print("❌ Could not find required tensors in batch file per schema")
                print("Available top-level keys:", list(batch_data.keys()))
                if isinstance(tensors, dict):
                    print("Tensors keys:", list(tensors.keys()))
                print(f"Expected input: {input_name}, target: {target_name}")
                self.wait_for_key()
                return False

            batch_dict = {'input': x_tensor, 'target': y_tensor, 'input_name': input_name, 'target_name': target_name}
            if attn_tensor is not None:
                batch_dict['attention_mask'] = attn_tensor
            self.current_batch = batch_dict
            self.current_sample_idx = 0

            batch_size, seq_len = x_tensor.shape
            print(f"✅ Batch file loaded:")
            print(f"   • Batch size: {batch_size}")
            print(f"   • Sequence length: {seq_len}")
            # Attention mask stats if present
            if attn_tensor is not None:
                total_ones = int(attn_tensor.sum().item())
                total_elems = int(attn_tensor.numel())
                total_zeros = total_elems - total_ones
                print(f" attention_mask: ones={total_ones}, zeros={total_zeros} (visible avg per sample: {total_ones/max(batch_size,1):.1f})")

            # Show statistics depending on target shape
            ignore_index = -100
            if y_tensor.dim() == 2 and y_tensor.shape[1] == seq_len:
                total_tokens = batch_size * seq_len
                masked_tokens = (y_tensor != ignore_index).sum().item()
                mask_percentage = (masked_tokens / max(total_tokens, 1)) * 100

                per_sample_masked = (y_tensor != ignore_index).sum(dim=1).float()
                per_sample_percentages = (per_sample_masked / max(seq_len, 1)) * 100
                sorted_percentages, _ = torch.sort(per_sample_percentages)
                min_masked_pct = sorted_percentages[0].item()
                max_masked_pct = sorted_percentages[-1].item()
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
            else:
                # Scalar/regression targets
                try:
                    targets_flat = y_tensor.view(-1).to(torch.float32)
                    print(f"   • Targets stats (min/mean/max): {targets_flat.min().item():.4f} / {targets_flat.mean().item():.4f} / {targets_flat.max().item():.4f}")
                except Exception:
                    print(f"   • Targets shape: {tuple(y_tensor.shape)}")

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
            if self.mask_token_id is not None and token_id == self.mask_token_id:
                result.append('[MASK]')
            elif getattr(self, 'cls_token_id', None) is not None and token_id == self.cls_token_id:
                result.append('[CLS]')
            elif getattr(self, 'pad_token_id', None) is not None and token_id == self.pad_token_id:
                result.append('[PAD]')
            elif getattr(self, 'sep_token_id', None) is not None and token_id == self.sep_token_id:
                result.append('[SEP]')
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

        x_tensor = self.current_batch['input']
        y_tensor = self.current_batch['target']
        batch_size = x_tensor.shape[0]
        ignore_index = -100

        # Determine dataset style from targets shape
        is_token_targets = (y_tensor.dim() == 2 and y_tensor.shape[1] == x_tensor.shape[1])

        while True:
            self.print_header(f"Sample Navigation: {self.dataset_name}")

            attn = self.current_batch.get('attention_mask', None)
            if attn is not None:
                seq_len_display = int(attn[self.current_sample_idx].sum().item())
                ones = int(attn[self.current_sample_idx].sum().item())
                zeros = int(attn.shape[1] - ones)
            else:
                seq_len_display = x_tensor.shape[1]
                ones = seq_len_display
                zeros = 0
            print(f"📊 Current sample: {self.current_sample_idx + 1}/{batch_size} (sequence length: {seq_len_display})")
            # Display masking strategy per sample, if present in metadata
            try:
                if isinstance(self.file_metadata, dict) and 'masking_strategy' in self.file_metadata:
                    ms = self.file_metadata['masking_strategy']
                    if isinstance(ms, list) and len(ms) > self.current_sample_idx:
                        print(f"   • Masking: {ms[self.current_sample_idx]}")
            except Exception:
                pass
            if attn is not None:
                print(f"   • attention_mask -> ones: {ones}, zeros: {zeros}")
            if is_token_targets:
                print(f"🔧 Navigation: ← Previous | → Next | U Unmask | G Generate | Q Quit")
            else:
                print(f"🔧 Navigation: ← Previous | → Next | Q Quit")
            print("-" * 80)

            # Show current sample
            attn = self.current_batch.get('attention_mask', None)
            if attn is not None:
                valid_len = int(attn[self.current_sample_idx].sum().item())
                x_tokens = x_tensor[self.current_sample_idx, :valid_len]
            else:
                x_tokens = x_tensor[self.current_sample_idx]
            x_decoded = self.decode_tokens(x_tokens)
            print(f"📝 Input ({self.current_batch['input_name']}):  {repr(x_decoded)}")

            if is_token_targets:
                # Show per-token targets for masked positions
                y_tokens = y_tensor[self.current_sample_idx]
                y_decoded_parts = []
                masked_positions = []
                # Limit target iteration to visible region when attention_mask is provided
                limit = x_tokens.shape[0]
                for j, (x_tok, y_tok) in enumerate(zip(x_tokens.tolist(), y_tokens.tolist()[:limit])):
                    if y_tok != ignore_index:
                        y_decoded_parts.append(self.itos.get(y_tok, f'[UNK:{y_tok}]'))
                        masked_positions.append(j)
                    else:
                        y_decoded_parts.append('_')
                print(f"🎯 Target ({self.current_batch['target_name']}): {''.join(y_decoded_parts)}")
                print(f"🎭 Masked positions: {len(masked_positions)} positions")
            else:
                target_val = y_tensor[self.current_sample_idx].item() if y_tensor.dim() == 1 else y_tensor[self.current_sample_idx].squeeze().item()
                ratio_txt = ""
                try:
                    if isinstance(self.file_metadata, dict) and 'masking_ratio' in self.file_metadata:
                        mr = self.file_metadata['masking_ratio']
                        if isinstance(mr, list) and len(mr) > self.current_sample_idx:
                            ratio_txt = f" Masking ratio {float(mr[self.current_sample_idx]):.2f}"
                except Exception:
                    pass
                print(f"🎯 Target ({self.current_batch['target_name']}): {target_val:.4f}{ratio_txt}")

            print(f"\nCommands:")
            print(f"  ← / A - Previous sample")
            print(f"  → / D - Next sample")
            if is_token_targets:
                print(f"  U - Run model unmasking on this sample")
                print(f"  C - Critic view for this sample")
                print(f"  G - Generate test sample")
            print(f"  Q - Quit to main menu")

            key = self.wait_for_key("Enter command: ").lower()

            if key in ['q']:
                break
            elif key in ['a'] or (HAS_KEYBOARD and key == '\x1b'):  # Left arrow or A
                self.current_sample_idx = (self.current_sample_idx - 1) % batch_size
            elif key in ['d'] or (HAS_KEYBOARD and key == '\x1b'):  # Right arrow or D
                self.current_sample_idx = (self.current_sample_idx + 1) % batch_size
            elif key in ['u'] and is_token_targets:
                self.run_model_unmasking()
            elif key in ['c'] and is_token_targets:
                self.run_critic_view()
            elif key in ['g'] and is_token_targets:
                self.generate_test_sample()

    def run_model_unmasking(self):
        """Run model unmasking on current sample"""
        if self.current_batch is None:
            print("❌ No batch loaded")
            self.wait_for_key()
            return

        self.print_header("Model Unmasking")

        x_tensor = self.current_batch['input']
        y_tensor = self.current_batch['target']
        # Enforce only MLM-style is supported here
        if not (y_tensor.dim() == 2 and y_tensor.shape[1] == x_tensor.shape[1]):
            print("Unmasking only supported for MLM-style datasets with per-token targets.")
            self.wait_for_key()
            return
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
                    # Pass attention_mask if available
                    attn_full = self.current_batch.get('attention_mask', None)
                    attn_sample = None
                    if attn_full is not None:
                        attn_sample = attn_full[self.current_sample_idx:self.current_sample_idx+1].to(DEVICE)
                    model_output = self.model(sample_tokens, targets=dummy_targets, attention_mask=attn_sample)

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
                    y_tensor = self.current_batch['target']
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


    def run_critic_view(self):
        """Inspect critic inputs/targets/predictions for current sample using shared builder."""
        if self.current_batch is None or self.model is None:
            print("❌ No batch/model loaded")
            self.wait_for_key()
            return

        self.print_header("Critic View (per-sample)")

        x_tensor = self.current_batch['input']
        y_tensor = self.current_batch['target']
        # Only MLM-style supported (token targets)
        if not (y_tensor.dim() == 2 and y_tensor.shape[1] == x_tensor.shape[1]):
            print("Critic view is only available for MLM-style datasets with per-token targets.")
            self.wait_for_key()
            return

        sample_tokens = x_tensor[self.current_sample_idx:self.current_sample_idx+1].to(DEVICE)
        targets = y_tensor[self.current_sample_idx:self.current_sample_idx+1].to(DEVICE)

        # Resolve IDs from model config with fallbacks to dataset meta for display
        cfg = getattr(self.model, 'config', object())
        mask_token_id = getattr(cfg, 'mask_token_id', None)
        if mask_token_id is None:
            mask_token_id = self.mask_token_id
        pad_token_id = getattr(cfg, 'pad_token_id', getattr(self, 'pad_token_id', None))
        ignore_index = int(getattr(cfg, 'ignore_index', -100))
        scope = getattr(cfg, 'critic_target_scope', 'masked_and_ignore')

        try:
            with torch.no_grad():
                with self.ctx:
                    dummy_targets = torch.zeros_like(sample_tokens)
                    model_output = self.model(sample_tokens, targets=dummy_targets)
                    logits = model_output[0] if isinstance(model_output, tuple) else model_output

            # Build artifacts via shared helper
            artifacts = build_critic_artifacts_from_logits(
                idx=sample_tokens,
                logits=logits,
                targets=targets,
                mask_token_id=int(mask_token_id),
                ignore_index=ignore_index,
                pad_token_id=pad_token_id,
                scope=scope,
            )
            critic_input = artifacts['critic_input']
            critic_target = artifacts['critic_target']
            critic_valid = artifacts['critic_valid']
            pred_tokens = artifacts['pred_tokens']

            # Prepare critic predictions if critic head is available
            has_critic = getattr(cfg, 'add_critic_head', False) and hasattr(self.model, 'critic_head')
            probs = None
            if has_critic:
                critic_logits = self.model.critic_scores(critic_input)
                probs = torch.sigmoid(critic_logits)

            # Display decoded sequences (no repr; allow real newlines)
            original_text = self.decode_tokens(sample_tokens[0])
            critic_text = self.decode_tokens(critic_input[0])
            print("📝 Input:")
            print(original_text)
            print()
            print("🧪 Critic input (filled masks with LM samples):")
            # Color strictly by critic target: green=0, red=1, yellow=invalid
            cls_id = getattr(self, 'cls_token_id', None)
            sep_id = getattr(self, 'sep_token_id', None)
            def _tok_str(tid: int) -> str:
                if mask_token_id is not None and tid == mask_token_id:
                    return '[MASK]'
                if cls_id is not None and tid == cls_id:
                    return '[CLS]'
                if pad_token_id is not None and tid == pad_token_id:
                    return '[PAD]'
                if sep_id is not None and tid == sep_id:
                    return '[SEP]'
                return self.itos[tid] if tid < len(self.itos) else f'[UNK:{tid}]'
            row_inp = critic_input[0]
            row_tgt = critic_target[0]
            row_val = critic_valid[0]
            parts = []
            for i in range(row_inp.shape[0]):
                tid = int(row_inp[i].item())
                s = _tok_str(tid)
                if not bool(row_val[i].item()):
                    parts.append(f"\033[33m{s}\033[0m")  # yellow = invalid
                elif int(row_tgt[i].item()) == 0:
                    parts.append(f"\033[32m{s}\033[0m")  # green = target 0
                else:
                    parts.append(f"\033[31m{s}\033[0m")  # red = target 1
            print(''.join(parts))
            print("    \033[32m🟢 target=0\033[0m  \033[31m🔴 target=1\033[0m  \033[33m🟡 invalid\033[0m")
            print()

            # Summaries
            masked_positions = (sample_tokens[0] == mask_token_id)
            masked_cnt = int(masked_positions.sum().item())
            ignore_cnt = int((targets[0] == ignore_index).sum().item())
            valid_cnt = int(critic_valid[0].sum().item())
            zeros_cnt = int((critic_valid[0] & (critic_target[0] == 0)).sum().item())
            ones_cnt = int((critic_valid[0] & (critic_target[0] == 1)).sum().item())
            print(f"📊 Counts: masked={masked_cnt}, ignore={ignore_cnt}, valid={valid_cnt}, target0={zeros_cnt}, target1={ones_cnt}")

            # If critic head available, compute probabilities and basic stats
            has_critic = getattr(cfg, 'add_critic_head', False) and hasattr(self.model, 'critic_head')
            if has_critic:
                # probs computed earlier
                t0_mask = critic_valid & (critic_target == 0)
                t1_mask = critic_valid & (critic_target == 1)
                def _percentiles(vals: torch.Tensor):
                    vals = vals.view(-1)
                    if vals.numel() == 0:
                        return float('nan'), float('nan'), float('nan')
                    mean = float(vals.mean().item())
                    try:
                        p10 = float(torch.quantile(vals, torch.tensor(0.1)).item())
                        p90 = float(torch.quantile(vals, torch.tensor(0.9)).item())
                    except Exception:
                        sorted_vals, _ = torch.sort(vals)
                        n = sorted_vals.numel()
                        i10 = max(int(0.1 * (n - 1)), 0)
                        i90 = max(int(0.9 * (n - 1)), 0)
                        p10 = float(sorted_vals[i10].item())
                        p90 = float(sorted_vals[i90].item())
                    return mean, p10, p90
                if t0_mask.any():
                    m0, p10_0, p90_0 = _percentiles(probs[t0_mask])
                    print(f"🔵 Critic probs for target0: mean={m0:.4f}, p10={p10_0:.4f}, p90={p90_0:.4f}")
                else:
                    print("🔵 Critic probs for target0: n/a")
                if t1_mask.any():
                    m1, p10_1, p90_1 = _percentiles(probs[t1_mask])
                    print(f"🟠 Critic probs for target1: mean={m1:.4f}, p10={p10_1:.4f}, p90={p90_1:.4f}")
                else:
                    print("🟠 Critic probs for target1: n/a")

            # Grid views (no truncation)
            seq_len = critic_target.shape[1]
            cols = 64  # reasonable width for console
            print()
            print("🎯 Critic target grid ( - = invalid; color indicates match vs prediction when critic is present ):")
            vt_bool = critic_valid[0].bool()
            tt_int = critic_target[0].to(torch.int)
            use_preds = (probs is not None)
            pred_bin = (probs[0] > 0.5).to(torch.int) if use_preds else None
            for start in range(0, seq_len, cols):
                end = min(start + cols, seq_len)
                row_parts = []
                for i in range(start, end):
                    if not vt_bool[i]:
                        row_parts.append("\033[33m-\033[0m")  # yellow invalid
                    else:
                        ch = '0' if tt_int[i] == 0 else '1'
                        if use_preds:
                            ok = (int(pred_bin[i].item()) == int(tt_int[i].item()))
                            row_parts.append(f"\033[32m{ch}\033[0m" if ok else f"\033[31m{ch}\033[0m")
                        else:
                            # fallback: class color if no critic
                            row_parts.append("\033[32m0\033[0m" if tt_int[i] == 0 else "\033[31m1\033[0m")
                row = ''.join(row_parts)
                print(f"[{start:04d}-{end-1:04d}] {row}")

            # Also show masked positions as a grid (M = masked, . = not masked)
            print()
            print("🎭 Masked positions (M = masked, . = not masked):")
            mp_bool = masked_positions.bool()
            for start in range(0, seq_len, cols):
                end = min(start + cols, seq_len)
                row = ''.join('M' if mp_bool[i] else '.' for i in range(start, end))
                print(f"[{start:04d}-{end-1:04d}] {row}")

        except Exception as e:
            print(f"❌ Error during critic view: {e}")

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

        # Only supported for MLM-style datasets with per-token targets
        x_tensor = self.current_batch['input']
        y_tensor = self.current_batch['target']
        if not (y_tensor.dim() == 2 and y_tensor.shape[1] == x_tensor.shape[1]):
            print("This action is only available for MLM-style datasets with token targets.")
            self.wait_for_key()
            return

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

            # Determine content (non-PAD) length; only mask within content
            pad_token_id = getattr(self, 'pad_token_id', None)
            seq_len = int(base_sample.shape[1])
            if pad_token_id is not None:
                pad_pos = (base_sample[0] == pad_token_id).nonzero(as_tuple=True)[0]
                content_len = int(pad_pos[0].item()) if pad_pos.numel() > 0 else seq_len
            else:
                content_len = seq_len
            if content_len <= 0:
                print("No content tokens before [PAD]; nothing to mask.")
                self.wait_for_key()
                return

            content_only = base_sample[:, :content_len]

            # Apply masking to content portion only
            if strategy_choice == 1:  # Random masking
                print("\nRandom Masking Parameters:")
                mask_prob = float(input("Mask probability (0.0-1.0, default 0.3): ") or "0.3")

                masked_content, mask_content = apply_random_masking_cpu(
                    content_only, mask_prob, self.mask_token_id, self.vocab_size - 1, rng
                )

            elif strategy_choice == 2:  # Sticky masking
                print("\nSticky Masking Parameters:")
                target_ratio = float(input("Target masked ratio (0.0-1.0, default 0.3): ") or "0.3")
                p1_prob = float(input("P1 probability (default 0.1): ") or "0.1")
                p2_prob = float(input("P2 probability (default 0.7): ") or "0.7")

                masked_content, mask_content = apply_target_driven_sticky_masking_cpu(
                    content_only, target_ratio, p1_prob, p2_prob,
                    self.mask_token_id, self.vocab_size - 1, rng
                )

            elif strategy_choice == 3:  # Span masking
                print("\nSpan Masking Parameters:")
                spans_count = int(input("Number of spans (default 3): ") or "3")

                masked_content, mask_content = apply_span_masking_cpu(
                    content_only, spans_count, self.mask_token_id, self.vocab_size - 1, rng
                )

            # Reconstruct full-length outputs with PAD preserved and mask limited to content
            masked_x_full = base_sample.clone()
            masked_x_full[:, :content_len] = masked_content
            mask_full = torch.zeros_like(base_sample, dtype=torch.bool)
            mask_full[:, :content_len] = mask_content

            # Show results
            self.print_header("Generated Test Sample")

            masked_text = self.decode_tokens(masked_x_full[0])
            mask_count = int(mask_full.sum().item())
            total_tokens = int(content_len)
            mask_percentage = (mask_count / max(total_tokens, 1)) * 100.0

            print(f"📊 Masking Statistics:")
            print(f"   • Strategy: {strategies[strategy_choice - 1]}")
            print(f"   • Masked tokens: {mask_count}/{total_tokens} ({mask_percentage:.1f}%)")

            print(f"\n📝 Original: {repr(original_text)}")
            print(f"🎭 Masked:   {repr(masked_text)}")

            # Show masked positions (global indices)
            masked_positions = mask_full[0].nonzero(as_tuple=True)[0].tolist()
            print(f"\n🎯 Masked positions: {masked_positions}")

            # Option to unmask with model
            print(f"\nOptions:")
            print(f"  U - Run model unmasking on this sample")
            print(f"  Any other key - Return to navigation")

            key = self.wait_for_key("Enter choice: ").lower()

            if key == 'u':
                self.unmask_generated_sample(masked_x_full, mask_full, original_sample_tokens)
                return

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

    def validate_file(self):
        """Validate the current batch file for correctness"""
        if self.current_batch is None:
            print("❌ No batch loaded")
            self.wait_for_key()
            return

        self.print_header("File Validation")

        x_tensor = self.current_batch['input']
        y_tensor = self.current_batch['target']
        attn_tensor = self.current_batch.get('attention_mask', None)

        batch_size, seq_len = x_tensor.shape
        ignore_index = -100

        print(f"📊 Validating batch file:")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Sequence length: {seq_len}")
        print()

        # Validation checks
        validation_passed = True

        # 1) X, targets, and masks have all same length
        print("1️⃣ Checking tensor dimensions...")
        if y_tensor.shape != x_tensor.shape:
            print(f"   ❌ Shape mismatch: X={x_tensor.shape}, Y={y_tensor.shape}")
            validation_passed = False
        else:
            print(f"   ✅ X and Y shapes match: {x_tensor.shape}")

        if attn_tensor is not None:
            if attn_tensor.shape != x_tensor.shape:
                print(f"   ❌ Attention mask shape mismatch: X={x_tensor.shape}, attention_mask={attn_tensor.shape}")
                validation_passed = False
            else:
                print(f"   ✅ Attention mask shape matches: {attn_tensor.shape}")
        else:
            print(f"   ⚠️  No attention mask found")

        # 2) Length is constant for all samples in a file
        print("\n2️⃣ Checking sequence length consistency...")
        # All tensors must have identical shape - this is always required
        x_shapes_consistent = all(x_tensor[i].shape == x_tensor[0].shape for i in range(batch_size))
        y_shapes_consistent = all(y_tensor[i].shape == y_tensor[0].shape for i in range(batch_size))

        if not x_shapes_consistent:
            print(f"   ❌ X tensor shapes are inconsistent across samples")
            validation_passed = False
        elif not y_shapes_consistent:
            print(f"   ❌ Y tensor shapes are inconsistent across samples")
            validation_passed = False
        else:
            print(f"   ✅ All samples have consistent tensor shape: {x_tensor[0].shape}")

        if attn_tensor is not None:
            attn_shapes_consistent = all(attn_tensor[i].shape == attn_tensor[0].shape for i in range(batch_size))
            if not attn_shapes_consistent:
                print(f"   ❌ Attention mask shapes are inconsistent across samples")
                validation_passed = False

        # 3) Check [PAD] token presence and structure
        print("\n3️⃣ Checking [PAD] token structure...")
        pad_token_id = getattr(self, 'pad_token_id', None)
        if pad_token_id is not None:
            pad_counts = (x_tensor == pad_token_id).sum(dim=1)
            samples_with_no_pad = (pad_counts == 0).sum().item()
            samples_with_pad = (pad_counts > 0).sum().item()

            print(f"   • Samples with no [PAD]: {samples_with_no_pad}/{batch_size}")
            print(f"   • Samples with [PAD]: {samples_with_pad}/{batch_size}")

            # Check that [PAD] tokens are at the end
            pad_structure_violations = 0
            for b in range(batch_size):
                if pad_counts[b] > 0:
                    # Find first [PAD] position
                    pad_positions = (x_tensor[b] == pad_token_id).nonzero(as_tuple=True)[0]
                    first_pad = pad_positions[0].item()
                    # Check if all positions from first_pad onwards are [PAD]
                    remaining_positions = x_tensor[b, first_pad:]
                    if not (remaining_positions == pad_token_id).all():
                        pad_structure_violations += 1

            if pad_structure_violations > 0:
                print(f"   ❌ {pad_structure_violations} samples have non-contiguous [PAD] tokens")
                validation_passed = False
            else:
                print(f"   ✅ All [PAD] tokens are properly positioned at sequence ends")
        else:
            print(f"   ⚠️  [PAD] token ID not found in meta")

        # 4) Target has non-zero number of elements with value not equal to ignore_index
        print("\n4️⃣ Checking target supervision...")
        supervised_counts = (y_tensor != ignore_index).sum(dim=1)
        samples_with_supervision = (supervised_counts > 0).sum().item()
        min_supervised = supervised_counts.min().item()
        max_supervised = supervised_counts.max().item()

        print(f"   • Samples with supervision: {samples_with_supervision}/{batch_size}")
        print(f"   • Supervised tokens per sample: min={min_supervised}, max={max_supervised}")

        if samples_with_supervision != batch_size:
            print(f"   ❌ Not all samples have supervised targets")
            validation_passed = False
        elif min_supervised == 0:
            print(f"   ❌ Some samples have zero supervised targets")
            validation_passed = False
        else:
            print(f"   ✅ All samples have supervised targets")

        # 5) Check target positions (should not be in [PAD] regions)
        print("\n5️⃣ Checking target positions...")
        if pad_token_id is not None:
            # Check if any supervised targets are in [PAD] regions
            violations = 0
            for b in range(batch_size):
                # Find [PAD] positions
                pad_positions = (x_tensor[b] == pad_token_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    # Check if any supervised targets are in [PAD] positions
                    supervised_positions = (y_tensor[b] != ignore_index).nonzero(as_tuple=True)[0]
                    pad_set = set(pad_positions.tolist())
                    supervised_set = set(supervised_positions.tolist())
                    if pad_set.intersection(supervised_set):
                        violations += 1

            print(f"   • Samples with targets in [PAD] regions: {violations}/{batch_size}")
            if violations > 0:
                print(f"   ❌ Some targets are positioned in [PAD] regions")
                validation_passed = False
            else:
                print(f"   ✅ No targets in [PAD] regions")
        else:
            print(f"   ✅ No [PAD] token validation needed")

        # 6) Check attention mask structure (should not be present in new approach)
        print("\n6️⃣ Checking attention mask structure...")
        if attn_tensor is not None:
            print(f"   ⚠️  Attention mask present but not expected in [PAD] token approach")
            print(f"   ⚠️  Consider removing attention_mask from batch files")
        else:
            print(f"   ✅ No attention mask (correct for [PAD] token approach)")

        # 7) Print percentage of targets not equal to ignore_index (min, max, median across all samples)
        print("\n7️⃣ Target supervision statistics...")
        supervision_percentages = (supervised_counts.float() / seq_len * 100)
        sorted_percentages, _ = torch.sort(supervision_percentages)

        min_pct = sorted_percentages[0].item()
        max_pct = sorted_percentages[-1].item()
        median_idx = batch_size // 2
        median_pct = sorted_percentages[median_idx].item()
        mean_pct = supervision_percentages.mean().item()

        print(f"   • Supervision percentage per sample:")
        print(f"     - Min: {min_pct:.2f}%")
        print(f"     - Median: {median_pct:.2f}%")
        print(f"     - Mean: {mean_pct:.2f}%")
        print(f"     - Max: {max_pct:.2f}%")

        # Final result
        print(f"\n{'='*60}")
        if validation_passed:
            print(f"✅ VALIDATION PASSED: File structure is correct")
        else:
            print(f"❌ VALIDATION FAILED: File has structural issues")
        print(f"{'='*60}")

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
                        current_input = self.current_batch.get('input') if isinstance(self.current_batch, dict) else None
                        num_samples = int(current_input.shape[0]) if current_input is not None else 0
                        batch_info = f"📂 Change Data File (Current: {num_samples} samples)"
                        options.append(batch_info)

                    if self.current_batch is not None:
                        options.append("🔍 Navigate Samples")
                        options.append("✅ File Validation")

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
            elif "File Validation" in options[choice - 1]:
                self.validate_file()
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