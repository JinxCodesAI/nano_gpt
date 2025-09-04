"""
Model Quality Assessment Script for Diffusion Models

Evaluates relative quality of multiple diffusion models through ELO rating system.
Each model generates samples, and a SEQUENCE_SCORER judge model rates all samples.
Models compete in tournaments based on judge ratings (lower scores = better quality).
"""

import os
import pickle
import math
import random
import sys
import time
import numpy as np
from collections import Counter, defaultdict
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT, ModelMode
from sample_utils import (
    calculate_selfconfidence_ratio,
    calculate_sequence_scores,
    predict_and_sample_tokens,
    apply_remasking_step,
    linear_remasking_schedule
)
from training_utils import TrainingContext

# Handle old module references in checkpoints
class ModuleMapper:
    """Maps old module names to new ones for checkpoint loading"""
    def __init__(self):
        self.module_map = {
            'train_utils': 'training_utils',  # Handle renamed module
        }
    
    def __enter__(self):
        self._original_import = __builtins__.__import__
        def custom_import(name, *args, **kwargs):
            if name in self.module_map:
                name = self.module_map[name]
            return self._original_import(name, *args, **kwargs)
        __builtins__.__import__ = custom_import
        
        # Also handle sys.modules mapping
        if 'train_utils' not in sys.modules and 'training_utils' in sys.modules:
            sys.modules['train_utils'] = sys.modules['training_utils']
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        __builtins__.__import__ = self._original_import
        if 'train_utils' in sys.modules and 'training_utils' in sys.modules:
            if sys.modules['train_utils'] is sys.modules['training_utils']:
                del sys.modules['train_utils']


class GroundTruthModel:
    """
    A fake model that generates samples by taking them from validation set.
    This serves as ground truth baseline in tournament comparisons.
    """

    def __init__(self, config, vocab_info):
        self.config = config
        self.vocab_info = vocab_info
        self.validation_data = None
        self.sample_index = 0
        self._load_validation_data()

    def _load_validation_data(self):
        """Load validation data from val.bin file"""
        # Determine dataset name from vocab_info
        dataset_name = self.vocab_info.get('dataset_name', 'shakespeare_char')
        data_dir = os.path.join('data', dataset_name)
        val_path = os.path.join(data_dir, 'val.bin')

        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")

        # Load validation data using memory mapping for efficiency
        self.validation_data = np.memmap(val_path, dtype=np.uint16, mode='r')
        print(f"Ground truth model loaded validation data: {len(self.validation_data)} tokens from {val_path}")

    def generate_samples(self, batch_size, sequence_length):
        """Generate samples by extracting sequences from validation data"""
        if self.validation_data is None:
            raise RuntimeError("Validation data not loaded")

        if len(self.validation_data) < sequence_length:
            raise ValueError(f"Validation data ({len(self.validation_data)} tokens) is shorter than sequence_length ({sequence_length})")

        samples = []
        max_start_idx = len(self.validation_data) - sequence_length

        if max_start_idx <= 0:
            raise ValueError(f"Validation data too short for sequence length {sequence_length}")

        for i in range(batch_size):
            # Use deterministic sampling with wraparound to ensure reproducibility
            # Spread samples across the validation set to get diverse examples
            start_idx = (self.sample_index + i * 1000) % max_start_idx

            # Extract sequence from validation data
            sequence = self.validation_data[start_idx:start_idx + sequence_length]
            sample_tokens = sequence.astype(np.int64).tolist()
            sample_text = self.vocab_info['decode'](sample_tokens)

            samples.append({
                'tokens': sample_tokens,
                'text': sample_text,
                'model_id': 'ground_truth',
                'sample_id': i
            })

        # Update sample index for next batch to avoid repetition
        self.sample_index = (self.sample_index + batch_size * 1000) % max_start_idx

        return samples


def diffusion_generate(model, batch_size, total_length, iterations, remasking_model, mask_token_id,
                      randomness_strength, decode_fn, decode_mask_fn, device, vocab_size, itos, stoi,
                      start_ratio=0.99, end_ratio=0.05, verbose=True, temperature=1.0,
                      top_p=1.0, schedule_type='linear', masking_ratios=None, repetition_penalty=1.0,
                      repetition_window=10, log_debug=False, intelligent_remasking=False):
    """
    Generate text samples using diffusion-based iterative demasking
    
    Modified version that doesn't rely on global variables
    
    Args:
        model: Trained diffusion model
        batch_size: Number of samples to generate
        total_length: Length of sequence to generate
        iterations: Number of demasking iterations
        remasking_model: Optional remasking_binary model
        mask_token_id: ID of mask token
        randomness_strength: Balance between random and model-guided remasking (0-1)
        decode_fn: Function to decode tokens to text
        decode_mask_fn: Function to decode with mask characters
        device: Device to run on
        vocab_size: Size of vocabulary
        itos: Index to string mapping
        stoi: String to index mapping
        start_ratio: Starting ratio for linear schedule
        end_ratio: Ending ratio for linear schedule
        verbose: Whether to print progress
        temperature: Temperature for sampling
        top_p: Nucleus sampling parameter
        schedule_type: 'linear' or 'custom' - type of masking schedule to use
        masking_ratios: Array of masking ratios for 'custom' schedule
        repetition_penalty: Penalty for repeating recent tokens
        repetition_window: Window size for repetition penalty
        log_debug: Whether to do detailed debug logging
        intelligent_remasking: Enable selfmasking using base model when remasking_model is None

    Returns:
        Generated tokens (batch_size, total_length)
    """
    # Start with all positions masked
    tokens = torch.full((batch_size, total_length), mask_token_id, dtype=torch.long, device=device)
    
    if verbose:
        print(f"Starting diffusion generation: {batch_size} samples, {iterations} iterations")
        print(f"Total length: {total_length} (all tokens start masked)")
        if remasking_model is not None:
            print(f"Using remasking_binary model with randomness_strength={randomness_strength}")
        elif intelligent_remasking:
            print(f"Using intelligent selfmasking with randomness_strength={randomness_strength}")
        else:
            print("Using pure random remasking")
        print("=" * 80)
    
    for iteration in range(iterations):
        if verbose:
            masked_positions = (tokens == mask_token_id)
            num_masked_per_sample = masked_positions.sum(dim=1)
            avg_masked = num_masked_per_sample.float().mean().item()
            print(f"\nIteration {iteration + 1}/{iterations}")
            print(f"Average tokens masked: {avg_masked:.1f}/{total_length} ({avg_masked/total_length*100:.1f}%)")
        
        # Step 1: Predict tokens for all masked positions
        tokens, prediction_tokens = predict_and_sample_tokens(
            model=model,
            tokens=tokens,
            mask_token_id=mask_token_id,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            vocab_size=vocab_size,
            device=device,
            debug_logging_fn=None,  # Disable debug logging for evaluation
            itos=itos,
            stoi=stoi,
            verbose=verbose,
            log_debug=log_debug
        )
        
        # Step 2: Remask tokens for next iteration (except final iteration)
        if iteration < iterations - 1:
            tokens = apply_remasking_step(
                tokens=tokens,
                prediction_tokens=prediction_tokens,
                iteration=iteration,
                iterations=iterations,
                schedule_type=schedule_type,
                masking_ratios=masking_ratios,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                remasking_model=remasking_model,
                randomness_strength=randomness_strength,
                mask_token_id=mask_token_id,
                device=device,
                base_model=model,
                intelligent_remasking=intelligent_remasking,
                verbose=verbose
            )
        
        if verbose:
            print("=" * 80)
    
    return tokens


# Configuration
EVALUATION_CONFIG = {
    # Model configuration
    'checkpoints': [
        'ground_truth',  # Ground truth baseline using validation data
        'optimal5_bert_8000_better.pt',
        'optimal5_6600.pt',
        'optimal5_2000.pt',
        'optimal_5_8000_span.pt',
        'optimal2_8000.pt',
        '35.75_58.2_UM.pt',
        'optimal2_6400.pt',
        'optimal5_bert_8000.pt',
        'optimal_2400.pt',
        # Add more model checkpoints here for comparison
        # 'model2.pt',
        # 'model3.pt',
    ],
    'remasking_checkpoint_name': None,  # Optional: remasking_binary model checkpoint
    'judge_model': '1600_2_scorer.pt',  # '35.75_58.2_UM.pt',  # SEQUENCE_SCORER model to use as the single judge (lower scores = better)
    
    # Evaluation parameters
    'batch_size': 32,           # N samples per model per batch (generation) - reduced for testing
    'rating_batch_size': 64,    # Batch size for rating samples (GPU optimization) - reduced for testing
    'num_challenges': 1000,     # P tournament rounds before stopping - reduced for testing
    'sequence_length': 1024,   # Total length of generated sequence - reduced for testing
    'seed': 234,
    'device': 'cuda',
    'dtype': 'float16',
    'compile': False,
    
    # Generation parameters (copied from sample.py)
    'temperature': 0.8,
    'top_p': 0.95,
    'repetition_penalty': 1.0,
    'repetition_window': 10,
    'diffusion_iterations': 25,
    'start_ratio': 0.99,
    'end_ratio': 0.05,
    'randomness_strength': 0,
    'intelligent_remasking': True,  # Enable selfmasking for consistent evaluation

    # Schedule parameters
    'schedule_type': 'custom',
    'masking_ratios': [0.85,0.816,0.782,0.748,0.714,0.68,0.646,0.612,0.578,0.544,0.51,0.476,0.442,0.408,0.374,0.34,0.306,0.272,0.238,0.204,0.17,0.136,0.102,0.068,0.034],
    
    # Output configuration
    'out_dir': 'out',
    'results_file': 'model_evaluation_results.txt',
    'verbose': True,
}

# Allow configuration override from command line
exec(open('configurator.py').read())

# Validate configuration
if len(EVALUATION_CONFIG['checkpoints']) < 2:
    raise ValueError("Need at least 2 models for comparison")

if EVALUATION_CONFIG['schedule_type'] == 'custom':
    EVALUATION_CONFIG['diffusion_iterations'] = len(EVALUATION_CONFIG['masking_ratios'])


class ModelLoader:
    """Handles loading and managing multiple model checkpoints"""

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_names = []
        self.vocab_info = None
        self.remasking_model = None
        self.judge_model_id = None
        self.judge_model_type = None  # Will store ModelMode of judge model
        self.ground_truth_models = {}  # Store ground truth models separately
        
    def load_all_models(self):
        """Load all model checkpoints and vocabulary"""
        print(f"Loading {len(self.config['checkpoints'])} models...")
        
        # Load vocabulary from first model (assuming all models use same vocab)
        self._load_vocabulary()
        
        # Load each model checkpoint
        for i, checkpoint_name in enumerate(self.config['checkpoints']):
            model_id = f"model_{i}"
            self.model_names.append(model_id)

            # Check if this is a ground truth model
            if checkpoint_name == 'ground_truth':
                # Create ground truth model
                ground_truth_model = GroundTruthModel(self.config, self.vocab_info)
                self.models[model_id] = ground_truth_model
                self.ground_truth_models[model_id] = ground_truth_model
                print(f"  ✓ {model_id} loaded as ground truth model")
            else:
                # Load regular model checkpoint
                model = self._load_single_model(checkpoint_name, model_id)
                self.models[model_id] = model

                # Track which model is the judge (if it's in the checkpoints list)
                if checkpoint_name == self.config['judge_model']:
                    self.judge_model_id = model_id
                    print(f"  ✓ {model_id} identified as judge model")

        # Load optional remasking model
        self._load_remasking_model()

        # Load judge model (can be separate from evaluation models)
        self._load_judge_model()

        # Validate judge model type
        if self.judge_model_type != ModelMode.SEQUENCE_SCORER:
            raise ValueError(f"Judge model must be SEQUENCE_SCORER type, but got {self.judge_model_type.value}")

        print(f"Successfully loaded {len(self.models)} models")
        return self.models, self.vocab_info, self.remasking_model, self.judge_model_id, self.judge_model_type
    
    def _load_single_model(self, checkpoint_name, model_id):
        """Load a single model checkpoint"""
        ckpt_path = os.path.join(self.config['out_dir'], checkpoint_name)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        print(f"Loading {model_id} from {checkpoint_name}...")
        with ModuleMapper():
            checkpoint = torch.load(ckpt_path, map_location=self.config['device'], weights_only=False)
        
        # Create model
        model_args = checkpoint['model_args']
        if 'attention_type' not in model_args:
            model_args['attention_type'] = 'causal'  # Backward compatibility
            
        model_config = GPTConfig(**model_args)
        model = GPT(model_config)
        
        # Load weights
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        model.eval()
        model.to(self.config['device'])
        if self.config['compile']:
            model = torch.compile(model)
            
        print(f"  {model_id} loaded (attention: {model_config.attention_type})")
        return model
    
    def _load_vocabulary(self):
        """Load vocabulary information from the first real model (skip ground_truth)"""
        # Find first non-ground_truth checkpoint for vocabulary loading
        first_real_checkpoint = None
        for checkpoint_name in self.config['checkpoints']:
            if checkpoint_name != 'ground_truth':
                first_real_checkpoint = checkpoint_name
                break

        if first_real_checkpoint is None:
            raise ValueError("No real model checkpoints found - need at least one non-ground_truth model for vocabulary loading")

        # Use same vocabulary loading logic as sample.py
        ckpt_path = os.path.join(self.config['out_dir'], first_real_checkpoint)
        with ModuleMapper():
            checkpoint = torch.load(ckpt_path, map_location=self.config['device'], weights_only=False)
        
        # Determine dataset name
        dataset_name = None
        if 'config' in checkpoint:
            config = checkpoint['config']
            if hasattr(config, 'get'):
                dataset_name = config.get('dataset')
            elif hasattr(config, '__getitem__'):
                try:
                    dataset_name = config['dataset']
                except (KeyError, TypeError):
                    pass
        
        if not dataset_name:
            if 'shakespeare' in first_real_checkpoint.lower():
                dataset_name = 'shakespeare_char'
            else:
                dataset_name = 'shakespeare_char'
        
        # Load meta.pkl
        meta_path = os.path.join('data', dataset_name, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
            
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = meta['vocab_size']
        mask_token_id = vocab_size  # Mask token ID
        
        def decode(token_ids):
            result = []
            for token_id in token_ids:
                if token_id == mask_token_id:
                    result.append('[MASK]')
                elif token_id < len(itos):
                    result.append(itos[token_id])
                else:
                    result.append('[UNK]')
            return ''.join(result)
        
        def decode_with_mask_char(token_ids, mask_char='#'):
            result = []
            for token_id in token_ids:
                if token_id == mask_token_id:
                    result.append(mask_char)
                elif token_id < len(itos):
                    result.append(itos[token_id])
                else:
                    result.append('[UNK]')
            return ''.join(result)
        
        self.vocab_info = {
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size,
            'mask_token_id': mask_token_id,
            'decode': decode,
            'decode_mask_fn': decode_with_mask_char,
            'dataset_name': dataset_name
        }
        
        print(f"Vocabulary loaded: size={vocab_size}, mask_token_id={mask_token_id}, dataset={dataset_name}")
    
    def _load_remasking_model(self):
        """Load optional remasking model"""
        remasking_checkpoint_name = self.config.get('remasking_checkpoint_name')
        if remasking_checkpoint_name is None:
            return
            
        remasking_ckpt_path = os.path.join(self.config['out_dir'], remasking_checkpoint_name)
        if not os.path.exists(remasking_ckpt_path):
            print(f"Remasking checkpoint not found: {remasking_ckpt_path}")
            return
            
        print(f"Loading remasking model from {remasking_checkpoint_name}...")
        with ModuleMapper():
            remasking_checkpoint = torch.load(remasking_ckpt_path, map_location=self.config['device'], weights_only=False)
        
        remasking_model_args = remasking_checkpoint['model_args']
        if 'attention_type' not in remasking_model_args:
            remasking_model_args['attention_type'] = 'causal'
        
        # Verify it's a binary classification model
        if not remasking_model_args.get('binary_classification', False):
            print("Warning: Remasking model is not binary classification, skipping")
            return
            
        remasking_config = GPTConfig(**remasking_model_args)
        self.remasking_model = GPT(remasking_config)
        
        # Load weights
        remasking_state_dict = remasking_checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(remasking_state_dict.items()):
            if k.startswith(unwanted_prefix):
                remasking_state_dict[k[len(unwanted_prefix):]] = remasking_state_dict.pop(k)
        self.remasking_model.load_state_dict(remasking_state_dict)
        
        self.remasking_model.eval()
        self.remasking_model.to(self.config['device'])
        if self.config['compile']:
            self.remasking_model = torch.compile(self.remasking_model)
            
        print("Remasking model loaded (binary classification)")

    def _load_judge_model(self):
        """Load judge model (can be separate from evaluation models)"""
        judge_checkpoint_name = self.config['judge_model']

        # Check if judge model was already loaded as part of evaluation models
        if self.judge_model_id is not None:
            judge_model = self.models[self.judge_model_id]
            self.judge_model_type = judge_model.config.mode
            print(f"  Judge model type: {self.judge_model_type.value}")
            return

        # Load judge model separately
        print(f"Loading separate judge model from {judge_checkpoint_name}...")
        judge_model = self._load_single_model(judge_checkpoint_name, "judge")

        # Store judge model separately (not in evaluation models)
        self.judge_model_id = "judge"
        self.models[self.judge_model_id] = judge_model
        self.judge_model_type = judge_model.config.mode

        print(f"  ✓ Separate judge model loaded")
        print(f"  Judge model type: {self.judge_model_type.value}")


class SampleGenerator:
    """Generates samples from loaded models using diffusion process"""

    def __init__(self, config, vocab_info, remasking_model=None):
        self.config = config
        self.vocab_info = vocab_info
        self.remasking_model = remasking_model

        # Set up device and dtype context
        device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Timing tracking
        self.model_generation_times = {}  # Track time per model
        self.total_generation_time = 0.0
    
    def generate_samples_for_model(self, model, model_id):
        """Generate samples for a single model"""
        print(f"\nGenerating samples for {model_id}...")

        # Start timing for this model
        model_start_time = time.time()

        # Check if this is a ground truth model
        if isinstance(model, GroundTruthModel):
            # Use ground truth model's generate_samples method
            samples = model.generate_samples(
                batch_size=self.config['batch_size'],
                sequence_length=self.config['sequence_length']
            )
            # Record timing for ground truth model
            model_end_time = time.time()
            self.model_generation_times[model_id] = model_end_time - model_start_time
            print(f"Generated {len(samples)} ground truth samples for {model_id} in {self.model_generation_times[model_id]:.2f}s")
            return samples

        # Regular diffusion model generation
        with torch.no_grad():
            with self.ctx:
                generated_tokens = diffusion_generate(
                    model=model,
                    batch_size=self.config['batch_size'],
                    total_length=self.config['sequence_length'],
                    iterations=self.config['diffusion_iterations'],
                    remasking_model=self.remasking_model,
                    mask_token_id=self.vocab_info['mask_token_id'],
                    randomness_strength=self.config['randomness_strength'],
                    decode_fn=self.vocab_info['decode'],
                    decode_mask_fn=self.vocab_info['decode_mask_fn'],
                    device=self.config['device'],
                    vocab_size=self.vocab_info['vocab_size'],
                    itos=self.vocab_info['itos'],
                    stoi=self.vocab_info['stoi'],
                    start_ratio=self.config['start_ratio'],
                    end_ratio=self.config['end_ratio'],
                    verbose=False,  # Suppress detailed generation logs
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    schedule_type=self.config['schedule_type'],
                    masking_ratios=self.config.get('masking_ratios'),
                    repetition_penalty=self.config['repetition_penalty'],
                    repetition_window=self.config['repetition_window'],
                    log_debug=False,
                    intelligent_remasking=self.config['intelligent_remasking']
                )

        # Record timing for diffusion model
        model_end_time = time.time()
        self.model_generation_times[model_id] = model_end_time - model_start_time

        # Convert to list of samples
        samples = []
        for i in range(self.config['batch_size']):
            sample_tokens = generated_tokens[i].tolist()
            sample_text = self.vocab_info['decode'](sample_tokens)
            samples.append({
                'tokens': sample_tokens,
                'text': sample_text,
                'model_id': model_id,
                'sample_id': i
            })

        print(f"Generated {len(samples)} samples for {model_id} in {self.model_generation_times[model_id]:.2f}s")
        return samples
    
    def generate_all_samples(self, models):
        """Generate samples for all models (excluding judge model)"""
        print(f"\n{'='*80}")
        print("SAMPLE GENERATION PHASE")
        print(f"{'='*80}")

        # Start timing for entire generation phase
        phase_start_time = time.time()

        all_samples = {}
        for model_id, model in models.items():
            # Skip judge model - it's only used for rating, not generation
            if model_id == "judge":
                continue
            all_samples[model_id] = self.generate_samples_for_model(model, model_id)

        # End timing for entire generation phase
        phase_end_time = time.time()
        self.total_generation_time = phase_end_time - phase_start_time

        total_samples = sum(len(samples) for samples in all_samples.values())
        print(f"\nGenerated {total_samples} total samples across {len(all_samples)} models")

        # Calculate and display tokens per second metrics
        self._display_tokens_per_second_metrics(all_samples)

        return all_samples

    def _display_tokens_per_second_metrics(self, all_samples):
        """Calculate and display tokens per second metrics"""
        if self.total_generation_time <= 0:
            print("Warning: No valid generation time recorded")
            return

        # Count models (excluding ground truth for step-based calculation)
        num_models = len(all_samples)
        num_diffusion_models = sum(1 for model_id in all_samples.keys()
                                 if not any('ground_truth' in str(sample.get('model_id', ''))
                                          for sample in all_samples[model_id]))

        # Calculate total tokens generated (final tokens) - ONLY from diffusion models
        total_final_tokens = 0
        for model_id, samples in all_samples.items():
            # Check if this is a ground truth model by looking at the first sample
            if samples and 'ground_truth' in str(samples[0].get('model_id', '')):
                # Skip ground truth model tokens - they don't use diffusion
                continue
            else:
                # Count tokens from diffusion models only
                for sample in samples:
                    total_final_tokens += len(sample['tokens'])

        # Calculate total unmasking operations (every step tokens)
        # For ground truth models: 0 operations (they don't use diffusion)
        # For diffusion models: batch_size * iterations per model
        total_unmasking_operations = 0
        for model_id, samples in all_samples.items():
            # Check if this is a ground truth model by looking at the first sample
            if samples and 'ground_truth' in str(samples[0].get('model_id', '')):
                # Ground truth model: no unmasking operations
                continue
            else:
                # Diffusion model: batch_size * sequence_length * iterations
                batch_size = len(samples)
                sequence_length = self.config['sequence_length']
                iterations = self.config['diffusion_iterations']
                total_unmasking_operations += batch_size * sequence_length * iterations

        # Calculate metrics
        final_tokens_per_sec = total_final_tokens / self.total_generation_time
        step_tokens_per_sec = total_unmasking_operations / self.total_generation_time

        print(f"\n{'='*60}")
        print("TOKENS PER SECOND METRICS")
        print(f"{'='*60}")
        print(f"Total generation time: {self.total_generation_time:.2f}s")
        print(f"Total models processed: {num_models} ({num_diffusion_models} diffusion + {num_models - num_diffusion_models} ground truth)")
        print(f"")
        print(f"1) Final tokens/sec (diffusion models only):")
        print(f"   Total final tokens generated: {total_final_tokens:,}")
        print(f"   Rate: {final_tokens_per_sec:.1f} tokens/sec")
        print(f"")
        print(f"2) Every step tokens/sec (diffusion models only):")
        print(f"   Total unmasking operations: {total_unmasking_operations:,}")
        print(f"   Rate: {step_tokens_per_sec:.1f} operations/sec")
        print(f"{'='*60}")


class ModelRater:
    """Rates samples using model likelihood scores"""
    
    def __init__(self, config, vocab_info):
        self.config = config
        self.vocab_info = vocab_info
        
        # Set up device and dtype context
        device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
        self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    def rate_samples_with_model(self, model, samples, rater_model_id, model_type, batch_size=None):
        """Rate a list of samples using a specific model - GPU optimized with batching"""
        if batch_size is None:
            batch_size = self.config.get('rating_batch_size', 8)  # Default batch size for rating

        ratings = []

        with torch.no_grad():
            # Process samples in batches for GPU efficiency
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]

                # Create batched tensor from all samples in this batch
                batch_tokens = []
                for sample in batch_samples:
                    batch_tokens.append(sample['tokens'])

                # Convert to tensor (batch_size, sequence_length)
                tokens_tensor = torch.tensor(batch_tokens, device=self.config['device'])

                # Calculate scores based on model type
                if model_type == ModelMode.SEQUENCE_SCORER:
                    # Use sequence scoring for SEQUENCE_SCORER models
                    cls_token_id = model.config.cls_token_id
                    if cls_token_id is None:
                        raise ValueError("SEQUENCE_SCORER model must have cls_token_id configured")

                    scores = calculate_sequence_scores(
                        model=model,
                        tokens=tokens_tensor,
                        cls_token_id=cls_token_id,
                        device=self.config['device'],
                        ctx=self.ctx
                    )
                else:
                    raise ValueError(f"Unsupported model type for rating: {model_type}")

                # Store ratings for each sample in the batch
                for j, sample in enumerate(batch_samples):
                    rating = {
                        'sample_model_id': sample['model_id'],
                        'sample_id': sample['sample_id'],
                        'rater_model_id': rater_model_id,
                        'confidence_score': scores[j],  # j-th sample in batch
                        'log_prob': scores[j]  # Keep same field name for compatibility
                    }
                    ratings.append(rating)

        return ratings
    
    def rate_all_samples(self, models, all_samples, judge_model_id, judge_model_type):
        """Use only the specified judge model to rate all samples"""
        judge_checkpoint = self.config['judge_model']
        print(f"\n{'='*80}")
        print(f"SINGLE-MODEL RATING PHASE ({judge_model_id} - {judge_checkpoint} as {judge_model_type.value} judge)")
        print(f"{'='*80}")

        if judge_model_id not in models:
            raise ValueError(f"Judge model {judge_model_id} not found in loaded models")

        judge_model = models[judge_model_id]
        all_ratings = {}

        print(f"\n{judge_model_id} ({judge_checkpoint}) rating all samples using {judge_model_type.value} scoring...")
        model_ratings = {}

        for sample_model_id, samples in all_samples.items():
            batch_size = self.config.get('rating_batch_size', 8)
            num_batches = (len(samples) + batch_size - 1) // batch_size
            print(f"  Rating {len(samples)} samples from {sample_model_id} (using {num_batches} batches of {batch_size})...")

            ratings = self.rate_samples_with_model(judge_model, samples, judge_model_id, judge_model_type)
            model_ratings[sample_model_id] = ratings
            print(f"  ✓ Completed rating {len(ratings)} samples from {sample_model_id}")

        all_ratings[judge_model_id] = model_ratings

        print(f"\nRating phase complete - {judge_model_id} rated all samples using {judge_model_type.value}")
        return all_ratings


class SampleStats:
    """Tracks statistics for individual samples"""
    
    def __init__(self):
        self.comparisons = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def add_result(self, result):
        """Add a tournament result (WIN/LOSE/DRAW)"""
        self.comparisons += 1
        if result == 'WIN':
            self.wins += 1
        elif result == 'LOSE':
            self.losses += 1
        elif result == 'DRAW':
            self.draws += 1
    
    def get_points(self):
        """Calculate points using win=3, draw=1, loss=0"""
        return self.wins * 3 + self.draws * 1
    
    def get_win_rate(self):
        """Calculate win percentage"""
        if self.comparisons == 0:
            return 0.0
        return (self.wins / self.comparisons) * 100
    
    def get_avg_points(self):
        """Calculate average points per comparison"""
        if self.comparisons == 0:
            return 0.0
        return self.get_points() / self.comparisons
    
    def __str__(self):
        if self.comparisons == 0:
            return "No comparisons"
        points = self.get_points()
        win_rate = self.get_win_rate()
        avg_points = self.get_avg_points()
        return f"W:{self.wins} D:{self.draws} L:{self.losses} ({self.comparisons} games, {points} pts, {avg_points:.2f} avg pts, {win_rate:.1f}% win rate)"


class ELOTracker:
    """Manages ELO ratings and tournament statistics for models"""
    
    def __init__(self, model_names):
        self.model_names = model_names
        self.num_models = len(model_names)
        self.ratings = {name: 1000.0 for name in model_names}
        self.num_rounds = {name: 0 for name in model_names}
        self.rating_history = {name: [1000.0] for name in model_names}
        
        # Sample-level statistics
        self.sample_stats = {}  # {(model_id, sample_id): SampleStats}
    
    def get_sample_key(self, model_id, sample_id):
        """Get key for sample statistics"""
        return (model_id, sample_id)
    
    def initialize_sample_stats(self, all_samples):
        """Initialize statistics for all samples"""
        for model_id, samples in all_samples.items():
            for sample in samples:
                key = self.get_sample_key(model_id, sample['sample_id'])
                self.sample_stats[key] = SampleStats()
    
    def calculate_elo_change(self, rating1, rating2, result):
        """Calculate ELO rating change for player 1"""
        K = 32  # ELO K-factor
        
        # Expected score
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        
        # Actual score
        if result == 'WIN':
            actual1 = 1.0
        elif result == 'LOSE':
            actual1 = 0.0
        else:  # DRAW
            actual1 = 0.5
        
        # Rating change
        change = K * (actual1 - expected1)
        return change
    
    def update_elo(self, model1_id, model2_id, result):
        """Update ELO ratings after a match"""
        rating1 = self.ratings[model1_id]
        rating2 = self.ratings[model2_id]
        
        change1 = self.calculate_elo_change(rating1, rating2, result)
        change2 = -change1  # Zero-sum
        
        self.ratings[model1_id] += change1
        self.ratings[model2_id] += change2
        
        # Update round counts
        self.num_rounds[model1_id] += 1
        self.num_rounds[model2_id] += 1
        
        # Store rating history
        self.rating_history[model1_id].append(self.ratings[model1_id])
        self.rating_history[model2_id].append(self.ratings[model2_id])
    
    def update_sample_stats(self, model1_id, sample1_id, model2_id, sample2_id, result):
        """Update sample-level statistics"""
        key1 = self.get_sample_key(model1_id, sample1_id)
        key2 = self.get_sample_key(model2_id, sample2_id)
        
        # Update for sample 1
        self.sample_stats[key1].add_result(result)
        
        # Update for sample 2 (opposite result)
        if result == 'WIN':
            opposite_result = 'LOSE'
        elif result == 'LOSE':
            opposite_result = 'WIN'
        else:
            opposite_result = 'DRAW'
        
        self.sample_stats[key2].add_result(opposite_result)
    
    def get_rankings(self):
        """Get models ranked by ELO rating"""
        return sorted(self.model_names, key=lambda x: self.ratings[x], reverse=True)
    
    def get_sample_scores(self, model_id, sample_id, all_ratings):
        """Get judge model score for a specific sample"""
        sample_scores = {}
        
        # With single judge, get rating from the one judge model
        judge_model_id = list(all_ratings.keys())[0]  # Should be only one judge
        judge_ratings = all_ratings[judge_model_id]
        
        if model_id in judge_ratings:
            # Find the rating for this specific sample
            for rating in judge_ratings[model_id]:
                if rating['sample_id'] == sample_id:
                    sample_scores[judge_model_id] = rating['confidence_score']
                    break
        
        return sample_scores
    
    def get_top_samples(self, all_samples, n=3):
        """Get top N samples by points (excluding ground truth)"""
        return self._get_samples_by_rank(all_samples, n, reverse=True, exclude_ground_truth=True)

    def get_worst_samples(self, all_samples, n=3):
        """Get worst N samples by points (excluding ground truth)"""
        return self._get_samples_by_rank(all_samples, n, reverse=False, exclude_ground_truth=True)
    
    def _get_samples_by_rank(self, all_samples, n, reverse=True, exclude_ground_truth=False):
        """Get samples ranked by points and win rate"""
        sample_scores = []

        for (model_id, sample_id), stats in self.sample_stats.items():
            if stats.comparisons > 0:  # Only samples that participated in comparisons
                # Find the actual sample data
                sample_data = None
                if model_id in all_samples:
                    for sample in all_samples[model_id]:
                        if sample['sample_id'] == sample_id:
                            sample_data = sample
                            break

                if sample_data:
                    # Skip ground truth samples if requested
                    if exclude_ground_truth and sample_data.get('model_id') == 'ground_truth':
                        continue

                    sample_scores.append({
                        'model_id': model_id,
                        'sample_id': sample_id,
                        'sample_text': sample_data['text'],  # Store full text
                        'sample_text_preview': sample_data['text'][:200] + '...' if len(sample_data['text']) > 200 else sample_data['text'],
                        'stats': stats,
                        'points': stats.get_points()
                    })

        # Sort by average points per comparison, then by total points as tiebreaker
        sample_scores.sort(key=lambda x: (x['points'] / x['stats'].comparisons if x['stats'].comparisons > 0 else 0, x['points']), reverse=reverse)
        return sample_scores[:n]


class TournamentManager:
    """Manages pairwise tournament comparisons"""
    
    def __init__(self, config, elo_tracker, judge_model_type):
        self.config = config
        self.elo_tracker = elo_tracker
        self.judge_model_type = judge_model_type
        self.total_comparisons = 0
    
    def determine_winner(self, model1_ratings, model2_ratings):
        """Determine winner based on judge model scores"""
        # Get ratings from the single judge
        if len(model1_ratings) == 0 or len(model2_ratings) == 0:
            return 'DRAW'

        # With single judge, we have one score per sample
        model1_score = model1_ratings[0]['confidence_score']
        model2_score = model2_ratings[0]['confidence_score']

        # Calculate the absolute difference
        score_diff = abs(model1_score - model2_score)

        # Use fixed threshold for SEQUENCE_SCORER models (0-1 range)
        # If difference is less than 0.1, it's a draw
        threshold = 0.1

        if score_diff <= threshold:
            return 'DRAW'
        else:
            # For SEQUENCE_SCORER: lower score wins (as specified)
            # Only SEQUENCE_SCORER models are supported as judges
            if model1_score < model2_score:
                return 'WIN'
            else:
                return 'LOSE'
    
    def _select_weighted_sample(self, model_id, samples):
        """Select a sample with weighting favoring less popular (less compared) samples"""
        sample_weights = []
        
        for i, sample in enumerate(samples):
            sample_key = self.elo_tracker.get_sample_key(model_id, sample['sample_id'])
            if sample_key in self.elo_tracker.sample_stats:
                comparisons = self.elo_tracker.sample_stats[sample_key].comparisons
                # Invert the weight: more comparisons = lower weight
                # Add 1 to avoid division by zero and give new samples high weight
                weight = 1.0 / (comparisons + 1)
            else:
                # New sample with no comparisons gets maximum weight
                weight = 1.0
            sample_weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(sample_weights)
        if total_weight == 0:
            # Fallback to uniform random if all weights are zero
            return random.randint(0, len(samples) - 1)
        
        # Generate random number and select based on cumulative weights
        random_val = random.random() * total_weight
        cumulative_weight = 0
        
        for i, weight in enumerate(sample_weights):
            cumulative_weight += weight
            if random_val <= cumulative_weight:
                return i
        
        # Fallback (should not happen)
        return len(samples) - 1
    
    def run_single_tournament(self, all_samples, all_ratings):
        """Run a single tournament round"""
        model_names = list(all_samples.keys())
        
        # Generate random matchup
        model1_id = random.choice(model_names)
        model2_id = random.choice([m for m in model_names if m != model1_id])
        
        # Use weighted selection to favor less popular samples
        sample1_id = self._select_weighted_sample(model1_id, all_samples[model1_id])
        sample2_id = self._select_weighted_sample(model2_id, all_samples[model2_id])
        
        # Collect ratings for both samples from the single judge model
        model1_ratings = []
        model2_ratings = []
        
        # Find the judge model (should be only one in all_ratings)
        judge_model_id = list(all_ratings.keys())[0]
        judge_ratings = all_ratings[judge_model_id]
        
        # Get rating for sample1 from the judge
        if model1_id in judge_ratings:
            for rating in judge_ratings[model1_id]:
                if rating['sample_id'] == sample1_id:
                    model1_ratings.append(rating)
                    break
        
        # Get rating for sample2 from the judge
        if model2_id in judge_ratings:
            for rating in judge_ratings[model2_id]:
                if rating['sample_id'] == sample2_id:
                    model2_ratings.append(rating)
                    break
        
        if len(model1_ratings) == 0 or len(model2_ratings) == 0:
            return False  # Skip if no ratings available
        
        # Determine winner
        result = self.determine_winner(model1_ratings, model2_ratings)
        
        # Update ELO ratings
        self.elo_tracker.update_elo(model1_id, model2_id, result)
        
        # Update sample statistics
        self.elo_tracker.update_sample_stats(model1_id, sample1_id, model2_id, sample2_id, result)
        
        self.total_comparisons += 1
        
        if self.config['verbose'] and self.total_comparisons % 20 == 0:
            print(f"  Completed {self.total_comparisons} comparisons...")
        
        return True
    
    def run_tournaments(self, all_samples, all_ratings):
        """Run tournaments until stopping criteria met"""
        print(f"\n{'='*80}")
        print("TOURNAMENT PHASE")
        print(f"{'='*80}")
        
        print(f"Running tournaments until all models have {self.config['num_challenges']} matches...")
        
        while True:
            # Check if all models have enough rounds
            min_rounds = min(self.elo_tracker.num_rounds.values())
            if min_rounds >= self.config['num_challenges']:
                break
            
            # Run a single tournament
            success = self.run_single_tournament(all_samples, all_ratings)
            if not success and self.total_comparisons > 0:
                print("Warning: Could not generate valid comparison, continuing...")
        
        print(f"\nTournament phase complete: {self.total_comparisons} total comparisons")


class ModelEvaluator:
    """Main orchestrator for model evaluation"""
    
    def __init__(self, config):
        self.config = config
        
    def run_evaluation(self):
        """Run complete model evaluation pipeline"""
        print(f"Model Quality Assessment - Evaluating {len(self.config['checkpoints'])} models")
        print(f"Configuration: {self.config['batch_size']} samples per model, {self.config['num_challenges']} challenges")
        print(f"Sequence length: {self.config['sequence_length']}, Seed: {self.config['seed']}")
        
        # Stage 1: Load models
        model_loader = ModelLoader(self.config)
        models, vocab_info, remasking_model, judge_model_id, judge_model_type = model_loader.load_all_models()
        
        # Stage 2: Generate samples
        sample_generator = SampleGenerator(self.config, vocab_info, remasking_model)
        all_samples = sample_generator.generate_all_samples(models)
        
        # Stage 3: Rate samples
        model_rater = ModelRater(self.config, vocab_info)
        all_ratings = model_rater.rate_all_samples(models, all_samples, judge_model_id, judge_model_type)
        
        # Stage 4: Run tournaments
        elo_tracker = ELOTracker(model_loader.model_names)
        elo_tracker.initialize_sample_stats(all_samples)

        tournament_manager = TournamentManager(self.config, elo_tracker, judge_model_type)
        tournament_manager.run_tournaments(all_samples, all_ratings)
        
        # Stage 5: Generate results
        self.generate_results(elo_tracker, all_samples, model_loader.model_names, all_ratings)
        
        return elo_tracker, all_samples
    
    def generate_results(self, elo_tracker, all_samples, model_names, all_ratings):
        """Generate and display final results"""
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        
        # Model rankings with win/loss/draw statistics
        rankings = elo_tracker.get_rankings()
        print("\nModel Rankings (by ELO rating):")
        print("-" * 50)

        # Calculate win/loss/draw stats for each model
        model_stats = {}
        for model_id in rankings:
            wins = losses = draws = 0
            # Count wins/losses/draws from sample statistics
            for (sample_model_id, sample_id), sample_stat in elo_tracker.sample_stats.items():
                if sample_model_id == model_id:
                    wins += sample_stat.wins
                    losses += sample_stat.losses
                    draws += sample_stat.draws
            model_stats[model_id] = {'wins': wins, 'losses': losses, 'draws': draws}

        for i, model_id in enumerate(rankings):
            rating = elo_tracker.ratings[model_id]
            rounds = elo_tracker.num_rounds[model_id]
            checkpoint_name = self.config['checkpoints'][int(model_id.split('_')[1])]
            stats = model_stats[model_id]

            # Calculate judge model score statistics for this model
            judge_scores = []
            judge_model_id = list(all_ratings.keys())[0]  # Get the single judge model
            if model_id in all_ratings[judge_model_id]:
                for sample_rating in all_ratings[judge_model_id][model_id]:
                    judge_scores.append(sample_rating['confidence_score'])

            # Special display for ground truth model
            if checkpoint_name == 'ground_truth':
                print(f"{i+1}. {model_id} (Ground Truth Baseline)")
            else:
                print(f"{i+1}. {model_id} ({checkpoint_name})")
            print(f"   ELO: {rating:.1f} | Rounds: {rounds}")
            print(f"   W:{stats['wins']} D:{stats['draws']} L:{stats['losses']} | Win Rate: {stats['wins']/(stats['wins']+stats['losses']+stats['draws'])*100:.1f}%")

            # Add judge model score statistics
            if judge_scores:
                avg_score = sum(judge_scores) / len(judge_scores)
                min_score = min(judge_scores)
                max_score = max(judge_scores)
                print(f"   Judge Scores - Avg: {avg_score:.3f} | Lowest: {min_score:.3f} | Highest: {max_score:.3f}")
        
        # Top samples
        top_samples = elo_tracker.get_top_samples(all_samples, 3)
        print(f"\nTop 3 Samples (by points: win=3, draw=1, loss=0):")
        print("=" * 100)
        for i, sample_info in enumerate(top_samples):
            print(f"\n{i+1}. {sample_info['model_id']} Sample #{sample_info['sample_id']}")
            print(f"   Stats: {sample_info['stats']}")
            
            # Get individual model scores
            sample_scores = elo_tracker.get_sample_scores(
                sample_info['model_id'], 
                sample_info['sample_id'], 
                all_ratings
            )
            if sample_scores:
                scores_list = [f"{score:.3f}" for score in sorted(sample_scores.values())]
                print(f"   Scores: [{', '.join(scores_list)}]")
            
            print(f"   Full Text:")
            print(f"   {'-' * 90}")
            # Print full text with proper indentation
            full_text = sample_info['sample_text']
            # Split into lines and indent each line for better readability
            for line in full_text.split('\n'):
                print(f"   {line}")
            print(f"   {'-' * 90}")
        
        # Worst samples
        worst_samples = elo_tracker.get_worst_samples(all_samples, 3)
        print(f"\nWorst 3 Samples (by points: win=3, draw=1, loss=0):")
        print("=" * 100)
        for i, sample_info in enumerate(worst_samples):
            print(f"\n{i+1}. {sample_info['model_id']} Sample #{sample_info['sample_id']}")
            print(f"   Stats: {sample_info['stats']}")
            
            # Get individual model scores
            sample_scores = elo_tracker.get_sample_scores(
                sample_info['model_id'], 
                sample_info['sample_id'], 
                all_ratings
            )
            if sample_scores:
                scores_list = [f"{score:.3f}" for score in sorted(sample_scores.values())]
                print(f"   Scores: [{', '.join(scores_list)}]")
            
            print(f"   Full Text:")
            print(f"   {'-' * 90}")
            # Print full text with proper indentation
            full_text = sample_info['sample_text']
            # Split into lines and indent each line for better readability
            for line in full_text.split('\n'):
                print(f"   {line}")
            print(f"   {'-' * 90}")
        
        # Save results to file
        results_file = os.path.join(self.config['out_dir'], self.config['results_file'])
        with open(results_file, 'w') as f:
            f.write(f"Model Quality Assessment Results\n")
            f.write(f"Generated on: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"Configuration: {self.config['batch_size']} samples per model, {self.config['num_challenges']} challenges\n\n")
            
            f.write("Model Rankings (by ELO rating):\n")
            f.write("-" * 50 + "\n")
            for i, model_id in enumerate(rankings):
                rating = elo_tracker.ratings[model_id]
                rounds = elo_tracker.num_rounds[model_id]
                checkpoint_name = self.config['checkpoints'][int(model_id.split('_')[1])]
                stats = model_stats[model_id]

                # Calculate judge model score statistics for this model
                judge_scores = []
                judge_model_id = list(all_ratings.keys())[0]  # Get the single judge model
                if model_id in all_ratings[judge_model_id]:
                    for sample_rating in all_ratings[judge_model_id][model_id]:
                        judge_scores.append(sample_rating['confidence_score'])

                # Special display for ground truth model
                if checkpoint_name == 'ground_truth':
                    f.write(f"{i+1}. {model_id} (Ground Truth Baseline)\n")
                else:
                    f.write(f"{i+1}. {model_id} ({checkpoint_name})\n")
                f.write(f"   ELO: {rating:.1f} | Rounds: {rounds}\n")
                f.write(f"   W:{stats['wins']} D:{stats['draws']} L:{stats['losses']} | Win Rate: {stats['wins']/(stats['wins']+stats['losses']+stats['draws'])*100:.1f}%\n")

                # Add judge model score statistics to file
                if judge_scores:
                    avg_score = sum(judge_scores) / len(judge_scores)
                    min_score = min(judge_scores)
                    max_score = max(judge_scores)
                    f.write(f"   Judge Scores - Avg: {avg_score:.3f} | Lowest: {min_score:.3f} | Highest: {max_score:.3f}\n")
            
            f.write(f"\nTop 3 Samples (by points):\n")
            f.write("=" * 100 + "\n")
            for i, sample_info in enumerate(top_samples):
                f.write(f"\n{i+1}. {sample_info['model_id']} Sample #{sample_info['sample_id']}\n")
                f.write(f"   Stats: {sample_info['stats']}\n")
                
                # Write individual model scores
                sample_scores = elo_tracker.get_sample_scores(
                    sample_info['model_id'], 
                    sample_info['sample_id'], 
                    all_ratings
                )
                if sample_scores:
                    scores_list = [f"{score:.3f}" for score in sorted(sample_scores.values())]
                    f.write(f"   Scores: [{', '.join(scores_list)}]\n")
                
                f.write(f"   Full Text:\n")
                f.write(f"   {'-' * 90}\n")
                # Write full text with proper indentation
                full_text = sample_info['sample_text']
                for line in full_text.split('\n'):
                    f.write(f"   {line}\n")
                f.write(f"   {'-' * 90}\n")
            
            # Worst samples
            worst_samples = elo_tracker.get_worst_samples(all_samples, 3)
            f.write(f"\nWorst 3 Samples (by points):\n")
            f.write("=" * 100 + "\n")
            for i, sample_info in enumerate(worst_samples):
                f.write(f"\n{i+1}. {sample_info['model_id']} Sample #{sample_info['sample_id']}\n")
                f.write(f"   Stats: {sample_info['stats']}\n")
                
                # Write individual model scores
                sample_scores = elo_tracker.get_sample_scores(
                    sample_info['model_id'], 
                    sample_info['sample_id'], 
                    all_ratings
                )
                if sample_scores:
                    scores_list = [f"{score:.3f}" for score in sorted(sample_scores.values())]
                    f.write(f"   Scores: [{', '.join(scores_list)}]\n")
                
                f.write(f"   Full Text:\n")
                f.write(f"   {'-' * 90}\n")
                # Write full text with proper indentation
                full_text = sample_info['sample_text']
                for line in full_text.split('\n'):
                    f.write(f"   {line}\n")
                f.write(f"   {'-' * 90}\n")
        
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    # Initialize random seed
    torch.manual_seed(EVALUATION_CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(EVALUATION_CONFIG['seed'])
    random.seed(EVALUATION_CONFIG['seed'])
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Run evaluation
    evaluator = ModelEvaluator(EVALUATION_CONFIG)
    elo_tracker, all_samples = evaluator.run_evaluation()