#!/usr/bin/env python3
"""
Completion Variance Analysis Script

This script loads a language model and a scoring model, then analyzes how much
the scoring model's ratings vary for different completions generated from the
same masked input.

The script:
1. Loads two models: a language model (LANGUAGE_MODEL) and a scoring model (SEQUENCE_SCORER)
2. Reads samples from training set and applies random masking
3. Generates batch_size independent completions from the same masked input
4. Uses the scoring model to rate all completions
5. Analyzes variance in ratings and completion diversity

Usage:
    python completion_variance_analysis.py --language_model out/ckpt_unmasking_1000.pt \
                                          --scoring_model out/ckpt_sequence_scorer_500.pt \
                                          --batch_size 8 \
                                          --mask_ratio 0.3
"""

import os
import sys
import argparse
import pickle
import math
import random
from contextlib import nullcontext
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from collections import Counter

# Import project modules
from model import GPTConfig, GPT, ModelMode
from sample_utils import nucleus_sample, predict_and_sample_tokens, calculate_sequence_scores
from training_utils.data_loading_utils import load_memmap_data, sample_indices_random, vectorized_data_loading
from training_utils.masking_strategies import apply_random_masking_gpu, apply_span_masking_gpu, apply_target_driven_sticky_masking_gpu


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze variance in scoring model ratings for different completions from same input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument('--language_model', type=str, required=True,
                       help='Path to language model checkpoint (LANGUAGE_MODEL type)')
    parser.add_argument('--scoring_model', type=str, required=True,
                       help='Path to scoring model checkpoint (SEQUENCE_SCORER type)')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='shakespeare_char',
                       help='Dataset name (determines data directory)')
    parser.add_argument('--out_dir', type=str, default='out',
                       help='Directory containing model checkpoints')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Number of independent completions to generate from same input')
    parser.add_argument('--sequence_length', type=int, default=1024,
                       help='Length of sequences to work with')
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                       help='Fraction of tokens to mask (0.0 to 1.0)')

    # Masking strategy parameters
    parser.add_argument('--masking_strategy', type=str, default='random',
                       choices=['random', 'span', 'sticky'],
                       help='Masking strategy to use')
    parser.add_argument('--spans_count', type=int, default=3,
                       help='Number of spans to mask (for span masking strategy)')
    parser.add_argument('--p1_probability', type=float, default=None,
                       help='Probability of masking when no neighbors are masked (sticky masking). Default: mask_ratio/20')
    parser.add_argument('--p2_probability', type=float, default=None,
                       help='Probability of masking when neighbors are masked (sticky masking). Default: mask_ratio/5')

    # Generation quality parameters
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (higher = more random)')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Nucleus sampling parameter (1.0 = disabled)')
    
    # Analysis parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output (check completion diversity)')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Data type for inference')
    parser.add_argument('--compile', action='store_true',
                       help='Compile models with torch.compile')
    
    return parser.parse_args()


class CompletionVarianceAnalyzer:
    """Main class for analyzing completion variance"""
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.dtype = args.dtype
        
        # Set up device and context
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        # Initialize models and data
        self.language_model = None
        self.scoring_model = None
        self.vocab_info = None
        self.training_data = None
        
    def load_vocabulary(self) -> Dict[str, Any]:
        """Load vocabulary metadata from dataset"""
        data_dir = os.path.join('data', self.args.dataset)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = meta['vocab_size']
        mask_token_id = vocab_size
        cls_token_id = vocab_size + 5  # Reserved special token slot
        
        def decode(token_ids):
            """Decode token IDs to text"""
            result = []
            for token_id in token_ids:
                if token_id == mask_token_id:
                    result.append('[MASK]')
                elif token_id == cls_token_id:
                    result.append('[CLS]')
                elif token_id < len(itos):
                    result.append(itos[token_id])
                else:
                    result.append('[UNK]')
            return ''.join(result)
        
        vocab_info = {
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size,
            'mask_token_id': mask_token_id,
            'cls_token_id': cls_token_id,
            'extended_vocab_size': vocab_size + 15,
            'decode': decode,
            'dataset_name': self.args.dataset
        }
        
        if self.args.verbose:
            print(f"Vocabulary loaded: size={vocab_size}, mask_token_id={mask_token_id}, cls_token_id={cls_token_id}")
        
        return vocab_info
    
    def load_model(self, checkpoint_path: str, expected_mode: ModelMode) -> GPT:
        """Load a model from checkpoint and verify its mode"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if self.args.verbose:
            print(f"Loading model from {checkpoint_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_args = checkpoint['model_args']
        
        # Add backward compatibility for attention_type
        if 'attention_type' not in model_args:
            model_args['attention_type'] = 'causal'
        
        # Create model
        model_config = GPTConfig(**model_args)
        model = GPT(model_config)
        
        # Verify model mode
        if model_config.mode != expected_mode:
            raise ValueError(f"Expected {expected_mode.value} model, but got {model_config.mode.value}")
        
        # Load weights
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        if self.args.compile:
            model = torch.compile(model)
        
        if self.args.verbose:
            print(f"  Model loaded: {model_config.mode.value}, attention={model_config.attention_type}")
            print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
        
        return model
    
    def load_training_data(self):
        """Load training data for sampling"""
        data_dir = os.path.join('data', self.args.dataset)
        train_path = os.path.join(data_dir, 'train.bin')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        self.training_data = load_memmap_data(train_path)
        
        if self.args.verbose:
            print(f"Training data loaded: {len(self.training_data)} tokens")
    
    def sample_and_mask_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample data from training set and apply selected masking strategy"""
        # Sample random starting position
        max_start = len(self.training_data) - self.args.sequence_length
        if max_start <= 0:
            raise ValueError(f"Training data too short for sequence length {self.args.sequence_length}")

        # Use deterministic sampling for reproducibility
        torch.manual_seed(self.args.seed)
        start_idx = torch.randint(0, max_start, (1,)).item()

        # Extract sequence
        sequence_data = self.training_data[start_idx:start_idx + self.args.sequence_length]
        original_tokens = torch.from_numpy(sequence_data.astype(np.int64)).to(self.device)
        original_tokens = original_tokens.unsqueeze(0)  # Add batch dimension

        # Apply selected masking strategy
        if self.args.masking_strategy == 'random':
            masked_tokens, mask = apply_random_masking_gpu(
                original_tokens,
                self.args.mask_ratio,
                self.vocab_info['mask_token_id'],
                self.vocab_info['vocab_size']
            )
            strategy_info = f"random (ratio={self.args.mask_ratio})"

        elif self.args.masking_strategy == 'span':
            masked_tokens, mask = apply_span_masking_gpu(
                original_tokens,
                self.args.spans_count,
                self.vocab_info['mask_token_id'],
                self.vocab_info['vocab_size']
            )
            strategy_info = f"span (spans={self.args.spans_count})"

        elif self.args.masking_strategy == 'sticky':
            # Set default probabilities if not provided
            p1_prob = self.args.p1_probability if self.args.p1_probability is not None else self.args.mask_ratio / 20
            p2_prob = self.args.p2_probability if self.args.p2_probability is not None else self.args.mask_ratio / 5

            masked_tokens, mask = apply_target_driven_sticky_masking_gpu(
                original_tokens,
                self.args.mask_ratio,
                p1_prob,
                p2_prob,
                self.vocab_info['mask_token_id'],
                self.vocab_info['vocab_size']
            )
            strategy_info = f"sticky (target={self.args.mask_ratio}, p1={p1_prob:.4f}, p2={p2_prob:.4f})"

        else:
            raise ValueError(f"Unknown masking strategy: {self.args.masking_strategy}")

        if self.args.verbose:
            num_masked = mask.sum().item()
            print(f"Masking strategy: {strategy_info}")
            print(f"Masked {num_masked}/{self.args.sequence_length} tokens ({num_masked/self.args.sequence_length*100:.1f}%)")
            if self.args.debug:
                # Show masked sequence
                masked_text = self.vocab_info['decode'](masked_tokens[0].tolist())
                print(f"  Masked: {masked_text[:100]}{'...' if len(masked_text) > 100 else ''}")

        return masked_tokens, mask

    def generate_completions(self, masked_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Generate multiple independent completions from the same masked input"""
        # Create batch by repeating the same masked input
        batch_tokens = masked_tokens.repeat(self.args.batch_size, 1)

        # Find masked positions
        current_mask = (batch_tokens == self.vocab_info['mask_token_id'])

        if not current_mask.any():
            if self.args.verbose:
                print("No masked tokens found!")
            return batch_tokens

        with torch.no_grad():
            with self.ctx:
                # Single forward pass for all completions
                logits, _ = self.language_model(batch_tokens, None)

                # Sample new tokens for masked positions in each sequence
                for batch_idx in range(self.args.batch_size):
                    mask_positions = current_mask[batch_idx]
                    if mask_positions.any():
                        masked_logits = logits[batch_idx, mask_positions]

                        # Apply temperature and nucleus sampling
                        new_tokens = nucleus_sample(
                            masked_logits,
                            top_p=self.args.top_p,
                            temperature=self.args.temperature,
                            vocab_size=self.vocab_info['vocab_size']
                        )

                        batch_tokens[batch_idx, mask_positions] = new_tokens

        if self.args.debug:
            for i in range(min(3, self.args.batch_size)):  # Show first few completions
                completion_text = self.vocab_info['decode'](batch_tokens[i].tolist())
                print(f"  Completion {i}: {completion_text[:100]}{'...' if len(completion_text) > 100 else ''}")

        return batch_tokens

    def rate_completions(self, completions: torch.Tensor) -> List[float]:
        """Rate all completions using the scoring model"""
        with torch.no_grad():
            with self.ctx:
                # Calculate sequence scores for all completions at once
                scores = calculate_sequence_scores(
                    model=self.scoring_model,
                    tokens=completions,
                    cls_token_id=self.vocab_info['cls_token_id'],
                    device=self.device,
                    ctx=self.ctx
                )

                if self.args.debug:
                    for i, score in enumerate(scores[:3]):  # Show first 3
                        print(f"  Completion {i} rating: {score:.4f}")

        return scores

    def check_completion_diversity(self, completions: torch.Tensor) -> Dict[str, Any]:
        """Check if completions are actually different from each other"""
        batch_size = completions.shape[0]
        diversity_stats = {
            'unique_completions': 0,
            'total_completions': batch_size,
            'diversity_ratio': 0.0,
            'token_differences': [],
            'text_differences': []
        }

        # Convert completions to text for comparison
        completion_texts = []
        completion_tokens = []

        for i in range(batch_size):
            tokens = completions[i].tolist()
            text = self.vocab_info['decode'](tokens)
            completion_texts.append(text)
            completion_tokens.append(tokens)

        # Count unique completions
        unique_texts = set(completion_texts)
        diversity_stats['unique_completions'] = len(unique_texts)
        diversity_stats['diversity_ratio'] = len(unique_texts) / len(completion_texts)

        # Calculate pairwise token differences
        for i in range(len(completion_tokens)):
            for j in range(i + 1, len(completion_tokens)):
                tokens_i = completion_tokens[i]
                tokens_j = completion_tokens[j]

                # Count different tokens
                min_len = min(len(tokens_i), len(tokens_j))
                differences = sum(1 for k in range(min_len) if tokens_i[k] != tokens_j[k])

                # Add length difference if sequences have different lengths
                differences += abs(len(tokens_i) - len(tokens_j))

                diversity_stats['token_differences'].append(differences)

        # Store text differences for debugging
        if self.args.debug:
            diversity_stats['text_differences'] = completion_texts

        return diversity_stats

    def analyze_rating_variance(self, ratings: List[float], diversity_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze variance in ratings and compute statistics"""
        if not ratings:
            return {}

        ratings_array = np.array(ratings)

        variance_stats = {
            'num_ratings': len(ratings),
            'mean_rating': float(np.mean(ratings_array)),
            'std_rating': float(np.std(ratings_array)),
            'var_rating': float(np.var(ratings_array)),
            'min_rating': float(np.min(ratings_array)),
            'max_rating': float(np.max(ratings_array)),
            'range_rating': float(np.max(ratings_array) - np.min(ratings_array)),
            'median_rating': float(np.median(ratings_array)),
            'q25_rating': float(np.percentile(ratings_array, 25)),
            'q75_rating': float(np.percentile(ratings_array, 75)),
            'iqr_rating': float(np.percentile(ratings_array, 75) - np.percentile(ratings_array, 25)),
        }

        # Coefficient of variation (relative variability)
        if variance_stats['mean_rating'] != 0:
            variance_stats['cv_rating'] = variance_stats['std_rating'] / abs(variance_stats['mean_rating'])
        else:
            variance_stats['cv_rating'] = float('inf')

        # Add diversity information
        variance_stats.update({
            'diversity_ratio': diversity_stats['diversity_ratio'],
            'unique_completions': diversity_stats['unique_completions'],
            'total_completions': diversity_stats['total_completions'],
        })

        # Calculate average token differences if available
        if diversity_stats['token_differences']:
            token_diffs = np.array(diversity_stats['token_differences'])
            variance_stats.update({
                'mean_token_differences': float(np.mean(token_diffs)),
                'std_token_differences': float(np.std(token_diffs)),
                'max_token_differences': float(np.max(token_diffs)),
                'min_token_differences': float(np.min(token_diffs)),
            })

        return variance_stats

    def print_results(self, ratings: List[float], diversity_stats: Dict[str, Any], variance_stats: Dict[str, Any]):
        """Print analysis results"""
        print("=== Results ===")
        print(f"Ratings: {[f'{r:.4f}' for r in ratings]}")
        print(f"Mean ± Std: {variance_stats['mean_rating']:.4f} ± {variance_stats['std_rating']:.4f}")
        print(f"Range: [{variance_stats['min_rating']:.4f}, {variance_stats['max_rating']:.4f}] (span: {variance_stats['range_rating']:.4f})")
        print(f"Coefficient of Variation: {variance_stats['cv_rating']:.4f}")
        print(f"Diversity: {diversity_stats['unique_completions']}/{diversity_stats['total_completions']} unique ({diversity_stats['diversity_ratio']:.2f})")

        if diversity_stats['token_differences']:
            print(f"Token differences: mean={variance_stats['mean_token_differences']:.1f}, max={variance_stats['max_token_differences']:.0f}")

        if self.args.debug and 'text_differences' in diversity_stats:
            print("Completion texts:")
            for i, text in enumerate(diversity_stats['text_differences'][:3]):  # Show first 3
                print(f"  {i}: {text[:80]}{'...' if len(text) > 80 else ''}")

        print()



    def run_analysis(self):
        """Run the complete variance analysis"""
        print("=== Completion Variance Analysis ===")
        print(f"Language model: {self.args.language_model}")
        print(f"Scoring model: {self.args.scoring_model}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Masking strategy: {self.args.masking_strategy}")
        if self.args.masking_strategy == 'random':
            print(f"Mask ratio: {self.args.mask_ratio}")
        elif self.args.masking_strategy == 'span':
            print(f"Spans count: {self.args.spans_count}")
        elif self.args.masking_strategy == 'sticky':
            p1_prob = self.args.p1_probability if self.args.p1_probability is not None else self.args.mask_ratio / 20
            p2_prob = self.args.p2_probability if self.args.p2_probability is not None else self.args.mask_ratio / 5
            print(f"Target mask ratio: {self.args.mask_ratio}")
            print(f"P1 probability: {p1_prob:.4f}, P2 probability: {p2_prob:.4f}")
        print()

        # Set random seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # Load components
        self.vocab_info = self.load_vocabulary()
        self.language_model = self.load_model(
            os.path.join(self.args.out_dir, self.args.language_model),
            ModelMode.LANGUAGE_MODEL
        )
        self.scoring_model = self.load_model(
            os.path.join(self.args.out_dir, self.args.scoring_model),
            ModelMode.SEQUENCE_SCORER
        )
        self.load_training_data()

        print("All components loaded successfully!")
        print()

        # Sample and mask data
        masked_tokens, mask = self.sample_and_mask_data()

        # Generate multiple completions
        completions = self.generate_completions(masked_tokens, mask)

        # Rate completions with scoring model
        ratings = self.rate_completions(completions)

        # Check completion diversity
        diversity_stats = self.check_completion_diversity(completions)

        # Analyze rating variance
        variance_stats = self.analyze_rating_variance(ratings, diversity_stats)

        # Print results
        self.print_results(ratings, diversity_stats, variance_stats)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        analyzer = CompletionVarianceAnalyzer(args)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
