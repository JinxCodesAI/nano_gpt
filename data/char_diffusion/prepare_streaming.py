"""
Streaming provider for character-level BERT-style diffusion training.
Applies BERT-style masking (80% [MASK], 10% random, 10% unchanged) on Shakespeare text.
Uses built-in Python libraries only for CPU-optimized processing.
"""
from __future__ import annotations

import os
from typing import Dict, List, Any

import torch

from data.common.provider_base import DataProviderBase
from .file_utils import write_file_atomic, ensure_queue_dirs, get_backlog_size, get_max_sequence_number  
from .masking_utils import apply_stage_masking


def apply_bert_style_corruption_cpu(x: torch.Tensor, mask: torch.Tensor, 
                                  mask_token_id: int, vocab_size: int, rng) -> torch.Tensor:
    """
    Optimized BERT-style corruption using tensor operations.
    Applies 80/10/10 rule: 80% [MASK], 10% random token, 10% unchanged.
    
    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions to corrupt (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token
        vocab_size: Size of vocabulary for random token generation
        rng: Torch random number generator for consistent randomization
        
    Returns:
        corrupted_x: Input with BERT-style corruption applied
    """
    corrupted_x = x.clone()
    
    # Generate random values for all masked positions at once
    rand_vals = torch.rand(mask.shape, generator=rng)
    
    # Create masks for the three corruption types
    mask_positions = mask & (rand_vals < 0.8)  # 80%: [MASK] token
    random_positions = mask & (rand_vals >= 0.8) & (rand_vals < 0.9)  # 10%: random token
    # 10%: unchanged (no mask needed)
    
    # Apply [MASK] tokens
    corrupted_x[mask_positions] = mask_token_id
    
    # Apply random tokens
    if random_positions.any():
        num_random = random_positions.sum().item()
        random_tokens = torch.randint(0, vocab_size, (num_random,), generator=rng)
        corrupted_x[random_positions] = random_tokens
    
    return corrupted_x


class CharDiffusionProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        # Extract diffusion-specific config
        self.mask_probability = kwargs.pop('mask_probability', 0.15)
        self.mask_token_id = kwargs.pop('mask_token_id', None)  # Will be set after vocab creation
        self.ignore_index = kwargs.pop('ignore_index', -100)  # Default PyTorch ignore index
        
        # Extract stage-based configuration
        self.use_all_stages_for_training = kwargs.pop('use_all_stages_for_training', None)
        self.unmasking_stages = kwargs.pop('unmasking_stages', None)
        self.validation_stages = kwargs.pop('validation_stages', None)
        
        super().__init__(*args, **kwargs)
        
        # Load Shakespeare data - fail if not present
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        with open(input_file_path, 'r') as f:
            data = f.read()
        
        # Create vocabulary
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars) + 1  # +1 for [MASK] token
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Add [MASK] token
        self.mask_token_id = len(chars)
        self.stoi['[MASK]'] = self.mask_token_id
        self.itos[self.mask_token_id] = '[MASK]'
        
        # Create train/val splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        self.train_ids = [self.stoi[c] for c in train_data]
        self.val_ids = [self.stoi[c] for c in val_data]
        
        # Validate and initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()
        
        if self.verbose:
            print(f"CharDiffusionProvider initialized:")
            print(f"  vocab_size: {self.vocab_size} (including [MASK])")
            print(f"  mask_probability: {self.mask_probability}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  train_data_size: {len(self.train_ids)}")
            print(f"  val_data_size: {len(self.val_ids)}")
            if self.use_all_stages_for_training is not None:
                print(f"  use_all_stages_for_training: {self.use_all_stages_for_training}")
                print(f"  training_stages: {len(self.unmasking_stages) if self.unmasking_stages else 0}")
                print(f"  validation_stages: {len(self.validation_stages) if self.validation_stages else 0}")
    
    def _validate_stage_config(self):
        """Validate stage configuration and raise exceptions for unsupported options."""
        if self.use_all_stages_for_training is not None:
            if not self.use_all_stages_for_training:
                raise NotImplementedError("use_all_stages_for_training=False is not yet implemented")
            
            if not self.unmasking_stages:
                raise ValueError("unmasking_stages must be provided when use_all_stages_for_training=True")
                
            if not self.validation_stages:
                raise ValueError("validation_stages must be provided when use_all_stages_for_training=True")
    
    def _initialize_stage_distribution(self):
        """Initialize stage distribution for batch generation."""
        if self.use_all_stages_for_training:
            # Calculate how many batches of each stage type to generate per file
            self.train_stage_distribution = self._calculate_stage_distribution(self.unmasking_stages)
            self.val_stage_distribution = self._calculate_stage_distribution(self.validation_stages)
        else:
            self.train_stage_distribution = None
            self.val_stage_distribution = None
    
    def _apply_corruption(self, x: torch.Tensor, mask: torch.Tensor, rng) -> torch.Tensor:
        """Apply corruption to the masked positions.

        Subclasses can override this method to customize the corruption
        procedure while reusing mask generation logic.
        """
        return apply_bert_style_corruption_cpu(
            x, mask, self.mask_token_id, self.vocab_size - 1, rng
        )

    def _apply_stage_corruption(
        self,
        stage_corrupted_x: torch.Tensor,
        original_x: torch.Tensor,
        mask: torch.Tensor,
        rng,
    ) -> torch.Tensor:
        """Finalize corruption for stage-based masking.

        The default implementation trusts the stage masking output, which
        already applied BERT-style corruption.
        """
        return stage_corrupted_x

    def _calculate_stage_distribution(self, stages: List[Dict]) -> List[Dict]:
        """
        Calculate how many samples of each stage type to generate per file.
        
        Args:
            stages: List of stage configurations
            
        Returns:
            List of dicts with stage config and sample count
        """
        total_stages = len(stages)
        samples_per_stage = self.batches_per_file // total_stages
        remainder = self.batches_per_file % total_stages
        
        distribution = []
        for i, stage in enumerate(stages):
            # Distribute remainder across first stages
            count = samples_per_stage + (1 if i < remainder else 0)
            distribution.append({
                'config': stage,
                'count': count
            })
        
        return distribution

    def build_meta(self) -> Dict:
        """Build metadata for BERT-style masked language modeling."""
        if self.block_size is None:
            raise ValueError("block_size must be set for CharDiffusionProvider")
        
        return {
            "dataset_name": "char_diffusion",
            "training_type": "MLM",  # Masked Language Modeling
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
            "mask_probability": self.mask_probability,
            "ignore_index": self.ignore_index,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [self.block_size], "role": "target"},
            ],
        }

    def sample_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with stage-aware masking or default BERT-style masking."""
        if self.use_all_stages_for_training:
            return self._sample_stage_based_batch(split, rng)
        else:
            return self._sample_default_batch(split, rng)
    
    def _sample_default_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with default BERT-style masking."""
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - self.block_size
        
        # Use the provided RNG for proper randomization
        ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()
        
        # Create sequences as tensor
        x_list = [ids[i : i + self.block_size] for i in ix]
        x = torch.tensor(x_list, dtype=torch.long)
        
        # Sample mask using the configured probability
        mask = torch.rand(x.shape, generator=rng) < self.mask_probability

        # Apply corruption with subclass hook
        corrupted_x = self._apply_corruption(x, mask, rng)

        # Create labels: ignore_index for non-masked positions (ignored in loss), original token for masked
        labels = torch.where(mask, x, self.ignore_index)
        
        return {
            "x": corrupted_x,
            "y": labels,
        }
    
    def _sample_stage_based_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch using stage-based masking configuration."""
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - self.block_size
        
        # Get stage distribution for this split
        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution
        
        # Generate samples according to stage distribution
        all_x = []
        all_y = []
        
        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            
            if count == 0:
                continue
                
            # Sample sequences for this stage
            ix = torch.randint(0, max_start_idx, (count * self.batch_size,), generator=rng).tolist()
            x_list = [ids[i : i + self.block_size] for i in ix]
            x_stage = torch.tensor(x_list, dtype=torch.long).view(count, self.batch_size, self.block_size)
            
            # Apply stage-specific masking to each batch in this stage
            for batch_idx in range(count):
                batch_x = x_stage[batch_idx]
                
                # Apply stage-specific masking
                stage_corrupted_x, mask = apply_stage_masking(
                    batch_x, stage_config, self.mask_token_id,
                    self.vocab_size - 1, rng
                )

                corrupted_x = self._apply_stage_corruption(
                    stage_corrupted_x, batch_x, mask, rng
                )

                # Create labels: ignore_index for non-masked positions, original token for masked
                labels = torch.where(mask, batch_x, self.ignore_index)
                
                all_x.append(corrupted_x)
                all_y.append(labels)
        
        # Concatenate all batches and shuffle them randomly to distribute stages
        if all_x:
            combined_x = torch.cat(all_x, dim=0)
            combined_y = torch.cat(all_y, dim=0)
            
            # Randomly shuffle the batches to mix different stage types
            total_samples = combined_x.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)
            
            shuffled_x = combined_x[shuffle_indices]
            shuffled_y = combined_y[shuffle_indices]
            
            # Take only one batch worth of samples (the base class expects single batch)
            return {
                "x": shuffled_x[:self.batch_size],
                "y": shuffled_y[:self.batch_size],
            }
        else:
            # Fallback to default masking if no stages configured
            return self._sample_default_batch(split, rng)


def main():
    """Standalone entrypoint for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Streaming char diffusion provider")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--mask_probability', type=float, default=0.15)
    parser.add_argument('--batches_per_file', type=int, default=100)
    parser.add_argument('--max_backlog_files', type=int, default=2)
    parser.add_argument('--sleep_seconds', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--composition_config', type=str, default=None, 
                       help='Name of composition config to load (e.g., "complex")')

    args = parser.parse_args()

    # Load composition config if specified
    stage_kwargs = {}
    if args.composition_config:
        config_path = os.path.join(os.path.dirname(__file__), 'config', f'{args.composition_config}.py')
        if os.path.exists(config_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"{args.composition_config}_config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract stage-related parameters
            for attr_name in ['use_all_stages_for_training', 'unmasking_stages', 'validation_stages']:
                if hasattr(config_module, attr_name):
                    stage_kwargs[attr_name] = getattr(config_module, attr_name)
            print(f"Loaded composition config from {config_path}")
        else:
            print(f"Warning: composition config file not found at {config_path}")

    data_dir = os.path.dirname(__file__)
    provider = CharDiffusionProvider(
        data_dir=data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
        mask_probability=args.mask_probability,
        batches_per_file=args.batches_per_file,
        max_backlog_files=args.max_backlog_files,
        sleep_seconds=args.sleep_seconds,
        seed=args.seed,
        verbose=args.verbose,
        **stage_kwargs
    )
    provider.run()


# Explicit provider alias for prepare.py discovery
Provider = CharDiffusionProvider


if __name__ == "__main__":
    main()
