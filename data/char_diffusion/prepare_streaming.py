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

        device_arg = kwargs.pop('device', None)
        if device_arg is None:
            device_arg = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_obj = torch.device(device_arg)
        if device_obj.type == 'cuda' and not torch.cuda.is_available():
            device_obj = torch.device('cpu')
        self._tensor_device = device_obj
        
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

        if self.block_size is None:
            raise ValueError("block_size must be set for CharDiffusionProvider")
        self.train_ids_tensor = torch.tensor(self.train_ids, dtype=torch.long, device=self._tensor_device)
        self.val_ids_tensor = torch.tensor(self.val_ids, dtype=torch.long, device=self._tensor_device)
        self._sequence_offsets = torch.arange(self.block_size, dtype=torch.long, device=self._tensor_device)

        # Pre-compute candidate start positions aligned with line boundaries
        self._newline_token_ids = {self.stoi[ch] for ch in ("\n", "\r") if ch in self.stoi}
        self._train_line_start_indices = self._compute_line_start_indices(self.train_ids).to(self._tensor_device)
        self._val_line_start_indices = self._compute_line_start_indices(self.val_ids).to(self._tensor_device)
        
        # Validate and initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()
        self._stage_cycle_state = {"train": [], "val": []}
        
        if self.verbose:
            print(f"CharDiffusionProvider initialized:")
            print(f"  vocab_size: {self.vocab_size} (including [MASK])")
            print(f"  mask_probability: {self.mask_probability}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  train_data_size: {len(self.train_ids)}")
            print(f"  val_data_size: {len(self.val_ids)}")
            print(f"  compute_device: {self._tensor_device}")
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

    def _create_labels(
        self, original_x: torch.Tensor, mask: torch.Tensor, split: str
    ) -> torch.Tensor:
        """Create target labels for a batch.

        By default, labels are the original tokens where masking was applied
        and ``ignore_index`` everywhere else, matching the standard MLM
        objective. Subclasses can override this hook to customize label
        generation (e.g., to keep full targets for certain splits).
        """
        del split  # unused in the default implementation
        return torch.where(mask, original_x, self.ignore_index)

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

    def _get_stage_distribution(self, split: str) -> Optional[List[Dict]]:
        return self.train_stage_distribution if split == "train" else self.val_stage_distribution

    def _ensure_stage_cycle(self, split: str, rng) -> None:
        if not self.use_all_stages_for_training:
            return
        if self._stage_cycle_state[split]:
            return
        distribution = self._get_stage_distribution(split)
        if not distribution:
            raise ValueError("Stage distribution requested but not configured.")

        stage_pool: List[Dict] = []
        for info in distribution:
            stage_pool.extend([info["config"]] * info["count"])

        if not stage_pool:
            raise ValueError("Stage distribution produced no entries; check configuration.")

        perm = torch.randperm(
            len(stage_pool), generator=rng, device=self._tensor_device
        ).tolist()
        shuffled = [stage_pool[i] for i in perm]
        self._stage_cycle_state[split] = shuffled

    def _pop_stage_config(self, split: str, rng) -> Dict:
        if not self.use_all_stages_for_training:
            raise RuntimeError("Stage config requested but stage training is disabled.")
        self._ensure_stage_cycle(split, rng)
        if not self._stage_cycle_state[split]:
            self._ensure_stage_cycle(split, rng)
        config = self._stage_cycle_state[split].pop()
        if not self._stage_cycle_state[split]:
            # Mark cycle as completed; next request will refresh
            self._stage_cycle_state[split] = []
        return config

    def _compute_line_start_indices(self, ids: List[int]) -> torch.Tensor:
        """Identify start indices that align with the first non-newline character on each line."""
        if self.block_size is None:
            raise ValueError("block_size must be set before computing line-aligned starts")

        limit = len(ids) - self.block_size
        if limit < 0:
            raise ValueError("block_size is larger than the available sequence length")

        starts: List[int] = []
        last_was_newline = True  # treat start-of-file as a newline boundary
        newline_ids = self._newline_token_ids

        for idx, token_id in enumerate(ids):
            if token_id in newline_ids:
                last_was_newline = True
                continue
            if last_was_newline:
                if idx <= limit:
                    starts.append(idx)
                last_was_newline = False
                continue
            last_was_newline = False

        if not starts:
            raise ValueError(
                "No valid line-aligned start positions found; check input text or reduce block_size."
            )
        return torch.tensor(starts, dtype=torch.long)

    def _get_line_start_indices(self, split: str) -> torch.Tensor:
        return self._train_line_start_indices if split == "train" else self._val_line_start_indices

    def _sample_start_positions(
        self, start_candidates: torch.Tensor, num_samples: int, rng
    ) -> torch.Tensor:
        if start_candidates.numel() == 0:
            raise ValueError("No available start positions to sample from.")
        choice_indices = torch.randint(
            0,
            start_candidates.shape[0],
            (num_samples,),
            generator=rng,
            device=self._tensor_device,
        )
        return start_candidates[choice_indices]

    def _gather_sequences(
        self, ids_tensor: torch.Tensor, start_indices: torch.Tensor
    ) -> torch.Tensor:
        positions = start_indices.unsqueeze(1) + self._sequence_offsets
        flat_positions = positions.reshape(-1)
        gathered = ids_tensor[flat_positions]
        return gathered.view(start_indices.shape[0], self.block_size)

    def _prepare_generator(self, rng: torch.Generator) -> torch.Generator:
        if self._tensor_device.type == "cuda":
            seed = torch.randint(0, 2**31 - 1, (1,), generator=rng).item()
            device_rng = torch.Generator(device=self._tensor_device)
            device_rng.manual_seed(seed)
            return device_rng
        return rng

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
        active_rng = self._prepare_generator(rng)
        if self.use_all_stages_for_training:
            return self._sample_stage_based_batch(split, active_rng)
        else:
            return self._sample_default_batch(split, active_rng)
    
    def _sample_default_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with default BERT-style masking."""
        ids_tensor = self.train_ids_tensor if split == "train" else self.val_ids_tensor
        start_candidates = self._get_line_start_indices(split)
        start_indices = self._sample_start_positions(start_candidates, self.batch_size, rng)

        x = self._gather_sequences(ids_tensor, start_indices)
        
        # Sample mask using the configured probability
        mask = torch.rand(
            x.shape, generator=rng, device=self._tensor_device
        ) < self.mask_probability

        # Apply corruption with subclass hook
        corrupted_x = self._apply_corruption(x, mask, rng)

        # Create labels: ignore_index for non-masked positions (ignored in loss), original token for masked
        labels = self._create_labels(x, mask, split)

        return {
            "x": corrupted_x,
            "y": labels,
        }
    
    def _sample_stage_based_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch using stage-based masking configuration."""
        ids_tensor = self.train_ids_tensor if split == "train" else self.val_ids_tensor
        start_candidates = self._get_line_start_indices(split)

        stage_config = self._pop_stage_config(split, rng)

        start_indices = self._sample_start_positions(start_candidates, self.batch_size, rng)
        batch_x = self._gather_sequences(ids_tensor, start_indices)

        stage_corrupted_x, mask = apply_stage_masking(
            batch_x, stage_config, self.mask_token_id, self.vocab_size - 1, rng
        )

        corrupted_x = self._apply_stage_corruption(
            stage_corrupted_x, batch_x, mask, rng
        )

        labels = self._create_labels(batch_x, mask, split)

        return {"x": corrupted_x, "y": labels}


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
