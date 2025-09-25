"""
Streaming provider for character-level BERT-style diffusion training.
Applies BERT-style masking (80% [MASK], 10% random, 10% unchanged) on Shakespeare text.
Uses built-in Python libraries only for CPU-optimized processing.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Any

import torch

from data.common.provider_base import DataProviderBase
from data.common.line_aligned_utils import LineAlignedSequenceBuilder
from .file_utils import write_file_atomic, ensure_queue_dirs, get_backlog_size, get_max_sequence_number
from .masking_utils import apply_stage_masking


def apply_bert_style_corruption_cpu(x: torch.Tensor, mask: torch.Tensor,
                                  mask_token_id: int, base_vocab_size: int, rng) -> torch.Tensor:
    """
    Optimized BERT-style corruption using tensor operations.
    Applies 80/10/10 rule: 80% [MASK], 10% random token, 10% unchanged.

    Args:
        x: Original input tokens (batch_size, seq_len)
        mask: Boolean mask of positions to corrupt (batch_size, seq_len)
        mask_token_id: The ID of the [MASK] token
        base_vocab_size: Size of base character vocabulary (excludes special tokens like [MASK], [SEP])
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

    # Apply random tokens from base vocab only (exclude specials)
    if random_positions.any():
        num_random = random_positions.sum().item()
        random_tokens = torch.randint(0, base_vocab_size, (num_random,), generator=rng)
        corrupted_x[random_positions] = random_tokens

    return corrupted_x


def apply_random_masking_base_vocab_only(x: torch.Tensor, mask_probability: float,
                                       mask_token_id: int, base_vocab_size: int, rng=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized random masking for BERT-style training using tensor operations.
    Uses only base vocabulary for random token replacement (excludes special tokens).

    Args:
        x: Input tokens (batch_size, seq_len)
        mask_probability: Probability of masking each token (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token
        base_vocab_size: Size of base character vocabulary (excludes [MASK], [SEP])

    Returns:
        corrupted_x: Input with masking applied
        mask: Boolean mask indicating which positions were selected for prediction
    """
    # Generate random mask using provided RNG
    mask_tensor = torch.rand(x.shape, generator=rng)
    mask = mask_tensor < mask_probability

    # Apply BERT-style corruption with base vocab only
    corrupted_x = apply_bert_style_corruption_cpu(x, mask, mask_token_id, base_vocab_size, rng)

    return corrupted_x, mask




class CharDiffusionProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        # Accept full config for dataset-specific parsing
        cfg = kwargs.pop('config', {}) or {}
        # Extract diffusion-specific config
        self.mask_probability = cfg.get('mask_probability', 0.15)
        self.mask_token_id = cfg.get('mask_token_id', None)  # Will be set after vocab creation
        self.ignore_index = cfg.get('ignore_index', -100)  # Default PyTorch ignore index

        # Extract stage-based configuration
        self.use_all_stages_for_training = cfg.get('use_all_stages_for_training', None)
        self.unmasking_stages = cfg.get('unmasking_stages', None)
        self.validation_stages = cfg.get('validation_stages', None)

        # Model/training mode
        self.model_mode = cfg.get('model_mode', None)
        # Enable stage-based generation when use_all_stages_for_training is True
        self._stages_enabled = bool(self.use_all_stages_for_training)

        # Line-aligned sequences configuration (default True)
        self.enable_line_aligned_sequences = cfg.get('enable_line_aligned_sequences', True)

        super().__init__(*args, **kwargs)

        # Load Shakespeare data - fail if not present
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        with open(input_file_path, 'r') as f:
            data = f.read()

        # Create vocabulary (base chars + [MASK] + [SEP])
        chars = sorted(list(set(data)))
        self.base_vocab_size = len(chars)  # excludes special tokens
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Special tokens
        self.mask_token_id = self.base_vocab_size
        self.pad_token_id = self.base_vocab_size + 1
        self.stoi['[MASK]'] = self.mask_token_id
        self.itos[self.mask_token_id] = '[MASK]'
        self.stoi['[PAD]'] = self.pad_token_id
        self.itos[self.pad_token_id] = '[PAD]'
        self.vocab_size = self.base_vocab_size + 2
        self.newline_token_id = self.stoi.get('\n', None)

        # Create train/val splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        self.train_ids = [self.stoi[c] for c in train_data]
        self.val_ids = [self.stoi[c] for c in val_data]

        # Create line-aligned sequence builders if enabled
        if self.enable_line_aligned_sequences:
            # Split data into lines and create builders
            train_lines = train_data.splitlines(keepends=True)
            val_lines = val_data.splitlines(keepends=True)
            train_lines_ids = [[self.stoi[c] for c in line] for line in train_lines]
            val_lines_ids = [[self.stoi[c] for c in line] for line in val_lines]

            self.train_builder = LineAlignedSequenceBuilder(
                train_lines_ids, self.newline_token_id, self.pad_token_id
            )
            self.val_builder = LineAlignedSequenceBuilder(
                val_lines_ids, self.newline_token_id, self.pad_token_id
            )

            # Keep legacy data structures for backward compatibility
            self.train_lines = train_lines
            self.val_lines = val_lines
            self.train_lines_ids = train_lines_ids
            self.val_lines_ids = val_lines_ids
            self.train_line_lens = self.train_builder.line_lens
            self.val_line_lens = self.val_builder.line_lens
            self.train_cumsum = self.train_builder.cumsum
            self.val_cumsum = self.val_builder.cumsum
            self.train_line_offsets = self.train_builder.line_offsets
            self.val_line_offsets = self.val_builder.line_offsets
            self.train_tokens_flat = self.train_builder.tokens_flat
            self.val_tokens_flat = self.val_builder.tokens_flat
            self.train_valid_starts = self.train_builder.valid_starts
            self.val_valid_starts = self.val_builder.valid_starts
        else:
            # Legacy mode: no line-aligned builders
            self.train_builder = None
            self.val_builder = None


        if self.enable_line_aligned_sequences and self.newline_token_id is None:
            raise ValueError("enable_line_aligned_sequences=True requires '\n' to be in the vocabulary")


        # Validate and initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()

        if self.verbose:
            print(f"CharDiffusionProvider initialized:")
            print(f"  base_vocab_size: {self.base_vocab_size}")
            print(f"  vocab_size: {self.vocab_size} (including [MASK],[PAD])")
            print(f"  mask_probability: {self.mask_probability}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  pad_token_id: {self.pad_token_id}")
            print(f"  enable_line_aligned_sequences: {self.enable_line_aligned_sequences}")
            print(f"  train_data_size: {len(self.train_ids)}")
            print(f"  val_data_size: {len(self.val_ids)}")
            if self.use_all_stages_for_training is not None:
                print(f"  use_all_stages_for_training: {self.use_all_stages_for_training}")
                print(f"  training_stages: {len(self.unmasking_stages) if self.unmasking_stages else 0}")
                print(f"  validation_stages: {len(self.validation_stages) if self.validation_stages else 0}")

                if self.train_stage_distribution:
                    print(f"  train_stage_distribution:")
                    for i, stage_info in enumerate(self.train_stage_distribution):
                        config = stage_info['config']
                        count = stage_info['count']
                        stage_type = config.get('type', 'unknown')
                        if stage_type == 'sticky':
                            ratio = config.get('target_masked_ratio', 'unknown')
                            print(f"    Stage {i}: {stage_type} (ratio={ratio}) -> {count} samples per file")
                        elif stage_type == 'random':
                            ratio = config.get('max_masked_ratio', 'unknown')
                            print(f"    Stage {i}: {stage_type} (max_ratio={ratio}) -> {count} samples per file")
                        elif stage_type == 'span':
                            spans = config.get('spans_count', 'unknown')
                            print(f"    Stage {i}: {stage_type} (spans={spans}) -> {count} samples per file")
                        else:
                            print(f"    Stage {i}: {stage_type} -> {count} samples per file")

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
        if self._stages_enabled:
            # Calculate how many batches of each stage type to generate per file
            self.train_stage_distribution = self._calculate_stage_distribution(self.unmasking_stages)
            self.val_stage_distribution = self._calculate_stage_distribution(self.validation_stages)
        else:
            self.train_stage_distribution = None
            self.val_stage_distribution = None

    def _calculate_stage_distribution(self, stages: List[Dict]) -> List[Dict]:
        """
        Calculate how many samples of each stage type to generate per file.

        Args:
            stages: List of stage configurations

        Returns:
            List of dicts with stage config and sample count
        """
        total_stages = len(stages)
        total_samples = self.batches_per_file * self.batch_size
        samples_per_stage = total_samples // total_stages
        remainder = total_samples % total_stages

        distribution = []
        for i, stage in enumerate(stages):
            # Distribute remainder across first stages
            count = samples_per_stage + (1 if i < remainder else 0)
            if count > 0:  # Only include stages with samples
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
            "pad_token_id": self.pad_token_id,
            "mask_probability": self.mask_probability,
            "ignore_index": self.ignore_index,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [self.block_size], "role": "target"},
                {"name": "attention_mask", "dtype": "int64", "shape": [self.block_size], "role": "mask"},
            ],
        }

    def sample_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with stage-aware masking or default BERT-style masking."""
        if self._stages_enabled:
            # For stage-based generation, we need to generate all samples for the file at once
            # This method will be called by the base class for each batch, but we need to
            # coordinate across all batches in the file. We'll handle this differently.
            raise NotImplementedError("Stage-based sampling requires file-level generation")
        else:
            return self._sample_default_batch(split, rng)

    def _sample_default_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch with default BERT-style masking or line-aligned variable-length sequences."""
        if self.enable_line_aligned_sequences:
            # Step 1: Build variable-length line-aligned sequences
            x, content_lengths = self._build_line_aligned_variable_length(split, self.batch_size, rng)

            # Step 2: Apply random masking to variable-length content
            corrupted_x, labels = self._apply_masking_and_pad(
                x, content_lengths, self.mask_probability, self.mask_token_id, self.base_vocab_size, rng
            )

            return {
                "x": corrupted_x,
                "y": labels,
            }
        else:
            # Legacy fixed-window sampling
            ids = self.train_ids if split == "train" else self.val_ids
            max_start_idx = len(ids) - self.block_size
            ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()
            x_list = [ids[i : i + self.block_size] for i in ix]
            x = torch.tensor(x_list, dtype=torch.long)

            corrupted_x, mask = apply_random_masking_base_vocab_only(
                x, self.mask_probability, self.mask_token_id, self.base_vocab_size, rng
            )
            labels = torch.where(mask, x, torch.full_like(x, self.ignore_index))
            return {
                "x": corrupted_x,
                "y": labels,
            }

    def _sample_stage_based_batch(self, split: str, rng) -> Dict[str, Any]:
        """Sample a batch using stage-based masking configuration with line-aligned sequences."""
        # Get stage distribution for this split
        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution

        # Calculate total samples needed
        total_samples = sum(stage_info['count'] for stage_info in stage_distribution)
        if total_samples == 0:
            return self._sample_default_batch(split, rng)

        # Generate samples according to stage distribution
        all_x = []
        all_y = []
        stage_info_list = []

        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']

            if count == 0:
                continue

            # Step 1: Build variable-length line-aligned sequences
            x, content_lengths = self._build_line_aligned_variable_length(split, count, rng)

            # Step 2: Apply stage masking to variable-length content
            corrupted_x, labels = self._apply_stage_masking_and_pad(
                x, content_lengths, stage_config, self.mask_token_id, self.base_vocab_size, rng
            )

            all_x.append(corrupted_x)
            all_y.append(labels)

            # Track stage info for debugging
            stage_info_list.extend([stage_config] * count)

        # Concatenate all samples and shuffle them randomly to mix different stage types
        if all_x:
            combined_x = torch.cat(all_x, dim=0)
            combined_y = torch.cat(all_y, dim=0)

            # Randomly shuffle the samples to mix different stage types
            total_samples = combined_x.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)

            shuffled_x = combined_x[shuffle_indices]
            shuffled_y = combined_y[shuffle_indices]

            # Create shuffled stage info for debugging
            shuffled_stage_info = [stage_info_list[i] for i in shuffle_indices.tolist()]

            return {
                "x": shuffled_x,
                "y": shuffled_y,
                "stage_info": shuffled_stage_info  # Add stage info for debugging
            }
        else:
            # Fallback to default masking if no stages configured
            return self._sample_default_batch(split, rng)

    def produce_one_file(self, split: str, seq: int) -> None:
        """Override to handle stage-based generation at file level."""
        if self._stages_enabled:
            self._produce_stage_based_file(split, seq)
        else:
            # Use default file production for non-stage-based generation
            super().produce_one_file(split, seq)

    def _build_line_aligned_variable_length(self, split: str, count: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Step 1: Build variable-length line-aligned sequences (shared by all masking approaches).
        Returns (x, content_lengths) where x contains content and content_lengths tracks actual lengths.
        """
        if self.enable_line_aligned_sequences:
            # Use shared builder
            builder = self.train_builder if split == "train" else self.val_builder
            return builder.build_variable_length_sequences(count, self.block_size, rng)
        else:
            # Legacy fallback (should not be reached in normal operation)
            raise NotImplementedError("Legacy non-line-aligned mode not supported in this method")

    def _apply_masking_and_pad(self, x: torch.Tensor, content_lengths: torch.Tensor,
                              mask_probability: float, mask_token_id: int, base_vocab_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Step 2 & 3: Apply random masking to variable-length content, then pad with [PAD] tokens."""
        count = x.shape[0]
        corrupted_x = x.clone()
        labels = torch.full_like(x, self.ignore_index)

        for b in range(count):
            content_len = int(content_lengths[b].item())
            if content_len > 0:
                # Step 2: Apply masking only to the content portion
                content = x[b, :content_len].unsqueeze(0)  # Add batch dim
                corrupted_content, mask = apply_random_masking_base_vocab_only(
                    content, mask_probability, mask_token_id, base_vocab_size, rng
                )

                # Update corrupted sequence and labels for content only
                corrupted_x[b, :content_len] = corrupted_content.squeeze(0)
                labels[b, :content_len] = torch.where(
                    mask.squeeze(0), x[b, :content_len], self.ignore_index
                )

            # Step 3: Fill remaining positions with [PAD] tokens
            if content_lengths[b] < self.block_size:
                corrupted_x[b, content_lengths[b]:] = self.pad_token_id
                # labels already set to ignore_index for padding positions

        return corrupted_x, labels

    def _apply_stage_masking_and_pad(self, x: torch.Tensor, content_lengths: torch.Tensor,
                                    stage_config: Dict[str, Any], mask_token_id: int, base_vocab_size: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Step 2 & 3: Apply stage masking to variable-length content, then pad with [PAD] tokens."""
        count = x.shape[0]
        corrupted_x = x.clone()
        labels = torch.full_like(x, self.ignore_index)

        for b in range(count):
            content_len = int(content_lengths[b].item())
            if content_len > 0:
                # Step 2: Apply stage masking only to the content portion
                content = x[b, :content_len].unsqueeze(0)  # Add batch dim
                corrupted_content, mask = apply_stage_masking(
                    content, stage_config, mask_token_id, base_vocab_size, rng
                )

                # Update corrupted sequence and labels for content only
                corrupted_x[b, :content_len] = corrupted_content.squeeze(0)
                labels[b, :content_len] = torch.where(
                    mask.squeeze(0), x[b, :content_len], self.ignore_index
                )

        # Step 3: Apply padding - use shared builder if available
        if self.enable_line_aligned_sequences and hasattr(self, 'train_builder'):
            # We need to determine which builder to use, but we don't have split info here
            # Use train_builder as default (this method is called from stage-based generation)
            corrupted_x = self.train_builder.apply_padding(corrupted_x, content_lengths)
        else:
            # Legacy padding
            for b in range(count):
                if content_lengths[b] < self.block_size:
                    corrupted_x[b, content_lengths[b]:] = self.pad_token_id

        return corrupted_x, labels

    def _produce_stage_based_file(self, split: str, seq: int) -> None:
        """Generate an entire file with stage-based sampling using simplified line-aligned approach."""
        import time

        rng = torch.Generator()
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        # Use the same logic as _sample_stage_based_batch but for file-level generation
        # Get stage distribution for this split
        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution

        all_x = []
        all_y = []
        all_stage_info = []

        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            if count == 0:
                continue

            # Step 1: Build variable-length line-aligned sequences
            x, content_lengths = self._build_line_aligned_variable_length(split, count, rng)

            # Step 2 & 3: Apply stage masking and pad
            corrupted_x, labels = self._apply_stage_masking_and_pad(
                x, content_lengths, stage_config, self.mask_token_id, self.base_vocab_size, rng
            )

            all_x.append(corrupted_x)
            all_y.append(labels)
            all_stage_info.extend([stage_config] * count)

        if all_x:
            combined_x = torch.cat(all_x, dim=0)
            combined_y = torch.cat(all_y, dim=0)

            total_samples = combined_x.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)

            shuffled_x = combined_x[shuffle_indices]
            shuffled_y = combined_y[shuffle_indices]
            shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]

            tensors = {
                "x": shuffled_x,
                "y": shuffled_y,
            }

            metadata = {
                "batch_size": total_samples,
                "block_size": self.block_size,
                "stage_info": shuffled_stage_info,
            }

            # Write atomic (following base class pattern)
            d = self.train_dir if split == "train" else self.val_dir
            ts = int(time.time() * 1000)
            tmp_name = f".tmp-{ts}-{seq:06d}.pt"
            final_name = f"{ts}-{seq:06d}-{total_samples}.pt"
            tmp_path = os.path.join(d, tmp_name)
            final_path = os.path.join(d, final_name)
            torch.save({"tensors": tensors, "metadata": metadata}, tmp_path)
            os.replace(tmp_path, final_path)
            if self.verbose:
                print(f"[provider] produced stage-based file: {final_path}")

        if all_x:
            combined_x = torch.cat(all_x, dim=0)
            combined_y = torch.cat(all_y, dim=0)

            total_samples = combined_x.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)

            shuffled_x = combined_x[shuffle_indices]
            shuffled_y = combined_y[shuffle_indices]
            shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]

            tensors = {
                "x": shuffled_x,
                "y": shuffled_y,
            }

            metadata = {
                "batch_size": self.batch_size,
                "num_batches": self.batches_per_file,
                "file_idx": seq,
                "split": split,
                "produced_at": int(time.time() * 1000),
                "stage_info": shuffled_stage_info,
                "stage_distribution": stage_distribution,
            }

            d = self.train_dir if split == "train" else self.val_dir
            ts = metadata["produced_at"]
            tmp_name = f".tmp-{ts}-{seq:06d}.pt"
            final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
            tmp_path = os.path.join(d, tmp_name)
            final_path = os.path.join(d, final_name)
            torch.save({"tensors": tensors, "metadata": metadata}, tmp_path)
            os.replace(tmp_path, final_path)
            if self.verbose:
                print(f"[provider] produced stage-based file: {final_path}")
        else:
            super().produce_one_file(split, seq)


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