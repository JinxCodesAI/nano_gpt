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
from .masking_utils import apply_stage_masking, apply_random_masking_cpu


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


def apply_random_masking_cpu(x: torch.Tensor, mask_probability: float,
                           mask_token_id: int, base_vocab_size: int, rng=None, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized random masking for BERT-style training using tensor operations.

    Args:
        x: Input tokens (batch_size, seq_len)
        mask_probability: Probability of masking each token (0.0 to 1.0)
        mask_token_id: Token ID for [MASK] token
        base_vocab_size: Size of base character vocabulary (excludes [MASK], [SEP])
        attention_mask: Optional boolean/int mask where 1 denotes valid positions to consider

    Returns:
        corrupted_x: Input with masking applied
        mask: Boolean mask indicating which positions were selected for prediction
    """
    # Generate random mask using provided RNG
    mask_tensor = torch.rand(x.shape, generator=rng)
    mask = mask_tensor < mask_probability
    if attention_mask is not None:
        mask = mask & attention_mask.bool()

    # Apply BERT-style corruption
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
        self.sep_token_id = self.base_vocab_size + 1
        self.stoi['[MASK]'] = self.mask_token_id
        self.itos[self.mask_token_id] = '[MASK]'
        self.stoi['[SEP]'] = self.sep_token_id
        self.itos[self.sep_token_id] = '[SEP]'
        self.vocab_size = self.base_vocab_size + 2
        self.newline_token_id = self.stoi.get('\n', None)

        # Create train/val splits
        n = len(data)
        train_data = data[: int(n * 0.9)]
        val_data = data[int(n * 0.9) :]
        self.train_ids = [self.stoi[c] for c in train_data]
        self.val_ids = [self.stoi[c] for c in val_data]

        # Precompute line lists (with newlines kept) and their token ids for line-aligned sampling
        self.train_lines = train_data.splitlines(keepends=True)
        self.val_lines = val_data.splitlines(keepends=True)
        self.train_lines_ids = [[self.stoi[c] for c in line] for line in self.train_lines]
        self.val_lines_ids = [[self.stoi[c] for c in line] for line in self.val_lines]

        # Precompute tensors for efficient line packing
        self.train_line_lens = torch.tensor([len(x) for x in self.train_lines_ids], dtype=torch.long)
        self.val_line_lens = torch.tensor([len(x) for x in self.val_lines_ids], dtype=torch.long)
        self.train_cumsum = self.train_line_lens.cumsum(dim=0)
        self.val_cumsum = self.val_line_lens.cumsum(dim=0)
        self.train_line_offsets = torch.cat([torch.tensor([0], dtype=torch.long), self.train_cumsum[:-1]])
        self.val_line_offsets = torch.cat([torch.tensor([0], dtype=torch.long), self.val_cumsum[:-1]])
        self.train_tokens_flat = torch.tensor([t for line in self.train_lines_ids for t in line], dtype=torch.long)
        self.val_tokens_flat = torch.tensor([t for line in self.val_lines_ids for t in line], dtype=torch.long)
        # Valid start lines: skip blank-only lines (just a single '\n')
        self.train_valid_starts = torch.tensor([i for i, ids in enumerate(self.train_lines_ids) if len(ids) > 1], dtype=torch.long)
        self.val_valid_starts = torch.tensor([i for i, ids in enumerate(self.val_lines_ids) if len(ids) > 1], dtype=torch.long)


        if self.enable_line_aligned_sequences and self.newline_token_id is None:
            raise ValueError("enable_line_aligned_sequences=True requires '\n' to be in the vocabulary")


        # Validate and initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()

        if self.verbose:
            print(f"CharDiffusionProvider initialized:")
            print(f"  base_vocab_size: {self.base_vocab_size}")
            print(f"  vocab_size: {self.vocab_size} (including [MASK],[SEP])")
            print(f"  mask_probability: {self.mask_probability}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  sep_token_id: {self.sep_token_id}")
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
            "sep_token_id": self.sep_token_id,
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
            # Line-aligned, variable-length up to block_size with [SEP] and attention_mask
            lines_ids = self.train_lines_ids if split == "train" else self.val_lines_ids
            num_lines = len(lines_ids)
            if num_lines == 0:
                raise ValueError(f"No lines available for split {split}")

            x = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)
            attention_mask = torch.zeros((self.batch_size, self.block_size), dtype=torch.long)

            # Vectorized start selection (skip blank-only lines)
            valid_starts = self.train_valid_starts if split == "train" else self.val_valid_starts
            if valid_starts.numel() == 0:
                raise ValueError(f"No valid (non-blank) start lines for split {split}")
            starts = valid_starts[torch.randint(0, valid_starts.numel(), (self.batch_size,), generator=rng)]

            # Select precomputed tensors for split
            line_lens = self.train_line_lens if split == "train" else self.val_line_lens
            cumsum = self.train_cumsum if split == "train" else self.val_cumsum
            line_offsets = self.train_line_offsets if split == "train" else self.val_line_offsets
            tokens_flat = self.train_tokens_flat if split == "train" else self.val_tokens_flat

            # Compute last fully includable line index for each start using searchsorted on cumsum
            B = self.block_size - 1  # budget before [SEP]
            s_prev = torch.where(starts > 0, cumsum[starts - 1], torch.zeros_like(starts, dtype=torch.long))
            thresholds = s_prev + B
            last_idx = torch.searchsorted(cumsum, thresholds, right=True) - 1

            for b in range(self.batch_size):
                s = int(starts[b].item())
                e = int(last_idx[b].item())

                if e >= s:
                    # Sum of fully included lines [s..e]
                    sum_full = int((cumsum[e] - (cumsum[s - 1] if s > 0 else 0)).item())
                    remaining = B - sum_full

                    # Copy fully included contiguous region
                    start_ptr = int(line_offsets[s].item())
                    end_ptr_full = int((line_offsets[e] + line_lens[e]).item())
                    if sum_full > 0:
                        x[b, :sum_full] = tokens_flat[start_ptr:end_ptr_full]

                    valid_len = sum_full

                else:
                    # No full line fits; take truncated prefix of first line and force newline
                    budget = B
                    take = max(0, int(budget) - 1)
                    if take > 0:
                        p0 = int(line_offsets[s].item())
                        x[b, :take] = tokens_flat[p0:p0 + take]
                    valid_len = take
                    if self.newline_token_id is not None and budget > 0:
                        x[b, valid_len] = self.newline_token_id
                        valid_len += 1

                # Append [SEP] and set attention mask
                x[b, valid_len] = self.sep_token_id
                attention_mask[b, : valid_len + 1] = 1

            # Apply BERT-style masking only on valid positions (excluding [SEP])
            allowed_mask = attention_mask.bool() & (x != self.sep_token_id)
            corrupted_x, mask = apply_random_masking_cpu(
                x, self.mask_probability, self.mask_token_id, self.base_vocab_size, rng, attention_mask=allowed_mask
            )
            labels = torch.where(mask, x, torch.full_like(x, self.ignore_index))

            return {
                "x": corrupted_x,
                "y": labels,
                "attention_mask": attention_mask,
            }
        else:
            # Legacy fixed-window sampling
            ids = self.train_ids if split == "train" else self.val_ids
            max_start_idx = len(ids) - self.block_size
            ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()
            x_list = [ids[i : i + self.block_size] for i in ix]
            x = torch.tensor(x_list, dtype=torch.long)

            corrupted_x, mask = apply_random_masking_cpu(
                x, self.mask_probability, self.mask_token_id, self.base_vocab_size, rng
            )
            labels = torch.where(mask, x, torch.full_like(x, self.ignore_index))
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

        # Generate samples according to stage distribution within a single batch
        all_x = []
        all_y = []
        stage_info_list = []

        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']

            if count == 0:
                continue

            # Sample sequences for this stage
            ix = torch.randint(0, max_start_idx, (count,), generator=rng).tolist()
            x_list = [ids[i : i + self.block_size] for i in ix]
            x_stage = torch.tensor(x_list, dtype=torch.long)

            # Apply stage-specific masking
            corrupted_x, mask = apply_stage_masking(
                x_stage, stage_config, self.mask_token_id,
                self.base_vocab_size, rng
            )

            # Create labels: ignore_index for non-masked positions, original token for masked
            labels = torch.where(mask, x_stage, self.ignore_index)

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

    def _build_line_aligned(self, split: str, count: int, rng) -> tuple[torch.Tensor, torch.Tensor]:
        """Build line-aligned sequences (x, attention_mask) for 'count' samples.
        Reuses the same logic as in _sample_default_batch but for an arbitrary count.
        """
        lines_ids = self.train_lines_ids if split == "train" else self.val_lines_ids
        if len(lines_ids) == 0:
            raise ValueError(f"No lines available for split {split}")

        x = torch.zeros((count, self.block_size), dtype=torch.long)
        attention_mask = torch.zeros((count, self.block_size), dtype=torch.long)

        valid_starts = self.train_valid_starts if split == "train" else self.val_valid_starts
        if valid_starts.numel() == 0:
            raise ValueError(f"No valid (non-blank) start lines for split {split}")
        starts = valid_starts[torch.randint(0, valid_starts.numel(), (count,), generator=rng)]

        line_lens = self.train_line_lens if split == "train" else self.val_line_lens
        cumsum = self.train_cumsum if split == "train" else self.val_cumsum
        line_offsets = self.train_line_offsets if split == "train" else self.val_line_offsets
        tokens_flat = self.train_tokens_flat if split == "train" else self.val_tokens_flat

        B = self.block_size - 1
        s_prev = torch.where(starts > 0, cumsum[starts - 1], torch.zeros_like(starts, dtype=torch.long))
        thresholds = s_prev + B
        last_idx = torch.searchsorted(cumsum, thresholds, right=True) - 1

        for b in range(count):
            s = int(starts[b].item())
            e = int(last_idx[b].item())
            if e >= s:
                sum_full = int((cumsum[e] - (cumsum[s - 1] if s > 0 else 0)).item())
                start_ptr = int(line_offsets[s].item())
                end_ptr_full = int((line_offsets[e] + line_lens[e]).item())
                if sum_full > 0:
                    x[b, :sum_full] = tokens_flat[start_ptr:end_ptr_full]
                valid_len = sum_full
            else:
                budget = B
                take = max(0, int(budget) - 1)
                if take > 0:
                    p0 = int(line_offsets[s].item())
                    x[b, :take] = tokens_flat[p0:p0 + take]
                valid_len = take
                if self.newline_token_id is not None and budget > 0:
                    x[b, valid_len] = self.newline_token_id
                    valid_len += 1
            x[b, valid_len] = self.sep_token_id
            attention_mask[b, : valid_len + 1] = 1
        return x, attention_mask

    def _produce_stage_based_file(self, split: str, seq: int) -> None:
        """Generate an entire file with stage-based sampling using line-aligned sequences."""
        import time

        rng = torch.Generator()
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        # Get stage distribution for this split
        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution

        all_x = []
        all_y = []
        all_attn = []
        all_stage_info = []

        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            if count == 0:
                continue
            # Build line-aligned base x and attention masks
            base_x, attn = self._build_line_aligned(split, count, rng)
            # Apply stage masking
            stage_x, stage_mask = apply_stage_masking(
                base_x, stage_config, self.mask_token_id, self.base_vocab_size, rng
            )
            # Restrict masking to valid positions (exclude [SEP] and padding after it)
            allowed_mask = attn.bool() & (base_x != self.sep_token_id)
            sanitized_mask = stage_mask & allowed_mask
            sanitized_x = stage_x.clone()
            sanitized_x[~allowed_mask] = base_x[~allowed_mask]
            labels = torch.where(sanitized_mask, base_x, torch.full_like(base_x, self.ignore_index))

            all_x.append(sanitized_x)
            all_y.append(labels)
            all_attn.append(attn)
            all_stage_info.extend([stage_config] * count)

        if all_x:
            combined_x = torch.cat(all_x, dim=0)
            combined_y = torch.cat(all_y, dim=0)
            combined_attn = torch.cat(all_attn, dim=0)

            total_samples = combined_x.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)

            shuffled_x = combined_x[shuffle_indices]
            shuffled_y = combined_y[shuffle_indices]
            shuffled_attn = combined_attn[shuffle_indices]
            shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]

            tensors = {
                "x": shuffled_x,
                "y": shuffled_y,
                "attention_mask": shuffled_attn,
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