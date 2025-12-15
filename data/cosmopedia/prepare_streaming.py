"""Streaming provider for Cosmopedia dataset with on-the-fly tokenizer training and Discrete Diffusion."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Sequence, List

import torch
import datasets
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from data.common.provider_base import DataProviderBase
# Import corruption utils from the reference dataset location
try:
    from data.char_random_replacement.corruption_utils import (
        RandomReplacementCorruptor,
        apply_mixed_corruption,
        build_candidate_token_ids,
    )
    from data.char_diffusion.masking_utils import apply_stage_masking
except ImportError:
    # Fallback or error handling if path is different, but strict plan says to use this
    raise ImportError("Could not import corruption_utils or masking_utils")

class CosmopediaProvider(DataProviderBase):
    """
    Streams from HuggingFaceTB/cosmopedia, trains a BPE tokenizer if needed.
    Applies Discrete Diffusion (Random Replacement) corruption to BPE tokens using Stage-based masking.
    """

    DEFAULT_CONFIGS = ['web_samples_v1', 'web_samples_v2']

    def __init__(
        self,
        *args,
        vocab_size: int = 4096,
        tokenizer_train_samples: int = 100000,
        original_token_probability_multiplier: float = 1.0,
        train_corruption_mixture: Tuple[float, float, float] = (0.8, 0.2, 0.0),
        dataset_partial_targets: bool = False,
        use_all_stages_for_training: bool = False,
        unmasking_stages: Optional[List[Dict]] = None,
        validation_stages: Optional[List[Dict]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = int(vocab_size)
        self.tokenizer_train_samples = int(tokenizer_train_samples)
        
        # Corruption params
        self._original_multiplier = float(original_token_probability_multiplier)
        self._train_corruption_mixture = tuple(train_corruption_mixture)
        self._dataset_partial_targets = bool(dataset_partial_targets)
        
        # Stage configuration
        self.use_all_stages_for_training = use_all_stages_for_training
        self.unmasking_stages = unmasking_stages
        self.validation_stages = validation_stages
        
        self.tokenizer_path = os.path.join(self.data_dir, "tokenizer.json")
        self.tokenizer = self._load_or_train_tokenizer()
        
        # Initialize corruptor
        self._initialize_corruptor()
        
        # Validate/Init Stages
        self._validate_stage_config()
        self._initialize_stage_distribution()
        self._stage_cycle_state = {"train": [], "val": []}
        self._stage_mix_buffer = {"train": [], "val": []}
        
        self.configs = self.DEFAULT_CONFIGS

    def _load_or_train_tokenizer(self) -> Tokenizer:
        if os.path.exists(self.tokenizer_path):
            if self.verbose:
                print(f"Loading tokenizer from {self.tokenizer_path}")
            return Tokenizer.from_file(self.tokenizer_path)
        
        if self.verbose:
            print(f"Tokenizer not found at {self.tokenizer_path}. Training new tokenizer...")
        
        return self._train_tokenizer()

    def _train_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Ensure we have common special tokens
        special_tokens = ["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]"]
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )

        def iterator():
            count = 0
            for text in self._stream_from_configs(self.DEFAULT_CONFIGS, infinite=False):
                yield text
                count += 1
                if count >= self.tokenizer_train_samples:
                    break
        
        tokenizer.train_from_iterator(iterator(), trainer=trainer)
        tokenizer.save(self.tokenizer_path)
        print(f"Tokenizer trained and saved to {self.tokenizer_path} with vocab size {tokenizer.get_vocab_size()}")
        return tokenizer

    def _initialize_corruptor(self) -> None:
        # Get special token IDs
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have [MASK] token")
            
        # Identify excluded tokens (specials)
        excluded_ids = set()
        for token in ["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]"]:
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                excluded_ids.add(tid)
        
        candidate_ids = build_candidate_token_ids(
            self.tokenizer.get_vocab_size(), 
            excluded_token_ids=excluded_ids
        )
        
        self._corruptor = RandomReplacementCorruptor(
            candidate_ids,
            original_token_probability_multiplier=self._original_multiplier,
        )
        
        # Simple fragment sampler
        self._fragment_sampler = self._build_fragment_sampler()

    def _build_fragment_sampler(self):
         return lambda bs, rng: torch.full((bs, self.block_size), self.mask_token_id, dtype=torch.long)

    def _validate_stage_config(self):
        """Validate stage configuration."""
        if self.use_all_stages_for_training:
            if not self.unmasking_stages:
                raise ValueError("unmasking_stages must be provided when use_all_stages_for_training=True")
            if not self.validation_stages:
                raise ValueError("validation_stages must be provided when use_all_stages_for_training=True")

    def _initialize_stage_distribution(self):
        """Initialize stage distribution for batch generation."""
        if self.use_all_stages_for_training:
            self.train_stage_distribution = self._calculate_stage_distribution(self.unmasking_stages)
            self.val_stage_distribution = self._calculate_stage_distribution(self.validation_stages)
        else:
            self.train_stage_distribution = None
            self.val_stage_distribution = None

    def _calculate_stage_distribution(self, stages: List[Dict]) -> List[Dict]:
        total_stages = len(stages)
        samples_per_stage = self.batches_per_file // total_stages
        remainder = self.batches_per_file % total_stages
        
        distribution = []
        for i, stage in enumerate(stages):
            count = samples_per_stage + (1 if i < remainder else 0)
            distribution.append({
                'config': stage,
                'count': count
            })
        return distribution

    def _ensure_stage_cycle(self, split: str, rng) -> None:
        if not self.use_all_stages_for_training:
            return
        if self._stage_cycle_state[split]:
            return
        distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution
        
        stage_pool: List[Dict] = []
        for info in distribution:
            stage_pool.extend([info["config"]] * info["count"])

        perm = torch.randperm(len(stage_pool), generator=rng).tolist()
        shuffled = [stage_pool[i] for i in perm]
        self._stage_cycle_state[split] = shuffled

    def _stream_from_configs(self, config_names: Iterable[str], infinite: bool = False) -> Iterable[str]:
        while True:
            for config_name in config_names:
                if self.verbose:
                    print(f"Streaming from {config_name}...")
                try:
                    ds = datasets.load_dataset("HuggingFaceTB/cosmopedia", config_name, split="train", streaming=True)
                    for example in ds:
                        text = example.get('text', '')
                        if text:
                            yield text
                except Exception as e:
                    print(f"Error streaming {config_name}: {e}")
                    time.sleep(5)
            
            if not infinite:
                break

    def _get_infinite_stream(self):
        return self._stream_from_configs(self.configs, infinite=True)

    def _refill_stage_mix_buffer(self, split: str, rng) -> None:
        """Fetch enough data, apply various stage masks, and shuffle into a mixed buffer."""
        if self._stage_mix_buffer[split]:
            return

        if not hasattr(self, '_stream'):
            self._stream = self._get_infinite_stream()

        # 1. Determine which stages we need to satisfy the cycle
        self._ensure_stage_cycle(split, rng)
        stage_configs = []
        while self._stage_cycle_state[split]:
            stage_configs.append(self._stage_cycle_state[split].pop())
        
        # We need as many sequences as: len(stage_configs) * batch_size
        total_sequences_needed = len(stage_configs) * self.batch_size
        
        # 2. Fetch sequences
        sequences_x = []
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0

        while len(sequences_x) < total_sequences_needed:
            text = next(self._stream)
            ids = self.tokenizer.encode(text).ids
            if len(ids) > self.block_size:
                ids = ids[:self.block_size]
            
            row = torch.tensor(ids, dtype=torch.long)
            if len(ids) < self.block_size:
                needed = self.block_size - len(ids)
                padding = torch.full((needed,), pad_id, dtype=torch.long)
                row = torch.cat([row, padding])
            sequences_x.append(row)

        all_x = torch.stack(sequences_x) # [total_seqs, block_size]
        
        # 3. Apply masking per stage
        # We process in chunks of 'batch_size' for each stage config
        collected_mixed_batches = []
        
        ptr = 0
        for stage_config in stage_configs:
            batch_x_slice = all_x[ptr : ptr + self.batch_size]
            ptr += self.batch_size
            
            # Identify valid tokens (for stage masking to respect)
            # Actually apply_stage_masking doesn't take 'valid_mask' directly in signature 
            # but usually masking is agnostic or we ignore padding post-hoc.
            # Reference logic: masking_utils applies mask based on probabilities. 
            # We should ensure we don't mask padding? 
            # apply_stage_masking calls apply_bert_style_corruption_cpu which uses vocab_size.
            # Let's let it mask, but then we force padding back to PAD and ignore_index.
            
            # Apply stage masking -> returns corrupted_x (pre-corrupted/masked) AND mask boolean
            stage_corrupted_x, stage_mask = apply_stage_masking(
                batch_x_slice, stage_config, self.mask_token_id, self.tokenizer.get_vocab_size() - 1, rng
            )
            
            # Apply corruption (random replacement mixture) on the positions selected by stage_mask
            # NOTE: reference CharDiffusionProvider._apply_stage_corruption just returns stage_corrupted_x
            # BUT CharRandomReplacementProvider overrides it to use RandomReplacementCorruptor!
            # We must recreate that logic.
            
            # From reference:
            # _apply_stage_corruption(self, stage_corrupted_x, original_x, mask, rng)
            # return self._corruptor.corrupt(original_x, mask, rng) (if split != train, or via _apply_train_corruption)
            
            if split == 'train':
                 final_corrupted_x = apply_mixed_corruption(
                    batch_x_slice,
                    stage_mask,
                    rng,
                    random_corruptor=self._corruptor,
                    mask_token_id=self.mask_token_id,
                    fragment_sampler=self._fragment_sampler,
                    mixture_weights=self._train_corruption_mixture,
                )
            else:
                 final_corrupted_x = self._corruptor.corrupt(batch_x_slice, stage_mask, rng)

            # Create labels
            y = batch_x_slice.clone()
            
            # Enforce padding integrity (padding should not be masked or corrupted)
            is_padding = (batch_x_slice == pad_id)
            final_corrupted_x[is_padding] = pad_id # restore pads if corrupted
            
            if not self._dataset_partial_targets:
                 y[is_padding] = -100
                 # For full targets, we want to predict everything (except padding)
                 # Wait, Reference: `torch.where(mask, original_x, self.ignore_index)` for partial
                 # `original_x.clone()` for full.
                 pass
            else:
                 # Partial targets (only predict masked)
                 y = torch.where(stage_mask, batch_x_slice, torch.tensor(-100, dtype=torch.long))

            collected_mixed_batches.append({'x': final_corrupted_x, 'y': y})
            
        # 4. Shuffle the batches? 
        # Reference `_refill_stage_mix_buffer` calls `torch.randperm` on the stacked rows and re-batches.
        # This mixes the stages together in the buffer.
        
        stacked_x = torch.cat([b['x'] for b in collected_mixed_batches], dim=0)
        stacked_y = torch.cat([b['y'] for b in collected_mixed_batches], dim=0)
        
        perm = torch.randperm(stacked_x.shape[0], generator=rng)
        shuffled_x = stacked_x[perm]
        shuffled_y = stacked_y[perm]
        
        batched_x = torch.split(shuffled_x, self.batch_size)
        batched_y = torch.split(shuffled_y, self.batch_size)
        
        final_batches = []
        for bx, by in zip(batched_x, batched_y):
             if bx.shape[0] == self.batch_size:
                 final_batches.append({'x': bx, 'y': by})
                 
        self._stage_mix_buffer[split] = list(reversed(final_batches))

    def _sample_stage_based_batch(self, split: str, rng) -> Dict[str, Any]:
        self._refill_stage_mix_buffer(split, rng)
        return self._stage_mix_buffer[split].pop()

    def sample_batch(self, split: str, rng: torch.Generator) -> Dict[str, torch.Tensor]:
        if not hasattr(self, '_stream'):
            self._stream = self._get_infinite_stream()
            
        if self.use_all_stages_for_training:
             return self._sample_stage_based_batch(split, rng)
        
        # Fallback to simple logic (should not be reached if config is correct)
        raise NotImplementedError("Default non-stage batching not strictly implemented for full parity. Use stages.")

    def build_meta(self) -> Dict[str, Any]:
        return {
            "dataset_name": "cosmopedia",
            "training_type": "MLM",
            "vocab_size": self.vocab_size,
            "tokenizer_path": self.tokenizer_path,
            "corruption": {
                "type": "random_replacement",
                "original_token_probability_multiplier": self._original_multiplier,
            },
            "batch_schema": [
                {"name": "x", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "y", "dtype": "int64", "shape": [self.block_size], "role": "target"},
            ],
        }

# Explicit provider alias
Provider = CosmopediaProvider
