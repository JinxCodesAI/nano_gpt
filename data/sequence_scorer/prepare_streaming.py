from __future__ import annotations

import os
from typing import Dict, Any, List

import torch

from data.common.provider_base import DataProviderBase
from .mlm_inference import MLMInferenceEngine
from .synthetic_generation import (
    create_synthetic_text,
    create_stage_synthetic_text,
    add_cls_token,
)


class SequenceScorerProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        # Extract sequence scorer specific config
        self.mlm_checkpoint_path = kwargs.pop('mlm_checkpoint_path')
        self.cls_token_id = kwargs.pop('cls_token_id', 0)

        # Stage-based configuration (optional)
        self.use_all_stages_for_training = kwargs.pop('use_all_stages_for_training', None)
        self.unmasking_stages = kwargs.pop('unmasking_stages', None)
        self.validation_stages = kwargs.pop('validation_stages', None)

        # Simple masking configuration
        self.mask_probability_range = kwargs.pop('mask_probability_range', (0.1, 0.8))

        # Ignore diffusion-specific kwargs that prepare.py may pass
        kwargs.pop('mask_probability', None)
        kwargs.pop('mask_token_id', None)
        kwargs.pop('ignore_index', None)

        super().__init__(*args, **kwargs)

        # Initialize MLM inference engine (CPU for data generation)
        self.mlm_engine = MLMInferenceEngine(
            checkpoint_path=self.mlm_checkpoint_path,
            device='cpu',
            verbose=self.verbose,
        )

        # Load text and vocab consistent with MLM model
        self._load_text_data()

        # Validate/initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()

        if self.verbose:
            print("SequenceScorerProvider initialized:")
            print(f"  vocab_size: {self.vocab_size}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  cls_token_id: {self.cls_token_id}")

    def _validate_stage_config(self):
        if self.use_all_stages_for_training is not None:
            if not self.use_all_stages_for_training:
                raise NotImplementedError("use_all_stages_for_training=False is not yet implemented")
            if not self.unmasking_stages:
                raise ValueError("unmasking_stages must be provided when use_all_stages_for_training=True")
            if not self.validation_stages:
                raise ValueError("validation_stages must be provided when use_all_stages_for_training=True")

    def _initialize_stage_distribution(self):
        if self.use_all_stages_for_training:
            self.train_stage_distribution = self._calculate_stage_distribution(self.unmasking_stages)
            self.val_stage_distribution = self._calculate_stage_distribution(self.validation_stages)
        else:
            self.train_stage_distribution = None
            self.val_stage_distribution = None

    def _calculate_stage_distribution(self, stages: List[Dict]) -> List[Dict]:
        total_stages = len(stages)
        total_samples = self.batches_per_file * self.batch_size
        samples_per_stage = total_samples // total_stages
        remainder = total_samples % total_stages
        distribution = []
        for i, stage in enumerate(stages):
            count = samples_per_stage + (1 if i < remainder else 0)
            if count > 0:
                distribution.append({'config': stage, 'count': count})
        return distribution

    def build_meta(self) -> Dict:
        if self.block_size is None:
            raise ValueError("block_size must be set for SequenceScorerProvider")
        return {
            "dataset_name": "sequence_scorer",
            "training_type": "SEQUENCE_SCORING",
            "vocab_size": self.vocab_size,
            "cls_token_id": self.cls_token_id,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "input_ids", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "targets", "dtype": "float32", "shape": [], "role": "target"},
            ],
        }

    def _load_text_data(self) -> None:
        input_file_path = os.path.join(self.data_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input text file not found: {input_file_path}")
        with open(input_file_path, 'r') as f:
            data = f.read()

        # Use vocabulary from MLM model meta to ensure compatibility
        self.stoi = dict(self.mlm_engine.stoi)
        self.itos = dict(self.mlm_engine.itos)
        self.vocab_size = int(self.mlm_engine.vocab_size)
        self.mask_token_id = int(self.mlm_engine.mask_token_id)

        # Tokenize characters using stoi (fallback to 0 if missing)
        self.train_ids = [self.stoi.get(c, 0) for c in data[: int(len(data) * 0.9)]]
        self.val_ids = [self.stoi.get(c, 0) for c in data[int(len(data) * 0.9) :]]

        # Ensure [CLS] token exists in vocab mapping
        if self.cls_token_id not in self.itos:
            self.itos[self.cls_token_id] = '[CLS]'
        if '[CLS]' not in self.stoi:
            self.stoi['[CLS]'] = self.cls_token_id

    def sample_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        if self.use_all_stages_for_training:
            raise NotImplementedError("Stage-based sampling requires file-level generation")
        return self._sample_default_batch(split, rng)

    def _sample_default_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - (self.block_size - 1)  # leave space for [CLS]
        ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()

        # Build batch of original sequences (without CLS)
        original_sequences = [ids[i : i + (self.block_size - 1)] for i in ix]
        original_text = torch.tensor(original_sequences, dtype=torch.long)

        # Random mask ratio per sample
        min_ratio, max_ratio = self.mask_probability_range
        mask_ratios = torch.rand(self.batch_size, generator=rng) * (max_ratio - min_ratio) + min_ratio

        batch_inputs = []
        batch_targets = []
        for i in range(self.batch_size):
            synthetic_text, actual_synth = create_synthetic_text(
                original_text[i:i+1],
                mask_ratios[i].item(),
                self.mlm_engine,
                self.mask_token_id,
                rng,
                sampling_temperature=1.0,
                top_k=50,
            )
            input_with_cls = add_cls_token(synthetic_text, self.cls_token_id, self.block_size)
            batch_inputs.append(input_with_cls)
            batch_targets.append(actual_synth)

        input_ids = torch.cat(batch_inputs, dim=0)
        targets = torch.tensor(batch_targets, dtype=torch.float32)
        return {"input_ids": input_ids, "targets": targets}

    def _produce_stage_based_file(self, split: str, seq: int) -> None:
        import time
        rng = torch.Generator()
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - (self.block_size - 1)
        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution

        all_inputs: List[torch.Tensor] = []
        all_targets: List[float] = []
        all_stage_info: List[Dict[str, Any]] = []

        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            if count == 0:
                continue
            ix = torch.randint(0, max_start_idx, (count,), generator=rng).tolist()
            original_sequences = [ids[i : i + (self.block_size - 1)] for i in ix]
            original_text = torch.tensor(original_sequences, dtype=torch.long)

            synthetic_text, actual_synth = create_stage_synthetic_text(
                original_text,
                stage_config,
                self.mlm_engine,
                self.mask_token_id,
                rng,
                sampling_temperature=1.0,
                top_k=50,
            )

            for i in range(count):
                input_with_cls = add_cls_token(synthetic_text[i:i+1], self.cls_token_id, self.block_size)
                all_inputs.append(input_with_cls)
            all_targets.extend([actual_synth] * count)
            all_stage_info.extend([stage_config] * count)

        if all_inputs:
            combined_inputs = torch.cat(all_inputs, dim=0)
            combined_targets = torch.tensor(all_targets, dtype=torch.float32)
            total_samples = combined_inputs.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)
            shuffled_inputs = combined_inputs[shuffle_indices]
            shuffled_targets = combined_targets[shuffle_indices]
            shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]

            tensors = {"input_ids": shuffled_inputs, "targets": shuffled_targets}
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
                print(f"[sequence_scorer] produced stage-based file: {final_path}")
        else:
            super().produce_one_file(split, seq)


# Alias for prepare.py discovery
Provider = SequenceScorerProvider

