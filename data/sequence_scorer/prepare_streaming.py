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
    apply_stage_masking_direct,
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

        # Initialize MLM inference engine (prefer CUDA for unmasking if available)
        mlm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlm_engine = MLMInferenceEngine(
            checkpoint_path=self.mlm_checkpoint_path,
            device=mlm_device,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f"[sequence_scorer] Unmasking device: {mlm_device}")

        # Load text and vocab consistent with MLM model
        self._load_text_data()

        # Validate/initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()

        # Ensure meta reflects current config (e.g., updated CLS token and vocab size)
        self._ensure_meta_up_to_date()

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

    def _ensure_meta_up_to_date(self) -> None:
        desired = self.build_meta()
        try:
            import pickle
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "rb") as f:
                    existing = pickle.load(f)
                if existing.get("vocab_size") != desired.get("vocab_size") or existing.get("cls_token_id") != desired.get("cls_token_id"):
                    if self.verbose:
                        print("[sequence_scorer] updating meta.pkl to reflect vocab/CLS changes")
                    self.write_meta(desired)
            else:
                self.write_meta(desired)
        except Exception as e:
            # Fail loudly to avoid silent mismatches
            raise

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

        # Ensure [CLS] token exists in vocab mapping and extend vocab_size if needed
        if self.cls_token_id not in self.itos:
            self.itos[self.cls_token_id] = '[CLS]'
        if '[CLS]' not in self.stoi:
            self.stoi['[CLS]'] = self.cls_token_id
        if self.cls_token_id >= self.vocab_size:
            self.vocab_size = int(self.cls_token_id) + 1

    def sample_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        if self.use_all_stages_for_training:
            raise NotImplementedError("Stage-based sampling requires file-level generation")
        return self._sample_default_batch(split, rng)

    def produce_one_file(self, split: str, seq: int) -> None:
        """Override to handle stage-based generation at file level."""
        if self.use_all_stages_for_training:
            self._produce_stage_based_file(split, seq)
        else:
            super().produce_one_file(split, seq)

    def _sample_default_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        ids = self.train_ids if split == "train" else self.val_ids
        max_start_idx = len(ids) - (self.block_size - 1)  # leave space for [CLS]
        ix = torch.randint(0, max_start_idx, (self.batch_size,), generator=rng).tolist()

        # Build batch of original sequences (without CLS)
        original_sequences = [ids[i : i + (self.block_size - 1)] for i in ix]
        original_text = torch.tensor(original_sequences, dtype=torch.long)

        # Random mask ratio per sample (vectorized synthetic generation)
        min_ratio, max_ratio = self.mask_probability_range
        mask_ratios = torch.rand(self.batch_size, generator=rng) * (max_ratio - min_ratio) + min_ratio

        # Call into batched synthetic generation once
        synthetic_text, actual_synth = create_synthetic_text(
            original_text,
            mask_ratios,
            self.mlm_engine,
            self.mask_token_id,
            rng,
            sampling_temperature=1.0,
            top_k=50,
        )
        input_ids = add_cls_token(synthetic_text, self.cls_token_id, self.block_size)
        targets = actual_synth.float().cpu()
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

        import time
        total_stage_time = 0.0
        total_mask_time = 0.0
        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            if count == 0:
                continue
            ix = torch.randint(0, max_start_idx, (count,), generator=rng).tolist()
            original_sequences = [ids[i : i + (self.block_size - 1)] for i in ix]
            original_text = torch.tensor(original_sequences, dtype=torch.long)

            # Measure mask vs unmask separately
            t0 = time.time()
            masked_input, mask = apply_stage_masking_direct(
                original_text, stage_config, self.mask_token_id, self.vocab_size - 1, rng
            )
            t_mask_end = time.time()
            synthetic_text = self.mlm_engine.predict_masked_tokens(
                masked_input, mask, temperature=1.0, top_k=50
            )
            t1 = time.time()

            # per-sample syntheticity based on mask
            masked_counts = mask.sum(dim=1).to(torch.float32)
            actual_synth = (masked_counts / max(mask.shape[1], 1)).cpu()

            # Accumulate timings
            total_stage_time += (t1 - t0)
            total_mask_time += (t_mask_end - t0)

            # Append inputs and targets
            input_with_cls = add_cls_token(synthetic_text, self.cls_token_id, self.block_size)
            all_inputs.append(input_with_cls)
            all_targets.append(actual_synth)
            all_stage_info.extend([stage_config] * count)

        if all_inputs:
            combined_inputs = torch.cat(all_inputs, dim=0)
            # all_targets may be list of tensors; concatenate then cast
            combined_targets = torch.cat([t if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float32) for t in all_targets], dim=0).to(torch.float32)
            total_samples = combined_inputs.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)
            shuffled_inputs = combined_inputs[shuffle_indices]
            shuffled_targets = combined_targets[shuffle_indices]
            shuffled_stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]

            tensors = {"input_ids": shuffled_inputs, "targets": shuffled_targets}

            # Print per-file aggregate timing right after we finalize the tensors
            if self.verbose:
                total_ms = total_stage_time * 1000.0
                mask_ms = total_mask_time * 1000.0
                pct_remask = (mask_ms / max(total_ms, 1e-6)) * 100.0
                print(f"[sequence_scorer] batch gen avg per-file: {total_ms:.2f} ms, remasking {pct_remask:.1f}%")

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

