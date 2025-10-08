from __future__ import annotations

import os
from typing import Dict, Any, List

import torch
from core.common.utils import sequence_scorer_target_transform


from data.common.provider_base import DataProviderBase
from data.common.line_aligned_utils import LineAlignedSequenceBuilder, create_line_aligned_builder
from .mlm_inference import MLMInferenceEngine
from .synthetic_generation import (
    create_synthetic_text,
    create_stage_synthetic_text,
    add_cls_token,
    apply_stage_masking_direct,
    apply_line_masking_direct,
)


class SequenceScorerProvider(DataProviderBase):
    def __init__(self, *args, **kwargs) -> None:
        # Accept full config for dataset-specific parsing
        cfg = kwargs.pop('config', {}) or {}
        # Extract sequence scorer specific config
        self.mlm_checkpoint_path = cfg.get('mlm_checkpoint_path')
        # CLS token will be set from MLM vocab (should be at base_vocab_size + 2)
        self.cls_token_id = None  # Will be set in _load_text_data

        # Stage-based configuration (optional)
        self.use_all_stages_for_training = cfg.get('use_all_stages_for_training', None)
        self.unmasking_stages = cfg.get('unmasking_stages', None)
        self.validation_stages = cfg.get('validation_stages', None)

        # Simple masking configuration
        self.mask_probability_range = cfg.get('mask_probability_range', (0.1, 0.8))

        # Line-aligned sequences configuration (default True to match CharDiffusionProvider)
        self.enable_line_aligned_sequences = cfg.get('enable_line_aligned_sequences', True)

        super().__init__(*args, **kwargs)

        import time
        self._start_time = time.time()
        def _prefix():
            ms = int((time.time() - self._start_time) * 1000)
            return f"[{ms}ms] [sequence_scorer]"
        self._log_prefix = _prefix

        # Initialize MLM inference engine (prefer CUDA for unmasking if available)
        mlm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mlm_engine = MLMInferenceEngine(
            checkpoint_path=self.mlm_checkpoint_path,
            device=mlm_device,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f"{self._log_prefix()} Unmasking device: {mlm_device}")

        # Load text and vocab consistent with MLM model
        self._load_text_data()

        # Validate/initialize stage configuration
        self._validate_stage_config()
        self._initialize_stage_distribution()

        # Ensure meta reflects current config (e.g., updated CLS token and vocab size)
        self._ensure_meta_up_to_date()

        if self.verbose:
            print(f"{self._log_prefix()} SequenceScorerProvider initialized:")
            print(f"  vocab_size: {self.vocab_size}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  cls_token_id: {self.cls_token_id}")
            print(f"  enable_line_aligned_sequences: {self.enable_line_aligned_sequences}")
            if self.enable_line_aligned_sequences:
                print(f"  newline_token_id: {self.newline_token_id}")
                if hasattr(self, 'train_builder') and self.train_builder:
                    print(f"  train_lines: {len(self.train_builder.lines_ids)}")
                    print(f"  val_lines: {len(self.val_builder.lines_ids)}")
                    print(f"  train_valid_starts: {self.train_builder.valid_starts.numel()}")
                    print(f"  val_valid_starts: {self.val_builder.valid_starts.numel()}")


    def _validate_stage_config(self):
        if self.use_all_stages_for_training is not None:
            if not self.use_all_stages_for_training:
                raise ValueError("Unsupported config: use_all_stages_for_training=False. Set it to True or omit stage settings.")
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
            "pad_token_id": self.pad_token_id,
            "mask_token_id": self.mask_token_id,
            "stoi": self.stoi,
            "itos": self.itos,
            "batch_schema": [
                {"name": "input_ids", "dtype": "int64", "shape": [self.block_size], "role": "input"},
                {"name": "targets", "dtype": "float32", "shape": [], "role": "target"},
            ],
        }

    def _transform_ratio_to_target(self, x: torch.Tensor) -> torch.Tensor:
        """Map syntheticity ratio in [0,1] to target using shared non-linear transform."""
        return sequence_scorer_target_transform(x)

    def _load_text_data(self) -> None:
        # Load text data using shared method
        data = self._load_input_text()

        # Use vocabulary from MLM model meta to ensure compatibility
        self.stoi = dict(self.mlm_engine.stoi)
        self.itos = dict(self.mlm_engine.itos)

        # Get special token IDs from MLM vocab
        self.mask_token_id = int(self.mlm_engine.mask_token_id)

        # PAD should be in MLM vocab
        if '[PAD]' not in self.stoi:
            raise ValueError("MLM checkpoint missing [PAD] token - please regenerate char_diffusion data")
        self.pad_token_id = self.stoi['[PAD]']

        # CLS token: add if missing (backward compatibility with old checkpoints)
        if '[CLS]' not in self.stoi:
            # Old checkpoint without CLS - add it at the end
            mlm_vocab_size = int(self.mlm_engine.vocab_size)
            self.cls_token_id = mlm_vocab_size  # Add after all existing tokens
            self.stoi['[CLS]'] = self.cls_token_id
            self.itos[self.cls_token_id] = '[CLS]'
            self.vocab_size = mlm_vocab_size + 1  # Extended vocab

            if self.verbose:
                print(f"{self._log_prefix()} WARNING: MLM checkpoint missing [CLS] token")
                print(f"  Adding [CLS] at position {self.cls_token_id}")
                print(f"  Extended vocab_size: {self.vocab_size}")
        else:
            # New checkpoint with CLS already present
            self.cls_token_id = self.stoi['[CLS]']
            self.vocab_size = int(self.mlm_engine.vocab_size)

        if self.verbose:
            print(f"{self._log_prefix()} Vocabulary loaded from MLM:")
            print(f"  vocab_size: {self.vocab_size}")
            print(f"  mask_token_id: {self.mask_token_id}")
            print(f"  pad_token_id: {self.pad_token_id}")
            print(f"  cls_token_id: {self.cls_token_id}")

        if self.enable_line_aligned_sequences:
            # Use line-aligned sequence generation
            self.newline_token_id = self.stoi.get('\n', None)
            if self.newline_token_id is None:
                raise ValueError("enable_line_aligned_sequences=True requires '\\n' to be in the vocabulary")

            # Create line-aligned builders for train and validation data
            self.train_builder, self.val_builder, _ = create_line_aligned_builder(
                data,
                newline_token_id=self.newline_token_id,
                pad_token_id=self.pad_token_id,  # Use PAD tokens exactly like CharDiffusion
                stoi=self.stoi,
            )

            # Keep ids for compatibility with non-line-aligned code paths
            train_data, val_data = self._create_train_val_split(data)
            self.train_ids = [self.stoi.get(c, 0) for c in train_data]
            self.val_ids = [self.stoi.get(c, 0) for c in val_data]
        else:
            # Original approach: tokenize characters using stoi (fallback to 0 if missing)
            train_data, val_data = self._create_train_val_split(data)
            self.train_ids = [self.stoi.get(c, 0) for c in train_data]
            self.val_ids = [self.stoi.get(c, 0) for c in val_data]
            self.train_builder = None
            self.val_builder = None
            self.newline_token_id = None

    def sample_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        if self.use_all_stages_for_training:
            # In stage-based mode file generation is done in produce_one_file; fall back to default per-batch sampling here.
            return self._sample_default_batch(split, rng)
        return self._sample_default_batch(split, rng)

    def _sample_default_batch(self, split: str, rng) -> Dict[str, torch.Tensor]:
        if self.enable_line_aligned_sequences:
            # Use line-aligned sequence generation
            builder = self.train_builder if split == "train" else self.val_builder

            # Step 1: Build variable-length line-aligned sequences (without CLS)
            original_text, content_lengths = builder.build_variable_length_sequences(
                self.batch_size, self.block_size - 1, rng  # Leave space for CLS
            )

            # Apply padding to replace zeros with PAD tokens
            original_text = builder.apply_padding(original_text, content_lengths)

            # Step 2: Apply masking and synthetic generation to variable-length content
            min_ratio, max_ratio = self.mask_probability_range
            mask_ratios = torch.rand(self.batch_size, generator=rng) * (max_ratio - min_ratio) + min_ratio

            # Apply masking only to content portions
            synthetic_text = original_text.clone()
            actual_synth_list = []

            for b in range(self.batch_size):
                content_len = int(content_lengths[b].item())
                if content_len > 0:
                    # Extract content portion
                    content = original_text[b, :content_len].unsqueeze(0)

                    # Apply synthetic generation to content only
                    synth_content, synth_ratio = create_synthetic_text(
                        content,
                        mask_ratios[b:b+1],
                        self.mlm_engine,
                        self.mask_token_id,
                        rng,
                        sampling_temperature=1.0,
                        top_k=50,
                    )

                    # Update synthetic text with generated content
                    synthetic_text[b, :content_len] = synth_content.squeeze(0)
                    actual_synth_list.append(synth_ratio.item())
                else:
                    actual_synth_list.append(0.0)

            # Step 3: Add CLS token
            input_ids = add_cls_token(synthetic_text, self.cls_token_id, self.block_size, self.pad_token_id)
            # Apply non-linear transformation to targets
            raw_targets = torch.tensor(actual_synth_list, dtype=torch.float32)
            targets = sequence_scorer_target_transform(raw_targets)

            return {"input_ids": input_ids, "targets": targets, "masking_strategy": ["random"] * self.batch_size, "masking_ratio": [float(x) for x in raw_targets.tolist()]}
        else:
            # Original implementation
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
            input_ids = add_cls_token(synthetic_text, self.cls_token_id, self.block_size, self.pad_token_id)
            # Apply non-linear transformation to targets
            raw_targets = actual_synth.float().cpu()
            targets = sequence_scorer_target_transform(raw_targets)
            return {"input_ids": input_ids, "targets": targets, "masking_strategy": ["random"] * self.batch_size, "masking_ratio": [float(x) for x in raw_targets.tolist()]}

    def produce_one_file(self, split: str, seq: int) -> None:
        """Write unified array-of-batches format.

        Supports both default and stage-based generation.
        """
        if self.use_all_stages_for_training:
            return self._produce_stage_based_file(split, seq)
        # Default path (no stages): use base implementation which writes array-of-batches
        super().produce_one_file(split, seq)

    def _produce_stage_based_file(self, split: str, seq: int) -> None:
        import time
        rng = torch.Generator()
        per_seed = (self.seed * 1000003) ^ (hash(split) & 0xFFFFFFFF) ^ seq
        rng.manual_seed(per_seed)

        stage_distribution = self.train_stage_distribution if split == "train" else self.val_stage_distribution

        all_inputs: List[torch.Tensor] = []
        all_targets: List[float] = []
        all_stage_info: List[Dict[str, Any]] = []
        all_masking_ratio: List[float] = []

        import time
        total_stage_time = 0.0
        total_mask_time = 0.0
        for stage_info in stage_distribution:
            stage_config = stage_info['config']
            count = stage_info['count']
            if count == 0:
                continue

            if self.enable_line_aligned_sequences:
                # Use line-aligned sequence generation
                builder = self.train_builder if split == "train" else self.val_builder
                original_text, content_lengths = builder.build_variable_length_sequences(
                    count, self.block_size - 1, rng  # Leave space for CLS
                )

                # Apply padding to replace zeros with PAD tokens
                original_text = builder.apply_padding(original_text, content_lengths)

                # Ensure tensor is on CUDA if available (consistent with MLM engine device)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                original_text = original_text.to(device)
            else:
                # Original approach
                ids = self.train_ids if split == "train" else self.val_ids
                max_start_idx = len(ids) - (self.block_size - 1)
                ix = torch.randint(0, max_start_idx, (count,), generator=rng).tolist()
                original_sequences = [ids[i : i + (self.block_size - 1)] for i in ix]
                # Ensure tensor is on CUDA if available (consistent with MLM engine device)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                original_text = torch.tensor(original_sequences, dtype=torch.long, device=device)

            # Measure mask vs unmask separately
            t0 = time.time()

            # No config pollution - pass data directly to masking function

            # Handle line replacement differently - no MLM prediction needed
            if stage_config.get('type') == 'line':
                if self.enable_line_aligned_sequences:
                    # For line-aligned sequences with line replacement
                    content_only_text = original_text.clone()
                    pad_mask = (original_text == self.pad_token_id)
                    content_only_text[pad_mask] = 0

                    # Pass split data directly to masking function
                    builder = self.train_builder if split == "train" else self.val_builder
                    synthetic_text, mask = apply_line_masking_direct(
                        content_only_text, stage_config, builder.lines_ids, self.newline_token_id, self.pad_token_id, rng
                    )

                    # Restore PAD tokens in synthetic text
                    synthetic_text[pad_mask] = self.pad_token_id
                    mask[pad_mask] = False  # Don't count PAD tokens as replaced
                else:
                    # Line replacement requires line-aligned sequences
                    raise ValueError("Line masking requires enable_line_aligned_sequences=True")
            else:
                # Standard masking with MLM prediction
                if self.enable_line_aligned_sequences:
                    # For line-aligned sequences, we need to be careful about PAD tokens
                    # Only apply masking to content portions, not PAD tokens
                    content_only_text = original_text.clone()
                    # Replace PAD tokens with zeros temporarily for masking
                    pad_mask = (original_text == self.pad_token_id)
                    content_only_text[pad_mask] = 0

                    masked_input, mask = apply_stage_masking_direct(
                        content_only_text, stage_config, self.mask_token_id, self.mlm_engine.vocab_size - 1, rng
                    )

                    # Restore PAD tokens in masked input (they should not be masked)
                    masked_input[pad_mask] = self.pad_token_id
                    mask[pad_mask] = False  # Don't predict PAD tokens

                    # For MLM prediction, replace PAD tokens with zeros (MLM model doesn't know about PAD)
                    mlm_input = masked_input.clone()
                    mlm_input[pad_mask] = 0

                    synthetic_text = self.mlm_engine.predict_masked_tokens(
                        mlm_input, mask, temperature=1.0, top_k=50
                    )

                    # Restore PAD tokens in synthetic text
                    synthetic_text[pad_mask] = self.pad_token_id
                else:
                    masked_input, mask = apply_stage_masking_direct(
                        original_text, stage_config, self.mask_token_id, self.mlm_engine.vocab_size - 1, rng
                    )
                    synthetic_text = self.mlm_engine.predict_masked_tokens(
                        masked_input, mask, temperature=1.0, top_k=50
                    )

            t_mask_end = time.time()
            t1 = time.time()

            # per-sample syntheticity based on mask
            masked_counts = mask.sum(dim=1).to(torch.float32)

            # Compute per-sample masking ratio as (total length of replaced tokens) / seq_len
            # Note: PAD positions are already excluded from mask by setting mask[pad_mask] = False
            seq_len = mask.shape[1]
            raw_synth = (masked_counts / max(seq_len, 1)).cpu()

            # Apply non-linear transformation to targets
            actual_synth = sequence_scorer_target_transform(raw_synth)

            # Accumulate timings
            total_stage_time += (t1 - t0)
            total_mask_time += (t_mask_end - t0)

            # Append inputs and targets
            input_with_cls = add_cls_token(synthetic_text, self.cls_token_id, self.block_size, self.pad_token_id)
            all_inputs.append(input_with_cls)
            all_targets.append(actual_synth)
            # Masking ratio before target transform
            all_masking_ratio.extend([float(x) for x in raw_synth.tolist()])
            all_stage_info.extend([stage_config] * count)

        if all_inputs:
            combined_inputs = torch.cat(all_inputs, dim=0)
            combined_targets = torch.cat([t if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float32) for t in all_targets], dim=0).to(torch.float32)
            total_samples = combined_inputs.shape[0]
            shuffle_indices = torch.randperm(total_samples, generator=rng)
            inputs = combined_inputs[shuffle_indices]
            targets = combined_targets[shuffle_indices]
            stage_info = [all_stage_info[i] for i in shuffle_indices.tolist()]
            combined_masking_ratio = all_masking_ratio
            masking_ratio = [float(combined_masking_ratio[i]) for i in shuffle_indices.tolist()]

            # Optional: append ~10% zero-target extras for val, then reshuffle
            if split == "val":
                extra = int(inputs.shape[0] * 0.10)
                if extra > 0:
                    val_ids = self.val_ids
                    max_start_idx = len(val_ids) - (self.block_size - 1)
                    if max_start_idx > 0:
                        ix = torch.randint(0, max_start_idx, (extra,), generator=rng).tolist()
                        original_sequences = [val_ids[i : i + (self.block_size - 1)] for i in ix]
                        original_text = torch.tensor(original_sequences, dtype=torch.long)
                        input_ids_extra = add_cls_token(original_text, self.cls_token_id, self.block_size, self.pad_token_id)
                        raw_targets_extra = torch.zeros(extra, dtype=torch.float32)
                        targets_extra = self._transform_ratio_to_target(raw_targets_extra)
                        input_ids_extra = input_ids_extra.to(inputs.device)
                        targets_extra = targets_extra.to(targets.device)
                        inputs = torch.cat([inputs, input_ids_extra], dim=0)
                        targets = torch.cat([targets, targets_extra], dim=0)
                        stage_info.extend([{"extra_zero": True}] * extra)
                        masking_ratio.extend([0.0] * extra)
                # reshuffle after extras
                total = inputs.shape[0]
                perm = torch.randperm(total, generator=rng)
                inputs = inputs[perm]
                targets = targets[perm]
                stage_info = [stage_info[i] for i in perm.tolist()]
                masking_ratio = [float(masking_ratio[i]) for i in perm.tolist()]

            # Build batches_out
            batches_out = []
            total_needed = self.batches_per_file * self.batch_size
            # truncate or pad with last samples to exactly total_needed
            if inputs.shape[0] < total_needed:
                # simple wrap-around to fill
                reps = (total_needed + inputs.shape[0] - 1) // inputs.shape[0]
                inputs = inputs.repeat((reps, 1))[:total_needed]
                targets = targets.repeat(reps)[:total_needed]
                stage_info = (stage_info * reps)[:total_needed]
                masking_ratio = (masking_ratio * reps)[:total_needed]
            else:
                inputs = inputs[:total_needed]
                targets = targets[:total_needed]
                stage_info = stage_info[:total_needed]
                masking_ratio = masking_ratio[:total_needed]

            for i in range(self.batches_per_file):
                s = i * self.batch_size
                e = s + self.batch_size
                tens = {"input": inputs[s:e], "target": targets[s:e]}
                meta = {
                    "stage_info": stage_info[s:e],
                    "masking_ratio": masking_ratio[s:e],
                }
                batches_out.append({"tensors": tens, "metadata": meta})

            file_meta = {
                "batch_size": self.batch_size,
                "num_batches": self.batches_per_file,
                "file_idx": seq,
                "split": split,
                "produced_at": int(time.time() * 1000),
                "stage_distribution": stage_distribution,
            }

            d = self.train_dir if split == "train" else self.val_dir
            ts = file_meta["produced_at"]
            tmp_name = f".tmp-{ts}-{seq:06d}.pt"
            final_name = f"{ts}-{seq:06d}-{self.batches_per_file}.pt"
            tmp_path = os.path.join(d, tmp_name)
            final_path = os.path.join(d, final_name)
            torch.save({"batches": batches_out, "metadata": file_meta}, tmp_path)
            os.replace(tmp_path, final_path)
            if self.verbose:
                total_ms = total_stage_time * 1000.0
                avg_per_batch_ms = total_ms / max(self.batches_per_file, 1)
                mask_ms = total_mask_time * 1000.0
                unmask_ms = max(total_ms - mask_ms, 0.0)
                pct_unmask = (unmask_ms / max(total_ms, 1e-6)) * 100.0
                print(f"{self._log_prefix()} produced stage-based file: {final_path}")
                print(f"{self._log_prefix()} avg batch gen time: {avg_per_batch_ms:.2f} ms ({self.batches_per_file} batches), unmasking {pct_unmask:.1f}%")
        else:
            super().produce_one_file(split, seq)


# Alias for prepare.py discovery
Provider = SequenceScorerProvider

