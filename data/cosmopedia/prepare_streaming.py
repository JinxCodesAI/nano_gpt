"""Streaming provider for Cosmopedia dataset with on-the-fly tokenizer training and Discrete Diffusion."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional, Tuple, Sequence, List
import threading
import queue
from collections import defaultdict

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
    raise ImportError("Could not import corruption_utils or masking_utils")

class BufferedIterator:
    """
    Buffers items from an iterator using a background thread and a queue.
    This allows prefetching data (e.g. from network) while the main thread is processing.
    Stores chunks of items to reduce queue overhead.
    """
    def __init__(self, iterator, buffer_size=1000, chunk_size=100):
        self.iterator = iterator
        self.queue = queue.Queue(maxsize=max(1, buffer_size // chunk_size))
        self.chunk_size = chunk_size
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._fill_buffer, daemon=True)
        self.thread.start()

    def _fill_buffer(self):
        try:
            chunk = []
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                chunk.append(item)
                if len(chunk) >= self.chunk_size:
                    self.queue.put(chunk)
                    chunk = []
            if chunk:
                self.queue.put(chunk)
            self.queue.put(None) # Sentinel for success
        except Exception as e:
            self.queue.put(e) # Sentinel for error

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item # Returns a list (chunk)

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
        bpe_dropout: float = 0.0,
        min_token_count: int = -1,
        batch_items_per_sample: int = 1,
        buffer_size: int = 1000,
        buffer_chunk_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = int(vocab_size)
        self.tokenizer_train_samples = int(tokenizer_train_samples)
        self.bpe_dropout = float(bpe_dropout)
        self.min_token_count = int(min_token_count)
        self.batch_items_per_sample = int(batch_items_per_sample)
        self.buffer_size = int(buffer_size)
        self.buffer_chunk_size = int(buffer_chunk_size)

        if self.batch_size % self.batch_items_per_sample != 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be divisible by batch_items_per_sample ({self.batch_items_per_sample})")
        
        
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
        from collections import Counter
        import json

        # 1. Setup the Tokenizer & Trainer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.ByteLevel(add_prefix_space=False),
            pre_tokenizers.Digits(individual_digits=True)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        
        special_tokens = ["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]", "[DEL]", "[EOS]", "[BOS]"]
        # Add noise level tokens
        for i in range(1, 11):
            special_tokens.append(f"[NOISE{i}]")
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=100, # or 50
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )

        # 2. CACHE DATA: Stream samples once into memory
        print(f"Collecting {self.tokenizer_train_samples} samples for tokenizer training...")
        corpus_buffer = []
        count = 0
        
        # We assume _stream_from_configs yields chunks of strings
        stream_iter = self._stream_from_configs(self.DEFAULT_CONFIGS, infinite=False)
        
        for chunk in stream_iter:
            # We iterate through the chunk
            for text in chunk:
                corpus_buffer.append(text+"\n")
                count += 1
                if count >= self.tokenizer_train_samples:
                    break
            if count >= self.tokenizer_train_samples:
                break
        
        print(f"Collected {len(corpus_buffer)} samples. Starting BPE training...")

        # 3. TRAIN: Use the in-memory buffer
        tokenizer.train_from_iterator(corpus_buffer, trainer=trainer)
        
        # 4. COUNT: Use the same buffer to count frequencies
        # Now that tokenizer is trained, we can check what it learned
        print("Training complete. Counting token frequencies...")
        token_counts = Counter()
        
        # Batch encoding is much faster than looping
        # We encode in chunks to be safe with RAM if samples are huge
        chunk_size = 1000
        for i in range(0, len(corpus_buffer), chunk_size):
            batch = corpus_buffer[i : i + chunk_size]
            encodings = tokenizer.encode_batch(batch)
            for enc in encodings:
                token_counts.update(enc.ids)

        # 5. SAVE: Save both tokenizer and counts
        tokenizer.save(self.tokenizer_path)
        
        counts_path = self.tokenizer_path.replace(".json", "_counts.json")
        
        # Sort by count descending (largest at top, smallest at bottom)
        # Note: Python 3.7+ preserves insertion order.
        sorted_counts = dict(sorted(token_counts.items(), key=lambda item: item[1], reverse=True))
        
        with open(counts_path, "w") as f:
            # Convert int keys to string for valid JSON
            # We must iterate over the *sorted* dict to preserve order in the new string-keyed dict
            json.dump({str(k): v for k, v in sorted_counts.items()}, f, indent=2)

        print(f"Tokenizer saved to {self.tokenizer_path}")
        print(f"Token counts saved to {counts_path}")
        
        return tokenizer

    def _initialize_corruptor(self) -> None:
        # Get special token IDs
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]")
        
        if self.mask_token_id is None:
            raise ValueError("Tokenizer must have [MASK] token")
            
        # Ensure Noise Tokens exist (for backward compatibility with existing tokenizers)
        base_noise_token = "[NOISE1]"
        if self.tokenizer.token_to_id(base_noise_token) is None:
            print("Adding missing noise tokens to tokenizer...")
            new_tokens = [f"[NOISE{i}]" for i in range(1, 11)]
            self.tokenizer.add_special_tokens(new_tokens)
            self.tokenizer.save(self.tokenizer_path)
            # We also might need to update counts? No, counts are for vocab learning.
            # But the vocab size in meta might need updating if we are strict.
            # self.vocab_size usually fixed, but tokenizer.get_vocab_size() changes.
            print(f"Added {len(new_tokens)} noise tokens. New vocab size: {self.tokenizer.get_vocab_size()}")

        self.noise_token_ids = []
        for i in range(1, 11):
            tid = self.tokenizer.token_to_id(f"[NOISE{i}]")
            if tid is None:
                raise ValueError(f"Token [NOISE{i}] missing after addition attempt")
            self.noise_token_ids.append(tid)

        # Initialize rare tokens
        self._load_rare_tokens()
            
        # Identify excluded tokens (specials)
        excluded_ids = set()
        for token in ["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]", "[BOS]", "[EOS]"]:
            tid = self.tokenizer.token_to_id(token)
            if tid is not None:
                excluded_ids.add(tid)
        
        # Also exclude noise tokens from being candidates for random replacement
        for tid in self.noise_token_ids:
            excluded_ids.add(tid)
        
        candidate_ids = build_candidate_token_ids(
            self.tokenizer.get_vocab_size(), 
            excluded_token_ids=excluded_ids
        )
        
        # Also store BOS/EOS ids for later use in buffer refilling
        self.bos_token_id = self.tokenizer.token_to_id("[BOS]")
        self.eos_token_id = self.tokenizer.token_to_id("[EOS]")

        if self.bos_token_id is None or self.eos_token_id is None:
             # Fallback if they are not found (though they should be in special_tokens list during training)
             # But the user logic strictly requires them.
             print("Warning: [BOS] or [EOS] token not found in tokenizer. This might cause issues if they are expected.")
        
        self._corruptor = RandomReplacementCorruptor(
            candidate_ids,
            original_token_probability_multiplier=self._original_multiplier,
        )
        
        # Simple fragment sampler
        self._fragment_sampler = self._build_fragment_sampler()

    def _build_fragment_sampler(self):
         return lambda bs, rng: torch.full((bs, self.block_size), self.mask_token_id, dtype=torch.long)

    def _load_rare_tokens(self) -> None:
        """
        Loads rare tokens based on min_token_count if counts file exists.
        Precomputes a dense mapping tensor for vectorized masking.
        """
        self.rare_token_ids = set()
        self.rare_token_map = None # Tensor mapping: id -> unk_id or id
        
        if self.min_token_count < 0:
            return
            
        import json
        counts_path = self.tokenizer_path.replace(".json", "_counts.json")
        
        if not os.path.exists(counts_path):
            if self.verbose:
                print(f"Counts file not found at {counts_path}, skipping rare token masking.")
            return
            
        try:
            with open(counts_path, "r") as f:
                counts = json.load(f)
            
            # Counts keys are strings of token IDs
            count_hits = 0
            for tid_str, count in counts.items():
                if count < self.min_token_count:
                    tid = int(tid_str)
                    self.rare_token_ids.add(tid)
                    count_hits += 1
            
            if self.verbose:
                print(f"Identified {count_hits} rare tokens (count < {self.min_token_count}) to be masked as [UNK].")
            
            # Create vectorized map
            vocab_size = self.tokenizer.get_vocab_size()
            # Default: map x -> x
            self.rare_token_map = torch.arange(vocab_size, dtype=torch.long)
            # If tensor is on CPU, it's fine for now, we move to device usually later, 
            # but prepare_streaming seems to run on CPU mostly (data gen).
            
            # Map rare -> unk
            if self.unk_token_id is not None and self.rare_token_ids:
                rare_indices = torch.tensor(list(self.rare_token_ids), dtype=torch.long)
                self.rare_token_map[rare_indices] = self.unk_token_id
                
        except Exception as e:
            print(f"Error loading token counts for rare token masking: {e}")

    def _apply_rare_token_masking_vectorized(self, ids_tensor: torch.Tensor) -> torch.Tensor:
        """
        Replaces rare tokens with [UNK] token ID using vectorized lookup.
        ids_tensor: [batch, len] or [len]
        """
        if self.rare_token_map is None:
            return ids_tensor
            
        # Ensure map is on same device (likely CPU)
        if self.rare_token_map.device != ids_tensor.device:
            self.rare_token_map = self.rare_token_map.to(ids_tensor.device)
            
        # Clamp to avoid index error if ids > vocab_size (safety)
        max_id = self.rare_token_map.size(0) - 1
        safe_ids = torch.clamp(ids_tensor, max=max_id)
        
        return self.rare_token_map[safe_ids]


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

    def _fetch_dataset_info(self, config_names: Iterable[str]) -> Optional[List[int]]:
        """
        Attempts to fetch the number of examples for each config.
        Returns a list of counts if successful for ALL configs, otherwise None.
        """
        counts = []
        try:
            for config_name in config_names:
                if self.verbose:
                    print(f"Fetching info for {config_name}...")
                builder = datasets.load_dataset_builder("HuggingFaceTB/cosmopedia", config_name)
                # Builder info might be empty if not downloaded, but usually works for streaming
                if builder.info.splits and "train" in builder.info.splits:
                    dataset_count = builder.info.splits["train"].num_examples
                    if dataset_count is None or dataset_count <= 0:
                        print(f"Warning: Count for {config_name} is invalid/None")
                        return None
                    counts.append(dataset_count)
                else:
                    print(f"Warning: Could not find split info for {config_name}")
                    return None
            return counts
        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            return None

    def _stream_from_configs(self, config_names: Iterable[str], infinite: bool = False) -> Iterable[str]:
        # 1. Load all datasets
        loaded_datasets = []
        valid_configs = []
        for config_name in config_names:
            try:
                if self.verbose:
                    print(f"Loading dataset stream for verify: {config_name}")
                ds = datasets.load_dataset("HuggingFaceTB/cosmopedia", config_name, split="train", streaming=True)
                loaded_datasets.append(ds)
                valid_configs.append(config_name)
            except Exception as e:
                print(f"Error loading config {config_name}: {e}. Skipping.")
        
        if not loaded_datasets:
            print("No datasets loaded successfully.")
            return

        # 2. Determine probabilities
        counts = self._fetch_dataset_info(valid_configs)
        probabilities = None
        
        if counts:
            total = sum(counts)
            if total > 0:
                probabilities = [c / total for c in counts]
                print(f"Using proportional sampling: {dict(zip(valid_configs, probabilities))}")
            else:
                print("Total count is 0, falling back to equal interleaving.")
        else:
            print("Could not fetch all dataset sizes, falling back to equal interleaving (Round-Robin).")

        # 3. Interleave
        # stopping_strategy="all_exhausted" ensures we use all data (oversampling smaller ones if needed to match probabilities, 
        # or just cycling if probabilities are None - actually with None it cycles A,B,C until all exhausted)
        interleaved_ds = datasets.interleave_datasets(
            loaded_datasets, 
            probabilities=probabilities, 
            seed=self.seed,
            stopping_strategy="all_exhausted"
        )
        
        # 4. Stream with logging
        self.read_counts = defaultdict(int) 
        # Needs to track which config yielded the item. 
        # interleave_datasets doesn't natively yield the source.
        # BUT we can wrap the original datasets to inject their source info if really needed, 
        # or just track total. The user asked for "how many items has been read so far from each set".
        # To do that, we need to map the yielded item back to its source, or wrap the sources.
        
        # Let's wrap the datasets to identify them.
        def identify_source(example, source_name):
            example['__source_config__'] = source_name
            return example

        # Re-create loaded datasets with mapping
        # Note: streaming datasets map returns a new iterable
        labeled_datasets = []
        for ds, name in zip(loaded_datasets, valid_configs):
             # Fix lambda capture by binding name to n
             labeled_datasets.append(ds.map(lambda x, n=name: identify_source(x, n)))
             
        # Re-interleave labeled datasets
        interleaved_ds = datasets.interleave_datasets(
            labeled_datasets, 
            probabilities=probabilities, 
            seed=self.seed,
            stopping_strategy="all_exhausted"
        )

        # Iterator for interleaved dataset
        iterator = iter(interleaved_ds)

        total_read = 0
        LOG_INTERVAL = 1000

        if self.buffer_size > 0:
            if self.verbose:
                 print(f"Buffering stream with size {self.buffer_size}, chunk_size={self.buffer_chunk_size}...")
            # We use an internal variable to access the iterator directly in refill if we want structure,
            # but _stream_from_configs is a generator.
            buffered = BufferedIterator(iterator, buffer_size=self.buffer_size, chunk_size=self.buffer_chunk_size)
            # We yield chunks from 'buffered'
            for chunk in buffered:
                # Update stats
                for ex in chunk:
                    source = ex.get('__source_config__', 'unknown')
                    self.read_counts[source] += 1
                    total_read += 1
                
                if total_read % LOG_INTERVAL < len(chunk): # Approx logging
                     print(f"Read stats: {dict(self.read_counts)}")
                
                # The original filtered for `text`.
                yield [ex.get('text', '') for ex in chunk if ex.get('text', '')]
                
        else:
            # Fallback for no buffer (or tokenizer training if buffer_size=0?)
            # Just yield chunks of 1 or similar to keep API consistent?
            # Or keep original behavior?
            # Best to harmonize: Always yield lists of strings.
            buffer_acc = []
            for example in iterator:
                 source = example.get('__source_config__', 'unknown')
                 self.read_counts[source] += 1
                 total_read += 1
                 
                 if total_read % LOG_INTERVAL == 0:
                     print(f"Read stats: {dict(self.read_counts)}")
                 
                 text = example.get('text', '')
                 if text:
                     buffer_acc.append(text)
                 
                 if len(buffer_acc) >= self.buffer_chunk_size:
                     yield buffer_acc
                     buffer_acc = []
            if buffer_acc:
                yield buffer_acc
    
    # helper for infinite loop wrapper
    def _stream_infinite_wrapper(self, config_names):
        while True:
            yield from self._stream_from_configs(config_names, infinite=False)


    def _get_infinite_stream(self):
        return self._stream_infinite_wrapper(self.configs)

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
        bos_id = self.bos_token_id if self.bos_token_id is not None else self.tokenizer.token_to_id("[BOS]")
        eos_id = self.eos_token_id if self.eos_token_id is not None else self.tokenizer.token_to_id("[EOS]")
        
        # Effective max length for content is block_size - 3 (for BOS, NOISE, and EOS)
        # If block_size is small, this might be tight.
        max_content_len = self.block_size - 3
        if max_content_len < 1:
            raise ValueError(f"Block size {self.block_size} is too small to hold [BOS], [NOISE], content, and [EOS].")

        # Determine how many UNIQUE sequences we need to fetch from the stream
        if total_sequences_needed % self.batch_items_per_sample != 0:
             # Should be guaranteed by init check, but good for safety
             raise ValueError("total_sequences_needed not divisible by batch_items_per_sample")
        
        unique_needed = total_sequences_needed // self.batch_items_per_sample
        unique_needed = total_sequences_needed // self.batch_items_per_sample
        unique_sequences_x = []

        t_wait = 0.0
        t_tokenize = 0.0
        t_prepare = 0.0
        
        t0_loop = time.perf_counter()

        while len(unique_sequences_x) < unique_needed:
            t0 = time.perf_counter()
            # stream yields chunks now
            try:
                text_chunk = next(self._stream)
            except StopIteration:
                 # Should not happen in infinite loop wrapper, but if it does
                 break
            t1 = time.perf_counter()
            t_wait += (t1 - t0)
            
            # Apply BPE Dropout if applicable
            if hasattr(self.tokenizer.model, 'dropout'):
                self.tokenizer.model.dropout = self.bpe_dropout if split == 'train' else 0.0
            
            # Batch encode
            # encode_batch returns List[Encoding]
            # We assume text_chunk is List[str]
            encodings = self.tokenizer.encode_batch(text_chunk)
            
            t2 = time.perf_counter()
            t_tokenize += (t2 - t1)
            
            # Process each encoding
            for enc in encodings:
                ids = torch.tensor(enc.ids, dtype=torch.long)
                
                # Vectorized Rare Token Masking
                if self.min_token_count > -1:
                    ids = self._apply_rare_token_masking_vectorized(ids)
                
                # Truncate content if needed
                if ids.size(0) > max_content_len:
                    ids = ids[:max_content_len]
                
                # Construct sequence tensor
                # We need to assemble [BOS, NOISE, ...ids..., EOS, PAD...]
                # Pre-allocate buffer for speed? 
                
                row = torch.full((self.block_size,), pad_id, dtype=torch.long)
                curr_pos = 0
                if bos_id is not None:
                    row[curr_pos] = bos_id
                    curr_pos += 1
                    
                # Noise placeholder
                row[curr_pos] = pad_id
                curr_pos += 1
                
                # Content
                l = ids.size(0)
                row[curr_pos : curr_pos + l] = ids
                curr_pos += l
                
                if eos_id is not None:
                    row[curr_pos] = eos_id
                    # curr_pos += 1
                
                # Padding is already filled by torch.full
                
                unique_sequences_x.append(row)
                if len(unique_sequences_x) >= unique_needed:
                    break
            
            t3 = time.perf_counter()
            t_prepare += (t3 - t2)

        # Expand unique sequences: reuse each one batch_items_per_sample times
        # We want distinct corruptions for each copy, so we just duplicate the CLEAN inputs here.
        # The corruption loop later (step 3) processes each item in all_x independently with RNG,
        # so they will get different masks/replacements naturally.
        
        sequences_x = []
        for seq in unique_sequences_x:
            for _ in range(self.batch_items_per_sample):
                 sequences_x.append(seq)
                 
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
            # Enforce integrity of special tokens (PAD, BOS, EOS)
            # They should NEVER be masked or corrupted.
            # 1. Update mask to exclude them (so we don't try to predict them if we were using partial targets based on mask, 
            #    though for partial targets y is based on stage_mask)
            # 2. Restore them in final_corrupted_x
            
            protected_mask = (batch_x_slice == pad_id)
            if bos_id is not None:
                protected_mask |= (batch_x_slice == bos_id)
            if eos_id is not None:
                protected_mask |= (batch_x_slice == eos_id)
            
            # Ensure stage_mask does not include protected tokens
            # (In case random masking selected them)
            stage_mask = stage_mask & (~protected_mask)
            
            # Restore protected tokens in the corrupted input
            # This undoes any corruption that might have happened to them
            final_corrupted_x[protected_mask] = batch_x_slice[protected_mask]
            
            if not self._dataset_partial_targets:
                 is_padding = (batch_x_slice == pad_id)
                 y[is_padding] = -100
                 # For full targets, we want to predict everything (except padding)
                 # Wait, Reference: `torch.where(mask, original_x, self.ignore_index)` for partial
                 # `original_x.clone()` for full.
                 pass
            else:
                 # Partial targets (only predict masked)
                 y = torch.where(stage_mask, batch_x_slice, torch.tensor(-100, dtype=torch.long))

            # --- Inject Noise Token Logic ---
            # Calculate ratio of mismatch: input (final_corrupted_x) vs target/original (batch_x_slice)
            # Placeholder is at index 1 (if BOS) or 0 (if no BOS).
            # Both final_corrupted_x and batch_x_slice have PAD at the placeholder position (since it was protected).
            # So diffs are purely from corruption.
            
            diff_mask = (final_corrupted_x != batch_x_slice)
            diff_counts = diff_mask.sum(dim=1).float()
            seq_len = float(final_corrupted_x.shape[1])
            ratios = diff_counts / seq_len
            
            # Formula: 1-10% -> 1, ..., 91-100% -> 10.
            # ceil(ratio * 10)
            noise_indices = torch.ceil(ratios * 10).long()
            # Map 0 to NOISE1, and ensure range 1-10
            noise_indices = torch.clamp(noise_indices, 1, 10) 
            
            # Convert computed indices to token IDs
            # noise_token_ids[0] is [NOISE1]
            if hasattr(self, 'noise_token_ids') and self.noise_token_ids:
                noise_ids_tensor = torch.tensor(self.noise_token_ids, dtype=torch.long, device=final_corrupted_x.device)
                
                # noise_indices is now always >= 1, so we always have a token to insert
                # Subtract 1 because noise_indices 1 maps to noise_token_ids[0]
                tokens_to_insert = noise_ids_tensor[noise_indices - 1]
                
                target_idx = 1 if bos_id is not None else 0
                final_corrupted_x[:, target_idx] = tokens_to_insert
                # Ensure y ignores this position
                y[:, target_idx] = -100

                y[:, target_idx] = -100

            collected_mixed_batches.append({'x': final_corrupted_x, 'y': y})
            
        t_corrupt = time.perf_counter() - t0_loop - t_wait - t_tokenize - t_prepare
        
        # Log aggregated stats
        if self.verbose:
             total_ms = (time.perf_counter() - t0_loop) * 1000
             print(f"[profiler] refill {unique_needed} items: total={total_ms:.1f}ms | "
                   f"wait={t_wait*1000:.1f}ms ({(t_wait*1000)/unique_needed:.2f}ms/it), "
                   f"tok={t_tokenize*1000:.1f}ms, "
                   f"prep={t_prepare*1000:.1f}ms, "
                   f"corrupt={t_corrupt*1000:.1f}ms")
            
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
        stoi = self.tokenizer.get_vocab()
        itos = {v: k for k, v in stoi.items()}
        return {
            "dataset_name": "cosmopedia",
            "training_type": "MLM",
            "vocab_size": self.tokenizer.get_vocab_size(),
            "tokenizer_path": self.tokenizer_path,
            "stoi": stoi,
            "itos": itos,
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
