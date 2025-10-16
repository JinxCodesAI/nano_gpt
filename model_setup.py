from __future__ import annotations

import os
import pickle
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple

import torch
import tiktoken

from model import GPT, GPTConfig


class ModelSetup:
    """Load model and tokenizer resources shared across sampling strategies."""

    def __init__(
        self,
        *,
        init_from: str,
        out_dir: str,
        ckpt_name: str,
        device: str,
        dtype: str,
        compile_model: bool,
        start: str,
    ) -> None:
        self.init_from = init_from
        self.out_dir = out_dir
        self.ckpt_name = ckpt_name
        self.device = device
        self.dtype = dtype
        self.compile_model = compile_model
        self.start = start

        model, checkpoint = self._load_model()
        encode, decode, space_token_id = self._load_tokenizer(checkpoint)

        self.model = model
        self.encode = encode
        self.decode = decode
        self.space_token_id = space_token_id
        self.prompt = self._build_prompt(start_text=self._resolve_start_text())
        self.initial_length = self.prompt.size(0)
        self.block_size = self.model.config.block_size
        if self.initial_length > self.block_size:
            raise ValueError(
                f"Prompt is longer ({self.initial_length}) than model block size ({self.block_size})."
            )

        self.device_type = "cuda" if "cuda" in self.device else "cpu"
        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]

    def autocast_context(self):
        """Return the appropriate context manager for mixed precision."""
        if self.device_type == "cpu":
            return nullcontext()
        return torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

    def _load_model(self) -> Tuple[GPT, Optional[dict]]:
        if self.init_from == "resume":
            ckpt_path = os.path.join(self.out_dir, self.ckpt_name)
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint["model_args"])
            model = GPT(gptconf)
            state_dict = checkpoint["model"]
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
        elif self.init_from.startswith("gpt2"):
            model = GPT.from_pretrained(self.init_from, dict(dropout=0.0))
            checkpoint = None
        else:
            raise ValueError(f"Unknown init_from: {self.init_from}")

        model.eval()
        model.to(self.device)
        if self.compile_model:
            model = torch.compile(model)
        return model, checkpoint

    def _load_tokenizer(
        self, checkpoint: Optional[dict]
    ) -> Tuple[Callable[[str], Sequence[int]], Callable[[Sequence[int]], str], int]:
        load_meta = False
        meta_path: Optional[str] = None
        if (
            self.init_from == "resume"
            and checkpoint is not None
            and "config" in checkpoint
            and "dataset" in checkpoint["config"]
        ):
            meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
            load_meta = os.path.exists(meta_path)

        if load_meta and meta_path:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            stoi = meta["stoi"]
            itos = meta["itos"]
            if " " not in stoi:
                raise ValueError(
                    "Space character not found in dataset vocabulary; cannot perform space-based re-noising."
                )
            space_token_id = stoi[" "]
            encode_unknown_cache: set[str] = set()

            def encode(text: str) -> Sequence[int]:
                missing = set()
                ids = []
                for ch in text:
                    idx = stoi.get(ch)
                    if idx is None:
                        missing.add(ch)
                        idx = space_token_id
                    ids.append(idx)
                if missing:
                    unseen = missing.difference(encode_unknown_cache)
                    if unseen:
                        printable = ", ".join(repr(ch) for ch in sorted(unseen))
                        print(
                            f"Warning: substituting space for unknown characters: {printable}"
                        )
                        encode_unknown_cache.update(unseen)
                return ids

            decode = lambda token_ids: "".join(itos[i] for i in token_ids)
            return encode, decode, space_token_id

        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda text: enc.encode(text, allowed_special={"<|endoftext|>"})
        decode = lambda token_ids: enc.decode(token_ids)
        space_token_ids = encode(" ")
        if not space_token_ids:
            raise ValueError("Encoder did not return a token id for a single space character.")
        if len(space_token_ids) != 1:
            raise ValueError(f"Expected a single token id for space, got: {space_token_ids}")
        space_token_id = space_token_ids[0]
        return encode, decode, space_token_id

    def _resolve_start_text(self) -> str:
        if self.start.startswith("FILE:"):
            file_path = self.start[5:]
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return self.start

    def _build_prompt(self, start_text: str) -> torch.Tensor:
        start_ids = self.encode(start_text)
        prompt = torch.tensor(start_ids, dtype=torch.long, device=self.device)
        return prompt


__all__ = ["ModelSetup"]

