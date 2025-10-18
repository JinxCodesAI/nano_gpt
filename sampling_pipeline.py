"""Composable diffusion sampling pipeline with pluggable input transformations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

import torch

from sampling_utils import (
    apply_deletions,
    apply_insertions,
    apply_re_noise,
    build_cooldown_mask,
    compute_noise_ratio,
    deletion_scores_from_probs,
    score_gaps_for_insertion,
    select_topk_mask,
    uncertainty_from_logprobs,
)

# --------------------------------------------------------------------------- #
# Shared state containers


@dataclass
class DiffusionConfig:
    """Static configuration shared across pipeline iterations."""

    block_size: int
    initial_length: int
    max_iterations: int
    max_new_tokens: int
    prompt: torch.Tensor
    space_token_id: int
    temperature: float
    device: torch.device | str
    fix_prompt_during_diffusion: bool = True


@dataclass
class DiffusionState:
    """Mutable state exposed to transformations during sampling."""

    config: DiffusionConfig
    sample_index: int
    iteration: int
    x: torch.Tensor
    max_token_pos: int
    last_log_probs: torch.Tensor
    last_insert_indices: List[int] = field(default_factory=list)
    last_delete_indices: List[int] = field(default_factory=list)
    applied_insertions: List[int] = field(default_factory=list)
    applied_deletions: List[int] = field(default_factory=list)
    pre_deletion_tokens: Optional[List[int]] = None
    mean_log_prob: float = 0.0
    tokens_for_display: List[int] = field(default_factory=list)
    logits: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None

    @property
    def active_length(self) -> int:
        return max(0, self.max_token_pos - self.config.initial_length)


# --------------------------------------------------------------------------- #
# Transformation interfaces


class DiffusionTransformation(Protocol):
    """Protocol implemented by all pipeline transformations."""

    def initialize(self, state: DiffusionState) -> None:
        """Called once per sample before diffusion iterations begin."""

    def on_iteration_start(self, state: DiffusionState) -> None:
        """Called at the beginning of every iteration."""

    def before_model_forward(self, state: DiffusionState) -> None:
        """Executed after start hooks and before the model forward pass."""

    def after_sampling(self, state: DiffusionState) -> None:
        """Invoked after the inference step and token sampling."""

    def on_iteration_end(self, state: DiffusionState) -> None:
        """Called after all per-iteration work is complete."""

    def finalize(self, state: DiffusionState) -> None:
        """Executed once per sample after the final iteration."""


class BaseTransformation:
    """Convenience base class providing no-op implementations."""

    def initialize(self, state: DiffusionState) -> None:
        return None

    def on_iteration_start(self, state: DiffusionState) -> None:
        return None

    def before_model_forward(self, state: DiffusionState) -> None:
        return None

    def after_sampling(self, state: DiffusionState) -> None:
        return None

    def on_iteration_end(self, state: DiffusionState) -> None:
        return None

    def finalize(self, state: DiffusionState) -> None:
        return None


# --------------------------------------------------------------------------- #
# Concrete transformations


class EditScheduleTransformation(BaseTransformation):
    """Structural edit policy that proposes insertions and deletions."""

    def __init__(
        self,
        schedule: str,
        *,
        insert_ratio_start: float,
        insert_ratio_end: float,
        delete_ratio_start: float,
        delete_ratio_end: float,
        delete_margin: float,
        delete_lambda: float,
        cooldown_distance: int,
        length_target_mode: str = 'none',
    ) -> None:
        self.schedule = schedule
        self.insert_ratio_start = insert_ratio_start
        self.insert_ratio_end = insert_ratio_end
        self.delete_ratio_start = delete_ratio_start
        self.delete_ratio_end = delete_ratio_end
        self.delete_margin = delete_margin
        self.delete_lambda = delete_lambda
        self.cooldown_distance = cooldown_distance
        self.length_target_mode = length_target_mode

    @property
    def enabled(self) -> bool:
        return self.schedule in {'linear', 'cosine'}

    def on_iteration_start(self, state: DiffusionState) -> None:
        state.applied_insertions = []
        state.applied_deletions = []
        state.pre_deletion_tokens = None

        if not self.enabled or state.last_log_probs is None:
            return

        cfg = state.config
        active_len = state.active_length

        r_ins = compute_noise_ratio(
            state.iteration,
            cfg.max_iterations,
            self.schedule,
            self.insert_ratio_start,
            self.insert_ratio_end,
        )
        r_del = compute_noise_ratio(
            state.iteration,
            cfg.max_iterations,
            self.schedule,
            self.delete_ratio_start,
            self.delete_ratio_end,
        )

        k_ins = max(0, int(r_ins * active_len))
        k_del = max(0, int(r_del * active_len))

        if self.length_target_mode == 'to_max_new':
            target_len = min(cfg.block_size, cfg.initial_length + cfg.max_new_tokens)
            error = state.max_token_pos - target_len
            if error > 0:
                k_del = min(active_len, k_del + min(3, error))
            elif error < 0:
                extra = min(3, -error)
                capacity = max(0, cfg.block_size - state.max_token_pos)
                k_ins = min(k_ins + extra, capacity + active_len)

        # Insertion scoring and application
        u = uncertainty_from_logprobs(
            state.last_log_probs[:, :state.max_token_pos, :],
            state.x[:, :state.max_token_pos],
        )
        gap_scores = score_gaps_for_insertion(u)
        allowed_gap_count = max(0, gap_scores.numel() - cfg.initial_length)
        k_ins = min(k_ins, allowed_gap_count)
        cooldown_mask_gaps = build_cooldown_mask(
            gap_scores.numel(),
            state.last_insert_indices,
            self.cooldown_distance,
            gap_scores.device,
        )
        sel_gaps = select_topk_mask(
            gap_scores,
            k_ins,
            forbid_lo=0,
            forbid_hi=cfg.initial_length,
            additional_forbid=cooldown_mask_gaps,
        )
        gap_indices = torch.nonzero(sel_gaps, as_tuple=False).flatten().tolist()
        if gap_indices:
            state.max_token_pos, applied_insertions = apply_insertions(
                state.x,
                gap_indices,
                max_token_pos=state.max_token_pos,
                block_size=cfg.block_size,
                fill_id=cfg.space_token_id,
            )
        else:
            applied_insertions = []

        state.applied_insertions = applied_insertions

        # Deletion scoring and application
        last_probs = state.last_log_probs.exp()
        del_scores = deletion_scores_from_probs(
            last_probs[:, :state.max_token_pos, :],
            state.x[:, :state.max_token_pos],
            margin=self.delete_margin,
            lam=self.delete_lambda,
        )
        allowed_delete_count = max(0, del_scores.numel() - cfg.initial_length)
        k_del = min(k_del, allowed_delete_count)
        cooldown_mask_del = build_cooldown_mask(
            del_scores.numel(),
            state.last_delete_indices,
            self.cooldown_distance,
            del_scores.device,
        )
        sel_del = select_topk_mask(
            del_scores,
            k_del,
            forbid_lo=0,
            forbid_hi=cfg.initial_length,
            additional_forbid=cooldown_mask_del,
        )
        del_indices = torch.nonzero(sel_del, as_tuple=False).flatten().tolist()
        pre_deletion_tokens: Optional[List[int]] = None
        if del_indices:
            pre_deletion_tokens = state.x[0, :state.max_token_pos].tolist()
            state.max_token_pos, applied_deletions = apply_deletions(
                state.x,
                del_indices,
                max_token_pos=state.max_token_pos,
                initial_length=cfg.initial_length,
                fill_id=cfg.space_token_id,
                prior_insertions=applied_insertions,
            )
        else:
            applied_deletions = []

        state.pre_deletion_tokens = pre_deletion_tokens
        state.applied_deletions = applied_deletions
        state.last_insert_indices = applied_insertions
        state.last_delete_indices = applied_deletions


class NoiseScheduleTransformation(BaseTransformation):
    """Re-noise policy applied before the model forward pass."""

    def __init__(self, schedule: str, *, start: float, end: float) -> None:
        self.schedule = schedule
        self.start = start
        self.end = end

    @property
    def enabled(self) -> bool:
        return self.schedule in {'linear', 'cosine'}

    def before_model_forward(self, state: DiffusionState) -> None:
        if not self.enabled:
            return

        ratio = compute_noise_ratio(
            state.iteration,
            state.config.max_iterations,
            self.schedule,
            self.start,
            self.end,
        )
        if ratio <= 0.0:
            return

        state.x = apply_re_noise(
            x=state.x,
            max_token_pos=state.max_token_pos,
            initial_length=state.config.initial_length,
            replace_character_id=state.config.space_token_id,
            ratio=ratio,
            device=state.config.device,
        )


class DisplayTransformation(BaseTransformation):
    """Observer that mirrors the original DiffusionDisplay side effects."""

    def __init__(self, display_factory) -> None:
        self._display_factory = display_factory
        self._display = None

    def initialize(self, state: DiffusionState) -> None:
        self._display = self._display_factory()

    def on_iteration_start(self, state: DiffusionState) -> None:
        if self._display is None:
            return
        self._display.start_iteration(state.max_token_pos)

    def before_model_forward(self, state: DiffusionState) -> None:
        if self._display is None:
            return
        self._display.register_insertions(state.applied_insertions)
        self._display.capture_pre_deletion(state.pre_deletion_tokens, state.applied_deletions)
        self._display.register_deletions(state.applied_deletions)

    def after_sampling(self, state: DiffusionState) -> None:
        if self._display is None:
            return
        self._display.emit_iteration(
            iteration=state.iteration,
            sample_index=state.sample_index,
            mean_log_prob=state.mean_log_prob,
            insert_count=len(state.applied_insertions),
            delete_count=len(state.applied_deletions),
            tokens=state.tokens_for_display,
        )

    def finalize(self, state: DiffusionState) -> None:
        if self._display is None:
            return
        self._display.emit_final_tokens(state.tokens_for_display)


# --------------------------------------------------------------------------- #
# Pipeline driver


class DiffusionPipeline:
    """Core diffusion sampler with pluggable transformations."""

    def __init__(
        self,
        model,
        *,
        config: DiffusionConfig,
        transformations: Sequence[DiffusionTransformation],
    ) -> None:
        self.model = model
        self.config = config
        self.transformations: List[DiffusionTransformation] = list(transformations)

    def run_samples(self, num_samples: int) -> List[List[int]]:
        """Run the diffusion process for a batch of samples."""
        outputs: List[List[int]] = []
        for sample_index in range(num_samples):
            outputs.append(self._run_single_sample(sample_index))
        return outputs

    def _run_single_sample(self, sample_index: int) -> List[int]:
        cfg = self.config
        device = cfg.device

        seq_length = cfg.block_size
        max_token_pos = min(seq_length, cfg.initial_length + cfg.max_new_tokens)

        x = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        x[0, :cfg.initial_length] = cfg.prompt

        logits, _ = self.model(x)
        if cfg.temperature <= 0:
            raise ValueError("temperature must be greater than zero to perform sampling.")

        last_log_probs = torch.log_softmax(logits, dim=-1).detach()

        state = DiffusionState(
            config=cfg,
            sample_index=sample_index,
            iteration=0,
            x=x,
            max_token_pos=max_token_pos,
            last_log_probs=last_log_probs,
        )

        for transformation in self.transformations:
            transformation.initialize(state)

        while state.iteration < cfg.max_iterations:
            for transformation in self.transformations:
                transformation.on_iteration_start(state)

            for transformation in self.transformations:
                transformation.before_model_forward(state)

            logits, _ = self.model(state.x)
            logits = logits / cfg.temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            active_log_probs = log_probs[0, :state.max_token_pos, :].to(dtype=torch.float)
            active_probs = probs[0, :state.max_token_pos, :].to(dtype=torch.float)

            flat_probs = active_probs.view(-1, active_probs.size(-1))
            sampled_indices = torch.multinomial(flat_probs, 1)
            sampled = sampled_indices.view(1, -1)

            state.x[:, :state.max_token_pos] = sampled
            if cfg.fix_prompt_during_diffusion:
                state.x[0, :cfg.initial_length] = cfg.prompt
                sampled[:, :cfg.initial_length] = cfg.prompt

            current_tokens = sampled[0, :state.max_token_pos].unsqueeze(-1)
            iteration_log_probs = active_log_probs.gather(-1, current_tokens).squeeze(-1)
            state.mean_log_prob = iteration_log_probs.mean().item()

            if state.max_token_pos < cfg.block_size:
                state.x[:, state.max_token_pos:] = 0

            state.tokens_for_display = state.x[0, :state.max_token_pos].tolist()
            state.logits = logits
            state.log_probs = log_probs
            state.last_log_probs = log_probs.detach()

            for transformation in self.transformations:
                transformation.after_sampling(state)

            for transformation in self.transformations:
                transformation.on_iteration_end(state)

            state.iteration += 1

        state.tokens_for_display = state.x[0, :state.max_token_pos].tolist()

        for transformation in self.transformations:
            transformation.finalize(state)

        return state.tokens_for_display


__all__ = [
    'DiffusionConfig',
    'DiffusionPipeline',
    'DiffusionState',
    'DiffusionTransformation',
    'DisplayTransformation',
    'EditScheduleTransformation',
    'NoiseScheduleTransformation',
]
