"""
GRPO Training Step: Core logic for Group-Relative Policy Optimization.

This module implements the GRPO training step that:
1. Fetches masked inputs from the dataset
2. Generates k completions per input
3. Scores completions with a frozen judge model
4. Computes advantages (group-relative rewards)
5. Calculates GRPO loss with KL divergence penalty
6. Updates generator weights
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from sample_utils import predict_and_sample_tokens, calculate_judge_scores
from core.batch import unpack_batch


class GRPOTrainingStep:
    """
    Encapsulates one GRPO training iteration.
    
    The GRPO algorithm trains a generator model to produce high-quality completions
    by using a frozen judge model as a reward signal and computing group-relative
    advantages to reduce variance.
    """
    
    def __init__(
        self,
        generator_model,
        reference_model,
        judge_model,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        config: Dict[str, Any],
        ctx=None,
        ddp: bool = False,
    ):
        """
        Initialize GRPO training step.

        Args:
            generator_model: Model to train (will be updated)
            reference_model: Frozen copy of initial generator (for KL penalty)
            judge_model: Frozen judge model (for reward signal)
            optimizer: Optimizer for generator
            scaler: Gradient scaler for mixed precision
            config: Configuration dict with GRPO hyperparameters
            ctx: Context manager for autocast (nullcontext or torch.amp.autocast)
            ddp: Whether using DDP
        """
        self.generator = generator_model
        self.reference = reference_model
        self.judge = judge_model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.ctx = ctx if ctx is not None else nullcontext()
        self.ddp = bool(ddp)

        # Freeze reference and judge models
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        self.judge.eval()
        for param in self.judge.parameters():
            param.requires_grad = False

        # Extract config values
        self.group_size = int(config.get('group_size', 8))
        self.kl_beta = float(config.get('kl_beta', 0.1))
        self.grad_clip = float(config.get('grad_clip', 1.0))
        self.grad_accum_steps = int(config.get('gradient_accumulation_steps', 1))
        self.mask_token_id = int(config['mask_token_id'])
        self.pad_token_id = config.get('pad_token_id', None)
        self.base_vocab_size = config.get('base_vocab_size', None)
        self.vocab_size = int(config['vocab_size'])
        self.temperature = float(config.get('temperature', 0.8))
        self.top_p = float(config.get('top_p', 0.95))
        self.device = config['device']
        
    def execute_step(
        self,
        batch: Dict[str, torch.Tensor],
        consumer,
        device: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Execute one GRPO training step with gradient accumulation.

        Args:
            batch: Dict with 'x' (masked input) and 'y' (targets)
            consumer: DatasetConsumer for fetching next batch
            device: Device string

        Returns:
            Tuple of (loss, next_batch, metrics_dict)
        """
        # Accumulate metrics across micro-steps
        accumulated_metrics = {
            'pg_loss': 0.0,
            'kl_penalty': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'mean_advantage': 0.0,
            'mean_kl': 0.0,
            'mean_log_prob': 0.0,
        }

        last_loss = None

        # Timing for performance analysis
        import time
        step_start = time.perf_counter()

        for micro_step in range(self.grad_accum_steps):
            # DDP gradient sync only on last micro-step
            if self.ddp:
                self.generator.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )

            # Memory optimization: explicitly delete tensors after use
            # GRPO is memory-intensive due to:
            # - 3 forward passes (sampling, current policy, reference policy)
            # - group_size multiplier (B*k samples)
            # - Large logits tensors (B*k, T, V)

            # Unpack batch
            t0 = time.perf_counter()
            X, Y = unpack_batch(batch)  # X: (B, T), Y: (B, T)

            # Identify masked positions (positions we want to fill in)
            mask = (X == self.mask_token_id)  # (B, T)

            # Debug: Check how many masks we have
            num_masks_in_X = mask.sum().item()
            total_tokens = X.numel()
            mask_pct = 100.0 * num_masks_in_X / total_tokens if total_tokens > 0 else 0

            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 1. Iteration start: mem={mem_alloc:.0f}MB, time={0:.3f}s, masks_in_X={num_masks_in_X}/{total_tokens} ({mask_pct:.1f}%)")

            # STEP 1: Generate k completions per input
            # Repeat each input k times: (B*k, T)
            X_repeated = X.repeat_interleave(self.group_size, dim=0)
            mask_repeated = mask.repeat_interleave(self.group_size, dim=0)

            # Sample completions using existing infrastructure
            # Don't return logits to save memory - we'll recompute them later
            with torch.no_grad():
                completions = predict_and_sample_tokens(
                    model=self.generator,
                    tokens=X_repeated,
                    mask_token_id=self.mask_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    vocab_size=self.vocab_size,
                    device=device,
                    verbose=False,
                    return_logits=False,  # Save memory by not returning logits
                    pad_token_id=self.pad_token_id,
                    base_vocab_size=self.base_vocab_size
                )

            t1 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

            # Debug: Check if completions still have masks
            num_masks_in_completions = (completions == self.mask_token_id).sum().item()
            print(f"  [Micro {micro_step}] 2. Samples generated: mem={mem_alloc:.0f}MB, time={t1-t0:.3f}s, masks_in_completions={num_masks_in_completions}")

            # STEP 2: Score completions with judge
            with torch.no_grad():
                rewards = calculate_judge_scores(
                    judge_model=self.judge,
                    tokens=completions,
                    device=device,
                    ctx=self.ctx
                )  # (B*k,)

            t2 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 3. Samples scored: mem={mem_alloc:.0f}MB, time={t2-t1:.3f}s, rewards={rewards.mean().item():.4f}±{rewards.std().item():.4f}")

            # STEP 3: Calculate advantages (group-relative)
            # Reshape to (B, k) to compute per-group baseline
            B = X.shape[0]
            rewards_grouped = rewards.view(B, self.group_size)  # (B, k)
            baseline = rewards_grouped.mean(dim=1, keepdim=True)  # (B, 1)
            advantages = rewards_grouped - baseline  # (B, k)

            # Debug: Check if advantages are all zero
            if advantages.abs().max().item() < 1e-8:
                print(f"  WARNING: All advantages are near zero! rewards_grouped={rewards_grouped.flatten().tolist()}")

            # Normalize advantages for stability
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.view(-1)  # Flatten back to (B*k,)

            # STEP 4: Compute log-probabilities (with gradients)
            # For BERT-style models: pass masked input, get logits, extract probs for generated tokens
            # We need P(generated_token | masked_context) at each masked position
            with self.ctx:
                logits_current, _ = self.generator(X_repeated, targets=None)

            # Ensure shapes match
            if logits_current.shape[1] != completions.shape[1]:
                # Truncate or pad to match
                min_len = min(logits_current.shape[1], completions.shape[1])
                logits_current = logits_current[:, :min_len, :]
                completions_for_gather = completions[:, :min_len]
                mask_repeated_for_sum = mask_repeated[:, :min_len]
            else:
                completions_for_gather = completions
                mask_repeated_for_sum = mask_repeated

            # Compute log-probs and gather in one step to save memory
            log_probs_current = F.log_softmax(logits_current, dim=-1)  # (B*k, T', V)
            token_log_probs = torch.gather(
                log_probs_current,
                dim=-1,
                index=completions_for_gather.unsqueeze(-1)
            ).squeeze(-1)  # (B*k, T')

            # Free logits_current immediately to save memory
            del logits_current, log_probs_current

            # Sum only over masked positions (where actions were taken)
            sequence_log_probs = (token_log_probs * mask_repeated_for_sum.float()).sum(dim=1)  # (B*k,)

            # Debug: Check if log probs are reasonable
            num_masked = mask_repeated_for_sum.float().sum(dim=1).mean().item()
            total_masks_in_repeated = mask_repeated_for_sum.sum().item()
            if sequence_log_probs.abs().max().item() < 1e-8:
                print(f"  WARNING: All sequence_log_probs are near zero! num_masked={num_masked:.1f}, total_masks_in_repeated={total_masks_in_repeated}")

            # Free token_log_probs after computing sequence log probs
            del token_log_probs

            # STEP 5: Compute KL divergence to reference policy
            # Same as above: pass masked input to get reference policy probs
            with torch.no_grad():
                with self.ctx:
                    logits_ref, _ = self.reference(X_repeated, targets=None)

                # Ensure shapes match
                if logits_ref.shape[1] != completions.shape[1]:
                    min_len = min(logits_ref.shape[1], completions.shape[1])
                    logits_ref = logits_ref[:, :min_len, :]

                log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                token_log_probs_ref = torch.gather(
                    log_probs_ref,
                    dim=-1,
                    index=completions_for_gather.unsqueeze(-1)
                ).squeeze(-1)

                # Free reference logits immediately
                del logits_ref, log_probs_ref

                sequence_log_probs_ref = (token_log_probs_ref * mask_repeated_for_sum.float()).sum(dim=1)

                # Free token log probs
                del token_log_probs_ref

            kl_divergence = sequence_log_probs - sequence_log_probs_ref  # (B*k,)

            # STEP 6: Compute GRPO loss
            # Policy gradient term (maximize reward-weighted log-probs)
            pg_loss = -(sequence_log_probs * advantages.detach()).mean()

            # KL penalty term (prevent deviation from reference)
            kl_penalty = self.kl_beta * kl_divergence.mean()

            # Total loss (scaled for gradient accumulation)
            loss = (pg_loss + kl_penalty) / self.grad_accum_steps

            t3 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 4. Loss computed: mem={mem_alloc:.0f}MB, time={t3-t2:.3f}s, pg_loss={pg_loss.item():.6f}, kl={kl_penalty.item():.6f}, total={loss.item():.6f}")

            # STEP 7: Backward pass
            self.scaler.scale(loss).backward()

            # Accumulate metrics (extract scalars before deleting tensors)
            accumulated_metrics['pg_loss'] += float(pg_loss.item())
            accumulated_metrics['kl_penalty'] += float(kl_penalty.item())
            accumulated_metrics['mean_reward'] += float(rewards.mean().item())
            accumulated_metrics['std_reward'] += float(rewards.std().item())
            accumulated_metrics['mean_advantage'] += float(advantages.mean().item())
            accumulated_metrics['mean_kl'] += float(kl_divergence.mean().item())
            accumulated_metrics['mean_log_prob'] += float(sequence_log_probs.mean().item())

            last_loss = loss

            # Free all intermediate tensors to save memory
            del X, Y, mask, X_repeated, mask_repeated
            del completions, completions_for_gather, mask_repeated_for_sum
            del rewards, rewards_grouped, baseline, advantages
            del sequence_log_probs, sequence_log_probs_ref, kl_divergence
            del pg_loss, kl_penalty

            # Fetch next batch
            batch = consumer.get_batch('train', device)

        # Gradient clipping (after all micro-steps)
        if self.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.grad_clip
            )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        t_opt = time.perf_counter()
        mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"  5. Optimizer step finished: mem={mem_alloc:.0f}MB, time={t_opt-step_start:.3f}s total")

        # Average metrics across micro-steps
        metrics = {
            'loss': float(last_loss.item()) * self.grad_accum_steps,  # Unscale for logging
            'pg_loss': accumulated_metrics['pg_loss'] / self.grad_accum_steps,
            'kl_penalty': accumulated_metrics['kl_penalty'] / self.grad_accum_steps,
            'mean_reward': accumulated_metrics['mean_reward'] / self.grad_accum_steps,
            'std_reward': accumulated_metrics['std_reward'] / self.grad_accum_steps,
            'mean_advantage': accumulated_metrics['mean_advantage'] / self.grad_accum_steps,
            'mean_kl': accumulated_metrics['mean_kl'] / self.grad_accum_steps,
            'mean_log_prob': accumulated_metrics['mean_log_prob'] / self.grad_accum_steps,
        }

        return last_loss, batch, metrics

