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

            # Skip if no masks (nothing to learn)
            if num_masks_in_X == 0:
                print(f"  [Micro {micro_step}] WARNING: No masks in batch, skipping")
                batch = consumer.get_batch('train', device)
                micro_step -= 1  # Don't count this as a valid step
                continue

            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 1. Iteration start: mem={mem_alloc:.0f}MB, time={0:.3f}s, masks_in_X={num_masks_in_X}/{total_tokens} ({mask_pct:.1f}%)")

            t0 = time.perf_counter()

            # STEP 1: Generate k completions per input
            # Repeat each input k times: (B*k, T)
            X_repeated = X.repeat_interleave(self.group_size, dim=0)
            mask_repeated = mask.repeat_interleave(self.group_size, dim=0)

            # STEP 2: Forward pass WITH gradients and sample completions
            # CRITICAL: Use same logits for both sampling and gradient computation
            # This ensures policy gradient is computed on the correct distribution
            with self.ctx:
                completions, logits_current = predict_and_sample_tokens(
                    model=self.generator,
                    tokens=X_repeated,
                    mask_token_id=self.mask_token_id,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    vocab_size=self.vocab_size,
                    device=device,
                    verbose=True,  # Enable detailed timing logs
                    return_logits=True,  # Return logits for gradient computation
                    pad_token_id=self.pad_token_id,
                    base_vocab_size=self.base_vocab_size,
                    no_grad=False  # Enable gradients for policy gradient
                )

            t1 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 2. Forward and sample: mem={mem_alloc:.0f}MB, time={t1-t0:.3f}s")

            # Check if all completions are identical (sampling failed)
            # This can happen if all masks are filled with the same token
            if self.group_size > 1:
                first_completion = completions[0]
                all_identical = all(torch.equal(completions[i], first_completion) for i in range(1, self.group_size))
                if all_identical:
                    print(f"  WARNING: All {self.group_size} completions are identical! Skipping this batch.")
                    # Fetch next batch and restart this micro-step
                    batch = consumer.get_batch('train', device)
                    continue

            # STEP 3: Score completions with judge
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

            # STEP 4: Calculate advantages (group-relative)
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

            # STEP 5: Compute log-probabilities from the SAME logits used for sampling
            # Verify shapes match - no fallback, fail fast
            if logits_current.shape[1] != completions.shape[1]:
                raise RuntimeError(
                    f"Shape mismatch: logits_current.shape={logits_current.shape}, "
                    f"completions.shape={completions.shape}. "
                    f"Model forward pass returned wrong sequence length. "
                    f"Expected {completions.shape[1]}, got {logits_current.shape[1]}. "
                    f"This indicates a bug in the model or sampling code."
                )

            # Compute log-probs and gather in one step to save memory
            # IMPORTANT: completions and mask are constants (from sampling), detach to avoid unnecessary graph
            log_probs_current = F.log_softmax(logits_current, dim=-1)  # (B*k, T, V)
            token_log_probs = torch.gather(
                log_probs_current,
                dim=-1,
                index=completions.detach().unsqueeze(-1)
            ).squeeze(-1)  # (B*k, T)

            # Free log_probs_current to save memory
            del log_probs_current

            # Sum only over masked positions (where actions were taken)
            sequence_log_probs = (token_log_probs * mask_repeated.detach().float()).sum(dim=1)  # (B*k,)

            # Debug: Check if log probs are reasonable
            num_masked_per_seq = mask_repeated.float().sum(dim=1).mean().item()
            total_masks_in_repeated = mask_repeated.sum().item()

            # Only warn if we actually have zero masks (which shouldn't happen)
            if total_masks_in_repeated == 0:
                print(f"  WARNING: No masked positions found! This should not happen. Original masks_in_X={num_masks_in_X}")
            elif sequence_log_probs.abs().max().item() < 1e-8:
                # Log probs are zero despite having masks - this is suspicious
                avg_log_prob = token_log_probs[mask_repeated].mean().item() if total_masks_in_repeated > 0 else 0
                print(f"  WARNING: sequence_log_probs near zero despite {total_masks_in_repeated} masks! avg_token_log_prob={avg_log_prob:.6f}")

            # Free token_log_probs and logits_current after computing sequence log probs
            del token_log_probs, logits_current

            t3 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 4. Log probs computed: mem={mem_alloc:.0f}MB, time={t3-t2:.3f}s")

            # STEP 6: Compute KL divergence to reference policy
            # CRITICAL: Detach X_repeated to avoid building computation graph for reference model
            dummy_targets = torch.zeros_like(X_repeated).detach()

            with torch.no_grad():
                with self.ctx:
                    logits_ref, _ = self.reference(X_repeated.detach(), targets=dummy_targets)

                # Verify shapes match
                if logits_ref.shape[1] != completions.shape[1]:
                    raise RuntimeError(
                        f"Reference model shape mismatch: logits_ref.shape={logits_ref.shape}, "
                        f"completions.shape={completions.shape}"
                    )

                log_probs_ref = F.log_softmax(logits_ref, dim=-1)
                token_log_probs_ref = torch.gather(
                    log_probs_ref,
                    dim=-1,
                    index=completions.detach().unsqueeze(-1)
                ).squeeze(-1)

                # Free reference logits immediately
                del logits_ref, log_probs_ref

                sequence_log_probs_ref = (token_log_probs_ref * mask_repeated.detach().float()).sum(dim=1)

                # Free token log probs
                del token_log_probs_ref

            kl_divergence = sequence_log_probs - sequence_log_probs_ref  # (B*k,)

            t4 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 5. Reference policy forward: mem={mem_alloc:.0f}MB, time={t4-t3:.3f}s")

            # STEP 7: Compute GRPO loss
            # Policy gradient term (maximize reward-weighted log-probs)
            # Note: sequence_log_probs are large negative numbers (sum of log probs over ~400 tokens)
            # advantages are zero-mean, so pg_loss can be positive or negative
            # What matters: gradient direction is correct (minimize pg_loss → increase log_probs for positive advantages)
            pg_loss = -(sequence_log_probs * advantages.detach()).mean()

            # KL penalty term (prevent deviation from reference)
            kl_penalty = self.kl_beta * kl_divergence.mean()

            # Total loss (scaled for gradient accumulation)
            loss = (pg_loss + kl_penalty) / self.grad_accum_steps

            t5 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

            # Debug: Show actual values to understand the loss
            avg_seq_log_prob = sequence_log_probs.mean().item()
            avg_advantage = advantages.mean().item()
            avg_kl = kl_divergence.mean().item()

            print(f"  [Micro {micro_step}] 6. Loss computed: mem={mem_alloc:.0f}MB, time={t5-t4:.3f}s")
            print(f"      pg_loss={pg_loss.item():.4f}, kl_penalty={kl_penalty.item():.4f}, total={loss.item():.4f}")
            print(f"      avg_seq_log_prob={avg_seq_log_prob:.2f}, avg_advantage={avg_advantage:.4f}, avg_kl={avg_kl:.4f}")

            # STEP 8: Backward pass
            t6 = time.perf_counter()
            self.scaler.scale(loss).backward()
            t7 = time.perf_counter()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            print(f"  [Micro {micro_step}] 7. Backward: mem={mem_alloc:.0f}MB, time={t7-t6:.3f}s")

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
            del completions
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
            
        t__before_opt = time.perf_counter()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        t_opt = time.perf_counter()
        mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        print(f"  5. Optimizer step finished: mem={mem_alloc:.0f}MB, time={t_opt-t__before_opt:.3f}s optimizer, time={t_opt-step_start:.3f}s total")

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

