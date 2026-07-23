from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossWeights, TrainingConfig
from .model import HILDA


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def sample_condition_lengths(
    lengths: torch.Tensor, minimum: float, maximum: float
) -> torch.Tensor:
    fractions = torch.empty(
        len(lengths), device=lengths.device
    ).uniform_(minimum, maximum)
    conditions = torch.floor(lengths.float() * fractions).long()
    return torch.minimum(
        conditions.clamp_min(1), (lengths - 1).clamp_min(1)
    )


def health_alignment_losses(
    estimator: nn.Module | None,
    waveforms: torch.Tensor,
    target_soh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if estimator is None:
        zero = waveforms.sum() * 0.0
        return zero, zero, zero
    prediction = estimator(waveforms).reshape_as(target_soh)
    level = F.smooth_l1_loss(prediction, target_soh, beta=0.02)
    if prediction.numel() < 2:
        zero = level * 0.0
        return level, zero, zero
    predicted_delta = prediction[1:] - prediction[:-1]
    target_delta = target_soh[1:] - target_soh[:-1]
    delta = F.smooth_l1_loss(predicted_delta, target_delta, beta=0.005)
    monotonic = F.relu(predicted_delta).mean()
    return level, delta, monotonic


def waveform_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous_prediction: torch.Tensor | None,
    previous_target: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    waveform = F.smooth_l1_loss(prediction, target, beta=1.0)
    if prediction.shape[-1] % 3:
        raise ValueError("cycle feature dimension must contain three channels")
    points = prediction.shape[-1] // 3
    pred_curve = prediction.reshape(-1, points, 3)
    target_curve = target.reshape(-1, points, 3)
    critical_start = max(0, round(0.70 * points))
    critical = F.smooth_l1_loss(
        pred_curve[:, critical_start:],
        target_curve[:, critical_start:],
        beta=1.0,
    )

    def protocol_features(curve: torch.Tensor) -> torch.Tensor:
        integral = curve.mean(1)
        terminal = curve[:, -1]
        span = curve.amax(1) - curve.amin(1)
        third = curve[:, :, 2]
        third_channel = torch.stack(
            [third[:, -1] - third[:, 0], third.amax(1)], dim=1
        )
        return torch.cat([integral, terminal, span, third_channel], dim=1)

    protocol = F.smooth_l1_loss(
        protocol_features(pred_curve),
        protocol_features(target_curve),
        beta=0.1,
    )
    boundary = F.smooth_l1_loss(prediction[:1], target[:1], beta=1.0)
    if previous_prediction is not None and previous_target is not None:
        boundary = boundary + F.smooth_l1_loss(
            prediction[:1] - previous_prediction,
            target[:1] - previous_target,
            beta=1.0,
        )
    multiscale = waveform * 0.0
    count = 0
    for lag in (1, 4, 16):
        if prediction.shape[0] > lag:
            multiscale = multiscale + F.smooth_l1_loss(
                prediction[lag:] - prediction[:-lag],
                target[lag:] - target[:-lag],
                beta=1.0,
            )
            count += 1
    if count:
        multiscale = multiscale / count
    return {
        "flow_waveform": waveform,
        "critical_region": critical,
        "protocol_features": protocol,
        "boundary": boundary,
        "multiscale": multiscale,
    }


def _weighted_total(
    losses: Mapping[str, torch.Tensor], weights: LossWeights
) -> torch.Tensor:
    return (
        losses["latent"]
        + weights.health * losses["health"]
        + weights.health_delta * losses["health_delta"]
        + weights.health_monotonic * losses["health_monotonic"]
        + weights.reconstruction * losses["reconstruction"]
        + weights.waveform_delta * losses["waveform_delta"]
        + weights.kl * losses["kl"]
        + weights.reference * losses["reference"]
        + weights.flow_waveform * losses["flow_waveform"]
        + weights.critical_region * losses["critical_region"]
        + weights.protocol_features * losses["protocol_features"]
        + weights.boundary * losses["boundary"]
        + weights.multiscale * losses["multiscale"]
    )


def hilda_loss(
    model: HILDA,
    reference_codec: nn.Module,
    functional_estimator: nn.Module | None,
    scaled_waveforms: torch.Tensor,
    target_soh: torch.Tensor,
    lengths: torch.Tensor,
    waveform_center: torch.Tensor,
    waveform_scale: torch.Tensor,
    config: TrainingConfig,
    scheduled_sampling_probability: float,
    rollout_blocks: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Final HILDA joint objective on a batch of padded lifecycle tensors."""
    batch, max_length, feature_dim = scaled_waveforms.shape
    flat = scaled_waveforms.reshape(-1, feature_dim)
    mu, logvar = model.codec.encode_stats(flat)
    mu = mu.reshape(batch, max_length, -1)
    logvar = logvar.reshape_as(mu)
    states = model.normalize_latents(mu)
    positions = torch.arange(max_length, device=lengths.device)[None]
    valid = positions < lengths[:, None]
    states = torch.where(valid[:, :, None], states, torch.zeros_like(states))
    with torch.no_grad():
        reference_mu = reference_codec.encode(flat).reshape_as(mu)

    condition_lengths = sample_condition_lengths(
        lengths,
        config.min_observed_fraction,
        config.max_observed_fraction,
    )
    collected: dict[str, list[torch.Tensor]] = {}
    totals = []
    for row in range(batch):
        length = int(lengths[row])
        condition = int(condition_lengths[row])
        terminal = min(length - 1, max_length - 1)
        block_count = max(
            1,
            math.ceil(
                (terminal + 1 - condition) / model.flow.config.block_size
            ),
        )
        if (
            block_count > 1
            and torch.rand((), device=states.device)
            < model.flow.config.terminal_block_probability
        ):
            block_index = block_count - 1
        else:
            block_index = int(
                torch.randint(0, block_count, (1,), device=states.device)
            )
        start = condition + block_index * model.flow.config.block_size
        end = min(start + model.flow.config.block_size, max_length)
        history = states[row : row + 1, :start]

        if (
            rollout_blocks > 0
            and start > condition
            and torch.rand((), device=states.device)
            < scheduled_sampling_probability
        ):
            rollout_start = max(
                condition,
                start - rollout_blocks * model.flow.config.block_size,
            )
            generated_history = states[row : row + 1, :rollout_start]
            was_training = model.flow.training
            model.flow.eval()
            with torch.no_grad():
                while generated_history.shape[1] < start:
                    count = min(
                        model.flow.config.block_size,
                        start - generated_history.shape[1],
                    )
                    generated_history = torch.cat(
                        [
                            generated_history,
                            model.flow.generate_block(
                                generated_history,
                                count,
                                config.rollout_flow_steps,
                            ).detach(),
                        ],
                        dim=1,
                    )
            model.flow.train(was_training)
            history = generated_history

        target = states[row : row + 1, start:end]
        t = torch.rand(1, device=states.device) * 0.96 + 0.02
        noise = torch.randn_like(target)
        mixed = t[:, None, None] * target + (1.0 - t[:, None, None]) * noise
        prediction = model.flow.predict_x1(history, mixed, t, start)
        active = torch.arange(start, end, device=states.device) < length
        active_prediction = prediction[0, active]
        active_target = target[0, active]
        latent = F.mse_loss(active_prediction, active_target)

        predicted_latent = model.denormalize_latents(active_prediction)
        predicted_scaled = model.codec.decode(predicted_latent)
        target_scaled = scaled_waveforms[row, start:end][active]
        predicted_raw = (
            predicted_scaled
            * waveform_scale.to(
                device=predicted_scaled.device, dtype=predicted_scaled.dtype
            )
            + waveform_center.to(
                device=predicted_scaled.device, dtype=predicted_scaled.dtype
            )
        )
        health, health_delta, health_monotonic = health_alignment_losses(
            functional_estimator,
            predicted_raw,
            target_soh[row, start:end][active],
        )

        target_mu = mu[row, start:end][active]
        target_logvar = logvar[row, start:end][active]
        reconstructed = model.codec.decode(target_mu)
        reconstruction = F.mse_loss(reconstructed, target_scaled)
        if reconstructed.shape[0] > 1:
            waveform_delta = F.mse_loss(
                reconstructed[1:] - reconstructed[:-1],
                target_scaled[1:] - target_scaled[:-1],
            )
        else:
            waveform_delta = reconstruction * 0.0
        kl = model.codec.kl_standard_normal(target_mu, target_logvar)
        reference = F.mse_loss(
            target_mu, reference_mu[row, start:end][active]
        )
        previous = (
            scaled_waveforms[row, start - 1 : start] if start > 0 else None
        )
        shape = waveform_losses(
            predicted_scaled, target_scaled, previous, previous
        )
        losses = {
            "latent": latent,
            "health": health,
            "health_delta": health_delta,
            "health_monotonic": health_monotonic,
            "reconstruction": reconstruction,
            "waveform_delta": waveform_delta,
            "kl": kl,
            "reference": reference,
            **shape,
        }
        totals.append(_weighted_total(losses, config.weights))
        for name, value in losses.items():
            collected.setdefault(name, []).append(value.detach())

    return torch.stack(totals).mean(), {
        name: torch.stack(values).mean()
        for name, values in collected.items()
    }


class HILDATrainer:
    """One-step trainer with the released rollout curriculum and frozen guides."""

    def __init__(
        self,
        model: HILDA,
        reference_codec: nn.Module | None,
        functional_estimator: nn.Module | None,
        waveform_center: torch.Tensor,
        waveform_scale: torch.Tensor,
        config: TrainingConfig | None = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.reference_codec = freeze(
            reference_codec
            if reference_codec is not None
            else copy.deepcopy(model.codec)
        )
        self.functional_estimator = (
            freeze(functional_estimator)
            if functional_estimator is not None
            else None
        )
        if (
            functional_estimator is None
            and any(
                value > 0
                for value in (
                    self.config.weights.health,
                    self.config.weights.health_delta,
                    self.config.weights.health_monotonic,
                )
            )
        ):
            raise ValueError(
                "a frozen functional estimator is required by the loss weights"
            )
        self.waveform_center = waveform_center
        self.waveform_scale = waveform_scale
        self.optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.flow.parameters(),
                    "lr": self.config.flow_learning_rate,
                    "weight_decay": self.config.flow_weight_decay,
                },
                {
                    "params": model.codec.parameters(),
                    "lr": self.config.codec_learning_rate,
                    "weight_decay": self.config.codec_weight_decay,
                },
            ]
        )

    def train_step(
        self,
        scaled_waveforms: torch.Tensor,
        target_soh: torch.Tensor,
        lengths: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> dict[str, float]:
        self.model.train()
        self.reference_codec.eval()
        if self.functional_estimator is not None:
            self.functional_estimator.eval()
        progress = (step - 1) / max(total_steps - 1, 1)
        curriculum = self.config.rollout_curriculum
        stage = min(int(progress * len(curriculum)), len(curriculum) - 1)
        rollout_blocks = curriculum[stage]
        if rollout_blocks == 0:
            rollout_blocks = 1_000_000
        scheduled_probability = (
            self.config.scheduled_sampling_start
            + progress
            * (
                self.config.scheduled_sampling_end
                - self.config.scheduled_sampling_start
            )
        )
        total, losses = hilda_loss(
            self.model,
            self.reference_codec,
            self.functional_estimator,
            scaled_waveforms,
            target_soh,
            lengths,
            self.waveform_center,
            self.waveform_scale,
            self.config,
            scheduled_probability,
            rollout_blocks,
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.flow.parameters(), self.config.flow_gradient_clip
        )
        torch.nn.utils.clip_grad_norm_(
            self.model.codec.parameters(), self.config.codec_gradient_clip
        )
        self.optimizer.step()
        return {
            "total": float(total.detach()),
            **{name: float(value) for name, value in losses.items()},
            "scheduled_sampling": scheduled_probability,
            "rollout_blocks": float(rollout_blocks),
        }
