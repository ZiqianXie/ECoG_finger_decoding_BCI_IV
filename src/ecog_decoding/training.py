"""Training utilities for joint ECoG finger-trajectory decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
HISTORICAL_FINGERS = (0, 1, 2, 4)


@dataclass(frozen=True)
class AlignedSequence:
    ecog: np.ndarray
    target: np.ndarray


def align_causal_sequence(
    ecog: np.ndarray,
    target: np.ndarray,
    start_bin: int,
    stop_bin: int,
    history_bins: int = 25,
    samples_per_bin: int = 40,
) -> AlignedSequence:
    """Return ECoG context and labels with one output per requested target bin."""
    if ecog.ndim != 2 or target.ndim != 2:
        raise ValueError("ecog and target must both be two-dimensional")
    if history_bins < 1 or samples_per_bin < 1:
        raise ValueError("history_bins and samples_per_bin must be positive")
    if not history_bins - 1 <= start_bin < stop_bin <= target.shape[0]:
        raise ValueError("requested bins are outside the causal target range")
    context_start = start_bin - history_bins + 1
    raw_start = context_start * samples_per_bin
    raw_stop = stop_bin * samples_per_bin
    if raw_stop > ecog.shape[0]:
        raise ValueError("ECoG does not cover the requested target bins")
    values = np.array(ecog[raw_start:raw_stop].T, dtype=np.float32, copy=True)
    labels = np.array(target[start_bin:stop_bin], dtype=np.float32, copy=True)
    energy_bins = values.shape[1] // samples_per_bin
    output_steps = energy_bins - history_bins + 1
    if output_steps != labels.shape[0]:
        raise RuntimeError("internal ECoG/target alignment error")
    return AlignedSequence(ecog=values, target=labels)


def joint_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    movement_threshold: float = 0.08,
    movement_weight: float = 4.0,
    velocity_weight: float = 0.2,
    correlation_weight: float = 0.1,
    level_kind: str = "mse",
    huber_delta: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Movement-balanced level, velocity, and correlation objective."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching (batch,time,finger) shapes")
    weights = 1.0 + movement_weight * (target > movement_threshold).to(target.dtype)
    if level_kind == "mse":
        pointwise_level = (prediction - target).square()
    elif level_kind == "huber":
        pointwise_level = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
            beta=huber_delta,
        )
    else:
        raise ValueError("level_kind must be 'mse' or 'huber'")
    level = (weights * pointwise_level).sum() / weights.sum()
    if prediction.shape[1] > 1:
        velocity = torch.mean(
            torch.diff(prediction, dim=1).sub(torch.diff(target, dim=1)).square()
        )
    else:
        velocity = prediction.new_zeros(())
    pred_centered = prediction - prediction.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred_centered * target_centered, dim=1)
    # Add epsilon before sqrt: clamping afterward still leaves an infinite
    # sqrt derivative when a finger is constant within a short training crop.
    denominator = torch.sqrt(
        torch.sum(pred_centered.square(), dim=1)
        * torch.sum(target_centered.square(), dim=1)
        + 1.0e-7
    )
    correlation = numerator / denominator
    correlation_loss = 1.0 - correlation.mean()
    total = level + velocity_weight * velocity + correlation_weight * correlation_loss
    return total, {
        "level": level.detach(),
        "velocity": velocity.detach(),
        "correlation_loss": correlation_loss.detach(),
    }


def position_velocity_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    level_scale: torch.Tensor,
    velocity_scale: torch.Tensor,
    velocity_weight: float = 0.25,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Robust, scale-normalized trajectory and velocity objective."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    level_residual = (prediction - target) / level_scale.clamp_min(1.0e-6)
    level = F.smooth_l1_loss(
        level_residual, torch.zeros_like(level_residual), beta=beta
    )
    # Decoder batches are (batch, time); retain the same time axis for an
    # optional trailing finger dimension.
    time_axis = 0 if prediction.ndim == 1 else 1
    if prediction.shape[time_axis] > 1:
        velocity_residual = (
            torch.diff(prediction, dim=time_axis) - torch.diff(target, dim=time_axis)
        ) / velocity_scale.clamp_min(1.0e-6)
        velocity = F.smooth_l1_loss(
            velocity_residual, torch.zeros_like(velocity_residual), beta=beta
        )
    else:
        velocity = prediction.new_zeros(())
    total = level + velocity_weight * velocity
    return total, {"level": level.detach(), "velocity": velocity.detach()}


def trajectory_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    """Return per-finger and aggregate Pearson correlations plus RMSE."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have matching (time,finger) shapes")
    correlations: list[float] = []
    for finger in range(target.shape[1]):
        x = prediction[:, finger] - prediction[:, finger].mean()
        y = target[:, finger] - target[:, finger].mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        correlations.append(float(np.dot(x, y) / denominator) if denominator > 0 else 0.0)
    named = {
        FINGER_NAMES[index] if index < len(FINGER_NAMES) else str(index): value
        for index, value in enumerate(correlations)
    }
    historical = [correlations[index] for index in HISTORICAL_FINGERS if index < len(correlations)]
    return {
        "pearson_by_finger": named,
        "pearson_macro_five": float(np.mean(correlations)),
        "pearson_historical_four": float(np.mean(historical)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }
