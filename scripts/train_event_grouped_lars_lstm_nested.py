#!/usr/bin/env python3
"""Nested event-fold evaluation for the LARS-initialized nonlinear LSTM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from train_event_grouped_lars_lstm import (
    TARGETS,
    indices_from_intervals,
    pearson,
    plot_validation,
    predict_intervals,
    starts_from_intervals,
)
from train_exact_window_end_to_end import ExactWindowFingerDecoder

from ecog_decoding.training import FINGER_NAMES, joint_trajectory_loss


class KinematicLarsDecoder(ExactWindowFingerDecoder):
    """LARS-position decoder with coupled movement-state and velocity heads."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.state_head = torch.nn.Linear(self.lstm.hidden_size, 1)
        self.velocity_head = torch.nn.Linear(self.lstm.hidden_size, 1)
        torch.nn.init.normal_(self.state_head.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.normal_(self.velocity_head.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.state_head.bias)
        torch.nn.init.zeros_(self.velocity_head.bias)

    @torch.no_grad()
    def initialize_state_prior(self, movement_fraction: float) -> None:
        probability = float(np.clip(movement_fraction, 1.0e-4, 1.0 - 1.0e-4))
        self.state_head.bias.fill_(np.log(probability / (1.0 - probability)))

    def decode_with_auxiliary(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        standardized = (features - self.feature_mean) / self.feature_scale
        direct = self.direct(standardized)
        recurrent, _ = self.lstm(standardized)
        pre_activation = direct + self.temporal(recurrent)
        position = F.softplus(pre_activation, beta=10.0).squeeze(-1)
        state_logit = self.state_head(recurrent).squeeze(-1)
        velocity = self.velocity_head(recurrent).squeeze(-1)
        return position, state_logit, velocity

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decode_with_auxiliary(features)[0]


def kinematic_hurdle_loss(
    position: torch.Tensor,
    state_logit: torch.Tensor,
    velocity: torch.Tensor,
    target: torch.Tensor,
    movement_threshold: float,
    rest_threshold: float,
    state_weight: float,
    velocity_weight: float,
    consistency_weight: float,
    curvature_weight: float,
    sample_period: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    moving = target >= movement_threshold
    resting = target <= rest_threshold
    known = moving | resting
    state = moving.to(target.dtype)
    positive = moving[known].sum().clamp_min(1.0)
    negative = resting[known].sum().clamp_min(1.0)
    positive_weight = (negative / positive).clamp(1.0, 12.0)
    state_loss = F.binary_cross_entropy_with_logits(
        state_logit[known], state[known], pos_weight=positive_weight
    )
    rest_loss = position[resting].square().mean() if resting.any() else position.sum() * 0.0
    movement_loss = (
        F.smooth_l1_loss(position[moving], target[moving], beta=0.10)
        if moving.any()
        else position.sum() * 0.0
    )
    if target.shape[1] > 1:
        target_velocity = torch.diff(target, dim=1) / sample_period
        position_velocity = torch.diff(position, dim=1) / sample_period
        velocity_prediction = velocity[:, 1:]
        transition = moving[:, 1:] | moving[:, :-1]
        edge_weight = 1.0 + 4.0 * transition.to(target.dtype)
        velocity_loss = (
            edge_weight * (velocity_prediction - target_velocity).square()
        ).sum() / edge_weight.sum()
        consistency = (velocity_prediction - position_velocity).square().mean()
        if target.shape[1] > 2:
            target_acceleration = torch.diff(target_velocity, dim=1)
            predicted_acceleration = torch.diff(velocity_prediction, dim=1)
            curvature = F.smooth_l1_loss(
                predicted_acceleration, target_acceleration, beta=0.25
            )
        else:
            curvature = position.sum() * 0.0
    else:
        velocity_loss = consistency = curvature = position.sum() * 0.0
    total = (
        movement_loss
        + rest_loss
        + state_weight * state_loss
        + velocity_weight * velocity_loss
        + consistency_weight * consistency
        + curvature_weight * curvature
    )
    return total, {
        "movement": movement_loss.detach(),
        "rest": rest_loss.detach(),
        "state": state_loss.detach(),
        "velocity": velocity_loss.detach(),
        "consistency": consistency.detach(),
        "curvature": curvature.detach(),
    }


@torch.inference_mode()
def predict_kinematic_intervals(
    model: KinematicLarsDecoder,
    features: torch.Tensor,
    intervals: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_indices: list[np.ndarray] = []
    all_position: list[np.ndarray] = []
    all_probability: list[np.ndarray] = []
    all_velocity: list[np.ndarray] = []
    for start, stop in intervals:
        position, state_logit, velocity = model.decode_with_auxiliary(
            features[None, start:stop]
        )
        all_indices.append(np.arange(start, stop, dtype=np.int64))
        all_position.append(position[0].float().cpu().numpy())
        all_probability.append(torch.sigmoid(state_logit[0]).float().cpu().numpy())
        all_velocity.append(velocity[0].float().cpu().numpy())
    return (
        np.concatenate(all_indices),
        np.concatenate(all_position),
        np.concatenate(all_probability),
        np.concatenate(all_velocity),
    )


def plot_kinematic_validation(
    path: Path,
    indices: np.ndarray,
    target: np.ndarray,
    position: np.ndarray,
    probability: np.ndarray,
    velocity: np.ndarray,
) -> None:
    order = np.argsort(indices)
    index = indices[order]
    gap = np.r_[False, np.diff(index) > 1]
    time = index.astype(np.float64) / 25.0
    values = [target[order].copy(), position[order].copy(), probability[order].copy(), velocity[order].copy()]
    for value in values:
        value[gap] = np.nan
    target_velocity = np.r_[np.nan, np.diff(values[0]) / 0.04]
    figure, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(time, values[0], color="black", linewidth=0.9, label="cleaned target")
    axes[0].plot(time, values[1], color="#2563eb", linewidth=0.9, label="predicted position")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(time, values[2], color="#7c3aed", linewidth=0.8, label="movement probability")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].plot(time, target_velocity, color="black", linewidth=0.7, label="target velocity")
    axes[2].plot(time, values[3], color="#d97706", linewidth=0.7, label="predicted velocity")
    axes[2].legend(frameon=False, ncol=2)
    axes[2].set_xlabel("original training time (s); gaps reset recurrent state")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def intervals_from_mask(mask: np.ndarray) -> list[list[int]]:
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [[int(start), int(stop)] for start, stop in zip(starts, stops)]


def model_from_lars(
    feature_count: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    hidden_size: int,
    near_zero_std: float,
    device: torch.device,
    architecture: str,
    movement_fraction: float,
) -> ExactWindowFingerDecoder:
    model_class = KinematicLarsDecoder if architecture == "kinematic" else ExactWindowFingerDecoder
    model = model_class(
        input_channels=1,
        component_count=1,
        selected_indices=np.arange(feature_count),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_size=hidden_size,
        frontend="wavelet",
        head_initialization="lars_linear_regime",
        output_activation="linear",
    ).to(device)
    model.initialize_lars_linear_regime(
        coefficients, intercept, near_zero_std=near_zero_std
    )
    if isinstance(model, KinematicLarsDecoder):
        model.initialize_state_prior(movement_fraction)
    return model


def decoder_parameters(model: ExactWindowFingerDecoder) -> list[torch.nn.Parameter]:
    parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
    if isinstance(model, KinematicLarsDecoder):
        parameters += list(model.state_head.parameters())
        parameters += list(model.velocity_head.parameters())
    return parameters


def train_epochs(
    model: ExactWindowFingerDecoder,
    features: torch.Tensor,
    target: torch.Tensor,
    intervals: list[list[int]],
    epochs: int,
    seed: int,
    sequence_steps: int,
    sequence_stride: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    compile_model: bool,
    loss_name: str,
    movement_threshold: float,
    movement_weight: float,
    velocity_weight: float,
    correlation_weight: float,
    rest_threshold: float,
    state_weight: float,
    consistency_weight: float,
    curvature_weight: float,
) -> list[float]:
    if epochs <= 0:
        return []
    parameters = decoder_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    decode = (
        model.decode_with_auxiliary
        if isinstance(model, KinematicLarsDecoder)
        else model.decode
    )
    if compile_model:
        decode = torch.compile(decode, mode="reduce-overhead")
    starts = starts_from_intervals(intervals, sequence_steps, sequence_stride)
    if starts.size == 0:
        raise RuntimeError("no complete sequences remain in nested training fold")
    offsets = np.arange(sequence_steps, dtype=np.int64)
    rng = np.random.default_rng(seed)
    epoch_losses: list[float] = []
    for _ in range(epochs):
        rng.shuffle(starts)
        model.train()
        losses: list[float] = []
        for begin in range(0, starts.size, batch_size):
            chosen = starts[begin : begin + batch_size]
            indices = torch.as_tensor(chosen[:, None] + offsets[None], device=features.device)
            decoded = decode(features[indices])
            batch_target = target[indices]
            if isinstance(model, KinematicLarsDecoder):
                prediction, state_logit, velocity = decoded
                loss, _ = kinematic_hurdle_loss(
                    prediction,
                    state_logit,
                    velocity,
                    batch_target,
                    movement_threshold,
                    rest_threshold,
                    state_weight,
                    velocity_weight,
                    consistency_weight,
                    curvature_weight,
                )
            elif loss_name == "mse":
                prediction = decoded
                loss = (prediction - batch_target).square().mean()
            else:
                prediction = decoded
                loss, _ = joint_trajectory_loss(
                    prediction.unsqueeze(-1),
                    batch_target.unsqueeze(-1),
                    movement_threshold=movement_threshold,
                    movement_weight=movement_weight,
                    velocity_weight=velocity_weight,
                    correlation_weight=correlation_weight,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(float(np.mean(losses)))
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_nested_v1"))
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--sequence-steps", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--near-zero-std", type=float, default=1e-3)
    parser.add_argument("--loss", choices=("mse", "joint"), default="mse")
    parser.add_argument("--architecture", choices=("position", "kinematic"), default="position")
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--rest-threshold", type=float, default=0.04)
    parser.add_argument("--movement-weight", type=float, default=4.0)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--correlation-weight", type=float, default=0.1)
    parser.add_argument("--state-weight", type=float, default=0.20)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--curvature-weight", type=float, default=0.02)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    fold_path = args.fold_root / f"sub{args.subject}" / args.finger / "folds.json"
    definition = json.loads(fold_path.read_text())
    row_count = int(definition["training_rows"])
    outer = definition["folds"][args.fold]
    outer_training_intervals = outer["training_intervals_after_purge"]
    outer_validation_intervals = outer["validation_intervals"]
    outer_training_mask = np.zeros(row_count, dtype=bool)
    outer_training_mask[indices_from_intervals(outer_training_intervals)] = True

    feature_path = (
        args.feature_root / f"sub{args.subject}" / "train_initialized_window_features.npy"
    )
    features_all = np.load(feature_path, mmap_mode="r")[:row_count]
    finger_index = list(FINGER_NAMES).index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    target_all = np.load(
        prepared / f"train_glove_{TARGETS[args.subject]}.npy"
    )[24 : 24 + row_count, finger_index]
    raw_all = np.load(prepared / "train_glove_25hz_raw.npy")[
        24 : 24 + row_count, finger_index
    ]

    cache = (
        args.selection_cache_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}.npz"
    )
    if not cache.exists():
        raise FileNotFoundError(
            f"missing outer-training-only LARS cache {cache}; run the exploratory trainer first"
        )
    saved = np.load(cache)
    selected_source = saved["selected_source"]
    feature_mean = saved["feature_mean"]
    feature_scale = saved["feature_scale"]
    coefficients = saved["coefficients"]
    intercept = float(saved["intercept"])
    features = torch.from_numpy(
        np.asarray(features_all[:, selected_source], dtype=np.float32)
    ).to(device)
    target_tensor = torch.from_numpy(np.asarray(target_all, dtype=np.float32)).to(device)

    inner_records: list[dict[str, object]] = []
    selected_epochs: list[int] = []
    for inner_fold in range(3):
        if inner_fold == args.fold:
            continue
        inner_definition = definition["folds"][inner_fold]
        inner_training_mask = np.zeros(row_count, dtype=bool)
        inner_training_mask[
            indices_from_intervals(inner_definition["training_intervals_after_purge"])
        ] = True
        inner_training_mask &= outer_training_mask
        inner_validation_mask = np.zeros(row_count, dtype=bool)
        inner_validation_mask[
            indices_from_intervals(inner_definition["validation_intervals"])
        ] = True
        inner_validation_mask &= outer_training_mask
        inner_training_intervals = intervals_from_mask(inner_training_mask)
        inner_validation_intervals = intervals_from_mask(inner_validation_mask)
        if not inner_training_intervals or not inner_validation_intervals:
            raise RuntimeError("nested event split is empty")

        torch.manual_seed(args.seed)
        model = model_from_lars(
            len(selected_source),
            feature_mean,
            feature_scale,
            coefficients,
            intercept,
            args.hidden_size,
            args.near_zero_std,
            device,
            args.architecture,
            float(np.mean(target_all[inner_training_mask] >= args.movement_threshold)),
        )
        _, initialization = predict_intervals(
            model, features, inner_validation_intervals
        )
        inner_indices = indices_from_intervals(inner_validation_intervals)
        best_score = pearson(initialization, raw_all[inner_indices])
        best_epoch = 0
        history: list[dict[str, float]] = [
            {"epoch": 0, "validation_raw_pcc": best_score}
        ]
        parameters = decoder_parameters(model)
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        decode = (
            model.decode_with_auxiliary
            if isinstance(model, KinematicLarsDecoder)
            else model.decode
        )
        if args.compile:
            decode = torch.compile(decode, mode="reduce-overhead")
        starts = starts_from_intervals(
            inner_training_intervals, args.sequence_steps, args.sequence_stride
        )
        if starts.size == 0:
            raise RuntimeError("no complete sequences remain in inner training fold")
        offsets = np.arange(args.sequence_steps, dtype=np.int64)
        rng = np.random.default_rng(args.seed)
        # Train one epoch at a time so the inner fold alone selects the epoch.
        for epoch in range(1, args.max_epochs + 1):
            rng.shuffle(starts)
            model.train()
            losses: list[float] = []
            for begin in range(0, starts.size, args.batch_size):
                chosen = starts[begin : begin + args.batch_size]
                indices = torch.as_tensor(
                    chosen[:, None] + offsets[None], device=features.device
                )
                decoded = decode(features[indices])
                batch_target = target_tensor[indices]
                if isinstance(model, KinematicLarsDecoder):
                    prediction, state_logit, velocity = decoded
                    loss, _ = kinematic_hurdle_loss(
                        prediction,
                        state_logit,
                        velocity,
                        batch_target,
                        args.movement_threshold,
                        args.rest_threshold,
                        args.state_weight,
                        args.velocity_weight,
                        args.consistency_weight,
                        args.curvature_weight,
                    )
                elif args.loss == "mse":
                    prediction = decoded
                    loss = (prediction - batch_target).square().mean()
                else:
                    prediction = decoded
                    loss, _ = joint_trajectory_loss(
                        prediction.unsqueeze(-1),
                        batch_target.unsqueeze(-1),
                        movement_threshold=args.movement_threshold,
                        movement_weight=args.movement_weight,
                        velocity_weight=args.velocity_weight,
                        correlation_weight=args.correlation_weight,
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch == 1 or epoch % args.validation_interval == 0:
                order, prediction = predict_intervals(
                    model, features, inner_validation_intervals
                )
                score = pearson(prediction, raw_all[order])
                history.append(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "validation_raw_pcc": score,
                    }
                )
                if score > best_score + 1e-4:
                    best_score = score
                    best_epoch = epoch
        selected_epochs.append(best_epoch)
        inner_records.append(
            {
                "inner_validation_fold": inner_fold,
                "selected_epoch": best_epoch,
                "best_validation_raw_pcc": best_score,
                "history": history,
            }
        )

    final_epoch = int(np.rint(np.median(selected_epochs)))
    torch.manual_seed(args.seed)
    final_model = model_from_lars(
        len(selected_source),
        feature_mean,
        feature_scale,
        coefficients,
        intercept,
        args.hidden_size,
        args.near_zero_std,
        device,
        args.architecture,
        float(np.mean(target_all[outer_training_mask] >= args.movement_threshold)),
    )
    outer_order, initialized = predict_intervals(
        final_model, features, outer_validation_intervals
    )
    initialization_score = pearson(initialized, raw_all[outer_order])
    losses = train_epochs(
        final_model,
        features,
        target_tensor,
        outer_training_intervals,
        final_epoch,
        args.seed,
        args.sequence_steps,
        args.sequence_stride,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.compile,
        args.loss,
        args.movement_threshold,
        args.movement_weight,
        args.velocity_weight,
        args.correlation_weight,
        args.rest_threshold,
        args.state_weight,
        args.consistency_weight,
        args.curvature_weight,
    )
    outer_order, outer_prediction = predict_intervals(
        final_model, features, outer_validation_intervals
    )
    outer_score = pearson(outer_prediction, raw_all[outer_order])
    outer_cleaned_score = pearson(outer_prediction, target_all[outer_order])

    output = (
        args.output_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}"
        / f"seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_indices.npy", outer_order)
    np.save(output / "validation_prediction.npy", outer_prediction)
    np.save(output / "validation_raw_target.npy", raw_all[outer_order])
    np.save(output / "validation_cleaned_target.npy", target_all[outer_order])
    kinematic_metrics: dict[str, float] = {}
    if isinstance(final_model, KinematicLarsDecoder):
        kin_order, kin_position, probability, velocity = predict_kinematic_intervals(
            final_model, features, outer_validation_intervals
        )
        if not np.array_equal(kin_order, outer_order):
            raise RuntimeError("kinematic and trajectory validation order disagree")
        np.save(output / "validation_movement_probability.npy", probability)
        np.save(output / "validation_velocity.npy", velocity)
        contiguous = np.diff(kin_order) == 1
        target_velocity = np.diff(target_all[kin_order]) / 0.04
        predicted_velocity = velocity[1:]
        kinematic_metrics = {
            "velocity_pcc": pearson(
                predicted_velocity[contiguous], target_velocity[contiguous]
            ),
            "movement_state_accuracy": float(
                np.mean(
                    (probability >= 0.5)
                    == (target_all[kin_order] >= args.movement_threshold)
                )
            ),
        }
        plot_kinematic_validation(
            output / "validation_kinematics.png",
            kin_order,
            target_all[kin_order],
            kin_position,
            probability,
            velocity,
        )
    plot_validation(
        output / "validation_trajectory.png",
        outer_order,
        raw_all[outer_order].copy(),
        target_all[outer_order].copy(),
        initialized.copy(),
        outer_prediction.copy(),
    )
    summary = {
        "protocol": "nested per-finger event CV; inner folds select epoch; outer fold evaluated once",
        "subject": args.subject,
        "finger": args.finger,
        "fold": args.fold,
        "seed": args.seed,
        "released_test_touched": False,
        "official_final_validation_touched": False,
        "outer_validation_used_for_epoch_selection": False,
        "training_loss": args.loss,
        "architecture": args.architecture,
        "loss_weights": {
            "movement_threshold": args.movement_threshold,
            "movement_weight": args.movement_weight,
            "velocity_weight": args.velocity_weight,
            "correlation_weight": args.correlation_weight,
            "rest_threshold": args.rest_threshold,
            "state_weight": args.state_weight,
            "consistency_weight": args.consistency_weight,
            "curvature_weight": args.curvature_weight,
        },
        "inner_epoch_selection": inner_records,
        "selected_epoch": final_epoch,
        "initialized_outer_raw_pcc": initialization_score,
        "outer_raw_pcc": outer_score,
        "outer_cleaned_target_pcc": outer_cleaned_score,
        "kinematic_metrics": kinematic_metrics,
        "selected_feature_count": int(len(selected_source)),
        "epoch_training_losses": losses,
    }
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "selected_source_indices": selected_source,
            "summary": summary,
        },
        output / "model.pt",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"saved": str(output), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
