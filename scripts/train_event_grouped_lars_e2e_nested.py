#!/usr/bin/env python3
"""Nested event-fold evaluation of two-stage LARS-initialized end-to-end LSTMs.

The outer fold is evaluated once. Two inner event folds select training
duration. Stage one trains only the recurrent head from cached exact-window
features; stage two recomputes features from raw ECoG and fine-tunes the
ICA-initialized spatial projection and bior6.8 packet filters.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES, joint_trajectory_loss
from train_event_grouped_lars_lstm import (
    TARGETS,
    correlation_order,
    indices_from_intervals,
    pearson,
    starts_from_intervals,
)
from train_event_grouped_lars_lstm_nested import intervals_from_mask
from train_exact_window_end_to_end import ExactWindowFingerDecoder


def build_model(
    *, input_channels: int, ica: np.ndarray, selected: np.ndarray,
    mean: np.ndarray, scale: np.ndarray, coefficients: np.ndarray,
    intercept: float, hidden_size: int, near_zero_std: float,
    output_activation: str, device: torch.device,
) -> ExactWindowFingerDecoder:
    model = ExactWindowFingerDecoder(
        input_channels=input_channels,
        component_count=ica.shape[0],
        selected_indices=selected,
        feature_mean=mean,
        feature_scale=scale,
        hidden_size=hidden_size,
        frontend="asymmetric",
        head_initialization="lars_linear_regime",
        output_activation=output_activation,
    ).to(device)
    with torch.no_grad():
        model.spatial.weight[:, :, 0].copy_(torch.as_tensor(ica, device=device))
        model.initialize_lars_linear_regime(
            coefficients, intercept, near_zero_std=near_zero_std
        )
    return model


def event_grouped_cv_splits(
    intervals: list[list[int]], training_indices: np.ndarray, folds: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split disjoint event intervals without putting one interval in both sides."""
    assignments: list[list[list[int]]] = [[] for _ in range(folds)]
    loads = np.zeros(folds, dtype=np.int64)
    for interval in sorted(intervals, key=lambda value: value[1] - value[0], reverse=True):
        fold = int(np.argmin(loads))
        assignments[fold].append(interval)
        loads[fold] += interval[1] - interval[0]
    global_to_local = np.full(int(training_indices.max()) + 1, -1, dtype=np.int64)
    global_to_local[training_indices] = np.arange(training_indices.size)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_positions = np.arange(training_indices.size, dtype=np.int64)
    for intervals_for_fold in assignments:
        validation_global = indices_from_intervals(intervals_for_fold)
        validation_global = validation_global[validation_global < global_to_local.size]
        validation = global_to_local[validation_global]
        validation = validation[validation >= 0]
        mask = np.ones(training_indices.size, dtype=bool)
        mask[validation] = False
        training = all_positions[mask]
        if validation.size and training.size:
            splits.append((training, validation))
    if len(splits) < 2:
        raise RuntimeError("inner LARS requires at least two event-grouped subfolds")
    return splits


def fit_or_load_inner_lars(
    *,
    features_all: np.ndarray,
    target_all: np.ndarray,
    training_intervals: list[list[int]],
    cache: Path,
    max_features: int,
) -> dict[str, np.ndarray | float]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not cache.exists():
            training_indices = indices_from_intervals(training_intervals)
            prescreen = correlation_order(
                np.asarray(features_all[training_indices]), target_all[training_indices]
            )[:max_features]
            scaler = StandardScaler()
            selected_training = scaler.fit_transform(
                np.asarray(features_all[training_indices][:, prescreen])
            ).astype(np.float64, copy=False)
            splits = event_grouped_cv_splits(training_intervals, training_indices)
            lars = LassoLarsCV(cv=splits, max_iter=500, n_jobs=1)
            lars.fit(selected_training, target_all[training_indices])
            nonzero = np.flatnonzero(lars.coef_)
            if nonzero.size == 0:
                raise RuntimeError("inner event-grouped LARS selected no features")
            temporary = cache.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                selected_source=prescreen[nonzero],
                feature_mean=scaler.mean_[nonzero].astype(np.float32),
                feature_scale=scaler.scale_[nonzero].astype(np.float32),
                coefficients=lars.coef_[nonzero].astype(np.float32),
                intercept=np.asarray(float(lars.intercept_)),
                alpha=np.asarray(float(lars.alpha_)),
            )
            temporary.replace(cache)
        saved = np.load(cache)
        return {
            "selected_source": saved["selected_source"],
            "feature_mean": saved["feature_mean"],
            "feature_scale": saved["feature_scale"],
            "coefficients": saved["coefficients"],
            "intercept": float(saved["intercept"]),
            "alpha": float(saved["alpha"]),
        }


def trainable_parameters(model: ExactWindowFingerDecoder) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_optimizer(
    model: ExactWindowFingerDecoder,
    head_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
    if model.output_activation == "hurdle":
        head_parameters += list(model.movement_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": model.spatial.parameters(), "lr": 0.0},
            {"params": model.wavelet.parameters(), "lr": 0.0},
            {
                "params": head_parameters,
                "lr": head_learning_rate,
            },
        ],
        weight_decay=weight_decay,
    )


def padded_batches(
    starts: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    order = starts.copy()
    rng.shuffle(order)
    batches: list[np.ndarray] = []
    for begin in range(0, order.size, batch_size):
        chosen = order[begin : begin + batch_size]
        if chosen.size < batch_size:
            chosen = np.concatenate(
                (chosen, rng.choice(order, size=batch_size - chosen.size, replace=True))
            )
        batches.append(chosen)
    return batches


def batch_loss(
    decoded: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    observed: torch.Tensor,
    args: argparse.Namespace,
    amplitude_scale: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    if isinstance(decoded, tuple):
        _, state_logit, amplitude = decoded
        moving = observed >= args.movement_threshold
        state_loss = F.binary_cross_entropy_with_logits(
            state_logit,
            moving.to(observed.dtype),
            pos_weight=positive_weight,
        )
        if moving.any():
            amplitude_loss = F.mse_loss(
                amplitude[moving] / amplitude_scale,
                observed[moving] / amplitude_scale,
            )
        else:
            amplitude_loss = amplitude.sum() * 0.0
        # Weighted Bernoulli state likelihood plus a unit-variance
        # conditional Gaussian amplitude likelihood on moving bins.
        return state_loss + 0.5 * amplitude_loss
    prediction = decoded
    if args.loss == "mse":
        return (prediction - observed).square().mean()
    loss, _ = joint_trajectory_loss(
        prediction.unsqueeze(-1),
        observed.unsqueeze(-1),
        movement_threshold=args.movement_threshold,
        movement_weight=args.movement_weight,
        velocity_weight=args.velocity_weight,
        correlation_weight=args.correlation_weight,
    )
    return loss


@torch.inference_mode()
def predict_intervals(
    model: ExactWindowFingerDecoder,
    cached_features: torch.Tensor,
    raw_windows: torch.Tensor,
    intervals: list[list[int]],
    use_raw: bool,
    chunk_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_indices: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    for start, stop in intervals:
        for begin in range(start, stop, chunk_steps):
            end = min(stop, begin + chunk_steps)
            prediction = (
                model(raw_windows[begin:end][None])
                if use_raw
                else model.decode(cached_features[begin:end][None])
            )
            all_indices.append(np.arange(begin, end, dtype=np.int64))
            all_predictions.append(prediction[0].float().cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_predictions)


@torch.inference_mode()
def predict_hurdle_intervals(
    model: ExactWindowFingerDecoder,
    cached_features: torch.Tensor,
    raw_windows: torch.Tensor,
    intervals: list[list[int]],
    use_raw: bool,
    chunk_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model.output_activation != "hurdle":
        raise RuntimeError("hurdle diagnostics require hurdle output activation")
    model.eval()
    all_indices: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_amplitudes: list[np.ndarray] = []
    for start, stop in intervals:
        for begin in range(start, stop, chunk_steps):
            end = min(stop, begin + chunk_steps)
            prediction, state_logit, amplitude = (
                model.forward_with_hurdle(raw_windows[begin:end][None])
                if use_raw
                else model.decode_with_hurdle(cached_features[begin:end][None])
            )
            all_indices.append(np.arange(begin, end, dtype=np.int64))
            all_predictions.append(prediction[0].float().cpu().numpy())
            all_probabilities.append(torch.sigmoid(state_logit[0]).float().cpu().numpy())
            all_amplitudes.append(amplitude[0].float().cpu().numpy())
    return (
        np.concatenate(all_indices),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
        np.concatenate(all_amplitudes),
    )


def train_one_epoch(
    *,
    model: ExactWindowFingerDecoder,
    forward: object,
    decode: object,
    optimizer: torch.optim.Optimizer,
    cached_features: torch.Tensor,
    raw_windows: torch.Tensor,
    target: torch.Tensor,
    starts: np.ndarray,
    use_raw: bool,
    args: argparse.Namespace,
    rng: np.random.Generator,
    amplitude_scale: torch.Tensor,
    positive_weight: torch.Tensor,
) -> float:
    model.train()
    offsets = torch.arange(args.sequence_steps, device=target.device)
    losses: list[float] = []
    batch_size = args.unfrozen_batch_size if use_raw else args.batch_size
    for chosen in padded_batches(starts, batch_size, rng):
        origins = torch.as_tensor(chosen, device=target.device)
        indices = origins[:, None] + offsets[None]
        decoded = (
            forward(raw_windows[indices])
            if use_raw
            else decode(cached_features[indices])
        )
        loss = batch_loss(
            decoded, target[indices], args, amplitude_scale, positive_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters(model), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def compiled_calls(
    model: ExactWindowFingerDecoder, enabled: bool
) -> tuple[object, object]:
    forward = model.forward_with_hurdle if model.output_activation == "hurdle" else model
    decode = model.decode_with_hurdle if model.output_activation == "hurdle" else model.decode
    if not enabled:
        return forward, decode
    return (
        torch.compile(forward, mode="reduce-overhead"),
        torch.compile(decode, mode="reduce-overhead"),
    )


def hurdle_training_constants(
    target: torch.Tensor,
    training_intervals: list[list[int]],
    movement_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.as_tensor(
        indices_from_intervals(training_intervals), device=target.device
    )
    values = target[indices]
    moving = values >= movement_threshold
    if moving.any():
        amplitude_scale = torch.quantile(values[moving], 0.95).clamp_min(1.0e-3)
    else:
        amplitude_scale = values.new_tensor(1.0)
    positive_weight = ((~moving).sum() / moving.sum().clamp_min(1)).clamp(1.0, 12.0)
    return amplitude_scale.detach(), positive_weight.detach()


def train_with_validation(
    *,
    model: ExactWindowFingerDecoder,
    cached_features: torch.Tensor,
    raw_windows: torch.Tensor,
    target: torch.Tensor,
    raw_target: np.ndarray,
    training_intervals: list[list[int]],
    validation_intervals: list[list[int]],
    args: argparse.Namespace,
    seed: int,
) -> tuple[int, float, list[dict[str, object]]]:
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    forward, decode = compiled_calls(model, args.compile)
    amplitude_scale, positive_weight = hurdle_training_constants(
        target, training_intervals, args.movement_threshold
    )
    starts = starts_from_intervals(
        training_intervals, args.sequence_steps, args.sequence_stride
    )
    if starts.size == 0:
        raise RuntimeError("event fold has no complete training sequences")
    rng = np.random.default_rng(seed)
    order, initialized = predict_intervals(
        model, cached_features, raw_windows, validation_intervals, False,
        args.prediction_chunk_steps,
    )
    best_score = pearson(initialized, raw_target[order])
    best_epoch = 0
    history: list[dict[str, object]] = [
        {"epoch": 0, "validation_raw_pcc": best_score, "stage": "initialized"}
    ]
    for epoch in range(1, args.max_epochs + 1):
        use_raw = epoch > args.warmup_epochs
        if epoch == args.warmup_epochs + 1:
            optimizer.param_groups[0]["lr"] = args.spatial_learning_rate
            optimizer.param_groups[1]["lr"] = args.wavelet_learning_rate
        loss = train_one_epoch(
            model=model,
            forward=forward,
            decode=decode,
            optimizer=optimizer,
            cached_features=cached_features,
            raw_windows=raw_windows,
            target=target,
            starts=starts,
            use_raw=use_raw,
            args=args,
            rng=rng,
            amplitude_scale=amplitude_scale,
            positive_weight=positive_weight,
        )
        if (
            epoch == 1
            or epoch == args.warmup_epochs
            or epoch % args.validation_interval == 0
        ):
            order, estimate = predict_intervals(
                model, cached_features, raw_windows, validation_intervals,
                use_raw, args.prediction_chunk_steps,
            )
            score = pearson(estimate, raw_target[order])
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "validation_raw_pcc": score,
                    "stage": "end_to_end" if use_raw else "frozen_stem",
                }
            )
            if score > best_score + 1.0e-4:
                best_score = score
                best_epoch = epoch
    return best_epoch, best_score, history


def train_fixed_epochs(
    *,
    model: ExactWindowFingerDecoder,
    cached_features: torch.Tensor,
    raw_windows: torch.Tensor,
    target: torch.Tensor,
    training_intervals: list[list[int]],
    epochs: int,
    args: argparse.Namespace,
    seed: int,
) -> list[float]:
    if epochs <= 0:
        return []
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    forward, decode = compiled_calls(model, args.compile)
    amplitude_scale, positive_weight = hurdle_training_constants(
        target, training_intervals, args.movement_threshold
    )
    starts = starts_from_intervals(
        training_intervals, args.sequence_steps, args.sequence_stride
    )
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        use_raw = epoch > args.warmup_epochs
        if epoch == args.warmup_epochs + 1:
            optimizer.param_groups[0]["lr"] = args.spatial_learning_rate
            optimizer.param_groups[1]["lr"] = args.wavelet_learning_rate
        losses.append(
            train_one_epoch(
                model=model,
                forward=forward,
                decode=decode,
                optimizer=optimizer,
                cached_features=cached_features,
                raw_windows=raw_windows,
                target=target,
                starts=starts,
                use_raw=use_raw,
                args=args,
                rng=rng,
                amplitude_scale=amplitude_scale,
                positive_weight=positive_weight,
            )
        )
    return losses


def plot_result(
    path: Path,
    indices: np.ndarray,
    raw: np.ndarray,
    cleaned: np.ndarray,
    initialized: np.ndarray,
    prediction: np.ndarray,
) -> None:
    order = np.argsort(indices)
    index = indices[order]
    gap = np.r_[False, np.diff(index) > 1]
    time_axis = index / 25.0
    series = [
        raw[order].copy(), cleaned[order].copy(),
        initialized[order].copy(), prediction[order].copy(),
    ]
    for values in series:
        values[gap] = np.nan
    figure, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(time_axis, series[0], color="#64748b", linewidth=0.7, label="raw glove")
    axes[0].plot(time_axis, series[1], color="black", linewidth=0.8, label="cleaned target")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(time_axis, series[2], color="#a78bfa", linewidth=0.7, label="LARS-LSTM initialization")
    axes[1].plot(time_axis, series[3], color="#2563eb", linewidth=0.8, label="two-stage prediction")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xlabel("original training time (s); gaps reset recurrent state")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_hurdle_result(
    path: Path,
    indices: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    amplitude: np.ndarray,
    threshold: float,
) -> None:
    order = np.argsort(indices)
    index = indices[order]
    gap = np.r_[False, np.diff(index) > 1]
    time_axis = index / 25.0
    series = [target[order].copy(), prediction[order].copy(), probability[order].copy(), amplitude[order].copy()]
    for values in series:
        values[gap] = np.nan
    figure, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(time_axis, series[0], color="black", linewidth=0.8, label="cleaned target")
    axes[0].plot(time_axis, series[1], color="#2563eb", linewidth=0.8, label="gate × amplitude")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(time_axis, series[2], color="#db2777", linewidth=0.8, label="movement probability")
    axes[1].axhline(0.5, color="#94a3b8", linestyle="--", linewidth=0.7)
    axes[1].fill_between(
        time_axis, 0, 1,
        where=np.nan_to_num(series[0], nan=0.0) >= threshold,
        color="#f59e0b", alpha=0.12, label="target moving",
    )
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].legend(frameon=False, ncol=2)
    axes[2].plot(time_axis, series[3], color="#7c3aed", linewidth=0.8, label="conditional amplitude")
    axes[2].plot(time_axis, series[0], color="black", linewidth=0.6, alpha=0.7, label="target")
    axes[2].legend(frameon=False, ncol=2)
    axes[2].set_xlabel("original training time (s); gaps reset recurrent state")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--target",
        default=None,
        help="target file stem after train_glove_; defaults to the subject policy",
    )
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--inner-selection-cache-root", type=Path, default=Path("outputs/event_lars_inner_selection_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_lars_e2e_nested_v1"))
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--sequence-steps", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfrozen-batch-size", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--spatial-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--wavelet-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--near-zero-std", type=float, default=1.0e-3)
    parser.add_argument(
        "--output-activation",
        choices=("linear", "softplus", "hurdle"),
        default="linear",
    )
    parser.add_argument("--loss", choices=("mse", "joint"), default="mse")
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--movement-weight", type=float, default=4.0)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--correlation-weight", type=float, default=0.1)
    parser.add_argument("--prediction-chunk-steps", type=int, default=512)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    started = time.perf_counter()

    definition = json.loads(
        (args.fold_root / f"sub{args.subject}" / args.finger / "folds.json").read_text()
    )
    row_count = int(definition["training_rows"])
    outer = definition["folds"][args.fold]
    outer_training_intervals = outer["training_intervals_after_purge"]
    outer_validation_intervals = outer["validation_intervals"]
    outer_training_mask = np.zeros(row_count, dtype=bool)
    outer_training_mask[indices_from_intervals(outer_training_intervals)] = True

    prepared = args.prepared_root / f"sub{args.subject}"
    ecog_numpy = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    ecog = torch.from_numpy(np.array(ecog_numpy, copy=True)).to(device)
    raw_windows = ecog.unfold(0, args.window_samples, args.stride_samples)[:row_count]
    finger_index = list(FINGER_NAMES).index(args.finger)
    target_name = args.target or TARGETS[args.subject]
    target_all = np.load(prepared / f"train_glove_{target_name}.npy")[24 : 24 + row_count, finger_index]
    raw_all = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + row_count, finger_index]
    target = torch.as_tensor(target_all, dtype=torch.float32, device=device)

    all_fixed = np.load(
        args.feature_root / f"sub{args.subject}" / "train_initialized_window_features.npy",
        mmap_mode="r",
    )[:row_count]
    if args.target is None:
        selection_path = (
            args.selection_cache_root
            / f"sub{args.subject}"
            / args.finger
            / f"fold{args.fold}.npz"
        )
        saved = np.load(selection_path)
        outer_selection = {
            "selected_source": saved["selected_source"],
            "feature_mean": saved["feature_mean"],
            "feature_scale": saved["feature_scale"],
            "coefficients": saved["coefficients"],
            "intercept": float(saved["intercept"]),
        }
    else:
        outer_selection = fit_or_load_inner_lars(
            features_all=all_fixed,
            target_all=target_all,
            training_intervals=outer_training_intervals,
            cache=(
                args.selection_cache_root
                / target_name
                / f"sub{args.subject}"
                / args.finger
                / f"fold{args.fold}.npz"
            ),
            max_features=args.max_features,
        )
    selected = np.asarray(outer_selection["selected_source"], dtype=np.int64)
    mean = np.asarray(outer_selection["feature_mean"])
    scale = np.asarray(outer_selection["feature_scale"])
    coefficients = np.asarray(outer_selection["coefficients"])
    intercept = float(outer_selection["intercept"])
    cached_features = torch.as_tensor(
        np.asarray(all_fixed[:, selected], dtype=np.float32), device=device
    )
    ica = np.load(args.ica_root / f"sub{args.subject}" / "fastica_unmixing.npy")

    inner_records: list[dict[str, object]] = []
    selected_epochs: list[int] = []
    torch.manual_seed(args.seed)
    audit_model = build_model(
        input_channels=ecog.shape[1], ica=ica, selected=selected,
        mean=mean, scale=scale, coefficients=coefficients,
        intercept=intercept, hidden_size=args.hidden_size,
        near_zero_std=args.near_zero_std,
        output_activation=args.output_activation, device=device,
    )
    audit_indices = torch.linspace(
        0, row_count - 1, steps=min(row_count, 64), device=device
    ).round().long()
    with torch.inference_mode():
        raw_initial_features = audit_model.extract(raw_windows[audit_indices][None])[0]
    cached_initial_features = cached_features[audit_indices]
    feature_difference = raw_initial_features - cached_initial_features
    feature_audit = {
        "sample_count": int(audit_indices.numel()),
        "rmse": float(feature_difference.square().mean().sqrt().cpu()),
        "max_abs_error": float(feature_difference.abs().max().cpu()),
        "reference_rms": float(cached_initial_features.square().mean().sqrt().cpu()),
    }
    del audit_model, raw_initial_features, cached_initial_features, feature_difference
    torch.cuda.empty_cache()
    for inner_fold in range(3):
        if inner_fold == args.fold:
            continue
        inner = definition["folds"][inner_fold]
        training_mask = np.zeros(row_count, dtype=bool)
        training_mask[indices_from_intervals(inner["training_intervals_after_purge"])] = True
        training_mask &= outer_training_mask
        validation_mask = np.zeros(row_count, dtype=bool)
        validation_mask[indices_from_intervals(inner["validation_intervals"])] = True
        validation_mask &= outer_training_mask
        training_intervals = intervals_from_mask(training_mask)
        validation_intervals = intervals_from_mask(validation_mask)
        inner_selection = fit_or_load_inner_lars(
            features_all=all_fixed,
            target_all=target_all,
            training_intervals=training_intervals,
            cache=(
                args.inner_selection_cache_root
                / (target_name if args.target is not None else Path())
                / f"sub{args.subject}"
                / args.finger
                / f"outer{args.fold}_inner{inner_fold}.npz"
            ),
            max_features=args.max_features,
        )
        inner_selected = np.asarray(inner_selection["selected_source"], dtype=np.int64)
        inner_cached_features = torch.as_tensor(
            np.asarray(all_fixed[:, inner_selected], dtype=np.float32), device=device
        )
        torch.manual_seed(args.seed)
        model = build_model(
            input_channels=ecog.shape[1], ica=ica, selected=inner_selected,
            mean=np.asarray(inner_selection["feature_mean"]),
            scale=np.asarray(inner_selection["feature_scale"]),
            coefficients=np.asarray(inner_selection["coefficients"]),
            intercept=float(inner_selection["intercept"]),
            hidden_size=args.hidden_size,
            near_zero_std=args.near_zero_std,
            output_activation=args.output_activation, device=device,
        )
        best_epoch, best_score, history = train_with_validation(
            model=model, cached_features=inner_cached_features, raw_windows=raw_windows,
            target=target, raw_target=raw_all,
            training_intervals=training_intervals,
            validation_intervals=validation_intervals,
            args=args, seed=args.seed,
        )
        selected_epochs.append(best_epoch)
        inner_records.append(
            {
                "inner_validation_fold": inner_fold,
                "selected_epoch": best_epoch,
                "best_validation_raw_pcc": best_score,
                "lars_feature_count": int(inner_selected.size),
                "lars_alpha": float(inner_selection["alpha"]),
                "history": history,
            }
        )
        del model, inner_cached_features
        torch.cuda.empty_cache()

    selected_epoch = int(np.rint(np.median(selected_epochs)))
    torch.manual_seed(args.seed)
    final_model = build_model(
        input_channels=ecog.shape[1], ica=ica, selected=selected,
        mean=mean, scale=scale, coefficients=coefficients,
        intercept=intercept, hidden_size=args.hidden_size,
        near_zero_std=args.near_zero_std,
        output_activation=args.output_activation, device=device,
    )
    outer_order, initialized = predict_intervals(
        final_model, cached_features, raw_windows, outer_validation_intervals,
        False, args.prediction_chunk_steps,
    )
    losses = train_fixed_epochs(
        model=final_model, cached_features=cached_features,
        raw_windows=raw_windows, target=target,
        training_intervals=outer_training_intervals,
        epochs=selected_epoch, args=args, seed=args.seed,
    )
    use_raw = selected_epoch > args.warmup_epochs
    outer_order, prediction = predict_intervals(
        final_model, cached_features, raw_windows, outer_validation_intervals,
        use_raw, args.prediction_chunk_steps,
    )
    output = args.output_root / f"sub{args.subject}" / args.finger / f"fold{args.fold}" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_indices.npy", outer_order)
    np.save(output / "validation_prediction.npy", prediction)
    np.save(output / "validation_initialized_prediction.npy", initialized)
    np.save(output / "validation_raw_target.npy", raw_all[outer_order])
    np.save(output / "validation_cleaned_target.npy", target_all[outer_order])
    plot_result(
        output / "validation_trajectories.png", outer_order,
        raw_all[outer_order], target_all[outer_order], initialized, prediction,
    )
    hurdle_metrics: dict[str, float] = {}
    if args.output_activation == "hurdle":
        hurdle_order, hurdle_prediction, probability, amplitude = predict_hurdle_intervals(
            final_model, cached_features, raw_windows, outer_validation_intervals,
            use_raw, args.prediction_chunk_steps,
        )
        if not np.array_equal(hurdle_order, outer_order):
            raise RuntimeError("hurdle and trajectory validation order disagree")
        if not np.allclose(hurdle_prediction, prediction, atol=1.0e-6):
            raise RuntimeError("hurdle and trajectory predictions disagree")
        moving = target_all[outer_order] >= args.movement_threshold
        predicted_moving = probability >= 0.5
        true_positive = int(np.sum(moving & predicted_moving))
        false_positive = int(np.sum(~moving & predicted_moving))
        false_negative = int(np.sum(moving & ~predicted_moving))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        hurdle_metrics = {
            "movement_state_f1": float(
                2.0 * precision * recall / max(precision + recall, 1.0e-12)
            ),
            "movement_state_precision": float(precision),
            "movement_state_recall": float(recall),
            "mean_rest_probability": float(np.mean(probability[~moving])) if (~moving).any() else 0.0,
            "mean_movement_probability": float(np.mean(probability[moving])) if moving.any() else 0.0,
            "rest_prediction_rms": float(np.sqrt(np.mean(np.square(prediction[~moving])))) if (~moving).any() else 0.0,
            "movement_amplitude_rmse": float(np.sqrt(np.mean(np.square(amplitude[moving] - target_all[outer_order][moving])))) if moving.any() else 0.0,
        }
        np.save(output / "validation_movement_probability.npy", probability)
        np.save(output / "validation_conditional_amplitude.npy", amplitude)
        plot_hurdle_result(
            output / "validation_hurdle_components.png",
            outer_order, target_all[outer_order], prediction, probability,
            amplitude, args.movement_threshold,
        )
    torch.save(
        {
            "model_state_dict": copy.deepcopy(final_model.state_dict()),
            "feature_indices": selected,
            "selected_epoch": selected_epoch,
        },
        output / "model.pt",
    )
    report = {
        "protocol": "nested per-finger event folds; frozen LARS-LSTM then end-to-end ICA/bior6.8 fine-tuning",
        "subject": args.subject,
        "finger": args.finger,
        "fold": args.fold,
        "seed": args.seed,
        "official_final_validation_touched": False,
        "released_test_touched": False,
        "feature_count": int(selected.size),
        "cached_vs_raw_initial_feature_audit": feature_audit,
        "inner_folds": inner_records,
        "selected_epoch": selected_epoch,
        "selected_stage": "end_to_end" if use_raw else ("frozen_stem" if selected_epoch else "initialized"),
        "initialization_outer_raw_pcc": pearson(initialized, raw_all[outer_order]),
        "outer_validation_raw_pcc": pearson(prediction, raw_all[outer_order]),
        "outer_validation_cleaned_pcc": pearson(prediction, target_all[outer_order]),
        "training_losses": losses,
        "hurdle_metrics": hurdle_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "configuration": {
            "warmup_epochs": args.warmup_epochs,
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "spatial_learning_rate": args.spatial_learning_rate,
            "wavelet_learning_rate": args.wavelet_learning_rate,
            "loss": args.loss,
            "output_activation": args.output_activation,
            "target": target_name,
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    keys = (
        "subject", "finger", "fold", "seed", "selected_epoch",
        "selected_stage", "initialization_outer_raw_pcc",
        "outer_validation_raw_pcc", "runtime_seconds",
    )
    print(json.dumps({key: report[key] for key in keys}, indent=2), flush=True)


if __name__ == "__main__":
    main()
