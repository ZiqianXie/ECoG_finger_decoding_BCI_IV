#!/usr/bin/env python3
"""Train a LARS-initialized nonlinear LSTM on purged per-finger event folds."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_exact_window_end_to_end import ExactWindowFingerDecoder  # noqa: E402

from ecog_decoding.training import FINGER_NAMES  # noqa: E402


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def indices_from_intervals(intervals: list[list[int]]) -> np.ndarray:
    if not intervals:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def starts_from_intervals(
    intervals: list[list[int]], sequence_steps: int, stride: int
) -> np.ndarray:
    pieces = [
        np.arange(start, stop - sequence_steps + 1, stride, dtype=np.int64)
        for start, stop in intervals
        if stop - start >= sequence_steps
    ]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def correlation_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = x - x.mean(axis=0, keepdims=True)
    centered_y = y - y.mean()
    denominator = np.linalg.norm(centered_x, axis=0) * np.linalg.norm(centered_y)
    correlation = np.divide(
        centered_x.T @ centered_y,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float64),
        where=denominator > 0,
    )
    return np.argsort(np.abs(correlation))[::-1]


@torch.inference_mode()
def predict_intervals(
    model: ExactWindowFingerDecoder,
    features: torch.Tensor,
    intervals: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start, stop in intervals:
        indices.append(np.arange(start, stop, dtype=np.int64))
        predictions.append(
            model.decode(features[None, start:stop])[0].float().cpu().numpy()
        )
    return np.concatenate(indices), np.concatenate(predictions)


def plot_validation(
    path: Path,
    indices: np.ndarray,
    raw: np.ndarray,
    cleaned: np.ndarray,
    initialized: np.ndarray,
    trained: np.ndarray,
) -> None:
    order = np.argsort(indices)
    index = indices[order]
    gap = np.r_[False, np.diff(index) > 1]
    x = index.astype(np.float64) / 25.0
    series = [raw[order], cleaned[order], initialized[order], trained[order]]
    for values in series:
        values[gap] = np.nan
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(x, series[0], color="black", linewidth=1.0, label="raw glove")
    axes[0].plot(x, series[2], linewidth=0.9, label="LARS initialization")
    axes[0].plot(x, series[3], linewidth=1.0, label="trained LSTM")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].set_ylabel("raw coordinate")
    axes[1].plot(x, series[1], color="black", linewidth=1.0, label="cleaned target")
    axes[1].plot(x, series[3], linewidth=1.0, label="trained LSTM")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_ylabel("movement")
    axes[1].set_xlabel("original training time (s); gaps reset recurrent state")
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
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1"))
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--sequence-steps", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--near-zero-std", type=float, default=1e-3)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    prepared = args.prepared_root / f"sub{args.subject}"
    fold_path = args.fold_root / f"sub{args.subject}" / args.finger / "folds.json"
    definition = json.loads(fold_path.read_text())
    fold = definition["folds"][args.fold]
    training_intervals = fold["training_intervals_after_purge"]
    validation_intervals = fold["validation_intervals"]
    training_indices = indices_from_intervals(training_intervals)
    validation_indices = indices_from_intervals(validation_intervals)
    row_count = int(definition["training_rows"])

    feature_path = (
        args.feature_root / f"sub{args.subject}" / "train_initialized_window_features.npy"
    )
    features_all = np.load(feature_path, mmap_mode="r")[:row_count]
    finger = list(FINGER_NAMES).index(args.finger)
    target_all = np.load(prepared / f"train_glove_{TARGETS[args.subject]}.npy")[24 : 24 + row_count, finger]
    raw_all = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + row_count, finger]

    cache = (
        args.selection_cache_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}.npz"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache.exists():
            saved = np.load(cache)
            selected_source = saved["selected_source"]
            feature_mean = saved["feature_mean"]
            feature_scale = saved["feature_scale"]
            coefficients = saved["coefficients"]
            lars_intercept = float(saved["intercept"])
            lars_alpha = float(saved["alpha"])
            print(f"loaded_lars_cache={cache}", flush=True)
        else:
            prescreen = correlation_order(
                np.asarray(features_all[training_indices]), target_all[training_indices]
            )[: args.max_features]
            scaler = StandardScaler()
            selected_training = scaler.fit_transform(
                np.asarray(features_all[training_indices][:, prescreen])
            ).astype(np.float64, copy=False)
            global_to_training = np.full(row_count, -1, dtype=np.int64)
            global_to_training[training_indices] = np.arange(training_indices.size)
            cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
            for inner_fold in range(3):
                if inner_fold == args.fold:
                    continue
                inner_validation = indices_from_intervals(
                    definition["folds"][inner_fold]["validation_intervals"]
                )
                inner_training = indices_from_intervals(
                    definition["folds"][inner_fold]["training_intervals_after_purge"]
                )
                inner_validation_positions = global_to_training[inner_validation]
                inner_validation_positions = inner_validation_positions[inner_validation_positions >= 0]
                inner_training_positions = global_to_training[inner_training]
                inner_training_positions = inner_training_positions[inner_training_positions >= 0]
                if inner_validation_positions.size and inner_training_positions.size:
                    cv_splits.append((inner_training_positions, inner_validation_positions))
            if len(cv_splits) != 2:
                raise RuntimeError("event fold did not yield two valid inner LARS splits")
            lars = LassoLarsCV(cv=cv_splits, max_iter=500, n_jobs=1)
            lars.fit(selected_training, target_all[training_indices])
            nonzero = np.flatnonzero(lars.coef_)
            if nonzero.size == 0:
                raise RuntimeError("LARS selected no features")
            selected_source = prescreen[nonzero]
            feature_mean = scaler.mean_[nonzero].astype(np.float32)
            feature_scale = scaler.scale_[nonzero].astype(np.float32)
            coefficients = lars.coef_[nonzero].astype(np.float32)
            lars_intercept = float(lars.intercept_)
            lars_alpha = float(lars.alpha_)
            temporary = cache.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                selected_source=selected_source,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                coefficients=coefficients,
                intercept=np.asarray(lars_intercept),
                alpha=np.asarray(lars_alpha),
            )
            temporary.replace(cache)
            print(f"saved_lars_cache={cache}", flush=True)
    nonzero = np.arange(selected_source.size)
    features = torch.from_numpy(
        np.asarray(features_all[:, selected_source], dtype=np.float32)
    ).to(device)

    model = ExactWindowFingerDecoder(
        input_channels=1,
        component_count=1,
        selected_indices=np.arange(nonzero.size),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        hidden_size=args.hidden_size,
        frontend="wavelet",
        head_initialization="lars_linear_regime",
        output_activation="linear",
    ).to(device)
    model.initialize_lars_linear_regime(
        coefficients,
        lars_intercept,
        near_zero_std=args.near_zero_std,
    )
    validation_order, initialized = predict_intervals(
        model, features, validation_intervals
    )
    initialized_pcc = pearson(initialized, raw_all[validation_order])
    lars_validation = (
        (np.asarray(features_all[validation_order][:, selected_source]) - feature_mean)
        / feature_scale
    ) @ coefficients + lars_intercept
    initialization_fidelity = pearson(initialized, lars_validation)

    parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    decode = model.decode
    if args.compile:
        decode = torch.compile(decode, mode="reduce-overhead")
    starts = starts_from_intervals(
        training_intervals, args.sequence_steps, args.sequence_stride
    )
    if starts.size == 0:
        raise RuntimeError("no complete training sequences remain after purge")
    rng = np.random.default_rng(args.seed)
    best_score = initialized_pcc
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_prediction = initialized.copy()
    history: list[dict[str, float]] = [
        {"epoch": 0, "validation_raw_pcc": initialized_pcc}
    ]
    stale = 0
    offsets = np.arange(args.sequence_steps, dtype=np.int64)
    target_tensor = torch.from_numpy(np.asarray(target_all, dtype=np.float32)).to(device)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(starts)
        model.train()
        losses: list[float] = []
        for begin in range(0, starts.size, args.batch_size):
            chosen = starts[begin : begin + args.batch_size]
            indices = torch.as_tensor(chosen[:, None] + offsets[None], device=device)
            prediction = decode(features[indices])
            loss = (prediction - target_tensor[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % args.validation_interval == 0:
            order, prediction = predict_intervals(model, features, validation_intervals)
            score = pearson(prediction, raw_all[order])
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(losses)),
                    "validation_raw_pcc": score,
                }
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "validation_raw_pcc": score,
                        "best": best_score,
                    }
                ),
                flush=True,
            )
            if score > best_score + 1e-4:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_prediction = prediction.copy()
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

    output = (
        args.output_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}"
        / f"seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_indices.npy", validation_order)
    np.save(output / "validation_prediction.npy", best_prediction)
    np.save(output / "validation_raw_target.npy", raw_all[validation_order])
    np.save(output / "validation_cleaned_target.npy", target_all[validation_order])
    plot_validation(
        output / "validation_trajectory.png",
        validation_order,
        raw_all[validation_order].copy(),
        target_all[validation_order].copy(),
        initialized.copy(),
        best_prediction.copy(),
    )
    summary = {
        "protocol": "per-finger event-grouped outer fold; two event-grouped inner LARS splits",
        "subject": args.subject,
        "finger": args.finger,
        "fold": args.fold,
        "seed": args.seed,
        "released_test_touched": False,
        "official_final_validation_touched": False,
        "training_rows_after_purge": int(training_indices.size),
        "validation_rows": int(validation_indices.size),
        "selected_feature_count": int(nonzero.size),
        "selected_source_indices": selected_source.tolist(),
        "lars_alpha": lars_alpha,
        "lars_coefficients": coefficients.tolist(),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "lars_intercept": lars_intercept,
        "initialization_fidelity_pcc": initialization_fidelity,
        "initialized_validation_raw_pcc": initialized_pcc,
        "best_validation_raw_pcc": best_score,
        "best_epoch": best_epoch,
        "history": history,
    }
    torch.save(
        {
            "model_state_dict": best_state,
            "selected_source_indices": selected_source,
            "summary": summary,
        },
        output / "model.pt",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"saved": str(output), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
