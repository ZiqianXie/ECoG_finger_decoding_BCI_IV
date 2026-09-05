#!/usr/bin/env python3
"""Nested event-fold evaluation for the LARS-initialized nonlinear LSTM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

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
) -> ExactWindowFingerDecoder:
    model = ExactWindowFingerDecoder(
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
    return model


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
) -> list[float]:
    if epochs <= 0:
        return []
    parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    decode = model.decode
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
            prediction = decode(features[indices])
            batch_target = target[indices]
            if loss_name == "mse":
                loss = (prediction - batch_target).square().mean()
            else:
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
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--movement-weight", type=float, default=4.0)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--correlation-weight", type=float, default=0.1)
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
        parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        decode = model.decode
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
                prediction = decode(features[indices])
                batch_target = target_tensor[indices]
                if args.loss == "mse":
                    loss = (prediction - batch_target).square().mean()
                else:
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
        "loss_weights": {
            "movement_threshold": args.movement_threshold,
            "movement_weight": args.movement_weight,
            "velocity_weight": args.velocity_weight,
            "correlation_weight": args.correlation_weight,
        },
        "inner_epoch_selection": inner_records,
        "selected_epoch": final_epoch,
        "initialized_outer_raw_pcc": initialization_score,
        "outer_raw_pcc": outer_score,
        "outer_cleaned_target_pcc": outer_cleaned_score,
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
