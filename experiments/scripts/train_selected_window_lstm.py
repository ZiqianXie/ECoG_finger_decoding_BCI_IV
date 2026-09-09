#!/usr/bin/env python3
"""Train one small LSTM per finger on validation-selected exact-window features."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark_ridge_target_variants import ridge_fit
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator else 0.0


class FingerWindowLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        self.direct = nn.Linear(feature_count, 1)
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.temporal = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        standardized = (x - self.feature_mean) / self.feature_scale
        direct = self.direct(standardized)
        recurrent, _ = self.lstm(standardized)
        return torch.relu(direct + self.temporal(recurrent)).squeeze(-1)


@torch.inference_mode()
def predict(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
    return model(tensor).squeeze(0).float().cpu().numpy()


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean one-minus-correlation across sequences in a minibatch."""
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = torch.sum(prediction * target, dim=-1)
    denominator = torch.linalg.vector_norm(prediction, dim=-1) * torch.linalg.vector_norm(
        target, dim=-1
    )
    return 1.0 - torch.mean(numerator / denominator.clamp_min(1.0e-8))


def training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    movement_threshold: float,
    movement_loss_weight: float,
    correlation_loss_weight: float,
    derivative_correlation_weight: float,
) -> torch.Tensor:
    weights = torch.where(
        target >= movement_threshold,
        torch.as_tensor(movement_loss_weight, device=target.device),
        torch.ones((), device=target.device),
    )
    loss = torch.sum(weights * (prediction - target).square()) / weights.sum()
    if correlation_loss_weight:
        loss = loss + correlation_loss_weight * correlation_loss(prediction, target)
    if derivative_correlation_weight:
        loss = loss + derivative_correlation_weight * correlation_loss(
            torch.diff(prediction, dim=-1), torch.diff(target, dim=-1)
        )
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_sklearn_v1"))
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/selected_window_lstm_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--finger-targets", nargs=5,
                        help="optional thumb/index/middle/ring/little target names")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--direct-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--movement-loss-weight", type=float, default=1.0)
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    parser.add_argument("--derivative-correlation-weight", type=float, default=0.0)
    parser.add_argument(
        "--only-fingers",
        nargs="+",
        choices=FINGER_NAMES,
        help="train only these fingers; omitted output columns are NaN",
    )
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--refit-full", action="store_true",
                        help="after validation selection, retrain for the chosen epoch count on all 400 s")
    parser.add_argument(
        "--refit-epochs",
        type=int,
        help="fixed full-data refit duration; defaults to the validation-selected epoch",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prepared = args.prepared_root / f"sub{args.subject}"
    features = args.feature_root / f"sub{args.subject}"
    selection = args.selection_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    split = int(json.loads((prepared / "metadata.json").read_text())["target_fit_samples_25hz"])
    offset = args.history - 1
    train_all = np.load(features / "train_initialized_window_features.npy", mmap_mode="r")
    test_all = np.load(features / "test_initialized_window_features.npy", mmap_mode="r")
    target_names = list(args.finger_targets or [args.target] * 5)
    target_cache = {
        name: np.load(prepared / f"train_glove_{name}.npy")[offset:]
        for name in set(target_names)
    }
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[offset:]
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    train_count = split - offset
    selection_summary = json.loads((selection / "summary.json").read_text())
    device = torch.device(args.device)
    validation_prediction = np.full_like(raw_train[train_count:], np.nan, dtype=np.float32)
    test_prediction = np.full_like(raw_test, np.nan, dtype=np.float32)
    report: dict[str, object] = {}
    selected_fingers = set(args.only_fingers or FINGER_NAMES)

    for finger, name in enumerate(FINGER_NAMES):
        if name not in selected_fingers:
            report[name] = {"status": "skipped"}
            continue
        target = target_cache[target_names[finger]]
        indices = np.asarray(
            selection_summary["per_finger"][name]["selected_source_indices"], dtype=np.int64
        )
        if indices.size == 0:
            raise ValueError(f"LARS selected no features for {name}")
        x_train = np.asarray(train_all[:train_count, indices], dtype=np.float32)
        x_validation = np.asarray(train_all[train_count:, indices], dtype=np.float32)
        x_test = np.asarray(test_all[:, indices], dtype=np.float32)
        y_train = np.asarray(target[:train_count, finger], dtype=np.float32)
        y_validation_raw = raw_train[train_count:, finger]
        mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        ridge = ridge_fit(x_train, y_train, 1.0e-3, device)
        ridge_mean, ridge_scale, target_mean, ridge_weight = ridge
        model = FingerWindowLSTM(indices.size, args.hidden_size).to(device)
        with torch.no_grad():
            model.feature_mean.copy_(torch.from_numpy(mean).to(device))
            model.feature_scale.copy_(torch.from_numpy(scale).to(device))
            model.direct.weight.copy_(torch.from_numpy(ridge_weight[None]).to(device))
            model.direct.bias.fill_(target_mean)
        train_model: nn.Module = model
        if args.compile:
            train_model = torch.compile(model, mode="reduce-overhead")
        optimizer = torch.optim.AdamW(
            [
                {"params": model.direct.parameters(), "lr": args.direct_learning_rate},
                {"params": list(model.lstm.parameters()) + list(model.temporal.parameters()), "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        starts = np.arange(0, train_count - args.sequence_steps + 1, args.sequence_stride)
        initial_estimate = predict(model, x_validation, device)
        best_score = pearson(initial_estimate, y_validation_raw)
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale = 0
        history = [
            {
                "epoch": 0,
                "loss": None,
                "validation_raw_r": best_score,
            }
        ]
        print(f"finger={name} epoch=0 val_r={best_score:.4f}", flush=True)
        for epoch in range(1, args.epochs + 1):
            np.random.shuffle(starts)
            model.train()
            losses = []
            for begin in range(0, starts.size, args.batch_size):
                batch_starts = starts[begin : begin + args.batch_size]
                xb = np.stack([x_train[s : s + args.sequence_steps] for s in batch_starts])
                yb = np.stack([y_train[s : s + args.sequence_steps] for s in batch_starts])
                prediction = train_model(torch.from_numpy(xb).to(device))
                observed = torch.from_numpy(yb).to(device)
                loss = training_loss(
                    prediction,
                    observed,
                    args.movement_threshold,
                    args.movement_loss_weight,
                    args.correlation_loss_weight,
                    args.derivative_correlation_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch == 1 or epoch % args.validation_interval == 0:
                estimate = predict(model, x_validation, device)
                score = pearson(estimate, y_validation_raw)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_raw_r": score})
                print(f"finger={name} epoch={epoch} val_r={score:.4f} best={best_score:.4f}", flush=True)
                if score > best_score + 1.0e-4:
                    best_score = score
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
        model.load_state_dict(best_state)
        validation_prediction[:, finger] = predict(model, x_validation, device)
        final_state = best_state
        if args.refit_full:
            refit_epochs = args.refit_epochs if args.refit_epochs is not None else best_epoch
            if refit_epochs < 0:
                raise ValueError("--refit-epochs must be non-negative")
            x_full = np.asarray(train_all[:, indices], dtype=np.float32)
            y_full = np.asarray(target[:, finger], dtype=np.float32)
            full_mean = x_full.mean(axis=0, dtype=np.float64).astype(np.float32)
            full_scale = x_full.std(axis=0, dtype=np.float64).astype(np.float32)
            full_scale[full_scale < 1.0e-6] = 1.0
            full_ridge = ridge_fit(x_full, y_full, 1.0e-3, device)
            _, _, full_target_mean, full_ridge_weight = full_ridge
            torch.manual_seed(args.seed)
            refit = FingerWindowLSTM(indices.size, args.hidden_size).to(device)
            with torch.no_grad():
                refit.feature_mean.copy_(torch.from_numpy(full_mean).to(device))
                refit.feature_scale.copy_(torch.from_numpy(full_scale).to(device))
                refit.direct.weight.copy_(torch.from_numpy(full_ridge_weight[None]).to(device))
                refit.direct.bias.fill_(full_target_mean)
            refit_train: nn.Module = refit
            if args.compile:
                refit_train = torch.compile(refit, mode="reduce-overhead")
            refit_optimizer = torch.optim.AdamW(
                [
                    {"params": refit.direct.parameters(), "lr": args.direct_learning_rate},
                    {"params": list(refit.lstm.parameters()) + list(refit.temporal.parameters()), "lr": args.learning_rate},
                ],
                weight_decay=args.weight_decay,
            )
            full_starts = np.arange(0, x_full.shape[0] - args.sequence_steps + 1, args.sequence_stride)
            for _ in range(refit_epochs):
                np.random.shuffle(full_starts)
                refit.train()
                for begin in range(0, full_starts.size, args.batch_size):
                    batch_starts = full_starts[begin : begin + args.batch_size]
                    xb = np.stack([x_full[s : s + args.sequence_steps] for s in batch_starts])
                    yb = np.stack([y_full[s : s + args.sequence_steps] for s in batch_starts])
                    estimate = refit_train(torch.from_numpy(xb).to(device))
                    observed = torch.from_numpy(yb).to(device)
                    loss = training_loss(
                        estimate,
                        observed,
                        args.movement_threshold,
                        args.movement_loss_weight,
                        args.correlation_loss_weight,
                        args.derivative_correlation_weight,
                    )
                    refit_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(refit.parameters(), 1.0)
                    refit_optimizer.step()
            model = refit
            final_state = copy.deepcopy(refit.state_dict())
        test_prediction[:, finger] = predict(model, x_test, device)
        torch.save({"model_state_dict": final_state, "feature_indices": indices}, output / f"{name}.pt")
        report[name] = {
            "target": target_names[finger],
            "feature_count": int(indices.size),
            "best_epoch": best_epoch,
            "best_validation_raw_r": best_score,
            "refit_on_all_training_data": args.refit_full,
            "refit_epochs": (
                args.refit_epochs if args.refit_full and args.refit_epochs is not None
                else best_epoch if args.refit_full else 0
            ),
            "history": history,
        }
    metrics = trajectory_metrics(test_prediction, raw_test)
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    summary = {
        "subject": args.subject,
        "method": (
            f"five independent {args.hidden_size}-unit LSTMs on "
            "LARS-selected exact-window features"
        ),
        "target": args.target,
        "finger_targets": dict(zip(FINGER_NAMES, target_names)),
        "feature_root": str(args.feature_root),
        "selection_root": str(args.selection_root),
        "architecture": {
            "hidden_size": args.hidden_size,
            "history_bins": args.history,
            "sequence_steps": args.sequence_steps,
            "sequence_stride": args.sequence_stride,
            "batch_size": args.batch_size,
        },
        "training_objective": {
            "movement_threshold": args.movement_threshold,
            "movement_loss_weight": args.movement_loss_weight,
            "correlation_loss_weight": args.correlation_loss_weight,
            "derivative_correlation_weight": args.derivative_correlation_weight,
        },
        "per_finger": report,
        "validation_raw_metrics": trajectory_metrics(validation_prediction, raw_train[train_count:]),
        "test_raw_metrics": metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
