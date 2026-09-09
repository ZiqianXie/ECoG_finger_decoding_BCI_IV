#!/usr/bin/env python3
"""Blocked-fold, finger-specific movement-state and amplitude decoder.

The decoder is a two-part hurdle model: a Bernoulli movement state and a
conditional positive amplitude. Architecture selection uses out-of-fold
predictions inside the fit partition. The untouched chronological validation
partition is used only for lag selection and final candidate comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ecog_decoding.training import FINGER_NAMES
try:
    from benchmark_fixed_lars import named_subset_metrics
    from crossvalidate_selected_ridge import purged_block_folds
    from optimize_prediction_lag import shifted
except ModuleNotFoundError:  # imported as scripts.train_blocked_hurdle_gru in tests
    from scripts.benchmark_fixed_lars import named_subset_metrics
    from scripts.crossvalidate_selected_ridge import purged_block_folds
    from scripts.optimize_prediction_lag import shifted


@dataclass(frozen=True)
class Profile:
    name: str
    hidden_size: int
    dropout: float


PROFILES = (
    Profile("compact", hidden_size=24, dropout=0.10),
    Profile("wide", hidden_size=48, dropout=0.20),
)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64) - np.mean(left)
    right = np.asarray(right, dtype=np.float64) - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def state_f1(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    predicted = prediction >= threshold
    observed = target >= threshold
    true_positive = np.sum(predicted & observed)
    false_positive = np.sum(predicted & ~observed)
    false_negative = np.sum(~predicted & observed)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return float(2.0 * precision * recall / max(precision + recall, 1.0e-12))


def morphology(
    prediction: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, float]:
    moving = target >= threshold
    rest_rms = float(np.sqrt(np.mean(np.square(prediction[~moving]))))
    movement_rms = float(np.sqrt(np.mean(np.square(target[moving]))))
    peak_ratio = float(
        np.quantile(prediction[moving], 0.95)
        / max(float(np.quantile(target[moving], 0.95)), 1.0e-8)
    )
    pcc = pearson(prediction, target)
    derivative = pearson(np.diff(prediction), np.diff(target))
    f1 = state_f1(prediction, target, threshold)
    score = (
        0.45 * pcc
        + 0.20 * derivative
        + 0.20 * f1
        - 0.10 * rest_rms / max(movement_rms, 1.0e-8)
        - 0.05 * abs(np.log(max(peak_ratio, 1.0e-4)))
    )
    return {
        "score": score,
        "pcc_cleaned": pcc,
        "derivative_pcc": derivative,
        "state_f1": f1,
        "rest_rms": rest_rms,
        "movement_peak_ratio": peak_ratio,
    }


class HurdleGRU(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        dropout: float,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        amplitude_scale: float,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean))
        self.register_buffer("feature_scale", torch.as_tensor(feature_scale))
        self.register_buffer("amplitude_scale", torch.tensor(float(amplitude_scale)))
        self.input = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.state_head = nn.Linear(hidden_size, 1)
        self.amplitude_head = nn.Linear(hidden_size, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standardized = (values - self.feature_mean) / self.feature_scale
        encoded = self.input(standardized)
        encoded, _ = self.gru(encoded)
        encoded = self.dropout(encoded)
        state_logit = self.state_head(encoded).squeeze(-1)
        amplitude = F.softplus(self.amplitude_head(encoded).squeeze(-1))
        return state_logit, amplitude * self.amplitude_scale


def hurdle_loss(
    state_logit: torch.Tensor,
    amplitude: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    amplitude_scale: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    state = (target >= threshold).to(target.dtype)
    state_loss = F.binary_cross_entropy_with_logits(
        state_logit, state, pos_weight=positive_weight
    )
    moving = state.bool()
    if moving.any():
        amplitude_loss = F.mse_loss(
            amplitude[moving] / amplitude_scale,
            target[moving] / amplitude_scale,
        )
    else:
        amplitude_loss = amplitude.sum() * 0.0
    # Bernoulli state likelihood plus a unit-variance conditional Gaussian
    # amplitude likelihood on movement bins.
    return state_loss + 0.5 * amplitude_loss


def contiguous_starts(
    indices: np.ndarray, sequence_steps: int, stride: int
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    runs = np.split(indices, breaks)
    starts: list[int] = []
    for run in runs:
        if run.size >= sequence_steps:
            starts.extend(range(int(run[0]), int(run[-1]) - sequence_steps + 2, stride))
    if not starts:
        raise ValueError("no contiguous training sequences remain")
    return np.asarray(starts, dtype=np.int64)


@torch.inference_mode()
def predict(
    model: HurdleGRU, values: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0).to(device)
    state_logit, amplitude = model(tensor)
    probability = torch.sigmoid(state_logit)
    prediction = probability * amplitude
    return (
        prediction.squeeze(0).float().cpu().numpy(),
        probability.squeeze(0).float().cpu().numpy(),
        amplitude.squeeze(0).float().cpu().numpy(),
    )


def fit_fold(
    values: np.ndarray,
    target: np.ndarray,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    profile: Profile,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[HurdleGRU, dict[str, object]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    fit_values = values[training_indices]
    mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    moving = target[training_indices] >= args.movement_threshold
    amplitude_scale = float(np.quantile(target[training_indices][moving], 0.95))
    amplitude_scale = max(amplitude_scale, 1.0e-3)
    positive_weight = float(np.sum(~moving) / max(np.sum(moving), 1))
    positive_weight = float(np.clip(positive_weight, 1.0, 12.0))
    model = HurdleGRU(
        values.shape[1],
        profile.hidden_size,
        profile.dropout,
        mean,
        scale,
        amplitude_scale,
    ).to(device)
    train_model: nn.Module = model
    if args.compile:
        train_model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    starts = contiguous_starts(
        training_indices, args.sequence_steps, args.sequence_stride
    )
    positive_weight_tensor = torch.tensor(positive_weight, device=device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(starts)
        model.train()
        losses = []
        for begin in range(0, starts.size, args.batch_size):
            batch_starts = starts[begin : begin + args.batch_size]
            x_batch = np.stack(
                [values[start : start + args.sequence_steps] for start in batch_starts]
            )
            y_batch = np.stack(
                [target[start : start + args.sequence_steps] for start in batch_starts]
            )
            state_logit, amplitude = train_model(
                torch.from_numpy(x_batch).to(device)
            )
            observed = torch.from_numpy(y_batch).to(device)
            loss = hurdle_loss(
                state_logit,
                amplitude,
                observed,
                args.movement_threshold,
                model.amplitude_scale,
                positive_weight_tensor,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        if epoch == 1 or epoch % args.validation_interval == 0:
            model.eval()
            x_validation = torch.from_numpy(values[validation_indices]).unsqueeze(0).to(device)
            y_validation = torch.from_numpy(target[validation_indices]).unsqueeze(0).to(device)
            with torch.inference_mode():
                logit, amplitude = model(x_validation)
                validation_loss = float(
                    hurdle_loss(
                        logit,
                        amplitude,
                        y_validation,
                        args.movement_threshold,
                        model.amplitude_scale,
                        positive_weight_tensor,
                    )
                )
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": float(np.mean(losses)),
                    "inner_validation_hurdle_nll": validation_loss,
                }
            )
            if validation_loss < best_loss - args.minimum_improvement:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_inner_validation_hurdle_nll": best_loss,
        "positive_weight": positive_weight,
        "amplitude_scale": amplitude_scale,
        "history": history,
    }


def select_lag(
    prediction: np.ndarray,
    target: np.ndarray,
    maximum: int,
    minimum_gain: float,
) -> tuple[int, dict[str, float]]:
    scores = {lag: pearson(shifted(prediction, lag), target) for lag in range(-maximum, maximum + 1)}
    best = max(scores, key=scores.get)
    if scores[best] < scores[0] + minimum_gain:
        best = 0
    return int(best), {str(key): value for key, value in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/blocked_hurdle_gru_v1")
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--fingers", nargs="+", choices=FINGER_NAMES)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--purge", type=int, default=None)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-improvement", type=float, default=1.0e-4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--max-lag-bins", type=int, default=4)
    parser.add_argument("--lag-minimum-validation-gain", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if str(args.device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")

    prepared = args.prepared_root / f"sub{args.subject}"
    feature_root = args.feature_root / f"sub{args.subject}"
    selection_root = args.selection_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    fit_count = split - offset
    purge = offset if args.purge is None else args.purge
    device = torch.device(args.device)
    features = np.load(
        feature_root / "train_initialized_window_features.npy", mmap_mode="r"
    )
    test_features = np.load(
        feature_root / "test_initialized_window_features.npy", mmap_mode="r"
    )
    cleaned = np.load(prepared / f"train_glove_{args.target}.npy")[offset:]
    raw = np.load(prepared / "train_glove_25hz_raw.npy")[offset:]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    selection = json.loads((selection_root / "summary.json").read_text())
    names = list(args.fingers or FINGER_NAMES)
    chosen = [FINGER_NAMES.index(name) for name in names]
    folds = purged_block_folds(fit_count, args.folds, purge)

    validation_prediction = np.full((features.shape[0] - fit_count, 5), np.nan, dtype=np.float32)
    test_prediction = np.full((test_features.shape[0], 5), np.nan, dtype=np.float32)
    validation_probability = np.full_like(validation_prediction, np.nan)
    test_probability = np.full_like(test_prediction, np.nan)
    validation_amplitude = np.full_like(validation_prediction, np.nan)
    test_amplitude = np.full_like(test_prediction, np.nan)
    report: dict[str, object] = {}

    for finger, name in zip(chosen, names, strict=True):
        indices = np.asarray(
            selection["per_finger"][name]["selected_source_indices"], dtype=np.int64
        )
        x_fit = np.asarray(features[:fit_count, indices], dtype=np.float32)
        x_validation = np.asarray(features[fit_count:, indices], dtype=np.float32)
        x_test = np.asarray(test_features[:, indices], dtype=np.float32)
        y_fit = np.asarray(cleaned[:fit_count, finger], dtype=np.float32)
        profile_outputs: dict[str, object] = {}

        for profile_index, profile in enumerate(PROFILES):
            oof_prediction = np.zeros(fit_count, dtype=np.float32)
            validation_parts = []
            test_parts = []
            probability_parts = []
            test_probability_parts = []
            amplitude_parts = []
            test_amplitude_parts = []
            fold_reports = []
            for fold_index, (training, held_out) in enumerate(folds):
                model, fold_report = fit_fold(
                    x_fit,
                    y_fit,
                    training,
                    held_out,
                    profile,
                    args,
                    args.seed + 1000 * finger + 100 * profile_index + fold_index,
                    device,
                )
                oof_prediction[held_out] = predict(model, x_fit[held_out], device)[0]
                val_pred, val_prob, val_amp = predict(model, x_validation, device)
                tst_pred, tst_prob, tst_amp = predict(model, x_test, device)
                validation_parts.append(val_pred)
                test_parts.append(tst_pred)
                probability_parts.append(val_prob)
                test_probability_parts.append(tst_prob)
                amplitude_parts.append(val_amp)
                test_amplitude_parts.append(tst_amp)
                fold_reports.append(fold_report)
                torch.save(
                    {"model_state_dict": model.state_dict(), "feature_indices": indices},
                    output / f"{name}_{profile.name}_fold{fold_index}.pt",
                )
            oof_metrics = morphology(
                oof_prediction, y_fit, args.movement_threshold
            )
            profile_outputs[profile.name] = {
                "profile": {
                    "hidden_size": profile.hidden_size,
                    "dropout": profile.dropout,
                },
                "oof_morphology": oof_metrics,
                "folds": fold_reports,
                "validation_prediction": np.mean(validation_parts, axis=0),
                "test_prediction": np.mean(test_parts, axis=0),
                "validation_probability": np.mean(probability_parts, axis=0),
                "test_probability": np.mean(test_probability_parts, axis=0),
                "validation_amplitude": np.mean(amplitude_parts, axis=0),
                "test_amplitude": np.mean(test_amplitude_parts, axis=0),
            }
            print(
                f"finger={name} profile={profile.name} "
                f"oof_morphology={oof_metrics['score']:.4f}",
                flush=True,
            )

        winner = max(
            profile_outputs,
            key=lambda key: profile_outputs[key]["oof_morphology"]["score"],
        )
        selected = profile_outputs[winner]
        lag, lag_scores = select_lag(
            selected["validation_prediction"],
            raw[fit_count:, finger],
            args.max_lag_bins,
            args.lag_minimum_validation_gain,
        )
        validation_prediction[:, finger] = shifted(selected["validation_prediction"], lag)
        test_prediction[:, finger] = shifted(selected["test_prediction"], lag)
        validation_probability[:, finger] = shifted(selected["validation_probability"], lag)
        test_probability[:, finger] = shifted(selected["test_probability"], lag)
        validation_amplitude[:, finger] = shifted(selected["validation_amplitude"], lag)
        test_amplitude[:, finger] = shifted(selected["test_amplitude"], lag)
        report[name] = {
            "feature_count": int(indices.size),
            "profile_selected_by_inner_oof": winner,
            "profile_oof": {
                key: value["oof_morphology"] for key, value in profile_outputs.items()
            },
            "selected_lag_bins": lag,
            "selected_lag_seconds": lag / 25.0,
            "outer_validation_lag_scores": lag_scores,
            "folds": selected["folds"],
        }

    validation_raw = raw[fit_count:, chosen]
    test_raw_selected = test_raw[:, chosen]
    result = {
        "subject": args.subject,
        "method": "finger-specific Bernoulli-state/conditional-amplitude hurdle GRU",
        "architecture_selection": "profile selected by purged blocked out-of-fold morphology inside fit partition",
        "outer_validation_use": "lag selection and final candidate comparison only",
        "test_labels_used_for_selection": False,
        "feature_root": str(args.feature_root),
        "selection_root": str(args.selection_root),
        "target": args.target,
        "fingers": report,
        "validation_raw_metrics": named_subset_metrics(
            validation_prediction[:, chosen], validation_raw, names
        ),
        "test_raw_metrics": named_subset_metrics(
            test_prediction[:, chosen], test_raw_selected, names
        ),
    }
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    np.save(output / "validation_state_probability.npy", validation_probability, allow_pickle=False)
    np.save(output / "test_state_probability.npy", test_probability, allow_pickle=False)
    np.save(output / "validation_amplitude.npy", validation_amplitude, allow_pickle=False)
    np.save(output / "test_amplitude.npy", test_amplitude, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["validation_raw_metrics"], indent=2), flush=True)
    print(json.dumps(result["test_raw_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
