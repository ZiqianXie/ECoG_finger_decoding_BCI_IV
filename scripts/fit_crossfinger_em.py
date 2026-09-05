#!/usr/bin/env python3
"""Fit subject-specific latent finger trajectories and a sparse cross-talk map.

This is a generalized-EM/MAP diagnostic.  The neural evidence is reconstructed
entirely from out-of-fold predictions, and no official validation or released
test target is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

from diagnose_cross_finger_oof import reconstruct_oof
from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def calibrate_neural_evidence(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    calibrated = np.empty_like(prediction, dtype=np.float64)
    records: list[dict[str, float]] = []
    for finger in range(prediction.shape[1]):
        x = np.asarray(prediction[:, finger], dtype=np.float64)
        y = np.asarray(target[:, finger], dtype=np.float64)
        variance = float(np.sum(np.square(x - x.mean())))
        slope = max(0.0, float(np.sum((x - x.mean()) * (y - y.mean()))) / max(variance, 1e-12))
        intercept = max(0.0, float(y.mean() - slope * x.mean()))
        calibrated[:, finger] = np.clip(intercept + slope * x, 0.0, 2.5)
        records.append({"slope": slope, "intercept": intercept})
    return calibrated, records


def initialize_mixing(
    observed: np.ndarray, maximum_cross_talk: float
) -> np.ndarray:
    """Estimate a conservative mixing seed from glove-dominant movement bins."""
    fingers = observed.shape[1]
    mixing = np.eye(fingers, dtype=np.float64)
    dominant = np.argmax(observed, axis=1)
    for latent_finger in range(fingers):
        moving = observed[:, latent_finger] >= max(
            0.20, float(np.quantile(observed[:, latent_finger], 0.75))
        )
        selected = moving & (dominant == latent_finger)
        x = observed[selected, latent_finger]
        denominator = float(x @ x)
        if denominator <= 1e-12:
            continue
        for observed_finger in range(fingers):
            if observed_finger == latent_finger:
                continue
            coefficient = float(x @ observed[selected, observed_finger]) / denominator
            mixing[observed_finger, latent_finger] = np.clip(
                coefficient, 0.0, maximum_cross_talk
            )
    return mixing


def temporal_gradient(values: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(values)
    difference = values[1:] - values[:-1]
    gradient[:-1] -= difference
    gradient[1:] += difference
    return gradient


def objective(
    observed: np.ndarray,
    latent: np.ndarray,
    mixing: np.ndarray,
    evidence: np.ndarray,
    neural_weight: float,
    sparsity: float,
    temporal_weight: float,
) -> dict[str, float]:
    reconstruction = 0.5 * float(np.mean(np.square(latent @ mixing.T - observed)))
    neural = 0.5 * neural_weight * float(np.mean(np.square(latent - evidence)))
    sparse = sparsity * float(np.mean(latent))
    temporal = 0.5 * temporal_weight * float(np.mean(np.square(np.diff(latent, axis=0))))
    return {
        "total": reconstruction + neural + sparse + temporal,
        "reconstruction": reconstruction,
        "neural": neural,
        "sparsity": sparse,
        "temporal": temporal,
    }


def e_step(
    observed: np.ndarray,
    initial: np.ndarray,
    mixing: np.ndarray,
    evidence: np.ndarray,
    neural_weight: float,
    sparsity: float,
    temporal_weight: float,
    steps: int,
) -> np.ndarray:
    latent = initial.copy()
    spectral_norm = float(np.linalg.norm(mixing, ord=2) ** 2)
    step_size = 0.9 / max(spectral_norm + neural_weight + 4.0 * temporal_weight, 1e-6)
    for _ in range(steps):
        gradient = (latent @ mixing.T - observed) @ mixing
        gradient += neural_weight * (latent - evidence)
        gradient += temporal_weight * temporal_gradient(latent)
        latent = np.maximum(latent - step_size * gradient - step_size * sparsity, 0.0)
    return latent


def m_step(
    observed: np.ndarray,
    latent: np.ndarray,
    cross_l1: float,
    maximum_cross_talk: float,
) -> np.ndarray:
    fingers = observed.shape[1]
    mixing = np.eye(fingers, dtype=np.float64)
    for observed_finger in range(fingers):
        other = [finger for finger in range(fingers) if finger != observed_finger]
        residual = observed[:, observed_finger] - latent[:, observed_finger]
        estimator = Lasso(
            alpha=cross_l1,
            fit_intercept=False,
            positive=True,
            max_iter=20_000,
            tol=1e-8,
        )
        estimator.fit(latent[:, other], residual)
        mixing[observed_finger, other] = np.clip(
            estimator.coef_, 0.0, maximum_cross_talk
        )
    return mixing


def render(
    destination: Path,
    observed: np.ndarray,
    latent: np.ndarray,
    mixing: np.ndarray,
    evidence: np.ndarray,
    subject: int,
) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(mixing, cmap="Blues", vmin=0.0, vmax=max(1.0, float(mixing.max())))
    axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(5), FINGER_NAMES)
    axis.set_xlabel("latent intended finger")
    axis.set_ylabel("observed glove channel")
    axis.set_title(f"Subject {subject}: fitted near-identity cross-talk map")
    for row in range(5):
        for column in range(5):
            axis.text(column, row, f"{mixing[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(destination / "crosstalk_matrix.png", dpi=180)
    plt.close(figure)

    time = np.arange(observed.shape[0]) / 25.0
    figure, axes = plt.subplots(5, 1, figsize=(17, 13), sharex=True)
    for finger, axis in enumerate(axes):
        axis.plot(time, observed[:, finger], color="black", linewidth=0.75, label="baseline-corrected glove")
        axis.plot(time, latent[:, finger], color="#2563eb", linewidth=0.8, label="EM latent intent")
        axis.plot(time, evidence[:, finger], color="#d97706", linewidth=0.55, alpha=0.65, label="calibrated OOF neural evidence")
        axis.set_ylabel(FINGER_NAMES[finger])
        if finger == 0:
            axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("training-partition time (s)")
    figure.suptitle(f"Subject {subject}: generalized-EM latent movement diagnostic")
    figure.tight_layout()
    figure.savefig(destination / "latent_trajectory_diagnosis.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--prediction-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_nested_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/crossfinger_em_v1"))
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--e-steps", type=int, default=150)
    parser.add_argument("--neural-weight", type=float, default=0.35)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--temporal-weight", type=float, default=0.20)
    parser.add_argument("--cross-l1", type=float, default=0.0002)
    parser.add_argument("--maximum-cross-talk", type=float, default=0.75)
    args = parser.parse_args()

    prediction = reconstruct_oof(
        args.prediction_root, args.fold_root, args.subject, list(args.seeds)
    )
    rows = prediction.shape[0]
    prepared = args.prepared_root / f"sub{args.subject}"
    cleaned = np.asarray(
        np.load(prepared / f"train_glove_{TARGETS[args.subject]}.npy")[24 : 24 + rows],
        dtype=np.float64,
    )
    scale = np.maximum(np.quantile(np.maximum(cleaned, 0.0), 0.99, axis=0), 1e-4)
    observed = np.clip(cleaned / scale, 0.0, 2.5)
    evidence, calibration = calibrate_neural_evidence(prediction, observed)
    mixing = initialize_mixing(observed, args.maximum_cross_talk)
    initial_mixing = mixing.copy()
    latent = np.maximum(
        (observed @ np.linalg.pinv(mixing).T + args.neural_weight * evidence)
        / (1.0 + args.neural_weight),
        0.0,
    )
    history: list[dict[str, object]] = []
    for iteration in range(1, args.iterations + 1):
        latent = e_step(
            observed,
            latent,
            mixing,
            evidence,
            args.neural_weight,
            args.sparsity,
            args.temporal_weight,
            args.e_steps,
        )
        mixing = m_step(
            observed, latent, args.cross_l1, args.maximum_cross_talk
        )
        parts = objective(
            observed,
            latent,
            mixing,
            evidence,
            args.neural_weight,
            args.sparsity,
            args.temporal_weight,
        )
        record: dict[str, object] = {
            "iteration": iteration,
            **parts,
            "maximum_off_diagonal": float((mixing - np.eye(5)).max()),
        }
        history.append(record)
        if iteration == 1 or iteration % 5 == 0 or iteration == args.iterations:
            print(json.dumps(record), flush=True)

    destination = args.output_root / f"sub{args.subject}"
    destination.mkdir(parents=True, exist_ok=True)
    latent_physical = latent * scale
    physical_mixing = mixing * scale[:, None] / scale[None, :]
    np.save(destination / "latent_targets.npy", latent_physical.astype(np.float32))
    np.save(destination / "normalized_latent_targets.npy", latent.astype(np.float32))
    np.save(destination / "normalized_crosstalk_matrix.npy", mixing.astype(np.float32))
    np.save(destination / "physical_crosstalk_matrix.npy", physical_mixing.astype(np.float32))
    render(destination, observed, latent, mixing, evidence, args.subject)
    reconstruction = latent @ mixing.T
    report = {
        "protocol": "subject-specific generalized-EM/MAP diagnostic using OOF neural evidence",
        "subject": args.subject,
        "official_final_validation_touched": False,
        "released_test_touched": False,
        "parameters": {
            "iterations": args.iterations,
            "e_steps": args.e_steps,
            "neural_weight": args.neural_weight,
            "sparsity": args.sparsity,
            "temporal_weight": args.temporal_weight,
            "cross_l1": args.cross_l1,
            "maximum_cross_talk": args.maximum_cross_talk,
        },
        "neural_calibration": calibration,
        "normalized_crosstalk_matrix": mixing.tolist(),
        "initial_normalized_crosstalk_matrix": initial_mixing.tolist(),
        "physical_crosstalk_matrix": physical_mixing.tolist(),
        "normalized_reconstruction_rmse": float(np.sqrt(np.mean(np.square(reconstruction - observed)))),
        "latent_to_observed_pcc": [
            float(np.corrcoef(latent[:, finger], observed[:, finger])[0, 1])
            for finger in range(5)
        ],
        "history": history,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"saved": str(destination), **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
