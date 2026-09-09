#!/usr/bin/env python3
"""Joint CSP decoder with explicit per-finger movement-state supervision."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmark_ridge_target_variants import lagged
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics
from train_csp_residual_ssm import CSPResidualSSM, correlation_loss, initialize_ridge


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def morphology_score(prediction: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """Validation-only score favoring usable movement shape over raw PCC alone."""
    scores: list[float] = []
    for finger in range(target.shape[1]):
        truth = target[:, finger]
        estimate = prediction[:, finger]
        moving = truth >= threshold
        predicted_moving = estimate >= threshold
        tp = np.sum(moving & predicted_moving)
        fp = np.sum(~moving & predicted_moving)
        fn = np.sum(moving & ~predicted_moving)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        peak_truth = np.quantile(truth[moving], 0.95) if moving.any() else 1.0
        peak_prediction = np.quantile(estimate[moving], 0.95) if moving.any() else 0.0
        peak_ratio = max(float(peak_prediction / max(peak_truth, 1e-6)), 1e-6)
        peak_score = np.exp(-abs(np.log(peak_ratio)))
        rest_rms = np.sqrt(np.mean(estimate[~moving] ** 2)) if (~moving).any() else 0.0
        scale = max(float(peak_truth), 1e-6)
        scores.append(
            0.50 * pearson(estimate, truth)
            + 0.20 * pearson(np.diff(estimate), np.diff(truth))
            + 0.15 * f1
            + 0.10 * peak_score
            - 0.05 * rest_rms / scale
        )
    return float(np.mean(scores))


def state_metrics(probability: np.ndarray, state: np.ndarray) -> dict[str, object]:
    predicted = probability >= 0.5
    per_finger: dict[str, object] = {}
    for finger, name in enumerate(FINGER_NAMES):
        truth = state[:, finger].astype(bool)
        guess = predicted[:, finger]
        tp = int(np.sum(truth & guess))
        fp = int(np.sum(~truth & guess))
        fn = int(np.sum(truth & ~guess))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_finger[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "movement_fraction": float(truth.mean()),
        }
    return per_finger


@torch.inference_mode()
def predict_components(
    model: CSPResidualSSM, energy: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    amplitude, logits = model.forward_components(energy)
    return (
        amplitude.squeeze(0).float().cpu().numpy(),
        torch.sigmoid(logits).squeeze(0).float().cpu().numpy(),
    )


def candidate_predictions(
    amplitude: np.ndarray, probability: np.ndarray, floor: float
) -> dict[str, np.ndarray]:
    return {
        "continuous": amplitude,
        "soft_gate": amplitude * (floor + (1.0 - floor) * probability),
        "hard_gate": amplitude * (probability >= 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/state_aware_csp_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument(
        "--backbone",
        choices=("gru", "lstm", "linear_attention", "mamba", "tcn"),
        default="lstm",
    )
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--correlation-weight", type=float, default=0.0)
    parser.add_argument("--derivative-weight", type=float, default=0.0)
    parser.add_argument("--derivative-correlation-weight", type=float, default=0.0)
    parser.add_argument("--movement-loss-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("raw_pcc", "morphology"), default="raw_pcc")
    parser.add_argument("--gate-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    root = args.prepared_root / f"sub{args.subject}"
    csp_root = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_energy = np.load(csp_root / "train_csp_energy.npy")
    test_energy = np.load(csp_root / "test_csp_energy.npy")
    target = np.load(root / f"train_glove_{args.target}.npy")
    raw_train = np.load(root / "train_glove_25hz_raw.npy")
    raw_test = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    train_x = lagged(train_energy, args.history)[: split - offset]
    train_y = target[offset:split]
    train_state = (train_y >= args.movement_threshold).astype(np.float32)

    model = CSPResidualSSM(
        train_energy.shape[1],
        args.history,
        temporal_backbone=args.backbone,
    ).to(device)
    ridge_audit = initialize_ridge(model, train_x, train_y, args.top_features, device)
    train_tensor = torch.from_numpy(train_energy[:split]).unsqueeze(0).to(device)
    validation_tensor = torch.from_numpy(train_energy[split - offset :]).unsqueeze(0).to(device)
    test_tensor = torch.from_numpy(test_energy).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(train_y).unsqueeze(0).to(device)
    state_tensor = torch.from_numpy(train_state).unsqueeze(0).to(device)
    movement_counts = state_tensor.sum(dim=1)
    pos_weight = ((state_tensor.shape[1] - movement_counts) / movement_counts.clamp_min(1)).squeeze(0)

    initial_validation, initial_validation_state = predict_components(model, validation_tensor)
    initial_test, initial_test_state = predict_components(model, test_tensor)
    if args.selection_metric == "morphology":
        best_score = morphology_score(initial_validation, target[split:], args.movement_threshold)
    else:
        best_score = float(trajectory_metrics(initial_validation, raw_train[split:])["pearson_historical_four"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    executable = torch.compile(model, mode="reduce-overhead")
    direct_ids = {id(parameter) for parameter in model.direct.parameters()}
    residual_parameters = [parameter for parameter in model.parameters() if id(parameter) not in direct_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.direct.parameters(), "lr": 2e-5},
            {"params": residual_parameters, "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    history: list[dict[str, object]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        amplitude, logits = executable.forward_components(train_tensor)
        sample_weight = 1.0 + (args.movement_loss_weight - 1.0) * state_tensor
        level = torch.mean(sample_weight * (amplitude - target_tensor).square())
        state_loss = F.binary_cross_entropy_with_logits(logits, state_tensor, pos_weight=pos_weight)
        corr = correlation_loss(amplitude, target_tensor)
        derivative = F.mse_loss(
            amplitude[:, 1:] - amplitude[:, :-1],
            target_tensor[:, 1:] - target_tensor[:, :-1],
        )
        derivative_correlation = correlation_loss(
            amplitude[:, 1:] - amplitude[:, :-1],
            target_tensor[:, 1:] - target_tensor[:, :-1],
        )
        loss = (
            level
            + args.state_weight * state_loss
            + args.correlation_weight * corr
            + args.derivative_weight * derivative
            + args.derivative_correlation_weight * derivative_correlation
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation_amplitude, validation_probability = predict_components(executable, validation_tensor)
            candidates = candidate_predictions(validation_amplitude, validation_probability, args.gate_floor)
            if args.selection_metric == "morphology":
                candidate_scores = {
                    name: morphology_score(value, target[split:], args.movement_threshold)
                    for name, value in candidates.items()
                }
            else:
                candidate_scores = {
                    name: float(trajectory_metrics(value, raw_train[split:])["pearson_historical_four"])
                    for name, value in candidates.items()
                }
            score = max(candidate_scores.values())
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach()),
                "level": float(level.detach()),
                "state_loss": float(state_loss.detach()),
                "correlation_loss": float(corr.detach()),
                "derivative_loss": float(derivative.detach()),
                "derivative_correlation_loss": float(derivative_correlation.detach()),
                "gradient_norm": float(gradient_norm),
                "validation_candidates": candidate_scores,
            })
            print(f"epoch={epoch} val_best_r4={score:.4f} candidates={candidate_scores}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 20:
                break

    model.load_state_dict(best_state)
    validation_amplitude, validation_probability = predict_components(executable, validation_tensor)
    test_amplitude, test_probability = predict_components(executable, test_tensor)
    validation_candidates = candidate_predictions(validation_amplitude, validation_probability, args.gate_floor)
    test_candidates = candidate_predictions(test_amplitude, test_probability, args.gate_floor)
    selected_modes: dict[str, str] = {}
    validation_prediction = np.zeros_like(validation_amplitude)
    test_prediction = np.zeros_like(test_amplitude)
    for finger, name in enumerate(FINGER_NAMES):
        selected = max(
            validation_candidates,
            key=lambda mode: pearson(validation_candidates[mode][:, finger], raw_train[split:, finger]),
        )
        selected_modes[name] = selected
        validation_prediction[:, finger] = validation_candidates[selected][:, finger]
        test_prediction[:, finger] = test_candidates[selected][:, finger]

    validation_state_truth = (target[split:] >= args.movement_threshold).astype(np.float32)
    report = {
        "subject": args.subject,
        "method": "CSP ridge amplitude plus recurrent movement-state auxiliary task",
        "target": args.target,
        "temporal_backbone": args.backbone,
        "movement_threshold": args.movement_threshold,
        "state_weight": args.state_weight,
        "correlation_weight": args.correlation_weight,
        "derivative_weight": args.derivative_weight,
        "derivative_correlation_weight": args.derivative_correlation_weight,
        "movement_loss_weight": args.movement_loss_weight,
        "selection_metric": args.selection_metric,
        "gate_floor": args.gate_floor,
        "best_epoch": best_epoch,
        "best_validation_historical_four": best_score,
        "selected_gate_mode_by_finger": selected_modes,
        "initial_validation_raw_metrics": trajectory_metrics(initial_validation, raw_train[split:]),
        "initial_test_raw_metrics": trajectory_metrics(initial_test, raw_test),
        "validation_raw_metrics": trajectory_metrics(validation_prediction, raw_train[split:]),
        "test_raw_metrics": trajectory_metrics(test_prediction, raw_test),
        "validation_candidate_metrics": {
            name: trajectory_metrics(value, raw_train[split:]) for name, value in validation_candidates.items()
        },
        "test_candidate_metrics": {
            name: trajectory_metrics(value, raw_test) for name, value in test_candidates.items()
        },
        "validation_state_metrics": state_metrics(validation_probability, validation_state_truth),
        "ridge_audit": ridge_audit,
        "history": history,
        "compiled": True,
        "compile_mode": "reduce-overhead",
    }
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    np.save(output / "validation_amplitude.npy", validation_amplitude, allow_pickle=False)
    np.save(output / "test_amplitude.npy", test_amplitude, allow_pickle=False)
    np.save(output / "validation_state_probability.npy", validation_probability, allow_pickle=False)
    np.save(output / "test_state_probability.npy", test_probability, allow_pickle=False)
    torch.save(best_state, output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
