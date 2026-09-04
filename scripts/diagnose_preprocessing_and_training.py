#!/usr/bin/env python3
"""Audit target assignment, gradient flow, and a band-energy ridge control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ecog_decoding.models import EcogTrajectoryDecoder, WaveletPacketEnergy
from ecog_decoding.preprocessing import MovementCorrectionModel, apply_movement_corrector
from ecog_decoding.training import align_causal_sequence, joint_trajectory_loss, trajectory_metrics


FINGERS = ("thumb", "index", "middle", "ring", "little")


def load_array(root: Path, subject: int, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(root / f"sub{subject}" / f"{name}.npy", mmap_mode="r" if mmap else None)


def movement_model(metadata: dict[str, object]) -> MovementCorrectionModel:
    config = metadata["config"]
    return MovementCorrectionModel(
        scale=np.asarray(metadata["target_scale"], dtype=np.float64),
        coupling_matrix=np.asarray(metadata["finger_coupling_matrix"], dtype=np.float64),
        sampling_rate_hz=float(config["glove_rate_hz"]),
        baseline_smoothness=float(config["baseline_smoothness"]),
        baseline_asymmetry=float(config["baseline_asymmetry"]),
        baseline_iterations=int(config["baseline_iterations"]),
        activation_threshold=float(config["movement_activation_threshold"]),
        state_transition_penalty=float(config["movement_state_transition_penalty"]),
        coupling_minimum_activation=float(config["coupling_minimum_activation"]),
    )


def dominant_states(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(values, axis=1)
    state = order[:, -1].astype(np.int16)
    top = values[np.arange(values.shape[0]), order[:, -1]]
    second = values[np.arange(values.shape[0]), order[:, -2]]
    state[top < threshold] = -1
    return state, top - second


def confusion(reference: np.ndarray, estimate: np.ndarray) -> list[list[int]]:
    # State order is rest, thumb, index, middle, ring, little.
    matrix = np.zeros((6, 6), dtype=np.int64)
    np.add.at(matrix, (reference.astype(int) + 1, estimate.astype(int) + 1), 1)
    return matrix.tolist()


def assignment_audit(
    decoded: np.ndarray,
    multifinger: np.ndarray,
    events: tuple[tuple[int, int, int], ...],
    threshold: float,
) -> dict[str, object]:
    dominant, margin = dominant_states(multifinger, threshold)
    both_active = (decoded >= 0) & (dominant >= 0)
    decisive = both_active & (margin >= 0.10)
    event_peak_matches: list[bool] = []
    event_integral_matches: list[bool] = []
    event_margins: list[float] = []
    event_rows: list[dict[str, object]] = []
    for start, stop, assigned in events:
        segment = multifinger[start:stop]
        if not segment.size:
            continue
        peak_time = int(np.argmax(np.max(segment, axis=1)))
        peak_finger = int(np.argmax(segment[peak_time]))
        totals = np.sum(segment, axis=0)
        integrated_finger = int(np.argmax(totals))
        sorted_totals = np.sort(totals)
        integral_margin = float(
            (sorted_totals[-1] - sorted_totals[-2]) / max(sorted_totals[-1], 1e-8)
        )
        event_peak_matches.append(assigned == peak_finger)
        event_integral_matches.append(assigned == integrated_finger)
        event_margins.append(integral_margin)
        if assigned != integrated_finger:
            event_rows.append(
                {
                    "start_bin": int(start),
                    "stop_bin": int(stop),
                    "assigned": FINGERS[assigned],
                    "peak_measured": FINGERS[peak_finger],
                    "integrated_measured": FINGERS[integrated_finger],
                    "integral_margin": integral_margin,
                }
            )
    clear_events = np.asarray(event_margins) >= 0.10
    integral_matches = np.asarray(event_integral_matches, dtype=bool)
    return {
        "sample_confusion_rows_decoded_cols_dominant": confusion(decoded, dominant),
        "sample_labels": ("rest",) + FINGERS,
        "both_active_samples": int(np.sum(both_active)),
        "sample_agreement_when_both_active": float(np.mean(decoded[both_active] == dominant[both_active])) if np.any(both_active) else None,
        "decisive_samples_margin_ge_0p10": int(np.sum(decisive)),
        "sample_agreement_decisive": float(np.mean(decoded[decisive] == dominant[decisive])) if np.any(decisive) else None,
        "event_count": len(event_peak_matches),
        "event_agreement_peak": float(np.mean(event_peak_matches)) if event_peak_matches else None,
        "event_agreement_integrated": float(np.mean(event_integral_matches)) if event_integral_matches else None,
        "clear_event_count_margin_ge_0p10": int(np.sum(clear_events)),
        "clear_event_agreement_integrated": float(np.mean(integral_matches[clear_events])) if np.any(clear_events) else None,
        "mismatched_events": event_rows,
    }


def make_model(config: dict[str, object], input_channels: int, device: torch.device) -> EcogTrajectoryDecoder:
    decoder = config["decoder"]
    frontend = config["wavelet_packet_frontend"]
    return EcogTrajectoryDecoder(
        input_channels=input_channels,
        spatial_components=decoder["spatial_components"],
        feature_width=int(decoder["feature_width"]),
        temporal_backbone=str(decoder["temporal_backbone"]),
        temporal_layers=int(decoder["temporal_layers"]),
        state_size=int(decoder["state_size"]),
        dropout=float(decoder["dropout"]),
        output_fingers=int(decoder["output_fingers"]),
        cnn_history_bins=int(decoder["cnn_history_bins"]),
        wavelet_frontend=WaveletPacketEnergy(
            wavelet=str(frontend["wavelet"]),
            levels=int(frontend["levels"]),
            kernel_size=int(frontend["kernel_size"]),
            trainable=bool(frontend["trainable"]),
            padding_mode=str(frontend["padding_mode"]),
            energy_window_samples=int(frontend["energy_window_samples"]),
            energy_stride_samples=int(frontend["energy_stride_samples"]),
        ),
    ).to(device)


def norm_dict(parameters: list[torch.nn.Parameter], gradients: bool = False) -> float:
    values = []
    for parameter in parameters:
        value = parameter.grad if gradients else parameter.detach()
        if value is not None:
            values.append(torch.sum(value.float().square()))
    return float(torch.sqrt(torch.stack(values).sum()).cpu()) if values else 0.0


def gradient_audit(
    config: dict[str, object], prepared_root: Path, training_root: Path,
    subject: int, device: torch.device,
) -> dict[str, object]:
    metadata = json.loads((prepared_root / f"sub{subject}" / "metadata.json").read_text())
    ecog = load_array(prepared_root, subject, "train_ecog")
    intended = load_array(prepared_root, subject, "train_target_intended")
    split = int(metadata["target_fit_samples_25hz"])
    history = int(config["decoder"]["cnn_history_bins"])
    sequence = align_causal_sequence(ecog, intended, history - 1, split, history)

    seed = int(config["training"]["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = make_model(config, ecog.shape[1], device)
    ica = np.load(training_root / f"sub{subject}" / "fastica_unmixing.npy")
    with torch.no_grad():
        initial.spatial_projection.weight[:, :, 0].copy_(torch.as_tensor(ica, device=device))
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}

    checkpoint = torch.load(training_root / f"sub{subject}" / "best_model.pt", map_location=device, weights_only=False)
    model = make_model(config, ecog.shape[1], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    prediction = model(torch.from_numpy(sequence.ecog).unsqueeze(0).to(device))
    target = torch.from_numpy(sequence.target).unsqueeze(0).to(device)
    training = config["training"]
    loss, parts = joint_trajectory_loss(
        prediction, target,
        movement_threshold=float(training["movement_threshold"]),
        movement_weight=float(training["movement_weight"]),
        velocity_weight=float(training["velocity_loss_weight"]),
        correlation_weight=float(training["correlation_loss_weight"]),
    )
    loss.backward()

    groups = {
        "spatial_projection": model.spatial_projection,
        "wavelet_frontend": model.wavelet_frontend,
        "feature_projection": model.feature_projection,
        "temporal": model.temporal,
        "output": model.output,
    }
    group_results: dict[str, object] = {}
    named = dict(model.named_parameters())
    for group_name, module in groups.items():
        params = list(module.parameters())
        parameter_norm = norm_dict(params)
        gradient_norm = norm_dict(params, gradients=True)
        prefixes = [name for name, parameter in named.items() if any(parameter is p for p in params)]
        squared_drift = []
        squared_initial = []
        for name in prefixes:
            current = named[name].detach().float()
            reference = initial_state[name].float()
            squared_drift.append(torch.sum((current - reference).square()))
            squared_initial.append(torch.sum(reference.square()))
        drift = float(torch.sqrt(torch.stack(squared_drift).sum()).cpu())
        initial_norm = float(torch.sqrt(torch.stack(squared_initial).sum()).cpu())
        group_results[group_name] = {
            "parameter_norm": parameter_norm,
            "gradient_norm": gradient_norm,
            "gradient_over_parameter": gradient_norm / max(parameter_norm, 1e-12),
            "drift_from_initialization": drift,
            "drift_over_initial": drift / max(initial_norm, 1e-12),
        }
    pred = prediction.detach().squeeze(0).cpu().numpy()
    return {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "loss": float(loss.detach()),
        "loss_parts": {key: float(value) for key, value in parts.items()},
        "prediction_mean": pred.mean(axis=0).tolist(),
        "prediction_std": pred.std(axis=0).tolist(),
        "groups": group_results,
    }


@torch.inference_mode()
def initialized_energy(
    ecog: np.ndarray, ica: np.ndarray, config: dict[str, object], device: torch.device
) -> np.ndarray:
    frontend_config = config["wavelet_packet_frontend"]
    frontend = WaveletPacketEnergy(
        wavelet=str(frontend_config["wavelet"]),
        levels=int(frontend_config["levels"]),
        kernel_size=int(frontend_config["kernel_size"]),
        trainable=False,
        padding_mode=str(frontend_config["padding_mode"]),
        energy_window_samples=int(frontend_config["energy_window_samples"]),
        energy_stride_samples=int(frontend_config["energy_stride_samples"]),
    ).to(device).eval()
    x = torch.from_numpy(np.asarray(ecog).T.copy()).unsqueeze(0).to(device)
    spatial = torch.nn.functional.conv1d(x, torch.as_tensor(ica, device=device)[:, :, None])
    energy = frontend(spatial).squeeze(0).permute(2, 0, 1).flatten(1)
    return energy.float().cpu().numpy()


def lagged(values: np.ndarray, history: int) -> np.ndarray:
    windows = np.lib.stride_tricks.sliding_window_view(values, history, axis=0)
    # sliding_window_view returns (time, feature, lag); match the network's lag-major flattening.
    return np.ascontiguousarray(windows.transpose(0, 2, 1).reshape(windows.shape[0], -1))


def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def fit_gpu_ridge(
    x: np.ndarray, y: np.ndarray, alpha: float, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    xt = torch.from_numpy((x - mean) / scale).to(device)
    yt = torch.from_numpy(np.array(y, dtype=np.float32, copy=True)).to(device)
    y_mean = float(yt.mean().cpu())
    yt = yt - y_mean
    gram = xt.T @ xt / xt.shape[0]
    rhs = xt.T @ yt / xt.shape[0]
    gram.diagonal().add_(alpha)
    weight = torch.linalg.solve(gram, rhs)
    return mean, scale, y_mean, weight.cpu().numpy()


def ridge_predict(x: np.ndarray, fit: tuple[np.ndarray, np.ndarray, float, np.ndarray]) -> np.ndarray:
    mean, scale, y_mean, weight = fit
    return ((x - mean) / scale) @ weight + y_mean


def run_ridge_target(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    feature_count: int,
    history: int,
    top_features: int,
    device: torch.device,
    validation_raw: np.ndarray | None = None,
    test_raw: np.ndarray | None = None,
) -> dict[str, object]:
    inner_stop = int(round(train_x.shape[0] * 0.8))
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    validation_prediction = np.zeros_like(validation_y, dtype=np.float32)
    test_prediction = np.zeros_like(test_y, dtype=np.float32)
    finger_results: dict[str, object] = {}
    for finger, name in enumerate(FINGERS):
        correlations = pearson_columns(train_x[:inner_stop], train_y[:inner_stop, finger])
        selected = np.argpartition(np.abs(correlations), -top_features)[-top_features:]
        selected = selected[np.argsort(np.abs(correlations[selected]))[::-1]]
        inner_scores: dict[str, float] = {}
        best_alpha = alphas[0]
        best_score = -float("inf")
        for alpha in alphas:
            fit = fit_gpu_ridge(
                train_x[:inner_stop, selected], train_y[:inner_stop, finger], alpha, device
            )
            estimate = ridge_predict(train_x[inner_stop:, selected], fit)
            score = pearson(estimate, train_y[inner_stop:, finger])
            inner_scores[str(alpha)] = score
            if score > best_score:
                best_score = score
                best_alpha = alpha
        fit = fit_gpu_ridge(train_x[:, selected], train_y[:, finger], best_alpha, device)
        validation_prediction[:, finger] = ridge_predict(validation_x[:, selected], fit)
        test_prediction[:, finger] = ridge_predict(test_x[:, selected], fit)
        decoded_features = []
        for index in selected[:20]:
            lag_index, energy_index = divmod(int(index), feature_count)
            component, band = divmod(energy_index, 8)
            decoded_features.append(
                {"lag_bins_ago": history - 1 - lag_index, "spatial_component": component, "band": band, "training_abs_r": float(abs(correlations[index]))}
            )
        finger_results[name] = {
            "best_alpha": best_alpha,
            "inner_validation_r": best_score,
            "alpha_scores": inner_scores,
            "top_features": decoded_features,
        }
    result = {
        "finger_models": finger_results,
        "validation_metrics": trajectory_metrics(validation_prediction, validation_y),
        "test_metrics": trajectory_metrics(test_prediction, test_y),
    }
    if validation_raw is not None:
        result["validation_raw_metrics"] = trajectory_metrics(validation_prediction, validation_raw)
    if test_raw is not None:
        result["test_raw_metrics"] = trajectory_metrics(test_prediction, test_raw)
    return result


def ridge_audit(
    config: dict[str, object], prepared_root: Path, training_root: Path,
    subject: int, device: torch.device, top_features: int,
) -> dict[str, object]:
    metadata = json.loads((prepared_root / f"sub{subject}" / "metadata.json").read_text())
    ecog = load_array(prepared_root, subject, "train_ecog")
    test_ecog = load_array(prepared_root, subject, "test_ecog")
    intended = np.asarray(load_array(prepared_root, subject, "train_target_intended"))
    paper_train_path = prepared_root / f"sub{subject}" / "train_glove_paper_baseline_only.npy"
    paper_test_path = prepared_root / f"sub{subject}" / "test_glove_paper_baseline_only.npy"
    if not paper_train_path.exists() or not paper_test_path.exists():
        raise FileNotFoundError(
            "published baseline targets are missing; run scripts/prepare_paper_baseline_targets.py"
        )
    detrended = np.load(paper_train_path)
    detrended_test = np.load(paper_test_path)
    raw_train = np.asarray(load_array(prepared_root, subject, "train_glove_25hz_raw"))
    raw_test = np.asarray(load_array(prepared_root, subject, "test_glove_25hz_raw"))
    ica = np.load(training_root / f"sub{subject}" / "fastica_unmixing.npy")
    history = int(config["decoder"]["cnn_history_bins"])
    split = int(metadata["target_fit_samples_25hz"])
    train_energy = initialized_energy(ecog, ica, config, device)
    test_energy = initialized_energy(test_ecog, ica, config, device)
    train_x_all = lagged(train_energy, history)
    test_x = lagged(test_energy, history)
    offset = history - 1
    train_count = split - offset
    train_x = train_x_all[:train_count]
    validation_x = train_x_all[train_count:]
    common = {
        "feature_count": train_energy.shape[1],
        "history": history,
        "top_features": top_features,
        "device": device,
    }
    intended_result = run_ridge_target(
        train_x, intended[offset:split], validation_x, raw_train[split:intended.shape[0]],
        test_x, raw_test[offset:], **common,
    )
    baseline_result = run_ridge_target(
        train_x, detrended[offset:split], validation_x, detrended[split:],
        test_x, detrended_test[offset:],
        validation_raw=raw_train[split:intended.shape[0]], test_raw=raw_test[offset:],
        **common,
    )
    return {
        "method": "initialized FastICA plus fixed bior6.8 energy; train-only correlation screen; GPU ridge",
        "candidate_features": int(train_x.shape[1]),
        "selected_features_per_finger": int(top_features),
        "train_samples": int(train_x.shape[0]),
        "validation_samples": int(validation_x.shape[0]),
        "test_samples": int(test_x.shape[0]),
        "cleaned_intended_training_raw_evaluation": intended_result,
        "published_baseline_only_no_winner_assignment": baseline_result,
    }


def prediction_assignment_audit(
    prediction: np.ndarray, corrected_states: np.ndarray, multifinger: np.ndarray,
    threshold: float, offset: int,
) -> dict[str, object]:
    prediction_state = np.argmax(prediction, axis=1).astype(np.int16)
    decoded = corrected_states[offset:offset + prediction.shape[0]]
    dominant, margin = dominant_states(multifinger[offset:offset + prediction.shape[0]], threshold)
    decoded_active = decoded >= 0
    dominant_active = dominant >= 0
    disagree = decoded_active & dominant_active & (decoded != dominant)
    decisive_disagree = disagree & (margin >= 0.10)
    return {
        "prediction_vs_corrected_when_corrected_active": float(np.mean(prediction_state[decoded_active] == decoded[decoded_active])) if np.any(decoded_active) else None,
        "prediction_vs_measured_dominant_when_measured_active": float(np.mean(prediction_state[dominant_active] == dominant[dominant_active])) if np.any(dominant_active) else None,
        "corrected_vs_measured_disagreement_samples": int(np.sum(disagree)),
        "on_disagreements_prediction_matches_corrected": float(np.mean(prediction_state[disagree] == decoded[disagree])) if np.any(disagree) else None,
        "on_disagreements_prediction_matches_measured": float(np.mean(prediction_state[disagree] == dominant[disagree])) if np.any(disagree) else None,
        "decisive_disagreement_samples": int(np.sum(decisive_disagree)),
        "on_decisive_disagreements_prediction_matches_corrected": float(np.mean(prediction_state[decisive_disagree] == decoded[decisive_disagree])) if np.any(decisive_disagree) else None,
        "on_decisive_disagreements_prediction_matches_measured": float(np.mean(prediction_state[decisive_disagree] == dominant[decisive_disagree])) if np.any(decisive_disagree) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed"))
    parser.add_argument("--training-root", type=Path, default=Path("outputs/training"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--top-features", type=int, default=512)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for subject in args.subjects:
        print(f"subject={subject} assignment audit", flush=True)
        metadata = json.loads((args.prepared_root / f"sub{subject}" / "metadata.json").read_text())
        threshold = float(metadata["config"]["movement_activation_threshold"])
        raw_train = np.asarray(load_array(args.prepared_root, subject, "train_glove_25hz_raw"))
        raw_test = np.asarray(load_array(args.prepared_root, subject, "test_glove_25hz_raw"))
        model = movement_model(metadata)
        corrected_train = apply_movement_corrector(raw_train, model)
        corrected_test = apply_movement_corrector(raw_test, model)
        result: dict[str, object] = {
            "subject": subject,
            "coupling_matrix_rows_measured_cols_intended": model.coupling_matrix.tolist(),
            "train_assignment": assignment_audit(corrected_train.active_finger, corrected_train.detrended_multifinger, corrected_train.events, threshold),
            "test_assignment": assignment_audit(corrected_test.active_finger, corrected_test.detrended_multifinger, corrected_test.events, threshold),
        }
        prediction_path = args.training_root / f"sub{subject}" / "test_prediction.npy"
        if prediction_path.exists():
            prediction = np.load(prediction_path)
            result["trained_prediction_assignment"] = prediction_assignment_audit(
                prediction, corrected_test.active_finger, corrected_test.detrended_multifinger,
                threshold, int(config["decoder"]["cnn_history_bins"]) - 1,
            )
        print(f"subject={subject} gradient audit", flush=True)
        result["gradient_audit"] = gradient_audit(
            config, args.prepared_root, args.training_root, subject, device
        )
        print(f"subject={subject} ridge audit", flush=True)
        result["ridge_audit"] = ridge_audit(
            config, args.prepared_root, args.training_root, subject, device, args.top_features
        )
        output = args.output_root / f"sub{subject}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        summaries.append(result)
        print(json.dumps({
            "subject": subject,
            "train_clear_event_assignment": result["train_assignment"]["clear_event_agreement_integrated"],
            "gradient_norms": {key: value["gradient_norm"] for key, value in result["gradient_audit"]["groups"].items()},
            "ridge_cleaned_validation_r4": result["ridge_audit"]["cleaned_intended_training_raw_evaluation"]["validation_metrics"]["pearson_historical_four"],
            "ridge_cleaned_test_r4": result["ridge_audit"]["cleaned_intended_training_raw_evaluation"]["test_metrics"]["pearson_historical_four"],
            "ridge_paper_baseline_validation_r4": result["ridge_audit"]["published_baseline_only_no_winner_assignment"]["validation_metrics"]["pearson_historical_four"],
            "ridge_paper_baseline_test_r4": result["ridge_audit"]["published_baseline_only_no_winner_assignment"]["test_metrics"]["pearson_historical_four"],
        }), flush=True)
    (args.output_root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
