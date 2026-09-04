#!/usr/bin/env python3
"""Train subject-specific joint five-finger ECoG decoders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ecog_decoding.models import EcogTrajectoryDecoder, WaveletPacketEnergy
from ecog_decoding.training import (
    align_causal_sequence,
    joint_trajectory_loss,
    trajectory_metrics,
)


def load_array(root: Path, subject: int, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(root / f"sub{subject}" / f"{name}.npy", mmap_mode="r" if mmap else None)


def tensor(sequence: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(sequence).unsqueeze(0).to(device=device, non_blocking=True)


@torch.inference_mode()
def predict(executable: torch.nn.Module, ecog: torch.Tensor) -> np.ndarray:
    executable.eval()
    return executable(ecog).squeeze(0).float().cpu().numpy()


@torch.inference_mode()
def history_features(model: EcogTrajectoryDecoder, ecog: torch.Tensor) -> np.ndarray:
    """Return the exact flattened ICA/wavelet history seen by the decoder."""
    spatial = model.spatial_projection(ecog)
    energy = model.wavelet_frontend(spatial)
    per_bin = energy.permute(0, 3, 1, 2).flatten(start_dim=2)
    history = per_bin.unfold(1, model.cnn_history_bins, 1)
    return history.permute(0, 1, 3, 2).flatten(start_dim=2).squeeze(0).cpu().numpy()


def ridge_initializers(
    features: np.ndarray,
    target: np.ndarray,
    top_features: int,
    alphas: tuple[float, ...],
    device: torch.device,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    dict[str, object],
]:
    """Fit train-only screened ridge models for exact direct-head initialization."""
    inner_stop = int(round(0.8 * features.shape[0]))
    fits: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]] = []
    audit: dict[str, object] = {}
    for finger in range(target.shape[1]):
        x_screen = features[:inner_stop]
        y_screen = target[:inner_stop, finger]
        xc = x_screen - x_screen.mean(axis=0, keepdims=True)
        yc = y_screen - y_screen.mean()
        denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
        correlations = np.divide(
            xc.T @ yc,
            denominator,
            out=np.zeros(features.shape[1], dtype=np.float32),
            where=denominator > 0,
        )
        count = min(top_features, correlations.size)
        selected = np.argpartition(np.abs(correlations), -count)[-count:]
        selected = selected[np.argsort(np.abs(correlations[selected]))[::-1]]

        best_alpha = alphas[0]
        best_score = -float("inf")
        alpha_scores: dict[str, float] = {}
        for alpha in alphas:
            mean = features[:inner_stop, selected].mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = features[:inner_stop, selected].std(axis=0, dtype=np.float64).astype(np.float32)
            scale[scale < 1e-6] = 1.0
            x = torch.from_numpy((features[:inner_stop, selected] - mean) / scale).to(device)
            y = torch.from_numpy(np.asarray(target[:inner_stop, finger], dtype=np.float32)).to(device)
            y_mean = float(y.mean())
            y = y - y_mean
            gram = x.T @ x / x.shape[0]
            rhs = x.T @ y / x.shape[0]
            gram.diagonal().add_(alpha)
            weight = torch.linalg.solve(gram, rhs).cpu().numpy()
            estimate = ((features[inner_stop:, selected] - mean) / scale) @ weight + y_mean
            observed = target[inner_stop:, finger]
            estimate_centered = estimate - estimate.mean()
            observed_centered = observed - observed.mean()
            norm = np.linalg.norm(estimate_centered) * np.linalg.norm(observed_centered)
            score = float(estimate_centered @ observed_centered / norm) if norm > 0 else 0.0
            alpha_scores[str(alpha)] = score
            if score > best_score:
                best_score = score
                best_alpha = alpha

        mean = features[:, selected].mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features[:, selected].std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        x = torch.from_numpy((features[:, selected] - mean) / scale).to(device)
        y = torch.from_numpy(np.asarray(target[:, finger], dtype=np.float32)).to(device)
        y_mean = float(y.mean())
        y = y - y_mean
        gram = x.T @ x / x.shape[0]
        rhs = x.T @ y / x.shape[0]
        gram.diagonal().add_(best_alpha)
        weight = torch.linalg.solve(gram, rhs).cpu().numpy()
        fits.append((selected, mean, scale, y_mean, weight))
        audit[str(finger)] = {
            "best_alpha": best_alpha,
            "inner_validation_r": best_score,
            "alpha_scores": alpha_scores,
            "selected_feature_indices": selected.tolist(),
        }
    return fits, audit


def train_subject(args: argparse.Namespace, config: dict[str, object], subject: int) -> dict[str, object]:
    started = time.perf_counter()
    prepared_root = Path(args.prepared_root)
    output_dir = Path(args.output_root) / f"sub{subject}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((prepared_root / f"sub{subject}" / "metadata.json").read_text())
    ecog = load_array(prepared_root, subject, "train_ecog")
    target_array = str(config["training"].get("target_array", "train_target_intended"))
    intended = load_array(prepared_root, subject, target_array)
    raw_glove = load_array(prepared_root, subject, "train_glove_25hz_raw")
    test_ecog = load_array(prepared_root, subject, "test_ecog")
    test_raw = load_array(prepared_root, subject, "test_glove_25hz_raw")

    decoder_config = config["decoder"]
    frontend_config = config["wavelet_packet_frontend"]
    training_config = config["training"]
    runtime_config = config["runtime"]
    history_bins = int(decoder_config["cnn_history_bins"])
    split_bin = int(metadata["target_fit_samples_25hz"])
    train_start = history_bins - 1
    train_stop = split_bin
    if args.max_train_steps is not None:
        train_stop = min(train_stop, train_start + args.max_train_steps)
    train = align_causal_sequence(ecog, intended, train_start, train_stop, history_bins)
    validation = align_causal_sequence(ecog, intended, split_bin, intended.shape[0], history_bins)
    validation_raw = np.array(raw_glove[split_bin:intended.shape[0]], dtype=np.float32, copy=True)
    test_start = history_bins - 1
    test = align_causal_sequence(test_ecog, test_raw, test_start, test_raw.shape[0], history_bins)

    seed = int(training_config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision(str(runtime_config["float32_matmul_precision"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    frontend = WaveletPacketEnergy(
        wavelet=str(frontend_config["wavelet"]),
        levels=int(frontend_config["levels"]),
        kernel_size=int(frontend_config["kernel_size"]),
        trainable=bool(frontend_config["trainable"]),
        padding_mode=str(frontend_config["padding_mode"]),
        energy_window_samples=int(frontend_config["energy_window_samples"]),
        energy_stride_samples=int(frontend_config["energy_stride_samples"]),
    )
    model = EcogTrajectoryDecoder(
        input_channels=ecog.shape[1],
        spatial_components=decoder_config["spatial_components"],
        feature_width=int(decoder_config["feature_width"]),
        temporal_backbone=str(decoder_config["temporal_backbone"]),
        temporal_layers=int(decoder_config["temporal_layers"]),
        state_size=int(decoder_config["state_size"]),
        dropout=float(decoder_config["dropout"]),
        output_fingers=int(decoder_config["output_fingers"]),
        cnn_history_bins=history_bins,
        wavelet_frontend=frontend,
        direct_linear_head=bool(decoder_config.get("direct_linear_head", False)),
        output_activation=str(decoder_config.get("output_activation", "sigmoid")),
        zero_initialize_residual=bool(decoder_config.get("zero_initialize_residual", False)),
    ).to(device)
    configured_ica_device = str(decoder_config["fastica_device"])
    ica_device = str(device) if configured_ica_device == "cuda" and device.type == "cuda" else configured_ica_device
    if args.fastica_root is not None:
        ica_path = Path(args.fastica_root) / f"sub{subject}" / "fastica_unmixing.npy"
        ica_weights = np.load(ica_path)
        expected_shape = (
            model.spatial_projection.out_channels,
            model.spatial_projection.in_channels,
        )
        if ica_weights.shape != expected_shape:
            raise ValueError(
                f"cached FastICA shape {ica_weights.shape} does not match {expected_shape}"
            )
        with torch.no_grad():
            model.spatial_projection.weight[:, :, 0].copy_(
                torch.as_tensor(
                    ica_weights,
                    dtype=model.spatial_projection.weight.dtype,
                    device=device,
                )
            )
    else:
        ica_weights = model.initialize_spatial_from_fastica(
            np.asarray(ecog[: split_bin * 40]),
            max_samples=int(decoder_config["fastica_max_samples"]),
            random_state=int(decoder_config["fastica_random_state"]),
            backend=str(decoder_config["fastica_backend"]),
            device=ica_device,
        )
    np.save(output_dir / "fastica_unmixing.npy", ica_weights)

    train_x = tensor(train.ecog, device)
    train_y = tensor(train.target, device)
    validation_x = tensor(validation.ecog, device)
    initial_validation_metrics = None
    if bool(training_config.get("ridge_initialize", False)):
        if model.direct_output is None:
            raise ValueError("ridge_initialize requires decoder.direct_linear_head=true")
        initialized_features = history_features(model, train_x)
        fits, ridge_audit = ridge_initializers(
            initialized_features,
            train.target,
            top_features=int(training_config.get("ridge_top_features", 512)),
            alphas=tuple(float(value) for value in training_config.get(
                "ridge_alphas", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
            )),
            device=device,
        )
        model.initialize_direct_output_from_ridge(fits)
        initial_validation_prediction = predict(model, validation_x)
        initial_validation_metrics = trajectory_metrics(
            initial_validation_prediction, validation_raw
        )
        np.save(
            output_dir / "validation_prediction_initial_ridge.npy",
            initial_validation_prediction,
        )
        (output_dir / "ridge_initialization.json").write_text(
            json.dumps(
                {
                    "models": ridge_audit,
                    "validation_raw_metrics": initial_validation_metrics,
                },
                indent=2,
            )
        )

    compile_enabled = bool(runtime_config["compile"]) and not args.no_compile
    executable = (
        torch.compile(model, mode=str(runtime_config["compile_mode"]))
        if compile_enabled
        else model
    )
    base_learning_rate = float(training_config["learning_rate"])
    slow_scale = float(training_config.get("initialized_learning_rate_scale", 1.0))
    slow_parameters: list[torch.nn.Parameter] = []
    for module in (model.spatial_projection, model.wavelet_frontend, model.direct_output):
        if module is not None:
            slow_parameters.extend(module.parameters())
    slow_ids = {id(parameter) for parameter in slow_parameters}
    residual_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in slow_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": slow_parameters, "lr": base_learning_rate * slow_scale},
            {"params": residual_parameters, "lr": base_learning_rate},
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    validation_interval = int(training_config["validation_interval"])
    patience = int(training_config["patience_validations"])
    best_score = (
        float(initial_validation_metrics["pearson_historical_four"])
        if initial_validation_metrics is not None
        else -float("inf")
    )
    best_epoch = 0
    stale_validations = 0
    history: list[dict[str, object]] = []
    checkpoint_path = output_dir / "best_model.pt"
    if initial_validation_metrics is not None:
        torch.save(
            {
                "subject": subject,
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "model_config": config,
                "validation_raw_metrics": initial_validation_metrics,
            },
            checkpoint_path,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = executable(train_x)
        loss, parts = joint_trajectory_loss(
            prediction,
            train_y,
            movement_threshold=float(training_config["movement_threshold"]),
            movement_weight=float(training_config["movement_weight"]),
            velocity_weight=float(training_config["velocity_loss_weight"]),
            correlation_weight=float(training_config["correlation_loss_weight"]),
            level_kind=str(training_config.get("level_loss", "mse")),
            huber_delta=float(training_config.get("huber_delta", 0.10)),
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training_config["gradient_clip_norm"])
        )
        optimizer.step()
        record: dict[str, object] = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "level_loss": float(parts["level"]),
            "velocity_loss": float(parts["velocity"]),
            "correlation_loss": float(parts["correlation_loss"]),
            "gradient_norm": float(gradient_norm),
        }
        should_validate = epoch == 1 or epoch % validation_interval == 0 or epoch == epochs
        if should_validate:
            val_prediction = predict(executable, validation_x)
            val_metrics = trajectory_metrics(val_prediction, validation_raw)
            score = float(val_metrics["pearson_historical_four"])
            record["validation_raw"] = val_metrics
            print(
                f"subject={subject} epoch={epoch} loss={float(loss):.6f} "
                f"val_r4={score:.4f} val_r5={float(val_metrics['pearson_macro_five']):.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_validations = 0
                torch.save(
                    {
                        "subject": subject,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "model_config": config,
                        "validation_raw_metrics": val_metrics,
                    },
                    checkpoint_path,
                )
                np.save(output_dir / "validation_prediction_best.npy", val_prediction)
            else:
                stale_validations += 1
        history.append(record)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if should_validate and stale_validations >= patience:
            print(f"subject={subject} early_stop epoch={epoch}", flush=True)
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_x = tensor(test.ecog, device)
    test_prediction = predict(executable, test_x)
    test_metrics = trajectory_metrics(test_prediction, test.target)
    np.save(output_dir / "test_prediction.npy", test_prediction)
    summary = {
        "subject": subject,
        "best_epoch": best_epoch,
        "best_validation_historical_four": best_score,
        "validation_output_steps": int(validation.target.shape[0]),
        "training_output_steps": int(train.target.shape[0]),
        "test_output_steps": int(test.target.shape[0]),
        "test_excluded_initial_bins": test_start,
        "test_raw_metrics": test_metrics,
        "compiled": compile_enabled,
        "training_target_array": target_array,
        "initial_validation_raw_metrics": initial_validation_metrics,
        "ridge_initialized": bool(training_config.get("ridge_initialize", False)),
        "output_activation": str(decoder_config.get("output_activation", "sigmoid")),
        "compile_mode": str(runtime_config["compile_mode"]) if compile_enabled else None,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--prepared-root", default="outputs/preprocessed")
    parser.add_argument("--output-root", default="outputs/training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--target-array")
    parser.add_argument(
        "--loss-profile",
        choices=("configured", "paper_mse", "balanced", "correlation_huber"),
        default="configured",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fastica-root")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--backbone",
        choices=("diagonal_ssm", "gru", "lstm", "linear_attention", "mamba"),
    )
    parser.add_argument(
        "--output-activation",
        choices=("sigmoid", "relu", "identity"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.target_array is not None:
        config["training"]["target_array"] = args.target_array
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.backbone is not None:
        config["decoder"]["temporal_backbone"] = args.backbone
    if args.output_activation is not None:
        config["decoder"]["output_activation"] = args.output_activation
    if args.loss_profile == "paper_mse":
        config["training"].update(
            movement_weight=0.0,
            velocity_loss_weight=0.0,
            correlation_loss_weight=0.0,
            level_loss="mse",
        )
    elif args.loss_profile == "balanced":
        config["training"].update(
            movement_weight=4.0,
            velocity_loss_weight=0.2,
            correlation_loss_weight=0.1,
            level_loss="mse",
        )
    elif args.loss_profile == "correlation_huber":
        config["training"].update(
            movement_weight=2.0,
            velocity_loss_weight=0.05,
            correlation_loss_weight=0.3,
            level_loss="huber",
            huber_delta=0.10,
        )
    summaries = [train_subject(args, config, subject) for subject in args.subjects]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
