#!/usr/bin/env python3
"""Train five independent ridge-initialized CSP/SSM finger decoders."""

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


def own_csp_columns(finger: int, band_count: int = 7) -> np.ndarray:
    # benchmark_csp_ridge writes 4 filters/finger and 20 filters/band.
    return np.concatenate(
        [np.arange(band * 20 + finger * 4, band * 20 + finger * 4 + 4) for band in range(band_count)]
    )


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


@torch.inference_mode()
def predict(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(x).squeeze(0).squeeze(-1).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/separate_csp_ssm_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--feature-scope", choices=("own", "all"), default="own")
    parser.add_argument("--loss-profile", choices=("mse", "correlation"), default="mse")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--training-mode", choices=("full", "segments"), default="full")
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--direct-learning-rate", type=float, default=2e-5)
    parser.add_argument("--residual-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-history-input", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone",
        choices=("ssm", "gru", "lstm", "linear_attention", "mamba"),
        default="ssm",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    root = args.prepared_root / f"sub{args.subject}"
    csp_root = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    all_train_energy = np.load(csp_root / "train_csp_energy.npy")
    all_test_energy = np.load(csp_root / "test_csp_energy.npy")
    target = np.load(root / f"train_glove_{args.target}.npy")
    raw_train = np.load(root / "train_glove_25hz_raw.npy")
    raw_test = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    validation_predictions = np.zeros_like(raw_train[split:], dtype=np.float32)
    test_predictions = np.zeros_like(raw_test, dtype=np.float32)
    initial_validation_predictions = np.zeros_like(validation_predictions)
    initial_test_predictions = np.zeros_like(test_predictions)
    finger_reports: dict[str, object] = {}

    for finger, finger_name in enumerate(FINGER_NAMES):
        np.random.seed(args.seed + finger)
        torch.manual_seed(args.seed + finger)
        torch.cuda.manual_seed_all(args.seed + finger)
        columns = (
            own_csp_columns(finger)
            if args.feature_scope == "own"
            else np.arange(all_train_energy.shape[1])
        )
        train_energy = np.ascontiguousarray(all_train_energy[:, columns])
        test_energy = np.ascontiguousarray(all_test_energy[:, columns])
        train_x_all = lagged(train_energy, args.history)
        train_count = split - offset
        train_x = train_x_all[:train_count]
        train_y = target[offset:split, finger : finger + 1]
        model = CSPResidualSSM(
            train_energy.shape[1],
            args.history,
            width=args.width,
            layers=args.layers,
            state_size=16,
            dropout=args.dropout,
            output_fingers=1,
            temporal_backbone=args.backbone,
            temporal_history_input=args.temporal_history_input,
        ).to(device)
        ridge_audit = initialize_ridge(
            model,
            train_x,
            train_y,
            min(args.top_features, train_x.shape[1]),
            device,
        )
        train_tensor = torch.from_numpy(train_energy[:split]).unsqueeze(0).to(device)
        validation_tensor = torch.from_numpy(train_energy[split - offset :]).unsqueeze(0).to(device)
        test_tensor = torch.from_numpy(test_energy).unsqueeze(0).to(device)
        target_tensor = torch.from_numpy(train_y).unsqueeze(0).to(device)
        initial_validation = predict(model, validation_tensor)
        initial_test = predict(model, test_tensor)
        initial_validation_predictions[:, finger] = initial_validation
        initial_test_predictions[:, finger] = initial_test
        best_score = pearson(initial_validation, raw_train[split:, finger])
        best_epoch = 0
        best_state = copy.deepcopy(model.state_dict())

        executable = torch.compile(model, mode="reduce-overhead")
        direct_ids = {id(parameter) for parameter in model.direct.parameters()}
        residual_parameters = [parameter for parameter in model.parameters() if id(parameter) not in direct_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.direct.parameters(), "lr": args.direct_learning_rate},
                {"params": residual_parameters, "lr": args.residual_learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        if args.training_mode == "segments":
            if args.sequence_steps < 1 or args.sequence_stride < 1 or args.batch_size < 1:
                raise ValueError("sequence and batch parameters must be positive")
            output_steps = split - offset
            if args.sequence_steps > output_steps:
                raise ValueError("sequence-steps exceeds the available training output")
            starts = list(range(0, output_steps - args.sequence_steps + 1, args.sequence_stride))
            final_start = output_steps - args.sequence_steps
            if starts[-1] != final_start:
                starts.append(final_start)
            segmented_energy = torch.stack(
                [
                    train_tensor[0, start : start + args.sequence_steps + offset]
                    for start in starts
                ]
            )
            segmented_target = torch.stack(
                [
                    target_tensor[0, start : start + args.sequence_steps]
                    for start in starts
                ]
            )
        else:
            starts = [0]
            segmented_energy = train_tensor
            segmented_target = target_tensor
        history = []
        stale = 0
        optimizer_steps = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(segmented_energy.shape[0], device=device)
            epoch_loss = 0.0
            epoch_level = 0.0
            epoch_corr = 0.0
            batch_count = 0
            for batch_start in range(0, order.numel(), args.batch_size):
                batch_index = order[batch_start : batch_start + args.batch_size]
                batch_energy = segmented_energy[batch_index]
                batch_target = segmented_target[batch_index]
                estimate = executable(batch_energy)
                level = F.mse_loss(estimate, batch_target)
                corr = correlation_loss(estimate, batch_target)
                loss = level if args.loss_profile == "mse" else level + 0.25 * corr
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
                epoch_level += float(level.detach())
                epoch_corr += float(corr.detach())
                batch_count += 1
            epoch_loss /= batch_count
            epoch_level /= batch_count
            epoch_corr /= batch_count
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                validation = predict(executable, validation_tensor)
                score = pearson(validation, raw_train[split:, finger])
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_steps": optimizer_steps,
                        "loss": epoch_loss,
                        "level": epoch_level,
                        "correlation_loss": epoch_corr,
                        "validation_r": score,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                print(
                    f"finger={finger_name} epoch={epoch} val_r={score:.4f} best={best_score:.4f}",
                    flush=True,
                )
                if stale >= 12:
                    break

        model.load_state_dict(best_state)
        validation_predictions[:, finger] = predict(executable, validation_tensor)
        test_predictions[:, finger] = predict(executable, test_tensor)
        finger_dir = output / finger_name
        finger_dir.mkdir(exist_ok=True)
        torch.save(best_state, finger_dir / "best_model.pt")
        finger_reports[finger_name] = {
            "feature_scope": args.feature_scope,
            "per_bin_features": int(train_energy.shape[1]),
            "best_epoch": best_epoch,
            "best_validation_r": best_score,
            "initial_validation_r": pearson(initial_validation, raw_train[split:, finger]),
            "initial_test_r": pearson(initial_test, raw_test[:, finger]),
            "test_r": pearson(test_predictions[:, finger], raw_test[:, finger]),
            "ridge_audit": ridge_audit["0"],
            "history": history,
            "optimizer_steps": optimizer_steps,
            "training_sequences": len(starts),
        }
        del executable, model, optimizer, train_tensor, validation_tensor, test_tensor
        torch.cuda.empty_cache()

    report = {
        "subject": args.subject,
        "method": f"five fully independent CSP-ridge plus zero-residual {args.backbone} models",
        "target": args.target,
        "feature_scope": args.feature_scope,
        "loss_profile": args.loss_profile,
        "temporal_backbone": args.backbone,
        "training_mode": args.training_mode,
        "sequence_steps": args.sequence_steps if args.training_mode == "segments" else None,
        "sequence_stride": args.sequence_stride if args.training_mode == "segments" else None,
        "batch_size": args.batch_size,
        "width": args.width,
        "layers": args.layers,
        "temporal_history_input": args.temporal_history_input,
        "initial_validation_raw_metrics": trajectory_metrics(initial_validation_predictions, raw_train[split:]),
        "initial_test_raw_metrics": trajectory_metrics(initial_test_predictions, raw_test),
        "validation_raw_metrics": trajectory_metrics(validation_predictions, raw_train[split:]),
        "test_raw_metrics": trajectory_metrics(test_predictions, raw_test),
        "fingers": finger_reports,
        "compiled": True,
        "compile_mode": "reduce-overhead",
    }
    np.save(output / "validation_prediction_initial.npy", initial_validation_predictions, allow_pickle=False)
    np.save(output / "test_prediction_initial.npy", initial_test_predictions, allow_pickle=False)
    np.save(output / "validation_prediction.npy", validation_predictions, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_predictions, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
