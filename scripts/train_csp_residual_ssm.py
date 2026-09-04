#!/usr/bin/env python3
"""Train a zero-residual SSM on top of an exactly initialized CSP-ridge path."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_ridge_target_variants import correlation_screen, lagged, ridge_fit, ridge_predict
from ecog_decoding.models import CausalDilatedTCN, CausalLinearAttention, DiagonalSSM, MambaSequence
from ecog_decoding.training import trajectory_metrics


class CSPResidualSSM(nn.Module):
    def __init__(
        self,
        per_bin_features: int,
        history: int,
        width: int = 96,
        layers: int = 3,
        state_size: int = 16,
        dropout: float = 0.1,
        output_fingers: int = 5,
        temporal_backbone: str = "ssm",
        temporal_history_input: bool = False,
    ) -> None:
        super().__init__()
        self.history = int(history)
        flat_features = per_bin_features * history
        self.register_buffer("feature_mean", torch.zeros(flat_features))
        self.register_buffer("feature_scale", torch.ones(flat_features))
        self.direct = nn.Linear(flat_features, output_fingers)
        self.temporal_history_input = bool(temporal_history_input)
        temporal_input_features = flat_features if temporal_history_input else per_bin_features
        self.input_projection = nn.Sequential(
            nn.LayerNorm(temporal_input_features),
            nn.Linear(temporal_input_features, width),
            nn.SiLU(),
        )
        if temporal_backbone == "ssm":
            self.temporal = DiagonalSSM(
                width,
                layers=layers,
                state_size=state_size,
                dropout=dropout,
            )
        elif temporal_backbone in {"gru", "lstm"}:
            recurrent = nn.GRU if temporal_backbone == "gru" else nn.LSTM
            self.temporal = recurrent(
                width,
                width,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
        elif temporal_backbone == "linear_attention":
            self.temporal = CausalLinearAttention(
                width,
                layers=layers,
                heads=4,
                dropout=dropout,
            )
        elif temporal_backbone == "mamba":
            self.temporal = MambaSequence(
                width,
                layers=layers,
                state_size=state_size,
                dropout=dropout,
            )
        elif temporal_backbone == "tcn":
            self.temporal = CausalDilatedTCN(width, layers=layers, dropout=dropout)
        else:
            raise ValueError(
                "temporal_backbone must be 'ssm', 'gru', 'lstm', 'linear_attention', "
                "'mamba', or 'tcn'"
            )
        self.temporal_backbone = temporal_backbone
        self.residual = nn.Linear(width, output_fingers)
        self.state_classifier = nn.Linear(width, output_fingers)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward_components(
        self, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = energy.unfold(1, self.history, 1)
        flattened = history.permute(0, 1, 3, 2).flatten(start_dim=2)
        standardized = (flattened - self.feature_mean) / self.feature_scale
        direct = self.direct(standardized)
        temporal_input = standardized if self.temporal_history_input else energy
        temporal = self.temporal(self.input_projection(temporal_input))
        if self.temporal_backbone in {"gru", "lstm"}:
            temporal, _ = temporal
        if not self.temporal_history_input:
            temporal = temporal[:, self.history - 1 :]
        return direct + self.residual(temporal), self.state_classifier(temporal)

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        prediction, _ = self.forward_components(energy)
        return prediction


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = torch.sum(prediction * target, dim=1)
    denominator = torch.sqrt(
        torch.sum(prediction.square(), dim=1)
        * torch.sum(target.square(), dim=1)
        + 1e-7
    )
    return 1.0 - (numerator / denominator).mean()


def initialize_ridge(
    model: CSPResidualSSM,
    train_x: np.ndarray,
    train_y: np.ndarray,
    top_features: int,
    device: torch.device,
) -> dict[str, object]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    model.feature_mean.copy_(torch.from_numpy(mean).to(device))
    model.feature_scale.copy_(torch.from_numpy(scale).to(device))
    model.direct.weight.data.zero_()
    model.direct.bias.data.zero_()
    inner_stop = int(round(0.8 * train_x.shape[0]))
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    audit: dict[str, object] = {}
    for finger in range(train_y.shape[1]):
        selected = correlation_screen(
            train_x[:inner_stop],
            train_y[:inner_stop, finger],
            top_features,
        )
        best_alpha = alphas[0]
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for alpha in alphas:
            fit = ridge_fit(
                train_x[:inner_stop, selected],
                train_y[:inner_stop, finger],
                alpha,
                device,
            )
            prediction = ridge_predict(train_x[inner_stop:, selected], fit)
            observed = train_y[inner_stop:, finger]
            pc = prediction - prediction.mean()
            oc = observed - observed.mean()
            norm = np.linalg.norm(pc) * np.linalg.norm(oc)
            score = float(pc @ oc / norm) if norm > 0 else 0.0
            scores[str(alpha)] = score
            if score > best_score:
                best_score = score
                best_alpha = alpha
        fit = ridge_fit(train_x[:, selected], train_y[:, finger], best_alpha, device)
        selected_mean, selected_scale, target_mean, weights = fit
        # The module standardizes all features using the full-training moments.
        # ridge_fit used the same moments on each selected column.
        np.testing.assert_allclose(selected_mean, mean[selected], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(selected_scale, scale[selected], rtol=1e-5, atol=1e-6)
        model.direct.weight.data[finger, selected] = torch.from_numpy(weights).to(device)
        model.direct.bias.data[finger] = target_mean
        audit[str(finger)] = {
            "best_alpha": best_alpha,
            "inner_validation_r": best_score,
            "alpha_scores": scores,
            "selected_feature_indices": selected.tolist(),
        }
    return audit


@torch.inference_mode()
def prediction(model: nn.Module, values: torch.Tensor) -> np.ndarray:
    model.eval()
    return model(values).squeeze(0).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/csp_residual_ssm_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--loss-profile", choices=("mse", "correlation"), default="mse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone",
        choices=("ssm", "gru", "lstm", "linear_attention", "mamba", "tcn"),
        default="ssm",
    )
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
    train_x_all = lagged(train_energy, args.history)
    train_count = split - offset
    train_x = train_x_all[:train_count]
    train_y = target[offset:split]

    model = CSPResidualSSM(
        train_energy.shape[1],
        args.history,
        temporal_backbone=args.backbone,
    ).to(device)
    ridge_audit = initialize_ridge(
        model,
        train_x,
        train_y,
        args.top_features,
        device,
    )
    train_energy_tensor = torch.from_numpy(train_energy[:split]).unsqueeze(0).to(device)
    validation_energy_tensor = torch.from_numpy(train_energy[split - offset :]).unsqueeze(0).to(device)
    test_energy_tensor = torch.from_numpy(test_energy).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(train_y).unsqueeze(0).to(device)
    initial_validation = prediction(model, validation_energy_tensor)
    initial_test = prediction(model, test_energy_tensor)
    initial_validation_metrics = trajectory_metrics(initial_validation, raw_train[split:])
    initial_test_metrics = trajectory_metrics(initial_test, raw_test)
    best_score = float(initial_validation_metrics["pearson_historical_four"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    executable = torch.compile(model, mode="reduce-overhead")
    direct_ids = {id(parameter) for parameter in model.direct.parameters()}
    residual_parameters = [p for p in model.parameters() if id(p) not in direct_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.direct.parameters(), "lr": 2e-5},
            {"params": residual_parameters, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        estimate = executable(train_energy_tensor)
        level = F.mse_loss(estimate, target_tensor)
        corr = correlation_loss(estimate, target_tensor)
        loss = level if args.loss_profile == "mse" else level + 0.25 * corr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            validation_prediction = prediction(executable, validation_energy_tensor)
            metrics = trajectory_metrics(validation_prediction, raw_train[split:])
            score = float(metrics["pearson_historical_four"])
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "level": float(level.detach()),
                    "correlation_loss": float(corr.detach()),
                    "gradient_norm": float(gradient_norm),
                    "validation_raw": metrics,
                }
            )
            print(f"epoch={epoch} val_r4={score:.4f} loss={float(loss):.5f}", flush=True)
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
    validation_prediction = prediction(executable, validation_energy_tensor)
    test_prediction = prediction(executable, test_energy_tensor)
    report = {
        "subject": args.subject,
        "method": f"CSP band energy plus exactly ridge-initialized direct path and zero-residual {args.backbone}",
        "target": args.target,
        "loss_profile": args.loss_profile,
        "temporal_backbone": args.backbone,
        "best_epoch": best_epoch,
        "best_validation_historical_four": best_score,
        "initial_validation_raw_metrics": initial_validation_metrics,
        "initial_test_raw_metrics": initial_test_metrics,
        "validation_raw_metrics": trajectory_metrics(validation_prediction, raw_train[split:]),
        "test_raw_metrics": trajectory_metrics(test_prediction, raw_test),
        "ridge_audit": ridge_audit,
        "history": history,
        "compiled": True,
        "compile_mode": "reduce-overhead",
    }
    np.save(output / "validation_prediction_initial.npy", initial_validation, allow_pickle=False)
    np.save(output / "test_prediction_initial.npy", initial_test, allow_pickle=False)
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    torch.save(best_state, output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
