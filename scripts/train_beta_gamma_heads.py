#!/usr/bin/env python3
"""Mechanistic beta-gate and finger-specific high-gamma amplitude decoder."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmark_ridge_target_variants import lagged
from ecog_decoding.models import CausalDilatedTCN
from ecog_decoding.training import trajectory_metrics
from train_csp_residual_ssm import correlation_loss, initialize_ridge
from train_state_aware_csp import morphology_score, state_metrics


class BetaGammaHeads(nn.Module):
    """Shared beta gate plus communicating finger-specific high-gamma heads."""

    def __init__(
        self,
        per_bin_features: int,
        history: int,
        fusion: str,
        beta_shuffle_bins: int = 0,
        width: int = 96,
        layers: int = 3,
        gate_floor: float = 0.10,
    ) -> None:
        super().__init__()
        if per_bin_features % 7:
            raise ValueError("expected seven equal-width frequency-band feature groups")
        per_band = per_bin_features // 7
        beta = torch.arange(2 * per_band, 3 * per_band)
        high_gamma = torch.arange(4 * per_band, 6 * per_band)
        if fusion == "beta_only":
            direct_indices = beta
        elif fusion == "gamma_only":
            direct_indices = high_gamma
        else:
            direct_indices = torch.cat((beta, high_gamma))
        self.register_buffer("beta_indices", beta)
        self.register_buffer("gamma_indices", high_gamma)
        self.register_buffer("direct_indices", direct_indices)
        self.history = int(history)
        self.fusion = fusion
        self.beta_shuffle_bins = int(beta_shuffle_bins)
        self.gate_floor = float(gate_floor)
        flat = direct_indices.numel() * history
        self.register_buffer("feature_mean", torch.zeros(flat))
        self.register_buffer("feature_scale", torch.ones(flat))
        self.direct = nn.Linear(flat, 5)
        self.beta_projection = nn.Sequential(nn.LayerNorm(per_band), nn.Linear(per_band, width), nn.SiLU())
        self.gamma_projection = nn.Sequential(
            nn.LayerNorm(2 * per_band), nn.Linear(2 * per_band, width), nn.SiLU()
        )
        self.beta_temporal = CausalDilatedTCN(width, layers=layers, dropout=0.1)
        self.gamma_temporal = CausalDilatedTCN(width, layers=layers, dropout=0.1)
        self.film = nn.Linear(width, 2 * width)
        self.gate_head = nn.Linear(width, 5)
        self.beta_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(5))
        self.gamma_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(5))
        self.rest_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(5))
        for heads in (self.beta_heads, self.gamma_heads, self.rest_heads):
            for head in heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, 2.2)

    @staticmethod
    def apply_heads(heads: nn.ModuleList, values: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(values) for head in heads], dim=-1)

    def feature_streams(self, energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = energy.index_select(-1, self.beta_indices)
        if self.beta_shuffle_bins:
            beta = torch.roll(beta, self.beta_shuffle_bins, dims=1)
        gamma = energy.index_select(-1, self.gamma_indices)
        return beta, gamma

    def direct_features(self, energy: torch.Tensor) -> torch.Tensor:
        beta, gamma = self.feature_streams(energy)
        if self.fusion == "beta_only":
            source = beta
        elif self.fusion == "gamma_only":
            source = gamma
        else:
            source = torch.cat((beta, gamma), dim=-1)
        history = source.unfold(1, self.history, 1)
        return history.permute(0, 1, 3, 2).flatten(start_dim=2)

    def forward_components(
        self, energy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = self.direct_features(energy)
        direct = self.direct((flattened - self.feature_mean) / self.feature_scale)
        beta, gamma = self.feature_streams(energy)
        beta_hidden = self.beta_temporal(self.beta_projection(beta))[:, self.history - 1 :]
        gamma_hidden = self.gamma_temporal(self.gamma_projection(gamma))[:, self.history - 1 :]
        gate_logits = self.gate_head(beta_hidden)
        beta_value = self.apply_heads(self.beta_heads, beta_hidden)
        if self.fusion in {"gated", "shuffled_beta"}:
            scale, shift = self.film(beta_hidden).chunk(2, dim=-1)
            gamma_hidden = gamma_hidden * (1.0 + 0.25 * torch.tanh(scale)) + 0.25 * shift
        gamma_value = self.apply_heads(self.gamma_heads, gamma_hidden)
        rest_value = self.apply_heads(self.rest_heads, gamma_hidden)

        if self.fusion == "beta_only":
            amplitude = direct + beta_value
            prediction = amplitude
        elif self.fusion == "gamma_only":
            amplitude = direct + gamma_value
            prediction = amplitude
        elif self.fusion == "additive":
            amplitude = direct + beta_value + gamma_value
            prediction = amplitude
        else:
            amplitude = direct + gamma_value + 0.25 * beta_value
            probability = torch.sigmoid(gate_logits)
            gate = self.gate_floor + (1.0 - self.gate_floor) * probability
            prediction = gate * amplitude + 0.10 * rest_value
        return prediction, amplitude, gate_logits, rest_value

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        return self.forward_components(energy)[0]


@torch.inference_mode()
def predict(model: BetaGammaHeads, energy: torch.Tensor) -> tuple[np.ndarray, ...]:
    model.eval()
    values = model.forward_components(energy)
    prediction, amplitude, logits, residual = values
    return tuple(
        value.squeeze(0).float().cpu().numpy()
        for value in (prediction, amplitude, torch.sigmoid(logits), residual)
    )


def transformed_energy(model: BetaGammaHeads, energy: np.ndarray) -> np.ndarray:
    beta = energy[:, model.beta_indices.cpu().numpy()]
    if model.beta_shuffle_bins:
        beta = np.roll(beta, model.beta_shuffle_bins, axis=0)
    gamma = energy[:, model.gamma_indices.cpu().numpy()]
    if model.fusion == "beta_only":
        return beta
    if model.fusion == "gamma_only":
        return gamma
    return np.concatenate((beta, gamma), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/beta_gamma_heads_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument(
        "--fusion",
        choices=("beta_only", "gamma_only", "additive", "gated", "shuffled_beta"),
        default="gated",
    )
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--movement-loss-weight", type=float, default=3.0)
    parser.add_argument("--state-weight", type=float, default=0.25)
    parser.add_argument("--gate-pretrain-epochs", type=int, default=0)
    parser.add_argument("--amplitude-weight", type=float, default=0.25)
    parser.add_argument("--derivative-correlation-weight", type=float, default=0.10)
    parser.add_argument("--gate-floor", type=float, default=0.10)
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
    output = args.output_root / args.fusion / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_energy = np.load(csp_root / "train_csp_energy.npy")
    test_energy = np.load(csp_root / "test_csp_energy.npy")
    target = np.load(root / f"train_glove_{args.target}.npy")
    test_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    raw_train = np.load(root / "train_glove_25hz_raw.npy")
    raw_test = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    shuffle = 137 if args.fusion == "shuffled_beta" else 0
    model = BetaGammaHeads(
        train_energy.shape[1], args.history, args.fusion, shuffle, gate_floor=args.gate_floor
    ).to(device)
    direct_energy = transformed_energy(model, train_energy)
    train_x_all = lagged(direct_energy, args.history)
    train_count = split - offset
    train_y = target[offset:split]
    ridge_audit = initialize_ridge(
        model, train_x_all[:train_count], train_y, args.top_features, device
    )

    train_tensor = torch.from_numpy(train_energy[:split]).unsqueeze(0).to(device)
    validation_tensor = torch.from_numpy(train_energy[split - offset :]).unsqueeze(0).to(device)
    test_tensor = torch.from_numpy(test_energy).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(train_y).unsqueeze(0).to(device)
    state_tensor = (target_tensor >= args.movement_threshold).float()
    movement_counts = state_tensor.sum(dim=1)
    pos_weight = ((state_tensor.shape[1] - movement_counts) / movement_counts.clamp_min(1)).squeeze(0)

    gate_pretraining: list[dict[str, float]] = []
    if args.gate_pretrain_epochs > 0 and args.fusion in {"gated", "shuffled_beta"}:
        gate_parameters = list(model.beta_projection.parameters())
        gate_parameters += list(model.beta_temporal.parameters())
        gate_parameters += list(model.gate_head.parameters())
        gate_optimizer = torch.optim.AdamW(gate_parameters, lr=1e-3, weight_decay=1e-4)
        validation_state = (target[split:] >= args.movement_threshold).astype(np.float32)
        best_gate_f1 = -float("inf")
        best_gate_state = copy.deepcopy(model.state_dict())
        for epoch in range(1, args.gate_pretrain_epochs + 1):
            model.train()
            _, _, logits, _ = model.forward_components(train_tensor)
            gate_loss = F.binary_cross_entropy_with_logits(
                logits, state_tensor, pos_weight=pos_weight
            )
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate_parameters, 1.0)
            gate_optimizer.step()
            if epoch == 1 or epoch % 5 == 0:
                probability = predict(model, validation_tensor)[2]
                metrics = state_metrics(probability, validation_state)
                macro_f1 = float(np.mean([value["f1"] for value in metrics.values()]))
                gate_pretraining.append(
                    {"epoch": epoch, "loss": float(gate_loss.detach()), "validation_macro_f1": macro_f1}
                )
                print(
                    f"fusion={args.fusion} gate_epoch={epoch} gate_f1={macro_f1:.4f}",
                    flush=True,
                )
                if macro_f1 > best_gate_f1:
                    best_gate_f1 = macro_f1
                    best_gate_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_gate_state)

    initial_validation = predict(model, validation_tensor)[0]
    best_score = morphology_score(initial_validation, target[split:], args.movement_threshold)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    executable = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        estimate, amplitude, logits, rest_value = executable.forward_components(train_tensor)
        weights = 1.0 + (args.movement_loss_weight - 1.0) * state_tensor
        level = torch.mean(weights * (estimate - target_tensor).square())
        state_loss = F.binary_cross_entropy_with_logits(logits, state_tensor, pos_weight=pos_weight)
        amplitude_loss = torch.sum(state_tensor * (amplitude - target_tensor).square()) / state_tensor.sum().clamp_min(1)
        derivative_correlation = correlation_loss(
            estimate[:, 1:] - estimate[:, :-1], target_tensor[:, 1:] - target_tensor[:, :-1]
        )
        rest_penalty = torch.sum((1.0 - state_tensor) * rest_value.square()) / (1.0 - state_tensor).sum().clamp_min(1)
        loss = (
            level
            + args.state_weight * state_loss
            + args.amplitude_weight * amplitude_loss
            + args.derivative_correlation_weight * derivative_correlation
            + 0.05 * rest_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0:
            validation_prediction = predict(executable, validation_tensor)[0]
            score = morphology_score(validation_prediction, target[split:], args.movement_threshold)
            history.append({"epoch": epoch, "loss": float(loss.detach()), "validation_morphology": score})
            print(f"fusion={args.fusion} epoch={epoch} morphology={score:.4f}", flush=True)
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
    validation_prediction, validation_amplitude, validation_probability, validation_residual = predict(model, validation_tensor)
    test_prediction, test_amplitude, test_probability, test_residual = predict(model, test_tensor)
    report = {
        "subject": args.subject,
        "fusion": args.fusion,
        "beta_band_hz": [12, 30],
        "high_gamma_bands_hz": [[65, 95], [105, 145]],
        "beta_shuffle_bins": shuffle,
        "gate_floor": args.gate_floor,
        "best_epoch": best_epoch,
        "best_validation_morphology": best_score,
        "gate_pretrain_epochs": args.gate_pretrain_epochs,
        "gate_pretraining": gate_pretraining,
        "validation_cleaned_metrics": trajectory_metrics(validation_prediction, target[split:]),
        "test_cleaned_metrics": trajectory_metrics(test_prediction, test_target),
        "validation_raw_metrics": trajectory_metrics(validation_prediction, raw_train[split:]),
        "test_raw_metrics": trajectory_metrics(test_prediction, raw_test),
        "validation_state_metrics": state_metrics(
            validation_probability, (target[split:] >= args.movement_threshold).astype(np.float32)
        ),
        "ridge_audit": ridge_audit,
        "history": history,
        "compiled": True,
        "compile_mode": "reduce-overhead",
    }
    for name, value in (
        ("validation_prediction", validation_prediction),
        ("test_prediction", test_prediction),
        ("validation_amplitude", validation_amplitude),
        ("test_amplitude", test_amplitude),
        ("validation_state_probability", validation_probability),
        ("test_state_probability", test_probability),
        ("validation_rest_residual", validation_residual),
        ("test_rest_residual", test_residual),
    ):
        np.save(output / f"{name}.npy", value, allow_pickle=False)
    torch.save(best_state, output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
