#!/usr/bin/env python3
"""Supervised-contrastive pretraining on fixed ICA/wavelet history features."""

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
from ecog_decoding.training import trajectory_metrics


class ContrastiveRegressor(nn.Module):
    def __init__(self, input_features: int, width: int = 256, embedding: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_features, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, embedding),
            nn.GELU(),
        )
        self.output = nn.Linear(embedding, 5)

    def encode(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        values = self.encoder(x)
        return F.normalize(values, dim=-1) if normalize else values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode(x))


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(embedding.shape[0], dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * ~identity
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    return -(
        (positive * log_probability).sum(dim=1)[usable]
        / positive_count[usable]
    ).mean()


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    numerator = torch.sum(prediction * target, dim=0)
    denominator = torch.sqrt(
        torch.sum(prediction.square(), dim=0) * torch.sum(target.square(), dim=0) + 1e-7
    )
    return 1.0 - (numerator / denominator).mean()


def balanced_indices(
    labels: np.ndarray,
    per_class: int,
    generator: np.random.Generator,
) -> np.ndarray:
    selections = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        selections.append(generator.choice(candidates, per_class, replace=candidates.size < per_class))
    indices = np.concatenate(selections)
    generator.shuffle(indices)
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/ridge_target_benchmark_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/contrastive_regression_v2"))
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--selected-features", type=int, default=1024)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--regression-epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    root = args.prepared_root / f"sub{args.subject}"
    feature_root = args.feature_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    target = np.load(root / f"train_glove_{args.target}.npy")
    train_raw = np.load(root / "train_glove_25hz_raw.npy")
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    train_energy = np.load(feature_root / "train_initialized_energy.npy")
    test_energy = np.load(feature_root / "test_initialized_energy.npy")
    features_all = lagged(train_energy, args.history)
    test_features = lagged(test_energy, args.history)
    train_count = split - offset
    train_features = features_all[:train_count]
    validation_features = features_all[train_count:]
    train_target = target[offset:split]
    validation_target = target[split:]
    validation_raw = train_raw[split:]

    correlations = []
    centered_x = train_features - train_features.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(centered_x * centered_x, axis=0))
    for finger in range(train_target.shape[1]):
        centered_y = train_target[:, finger] - train_target[:, finger].mean()
        denominator = x_norm * np.linalg.norm(centered_y)
        correlations.append(
            np.abs(
                np.divide(
                    centered_x.T @ centered_y,
                    denominator,
                    out=np.zeros(train_features.shape[1], dtype=np.float32),
                    where=denominator > 0,
                )
            )
        )
    score = np.max(np.stack(correlations), axis=0)
    count = min(args.selected_features, score.size)
    selected = np.argpartition(score, -count)[-count:]
    selected = selected[np.argsort(score[selected])[::-1]]
    mean = train_features[:, selected].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_features[:, selected].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    train_x = torch.from_numpy((train_features[:, selected] - mean) / scale).to(device)
    validation_x = torch.from_numpy((validation_features[:, selected] - mean) / scale).to(device)
    test_x = torch.from_numpy((test_features[:, selected] - mean) / scale).to(device)
    train_y = torch.from_numpy(train_target.astype(np.float32)).to(device)
    labels = np.argmax(train_target, axis=1).astype(np.int64) + 1
    labels[np.max(train_target, axis=1) < 0.10] = 0

    model = ContrastiveRegressor(count).to(device)
    executable = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    pretrain_history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        indices = balanced_indices(labels, per_class=64, generator=generator)
        batch = train_x[indices]
        batch_labels = torch.from_numpy(labels[indices]).to(device)
        view_a = F.dropout(batch, p=0.10, training=True) + 0.02 * torch.randn_like(batch)
        view_b = F.dropout(batch, p=0.10, training=True) + 0.02 * torch.randn_like(batch)
        embedding = torch.cat(
            (model.encode(view_a, normalize=True), model.encode(view_b, normalize=True)),
            dim=0,
        )
        repeated_labels = torch.cat((batch_labels, batch_labels), dim=0)
        loss = supervised_contrastive_loss(embedding, repeated_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pretrain_history.append(float(loss.detach()))
        if epoch == 1 or epoch % 20 == 0:
            print(f"contrastive_epoch={epoch} loss={float(loss):.5f}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_score = -float("inf")
    best_state = None
    regression_history = []
    batch_size = 512
    for epoch in range(1, args.regression_epochs + 1):
        order = generator.permutation(train_x.shape[0])
        model.train()
        losses = []
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            prediction = executable(train_x[indices])
            level = F.mse_loss(prediction, train_y[indices])
            loss = level + 0.20 * correlation_loss(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.regression_epochs:
            model.eval()
            with torch.inference_mode():
                validation_prediction = executable(validation_x).float().cpu().numpy()
            metrics = trajectory_metrics(validation_prediction, validation_raw)
            score = float(metrics["pearson_historical_four"])
            regression_history.append(
                {"epoch": epoch, "loss": float(np.mean(losses)), "validation_raw": metrics}
            )
            print(f"regression_epoch={epoch} val_r4={score:.4f}", flush=True)
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = executable(validation_x).float().cpu().numpy()
        test_prediction = executable(test_x).float().cpu().numpy()
    report = {
        "subject": args.subject,
        "method": "balanced supervised contrastive movement-state pretraining followed by joint regression",
        "target": args.target,
        "selected_features": count,
        "best_validation_historical_four": best_score,
        "validation_raw_metrics": trajectory_metrics(validation_prediction, validation_raw),
        "test_raw_metrics": trajectory_metrics(test_prediction, test_raw),
        "pretrain_history": pretrain_history,
        "regression_history": regression_history,
    }
    np.save(output / "selected_feature_indices.npy", selected, allow_pickle=False)
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    torch.save(model.state_dict(), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["test_raw_metrics"]), flush=True)


if __name__ == "__main__":
    main()
