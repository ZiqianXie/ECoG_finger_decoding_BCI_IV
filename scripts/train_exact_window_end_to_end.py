#!/usr/bin/env python3
"""Fine-tune the paper front end on independent 1 s windows, one finger at a time.

The cached exact-window features are used only for LARS feature selection and
normalization.  During training the selected features are recomputed from raw
ECoG so the ICA-initialized spatial projection and bior6.8 wavelet packet can
adapt jointly with the 10-unit LSTM.
"""

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
from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.training import FINGER_NAMES


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


class ExactWindowFingerDecoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        component_count: int,
        selected_indices: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(input_channels, component_count, 1, bias=False)
        self.wavelet = WaveletPacketEnergy(
            wavelet="bior6.8",
            levels=3,
            kernel_size=17,
            trainable=True,
            padding_mode="constant",
            energy_window_samples=40,
            energy_stride_samples=40,
        )
        self.register_buffer("selected_indices", torch.as_tensor(selected_indices, dtype=torch.long))
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(feature_scale, dtype=torch.float32))
        self.direct = nn.Linear(selected_indices.size, 1)
        self.lstm = nn.LSTM(selected_indices.size, hidden_size, batch_first=True)
        self.temporal = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    def extract(self, windows: torch.Tensor) -> torch.Tensor:
        """Return selected features for (batch, steps, channels, 1000)."""
        batch, steps, channels, samples = windows.shape
        flattened = windows.reshape(batch * steps, channels, samples)
        spatial = self.spatial(flattened)
        energy = self.wavelet(spatial).flatten(1)
        selected = energy.index_select(1, self.selected_indices)
        return selected.reshape(batch, steps, -1)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.feature_mean) / self.feature_scale
        direct = self.direct(standardized)
        recurrent, _ = self.lstm(standardized)
        return torch.relu(direct + self.temporal(recurrent)).squeeze(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.decode(self.extract(windows))


def make_windows(ecog: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(
        ecog, window_shape=window_samples, axis=0
    )[::stride_samples]


@torch.inference_mode()
def predict(
    model: ExactWindowFingerDecoder,
    windows: np.ndarray,
    device: torch.device,
    chunk_steps: int,
) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, windows.shape[0], chunk_steps):
        batch = torch.from_numpy(np.ascontiguousarray(windows[start : start + chunk_steps]))
        features = model.extract(batch[None].to(device)).squeeze(0)
        chunks.append(features.cpu())
    features = torch.cat(chunks, dim=0)[None].to(device)
    return model.decode(features).squeeze(0).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_sklearn_v1"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/ica_sklearn_constant_v1"))
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/exact_window_end_to_end_v1"))
    parser.add_argument("--target", default="local_w1_q10")
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=["index", "middle"])
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--frontend-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--direct-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--prediction-chunk-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    prepared = args.prepared_root / f"sub{args.subject}"
    feature_dir = args.feature_root / f"sub{args.subject}"
    selection_dir = args.selection_root / f"sub{args.subject}"
    ica_dir = args.ica_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_count = split - offset
    train_ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(prepared / "test_ecog.npy", mmap_mode="r")
    train_windows_all = make_windows(train_ecog, args.window_samples, args.stride_samples)
    test_windows = make_windows(test_ecog, args.window_samples, args.stride_samples)
    # Row zero of the exact feature cache predicts glove bin history-1.
    target = np.load(prepared / f"train_glove_{args.target}.npy")[offset:]
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[offset:]
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    fixed_train = np.load(feature_dir / "train_initialized_window_features.npy", mmap_mode="r")
    ica = np.load(ica_dir / "fastica_unmixing.npy")
    selection = json.loads((selection_dir / "summary.json").read_text())
    validation_prediction = np.full_like(raw_train[train_count:], np.nan, dtype=np.float32)
    test_prediction = np.full_like(raw_test, np.nan, dtype=np.float32)
    report: dict[str, object] = {}

    for finger_name in args.fingers:
        finger = list(FINGER_NAMES).index(finger_name)
        indices = np.asarray(
            selection["per_finger"][finger_name]["selected_source_indices"], dtype=np.int64
        )
        initial = np.asarray(fixed_train[:train_count, indices], dtype=np.float32)
        mean = initial.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = initial.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        y_train = np.asarray(target[:train_count, finger], dtype=np.float32)
        _, _, target_mean, ridge_weight = ridge_fit(initial, y_train, 1.0e-3, device)
        del initial

        model = ExactWindowFingerDecoder(
            train_ecog.shape[1], ica.shape[0], indices, mean, scale, args.hidden_size
        ).to(device)
        with torch.no_grad():
            model.spatial.weight[:, :, 0].copy_(torch.from_numpy(ica).to(device))
            model.direct.weight.copy_(torch.from_numpy(ridge_weight[None]).to(device))
            model.direct.bias.fill_(target_mean)
        train_model: nn.Module = model
        if args.compile:
            train_model = torch.compile(model, mode="reduce-overhead")
        optimizer = torch.optim.AdamW(
            [
                {"params": model.spatial.parameters(), "lr": args.frontend_learning_rate},
                {"params": model.wavelet.parameters(), "lr": args.frontend_learning_rate},
                {"params": model.direct.parameters(), "lr": args.direct_learning_rate},
                {"params": list(model.lstm.parameters()) + list(model.temporal.parameters()), "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        starts = np.arange(0, train_count - args.sequence_steps + 1, args.sequence_stride)
        best_score = -float("inf")
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            np.random.shuffle(starts)
            model.train()
            losses: list[float] = []
            for begin in range(0, starts.size, args.batch_size):
                selected_starts = starts[begin : begin + args.batch_size]
                xb = np.stack(
                    [np.ascontiguousarray(train_windows_all[s : s + args.sequence_steps]) for s in selected_starts]
                )
                yb = np.stack([y_train[s : s + args.sequence_steps] for s in selected_starts])
                prediction = train_model(torch.from_numpy(xb).to(device))
                observed = torch.from_numpy(yb).to(device)
                loss = torch.mean((prediction - observed).square())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if epoch == 1 or epoch % args.validation_interval == 0:
                estimate = predict(
                    model,
                    train_windows_all[train_count:],
                    device,
                    args.prediction_chunk_steps,
                )
                score = pearson(estimate, raw_train[train_count:, finger])
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_raw_r": score})
                print(
                    f"finger={finger_name} epoch={epoch} loss={np.mean(losses):.6f} "
                    f"val_r={score:.4f} best={best_score:.4f}",
                    flush=True,
                )
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
        validation_prediction[:, finger] = predict(
            model, train_windows_all[train_count:], device, args.prediction_chunk_steps
        )
        test_prediction[:, finger] = predict(model, test_windows, device, args.prediction_chunk_steps)
        torch.save(
            {"model_state_dict": best_state, "feature_indices": indices},
            output / f"{finger_name}.pt",
        )
        report[finger_name] = {
            "feature_count": int(indices.size),
            "best_epoch": best_epoch,
            "validation_raw_r": best_score,
            "test_raw_r": pearson(test_prediction[:, finger], raw_test[:, finger]),
            "history": history,
        }

    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    summary = {
        "subject": args.subject,
        "method": "independent exact-window trainable ICA/wavelet plus 10-unit LSTM",
        "target": args.target,
        "fingers": args.fingers,
        "selection_root": str(args.selection_root),
        "per_finger": report,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
