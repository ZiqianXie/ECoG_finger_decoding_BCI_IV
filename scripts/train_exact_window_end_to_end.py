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
from ecog_decoding.models import AsymmetricWaveletPacketEnergy, WaveletPacketEnergy
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
        wavelet_levels: int = 3,
        frontend: str = "wavelet",
    ) -> None:
        super().__init__()
        self.spatial = nn.Conv1d(input_channels, component_count, 1, bias=False)
        if frontend == "asymmetric":
            self.wavelet = AsymmetricWaveletPacketEnergy(
                wavelet="bior6.8",
                split_parents=(1, 3),
                kernel_size=17,
                trainable=True,
                padding_mode="constant",
                energy_window_samples=40,
                energy_stride_samples=40,
            )
        elif frontend == "wavelet":
            self.wavelet = WaveletPacketEnergy(
                wavelet="bior6.8",
                levels=wavelet_levels,
                kernel_size=17,
                trainable=True,
                padding_mode="constant",
                energy_window_samples=40,
                energy_stride_samples=40,
            )
        else:
            raise ValueError(f"unsupported frontend {frontend!r}")
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
    parser.add_argument("--wavelet-levels", type=int, choices=(3, 4), default=3)
    parser.add_argument("--frontend", choices=("wavelet", "asymmetric"), default="wavelet")
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--frontend-learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--spatial-learning-rate",
        type=float,
        default=None,
        help="optional ICA projection rate; defaults to --frontend-learning-rate",
    )
    parser.add_argument(
        "--wavelet-learning-rate",
        type=float,
        default=None,
        help="optional wavelet/LMP rate; defaults to --frontend-learning-rate",
    )
    parser.add_argument(
        "--frontend-warmup-epochs",
        type=int,
        default=0,
        help="keep ICA/wavelet parameters frozen for this many head-training epochs",
    )
    parser.add_argument("--direct-learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--learning-rate-decay-per-epoch",
        type=float,
        default=0.0,
        help="inverse-time decay: lr(epoch) = lr0 / (1 + decay * (epoch - 1))",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--prediction-chunk-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="optional model-initialization seed; defaults to --seed",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="optional minibatch-order seed; defaults to --seed",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.frontend_warmup_epochs < 0:
        parser.error("--frontend-warmup-epochs must be nonnegative")
    if args.learning_rate_decay_per_epoch < 0:
        parser.error("--learning-rate-decay-per-epoch must be nonnegative")
    for name in (
        "learning_rate",
        "frontend_learning_rate",
        "direct_learning_rate",
        "weight_decay",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.spatial_learning_rate is not None and args.spatial_learning_rate < 0:
        parser.error("--spatial-learning-rate must be nonnegative")
    if args.wavelet_learning_rate is not None and args.wavelet_learning_rate < 0:
        parser.error("--wavelet-learning-rate must be nonnegative")
    spatial_learning_rate = (
        args.frontend_learning_rate
        if args.spatial_learning_rate is None
        else args.spatial_learning_rate
    )
    wavelet_learning_rate = (
        args.frontend_learning_rate
        if args.wavelet_learning_rate is None
        else args.wavelet_learning_rate
    )

    model_seed = args.seed if args.model_seed is None else args.model_seed
    shuffle_seed = args.seed if args.shuffle_seed is None else args.shuffle_seed
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    shuffle_rng = np.random.RandomState(shuffle_seed)
    if str(args.device).startswith("cuda"):
        torch.set_float32_matmul_precision("high")
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
            train_ecog.shape[1],
            ica.shape[0],
            indices,
            mean,
            scale,
            args.hidden_size,
            wavelet_levels=args.wavelet_levels,
            frontend=args.frontend,
        ).to(device)
        with torch.no_grad():
            model.spatial.weight[:, :, 0].copy_(torch.from_numpy(ica).to(device))
            model.direct.weight.copy_(torch.from_numpy(ridge_weight[None]).to(device))
            model.direct.bias.fill_(target_mean)
        train_model: nn.Module = model
        if args.compile:
            train_model = torch.compile(model, mode="reduce-overhead")
        initial_spatial_lr = 0.0 if args.frontend_warmup_epochs > 0 else spatial_learning_rate
        initial_wavelet_lr = 0.0 if args.frontend_warmup_epochs > 0 else wavelet_learning_rate
        optimizer = torch.optim.AdamW(
            [
                {"params": model.spatial.parameters(), "lr": initial_spatial_lr},
                {"params": model.wavelet.parameters(), "lr": initial_wavelet_lr},
                {"params": model.direct.parameters(), "lr": args.direct_learning_rate},
                {"params": list(model.lstm.parameters()) + list(model.temporal.parameters()), "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        starts = np.arange(0, train_count - args.sequence_steps + 1, args.sequence_stride)
        initial_validation_estimate = predict(
            model,
            train_windows_all[train_count:],
            device,
            args.prediction_chunk_steps,
        )
        best_score = pearson(initial_validation_estimate, raw_train[train_count:, finger])
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = [
            {"epoch": 0, "validation_raw_r": best_score}
        ]
        print(
            f"finger={finger_name} epoch=0 initialized_val_r={best_score:.4f}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            decay_scale = 1.0 / (
                1.0 + args.learning_rate_decay_per_epoch * (epoch - 1)
            )
            optimizer.param_groups[0]["lr"] = (
                0.0
                if epoch <= args.frontend_warmup_epochs
                else spatial_learning_rate * decay_scale
            )
            optimizer.param_groups[1]["lr"] = (
                0.0
                if epoch <= args.frontend_warmup_epochs
                else wavelet_learning_rate * decay_scale
            )
            optimizer.param_groups[2]["lr"] = args.direct_learning_rate * decay_scale
            optimizer.param_groups[3]["lr"] = args.learning_rate * decay_scale
            if args.frontend_warmup_epochs > 0 and epoch == args.frontend_warmup_epochs + 1:
                for group in optimizer.param_groups[:2]:
                    for parameter in group["params"]:
                        optimizer.state.pop(parameter, None)
                print(
                    f"finger={finger_name} unfreeze_frontend epoch={epoch} "
                    f"spatial_lr={optimizer.param_groups[0]['lr']:g} "
                    f"wavelet_lr={optimizer.param_groups[1]['lr']:g}",
                    flush=True,
                )
            shuffle_rng.shuffle(starts)
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
            "initialized_validation_raw_r": history[0]["validation_raw_r"],
            "best_epoch": best_epoch,
            "validation_raw_r": best_score,
            "test_raw_r": pearson(test_prediction[:, finger], raw_test[:, finger]),
            "history": history,
        }

    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    summary = {
        "subject": args.subject,
        "method": (
            f"independent exact-window trainable ICA/{args.frontend} "
            f"plus {args.hidden_size}-unit LSTM"
        ),
        "target": args.target,
        "fingers": args.fingers,
        "wavelet_levels": args.wavelet_levels,
        "frontend": args.frontend,
        "selection_root": str(args.selection_root),
        "optimization": {
            "learning_rate": args.learning_rate,
            "frontend_learning_rate": args.frontend_learning_rate,
            "spatial_learning_rate": spatial_learning_rate,
            "wavelet_learning_rate": wavelet_learning_rate,
            "frontend_warmup_epochs": args.frontend_warmup_epochs,
            "direct_learning_rate": args.direct_learning_rate,
            "learning_rate_decay_per_epoch": args.learning_rate_decay_per_epoch,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "sequence_steps": args.sequence_steps,
            "sequence_stride": args.sequence_stride,
            "epochs": args.epochs,
            "validation_interval": args.validation_interval,
            "patience": args.patience,
            "seed": args.seed,
            "model_seed": model_seed,
            "shuffle_seed": shuffle_seed,
            "compiled_reduce_overhead": args.compile,
        },
        "per_finger": report,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
