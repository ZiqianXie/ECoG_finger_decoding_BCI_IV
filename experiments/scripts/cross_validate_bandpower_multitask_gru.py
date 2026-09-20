#!/usr/bin/env python3
"""Outer-split multitask GRU over all-electrode band-power features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cross_validate_electrode_bandpower_ridge import load_band_energy, pearson
from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET


def interval_rows(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def sample_interval_batch(
    rng: np.random.Generator,
    intervals: list[list[int]],
    batch_size: int,
    sequence_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample intervals uniformly and pad short intervals with a false mask."""
    chosen = rng.integers(0, len(intervals), size=batch_size)
    indices = np.empty((batch_size, sequence_steps), dtype=np.int64)
    mask = np.zeros((batch_size, sequence_steps), dtype=bool)
    for row, interval_index in enumerate(chosen):
        start, stop = intervals[int(interval_index)]
        length = stop - start
        if length >= sequence_steps:
            begin = int(rng.integers(start, stop - sequence_steps + 1))
            indices[row] = np.arange(begin, begin + sequence_steps)
            mask[row] = True
        else:
            indices[row, :length] = np.arange(start, stop)
            indices[row, length:] = stop - 1
            mask[row, :length] = True
    return indices, mask


def ridge_skip(
    features: np.ndarray,
    targets: np.ndarray,
    penalty: float,
) -> np.ndarray:
    gram = features.T @ features
    cross = features.T @ targets
    return np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0], dtype=np.float32), cross
    ).astype(np.float32)


def screen_features(
    features: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select current-bin features by outer-training absolute correlation."""
    values = np.asarray(features[training], dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    centered_target = target[training] - target[training].mean()
    denominator = np.sqrt(np.sum(centered * centered, axis=0)) * np.sqrt(
        np.sum(centered_target * centered_target)
    )
    correlation = np.divide(
        centered.T @ centered_target,
        denominator,
        out=np.zeros(values.shape[1], dtype=np.float32),
        where=denominator > 0,
    )
    count = min(count, correlation.size)
    selected = np.argpartition(np.abs(correlation), -count)[-count:]
    return selected[np.argsort(np.abs(correlation[selected]))[::-1]].astype(np.int64)


class BandpowerMultitaskGRU(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        direct_coefficients: np.ndarray,
    ) -> None:
        super().__init__()
        output_count = direct_coefficients.shape[1]
        self.direct = nn.Linear(feature_count, output_count, bias=False)
        with torch.no_grad():
            self.direct.weight.copy_(torch.from_numpy(direct_coefficients.T))
        self.direct.requires_grad_(False)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Linear(2 * hidden_size, output_count)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        direct = self.direct(features)
        recurrent, _ = self.gru(self.input_projection(features))
        return direct + self.output(recurrent)


@torch.inference_mode()
def predict_intervals(
    model: BandpowerMultitaskGRU,
    features: torch.Tensor,
    intervals: list[list[int]],
    rows: int,
) -> np.ndarray:
    model.eval()
    prediction = np.full(
        (rows, model.direct.out_features), np.nan, dtype=np.float32
    )
    for start, stop in intervals:
        estimate = model(features[None, start:stop]).squeeze(0).cpu().numpy()
        prediction[start:stop] = estimate
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--band-energy-cache", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--sequence-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ridge-penalty", type=float, default=100.0)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--movement-weight", type=float, default=2.0)
    parser.add_argument("--target-loss-weight", type=float, default=2.0)
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--top-features", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.095)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(
        args.hidden_size,
        args.sequence_steps,
        args.batch_size,
        args.updates,
        args.top_features,
    ) <= 0:
        parser.error("model and training sizes must be positive")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    target_finger = list(FINGER_NAMES).index(args.finger)
    energy = np.asarray(
        load_band_energy(
            args.prepared_root,
            args.subject,
            40,
            args.band_energy_cache,
        ),
        dtype=np.float32,
    )
    rows = int(
        np.load(
            args.cache_root / "cache" / "outer0" / "outer" / "target.npy",
            mmap_mode="r",
        ).shape[0]
    )
    all_features_np = energy[OFFSET : OFFSET + rows]
    raw = np.array(
        np.load(
            args.prepared_root
            / f"sub{args.subject}"
            / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + rows],
        dtype=np.float32,
        copy=True,
    )
    stitched = np.full(rows, np.nan, dtype=np.float32)
    outer_records = []
    for outer_fold in range(3):
        started = time.perf_counter()
        split = json.loads(
            (
                args.cache_root
                / "cache"
                / f"outer{outer_fold}"
                / "outer"
                / "split.json"
            ).read_text()
        )
        training_intervals = split["training_intervals"]
        validation_intervals = split["validation_intervals"]
        training = interval_rows(training_intervals)
        validation = interval_rows(validation_intervals)
        if args.target_only:
            selected_features = screen_features(
                all_features_np,
                raw[:, target_finger],
                training,
                args.top_features,
            )
            features_np = all_features_np[:, selected_features]
            target_np = raw[:, [target_finger]]
            local_target_finger = 0
        else:
            selected_features = np.arange(all_features_np.shape[1])
            features_np = all_features_np
            target_np = raw
            local_target_finger = target_finger
        feature_mean = features_np[training].mean(axis=0)
        feature_scale = features_np[training].std(axis=0)
        feature_scale = np.maximum(feature_scale, 1.0e-6)
        target_mean = target_np[training].mean(axis=0)
        target_scale = np.maximum(target_np[training].std(axis=0), 1.0e-6)
        standardized_features = (
            (features_np - feature_mean) / feature_scale
        ).astype(np.float32)
        standardized_target = ((target_np - target_mean) / target_scale).astype(
            np.float32
        )
        direct = ridge_skip(
            standardized_features[training],
            standardized_target[training],
            args.ridge_penalty,
        )
        model = BandpowerMultitaskGRU(
            standardized_features.shape[1], args.hidden_size, direct
        ).to(device)
        features = torch.from_numpy(standardized_features).to(device)
        target = torch.from_numpy(standardized_target).to(device)
        raw_device = torch.from_numpy(target_np).to(device)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        output_weights = torch.ones(target_np.shape[1], device=device)
        output_weights[local_target_finger] = args.target_loss_weight
        model.train()
        losses = []
        for update in range(1, args.updates + 1):
            indices_np, mask_np = sample_interval_batch(
                rng,
                training_intervals,
                args.batch_size,
                args.sequence_steps,
            )
            indices = torch.from_numpy(indices_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            prediction = model(features[indices])
            observed = target[indices]
            movement = raw_device[indices] > args.movement_threshold
            weights = (1.0 + args.movement_weight * movement) * output_weights
            weights = weights * mask[..., None]
            loss = (weights * (prediction - observed).square()).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if update % 500 == 0:
                print(
                    json.dumps(
                        {
                            "outer_fold": outer_fold,
                            "update": update,
                            "mean_recent_loss": float(np.mean(losses[-100:])),
                        }
                    ),
                    flush=True,
                )

        estimate_standardized = predict_intervals(
            model, features, validation_intervals, rows
        )
        estimate = (
            estimate_standardized[validation, local_target_finger]
            * target_scale[local_target_finger]
            + target_mean[local_target_finger]
        )
        estimate = np.maximum(estimate, 0.0)
        stitched[validation] = estimate
        record = {
            "outer_fold": outer_fold,
            "selected_feature_count": int(selected_features.size),
            "outer_raw_pcc": pearson(estimate, raw[validation, target_finger]),
            "final_mean_loss": float(np.mean(losses[-100:])),
            "runtime_seconds": time.perf_counter() - started,
        }
        outer_records.append(record)
        print(json.dumps(record), flush=True)

    observed = np.isfinite(stitched)
    pcc = pearson(stitched[observed], raw[observed, target_finger])
    gain = pcc - args.fixed_reference_pcc
    mode = (
        "target-only GRU over outer-training-screened electrode-band features"
        if args.target_only
        else "multitask GRU over every electrode-band feature"
    )
    report = {
        "protocol": (
            f"fixed-schedule outer-split bidirectional {mode}; scaling and ridge "
            "skip refit in outer training; released test untouched"
        ),
        "released_test_touched": False,
        "outer_validation_used_for_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "settings": vars(args) | {"cache_root": str(args.cache_root), "prepared_root": str(args.prepared_root), "band_energy_cache": str(args.band_energy_cache), "output": str(args.output)},
        "outer_records": outer_records,
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_fixed_reference_gain_threshold": gain >= args.gain_threshold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "selected_oof.npy", stitched, allow_pickle=False)
    np.save(
        args.output / "target_oof.npy",
        raw[:, target_finger],
        allow_pickle=False,
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
