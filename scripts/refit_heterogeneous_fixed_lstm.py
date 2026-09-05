#!/usr/bin/env python3
"""Refit one six-seed member on selected joint ICA-wavelet/CSP features."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from refit_frozen_event_model import frozen_epoch, resolve_options
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from train_event_grouped_lars_e2e_nested import padded_batches, starts_from_intervals
from train_exact_window_end_to_end import ExactWindowFingerDecoder


@torch.inference_mode()
def predict(model: ExactWindowFingerDecoder, features: torch.Tensor) -> np.ndarray:
    model.eval()
    return model.decode(features[None])[0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/heterogeneous_full_features_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/full_development_event_refit.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heterogeneous_six_seed_refit_v1"))
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    started = time.perf_counter()

    ensemble_map = yaml.safe_load(args.ensemble_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    options = resolve_options(ensemble_map, args.subject, args.finger)
    selected_epoch, outer_epochs, epoch_rule = frozen_epoch(
        options["input_root"], args.subject, args.finger, args.seed,
        options["epoch_reference_seeds"],
    )
    feature_dir = args.feature_root / f"sub{args.subject}" / args.finger
    selection = np.load(feature_dir / "selection.npz")
    train_features_numpy = np.load(feature_dir / "train_selected_features.npy")
    test_features_numpy = np.load(feature_dir / "test_selected_features.npy")
    train_features = torch.as_tensor(train_features_numpy, dtype=torch.float32, device=device)
    test_features = torch.as_tensor(test_features_numpy, dtype=torch.float32, device=device)
    feature_count = train_features.shape[1]

    subject_targets = target_map.get(args.subject, target_map.get(str(args.subject)))
    target_policy = str(subject_targets[args.finger]).removesuffix("_split_safe")
    finger_index = list(FINGER_NAMES).index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    target_numpy = np.load(prepared / f"train_glove_{target_policy}.npy")[24:, finger_index]
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[24:, finger_index]
    target = torch.as_tensor(target_numpy, dtype=torch.float32, device=device)

    model = ExactWindowFingerDecoder(
        input_channels=1,
        component_count=1,
        selected_indices=np.arange(feature_count),
        feature_mean=np.asarray(selection["feature_mean"]),
        feature_scale=np.asarray(selection["feature_scale"]),
        hidden_size=args.hidden_size,
        frontend="wavelet",
        head_initialization="lars_linear_regime",
        output_activation=str(options["output_activation"]),
    ).to(device)
    model.initialize_lars_linear_regime(
        np.asarray(selection["coefficients"]),
        float(selection["intercept"]),
        near_zero_std=args.near_zero_std,
    )
    initialized = predict(model, train_features)
    parameters = list(model.lstm.parameters()) + list(model.temporal.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
    )
    decode = model.decode
    if args.compile:
        decode = torch.compile(decode, mode="reduce-overhead")
    sequence_steps = int(options["sequence_steps"])
    sequence_stride = int(options["sequence_stride"])
    starts = starts_from_intervals([[0, train_features.shape[0]]], sequence_steps, sequence_stride)
    offsets = torch.arange(sequence_steps, device=device)
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    for epoch in range(1, selected_epoch + 1):
        model.train()
        epoch_losses = []
        for chosen in padded_batches(starts, int(options["batch_size"]), rng):
            origins = torch.as_tensor(chosen, device=device)
            indices = origins[:, None] + offsets[None]
            prediction = decode(train_features[indices])
            loss = (prediction - target[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))

    fitted_train = predict(model, train_features)
    prediction = predict(model, test_features)
    output = args.output_root / f"sub{args.subject}" / args.finger / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "released_test_prediction.npy", prediction)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "selected_epoch": selected_epoch,
        "outer_fold_selected_epochs_by_reference_seed": outer_epochs,
        "epoch_rule": epoch_rule,
    }, output / "model.pt")

    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:, finger_index]
    cleaned_test = np.load(prepared / f"test_glove_{target_policy}.npy")[24:, finger_index]
    metrics = morphology_metrics(prediction, cleaned_test, movement_groups(cleaned_test, 0.08))
    metrics["raw_pcc"] = pearson(prediction, raw_test)
    report = {
        "protocol": "six-seed fixed-joint-feature LARS-initialized LSTM full-development refit",
        "subject": args.subject,
        "finger": args.finger,
        "seed": args.seed,
        "selected_epoch": selected_epoch,
        "epoch_rule": epoch_rule,
        "outer_fold_selected_epochs_by_reference_seed": outer_epochs,
        "epoch_source": "matched baseline OOF model; joint nonlinear duration was not reselected",
        "target_policy": target_policy,
        "feature_count": int(feature_count),
        "initialized_full_train_raw_pcc": pearson(initialized, raw_train),
        "fitted_full_train_raw_pcc": pearson(fitted_train, raw_train),
        "training_losses": losses,
        "released_test_used_for_feature_family_selection": False,
        "released_test_role": "retrospective paper-comparison evaluation",
        "released_test_metrics": metrics,
        "prediction_sd": float(np.std(prediction)),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "subject": args.subject, "finger": args.finger, "seed": args.seed,
        "selected_epoch": selected_epoch, "raw_test_pcc": metrics["raw_pcc"],
        "runtime_seconds": report["runtime_seconds"],
    }), flush=True)


if __name__ == "__main__":
    main()
