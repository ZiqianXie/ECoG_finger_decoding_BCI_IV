#!/usr/bin/env python3
"""Refit one frozen per-finger model on all development data, then test once."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from train_event_grouped_lars_e2e_nested import (
    build_model,
    compact_spatial_dictionary,
    fit_or_load_inner_lars,
    load_perfinger_cached_features,
    predict_intervals,
    resolve_perfinger_cache_root,
    train_fixed_epochs,
)
from train_event_grouped_lars_lstm import starts_from_intervals


def resolve_options(
    ensemble_map: dict[str, object], subject: int, finger: str
) -> dict[str, object]:
    defaults = ensemble_map.get("default", {})
    subjects = ensemble_map.get("subjects", {})
    if not isinstance(defaults, dict) or not isinstance(subjects, dict):
        raise TypeError("ensemble map default and subjects entries must be mappings")
    subject_map = subjects.get(subject, subjects.get(str(subject), {}))
    if not isinstance(subject_map, dict):
        raise TypeError(f"ensemble map subject {subject} must be a mapping")
    finger_map = subject_map.get(finger, {})
    if not isinstance(finger_map, dict):
        raise TypeError(f"ensemble map S{subject} {finger} must be a mapping")
    options = dict(defaults)
    options.update(finger_map)
    options["input_root"] = Path(str(options["input_root"]))
    options["seeds"] = tuple(int(value) for value in options["seeds"])
    options["epoch_reference_seeds"] = tuple(
        int(value) for value in options.get("epoch_reference_seeds", options["seeds"])
    )
    return options


def frozen_epoch(
    source_root: Path,
    subject: int,
    finger: str,
    seed: int,
    reference_seeds: tuple[int, ...],
) -> tuple[int, dict[str, list[int]], str]:
    """Reuse a seed's OOF duration when available, otherwise use the pooled rule."""
    epochs_by_seed: dict[str, list[int]] = {}
    pooled_epochs: list[int] = []
    for reference_seed in reference_seeds:
        seed_epochs = []
        for fold in range(3):
            path = (
                source_root
                / f"sub{subject}"
                / finger
                / f"fold{fold}"
                / f"seed{reference_seed}"
                / "summary.json"
            )
            seed_epochs.append(int(json.loads(path.read_text())["selected_epoch"]))
        epochs_by_seed[str(reference_seed)] = seed_epochs
        pooled_epochs.extend(seed_epochs)
    direct_epochs = epochs_by_seed.get(str(seed))
    if direct_epochs is not None:
        return (
            int(np.rint(np.median(direct_epochs))),
            epochs_by_seed,
            "rounded median of this seed's three outer-fold selected epochs",
        )
    return (
        int(np.rint(np.median(pooled_epochs))),
        epochs_by_seed,
        "rounded pooled median across outer folds and reference seeds",
    )


def update_equivalent_schedule(
    *,
    epochs_by_seed: dict[str, list[int]],
    seed: int,
    definition: dict[str, object],
    sequence_steps: int,
    sequence_stride: int,
    batch_size: int,
    unfrozen_batch_size: int,
    warmup_epochs: int,
    full_rows: int,
) -> tuple[int, int, dict[str, object]]:
    """Transfer fold-selected optimization exposure, rather than epoch count.

    A full-development epoch contains roughly twice as many sequence batches as
    an outer-fold epoch.  Reusing the fold epoch count therefore overtrains the
    final model.  Frozen-head and end-to-end updates are transferred separately
    because they use different batch sizes.
    """
    direct = epochs_by_seed.get(str(seed))
    selected = (
        [(fold, epoch) for fold, epoch in enumerate(direct)]
        if direct is not None
        else [
            (fold, epoch)
            for epochs in epochs_by_seed.values()
            for fold, epoch in enumerate(epochs)
        ]
    )
    records = []
    head_updates = []
    end_to_end_updates = []
    for fold, epoch in selected:
        intervals = definition["folds"][fold]["training_intervals_after_purge"]
        starts = starts_from_intervals(
            intervals, sequence_steps, sequence_stride
        )
        head_batches = int(np.ceil(starts.size / batch_size))
        end_to_end_batches = int(np.ceil(starts.size / unfrozen_batch_size))
        head_epoch_count = min(epoch, warmup_epochs)
        end_to_end_epoch_count = max(epoch - warmup_epochs, 0)
        head_update_count = head_epoch_count * head_batches
        end_to_end_update_count = end_to_end_epoch_count * end_to_end_batches
        head_updates.append(head_update_count)
        end_to_end_updates.append(end_to_end_update_count)
        records.append(
            {
                "fold": fold,
                "selected_epoch": epoch,
                "sequence_starts": int(starts.size),
                "frozen_batches_per_epoch": head_batches,
                "end_to_end_batches_per_epoch": end_to_end_batches,
                "frozen_updates": head_update_count,
                "end_to_end_updates": end_to_end_update_count,
            }
        )
    full_starts = starts_from_intervals(
        [[0, full_rows]], sequence_steps, sequence_stride
    )
    full_head_batches = int(np.ceil(full_starts.size / batch_size))
    full_end_to_end_batches = int(
        np.ceil(full_starts.size / unfrozen_batch_size)
    )

    def equivalent_epochs(updates: list[int], batches: int) -> int:
        median_updates = float(np.median(updates))
        if median_updates <= 0:
            return 0
        return max(1, int(np.rint(median_updates / batches)))

    transferred_warmup = equivalent_epochs(head_updates, full_head_batches)
    transferred_end_to_end = equivalent_epochs(
        end_to_end_updates, full_end_to_end_batches
    )
    selected_epoch = transferred_warmup + transferred_end_to_end
    audit = {
        "rule": (
            "rounded median fold optimizer updates transferred separately for "
            "the frozen-head and end-to-end stages"
        ),
        "reference": records,
        "median_frozen_updates": float(np.median(head_updates)),
        "median_end_to_end_updates": float(np.median(end_to_end_updates)),
        "full_sequence_starts": int(full_starts.size),
        "full_frozen_batches_per_epoch": full_head_batches,
        "full_end_to_end_batches_per_epoch": full_end_to_end_batches,
        "transferred_warmup_epochs": transferred_warmup,
        "transferred_end_to_end_epochs": transferred_end_to_end,
        "transferred_total_epochs": selected_epoch,
    }
    return selected_epoch, transferred_warmup, audit


def lars_chunks(row_count: int, count: int = 12) -> list[list[int]]:
    """Contiguous full-coverage chunks used only to choose the full-data LARS alpha."""
    edges = np.linspace(0, row_count, count + 1, dtype=np.int64)
    return [[int(start), int(stop)] for start, stop in zip(edges[:-1], edges[1:]) if stop > start]


def plot_full_trajectory(
    path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    time_axis = np.arange(raw.size) / 25.0
    figure, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    axes[0].plot(time_axis, raw, color="#94a3b8", linewidth=0.55, label="raw glove")
    axes[0].plot(time_axis, cleaned, color="black", linewidth=0.65, label="cleaned target")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(time_axis, cleaned, color="black", linewidth=0.65, label="cleaned target")
    axes[1].plot(time_axis, prediction, color="#2563eb", linewidth=0.65, label="prediction")
    axes[1].axhline(0.0, color="#cbd5e1", linewidth=0.5)
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xlabel("released-test time (s)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_events(
    path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    groups = movement_groups(cleaned, 0.08)
    selected = sorted(
        groups,
        key=lambda group: float(np.max(cleaned[group["start"] : group["stop"]])),
        reverse=True,
    )[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 10))
    for axis, group in zip(axes.flat, selected):
        start, stop = int(group["start"]), int(group["stop"])
        time_axis = np.arange(start, stop) / 25.0
        axis.plot(time_axis, raw[start:stop], color="#94a3b8", linewidth=0.7, label="raw glove")
        axis.plot(time_axis, cleaned[start:stop], color="black", linewidth=0.9, label="cleaned target")
        axis.plot(time_axis, prediction[start:stop], color="#2563eb", linewidth=0.9, label="prediction")
        axis.set_title(f"{start / 25:.1f}-{stop / 25:.1f} s", fontsize=9)
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ensemble-map", type=Path, default=Path("configs/final_event_ensemble.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument(
        "--spatial-cache-root",
        type=Path,
        default=None,
        help=(
            "optional full-refit cache containing train/test_features.npy and "
            "spatial_weights.npy under full/"
        ),
    )
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/frozen_event_full_refit_lars_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/frozen_event_full_refit_v1"))
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--prediction-chunk-steps", type=int, default=512)
    parser.add_argument("--near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--candidate-scale", type=float, default=1.0)
    parser.add_argument(
        "--epoch-override",
        type=int,
        default=None,
        help="diagnostic fixed epoch count; omit to use the frozen OOF median",
    )
    parser.add_argument(
        "--schedule-transfer",
        choices=("epochs", "updates"),
        default="epochs",
        help=(
            "reuse fold epoch counts verbatim or preserve the number of optimizer "
            "updates when refitting on the larger full-development set"
        ),
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path(
            "outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"
        ),
    )
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
    if args.seed not in options["seeds"]:
        raise ValueError(f"seed {args.seed} is not frozen for S{args.subject} {args.finger}")
    selected_epoch, outer_epochs, epoch_rule = frozen_epoch(
        options["input_root"],
        args.subject,
        args.finger,
        args.seed,
        options["epoch_reference_seeds"],
    )
    if args.epoch_override is not None:
        if args.epoch_override < 0:
            parser.error("--epoch-override must be nonnegative")
        selected_epoch = args.epoch_override
    subject_targets = target_map.get(args.subject, target_map.get(str(args.subject)))
    target_policy = str(subject_targets[args.finger])
    full_target_policy = target_policy.removesuffix("_split_safe")
    finger_index = list(FINGER_NAMES).index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    max_features = int(options.get("max_features", args.max_features))
    hidden_size = int(options.get("hidden_size", args.hidden_size))
    near_zero_std = float(options.get("near_zero_std", args.near_zero_std))
    candidate_scale = float(options.get("candidate_scale", args.candidate_scale))

    train_ecog_numpy = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    train_ecog = torch.from_numpy(np.array(train_ecog_numpy, copy=True)).to(device)
    train_windows = train_ecog.unfold(0, args.window_samples, args.stride_samples)
    row_count = int(train_windows.shape[0])
    transferred_warmup_epochs = int(options["warmup_epochs"])
    schedule_audit: dict[str, object] = {
        "rule": "verbatim fold-selected epoch count"
    }
    if args.schedule_transfer == "updates" and args.epoch_override is None:
        definition = json.loads(
            (
                args.fold_root
                / f"sub{args.subject}"
                / args.finger
                / "folds.json"
            ).read_text()
        )
        selected_epoch, transferred_warmup_epochs, schedule_audit = (
            update_equivalent_schedule(
                epochs_by_seed=outer_epochs,
                seed=args.seed,
                definition=definition,
                sequence_steps=int(options["sequence_steps"]),
                sequence_stride=int(options["sequence_stride"]),
                batch_size=int(options["batch_size"]),
                unfrozen_batch_size=int(options["unfrozen_batch_size"]),
                warmup_epochs=int(options["warmup_epochs"]),
                full_rows=row_count,
            )
        )
        epoch_rule = str(schedule_audit["rule"])
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + row_count, finger_index]
    target_train = np.load(prepared / f"train_glove_{full_target_policy}.npy")[24 : 24 + row_count, finger_index]
    target_tensor = torch.as_tensor(target_train, dtype=torch.float32, device=device)
    configured_spatial_cache_root = options.get("spatial_cache_root")
    spatial_cache_base = (
        args.spatial_cache_root
        if args.spatial_cache_root is not None
        else (
            Path(str(configured_spatial_cache_root))
            if configured_spatial_cache_root is not None
            else None
        )
    )
    spatial_cache_root = (
        resolve_perfinger_cache_root(
            spatial_cache_base, args.subject, args.finger
        )
        if spatial_cache_base is not None
        else None
    )
    if spatial_cache_root is None:
        fixed_all = np.load(
            args.feature_root
            / f"sub{args.subject}"
            / "train_initialized_window_features.npy",
            mmap_mode="r",
        )[:row_count]
        ica = np.load(
            args.ica_root / f"sub{args.subject}" / "fastica_unmixing.npy"
        )
    else:
        full_spatial_cache = spatial_cache_root / "full"
        fixed_all = load_perfinger_cached_features(
            full_spatial_cache,
            args.feature_root
            / f"sub{args.subject}"
            / "train_initialized_window_features.npy",
            rows=row_count,
            combined_name="train_features.npy",
            csp_name="train_csp_features.npy",
        )
        ica = np.load(full_spatial_cache / "spatial_weights.npy")
    selection = fit_or_load_inner_lars(
        features_all=fixed_all,
        target_all=target_train,
        training_intervals=lars_chunks(row_count),
        cache=args.selection_cache_root / full_target_policy / f"sub{args.subject}" / args.finger / "full.npz",
        max_features=max_features,
    )
    selected = np.asarray(selection["selected_source"], dtype=np.int64)
    cached_train = torch.as_tensor(
        np.asarray(fixed_all[:, selected], dtype=np.float32), device=device
    )
    model_ica, model_selected, used_spatial_rows = compact_spatial_dictionary(
        ica, selected, fixed_all.shape[1]
    )
    model = build_model(
        input_channels=train_ecog.shape[1],
        ica=model_ica,
        selected=model_selected,
        mean=np.asarray(selection["feature_mean"]),
        scale=np.asarray(selection["feature_scale"]),
        coefficients=np.asarray(selection["coefficients"]),
        intercept=float(selection["intercept"]),
        hidden_size=hidden_size,
        near_zero_std=near_zero_std,
        candidate_scale=candidate_scale,
        output_activation=str(options["output_activation"]),
        device=device,
        movement_fraction=float(np.mean(target_train >= 0.08)),
    )
    audit_indices = torch.linspace(0, row_count - 1, steps=64, device=device).round().long()
    with torch.inference_mode():
        extracted = model.extract(train_windows[audit_indices][None])[0]
    difference = extracted - cached_train[audit_indices]
    feature_audit = {
        "rmse": float(difference.square().mean().sqrt().cpu()),
        "max_abs_error": float(difference.abs().max().cpu()),
        "reference_rms": float(cached_train[audit_indices].square().mean().sqrt().cpu()),
    }
    _, initialized_train = predict_intervals(
        model, cached_train, train_windows, [[0, row_count]], False, args.prediction_chunk_steps
    )
    train_args = SimpleNamespace(
        learning_rate=float(options["learning_rate"]),
        gate_learning_rate=1.0e-3,
        spatial_learning_rate=float(options["spatial_learning_rate"]),
        wavelet_learning_rate=float(options["wavelet_learning_rate"]),
        weight_decay=float(options["weight_decay"]),
        compile=args.compile,
        warmup_epochs=transferred_warmup_epochs,
        sequence_steps=int(options["sequence_steps"]),
        sequence_stride=int(options["sequence_stride"]),
        batch_size=int(options["batch_size"]),
        unfrozen_batch_size=int(options["unfrozen_batch_size"]),
        movement_threshold=0.08,
        movement_weight=4.0,
        velocity_weight=0.2,
        correlation_weight=0.1,
        loss=str(options["loss"]),
        cached_only=False,
    )
    losses = train_fixed_epochs(
        model=model,
        cached_features=cached_train,
        raw_windows=train_windows,
        target=target_tensor,
        training_intervals=[[0, row_count]],
        epochs=selected_epoch,
        args=train_args,
        seed=args.seed,
    )
    _, fitted_train = predict_intervals(
        model, cached_train, train_windows, [[0, row_count]], True, args.prediction_chunk_steps
    )
    trained_seconds = time.perf_counter() - started
    output = args.output_root / f"sub{args.subject}" / args.finger / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_indices": selected,
            "source_feature_indices": selected,
            "model_feature_indices": model_selected,
            "source_spatial_rows": used_spatial_rows,
            "selected_epoch": selected_epoch,
            "outer_fold_selected_epochs_by_reference_seed": outer_epochs,
            "epoch_rule": epoch_rule,
        },
        output / "model.pt",
    )
    training_report = {
        "protocol": "frozen OOF-selected LARS-initialized LSTM refit on all development rows",
        "subject": args.subject,
        "finger": args.finger,
        "seed": args.seed,
        "official_final_validation_incorporated_into_training": True,
        "released_test_touched_during_model_selection": False,
        "source_cv_root": str(options["input_root"]),
        "outer_fold_selected_epochs_by_reference_seed": outer_epochs,
        "epoch_reference_seeds": list(options["epoch_reference_seeds"]),
        "selected_epoch": selected_epoch,
        "epoch_rule": (
            "explicit diagnostic override"
            if args.epoch_override is not None
            else epoch_rule
        ),
        "schedule_transfer": args.schedule_transfer,
        "schedule_transfer_audit": schedule_audit,
        "spatial_cache_root": (
            str(spatial_cache_root) if spatial_cache_root is not None else None
        ),
        "target_policy_selected_in_oof": target_policy,
        "full_refit_target_policy": full_target_policy,
        "feature_count": int(selected.size),
        "source_spatial_row_count": int(ica.shape[0]),
        "active_spatial_row_count": int(model_ica.shape[0]),
        "source_spatial_rows": used_spatial_rows.tolist(),
        "lars_alpha": float(selection["alpha"]),
        "linear_initializer": str(selection["selection_method"]),
        "lars_candidate_scale": candidate_scale,
        "cached_vs_raw_initial_feature_audit": feature_audit,
        "initialized_full_train_raw_pcc": pearson(initialized_train, raw_train),
        "fitted_full_train_raw_pcc": pearson(fitted_train, raw_train),
        "training_losses": losses,
        "training_runtime_seconds": trained_seconds,
        "configuration": {
            **{
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in options.items()
                if key not in {"seeds", "epoch_reference_seeds"}
            },
            "effective_warmup_epochs": transferred_warmup_epochs,
            "effective_total_epochs": selected_epoch,
        },
    }
    (output / "training_summary.json").write_text(json.dumps(training_report, indent=2) + "\n")

    # The model and its complete training audit exist before any released-test
    # target is loaded. Test labels are used only for this terminal evaluation.
    test_ecog_numpy = np.load(prepared / "test_ecog.npy", mmap_mode="r")
    test_ecog = torch.from_numpy(np.array(test_ecog_numpy, copy=True)).to(device)
    test_windows = test_ecog.unfold(0, args.window_samples, args.stride_samples)
    test_rows = int(test_windows.shape[0])
    if spatial_cache_root is None:
        fixed_test = np.load(
            args.feature_root
            / f"sub{args.subject}"
            / "test_initialized_window_features.npy",
            mmap_mode="r",
        )[:test_rows]
    else:
        fixed_test = load_perfinger_cached_features(
            spatial_cache_root / "full",
            args.feature_root
            / f"sub{args.subject}"
            / "test_initialized_window_features.npy",
            rows=test_rows,
            combined_name="test_features.npy",
            csp_name="test_csp_features.npy",
        )
    cached_test = torch.as_tensor(
        np.asarray(fixed_test[:, selected], dtype=np.float32), device=device
    )
    _, prediction = predict_intervals(
        model, cached_test, test_windows, [[0, test_rows]], True, args.prediction_chunk_steps
    )
    np.save(output / "released_test_prediction.npy", prediction)

    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24 : 24 + test_rows, finger_index]
    cleaned_test = np.load(prepared / f"test_glove_{full_target_policy}.npy")[24 : 24 + test_rows, finger_index]
    groups = movement_groups(cleaned_test, 0.08)
    metrics = morphology_metrics(prediction, cleaned_test, groups)
    metrics["raw_pcc"] = pearson(prediction, raw_test)
    plot_full_trajectory(
        output / "released_test_full_trajectory.png",
        raw_test,
        cleaned_test,
        prediction,
        f"S{args.subject} {args.finger} seed {args.seed}: full-data refit",
    )
    plot_events(
        output / "released_test_events.png",
        raw_test,
        cleaned_test,
        prediction,
        f"S{args.subject} {args.finger} seed {args.seed}: strongest test events",
    )
    report = {
        **training_report,
        "released_test_touched": True,
        "released_test_role": "retrospective paper-comparison evaluation",
        "released_test_used_for_current_configuration_selection": False,
        "released_test_previously_inspected_during_reconstruction": True,
        "released_test_metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "subject": args.subject,
        "finger": args.finger,
        "seed": args.seed,
        "selected_epoch": selected_epoch,
        "raw_test_pcc": metrics["raw_pcc"],
        "cleaned_test_pcc": metrics["cleaned_pcc"],
        "runtime_seconds": report["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
