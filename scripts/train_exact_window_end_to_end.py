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
from torch.nn import functional as F

try:
    from scripts.benchmark_ridge_target_variants import lagged, ridge_fit
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from benchmark_ridge_target_variants import lagged, ridge_fit
from ecog_decoding.models import (
    AsymmetricWaveletPacketEnergy,
    CSPBandCorrectionEnergy,
    CSPSpatialProjection,
    WaveletPacketEnergy,
)
from ecog_decoding.training import FINGER_NAMES


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def parse_warm_start(value: str) -> tuple[str, Path]:
    finger, separator, directory = value.partition("=")
    if not separator or finger not in FINGER_NAMES or not directory:
        raise argparse.ArgumentTypeError(
            "warm start must be FINGER=SUBJECT_OUTPUT_DIRECTORY"
        )
    return finger, Path(directory)


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
        head_initialization: str = "residual_ridge",
        output_activation: str = "relu",
        softplus_beta: float = 10.0,
        csp_weights: np.ndarray | None = None,
        csp_correction_kernel_size: int = 33,
    ) -> None:
        super().__init__()
        self.frontend = frontend
        if frontend == "csp_band":
            if csp_weights is None:
                raise ValueError("csp_band frontend requires initialized CSP weights")
            self.spatial = CSPSpatialProjection(csp_weights)
            self.wavelet = CSPBandCorrectionEnergy(
                band_count=csp_weights.shape[0],
                kernel_size=csp_correction_kernel_size,
                energy_window_samples=40,
                energy_stride_samples=40,
                padding_mode="constant",
            )
        elif frontend == "asymmetric":
            self.spatial = nn.Conv1d(input_channels, component_count, 1, bias=False)
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
            self.spatial = nn.Conv1d(input_channels, component_count, 1, bias=False)
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
        self.head_initialization = head_initialization
        if head_initialization not in ("lars_linear_regime", "residual_ridge"):
            raise ValueError(f"unsupported head initialization {head_initialization!r}")
        if output_activation not in ("linear", "relu", "softplus"):
            raise ValueError(f"unsupported output activation {output_activation!r}")
        if softplus_beta <= 0:
            raise ValueError("softplus beta must be positive")
        self.output_activation = output_activation
        self.softplus_beta = softplus_beta
        self.lstm = nn.LSTM(selected_indices.size, hidden_size, batch_first=True)
        self.temporal = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.temporal.weight)
        nn.init.zeros_(self.temporal.bias)

    @torch.no_grad()
    def initialize_lars_linear_regime(
        self,
        coefficients: np.ndarray,
        intercept: float,
        unit: int = 0,
        candidate_scale: float = 1.0,
        near_zero_std: float = 1.0e-3,
        open_gate_bias: float = 3.0,
        forget_gate_bias: float = -3.0,
    ) -> None:
        """Seed one isolated nonlinear LSTM unit in its linear regime."""
        if not 0 <= unit < self.lstm.hidden_size:
            raise ValueError("unit is outside the hidden state")
        if candidate_scale <= 0:
            raise ValueError("candidate_scale must be positive")
        if near_zero_std < 0:
            raise ValueError("near_zero_std must be nonnegative")
        coefficients_t = torch.as_tensor(
            coefficients,
            dtype=self.lstm.weight_ih_l0.dtype,
            device=self.lstm.weight_ih_l0.device,
        )
        if coefficients_t.shape != (self.lstm.input_size,):
            raise ValueError("LARS coefficient count must equal the LSTM input size")
        self.direct.weight.zero_()
        self.direct.bias.zero_()
        self.direct.requires_grad_(False)
        hidden = self.lstm.hidden_size
        rows = [unit + gate * hidden for gate in range(4)]
        for row in rows:
            self.lstm.weight_ih_l0[row].normal_(0.0, near_zero_std)
            self.lstm.weight_hh_l0[row].normal_(0.0, near_zero_std)
            self.lstm.bias_ih_l0[row].zero_()
            self.lstm.bias_hh_l0[row].zero_()
        # Connections that would be exactly zero in the ideal isolated path
        # receive tiny random weights, matching the historical implementation.
        self.lstm.weight_hh_l0[:, unit].normal_(0.0, near_zero_std)
        input_row, forget_row, candidate_row, output_row = rows
        self.lstm.bias_ih_l0[input_row] = open_gate_bias
        self.lstm.bias_ih_l0[forget_row] = forget_gate_bias
        self.lstm.bias_ih_l0[output_row] = open_gate_bias
        self.lstm.bias_ih_l0[candidate_row] = candidate_scale * float(intercept)
        self.lstm.weight_ih_l0[candidate_row].copy_(
            candidate_scale * coefficients_t
        )
        open_gate = torch.sigmoid(
            torch.as_tensor(
                open_gate_bias,
                dtype=self.temporal.weight.dtype,
                device=self.temporal.weight.device,
            )
        )
        self.temporal.weight.zero_()
        if near_zero_std:
            self.temporal.weight.normal_(0.0, near_zero_std)
        self.temporal.weight[0, unit] = 1.0 / (
            candidate_scale * open_gate.square()
        )
        self.temporal.bias.zero_()

    def extract(self, windows: torch.Tensor) -> torch.Tensor:
        """Return selected features from exact one-second windows."""
        if self.frontend == "csp_band":
            batch, steps, bands, channels, samples = windows.shape
            flattened = windows.reshape(batch * steps, bands, channels, samples)
            spatial = self.spatial(flattened)
            # Fixed CSP/LARS features are flattened time-major, then
            # band/component-major within each 40 ms bin.
            energy = self.wavelet(spatial).permute(0, 3, 1, 2).flatten(1)
        else:
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
        pre_activation = direct + self.temporal(recurrent)
        if self.output_activation == "relu":
            pre_activation = torch.relu(pre_activation)
        elif self.output_activation == "softplus":
            pre_activation = F.softplus(pre_activation, beta=self.softplus_beta)
        return pre_activation.squeeze(-1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.decode(self.extract(windows))


def make_windows(ecog: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(
        ecog, window_shape=window_samples, axis=0
    )[::stride_samples]


@torch.inference_mode()
def predict(
    model: ExactWindowFingerDecoder,
    windows: np.ndarray | torch.Tensor,
    device: torch.device,
    chunk_steps: int,
) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, windows.shape[0], chunk_steps):
        values = windows[start : start + chunk_steps]
        batch = (
            values.to(device)
            if isinstance(values, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(values)).to(device)
        )
        features = model.extract(batch[None]).squeeze(0)
        chunks.append(features.cpu())
    features = torch.cat(chunks, dim=0)[None].to(device)
    return model.decode(features).squeeze(0).float().cpu().numpy()


@torch.inference_mode()
def cache_features_on_device(
    model: ExactWindowFingerDecoder,
    windows: np.ndarray | torch.Tensor,
    device: torch.device,
    chunk_steps: int,
) -> torch.Tensor:
    """Evaluate a frozen stem once and retain its compact output on the GPU."""
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, windows.shape[0], chunk_steps):
        values = windows[start : start + chunk_steps]
        batch = (
            values.to(device)
            if isinstance(values, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(values)).to(device)
        )
        chunks.append(model.extract(batch[None]).squeeze(0))
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_sklearn_v1"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/ica_sklearn_constant_v1"))
    parser.add_argument("--csp-root", type=Path, default=Path("outputs/csp_ridge_v2"))
    parser.add_argument(
        "--band-cache-root",
        type=Path,
        default=Path("/dev/shm/ecog_csp_band_cache"),
    )
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/exact_window_end_to_end_v1"))
    parser.add_argument("--target", default="local_w1_q10")
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=["index", "middle"])
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument(
        "--fit-end-index",
        type=int,
        default=None,
        help="optional exclusive row in the lagged training sequence for an inner fold",
    )
    parser.add_argument(
        "--validation-end-index",
        type=int,
        default=None,
        help="optional exclusive end of the contiguous inner-validation block",
    )
    parser.add_argument(
        "--no-validation-selection",
        action="store_true",
        help=(
            "train exactly --epochs epochs and evaluate the held-out segment only "
            "once afterward; use this for the final refit"
        ),
    )
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="do not read released test labels or emit test predictions during inner CV",
    )
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--wavelet-levels", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--frontend",
        choices=("wavelet", "asymmetric", "csp_band"),
        default="wavelet",
    )
    parser.add_argument("--csp-correction-kernel-size", type=int, default=33)
    parser.add_argument(
        "--head-initialization",
        choices=("lars_linear_regime", "residual_ridge"),
        default="residual_ridge",
        help="lars_linear_regime seeds one standard nonlinear LSTM unit near zero; residual_ridge preserves the earlier direct-skip experiment",
    )
    parser.add_argument(
        "--output-activation",
        choices=("linear", "relu", "softplus"),
        default="relu",
        help=(
            "linear avoids a dead-gradient null predictor during regression; "
            "relu preserves the original reconstruction experiment"
        ),
    )
    parser.add_argument(
        "--softplus-beta",
        type=float,
        default=10.0,
        help="steepness of the smooth nonnegative output when using softplus",
    )
    parser.add_argument(
        "--warm-start",
        action="append",
        type=parse_warm_start,
        default=[],
        help=(
            "repeatable FINGER=SUBJECT_OUTPUT_DIRECTORY mapping; the saved "
            "checkpoint must match the requested frontend, selection, and width"
        ),
    )
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--lars-near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--lars-open-gate-bias", type=float, default=3.0)
    parser.add_argument("--lars-forget-gate-bias", type=float, default=-3.0)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--unfrozen-batch-size",
        type=int,
        default=None,
        help=(
            "optional smaller batch size after the raw-window stem is unfrozen; "
            "defaults to --batch-size"
        ),
    )
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
    parser.add_argument(
        "--cache-raw-on-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep the raw recording on the accelerator and form zero-copy unfold "
            "window views instead of rebuilding NumPy minibatches"
        ),
    )
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
    warm_starts = dict(args.warm_start)
    if len(warm_starts) != len(args.warm_start):
        parser.error("each --warm-start finger may be specified only once")
    unused_warm_starts = set(warm_starts).difference(args.fingers)
    if unused_warm_starts:
        parser.error(
            "warm-start mappings supplied for unrequested fingers: "
            + ", ".join(sorted(unused_warm_starts))
        )
    if args.frontend_warmup_epochs < 0:
        parser.error("--frontend-warmup-epochs must be nonnegative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.unfrozen_batch_size is not None and args.unfrozen_batch_size < 1:
        parser.error("--unfrozen-batch-size must be positive")
    if args.learning_rate_decay_per_epoch < 0:
        parser.error("--learning-rate-decay-per-epoch must be nonnegative")
    if args.softplus_beta <= 0:
        parser.error("--softplus-beta must be positive")
    if args.csp_correction_kernel_size < 1 or args.csp_correction_kernel_size % 2 == 0:
        parser.error("--csp-correction-kernel-size must be a positive odd integer")
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
    unfrozen_batch_size = (
        args.batch_size
        if args.unfrozen_batch_size is None
        else args.unfrozen_batch_size
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
    csp_dir = args.csp_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    official_train_count = split - offset
    train_ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    test_ecog = np.load(prepared / "test_ecog.npy", mmap_mode="r")
    if args.frontend == "csp_band":
        band_cache = args.band_cache_root / f"sub{args.subject}"
        train_source = np.load(band_cache / "train_filtered_bands.npy", mmap_mode="r")
        test_source = np.load(band_cache / "test_filtered_bands.npy", mmap_mode="r")
    else:
        train_source = train_ecog
        test_source = test_ecog
    if args.cache_raw_on_device:
        train_recording = torch.from_numpy(np.asarray(train_source)).to(device)
        test_recording = torch.from_numpy(np.asarray(test_source)).to(device)
        if args.frontend == "csp_band":
            train_windows_all = train_recording.unfold(
                1, args.window_samples, args.stride_samples
            ).permute(1, 0, 2, 3)
            test_windows = test_recording.unfold(
                1, args.window_samples, args.stride_samples
            ).permute(1, 0, 2, 3)
        else:
            train_windows_all = train_recording.unfold(
                0, args.window_samples, args.stride_samples
            )
            test_windows = test_recording.unfold(
                0, args.window_samples, args.stride_samples
            )
        print(
            f"cached_raw_recordings_on_device train={tuple(train_recording.shape)} "
            f"test={tuple(test_recording.shape)}",
            flush=True,
        )
    else:
        if args.frontend == "csp_band":
            train_windows_all = np.lib.stride_tricks.sliding_window_view(
                train_source, window_shape=args.window_samples, axis=1
            )[:, :: args.stride_samples].transpose(1, 0, 2, 3)
            test_windows = np.lib.stride_tricks.sliding_window_view(
                test_source, window_shape=args.window_samples, axis=1
            )[:, :: args.stride_samples].transpose(1, 0, 2, 3)
        else:
            train_windows_all = make_windows(
                train_source, args.window_samples, args.stride_samples
            )
            test_windows = make_windows(
                test_source, args.window_samples, args.stride_samples
            )
    # Row zero of the exact feature cache predicts glove bin history-1.
    target = np.load(prepared / f"train_glove_{args.target}.npy")[offset:]
    raw_train = np.load(prepared / "train_glove_25hz_raw.npy")[offset:]
    raw_test = (
        None
        if args.skip_test_evaluation
        else np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    )
    train_count = (
        official_train_count if args.fit_end_index is None else args.fit_end_index
    )
    validation_stop = (
        raw_train.shape[0]
        if args.validation_end_index is None
        else args.validation_end_index
    )
    if not 1 <= train_count < validation_stop <= raw_train.shape[0]:
        parser.error(
            "require 1 <= fit-end-index < validation-end-index <= available rows"
        )
    if args.no_validation_selection and train_count != official_train_count:
        parser.error("final refit must use the complete official training partition")
    if args.frontend == "csp_band":
        fixed_train = lagged(
            np.load(feature_dir / "train_csp_energy.npy", mmap_mode="r"),
            args.history,
        )
        csp_summary = json.loads((csp_dir / "summary.json").read_text())
        csp_weights = np.stack(
            [
                np.load(csp_dir / f"csp_{float(low):g}_{float(high):g}hz.npy")
                for low, high in csp_summary["bands_hz"]
            ],
            axis=0,
        )
        component_count = int(csp_weights.shape[1])
        ica = None
    else:
        fixed_train = np.load(
            feature_dir / "train_initialized_window_features.npy", mmap_mode="r"
        )
        ica = np.load(ica_dir / "fastica_unmixing.npy")
        csp_weights = None
        component_count = int(ica.shape[0])
    selection = json.loads((selection_dir / "summary.json").read_text())
    validation_prediction = np.full_like(
        raw_train[train_count:validation_stop], np.nan, dtype=np.float32
    )
    test_prediction = np.full(
        (test_windows.shape[0], len(FINGER_NAMES)), np.nan, dtype=np.float32
    )
    report: dict[str, object] = {}

    for finger_name in args.fingers:
        finger = list(FINGER_NAMES).index(finger_name)
        indices = np.asarray(
            selection["per_finger"][finger_name]["selected_source_indices"], dtype=np.int64
        )
        selected_fit = selection["per_finger"][finger_name]
        initial = np.asarray(fixed_train[:train_count, indices], dtype=np.float32)
        if args.head_initialization == "lars_linear_regime":
            required = (
                "selected_standardized_coefficients",
                "selected_feature_mean",
                "selected_feature_scale",
                "intercept",
            )
            missing = [name for name in required if name not in selected_fit]
            if missing:
                raise ValueError(
                    "selection summary predates paper LARS initialization fields: "
                    + ", ".join(missing)
                )
            mean = np.asarray(selected_fit["selected_feature_mean"], dtype=np.float32)
            scale = np.asarray(selected_fit["selected_feature_scale"], dtype=np.float32)
            linear_weight = np.asarray(
                selected_fit["selected_standardized_coefficients"], dtype=np.float32
            )
            linear_intercept = float(selected_fit["intercept"])
        else:
            mean = initial.mean(axis=0, dtype=np.float64).astype(np.float32)
            scale = initial.std(axis=0, dtype=np.float64).astype(np.float32)
            y_train = np.asarray(target[:train_count, finger], dtype=np.float32)
            _, _, linear_intercept, linear_weight = ridge_fit(
                initial, y_train, 1.0e-3, device
            )
        scale[scale < 1.0e-6] = 1.0
        y_train = np.asarray(target[:train_count, finger], dtype=np.float32)
        device_train_target = (
            torch.from_numpy(y_train).to(device)
            if args.cache_raw_on_device or args.frontend_warmup_epochs > 0
            else None
        )
        initialization_sample = initial[: min(initial.shape[0], 4096)].copy()
        expected_initial = (
            (initialization_sample - mean) / scale @ linear_weight + linear_intercept
        )
        if args.output_activation == "relu":
            expected_initial = np.maximum(expected_initial, 0.0)
        elif args.output_activation == "softplus":
            expected_initial = np.logaddexp(
                0.0, args.softplus_beta * expected_initial
            ) / args.softplus_beta
        del initial

        model = ExactWindowFingerDecoder(
            train_ecog.shape[1],
            component_count,
            indices,
            mean,
            scale,
            args.hidden_size,
            wavelet_levels=args.wavelet_levels,
            frontend=args.frontend,
            head_initialization=args.head_initialization,
            output_activation=args.output_activation,
            softplus_beta=args.softplus_beta,
            csp_weights=csp_weights,
            csp_correction_kernel_size=args.csp_correction_kernel_size,
        ).to(device)
        with torch.no_grad():
            if args.frontend != "csp_band":
                assert ica is not None
                model.spatial.weight[:, :, 0].copy_(torch.from_numpy(ica).to(device))
            if args.head_initialization == "lars_linear_regime":
                model.initialize_lars_linear_regime(
                    linear_weight,
                    linear_intercept,
                    candidate_scale=args.lars_candidate_scale,
                    near_zero_std=args.lars_near_zero_std,
                    open_gate_bias=args.lars_open_gate_bias,
                    forget_gate_bias=args.lars_forget_gate_bias,
                )
            else:
                model.direct.weight.copy_(
                    torch.from_numpy(linear_weight[None]).to(device)
                )
                model.direct.bias.fill_(linear_intercept)
        with torch.inference_mode():
            observed_initial = model.decode(
                torch.from_numpy(initialization_sample[None]).to(device)
            ).squeeze(0).cpu().numpy()
        initialization_max_abs_error = float(
            np.max(np.abs(observed_initial - expected_initial))
        )
        initialization_rmse = float(
            np.sqrt(np.mean(np.square(observed_initial - expected_initial)))
        )
        initialization_pcc = pearson(observed_initial, expected_initial)
        if not np.isfinite(initialization_max_abs_error):
            raise RuntimeError("network linear initialization produced non-finite values")
        if args.head_initialization == "residual_ridge" and initialization_max_abs_error > 1.0e-5:
            raise RuntimeError(
                "network does not reproduce its linear initializer: "
                f"max_abs_error={initialization_max_abs_error:.3g}"
            )
        if args.head_initialization == "lars_linear_regime" and initialization_pcc < 0.98:
            raise RuntimeError(
                "LSTM linear-regime initialization is not sufficiently faithful: "
                f"pcc={initialization_pcc:.6f}"
            )
        warm_start_directory = warm_starts.get(finger_name)
        if warm_start_directory is not None:
            # These are project-local checkpoints produced by this repository.
            checkpoint = torch.load(
                warm_start_directory / f"{finger_name}.pt",
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_indices = np.asarray(
                checkpoint["feature_indices"], dtype=np.int64
            )
            if not np.array_equal(checkpoint_indices, indices):
                raise ValueError(
                    f"warm-start feature indices do not match for {finger_name}"
                )
            checkpoint_state = checkpoint["model_state_dict"]
            current_state = model.state_dict()
            common = set(current_state).intersection(checkpoint_state)
            mismatched = {
                name: (tuple(current_state[name].shape), tuple(checkpoint_state[name].shape))
                for name in common
                if current_state[name].shape != checkpoint_state[name].shape
            }
            missing = sorted(set(current_state).difference(checkpoint_state))
            unexpected = sorted(set(checkpoint_state).difference(current_state))
            if mismatched or missing or unexpected:
                raise ValueError(
                    f"warm-start architecture mismatch for {finger_name}: "
                    f"shape_mismatches={mismatched}, missing={missing}, "
                    f"unexpected={unexpected}"
                )
            model.load_state_dict(checkpoint_state)
            print(
                f"finger={finger_name} warm_start={warm_start_directory}",
                flush=True,
            )
        train_model: nn.Module = model
        train_decode = model.decode
        if args.compile:
            train_model = torch.compile(model, mode="reduce-overhead")
            train_decode = torch.compile(model.decode, mode="reduce-overhead")
        cached_train_features: torch.Tensor | None = None
        cached_validation_features: torch.Tensor | None = None
        if args.frontend_warmup_epochs > 0:
            cached_train_features = cache_features_on_device(
                model,
                train_windows_all[:train_count],
                device,
                args.prediction_chunk_steps,
            )
            cached_validation_features = cache_features_on_device(
                model,
                train_windows_all[train_count:validation_stop],
                device,
                args.prediction_chunk_steps,
            )
            print(
                f"finger={finger_name} cached_frozen_stem_features "
                f"train={tuple(cached_train_features.shape)} "
                f"validation={tuple(cached_validation_features.shape)}",
                flush=True,
            )
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
        if args.no_validation_selection:
            initial_validation_estimate = None
            best_score = float("nan")
        elif cached_validation_features is None:
            initial_validation_estimate = predict(
                model,
                train_windows_all[train_count:validation_stop],
                device,
                args.prediction_chunk_steps,
            )
            best_score = pearson(
                initial_validation_estimate,
                raw_train[train_count:validation_stop, finger],
            )
        else:
            with torch.inference_mode():
                initial_validation_estimate = (
                    model.decode(cached_validation_features[None])
                    .squeeze(0)
                    .float()
                    .cpu()
                    .numpy()
                )
            best_score = pearson(
                initial_validation_estimate,
                raw_train[train_count:validation_stop, finger],
            )
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 0
        stale = 0
        history: list[dict[str, float]] = []
        if not args.no_validation_selection:
            history.append({"epoch": 0, "validation_raw_r": best_score})
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
            active_batch_size = (
                args.batch_size
                if epoch <= args.frontend_warmup_epochs
                else unfrozen_batch_size
            )
            for begin in range(0, starts.size, active_batch_size):
                selected_starts = starts[begin : begin + active_batch_size]
                if epoch <= args.frontend_warmup_epochs:
                    if cached_train_features is None or device_train_target is None:
                        raise RuntimeError("frozen-stem cache was not initialized")
                    indices_t = torch.as_tensor(
                        selected_starts[:, None]
                        + np.arange(args.sequence_steps, dtype=np.int64)[None, :],
                        dtype=torch.long,
                        device=device,
                    )
                    prediction = train_decode(cached_train_features[indices_t])
                    observed = device_train_target[indices_t]
                elif isinstance(train_windows_all, torch.Tensor):
                    indices_t = torch.as_tensor(
                        selected_starts[:, None]
                        + np.arange(args.sequence_steps, dtype=np.int64)[None, :],
                        dtype=torch.long,
                        device=device,
                    )
                    prediction = train_model(train_windows_all[indices_t])
                    if device_train_target is None:
                        raise RuntimeError("device target cache was not initialized")
                    observed = device_train_target[indices_t]
                else:
                    xb = np.stack(
                        [
                            np.ascontiguousarray(
                                train_windows_all[s : s + args.sequence_steps]
                            )
                            for s in selected_starts
                        ]
                    )
                    yb = np.stack(
                        [y_train[s : s + args.sequence_steps] for s in selected_starts]
                    )
                    prediction = train_model(torch.from_numpy(xb).to(device))
                    observed = torch.from_numpy(yb).to(device)
                loss = torch.mean((prediction - observed).square())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            if (
                not args.no_validation_selection
                and (epoch == 1 or epoch % args.validation_interval == 0)
            ):
                if epoch <= args.frontend_warmup_epochs:
                    if cached_validation_features is None:
                        raise RuntimeError("frozen-stem cache was not initialized")
                    with torch.inference_mode():
                        estimate = (
                            model.decode(cached_validation_features[None])
                            .squeeze(0)
                            .float()
                            .cpu()
                            .numpy()
                        )
                else:
                    estimate = predict(
                        model,
                        train_windows_all[train_count:validation_stop],
                        device,
                        args.prediction_chunk_steps,
                    )
                score = pearson(
                    estimate, raw_train[train_count:validation_stop, finger]
                )
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
            elif args.no_validation_selection and (
                epoch == 1
                or epoch % args.validation_interval == 0
                or epoch == args.epochs
            ):
                history.append({"epoch": epoch, "loss": float(np.mean(losses))})
                print(
                    f"finger={finger_name} epoch={epoch} "
                    f"loss={np.mean(losses):.6f} selection_holdout_untouched",
                    flush=True,
                )
        if args.no_validation_selection:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = args.epochs
        model.load_state_dict(best_state)
        validation_prediction[:, finger] = predict(
            model,
            train_windows_all[train_count:validation_stop],
            device,
            args.prediction_chunk_steps,
        )
        if not args.skip_test_evaluation:
            test_prediction[:, finger] = predict(
                model, test_windows, device, args.prediction_chunk_steps
            )
        best_score = pearson(
            validation_prediction[:, finger],
            raw_train[train_count:validation_stop, finger],
        )
        torch.save(
            {"model_state_dict": best_state, "feature_indices": indices},
            output / f"{finger_name}.pt",
        )
        report[finger_name] = {
            "feature_count": int(indices.size),
            "head_initialization": args.head_initialization,
            "output_activation": args.output_activation,
            "softplus_beta": args.softplus_beta,
            "warm_start_directory": (
                str(warm_start_directory) if warm_start_directory is not None else None
            ),
            "lars_initialization": (
                {
                    "candidate_scale": args.lars_candidate_scale,
                    "near_zero_std": args.lars_near_zero_std,
                    "open_gate_bias": args.lars_open_gate_bias,
                    "forget_gate_bias": args.lars_forget_gate_bias,
                }
                if args.head_initialization == "lars_linear_regime"
                else None
            ),
            "initialization_max_abs_error": initialization_max_abs_error,
            "initialization_rmse": initialization_rmse,
            "initialization_pcc": initialization_pcc,
            "initialized_validation_raw_r": (
                history[0]["validation_raw_r"]
                if history and "validation_raw_r" in history[0]
                else None
            ),
            "best_epoch": best_epoch,
            "validation_raw_r": best_score,
            "test_raw_r": (
                None
                if raw_test is None
                else pearson(test_prediction[:, finger], raw_test[:, finger])
            ),
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
        "head_initialization": args.head_initialization,
        "output_activation": args.output_activation,
        "softplus_beta": args.softplus_beta,
        "csp_correction_kernel_size": (
            args.csp_correction_kernel_size if args.frontend == "csp_band" else None
        ),
        "selection_root": str(args.selection_root),
        "data_partition": {
            "official_training_rows": official_train_count,
            "fit_end_index": train_count,
            "validation_end_index": validation_stop,
            "validation_selection_enabled": not args.no_validation_selection,
            "released_test_evaluated": not args.skip_test_evaluation,
        },
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
            "unfrozen_batch_size": unfrozen_batch_size,
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
