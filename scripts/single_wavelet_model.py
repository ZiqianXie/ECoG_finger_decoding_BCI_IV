#!/usr/bin/env python3
"""Trainable components for the released single-wavelet decoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ecog_decoding.models import WaveletPacketEnergy
from single_wavelet_support import (
    HISTORY,
    SAMPLES_PER_BIN,
    frontend_frequency_order,
)


def leaky_integrate(innovation: torch.Tensor, decay: float) -> torch.Tensor:
    """Apply a causal leaky integrator along the penultimate (time) axis.

    This convolution form performs the full recurrence on the accelerator:
    ``state[t] = decay * state[t - 1] + innovation[t]``.  It avoids a Python
    time loop and remains differentiable with respect to every innovation.
    """
    if innovation.ndim < 2:
        raise ValueError("innovation must have time and feature dimensions")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be between zero and one")
    time_steps = innovation.shape[-2]
    channels = innovation.shape[-1]
    flat = innovation.reshape(-1, time_steps, channels).transpose(1, 2)
    powers = torch.arange(
        time_steps - 1,
        -1,
        -1,
        device=innovation.device,
        dtype=innovation.dtype,
    )
    kernel = torch.pow(innovation.new_tensor(decay), powers)
    weight = kernel.reshape(1, 1, time_steps).expand(channels, 1, time_steps)
    integrated = F.conv1d(
        flat, weight, padding=time_steps - 1, groups=channels
    )[..., :time_steps]
    return integrated.transpose(1, 2).reshape_as(innovation)


class OvercompleteWaveletPacketEnergy(WaveletPacketEnergy):
    """One packet tree retaining every complete level from depth 3 onward.

    Parent and child packet atoms overlap in frequency, so split-local sparse
    selection can choose the resolution supported by each finger. This is still
    one temporal tree: every deeper level is computed directly from the prior
    activations, with no parallel feature branch.
    """

    def __init__(self, levels: int = 4, **kwargs: object) -> None:
        if levels < 4:
            raise ValueError("overcomplete tree must extend beyond depth 3")
        super().__init__(levels=levels, **kwargs)
        self.retained_levels = tuple(range(3, levels + 1))
        self.output_band_level_counts = tuple(2**level for level in self.retained_levels)

    @property
    def output_band_count(self) -> int:
        return sum(self.output_band_level_counts)

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(
            f"d{level}_"
            + "".join(
                "L" if (band >> bit) & 1 == 0 else "H"
                for bit in reversed(range(level))
            )
            for level in self.retained_levels
            for band in range(2**level)
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        bands = x
        retained = []
        for level, layer in enumerate(self.layers):
            parent = bands
            bands = self._same_filter(parent, layer)
            bands = 1.7156 * torch.tanh((2.0 / 3.0) * bands)
            if self.interlevel_skip:
                bands = bands + self.skip_gates[level] * parent.repeat_interleave(
                    2, dim=1
                )
            if self.interlevel_normalization and (
                level < self.levels - 1 or self.normalize_final_level
            ):
                mean = bands.mean(dim=-1, keepdim=True)
                variance = bands.var(dim=-1, keepdim=True, unbiased=False)
                normalized = (bands - mean) * torch.rsqrt(
                    variance + self.normalization_epsilon
                )
                bands = bands + self.normalization_gates[level] * (
                    normalized - bands
                )
            if level >= 2:
                retained.append(bands)
        if not retained:
            raise RuntimeError("overcomplete tree did not produce depth-3 atoms")
        return torch.cat(retained, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, electrode, time); got {tuple(x.shape)}"
            )
        batch, electrodes, time = x.shape
        if time < self.energy_window_samples:
            raise ValueError("time dimension is shorter than the energy window")
        bands = self.transform(x.reshape(batch * electrodes, 1, time))
        energy = self._energy(bands)
        return energy.reshape(
            batch, electrodes, self.output_band_count, energy.shape[-1]
        )


class PaperEquationLSTM(nn.Module):
    """Single-layer gated recurrence using the equations printed in the paper.

    The input, forget, and output gates are sigmoid nonlinearities.  Unlike a
    standard PyTorch LSTM, the candidate state and exposed cell state do not
    pass through tanh: ``g_t`` is affine and ``h_t = o_t * c_t``.  The model is
    therefore still nonlinear and context dependent through its gates.
    """

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_first = bool(batch_first)
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        bound = self.hidden_size**-0.5
        for parameter in self.parameters():
            parameter.uniform_(-bound, bound)

    def forward(
        self,
        inputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if inputs.ndim != 3:
            raise ValueError("paper-equation LSTM input must be three-dimensional")
        sequence = inputs if self.batch_first else inputs.transpose(0, 1)
        batch = sequence.shape[0]
        if state is None:
            hidden = sequence.new_zeros(batch, self.hidden_size)
            cell = sequence.new_zeros(batch, self.hidden_size)
        else:
            hidden, cell = state
            hidden = hidden[0]
            cell = cell[0]

        outputs = []
        for step in range(sequence.shape[1]):
            gates = F.linear(
                sequence[:, step], self.weight_ih_l0, self.bias_ih_l0
            ) + F.linear(hidden, self.weight_hh_l0, self.bias_hh_l0)
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * candidate
            hidden = torch.sigmoid(output_gate) * cell
            outputs.append(hidden)

        output = torch.stack(outputs, dim=1)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, (hidden.unsqueeze(0), cell.unsqueeze(0))


class SingleWaveletDecoder(nn.Module):
    """One spatial convolution, one wavelet tree, and one nonlinear sequence head.

    The sparse LARS predictor is embedded in one LSTM unit so that the network
    starts near a useful linear solution. Alternatively, a zero-initialized
    LSTM or GRU residual can be added to the fixed LARS logit. Both variants
    retain the same single convolutional stem and Softplus output.
    """

    def __init__(
        self,
        spatial_weights: np.ndarray,
        initialization: dict[str, np.ndarray],
        hidden_size: int,
        lars_candidate_scale: float = 1.0,
        near_zero_std: float = 1.0e-3,
        open_gate_bias: float = 5.0,
        forget_gate_bias: float = -5.0,
        output_activation: str = "softplus",
        softplus_beta: float = 10.0,
        recurrent_cell: str = "standard",
        energy_window_samples: int = SAMPLES_PER_BIN,
        tap_resample_up: int = 5,
        tap_resample_down: int = 2,
        wavelet_frontend: str = "depth3",
        wavelet_interlevel_skip: bool = False,
        wavelet_interlevel_normalization: bool = False,
        wavelet_final_normalization: bool = True,
        residual_input: str = "selected",
        residual_history_bins: int = 1,
        residual_input_width: int | None = None,
        residual_include_direct: bool = False,
        movement_head: bool = False,
        velocity_head: bool = False,
        movement_modulation: bool = False,
        wavelet_signed_pooling: bool = False,
        residual_output_init_std: float = 0.0,
        residual_dynamics: str = "pointwise",
        residual_decay: float = 0.95,
    ) -> None:
        super().__init__()
        components, channels = spatial_weights.shape
        self.spatial = nn.Conv1d(channels, components, 1, bias=False)
        with torch.no_grad():
            self.spatial.weight.copy_(torch.from_numpy(spatial_weights)[:, :, None])

        frontend_levels = {
            "depth3": 3,
            "overcomplete_depth3_depth4": 4,
            "overcomplete_depth3_depth4_depth5": 5,
        }
        if wavelet_frontend not in frontend_levels:
            raise ValueError(f"unsupported wavelet frontend {wavelet_frontend!r}")
        wavelet_type = (
            OvercompleteWaveletPacketEnergy
            if frontend_levels[wavelet_frontend] > 3
            else WaveletPacketEnergy
        )
        self.wavelet = wavelet_type(
            wavelet="bior6.8",
            levels=frontend_levels[wavelet_frontend],
            kernel_size=17,
            trainable=True,
            padding_mode="constant",
            energy_window_samples=energy_window_samples,
            energy_stride_samples=energy_window_samples,
            tap_resample_up=tap_resample_up,
            tap_resample_down=tap_resample_down,
            interlevel_skip=wavelet_interlevel_skip,
            interlevel_normalization=wavelet_interlevel_normalization,
            normalize_final_level=wavelet_final_normalization,
        )
        self.wavelet_frontend = wavelet_frontend
        self.wavelet_band_count = int(
            getattr(self.wavelet, "output_band_count", 2**self.wavelet.levels)
        )
        self.samples_per_bin = int(energy_window_samples)
        self.register_buffer(
            "selected_indices",
            torch.as_tensor(initialization["selected_indices"], dtype=torch.long),
        )
        self.register_buffer(
            "frequency_order",
            torch.as_tensor(frontend_frequency_order(self.wavelet), dtype=torch.long),
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(initialization["feature_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(initialization["feature_scale"], dtype=torch.float32),
        )

        if residual_input not in (
            "selected",
            "candidate",
            "current_candidate",
            "causal_candidate",
            "selected_causal",
        ):
            raise ValueError(f"unsupported residual input {residual_input!r}")
        candidate_residual = residual_input in (
            "candidate",
            "current_candidate",
            "causal_candidate",
        )
        causal_residual = residual_input in ("causal_candidate", "selected_causal")
        if (
            residual_input != "selected"
            and recurrent_cell not in ("residual_lstm", "residual_gru")
        ):
            raise ValueError(
                "candidate residual input requires residual_lstm or residual_gru"
            )
        self.residual_input = residual_input
        if residual_dynamics not in ("pointwise", "leaky_velocity"):
            raise ValueError(f"unsupported residual dynamics {residual_dynamics!r}")
        if residual_dynamics != "pointwise" and recurrent_cell not in (
            "residual_lstm",
            "residual_gru",
        ):
            raise ValueError("leaky residual dynamics require a residual decoder")
        if not 0.0 <= residual_decay <= 1.0:
            raise ValueError("residual decay must be between zero and one")
        self.residual_dynamics = residual_dynamics
        self.residual_decay = float(residual_decay)
        if not 1 <= residual_history_bins <= HISTORY:
            raise ValueError(
                f"residual_history_bins must be between 1 and {HISTORY}"
            )
        if residual_input != "current_candidate" and residual_history_bins != 1:
            raise ValueError(
                "residual_history_bins only applies to current_candidate input"
            )
        self.residual_history_bins = int(residual_history_bins)
        if residual_input_width is not None and residual_input_width <= 0:
            raise ValueError("residual_input_width must be positive")
        if (
            residual_input
            not in ("current_candidate", "causal_candidate", "selected_causal")
            and residual_input_width is not None
        ):
            raise ValueError(
                "residual_input_width only applies to current/causal candidate input"
            )
        self.residual_input_width = residual_input_width
        if residual_include_direct and recurrent_cell not in (
            "residual_lstm",
            "residual_gru",
        ):
            raise ValueError(
                "residual_include_direct requires a residual recurrent cell"
            )
        self.residual_include_direct = bool(residual_include_direct)
        if candidate_residual:
            required = (
                "candidate_indices",
                "candidate_feature_mean",
                "candidate_feature_scale",
                "selected_candidate_positions",
            )
            missing = [name for name in required if name not in initialization]
            if missing:
                raise ValueError(
                    "candidate residual initialization is missing " + ", ".join(missing)
                )
            self.register_buffer(
                "candidate_indices",
                torch.as_tensor(initialization["candidate_indices"], dtype=torch.long),
            )
            self.register_buffer(
                "candidate_feature_mean",
                torch.as_tensor(
                    initialization["candidate_feature_mean"], dtype=torch.float32
                ),
            )
            self.register_buffer(
                "candidate_feature_scale",
                torch.as_tensor(
                    initialization["candidate_feature_scale"], dtype=torch.float32
                ),
            )
            self.register_buffer(
                "selected_candidate_positions",
                torch.as_tensor(
                    initialization["selected_candidate_positions"], dtype=torch.long
                ),
            )
            if residual_input == "current_candidate":
                per_bin = components * self.wavelet_band_count
                current_positions = (
                    np.asarray(initialization["current_candidate_positions"])
                    if self.residual_history_bins == 1
                    and "current_candidate_positions" in initialization
                    else np.flatnonzero(
                        np.asarray(initialization["candidate_indices"])
                        // per_bin
                        >= HISTORY - self.residual_history_bins
                    )
                )
                if np.asarray(current_positions).size == 0:
                    raise ValueError("candidate pool contains no current-bin features")
                self.register_buffer(
                    "current_candidate_positions",
                    torch.as_tensor(
                        current_positions,
                        dtype=torch.long,
                    ),
                )
            elif residual_input == "causal_candidate":
                required_causal = (
                    "causal_stream_positions",
                    "causal_feature_mean",
                    "causal_feature_scale",
                )
                missing_causal = [
                    name for name in required_causal if name not in initialization
                ]
                if missing_causal:
                    raise ValueError(
                        "causal residual initialization is missing "
                        + ", ".join(missing_causal)
                    )
                self.register_buffer(
                    "causal_stream_positions",
                    torch.as_tensor(
                        initialization["causal_stream_positions"], dtype=torch.long
                    ),
                )
                self.register_buffer(
                    "causal_feature_mean",
                    torch.as_tensor(
                        initialization["causal_feature_mean"], dtype=torch.float32
                    ),
                )
                self.register_buffer(
                    "causal_feature_scale",
                    torch.as_tensor(
                        initialization["causal_feature_scale"], dtype=torch.float32
                    ),
                )
        if residual_input == "selected_causal":
            required_selected_causal = (
                "selected_causal_stream_positions",
                "selected_causal_feature_mean",
                "selected_causal_feature_scale",
            )
            missing_selected_causal = [
                name
                for name in required_selected_causal
                if name not in initialization
            ]
            if missing_selected_causal:
                raise ValueError(
                    "selected-causal residual initialization is missing "
                    + ", ".join(missing_selected_causal)
                )
            self.register_buffer(
                "selected_causal_stream_positions",
                torch.as_tensor(
                    initialization["selected_causal_stream_positions"],
                    dtype=torch.long,
                ),
            )
            self.register_buffer(
                "selected_causal_feature_mean",
                torch.as_tensor(
                    initialization["selected_causal_feature_mean"],
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "selected_causal_feature_scale",
                torch.as_tensor(
                    initialization["selected_causal_feature_scale"],
                    dtype=torch.float32,
                ),
            )

        if output_activation not in ("linear", "softplus"):
            raise ValueError(f"unsupported output activation {output_activation!r}")
        if softplus_beta <= 0:
            raise ValueError("softplus beta must be positive")
        self.output_activation = output_activation
        self.softplus_beta = float(softplus_beta)
        self.head_initialization = "lars_linear_regime"

        feature_count = int(self.selected_indices.numel())
        if residual_input == "candidate":
            recurrent_feature_count = int(self.candidate_indices.numel())
        elif residual_input == "current_candidate":
            current_feature_count = int(self.current_candidate_positions.numel())
            if residual_input_width is not None and residual_input_width < current_feature_count:
                raise ValueError(
                    f"residual_input_width {residual_input_width} is smaller than "
                    f"the {current_feature_count} current candidates"
                )
            recurrent_feature_count = residual_input_width or current_feature_count
        elif causal_residual:
            positions = (
                self.causal_stream_positions
                if residual_input == "causal_candidate"
                else self.selected_causal_stream_positions
            )
            causal_feature_count = int(positions.numel())
            if (
                residual_input_width is not None
                and residual_input_width < causal_feature_count
            ):
                raise ValueError(
                    f"residual_input_width {residual_input_width} is smaller than "
                    f"the {causal_feature_count} causal candidates"
                )
            recurrent_feature_count = residual_input_width or causal_feature_count
        else:
            recurrent_feature_count = feature_count
        if self.residual_include_direct:
            recurrent_feature_count += 1
        self.direct = nn.Linear(feature_count, 1)
        self.residual_decoder = recurrent_cell in ("residual_lstm", "residual_gru")
        if recurrent_cell == "standard":
            self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        elif recurrent_cell == "paper_equations":
            self.lstm = PaperEquationLSTM(feature_count, hidden_size, batch_first=True)
        elif recurrent_cell == "residual_lstm":
            self.lstm = nn.LSTM(recurrent_feature_count, hidden_size, batch_first=True)
        elif recurrent_cell == "residual_gru":
            self.lstm = nn.GRU(recurrent_feature_count, hidden_size, batch_first=True)
        else:
            raise ValueError(f"unsupported recurrent cell {recurrent_cell!r}")
        self.recurrent_cell = recurrent_cell
        self.output = nn.Linear(hidden_size, 1)
        self.movement_output = nn.Linear(hidden_size, 1) if movement_head else None
        self.velocity_output = nn.Linear(hidden_size, 1) if velocity_head else None
        if movement_modulation and self.movement_output is None:
            raise ValueError("movement modulation requires a movement output head")
        self.movement_modulation = bool(movement_modulation)
        self.movement_gain = (
            nn.Parameter(torch.zeros(())) if movement_modulation else None
        )
        self.signed_pooling_gates = (
            nn.Parameter(torch.zeros(self.wavelet_band_count))
            if wavelet_signed_pooling
            else None
        )
        if residual_output_init_std < 0:
            raise ValueError("residual output initialization scale must be nonnegative")
        with torch.no_grad():
            self.direct.weight.copy_(
                torch.as_tensor(initialization["coefficients"])[None]
            )
            self.direct.bias.fill_(float(initialization["intercept"]))
        self.direct.requires_grad_(False)
        if candidate_residual:
            expanded_weight = torch.zeros(
                int(self.candidate_indices.numel()), dtype=self.direct.weight.dtype
            )
            selected_positions = self.selected_candidate_positions
            selected_scale = self.feature_scale
            candidate_scale = self.candidate_feature_scale.index_select(
                0, selected_positions
            )
            coefficient = self.direct.weight.detach()[0]
            expanded_weight[selected_positions] = (
                coefficient * candidate_scale / selected_scale
            )
            candidate_mean = self.candidate_feature_mean.index_select(
                0, selected_positions
            )
            expanded_bias = self.direct.bias.detach()[0] + torch.sum(
                coefficient * (candidate_mean - self.feature_mean) / selected_scale
            )
            self.register_buffer(
                "candidate_direct_weight", expanded_weight[None], persistent=False
            )
            self.register_buffer(
                "candidate_direct_bias", expanded_bias[None], persistent=False
            )
        if self.residual_decoder:
            self.head_initialization = (
                "near_zero_residual_on_lars"
                if residual_output_init_std
                else "zero_residual_on_lars"
            )
            if residual_output_init_std:
                nn.init.normal_(
                    self.output.weight, mean=0.0, std=residual_output_init_std
                )
            else:
                nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            self._initialize_lars_linear_regime(
                coefficients=np.asarray(initialization["coefficients"]),
                intercept=float(initialization["intercept"]),
                candidate_scale=lars_candidate_scale,
                near_zero_std=near_zero_std,
                open_gate_bias=open_gate_bias,
                forget_gate_bias=forget_gate_bias,
            )
        if self.movement_output is not None:
            with torch.no_grad():
                self.movement_output.weight.normal_(0.0, near_zero_std)
                self.movement_output.bias.zero_()
        if self.velocity_output is not None:
            with torch.no_grad():
                self.velocity_output.weight.normal_(0.0, near_zero_std)
                self.velocity_output.bias.zero_()

    @torch.no_grad()
    def _initialize_lars_linear_regime(
        self,
        coefficients: np.ndarray,
        intercept: float,
        candidate_scale: float,
        near_zero_std: float,
        open_gate_bias: float,
        forget_gate_bias: float,
        unit: int = 0,
    ) -> None:
        """Embed LARS in one standard LSTM unit without disabling nonlinearity."""
        if not 0 <= unit < self.lstm.hidden_size:
            raise ValueError("LARS unit is outside the LSTM hidden state")
        if candidate_scale <= 0:
            raise ValueError("lars_candidate_scale must be positive")
        if near_zero_std < 0:
            raise ValueError("near_zero_std must be nonnegative")

        coefficient = torch.as_tensor(
            coefficients,
            dtype=self.lstm.weight_ih_l0.dtype,
            device=self.lstm.weight_ih_l0.device,
        )
        if coefficient.shape != (self.lstm.input_size,):
            raise ValueError("LARS coefficient count must equal the LSTM input size")

        for parameter in (
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.output.weight,
        ):
            parameter.normal_(0.0, near_zero_std)
        self.lstm.bias_ih_l0.zero_()
        self.lstm.bias_hh_l0.zero_()
        self.output.bias.zero_()

        hidden = self.lstm.hidden_size
        input_row = unit
        forget_row = hidden + unit
        candidate_row = 2 * hidden + unit
        output_row = 3 * hidden + unit
        self.lstm.bias_ih_l0[input_row] = open_gate_bias
        self.lstm.bias_ih_l0[forget_row] = forget_gate_bias
        self.lstm.bias_ih_l0[output_row] = open_gate_bias
        self.lstm.bias_ih_l0[candidate_row] = candidate_scale * intercept
        self.lstm.weight_ih_l0[candidate_row].copy_(candidate_scale * coefficient)

        open_gate = torch.sigmoid(
            torch.as_tensor(
                open_gate_bias,
                dtype=self.output.weight.dtype,
                device=self.output.weight.device,
            )
        )
        self.output.weight[0, unit] = 1.0 / (
            candidate_scale * open_gate.square()
        )

    @property
    def context_samples(self) -> int:
        return self.wavelet.effective_kernel_size // 2

    def activate_output(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "softplus":
            return F.softplus(prediction, beta=self.softplus_beta)
        return prediction

    def _decode_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent_features, direct_features = self._prepare_features(features)
        direct_prediction = None
        if self.residual_decoder:
            direct_prediction = self._direct_prediction(
                recurrent_features, direct_features
            )
            if self.residual_include_direct:
                recurrent_features = torch.cat(
                    (recurrent_features, direct_prediction), dim=-1
                )
        recurrent, _ = self.lstm(recurrent_features)
        prediction = self.output(recurrent)
        if self.residual_decoder:
            if self.residual_dynamics == "leaky_velocity":
                prediction = leaky_integrate(prediction, self.residual_decay)
            prediction = direct_prediction + prediction
        trajectory = self.activate_output(prediction).squeeze(-1)
        if self.movement_modulation:
            movement_logit = self.movement_output(recurrent).squeeze(-1)
            movement_state = 2.0 * torch.sigmoid(movement_logit) - 1.0
            log_gain = np.log(3.0) * torch.tanh(self.movement_gain)
            trajectory = trajectory * torch.exp(log_gain * movement_state)
        return trajectory, recurrent

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        prediction, _ = self._decode_features(features)
        return prediction

    def decode_features_with_movement(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return trajectory and auxiliary target-finger movement logits."""
        if self.movement_output is None:
            raise RuntimeError("decoder has no auxiliary movement head")
        prediction, recurrent = self._decode_features(features)
        return prediction, self.movement_output(recurrent).squeeze(-1)

    def decode_features_with_auxiliary(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return trajectory followed by the enabled training-only outputs."""
        prediction, recurrent = self._decode_features(features)
        outputs = [prediction]
        if self.movement_output is not None:
            outputs.append(self.movement_output(recurrent).squeeze(-1))
        if self.velocity_output is not None:
            outputs.append(self.velocity_output(recurrent).squeeze(-1))
        if len(outputs) == 1:
            raise RuntimeError("decoder has no auxiliary output head")
        return tuple(outputs)

    def _prepare_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardize recurrent input and recover the selected LARS subset."""
        if self.residual_input in ("causal_candidate", "selected_causal"):
            candidate_mode = self.residual_input == "causal_candidate"
            direct_count = (
                int(self.candidate_indices.numel())
                if candidate_mode
                else int(self.selected_indices.numel())
            )
            direct_raw = features[..., :direct_count]
            causal = features[..., direct_count:]
            direct_mean = (
                self.candidate_feature_mean if candidate_mode else self.feature_mean
            )
            direct_scale = (
                self.candidate_feature_scale if candidate_mode else self.feature_scale
            )
            causal_mean = (
                self.causal_feature_mean
                if candidate_mode
                else self.selected_causal_feature_mean
            )
            causal_scale = (
                self.causal_feature_scale
                if candidate_mode
                else self.selected_causal_feature_scale
            )
            direct = (direct_raw - direct_mean) / direct_scale
            recurrent = (causal - causal_mean) / causal_scale
            if self.residual_input_width is not None:
                recurrent = F.pad(
                    recurrent,
                    (0, self.residual_input_width - recurrent.shape[-1]),
                )
            return recurrent, direct
        if self.residual_input in ("candidate", "current_candidate"):
            standardized = (
                features - self.candidate_feature_mean
            ) / self.candidate_feature_scale
            recurrent = (
                standardized
                if self.residual_input == "candidate"
                else standardized.index_select(-1, self.current_candidate_positions)
            )
            if self.residual_input_width is not None:
                recurrent = F.pad(
                    recurrent,
                    (0, self.residual_input_width - recurrent.shape[-1]),
                )
            return recurrent, standardized
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized, standardized

    def _direct_prediction(
        self,
        recurrent_features: torch.Tensor,
        direct_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.residual_input in (
            "candidate",
            "current_candidate",
            "causal_candidate",
        ):
            return F.linear(
                direct_features,
                self.candidate_direct_weight,
                self.candidate_direct_bias,
            )
        return self.direct(direct_features)

    def direct_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return the immutable LARS audit prediction for the same features."""
        recurrent_features, direct_features = self._prepare_features(features)
        return self.activate_output(
            self._direct_prediction(recurrent_features, direct_features)
        ).squeeze(-1)

    def extract_sequences(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Extract continuous-filter histories for several binned sequences."""
        context = self.context_samples
        filtered_bins = steps + HISTORY - 1
        samples = filtered_bins * self.samples_per_bin
        length = samples + 2 * context
        offsets = torch.arange(length, device=starts.device)
        sample_indices = starts[:, None] * self.samples_per_bin + offsets[None]
        segments = padded_ecog.index_select(0, sample_indices.reshape(-1)).reshape(
            starts.numel(), length, padded_ecog.shape[1]
        )

        spatial = self.spatial(segments.transpose(1, 2))
        batch, components, _ = spatial.shape
        bands = spatial.reshape(batch * components, 1, length)
        bands = self.wavelet.transform(bands)
        bands = bands[..., context : context + samples]

        energy = self.wavelet._energy(bands)
        if self.signed_pooling_gates is not None:
            signed_mean = F.avg_pool1d(
                bands,
                kernel_size=self.samples_per_bin,
                stride=self.samples_per_bin,
            )
            energy = energy + self.signed_pooling_gates[None, :, None] * signed_mean
        energy = energy.reshape(
            batch, components, self.wavelet_band_count, filtered_bins
        )
        per_bin = (
            energy.index_select(2, self.frequency_order)
            .permute(0, 3, 1, 2)
            .flatten(2)
        )
        history = per_bin.unfold(1, HISTORY, 1).permute(0, 1, 3, 2).flatten(2)
        if self.residual_input in ("causal_candidate", "selected_causal"):
            candidate_mode = self.residual_input == "causal_candidate"
            direct_indices = (
                self.candidate_indices if candidate_mode else self.selected_indices
            )
            causal_positions = (
                self.causal_stream_positions
                if candidate_mode
                else self.selected_causal_stream_positions
            )
            direct = history.index_select(2, direct_indices)
            causal = per_bin[:, HISTORY - 1 :, :].index_select(
                2, causal_positions
            )
            return torch.cat((direct, causal), dim=2)
        indices = (
            self.candidate_indices
            if self.residual_input in ("candidate", "current_candidate")
            else self.selected_indices
        )
        return history.index_select(2, indices)

    def forward(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        return self.decode_features(self.extract_sequences(padded_ecog, starts, steps))

    def forward_with_movement(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode_features_with_movement(
            self.extract_sequences(padded_ecog, starts, steps)
        )

    def forward_with_auxiliary(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> tuple[torch.Tensor, ...]:
        return self.decode_features_with_auxiliary(
            self.extract_sequences(padded_ecog, starts, steps)
        )


@torch.inference_mode()
def predict_intervals(
    model: SingleWaveletDecoder,
    features: torch.Tensor,
    intervals: list[list[int]],
    rows: int,
) -> np.ndarray:
    model.eval()
    prediction = np.full(rows, np.nan, dtype=np.float32)
    for begin, end in intervals:
        prediction[begin:end] = (
            model.decode_features(features[begin:end][None])[0].float().cpu().numpy()
        )
    return prediction


@torch.inference_mode()
def extract_all(
    model: SingleWaveletDecoder,
    padded_ecog: torch.Tensor,
    rows: int,
    chunk_steps: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for begin in range(0, rows, chunk_steps):
        steps = min(chunk_steps, rows - begin)
        chunks.append(
            model.extract_sequences(
                padded_ecog,
                torch.as_tensor([begin], device=padded_ecog.device),
                steps,
            )[0]
        )
    return torch.cat(chunks, dim=0)
