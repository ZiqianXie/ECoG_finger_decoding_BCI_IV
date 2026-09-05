"""Joint spatial-spectrotemporal models for ECoG trajectory decoding.

Electrodes are represented as input planes rather than as a convolutional
axis.  A two-dimensional kernel therefore learns a separate electrode weight
for every frequency/time offset without assuming that adjacent channel
numbers are physically adjacent.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import WaveletPacketEnergy, fit_fastica_spatial_weights


def wavelet_packet_frequency_order(levels: int) -> tuple[int, ...]:
    """Return natural-tree indices in monotonically increasing frequency order.

    Quadrature-mirror wavelet trees reverse the frequency orientation of every
    high-pass child.  Binary tree order is consequently not a valid image axis.
    Binary-reflected Gray-code order restores frequency adjacency.
    """
    if levels < 1:
        raise ValueError("levels must be positive")
    return tuple(index ^ (index >> 1) for index in range(2**levels))


def analytic_signal(x: torch.Tensor) -> torch.Tensor:
    """Construct a differentiable analytic signal along the final dimension."""
    if x.shape[-1] < 2:
        raise ValueError("analytic signal requires at least two time samples")
    sample_count = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    multiplier = torch.zeros(
        sample_count,
        dtype=x.dtype,
        device=x.device,
    )
    multiplier[0] = 1.0
    if sample_count % 2 == 0:
        multiplier[1 : sample_count // 2] = 2.0
        multiplier[sample_count // 2] = 1.0
    else:
        multiplier[1 : (sample_count + 1) // 2] = 2.0
    return torch.fft.ifft(spectrum * multiplier, dim=-1)


class HilbertQuadrature(nn.Module):
    """Real FIR approximation to a 90-degree Hilbert phase shift.

    Keeping the paired branch real avoids complex-valued compiler fallbacks and
    mirrors the historical idea of using two phase-shifted filters.  The fixed
    antisymmetric kernel is differentiable with respect to its input.
    """

    def __init__(self, kernel_size: int = 129) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        half = kernel_size // 2
        offsets = np.arange(-half, half + 1, dtype=np.float64)
        impulse = np.zeros(kernel_size, dtype=np.float64)
        odd = np.remainder(np.abs(offsets).astype(np.int64), 2) == 1
        impulse[odd] = 2.0 / (np.pi * offsets[odd])
        impulse *= np.hamming(kernel_size)
        # conv1d performs cross-correlation, hence the reversal.
        kernel = torch.as_tensor(impulse[::-1].copy(), dtype=torch.float32)
        self.register_buffer("kernel", kernel[None, None])
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, band, time); got {tuple(x.shape)}"
            )
        batch, bands, time = x.shape
        flattened = x.reshape(batch * bands, 1, time)
        half = self.kernel_size // 2
        mode = "reflect" if half < time else "replicate"
        shifted = F.conv1d(F.pad(flattened, (half, half), mode=mode), self.kernel)
        return shifted.reshape(batch, bands, time)


class WaveletPacketSpectrogram(nn.Module):
    """Create frequency-ordered energy and optional phase feature maps.

    Input has shape ``(batch, component, samples)``.  Output has shape
    ``(batch, component * representation_count, frequency, time_bin)``.
    Energy-only mode exactly reuses :class:`WaveletPacketEnergy`.  Phase mode
    appends amplitude-gated cosine and sine maps derived from a differentiable
    Hilbert transform of each final wavelet-packet path.
    """

    def __init__(
        self,
        wavelet: str = "bior6.8",
        levels: int = 3,
        kernel_size: int = 17,
        trainable: bool = True,
        padding_mode: str = "constant",
        energy_window_samples: int = 40,
        energy_stride_samples: int = 40,
        include_phase: bool = False,
        phase_epsilon: float = 1.0e-6,
        phase_kernel_size: int = 129,
    ) -> None:
        super().__init__()
        if phase_epsilon <= 0:
            raise ValueError("phase_epsilon must be positive")
        self.packet = WaveletPacketEnergy(
            wavelet=wavelet,
            levels=levels,
            kernel_size=kernel_size,
            trainable=trainable,
            padding_mode=padding_mode,
            energy_window_samples=energy_window_samples,
            energy_stride_samples=energy_stride_samples,
        )
        self.register_buffer(
            "frequency_order",
            torch.as_tensor(wavelet_packet_frequency_order(levels), dtype=torch.long),
            persistent=False,
        )
        self.include_phase = bool(include_phase)
        self.phase_epsilon = float(phase_epsilon)
        self.quadrature = HilbertQuadrature(phase_kernel_size)

    @property
    def frequency_count(self) -> int:
        return 2**self.packet.levels

    @property
    def representation_count(self) -> int:
        return 3 if self.include_phase else 1

    @property
    def representation_names(self) -> tuple[str, ...]:
        if self.include_phase:
            return ("log_energy", "amplitude_gated_cos_phase", "amplitude_gated_sin_phase")
        return ("log_energy",)

    def _pooled_phase(self, bands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quadrature = self.quadrature(bands)
        amplitude = torch.sqrt(
            bands.square() + quadrature.square() + self.phase_epsilon**2
        )
        gate = torch.log1p(amplitude)
        kernel = self.packet.energy_window_samples
        stride = self.packet.energy_stride_samples
        cosine = F.avg_pool1d(
            gate * bands / amplitude,
            kernel_size=kernel,
            stride=stride,
        )
        sine = F.avg_pool1d(
            gate * quadrature / amplitude,
            kernel_size=kernel,
            stride=stride,
        )
        return cosine, sine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, component, samples); got {tuple(x.shape)}"
            )
        batch, components, samples = x.shape
        if samples < self.packet.energy_window_samples:
            raise ValueError("time dimension is shorter than the energy window")

        bands = x.reshape(batch * components, 1, samples)
        for layer in self.packet.layers:
            bands = self.packet._same_filter(bands, layer)
            bands = 1.7156 * torch.tanh((2.0 / 3.0) * bands)

        energy = self.packet._energy(bands).index_select(1, self.frequency_order)
        time_bins = energy.shape[-1]
        energy = energy.reshape(batch, components, self.frequency_count, time_bins)
        if not self.include_phase:
            return energy

        cosine, sine = self._pooled_phase(bands)
        cosine = cosine.index_select(1, self.frequency_order).reshape(
            batch, components, self.frequency_count, time_bins
        )
        sine = sine.index_select(1, self.frequency_order).reshape(
            batch, components, self.frequency_count, time_bins
        )
        # Keep the three representations for each component adjacent so that a
        # downstream kernel can easily inspect amplitude and phase together.
        return torch.stack((energy, cosine, sine), dim=2).reshape(
            batch,
            components * self.representation_count,
            self.frequency_count,
            time_bins,
        )


class SoftScaleSpectroTemporalConv2d(nn.Module):
    """Mixture of full multichannel frequency-by-time convolutions.

    Every branch mixes all spatial/phase input planes.  Softmax gates select a
    kernel height and width separately for every output unit.  The gates can be
    optimized continuously and later pruned to one scale with
    :meth:`selected_kernel_sizes`.
    """

    def __init__(
        self,
        input_planes: int,
        output_units: int,
        kernel_sizes: Sequence[tuple[int, int]] = ((1, 3), (3, 5), (5, 9)),
        include_frequency_coordinates: bool = True,
        causal_time: bool = False,
        activation: str = "silu",
        normalization: str = "none",
    ) -> None:
        super().__init__()
        parsed = tuple((int(height), int(width)) for height, width in kernel_sizes)
        if input_planes < 1 or output_units < 1:
            raise ValueError("input_planes and output_units must be positive")
        if not parsed:
            raise ValueError("at least one kernel size is required")
        if any(height < 1 or width < 1 for height, width in parsed):
            raise ValueError("kernel sizes must be positive")
        if any(height % 2 == 0 or width % 2 == 0 for height, width in parsed):
            raise ValueError("kernel heights and widths must be odd")
        if len(set(parsed)) != len(parsed):
            raise ValueError("kernel sizes must be unique")
        if activation not in {"silu", "tanh", "identity"}:
            raise ValueError("activation must be 'silu', 'tanh', or 'identity'")
        if normalization not in {"none", "group"}:
            raise ValueError("normalization must be 'none' or 'group'")

        coordinate_planes = 2 if include_frequency_coordinates else 0
        self.branches = nn.ModuleList(
            nn.Conv2d(
                input_planes + coordinate_planes,
                output_units,
                kernel_size=size,
                padding=0,
            )
            for size in parsed
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(parsed), output_units))
        self.normalization = (
            nn.Identity() if normalization == "none" else nn.GroupNorm(1, output_units)
        )
        self.kernel_sizes = parsed
        self.input_planes = int(input_planes)
        self.output_units = int(output_units)
        self.include_frequency_coordinates = bool(include_frequency_coordinates)
        self.causal_time = bool(causal_time)
        self.activation = activation
        self.normalization_name = normalization

    def scale_probabilities(self) -> torch.Tensor:
        """Return ``(scale, output_unit)`` differentiable selection weights."""
        return torch.softmax(self.scale_logits, dim=0)

    def selected_kernel_sizes(self) -> tuple[tuple[int, int], ...]:
        """Return the maximum-probability kernel size for every output unit."""
        selected = self.scale_probabilities().detach().argmax(dim=0).cpu().tolist()
        return tuple(self.kernel_sizes[index] for index in selected)

    def scale_entropy(self) -> torch.Tensor:
        """Mean gate entropy, useful as an optional pruning regularizer."""
        probabilities = self.scale_probabilities()
        return -(probabilities * probabilities.clamp_min(1.0e-8).log()).sum(dim=0).mean()

    def _append_frequency_coordinates(self, x: torch.Tensor) -> torch.Tensor:
        if not self.include_frequency_coordinates:
            return x
        batch, _, frequencies, time = x.shape
        coordinate = torch.linspace(-1.0, 1.0, frequencies, dtype=x.dtype, device=x.device)
        coordinate = coordinate.view(1, 1, frequencies, 1).expand(batch, 1, frequencies, time)
        return torch.cat((x, coordinate, coordinate.square()), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"x must have shape (batch, input_plane, frequency, time); got {tuple(x.shape)}"
            )
        if x.shape[1] != self.input_planes:
            raise ValueError("input plane count does not match the configured convolution")
        augmented = self._append_frequency_coordinates(x)
        branch_outputs: list[torch.Tensor] = []
        for branch, (height, width) in zip(self.branches, self.kernel_sizes, strict=True):
            time_padding = (width - 1, 0) if self.causal_time else (width // 2, width // 2)
            padded = F.pad(
                augmented,
                (*time_padding, height // 2, height // 2),
            )
            branch_outputs.append(branch(padded))
        stacked = torch.stack(branch_outputs, dim=1)
        probabilities = self.scale_probabilities().T.reshape(
            1, len(self.branches), self.output_units, 1, 1
        )
        mixed = (stacked * probabilities).sum(dim=1)
        mixed = self.normalization(mixed)
        if self.activation == "silu":
            return F.silu(mixed)
        if self.activation == "tanh":
            return torch.tanh(mixed)
        return mixed


class SpectroTemporalWaveletDecoder(nn.Module):
    """Exact-window decoder with joint spatial-spectrotemporal filters.

    Input windows have shape ``(batch, sequence, electrode, samples)``.  The
    wavelet map is convolved over frequency and within-window time while each
    output kernel jointly mixes every spatial component.  An LSTM then models
    dependencies between consecutive trajectory bins.
    """

    def __init__(
        self,
        input_channels: int,
        window_samples: int = 1000,
        spatial_components: int | None = None,
        wavelet_levels: int = 3,
        energy_window_samples: int = 40,
        energy_stride_samples: int = 40,
        include_phase: bool = False,
        convolution_units: int = 16,
        kernel_sizes: Sequence[tuple[int, int]] = ((1, 3), (3, 5), (5, 9)),
        feature_width: int = 64,
        hidden_size: int = 16,
        recurrent_layers: int = 1,
        output_fingers: int = 1,
        dropout: float = 0.0,
        feature_normalization: str = "none",
        output_activation: str = "softplus",
        softplus_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if input_channels < 1 or window_samples < 1 or output_fingers < 1:
            raise ValueError("input, window, and output dimensions must be positive")
        if recurrent_layers < 1:
            raise ValueError("recurrent_layers must be positive")
        if output_activation not in {"linear", "relu", "softplus"}:
            raise ValueError("unsupported output activation")
        if feature_normalization not in {"none", "layer"}:
            raise ValueError("feature_normalization must be 'none' or 'layer'")
        if softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive")
        spatial_components = input_channels if spatial_components is None else int(spatial_components)
        if not 1 <= spatial_components <= input_channels:
            raise ValueError("spatial_components must be between 1 and input_channels")
        time_bins = 1 + (window_samples - energy_window_samples) // energy_stride_samples
        if time_bins < 1:
            raise ValueError("window_samples must be at least the energy window")

        self.spatial_projection = nn.Conv1d(
            input_channels,
            spatial_components,
            kernel_size=1,
            bias=False,
        )
        nn.init.orthogonal_(self.spatial_projection.weight[:, :, 0])
        self.spectrogram = WaveletPacketSpectrogram(
            levels=wavelet_levels,
            trainable=True,
            padding_mode="constant",
            energy_window_samples=energy_window_samples,
            energy_stride_samples=energy_stride_samples,
            include_phase=include_phase,
        )
        self.spectrotemporal = SoftScaleSpectroTemporalConv2d(
            input_planes=spatial_components * self.spectrogram.representation_count,
            output_units=convolution_units,
            kernel_sizes=kernel_sizes,
            include_frequency_coordinates=True,
        )
        flattened = convolution_units * self.spectrogram.frequency_count * time_bins
        self.feature_projection = nn.Sequential(
            nn.Identity()
            if feature_normalization == "none"
            else nn.LayerNorm(flattened),
            nn.Linear(flattened, feature_width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.recurrent = nn.LSTM(
            feature_width,
            hidden_size,
            num_layers=recurrent_layers,
            dropout=dropout if recurrent_layers > 1 else 0.0,
            batch_first=True,
        )
        self.direct_output = nn.Linear(feature_width, output_fingers)
        nn.init.normal_(self.direct_output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.direct_output.bias)
        self.output = nn.Linear(hidden_size, output_fingers)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.window_samples = int(window_samples)
        self.output_activation = output_activation
        self.softplus_beta = float(softplus_beta)
        self.feature_normalization = feature_normalization

    def initialize_spatial_from_fastica(
        self,
        training_ecog: np.ndarray,
        max_samples: int | None = 50_000,
        random_state: int = 0,
        backend: str = "sklearn",
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Initialize the spatial input planes using training-only FastICA."""
        weights = fit_fastica_spatial_weights(
            training_ecog,
            n_components=self.spatial_projection.out_channels,
            max_samples=max_samples,
            random_state=random_state,
            backend=backend,
            device=device,
        )
        with torch.no_grad():
            self.spatial_projection.weight[:, :, 0].copy_(
                torch.as_tensor(
                    weights,
                    dtype=self.spatial_projection.weight.dtype,
                    device=self.spatial_projection.weight.device,
                )
            )
        return weights

    def extract(self, windows: torch.Tensor) -> torch.Tensor:
        """Return one learned feature vector for every exact input window."""
        if windows.ndim != 4:
            raise ValueError(
                "windows must have shape (batch, sequence, channel, samples); "
                f"got {tuple(windows.shape)}"
            )
        batch, sequence, channels, samples = windows.shape
        if channels != self.spatial_projection.in_channels:
            raise ValueError("window channel count does not match the model")
        if samples != self.window_samples:
            raise ValueError(
                f"expected {self.window_samples} samples per window; got {samples}"
            )
        flattened = windows.reshape(batch * sequence, channels, samples)
        spatial = self.spatial_projection(flattened)
        time_frequency = self.spectrogram(spatial)
        motifs = self.spectrotemporal(time_frequency).flatten(start_dim=1)
        features = self.feature_projection(motifs)
        return features.reshape(batch, sequence, -1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.decode(self.extract(windows))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode a precomputed ``(batch, sequence, feature)`` tensor."""
        if features.ndim != 3:
            raise ValueError(
                "features must have shape (batch, sequence, feature); "
                f"got {tuple(features.shape)}"
            )
        recurrent, _ = self.recurrent(features)
        prediction = self.direct_output(features) + self.output(recurrent)
        if self.output_activation == "relu":
            prediction = torch.relu(prediction)
        elif self.output_activation == "softplus":
            prediction = F.softplus(prediction, beta=self.softplus_beta)
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        return prediction


def label_independent_morlet_centers(
    atom_count: int,
    minimum_hz: float,
    maximum_hz: float,
    scale_hz: float = 20.0,
) -> torch.Tensor:
    """Smoothly cover a spectrum without using task labels or tuned bands."""
    if atom_count < 2:
        raise ValueError("atom_count must be at least two")
    if not 0.0 < minimum_hz < maximum_hz:
        raise ValueError("require 0 < minimum_hz < maximum_hz")
    if scale_hz <= 0:
        raise ValueError("scale_hz must be positive")
    low = np.arcsinh(minimum_hz / scale_hz)
    high = np.arcsinh(maximum_hz / scale_hz)
    coordinates = np.linspace(low, high, atom_count)
    return torch.as_tensor(scale_hz * np.sinh(coordinates), dtype=torch.float32)


class MorletAtomBank(nn.Module):
    """Constrained complex filters with trainable center and temporal width.

    The returned map has shape ``(batch, component, representation, atom,
    time_bin)``. Representations are signed real coefficient, log energy, and
    the cosine/sine phase phasor. A separate signed-plus-log-energy broadband
    residual is returned with shape ``(batch, 2 * component, time_bin)``.
    """

    representation_names = (
        "signed_coefficient",
        "log_energy",
        "phase_cosine",
        "phase_sine",
    )

    def __init__(
        self,
        atom_count: int = 16,
        sampling_rate_hz: float = 1000.0,
        kernel_size: int = 1001,
        minimum_hz: float = 0.5,
        maximum_hz: float = 450.0,
        minimum_width_seconds: float = 0.001,
        maximum_width_seconds: float = 0.250,
        initial_cycles: float = 4.5,
        pool_samples: int = 40,
        trainable: bool = True,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least three")
        if not 0.0 < minimum_width_seconds < maximum_width_seconds:
            raise ValueError("invalid temporal-width bounds")
        if pool_samples < 1:
            raise ValueError("pool_samples must be positive")
        nyquist = sampling_rate_hz / 2.0
        if maximum_hz >= nyquist:
            raise ValueError("maximum_hz must be below Nyquist")
        if initial_cycles <= 0 or epsilon <= 0:
            raise ValueError("initial_cycles and epsilon must be positive")

        centers = label_independent_morlet_centers(
            atom_count, minimum_hz, maximum_hz
        )
        center_fraction = (centers - minimum_hz) / (maximum_hz - minimum_hz)
        center_fraction = center_fraction.clamp(1.0e-5, 1.0 - 1.0e-5)
        center_logits = torch.logit(center_fraction)
        widths = initial_cycles / (2.0 * torch.pi * centers)
        widths = widths.clamp(minimum_width_seconds, maximum_width_seconds)
        width_fraction = (widths - minimum_width_seconds) / (
            maximum_width_seconds - minimum_width_seconds
        )
        width_fraction = width_fraction.clamp(1.0e-5, 1.0 - 1.0e-5)

        self.center_logits = nn.Parameter(center_logits, requires_grad=trainable)
        self.width_logits = nn.Parameter(
            torch.logit(width_fraction), requires_grad=trainable
        )
        time = (
            torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        ) / sampling_rate_hz
        self.register_buffer("time_seconds", time)
        self.atom_count = int(atom_count)
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.kernel_size = int(kernel_size)
        self.minimum_hz = float(minimum_hz)
        self.maximum_hz = float(maximum_hz)
        self.minimum_width_seconds = float(minimum_width_seconds)
        self.maximum_width_seconds = float(maximum_width_seconds)
        self.pool_samples = int(pool_samples)
        self.epsilon = float(epsilon)

    def center_frequencies_hz(self) -> torch.Tensor:
        return self.minimum_hz + (self.maximum_hz - self.minimum_hz) * torch.sigmoid(
            self.center_logits
        )

    def temporal_widths_seconds(self) -> torch.Tensor:
        return self.minimum_width_seconds + (
            self.maximum_width_seconds - self.minimum_width_seconds
        ) * torch.sigmoid(self.width_logits)

    def complex_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        time = self.time_seconds[None]
        centers = self.center_frequencies_hz()[:, None]
        widths = self.temporal_widths_seconds()[:, None]
        envelope = torch.exp(-0.5 * (time / widths).square())
        angle = 2.0 * torch.pi * centers * time
        real = envelope * torch.cos(angle)
        imaginary = envelope * torch.sin(angle)
        # Remove residual DC caused by finite support, then normalize the
        # complex atom so learned gates cannot exploit arbitrary kernel scale.
        real = real - real.mean(dim=-1, keepdim=True)
        norm = torch.sqrt(
            (real.square() + imaginary.square()).sum(dim=-1, keepdim=True)
        ).clamp_min(self.epsilon)
        return real / norm, imaginary / norm

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, component, sample); got {tuple(x.shape)}"
            )
        batch, components, samples = x.shape
        if samples < self.pool_samples:
            raise ValueError("sample dimension is shorter than one output bin")
        real_kernel, imaginary_kernel = self.complex_kernels()
        kernels = torch.cat((real_kernel, imaginary_kernel), dim=0)[:, None]
        flattened = x.reshape(batch * components, 1, samples)
        # Only past samples enter an output. The symmetric atom is therefore
        # observed with a fixed half-kernel delay, but a blocked fold cannot
        # borrow ECoG from the following validation interval.
        if self.kernel_size < 257:
            filtered = F.conv1d(
                F.pad(flattened, (self.kernel_size - 1, 0)),
                kernels,
            )
        else:
            # For the long low-frequency atoms, FFT convolution is much
            # cheaper than repeating a 1 s direct FIR dot product. conv1d is
            # cross-correlation, so flipping the atom reproduces its causal
            # left-padded result exactly.
            transform_size = 1 << (samples + self.kernel_size - 2).bit_length()
            signal_spectrum = torch.fft.rfft(flattened[:, 0], n=transform_size)
            kernel_spectrum = torch.fft.rfft(
                kernels[:, 0].flip(-1), n=transform_size
            )
            filtered = torch.fft.irfft(
                signal_spectrum[:, None] * kernel_spectrum[None],
                n=transform_size,
            )[..., :samples]
        filtered = filtered.reshape(batch, components, 2, self.atom_count, samples)
        real = filtered[:, :, 0]
        imaginary = filtered[:, :, 1]
        amplitude = torch.sqrt(real.square() + imaginary.square() + self.epsilon**2)

        def pool(values: torch.Tensor) -> torch.Tensor:
            flat = values.reshape(batch * components, self.atom_count, samples)
            pooled = F.avg_pool1d(
                flat, kernel_size=self.pool_samples, stride=self.pool_samples
            )
            return pooled.reshape(
                batch, components, self.atom_count, pooled.shape[-1]
            )

        signed = pool(real)
        log_energy = torch.log1p(pool(real.square() + imaginary.square()))
        phase_gate = torch.log1p(amplitude)
        phase_cosine = pool(phase_gate * real / amplitude)
        phase_sine = pool(phase_gate * imaginary / amplitude)
        representation = torch.stack(
            (signed, log_energy, phase_cosine, phase_sine), dim=2
        )
        signed_broadband = F.avg_pool1d(
            x,
            kernel_size=self.pool_samples,
            stride=self.pool_samples,
        )
        broadband_energy = torch.log1p(
            F.avg_pool1d(
                x.square(), kernel_size=self.pool_samples, stride=self.pool_samples
            )
        )
        broadband = torch.cat((signed_broadband, broadband_energy), dim=1)
        return representation, broadband


class GroupedTopKAtomGate(nn.Module):
    """One gate per complex atom, shared by all of its representations."""

    def __init__(self, atom_count: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= atom_count:
            raise ValueError("top_k must be between one and atom_count")
        self.logits = nn.Parameter(torch.zeros(atom_count))
        self.atom_count = int(atom_count)
        self.top_k = int(top_k)
        self.register_buffer("temperature", torch.tensor(1.0), persistent=False)
        self.register_buffer("hard_fraction", torch.tensor(0.0), persistent=False)

    def set_schedule(self, temperature: float, hard_fraction: float) -> None:
        if temperature <= 0 or not 0.0 <= hard_fraction <= 1.0:
            raise ValueError("invalid Top-K schedule")
        self.temperature.fill_(temperature)
        self.hard_fraction.fill_(hard_fraction)

    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.logits / self.temperature, dim=0)

    def selected_indices(self) -> torch.Tensor:
        return torch.topk(self.logits.detach(), self.top_k).indices.sort().values

    def forward(self) -> torch.Tensor:
        soft = self.atom_count * self.probabilities()
        hard = torch.zeros_like(soft)
        hard.scatter_(0, self.selected_indices(), self.atom_count / self.top_k)
        straight_through = hard + soft - soft.detach()
        return torch.lerp(soft, straight_through, self.hard_fraction)


class ContinuousMorletDecoder(nn.Module):
    """Efficient contiguous-sequence decoder with an explicit broadband skip."""

    def __init__(
        self,
        input_components: int,
        input_channels: int | None = None,
        atom_count: int = 16,
        top_k: int | None = None,
        trainable_atoms: bool = True,
        kernel_size: int = 1001,
        convolution_units: int = 8,
        kernel_sizes: Sequence[tuple[int, int]] = ((1, 3), (3, 5), (5, 9)),
        feature_width: int = 64,
        hidden_size: int = 16,
        output_activation: str = "softplus",
        softplus_beta: float = 10.0,
    ) -> None:
        super().__init__()
        if input_components < 1:
            raise ValueError("input_components must be positive")
        if input_channels is not None and input_channels < input_components:
            raise ValueError("input_channels must be at least input_components")
        if output_activation not in {"linear", "relu", "softplus"}:
            raise ValueError("unsupported output activation")
        self.morlet = MorletAtomBank(
            atom_count=atom_count,
            kernel_size=kernel_size,
            trainable=trainable_atoms,
        )
        self.spatial_projection: nn.Module
        if input_channels is None:
            self.spatial_projection = nn.Identity()
        else:
            self.spatial_projection = nn.Conv1d(
                input_channels, input_components, kernel_size=1, bias=False
            )
            nn.init.orthogonal_(self.spatial_projection.weight[:, :, 0])
        self.atom_gate = (
            None if top_k is None else GroupedTopKAtomGate(atom_count, top_k)
        )
        self.spectrotemporal = SoftScaleSpectroTemporalConv2d(
            input_planes=input_components * len(self.morlet.representation_names),
            output_units=convolution_units,
            kernel_sizes=kernel_sizes,
            include_frequency_coordinates=True,
            causal_time=True,
            activation="silu",
            normalization="none",
        )
        self.feature_projection = nn.Linear(
            convolution_units * atom_count, feature_width
        )
        self.broadband_projection = nn.Linear(
            2 * input_components, feature_width, bias=False
        )
        self.recurrent = nn.LSTM(feature_width, hidden_size, batch_first=True)
        self.direct_output = nn.Linear(feature_width, 1)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.direct_output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.direct_output.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.input_components = int(input_components)
        self.atom_count = int(atom_count)
        self.output_activation = output_activation
        self.softplus_beta = float(softplus_beta)

    def set_gate_schedule(self, temperature: float, hard_fraction: float) -> None:
        if self.atom_gate is not None:
            self.atom_gate.set_schedule(temperature, hard_fraction)

    def compose_features(
        self,
        representation: torch.Tensor,
        broadband: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the trainable motif head to cached or live Morlet maps."""
        if representation.ndim != 5 or broadband.ndim != 3:
            raise ValueError("invalid Morlet representation or broadband shape")
        if self.atom_gate is not None:
            gate = self.atom_gate().reshape(1, 1, 1, self.atom_count, 1)
            representation = representation * gate
        batch, components, representations, atoms, time_bins = representation.shape
        maps = representation.reshape(
            batch, components * representations, atoms, time_bins
        )
        motifs = self.spectrotemporal(maps).permute(0, 3, 1, 2)
        features = self.feature_projection(motifs.flatten(start_dim=2))
        residual = self.broadband_projection(broadband.transpose(1, 2))
        return F.silu(features + residual)

    def extract(self, spatial_ecog: torch.Tensor) -> torch.Tensor:
        spatial_ecog = self.spatial_projection(spatial_ecog)
        representation, broadband = self.morlet(spatial_ecog)
        return self.compose_features(representation, broadband)

    def forward_from_representation(
        self,
        representation: torch.Tensor,
        broadband: torch.Tensor,
    ) -> torch.Tensor:
        """Decode cached fixed-filter maps without reconstructing them."""
        return self.decode(self.compose_features(representation, broadband))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.recurrent(features)
        prediction = self.direct_output(features) + self.output(recurrent)
        if self.output_activation == "relu":
            prediction = torch.relu(prediction)
        elif self.output_activation == "softplus":
            prediction = F.softplus(prediction, beta=self.softplus_beta)
        return prediction.squeeze(-1)

    def forward(self, spatial_ecog: torch.Tensor) -> torch.Tensor:
        return self.decode(self.extract(spatial_ecog))
