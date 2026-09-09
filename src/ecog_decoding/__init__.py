"""Reproducible ECoG finger-trajectory decoding."""

from .io import BAD_CHANNELS_ONE_BASED, SubjectData, load_subject
from .models import (
    AsymmetricWaveletPacketEnergy,
    CSPBandCorrectionEnergy,
    CSPSpatialProjection,
    DilatedWaveletFilterBank,
    WaveletPacketEnergy,
    compact_wavelet_taps,
    fit_fastica_spatial_weights,
    fixed_length_wavelet_taps,
)
from .preprocessing import (
    BaselineResult,
    EcogResult,
    MovementCorrectionModel,
    MovementCorrectionResult,
    apply_movement_corrector,
    asymmetric_baseline,
    baseline_correct,
    downsample_glove,
    fit_movement_corrector,
    local_baseline_correct,
    local_lower_envelope_baseline,
    preprocess_ecog,
)

__all__ = [
    "AsymmetricWaveletPacketEnergy",
    "BAD_CHANNELS_ONE_BASED",
    "BaselineResult",
    "CSPBandCorrectionEnergy",
    "CSPSpatialProjection",
    "DilatedWaveletFilterBank",
    "EcogResult",
    "MovementCorrectionModel",
    "MovementCorrectionResult",
    "SubjectData",
    "WaveletPacketEnergy",
    "apply_movement_corrector",
    "asymmetric_baseline",
    "baseline_correct",
    "compact_wavelet_taps",
    "downsample_glove",
    "fit_fastica_spatial_weights",
    "fit_movement_corrector",
    "fixed_length_wavelet_taps",
    "load_subject",
    "local_baseline_correct",
    "local_lower_envelope_baseline",
    "preprocess_ecog",
]
