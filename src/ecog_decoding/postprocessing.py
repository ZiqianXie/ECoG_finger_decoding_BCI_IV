"""Output-domain constraints for decoded finger trajectories."""

from __future__ import annotations

import numpy as np


def project_nonnegative(prediction: np.ndarray) -> np.ndarray:
    """Project baseline-corrected finger flexion onto its physical domain.

    The operation is deterministic and parameter-free.  Model outputs remain
    available separately when an unconstrained diagnostic is required.
    """
    values = np.asarray(prediction)
    if not np.isfinite(values).all():
        raise ValueError("prediction contains non-finite values")
    return np.maximum(values, 0.0)
