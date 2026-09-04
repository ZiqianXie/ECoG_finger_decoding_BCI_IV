"""Loading and validation for BCI Competition IV Data Set 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


BAD_CHANNELS_ONE_BASED: dict[int, tuple[int, ...]] = {
    1: (55,),
    2: (21, 38),
    # The paper reports channel 49 for subject 3.  A raw-data audit also found
    # a test-only, >250x variance burst on physical channel 50.  Keep both out:
    # retaining channel 50 destroys held-out features even though it looks
    # normal in the training recording.
    3: (49, 50),
}


@dataclass(frozen=True)
class SubjectData:
    subject: int
    train_ecog: np.ndarray
    test_ecog: np.ndarray
    train_glove: np.ndarray
    test_glove: np.ndarray


def load_subject(data_root: str | Path, subject: int) -> SubjectData:
    """Load one subject and enforce the public dataset contract."""
    if subject not in (1, 2, 3):
        raise ValueError(f"subject must be 1, 2, or 3; got {subject}")

    root = Path(data_root)
    comp_path = root / "mat" / f"sub{subject}_comp.mat"
    labels_path = root / "true_labels" / f"sub{subject}_testlabels.mat"
    if not comp_path.is_file():
        raise FileNotFoundError(comp_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    comp = loadmat(
        comp_path,
        variable_names=("train_data", "test_data", "train_dg"),
    )
    labels = loadmat(labels_path, variable_names=("test_dg",))

    train_ecog = np.asarray(comp["train_data"], dtype=np.float64)
    test_ecog = np.asarray(comp["test_data"], dtype=np.float64)
    train_glove = np.asarray(comp["train_dg"], dtype=np.float64)
    test_glove = np.asarray(labels["test_dg"], dtype=np.float64)

    expected_channels = {1: 62, 2: 48, 3: 64}[subject]
    expected = {
        "train_ecog": (400_000, expected_channels),
        "test_ecog": (200_000, expected_channels),
        "train_glove": (400_000, 5),
        "test_glove": (200_000, 5),
    }
    actual = {
        "train_ecog": train_ecog.shape,
        "test_ecog": test_ecog.shape,
        "train_glove": train_glove.shape,
        "test_glove": test_glove.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} has shape {actual[name]}, expected {shape}")

    return SubjectData(
        subject=subject,
        train_ecog=train_ecog,
        test_ecog=test_ecog,
        train_glove=train_glove,
        test_glove=test_glove,
    )
