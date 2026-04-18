"""MNIST CSV loader and dataset-path resolution."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .config import DATA_DIR


def get_data_path(small_dataset: bool, train: bool) -> str:
    """Return the filesystem path to one of the four shipped CSVs.

    ``small_dataset=True`` selects the 100/10-sample pair; otherwise the
    full 60K/10K pair. Raises :class:`FileNotFoundError` if the file isn't
    on disk under :data:`mnist_ann.config.DATA_DIR`.
    """
    if train:
        filename = "mnist_train_100.csv" if small_dataset else "mnist_train.csv"
    else:
        filename = "mnist_test_10.csv" if small_dataset else "mnist_test.csv"

    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {filename}")
    return path


def load_mnist_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST data from a CSV file.

    Auto-detects whether the first row is a header by trying to parse its
    first column as a number.

    Returns:
        X: Image data normalized to [0, 1], shape ``(784, n_samples)``
        labels: Integer labels, shape ``(n_samples,)``
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()

    has_header = False
    try:
        float(first_line.split(",")[0])
    except ValueError:
        has_header = True

    data = np.genfromtxt(filepath, delimiter=",", skip_header=1 if has_header else 0)

    # Drop any NaN rows from malformed lines.
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]

    labels = data[:, 0].astype(int)
    X = data[:, 1:].T / 255.0
    return X.astype(np.float32), labels
