"""MNIST CSV loader and dataset-path resolution.

Typical usage example::

    from mnist_ann.data import get_data_path, load_mnist_data

    path = get_data_path(small_dataset=True, train=True)  # 100-sample CSV
    X, labels = load_mnist_data(path)
    # X.shape == (784, 100); labels.shape == (100,)
"""

from __future__ import annotations

import os

import numpy as np

from .config import DATA_DIR


def get_data_path(small_dataset: bool, train: bool) -> str:
    """Return the filesystem path to one of the four shipped CSVs.

    ``small_dataset=True`` selects the 100/10-sample pair; otherwise the
    full 60K/10K pair. Existence is not pre-checked here -- let the actual
    read in :func:`load_mnist_data` fail with its own ``FileNotFoundError``
    so we don't TOCTOU between this function and the open().
    """
    if train:
        filename = "mnist_train_100.csv" if small_dataset else "mnist_train.csv"
    else:
        filename = "mnist_test_10.csv" if small_dataset else "mnist_test.csv"

    return os.path.join(DATA_DIR, filename)


def load_mnist_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST data from a CSV file.

    Auto-detects whether the first row is a header by trying to parse its
    first column as a number.

    Args:
        filepath: Absolute or relative path to an MNIST-style CSV (first
            column = label, remaining 784 columns = pixel intensities in
            ``[0, 255]``).

    Returns:
        A ``(X, labels)`` tuple:

          - ``X`` is a ``float32`` ``(784, n_samples)`` array, normalised
            into ``[0, 1]``.
          - ``labels`` is an ``int`` ``(n_samples,)`` array.

    Raises:
        FileNotFoundError: If ``filepath`` doesn't exist on disk.
        OSError: For other I/O failures (permissions, unreadable device).
        ValueError: If the CSV is malformed beyond the NaN-row filter's
            tolerance (e.g. wrong column count on every row).
    """
    with open(filepath) as f:
        first_line = f.readline().strip()

    has_header = False
    try:
        float(first_line.split(",")[0])
    except ValueError:
        has_header = True

    # ``dtype=np.float32`` upfront avoids a full float64 pass then cast.
    # Halves peak memory on the 97 MB training CSV.
    data = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1 if has_header else 0,
        dtype=np.float32,
    )

    # Drop any NaN rows from malformed lines.
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]

    labels = data[:, 0].astype(int)
    X = (data[:, 1:].T / np.float32(255.0)).astype(np.float32, copy=False)
    return X, labels
