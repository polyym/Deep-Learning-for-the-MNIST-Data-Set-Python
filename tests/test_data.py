"""Tests for the CSV loader and dataset-path resolver."""

from __future__ import annotations

import numpy as np
import pytest

from mnist_ann.data import get_data_path, load_mnist_data


class TestGetDataPath:
    """Resolver for the four shipped CSVs under ``DATA_DIR``."""

    def test_small_train_path_exists(self):
        path = get_data_path(small_dataset=True, train=True)
        assert path.endswith("mnist_train_100.csv")

    def test_small_test_path_exists(self):
        path = get_data_path(small_dataset=True, train=False)
        assert path.endswith("mnist_test_10.csv")

    def test_full_train_path_exists(self):
        path = get_data_path(small_dataset=False, train=True)
        assert path.endswith("mnist_train.csv")

    def test_full_test_path_exists(self):
        path = get_data_path(small_dataset=False, train=False)
        assert path.endswith("mnist_test.csv")

    def test_missing_file_raises(self, monkeypatch, tmp_path):
        # Point DATA_DIR at an empty directory so no CSV exists.
        monkeypatch.setattr("mnist_ann.data.DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            get_data_path(small_dataset=True, train=True)


class TestLoadMnistData:
    """CSV parsing, normalisation, and header autodetection."""

    def test_shape_and_dtype(self):
        path = get_data_path(small_dataset=True, train=False)  # 10 samples
        X, labels = load_mnist_data(path)
        assert X.shape == (784, 10)
        assert X.dtype == np.float32
        assert labels.shape == (10,)
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_values_normalised_to_unit_range(self):
        path = get_data_path(small_dataset=True, train=False)
        X, _ = load_mnist_data(path)
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_labels_are_valid_digits(self):
        path = get_data_path(small_dataset=True, train=False)
        _, labels = load_mnist_data(path)
        assert labels.min() >= 0
        assert labels.max() <= 9

    def test_autodetects_header(self, tmp_path):
        # Write a tiny CSV with a text header
        p = tmp_path / "with_header.csv"
        pixel_cols = ",".join([f"p{i}" for i in range(784)])
        p.write_text(
            f"label,{pixel_cols}\n"
            + "3," + ",".join(["128"] * 784) + "\n"
            + "7," + ",".join(["200"] * 784) + "\n"
        )
        X, labels = load_mnist_data(str(p))
        assert X.shape == (784, 2)
        assert list(labels) == [3, 7]

    def test_autodetects_no_header(self, tmp_path):
        p = tmp_path / "no_header.csv"
        p.write_text(
            "5," + ",".join(["100"] * 784) + "\n"
            + "1," + ",".join(["50"] * 784) + "\n"
        )
        X, labels = load_mnist_data(str(p))
        assert X.shape == (784, 2)
        assert list(labels) == [5, 1]
