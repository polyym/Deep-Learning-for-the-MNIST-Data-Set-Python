"""Exercise the GPU-specific branches without requiring a real CUDA device.

CI runners don't have GPUs, but the code paths that only fire when
``self.use_gpu`` is True (``xp.asarray`` conversions, ``to_cpu``,
per-batch index transfer) can still be driven with numpy standing in
for CuPy. This guards against regressions that would only surface on a
user's actual GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from mnist_ann import backend, network
from mnist_ann.network import NeuralNetwork


@pytest.fixture
def fake_gpu(monkeypatch):
    """Flip ``GPU_AVAILABLE`` on while keeping ``_xp`` as numpy.

    Lets the GPU code paths execute (asarray conversions, to_cpu, etc.)
    without needing CuPy installed. ``to_cpu`` no-ops on plain ndarrays
    because they don't expose ``.get()``.
    """
    monkeypatch.setattr(backend, "GPU_AVAILABLE", True)
    monkeypatch.setattr(backend, "_xp", np)
    # Module-level ``GPU_AVAILABLE`` was imported by reference into network.py;
    # patch that bound name too, or the constructor sees the old False value.
    monkeypatch.setattr(network, "GPU_AVAILABLE", True)
    monkeypatch.setattr(network, "_xp", np)
    yield


class TestGpuPathExecutes:
    """Smoke tests: GPU-flagged branches must not crash with numpy backend."""

    def test_constructor_picks_gpu_backend(self, fake_gpu):
        nn = NeuralNetwork(hidden_layers=(8, 8, 8), use_gpu=True)
        assert nn.use_gpu is True
        assert nn.xp is np  # fake: _xp is numpy

    def test_forward_and_predict_on_gpu_path(self, fake_gpu):
        nn = NeuralNetwork(hidden_layers=(8, 8, 8), use_gpu=True)
        X = np.random.rand(784, 3).astype(np.float32)
        classes, probs = nn.predict(X)
        assert classes.shape == (3,)
        assert probs.shape == (5, 3)

    def test_mini_batch_training_on_gpu_path(self, fake_gpu):
        """Exercises ``xp.asarray(batch_slice)`` per batch + loss sync."""
        nn = NeuralNetwork(hidden_layers=(8, 8, 8), learning_rate=0.05, use_gpu=True)
        X = np.random.rand(784, 16).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3, 5] * 3 + [0]))
        history = nn.train(X, Y, n_epochs=2, batch_size=4, show_progress_bar=False)
        assert history["cancelled"] is False
        assert history["diverged"] is False
        assert len(history["loss_history"]) == 2

    def test_evaluate_on_gpu_path(self, fake_gpu):
        nn = NeuralNetwork(hidden_layers=(8, 8, 8), use_gpu=True)
        X = np.random.rand(784, 4).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3]))
        results = nn.evaluate(X, Y)
        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert len(results["predictions"]) == 5  # one list per class
