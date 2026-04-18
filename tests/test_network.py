"""Minimal sanity tests for the NeuralNetwork model.

These exercise the pure-Python math (softmax batching, one-hot encoding,
forward/backward shape invariants, and cancellation plumbing) without
touching the filesystem or running a full training pass.
"""

from __future__ import annotations

import numpy as np

from mnist_ann.network import NeuralNetwork


def _nn(**kwargs):
    defaults = dict(
        hidden_layers=(32, 16, 8),
        learning_rate=0.01,
        digits_to_classify=(0, 1, 2, 3),
        use_gpu=False,
    )
    defaults.update(kwargs)
    return NeuralNetwork(**defaults)


class TestSigmoidDerivative:
    """Sigmoid derivative stays finite across the full float32 range."""

    def test_bounded_values(self):
        # Canonical max of sigmoid' is at x=0 where it equals 0.25.
        nn = _nn()
        x = np.array([[-1000.0, -100.0, 0.0, 100.0, 1000.0]], dtype=np.float32)
        d = nn.sigmoid_derivative(x)
        assert np.all(np.isfinite(d)), "sigmoid' produced NaN/inf"
        assert d.max() <= 0.25 + 1e-6
        assert d.min() >= 0.0

    def test_matches_sigmoid_identity(self):
        # sigmoid'(x) == sigmoid(x) * (1 - sigmoid(x)), exactly.
        nn = _nn()
        x = np.linspace(-10.0, 10.0, 50, dtype=np.float32).reshape(1, -1)
        d = nn.sigmoid_derivative(x)
        s = nn.sigmoid(x)
        assert np.allclose(d, s * (1.0 - s), atol=1e-6)


class TestSoftmax:
    """Column-wise softmax: each column of the output must sum to 1."""

    def test_column_sums_to_one_for_single_column(self):
        nn = _nn()
        x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        y = nn.softmax(x)
        assert y.shape == (5, 1)
        assert abs(y.sum() - 1.0) < 1e-5

    def test_each_column_sums_to_one_for_batch(self):
        # This is the bug the review fixed: softmax must reduce per-column,
        # not over the whole tensor.
        nn = _nn()
        x = np.random.randn(5, 4).astype(np.float32)
        y = nn.softmax(x)
        assert y.shape == (5, 4)
        col_sums = y.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-5)


class TestOneHotEncode:
    """Digit -> class-index encoding; everything else lands in 'None'."""

    def test_known_digit_maps_to_its_class(self):
        nn = _nn(digits_to_classify=(0, 1, 2, 3))
        labels = np.array([0, 1, 2, 3])
        Y = nn.one_hot_encode(labels)
        assert Y.shape == (5, 4)
        # Each sample lights up exactly one class
        assert np.all(Y.sum(axis=0) == 1.0)
        # Diagonal assignment: 0->row0, 1->row1, ...
        assert Y[0, 0] == 1.0
        assert Y[1, 1] == 1.0
        assert Y[2, 2] == 1.0
        assert Y[3, 3] == 1.0

    def test_unknown_digit_lands_in_none_class(self):
        nn = _nn(digits_to_classify=(0, 1, 2, 3))
        labels = np.array([5, 7, 9])
        Y = nn.one_hot_encode(labels)
        # All three samples should be in row 4 ("None")
        assert np.all(Y[4, :] == 1.0)
        assert Y[:4, :].sum() == 0.0


class TestForwardShapes:
    """``forward`` works for both single columns and batched input."""

    def test_forward_accepts_column_vector_input(self):
        nn = _nn(hidden_layers=(32, 16, 8))
        x = np.random.rand(784, 1).astype(np.float32)
        y, cache = nn.forward(x)
        assert y.shape == (5, 1)
        assert cache["a2"].shape == (32, 1)
        assert cache["a3"].shape == (16, 1)
        assert cache["a4"].shape == (8, 1)

    def test_forward_accepts_batch_input(self):
        nn = _nn(hidden_layers=(32, 16, 8))
        X = np.random.rand(784, 7).astype(np.float32)
        Y, _ = nn.forward(X)
        assert Y.shape == (5, 7)
        # Column-wise softmax: each column sums to 1.
        assert np.allclose(Y.sum(axis=0), 1.0, atol=1e-5)


class TestTrainingCancellation:
    """``should_cancel`` predicate plumbing through the training loop."""

    def _tiny_data(self, n=10):
        X = np.random.rand(784, n).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 5, 5])[:n]
        return X, labels

    def test_immediate_cancel_exits_before_any_epoch(self):
        nn = _nn()
        X, labels = self._tiny_data()
        Y = nn.one_hot_encode(labels)
        history = nn.train(
            X, Y,
            n_epochs=50,
            show_progress_bar=False,
            should_cancel=lambda: True,
        )
        assert history["cancelled"] is True
        assert history["loss_history"] == []

    def test_no_cancel_runs_to_completion(self):
        nn = _nn()
        X, labels = self._tiny_data()
        Y = nn.one_hot_encode(labels)
        history = nn.train(
            X, Y,
            n_epochs=3,
            show_progress_bar=False,
            should_cancel=lambda: False,
        )
        assert history["cancelled"] is False
        assert history["diverged"] is False
        assert len(history["loss_history"]) == 3


class TestUpdateWeightsShapes:
    """``update_weights`` must preserve bias shapes for any batch size."""

    def test_single_sample_preserves_bias_shape(self):
        nn = _nn(hidden_layers=(8, 8, 8))
        X = np.random.rand(784, 1).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0]))
        _, cache = nn.forward(X)
        grads = nn.backward_cb(Y, cache)
        nn.update_weights(grads)
        assert nn.b2.shape == (8, 1)
        assert nn.b3.shape == (8, 1)
        assert nn.b4.shape == (8, 1)
        assert nn.b5.shape == (5, 1)

    def test_batched_input_does_not_corrupt_bias_shape(self):
        # Pre-fix, biases would broadcast to (n, batch_size).
        nn = _nn(hidden_layers=(8, 8, 8))
        X = np.random.rand(784, 4).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3]))
        _, cache = nn.forward(X)
        grads = nn.backward_cb(Y, cache)
        nn.update_weights(grads)
        assert nn.b2.shape == (8, 1)
        assert nn.b3.shape == (8, 1)
        assert nn.b4.shape == (8, 1)
        assert nn.b5.shape == (5, 1)

    def test_batched_uhb_also_preserves_bias_shape(self):
        nn = _nn(hidden_layers=(8, 8, 8))
        X = np.random.rand(784, 4).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3]))
        _, cache = nn.forward(X)
        grads = nn.backward_uhb(Y, cache)
        nn.update_weights(grads)
        assert nn.b2.shape == (8, 1)
        assert nn.b5.shape == (5, 1)


class TestTrainingConverges:
    """A few epochs on well-conditioned toy data should reduce the loss.

    Guards against silent regressions in the forward/backward/update
    wiring; the shape tests pass if every array is the right shape but
    the gradients point the wrong way.
    """

    def test_loss_strictly_decreases_on_toy_data(self):
        np.random.seed(0)
        nn = _nn(hidden_layers=(16, 16, 8), learning_rate=0.1)
        # 16 samples, one per class (4 known + 'None' filler), repeated.
        labels = np.array([0, 1, 2, 3, 5, 0, 1, 2, 3, 5, 0, 1, 2, 3, 5, 0])
        X = np.random.rand(784, labels.size).astype(np.float32)
        Y = nn.one_hot_encode(labels)
        history = nn.train(X, Y, n_epochs=30, show_progress_bar=False)
        losses = history["loss_history"]
        assert len(losses) == 30
        assert losses[-1] < losses[0], (
            f"loss did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"
        )


class TestDivergenceGuard:
    """Training bails out early if weights drive the loss to NaN/inf."""

    def test_normal_training_is_not_flagged_diverged(self):
        nn = NeuralNetwork(
            hidden_layers=(8, 8, 8),
            learning_rate=0.01,
            digits_to_classify=(0, 1, 2, 3),
            use_gpu=False,
        )
        X = np.random.rand(784, 5).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3, 5]))
        history = nn.train(X, Y, n_epochs=2, show_progress_bar=False)
        assert history["diverged"] is False

    def test_nan_weights_trigger_diverged_flag(self):
        # Inject NaN weights and verify the guard catches it at the first
        # reporting checkpoint.
        nn = NeuralNetwork(
            hidden_layers=(8, 8, 8),
            learning_rate=0.01,
            digits_to_classify=(0, 1, 2, 3),
            use_gpu=False,
        )
        nn.W2 = np.full_like(nn.W2, np.nan)
        X = np.random.rand(784, 150).astype(np.float32)
        Y = nn.one_hot_encode(np.array([0, 1, 2, 3] * 37 + [5, 5]))
        callback_epochs = []
        history = nn.train(
            X, Y,
            n_epochs=3,
            show_progress_bar=False,
            progress_callback=lambda info: callback_epochs.append(info["epoch"]),
        )
        assert history["diverged"] is True
        # Must not have completed a full epoch with NaN weights.
        assert len(history["loss_history"]) == 0
