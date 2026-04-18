"""Feedforward neural network for MNIST digit classification.

Mirrors my own MATLAB implementation (``ann1923114.m``):

- 784 input units (28x28 images)
- 3 sigmoid hidden layers (user-configurable widths)
- 5 softmax output units (4 selected digits + "None")
- Two backprop variants: Calculus-Based (CB) and Unscaled Heuristic (UHB)
- Optional CuPy/GPU acceleration via :mod:`.backend`

Typical usage example::

    from mnist_ann import NeuralNetwork, load_mnist_data

    X_train, labels = load_mnist_data("data/mnist_train_100.csv")
    nn = NeuralNetwork(hidden_layers=(64, 32, 16), learning_rate=0.01)
    Y_train = nn.one_hot_encode(labels)
    nn.train(X_train, Y_train, n_epochs=50, batch_size=32)
    pred_classes, probs = nn.predict(X_train[:, :5])
"""

from __future__ import annotations

from collections.abc import Callable

import math

import numpy as np

from .backend import GPU_AVAILABLE, _xp, to_cpu
from .progress import ProgressBar


# Clip floor for log-softmax: big enough to dodge log(0) but well above float32's
# smallest subnormal so no rounding surprises. log(1e-15) ≈ -34.5, which is a
# finite, sane loss contribution for a saturated prediction.
_LOSS_EPSILON = 1e-15


def _is_finite(x: float) -> bool:
    """True iff ``x`` is a finite float (not NaN, not +/-inf)."""
    return math.isfinite(x)


class NeuralNetwork:
    """Feedforward ANN for MNIST digit classification.

    Supports CPU (NumPy) and GPU (CuPy) backends. The backend is selected
    at construction time based on ``use_gpu`` and the presence of a working
    CuPy install; otherwise NumPy is used.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, int, int] = (64, 32, 16),
        learning_rate: float = 0.01,
        digits_to_classify: tuple[int, int, int, int] = (0, 1, 2, 3),
        use_gpu: bool = True,
    ):
        """
        Args:
            hidden_layers: ``(U, V, W)`` neuron counts per hidden layer.
            learning_rate: SGD step size.
            digits_to_classify: Which four MNIST digits to recognise; anything
                else is classified as "None".
            use_gpu: Use CuPy if available. Silently falls back to CPU.
        """
        self.U, self.V, self.W = hidden_layers
        self.lr = learning_rate
        self.A, self.B, self.C, self.D = digits_to_classify

        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = _xp if self.use_gpu else np

        self.Ni = 784
        self.No = 5

        self._initialize_weights()

        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []

    # ------------------------------------------------------------------
    # Initialisation and activations
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Weights drawn uniformly from (-0.5, 0.5]; biases zeroed."""
        xp = self.xp

        self.W2 = 0.5 - xp.random.rand(self.Ni, self.U).astype(xp.float32)
        self.W3 = 0.5 - xp.random.rand(self.U, self.V).astype(xp.float32)
        self.W4 = 0.5 - xp.random.rand(self.V, self.W).astype(xp.float32)
        self.W5 = 0.5 - xp.random.rand(self.W, self.No).astype(xp.float32)

        self.b2 = xp.zeros((self.U, 1), dtype=xp.float32)
        self.b3 = xp.zeros((self.V, 1), dtype=xp.float32)
        self.b4 = xp.zeros((self.W, 1), dtype=xp.float32)
        self.b5 = xp.zeros((self.No, 1), dtype=xp.float32)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Elementwise logistic sigmoid, clipped to avoid overflow in ``exp``.

        The clip range ``[-80, 80]`` keeps ``exp(-x)`` inside float32's
        representable range (``exp(88.7) ~= float32 max``); past that, sigmoid
        is already saturated to 0 or 1 within float32 resolution anyway.
        """
        xp = self.xp
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -80.0, 80.0)))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Elementwise derivative of :meth:`sigmoid` at ``x`` (pre-activation).

        Uses the algebraic identity ``sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))``
        rather than the raw ``exp(-x) / (1 + exp(-x))**2`` form, which overflows
        in float32 for ``x <= -88`` (``exp(88) > float32 max``) and produces
        ``inf / inf = NaN``. The identity form is bounded in ``[0, 0.25]`` for
        any finite input, which matters most for the UHB backprop path where
        saturated weights would otherwise feed NaN back into the gradients.
        """
        s = self.sigmoid(x)
        return s * (1.0 - s)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax, computed column-wise."""
        xp = self.xp
        exp_x = xp.exp(x - xp.max(x, axis=0, keepdims=True))
        return exp_x / xp.sum(exp_x, axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    def one_hot_encode(self, labels: np.ndarray) -> np.ndarray:
        """One-hot encode labels into 5-class targets.

        Classes 0-3 map to the four configured digits (in order, see
        ``digits_to_classify`` in :meth:`__init__`); class 4 is "None" for
        any label that isn't one of those four.

        Args:
            labels: 1-D integer array of MNIST digit labels (0-9), or any
                array-like coercible via :func:`numpy.asarray`. GPU arrays
                are fine; they're moved to host for the encoding pass.

        Returns:
            ``(5, n_samples)`` one-hot float32 matrix on the active backend
            (GPU if ``use_gpu=True``, else CPU). Each column has exactly
            one ``1.0``.
        """
        xp = self.xp
        labels_cpu = np.asarray(to_cpu(labels))
        n_samples = labels_cpu.shape[0]
        Y = np.zeros((5, n_samples), dtype=np.float32)

        known = np.zeros(n_samples, dtype=bool)
        for cls_idx, digit in enumerate((self.A, self.B, self.C, self.D)):
            mask = labels_cpu == digit
            Y[cls_idx, mask] = 1.0
            known |= mask
        Y[4, ~known] = 1.0

        return xp.asarray(Y) if self.use_gpu else Y

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward propagate ``x`` through the network.

        Args:
            x: Input, either a single column vector ``(784,)`` / ``(784, 1)``
                or a batch ``(784, N)``. 1-D input is reshaped to
                ``(784, 1)`` before propagation; cached activations are
                sized to match the input's batch dimension.

        Returns:
            Tuple of ``(a5, cache)``:

              - ``a5``: ``(5, N)`` softmax output.
              - ``cache``: per-layer pre-activations ``n2/n3/n4/n5`` and
                post-activations ``a1/a2/a3/a4/a5``; consumed by the
                backward pass.
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        a1 = x

        n2 = self.W2.T @ a1 + self.b2
        a2 = self.sigmoid(n2)

        n3 = self.W3.T @ a2 + self.b3
        a3 = self.sigmoid(n3)

        n4 = self.W4.T @ a3 + self.b4
        a4 = self.sigmoid(n4)

        n5 = self.W5.T @ a4 + self.b5
        a5 = self.softmax(n5)

        cache = {
            "a1": a1,
            "n2": n2, "a2": a2,
            "n3": n3, "a3": a3,
            "n4": n4, "a4": a4,
            "n5": n5, "a5": a5,
        }
        return a5, cache

    def backward_uhb(self, y_true: np.ndarray, cache: dict) -> dict:
        """Unscaled Heuristic Backpropagation.

        Equivalent to ``diag(sigmoid'(n)) @ W @ S`` but uses an elementwise
        product -- O(n) instead of an O(n^2) dense matmul.
        """
        a1 = cache["a1"]
        a2, a3, a4, a5 = cache["a2"], cache["a3"], cache["a4"], cache["a5"]
        n2, n3, n4 = cache["n2"], cache["n3"], cache["n4"]

        d2 = self.sigmoid_derivative(n2)
        d3 = self.sigmoid_derivative(n3)
        d4 = self.sigmoid_derivative(n4)

        e5 = y_true - a5
        S5 = -e5

        e4 = self.W5 @ e5
        S4 = -2.0 * d4 * e4

        e3 = self.W4 @ e4
        S3 = -2.0 * d3 * e3

        e2 = self.W3 @ e3
        S2 = -2.0 * d2 * e2

        return {"S2": S2, "S3": S3, "S4": S4, "S5": S5,
                "a1": a1, "a2": a2, "a3": a3, "a4": a4}

    def backward_cb(self, y_true: np.ndarray, cache: dict) -> dict:
        """Calculus-Based Backpropagation (chain rule sensitivities)."""
        a1 = cache["a1"]
        a2, a3, a4, a5 = cache["a2"], cache["a3"], cache["a4"], cache["a5"]
        n2, n3, n4 = cache["n2"], cache["n3"], cache["n4"]

        d2 = self.sigmoid_derivative(n2)
        d3 = self.sigmoid_derivative(n3)
        d4 = self.sigmoid_derivative(n4)

        S5 = a5 - y_true
        S4 = d4 * (self.W5 @ S5)
        S3 = d3 * (self.W4 @ S4)
        S2 = d2 * (self.W3 @ S3)

        return {"S2": S2, "S3": S3, "S4": S4, "S5": S5,
                "a1": a1, "a2": a2, "a3": a3, "a4": a4}

    def update_weights(self, grads: dict) -> None:
        """Apply gradient-descent updates from a backward pass.

        Weight updates via ``a @ S.T`` naturally sum contributions across
        the batch dim. Bias updates must explicitly sum across axis=1 so
        they stay shape ``(n, 1)``; without this, batched inputs silently
        broadcast the bias to ``(n, N)`` and corrupt the network.

        Gradients are averaged over the batch (divided by ``S2.shape[1]``)
        so ``self.lr`` stays a per-sample step size independent of batch
        size. At batch size 1 the division is a no-op, preserving the
        original online-SGD semantics.

        Args:
            grads: Output of :meth:`backward_cb` or :meth:`backward_uhb`,
                i.e. a dict with keys ``S2``/``S3``/``S4``/``S5`` (layer
                sensitivities, each shape ``(n_layer, batch_size)``) and
                ``a1``/``a2``/``a3``/``a4`` (cached forward activations).
        """
        xp = self.xp
        S2, S3, S4, S5 = grads["S2"], grads["S3"], grads["S4"], grads["S5"]
        a1, a2, a3, a4 = grads["a1"], grads["a2"], grads["a3"], grads["a4"]

        batch_size = S2.shape[1]
        lr = self.lr / batch_size

        self.W5 = self.W5 - lr * (a4 @ S5.T)
        self.b5 = self.b5 - lr * xp.sum(S5, axis=1, keepdims=True)

        self.W4 = self.W4 - lr * (a3 @ S4.T)
        self.b4 = self.b4 - lr * xp.sum(S4, axis=1, keepdims=True)

        self.W3 = self.W3 - lr * (a2 @ S3.T)
        self.b3 = self.b3 - lr * xp.sum(S3, axis=1, keepdims=True)

        self.W2 = self.W2 - lr * (a1 @ S2.T)
        self.b2 = self.b2 - lr * xp.sum(S2, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Categorical cross-entropy, clipped to avoid ``log(0)``.

        Returns the **sum** over all classes and samples (not the mean), so
        training-loop scaling by ``n_samples / samples_done`` is consistent.
        """
        xp = self.xp
        y_pred = xp.clip(y_pred, _LOSS_EPSILON, 1 - _LOSS_EPSILON)
        loss = -xp.sum(y_true * xp.log(y_pred))
        return float(to_cpu(loss))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        n_epochs: int,
        batch_size: int = 32,
        backprop_method: str = "cb",
        progress_callback: Callable | None = None,
        show_progress_bar: bool = True,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict:
        """Train the network with mini-batch SGD.

        Args:
            X_train: ``(784, n_samples)`` inputs.
            Y_train: ``(5, n_samples)`` one-hot targets.
            n_epochs: Number of training epochs.
            batch_size: Samples per gradient update. ``1`` recovers online
                SGD. Clamped to ``n_samples`` for tiny datasets. Gradients
                are averaged over the batch in :meth:`update_weights`, so
                ``learning_rate`` keeps its per-sample interpretation.
            backprop_method: ``'cb'`` or ``'uhb'``.
            progress_callback: Optional callable receiving dict updates.
            show_progress_bar: Render a console progress bar.
            should_cancel: Optional zero-arg predicate. If it ever returns
                True, the loop exits at the next checkpoint and the returned
                history includes ``cancelled: True``.

        Returns:
            History dict with the keys ``loss_history``, ``accuracy_history``,
            ``cancelled`` (True iff ``should_cancel`` fired), and ``diverged``
            (True iff the running loss became NaN/inf and the loop bailed out
            early to avoid burning CPU on garbage gradients).

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        xp = self.xp

        if self.use_gpu:
            X_train = xp.asarray(X_train, dtype=xp.float32)
            Y_train = xp.asarray(Y_train, dtype=xp.float32)

        n_samples = X_train.shape[1]
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        batch_size = min(batch_size, n_samples)

        self.loss_history = []
        self.accuracy_history = []

        backward_fn = (
            self.backward_cb if backprop_method == "cb" else self.backward_uhb
        )

        progress_bar: ProgressBar | None = None
        if show_progress_bar:
            gpu_label = " [GPU]" if self.use_gpu else " [CPU]"
            progress_bar = ProgressBar(n_epochs, prefix=f"Training{gpu_label}: ")

        # Progress reports fire when cumulative samples cross a multiple of
        # ``report_interval``. Cancellation is cheap so it's checked at every
        # batch boundary.
        report_interval = min(1000, max(100, n_samples // 20))
        cancelled = False
        diverged = False

        for epoch in range(n_epochs):
            if should_cancel and should_cancel():
                cancelled = True
                break

            # CPU permutation; we slice GPU-resident matrices with a per-batch
            # ``xp.asarray`` of the slice, which is a tiny H2D transfer (one
            # int vector per batch) -- not the per-sample sync the original
            # online loop avoided.
            indices = np.random.permutation(n_samples)

            # Accumulate on-device; sync only on reporting/epoch boundaries.
            running_loss = xp.zeros((), dtype=xp.float32)
            running_correct = xp.zeros((), dtype=xp.int64)
            samples_seen = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_slice = indices[batch_start:batch_start + batch_size]
                batch_idx = xp.asarray(batch_slice) if self.use_gpu else batch_slice

                x = X_train[:, batch_idx]
                y = Y_train[:, batch_idx]

                y_pred, cache = self.forward(x)

                y_pred_clipped = xp.clip(y_pred, _LOSS_EPSILON, 1.0 - _LOSS_EPSILON)
                running_loss = running_loss - xp.sum(y * xp.log(y_pred_clipped))
                running_correct = running_correct + xp.sum(
                    xp.argmax(y_pred, axis=0) == xp.argmax(y, axis=0)
                )

                grads = backward_fn(y, cache)
                self.update_weights(grads)

                samples_prev = samples_seen
                samples_seen += len(batch_slice)

                # Fast cancel path: no sync, no callback.
                if should_cancel and should_cancel():
                    cancelled = True
                    break

                # Report + divergence guard when samples_seen crosses a
                # ``report_interval`` boundary. UHB at lr >> paper's 0.001
                # can push weights until sigmoid saturates and derivatives
                # blow up to NaN, so we stop early instead of burning CPU
                # on garbage gradients.
                crossed = (
                    samples_prev // report_interval
                    < samples_seen // report_interval
                )
                if progress_callback and crossed:
                    loss_so_far = float(to_cpu(running_loss))
                    correct_so_far = int(to_cpu(running_correct))
                    if not _is_finite(loss_so_far):
                        diverged = True
                        break
                    progress_callback({
                        "epoch": epoch + (samples_seen / n_samples),
                        "total_epochs": n_epochs,
                        "loss": loss_so_far * (n_samples / samples_seen),
                        "accuracy": 100.0 * correct_so_far / samples_seen,
                        "samples_done": samples_seen,
                        "total_samples": n_samples,
                    })

            if cancelled or diverged:
                break

            total_loss = float(to_cpu(running_loss))
            accuracy = 100.0 * int(to_cpu(running_correct)) / n_samples

            # Same divergence check at epoch boundary, in case the run blew
            # up between report intervals.
            if not _is_finite(total_loss):
                diverged = True
                break

            self.loss_history.append(total_loss)
            self.accuracy_history.append(accuracy)

            if progress_bar:
                progress_bar.update(epoch + 1, loss=total_loss, accuracy=accuracy)

            if progress_callback:
                progress_callback({
                    "epoch": epoch + 1,
                    "total_epochs": n_epochs,
                    "loss": total_loss,
                    "accuracy": accuracy,
                })

        if progress_bar:
            progress_bar.finish()

        return {
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history,
            "cancelled": cancelled,
            "diverged": diverged,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run a single batched forward pass and return (classes, probs).

        Args:
            X: Input, either ``(784,)`` for a single sample or ``(784, N)``
                for a batch. Moved onto the active backend if necessary.

        Returns:
            Tuple of ``(pred_classes, predictions)``, both returned on the
            host:

              - ``pred_classes``: ``(N,)`` argmax class indices in
                ``[0, 4]``.
              - ``predictions``: ``(5, N)`` softmax probabilities.
        """
        xp = self.xp
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions, _ = self.forward(X)
        pred_classes = xp.argmax(predictions, axis=0)
        return to_cpu(pred_classes), to_cpu(predictions)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """Score the network on ``(X, Y)`` and return metrics + confusion.

        Args:
            X: ``(784, n_samples)`` inputs.
            Y: ``(5, n_samples)`` one-hot targets (produced by
                :meth:`one_hot_encode`).

        Returns:
            Dict with:

              - ``accuracy``: Overall accuracy as a percentage.
              - ``per_class``: ``{label: {correct, total, accuracy}}``
                for each of the 5 classes.
              - ``confusion_matrix``: ``5x5`` nested list, rows = predicted
                class index, cols = true class index.
              - ``predictions``: ``(5, n_samples)`` softmax outputs as a
                nested list.
              - ``true_labels`` / ``pred_labels``: ``(n_samples,)`` class
                indices as flat lists.
        """
        xp = self.xp
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)
            Y = xp.asarray(Y, dtype=xp.float32)

        n_samples = X.shape[1]
        pred_classes, predictions = self.predict(X)

        true_classes = to_cpu(xp.argmax(Y, axis=0))
        pred_classes = to_cpu(pred_classes)
        predictions = to_cpu(predictions)

        accuracy = 100 * np.sum(pred_classes == true_classes) / n_samples

        # Vectorised 5x5 confusion: flatten (pred, true) into a single bincount
        # index so we skip the Python per-sample loop.
        flat = pred_classes * 5 + true_classes
        confusion = np.bincount(flat, minlength=25).reshape(5, 5)

        class_names = self.class_labels
        per_class = {}
        for c in range(5):
            total_c = int(np.sum(true_classes == c))
            correct_c = int(confusion[c, c])
            per_class[class_names[c]] = {
                "correct": correct_c,
                "total": total_c,
                "accuracy": 100 * correct_c / total_c if total_c > 0 else 0,
            }

        return {
            "accuracy": float(accuracy),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
            "predictions": predictions.tolist(),
            "true_labels": true_classes.tolist(),
            "pred_labels": pred_classes.tolist(),
        }

    @property
    def class_labels(self) -> tuple[str, str, str, str, str]:
        """Display labels for class indices 0..4 (four configured digits + "None")."""
        return (str(self.A), str(self.B), str(self.C), str(self.D), "None")

    def get_digit_label(self, class_idx: int) -> str:
        """Map a class index 0..4 to its human label ('0'..'9' or 'None').

        Args:
            class_idx: Network output index in ``[0, 4]``. Values outside
                that range are clamped to the "None" label rather than
                raising.

        Returns:
            The digit string (e.g. ``"3"``) for a configured class, or
            ``"None"`` for class 4 / out-of-range indices.
        """
        labels = self.class_labels
        return labels[class_idx] if 0 <= class_idx < len(labels) else "None"
