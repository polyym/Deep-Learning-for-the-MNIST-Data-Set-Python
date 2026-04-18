"""Feedforward neural network for MNIST digit classification.

Mirrors my own MATLAB implementation (``ann1923114.m``):

- 784 input units (28x28 images)
- 3 sigmoid hidden layers (user-configurable widths)
- 5 softmax output units (4 selected digits + "None")
- Two backprop variants: Calculus-Based (CB) and Unscaled Heuristic (UHB)
- Optional CuPy/GPU acceleration via :mod:`.backend`
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import math

import numpy as np

from .backend import GPU_AVAILABLE, _xp, to_cpu
from .progress import ProgressBar


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
        hidden_layers: Tuple[int, int, int] = (64, 32, 16),
        learning_rate: float = 0.01,
        digits_to_classify: Tuple[int, int, int, int] = (0, 1, 2, 3),
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

        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

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

    def sigmoid(self, x):
        """Elementwise logistic sigmoid, clipped to avoid overflow in ``exp``.

        The clip range ``[-80, 80]`` keeps ``exp(-x)`` inside float32's
        representable range (``exp(88.7) ~= float32 max``); past that, sigmoid
        is already saturated to 0 or 1 within float32 resolution anyway.
        """
        xp = self.xp
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -80.0, 80.0)))

    def sigmoid_derivative(self, x):
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

    def softmax(self, x):
        """Numerically stable softmax, computed column-wise."""
        xp = self.xp
        exp_x = xp.exp(x - xp.max(x, axis=0, keepdims=True))
        return exp_x / xp.sum(exp_x, axis=0, keepdims=True)

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------
    def one_hot_encode(self, labels) -> np.ndarray:
        """One-hot encode labels into 5-class targets.

        Classes 0-3 map to the configured digits; class 4 is "None".
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
    def forward(self, x) -> Tuple[np.ndarray, Dict]:
        """Forward propagate ``x`` through the network.

        ``x`` may be a single column vector ``(784,)`` / ``(784, 1)`` or a
        batch ``(784, N)``; the cached activations are sized accordingly.
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

    def backward_uhb(self, y_true, cache: Dict) -> Dict:
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

    def backward_cb(self, y_true, cache: Dict) -> Dict:
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

    def update_weights(self, grads: Dict) -> None:
        """Apply gradient-descent updates from a backward pass.

        Weight updates via ``a @ S.T`` naturally sum contributions across
        the batch dim. Bias updates must explicitly sum across axis=1 so
        they stay shape ``(n, 1)``; without this, batched inputs silently
        broadcast the bias to ``(n, N)`` and corrupt the network.
        """
        xp = self.xp
        S2, S3, S4, S5 = grads["S2"], grads["S3"], grads["S4"], grads["S5"]
        a1, a2, a3, a4 = grads["a1"], grads["a2"], grads["a3"], grads["a4"]

        self.W5 = self.W5 - self.lr * (a4 @ S5.T)
        self.b5 = self.b5 - self.lr * xp.sum(S5, axis=1, keepdims=True)

        self.W4 = self.W4 - self.lr * (a3 @ S4.T)
        self.b4 = self.b4 - self.lr * xp.sum(S4, axis=1, keepdims=True)

        self.W3 = self.W3 - self.lr * (a2 @ S3.T)
        self.b3 = self.b3 - self.lr * xp.sum(S3, axis=1, keepdims=True)

        self.W2 = self.W2 - self.lr * (a1 @ S2.T)
        self.b2 = self.b2 - self.lr * xp.sum(S2, axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def cross_entropy_loss(self, y_true, y_pred) -> float:
        """Categorical cross-entropy, clipped to avoid ``log(0)``.

        Returns the **sum** over all classes and samples (not the mean), so
        training-loop scaling by ``n_samples / samples_done`` is consistent.
        """
        xp = self.xp
        epsilon = 1e-15
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        loss = -xp.sum(y_true * xp.log(y_pred))
        return float(to_cpu(loss)) if self.use_gpu else float(loss)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        X_train,
        Y_train,
        n_epochs: int,
        backprop_method: str = "cb",
        progress_callback: Optional[Callable] = None,
        show_progress_bar: bool = True,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """Train the network with online SGD.

        Args:
            X_train: ``(784, n_samples)`` inputs.
            Y_train: ``(5, n_samples)`` one-hot targets.
            n_epochs: Number of training epochs.
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
        """
        xp = self.xp

        if self.use_gpu:
            X_train = xp.asarray(X_train, dtype=xp.float32)
            Y_train = xp.asarray(Y_train, dtype=xp.float32)

        n_samples = X_train.shape[1]
        self.loss_history = []
        self.accuracy_history = []

        backward_fn = self.backward_cb if backprop_method == "cb" else self.backward_uhb

        progress_bar: Optional[ProgressBar] = None
        if show_progress_bar:
            gpu_label = " [GPU]" if self.use_gpu else " [CPU]"
            progress_bar = ProgressBar(n_epochs, prefix=f"Training{gpu_label}: ")

        # Cancel checks fire ~every 100 samples (lock-free ``Event.is_set``
        # is cheap). Progress reports are rarer so UI updates don't thrash.
        cancel_check_interval = 100
        report_interval = min(1000, max(100, n_samples // 20))
        epsilon = 1e-15
        cancelled = False
        diverged = False

        for epoch in range(n_epochs):
            if should_cancel and should_cancel():
                cancelled = True
                break

            # CPU indices avoid per-sample GPU sync when slicing the GPU-resident
            # training matrices.
            indices = np.random.permutation(n_samples)

            # Accumulate on-device; sync only on reporting/epoch boundaries.
            running_loss = xp.zeros((), dtype=xp.float32)
            running_correct = xp.zeros((), dtype=xp.int64)

            for j in range(n_samples):
                i = int(indices[j])
                x = X_train[:, i:i + 1]
                y = Y_train[:, i:i + 1]

                y_pred, cache = self.forward(x)

                y_pred_clipped = xp.clip(y_pred, epsilon, 1.0 - epsilon)
                running_loss = running_loss - xp.sum(y * xp.log(y_pred_clipped))
                running_correct = running_correct + (xp.argmax(y_pred) == xp.argmax(y))

                grads = backward_fn(y, cache)
                self.update_weights(grads)

                # Fast cancel path: no sync, no callback.
                if should_cancel and (j + 1) % cancel_check_interval == 0 and should_cancel():
                    cancelled = True
                    break

                if (j + 1) % report_interval == 0 and progress_callback:
                    samples_done = j + 1
                    loss_so_far = float(to_cpu(running_loss))
                    correct_so_far = int(to_cpu(running_correct))
                    # Divergence guard: UHB at lr >> paper's 0.001 can push
                    # weights until sigmoid saturates and derivatives blow up
                    # to NaN. Stop early instead of wasting CPU on garbage.
                    if not _is_finite(loss_so_far):
                        diverged = True
                        break
                    progress_callback({
                        "epoch": epoch + (samples_done / n_samples),
                        "total_epochs": n_epochs,
                        "loss": loss_so_far * (n_samples / samples_done),
                        "accuracy": 100.0 * correct_so_far / samples_done,
                        "samples_done": samples_done,
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
    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Run a single batched forward pass and return (classes, probs)."""
        xp = self.xp
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions, _ = self.forward(X)
        pred_classes = xp.argmax(predictions, axis=0)
        return to_cpu(pred_classes), to_cpu(predictions)

    def evaluate(self, X, Y) -> Dict:
        """Score the network on ``(X, Y)`` and return metrics + confusion.

        Returns overall accuracy, per-class accuracy, the 5x5 confusion matrix,
        and arrays of true/predicted labels and predicted probabilities.
        """
        xp = self.xp
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)
            Y = xp.asarray(Y, dtype=xp.float32)

        n_samples = X.shape[1]
        pred_classes, predictions = self.predict(X)

        true_classes = to_cpu(xp.argmax(Y, axis=0)) if self.use_gpu else np.argmax(Y, axis=0)
        pred_classes = to_cpu(pred_classes) if self.use_gpu else pred_classes
        predictions = to_cpu(predictions) if self.use_gpu else predictions

        accuracy = 100 * np.sum(pred_classes == true_classes) / n_samples

        confusion = np.zeros((5, 5), dtype=int)
        for i in range(n_samples):
            confusion[pred_classes[i], true_classes[i]] += 1

        per_class = {}
        class_names = [str(self.A), str(self.B), str(self.C), str(self.D), "None"]
        for c in range(5):
            total_c = np.sum(true_classes == c)
            correct_c = confusion[c, c]
            per_class[class_names[c]] = {
                "correct": int(correct_c),
                "total": int(total_c),
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

    def get_digit_label(self, class_idx: int) -> str:
        """Map a class index 0..4 to its human label ('0'..'9' or 'None')."""
        labels = (str(self.A), str(self.B), str(self.C), str(self.D), "None")
        return labels[class_idx] if 0 <= class_idx < len(labels) else "None"
