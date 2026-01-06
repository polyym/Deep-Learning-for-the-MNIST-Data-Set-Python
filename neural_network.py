"""
Neural Network for MNIST Digit Classification
Converted from MATLAB implementation (ann1923114.m)

This module implements a feedforward neural network with:
- 784 input nodes (28x28 images)
- 3 hidden layers with configurable sizes
- 5 output nodes (4 selected digits + "None" class)
- Sigmoid activation for hidden layers
- Softmax activation for output layer
- Two backpropagation methods: Unscaled Heuristic (UHB) and Calculus-based (CB)
- Optional GPU acceleration via CuPy
"""

import sys
import time
from typing import Tuple, List, Dict, Optional, Callable

# =============================================================================
# GPU Detection and Array Backend
# =============================================================================

GPU_AVAILABLE = False
_xp = None  # Array module (numpy or cupy)

def _setup_cuda_path():
    """Try to set up CUDA environment on Windows."""
    import os
    import glob

    if os.name != 'nt':  # Not Windows
        return

    # Common CUDA installation paths on Windows
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_base):
        # Find the latest CUDA version
        versions = glob.glob(os.path.join(cuda_base, "v*"))
        if versions:
            cuda_path = max(versions)  # Get latest version
            os.environ.setdefault('CUDA_PATH', cuda_path)
            # Add bin directory to PATH for DLLs
            cuda_bin = os.path.join(cuda_path, 'bin')
            if cuda_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')


def _detect_gpu():
    """Detect GPU availability and set up the array backend."""
    global GPU_AVAILABLE, _xp

    # Try to set up CUDA paths on Windows
    _setup_cuda_path()

    try:
        import cupy as cp
        # Test if CUDA is actually available by running a real operation
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise RuntimeError("No CUDA devices found")

        # Test that we can actually allocate and use GPU memory
        # Use simple operations that don't require nvrtc compilation
        test_arr = cp.zeros(10, dtype=cp.float32)
        test_arr += 1.0
        result = float(cp.sum(test_arr))
        del test_arr

        if result != 10.0:
            raise RuntimeError("GPU computation test failed")

        _xp = cp
        GPU_AVAILABLE = True
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        print(f"GPU detected: {device_name} - using CuPy acceleration")
    except Exception as e:
        import numpy as np
        _xp = np
        GPU_AVAILABLE = False
        # Only show detailed error if CuPy was installed but failed
        try:
            import cupy
            print(f"GPU not available (CuPy error: {type(e).__name__}) - using NumPy (CPU)")
        except ImportError:
            print("GPU not available - using NumPy (CPU)")

# Detect GPU on module load
_detect_gpu()

def get_array_module():
    """Get the current array module (numpy or cupy)."""
    return _xp

def to_cpu(arr):
    """Convert array to CPU (numpy array)."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return arr

def to_gpu(arr):
    """Convert array to GPU if available."""
    if GPU_AVAILABLE:
        return _xp.asarray(arr)
    return arr

def is_gpu_enabled():
    """Check if GPU is available."""
    return GPU_AVAILABLE


# =============================================================================
# Progress Bar
# =============================================================================

class ProgressBar:
    """Simple progress bar for training visualization."""

    def __init__(self, total: int, width: int = 40, prefix: str = ''):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()

    def update(self, current: int, loss: float = None, accuracy: float = None):
        """Update progress bar."""
        self.current = current
        percent = current / self.total
        filled = int(self.width * percent)
        bar = '=' * filled + '>' + '.' * (self.width - filled - 1)

        elapsed = time.time() - self.start_time
        if current > 0:
            eta = elapsed * (self.total - current) / current
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        metrics = ""
        if loss is not None:
            metrics += f" | Loss: {loss:.4f}"
        if accuracy is not None:
            metrics += f" | Acc: {accuracy:.2f}%"

        line = f"\r{self.prefix}[{bar}] {current}/{self.total} ({percent*100:.1f}%) {eta_str}{metrics}"
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self):
        """Complete the progress bar."""
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\n{self.prefix}Training completed in {elapsed:.1f}s\n")
        sys.stdout.flush()


# =============================================================================
# Neural Network Class
# =============================================================================

class NeuralNetwork:
    """
    A feedforward neural network for MNIST digit classification.
    Supports both CPU (NumPy) and GPU (CuPy) computation.
    """

    def __init__(
        self,
        hidden_layers: Tuple[int, int, int] = (128, 64, 32),
        learning_rate: float = 0.01,
        digits_to_classify: Tuple[int, int, int, int] = (0, 1, 2, 3),
        use_gpu: bool = True
    ):
        """
        Initialize the neural network.

        Args:
            hidden_layers: Tuple of (U, V, W) - number of neurons in each hidden layer
            learning_rate: Learning rate for gradient descent
            digits_to_classify: Tuple of 4 digits (A, B, C, D) to classify
            use_gpu: Whether to use GPU if available (default: True)
        """
        self.U, self.V, self.W = hidden_layers
        self.lr = learning_rate
        self.A, self.B, self.C, self.D = digits_to_classify

        # Decide whether to use GPU
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = _xp if self.use_gpu else __import__('numpy')

        # Network architecture
        self.Ni = 784  # Input nodes (28x28)
        self.No = 5    # Output nodes (4 classes + None)

        # Initialize weights and biases
        self._initialize_weights()

        # Training history
        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    def _initialize_weights(self):
        """Initialize weights uniformly in [-0.5, 0.5] and biases to zero."""
        xp = self.xp

        # Weights between layers
        self.W2 = 0.5 - xp.random.rand(self.Ni, self.U).astype(xp.float32)
        self.W3 = 0.5 - xp.random.rand(self.U, self.V).astype(xp.float32)
        self.W4 = 0.5 - xp.random.rand(self.V, self.W).astype(xp.float32)
        self.W5 = 0.5 - xp.random.rand(self.W, self.No).astype(xp.float32)

        # Biases
        self.b2 = xp.zeros((self.U, 1), dtype=xp.float32)
        self.b3 = xp.zeros((self.V, 1), dtype=xp.float32)
        self.b4 = xp.zeros((self.W, 1), dtype=xp.float32)
        self.b5 = xp.zeros((self.No, 1), dtype=xp.float32)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        xp = self.xp
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        xp = self.xp
        clipped = xp.clip(x, -500, 500)
        exp_neg = xp.exp(-clipped)
        return exp_neg / ((1.0 + exp_neg) ** 2)

    def softmax(self, x):
        """Numerically stable softmax."""
        xp = self.xp
        exp_x = xp.exp(x - xp.max(x))
        return exp_x / xp.sum(exp_x)

    def one_hot_encode(self, labels) -> 'array':
        """
        Convert labels to one-hot encoding for 5 classes.
        Classes 1-4 are the specified digits, class 5 is "None".
        """
        xp = self.xp

        # Convert to CPU numpy for iteration if needed
        labels_cpu = to_cpu(labels) if hasattr(labels, 'get') else labels

        n_samples = len(labels_cpu)
        Y = xp.zeros((5, n_samples), dtype=xp.float32)

        # Use numpy for the loop, then convert result
        import numpy as np
        Y_cpu = np.zeros((5, n_samples), dtype=np.float32)

        for i, label in enumerate(labels_cpu):
            if label == self.A:
                Y_cpu[0, i] = 1
            elif label == self.B:
                Y_cpu[1, i] = 1
            elif label == self.C:
                Y_cpu[2, i] = 1
            elif label == self.D:
                Y_cpu[3, i] = 1
            else:
                Y_cpu[4, i] = 1  # "None" class

        if self.use_gpu:
            return xp.asarray(Y_cpu)
        return Y_cpu

    def forward(self, x) -> Tuple['array', Dict]:
        """
        Forward propagation through the network.

        Args:
            x: Input vector (784, 1)

        Returns:
            Output predictions and intermediate values for backprop
        """
        xp = self.xp

        # Ensure input is column vector
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        a1 = x

        # Layer 2
        n2 = self.W2.T @ a1 + self.b2
        a2 = self.sigmoid(n2)

        # Layer 3
        n3 = self.W3.T @ a2 + self.b3
        a3 = self.sigmoid(n3)

        # Layer 4
        n4 = self.W4.T @ a3 + self.b4
        a4 = self.sigmoid(n4)

        # Output layer (softmax)
        n5 = self.W5.T @ a4 + self.b5
        a5 = self.softmax(n5)

        cache = {
            'a1': a1, 'n2': n2, 'a2': a2,
            'n3': n3, 'a3': a3,
            'n4': n4, 'a4': a4,
            'n5': n5, 'a5': a5
        }

        return a5, cache

    def backward_uhb(self, y_true, cache: Dict) -> Dict:
        """
        Unscaled Heuristic Backpropagation (UHB).
        """
        xp = self.xp

        a1 = cache['a1']
        n2, a2 = cache['n2'], cache['a2']
        n3, a3 = cache['n3'], cache['a3']
        n4, a4 = cache['n4'], cache['a4']
        a5 = cache['a5']

        # Diagonal matrices of activation derivatives
        A2 = xp.diag(self.sigmoid_derivative(n2).flatten())
        A3 = xp.diag(self.sigmoid_derivative(n3).flatten())
        A4 = xp.diag(self.sigmoid_derivative(n4).flatten())

        # Error in output layer
        e5 = y_true - a5
        S5 = -e5

        # UHB: Error flows back through weights
        e4 = self.W5 @ e5
        S4 = -2 * A4 @ e4

        e3 = self.W4 @ e4
        S3 = -2 * A3 @ e3

        e2 = self.W3 @ e3
        S2 = -2 * A2 @ e2

        return {'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
                'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4}

    def backward_cb(self, y_true, cache: Dict) -> Dict:
        """
        Calculus-based Backpropagation (CB).
        """
        xp = self.xp

        a1 = cache['a1']
        n2, a2 = cache['n2'], cache['a2']
        n3, a3 = cache['n3'], cache['a3']
        n4, a4 = cache['n4'], cache['a4']
        a5 = cache['a5']

        # Diagonal matrices of activation derivatives
        A2 = xp.diag(self.sigmoid_derivative(n2).flatten())
        A3 = xp.diag(self.sigmoid_derivative(n3).flatten())
        A4 = xp.diag(self.sigmoid_derivative(n4).flatten())

        # Error in output layer
        e5 = y_true - a5
        S5 = -e5

        # CB: Chain rule for sensitivities
        S4 = A4 @ self.W5 @ S5
        S3 = A3 @ self.W4 @ S4
        S2 = A2 @ self.W3 @ S3

        return {'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
                'a1': a1, 'a2': a2, 'a3': a3, 'a4': a4}

    def update_weights(self, grads: Dict):
        """Update weights and biases using gradient descent."""
        S2, S3, S4, S5 = grads['S2'], grads['S3'], grads['S4'], grads['S5']
        a1, a2, a3, a4 = grads['a1'], grads['a2'], grads['a3'], grads['a4']

        self.W5 = self.W5 - self.lr * (a4 @ S5.T)
        self.b5 = self.b5 - self.lr * S5

        self.W4 = self.W4 - self.lr * (a3 @ S4.T)
        self.b4 = self.b4 - self.lr * S4

        self.W3 = self.W3 - self.lr * (a2 @ S3.T)
        self.b3 = self.b3 - self.lr * S3

        self.W2 = self.W2 - self.lr * (a1 @ S2.T)
        self.b2 = self.b2 - self.lr * S2

    def cross_entropy_loss(self, y_true, y_pred) -> float:
        """Calculate cross-entropy loss."""
        xp = self.xp
        epsilon = 1e-15
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        loss = -xp.sum(y_true * xp.log(y_pred))
        return float(to_cpu(loss)) if self.use_gpu else float(loss)

    def train(
        self,
        X_train,
        Y_train,
        n_epochs: int,
        backprop_method: str = 'cb',
        progress_callback: Optional[Callable] = None,
        show_progress_bar: bool = True
    ) -> Dict:
        """
        Train the neural network.

        Args:
            X_train: Training data (784, n_samples)
            Y_train: One-hot encoded labels (5, n_samples)
            n_epochs: Number of training epochs
            backprop_method: 'uhb' or 'cb'
            progress_callback: Optional callback for progress updates
            show_progress_bar: Whether to show progress bar in console

        Returns:
            Training history
        """
        xp = self.xp

        # Convert data to GPU if needed
        if self.use_gpu:
            X_train = xp.asarray(X_train, dtype=xp.float32)
            Y_train = xp.asarray(Y_train, dtype=xp.float32)

        n_samples = X_train.shape[1]
        self.loss_history = []
        self.accuracy_history = []

        backward_fn = self.backward_cb if backprop_method == 'cb' else self.backward_uhb

        # Initialize progress bar
        progress_bar = None
        if show_progress_bar:
            gpu_label = " [GPU]" if self.use_gpu else " [CPU]"
            progress_bar = ProgressBar(n_epochs, prefix=f"Training{gpu_label}: ")

        # For intra-epoch progress reporting (useful for large datasets)
        # Report progress every ~2 seconds worth of samples or 1000 samples, whichever is smaller
        intra_epoch_report_interval = min(1000, max(100, n_samples // 20))
        last_reported_loss = 0.0
        last_reported_accuracy = 0.0

        for epoch in range(n_epochs):
            # Shuffle training data
            if self.use_gpu:
                indices = xp.random.permutation(n_samples)
            else:
                indices = xp.random.permutation(n_samples)

            # Track running stats within epoch for progress reporting
            running_loss = 0.0
            running_correct = 0

            for j in range(n_samples):
                i = int(indices[j])
                x = X_train[:, i:i+1]
                y = Y_train[:, i:i+1]

                # Forward pass
                y_pred, cache = self.forward(x)

                # Track running stats
                running_loss += self.cross_entropy_loss(y, y_pred)
                pred_class = int(to_cpu(xp.argmax(y_pred))) if self.use_gpu else int(xp.argmax(y_pred))
                true_class = int(to_cpu(xp.argmax(y))) if self.use_gpu else int(xp.argmax(y))
                if pred_class == true_class:
                    running_correct += 1

                # Backward pass
                grads = backward_fn(y, cache)

                # Update weights
                self.update_weights(grads)

                # Report intra-epoch progress for large datasets
                if progress_callback and (j + 1) % intra_epoch_report_interval == 0:
                    samples_done = j + 1
                    # Calculate progress as fraction of epoch + samples within epoch
                    epoch_progress = epoch + (samples_done / n_samples)
                    current_loss = running_loss / samples_done
                    current_accuracy = 100 * running_correct / samples_done
                    progress_callback({
                        'epoch': epoch_progress,
                        'total_epochs': n_epochs,
                        'loss': current_loss * n_samples,  # Scale to match epoch loss
                        'accuracy': current_accuracy,
                        'samples_done': samples_done,
                        'total_samples': n_samples
                    })
                    last_reported_loss = current_loss * n_samples
                    last_reported_accuracy = current_accuracy

            # Calculate final epoch loss and accuracy (already tracked during training)
            total_loss = running_loss
            accuracy = 100 * running_correct / n_samples

            self.loss_history.append(total_loss)
            self.accuracy_history.append(accuracy)

            # Update progress bar
            if progress_bar:
                progress_bar.update(epoch + 1, loss=total_loss, accuracy=accuracy)

            if progress_callback:
                progress_callback({
                    'epoch': epoch + 1,
                    'total_epochs': n_epochs,
                    'loss': total_loss,
                    'accuracy': accuracy
                })

        # Finish progress bar
        if progress_bar:
            progress_bar.finish()

        return {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }

    def predict(self, X) -> Tuple['array', 'array']:
        """
        Make predictions on input data.

        Args:
            X: Input data (784, n_samples) or (784,)

        Returns:
            Predicted class indices and probabilities (as numpy arrays)
        """
        xp = self.xp
        import numpy as np

        # Convert to GPU if needed
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[1]
        predictions = xp.zeros((5, n_samples), dtype=xp.float32)

        for i in range(n_samples):
            y_pred, _ = self.forward(X[:, i:i+1])
            predictions[:, i] = y_pred.flatten()

        pred_classes = xp.argmax(predictions, axis=0)

        # Convert back to CPU numpy for API compatibility
        return to_cpu(pred_classes), to_cpu(predictions)

    def evaluate(self, X, Y) -> Dict:
        """
        Evaluate the network on test data.

        Returns accuracy, per-class accuracy, and confusion matrix.
        """
        import numpy as np
        xp = self.xp

        # Convert to GPU if needed
        if self.use_gpu:
            X = xp.asarray(X, dtype=xp.float32)
            Y = xp.asarray(Y, dtype=xp.float32)

        n_samples = X.shape[1]
        pred_classes, predictions = self.predict(X)

        # Get true classes (ensure CPU numpy)
        true_classes = to_cpu(xp.argmax(Y, axis=0)) if self.use_gpu else np.argmax(Y, axis=0)
        pred_classes = to_cpu(pred_classes) if self.use_gpu else pred_classes
        predictions = to_cpu(predictions) if self.use_gpu else predictions

        # Overall accuracy
        accuracy = 100 * np.sum(pred_classes == true_classes) / n_samples

        # Confusion matrix (5x5)
        confusion = np.zeros((5, 5), dtype=int)
        for i in range(n_samples):
            confusion[pred_classes[i], true_classes[i]] += 1

        # Per-class accuracy
        per_class = {}
        class_names = [str(self.A), str(self.B), str(self.C), str(self.D), 'None']
        for c in range(5):
            total_c = np.sum(true_classes == c)
            correct_c = confusion[c, c]
            per_class[class_names[c]] = {
                'correct': int(correct_c),
                'total': int(total_c),
                'accuracy': 100 * correct_c / total_c if total_c > 0 else 0
            }

        return {
            'accuracy': float(accuracy),
            'per_class': per_class,
            'confusion_matrix': confusion.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': true_classes.tolist(),
            'pred_labels': pred_classes.tolist()
        }

    def get_digit_label(self, class_idx: int) -> str:
        """Convert class index to digit label."""
        if class_idx == 0:
            return str(self.A)
        elif class_idx == 1:
            return str(self.B)
        elif class_idx == 2:
            return str(self.C)
        elif class_idx == 3:
            return str(self.D)
        else:
            return 'None'


# =============================================================================
# Data Loading
# =============================================================================

def load_mnist_data(filepath: str) -> Tuple['array', 'array']:
    """
    Load MNIST data from CSV file.

    Returns:
        X: Image data normalized to [0, 1], shape (784, n_samples)
        labels: Raw labels, shape (n_samples,)
    """
    import numpy as np

    # Try to detect if file has a header by reading first line
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()

    # Check if first line contains non-numeric header
    has_header = False
    try:
        # Try to parse first value as number
        first_val = first_line.split(',')[0]
        float(first_val)
    except ValueError:
        has_header = True

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1 if has_header else 0)

    # Filter out any rows with NaN values (from malformed lines)
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]

    labels = data[:, 0].astype(int)
    X = data[:, 1:].T / 255.0  # Normalize and transpose
    return X.astype(np.float32), labels
