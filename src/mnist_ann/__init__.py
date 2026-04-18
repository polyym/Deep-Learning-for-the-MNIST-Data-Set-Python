"""MNIST digit classification with a feedforward ANN and Flask API.

Ported from the MATLAB MA2647 ANN project. See ``app.create_app`` for the
WSGI entry point and ``network.NeuralNetwork`` for the model.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .backend import GPU_AVAILABLE, is_gpu_enabled
from .data import load_mnist_data
from .network import NeuralNetwork

try:
    # Single source of truth: read the version from pyproject.toml via the
    # installed distribution metadata ("mnist-ann" = distribution name).
    __version__ = _pkg_version("mnist-ann")
except PackageNotFoundError:
    # Package not installed (e.g. running directly from a source checkout
    # with PYTHONPATH=src instead of `pip install -e .`).
    __version__ = "0+unknown"

__all__ = [
    "NeuralNetwork",
    "load_mnist_data",
    "GPU_AVAILABLE",
    "is_gpu_enabled",
    "__version__",
]
