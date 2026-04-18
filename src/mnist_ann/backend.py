"""GPU detection and the NumPy/CuPy array-module abstraction.

Import this module once at startup; :data:`GPU_AVAILABLE` and :data:`_xp`
are set as a side effect. CPU-only callers get NumPy; GPU callers get CuPy
(if a working CUDA device is present).
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GPU_AVAILABLE: bool = False
_xp: Any = None  # bound to numpy or cupy by _detect_gpu()


def _setup_cuda_path() -> None:
    """On Windows, expose the latest CUDA toolkit DLLs to CuPy's loader."""
    if os.name != "nt":
        return

    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if not os.path.exists(cuda_base):
        return

    versions = glob.glob(os.path.join(cuda_base, "v*"))
    if not versions:
        return

    cuda_path = max(versions)  # latest version
    os.environ.setdefault("CUDA_PATH", cuda_path)
    cuda_bin = os.path.join(cuda_path, "bin")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")


_GPU_SELFTEST_SUM = 10.0  # expected ``cp.sum(cp.ones(10))``


def _detect_gpu() -> None:
    """Probe for a working CuPy+CUDA install and set the array backend.

    Side effects on the module globals:
      * :data:`GPU_AVAILABLE` -> ``True`` iff a CuPy device round-trip succeeds.
      * :data:`_xp` -> bound to ``cupy`` on success, otherwise ``numpy``.

    Runs once at import time; callers should treat the module globals as
    read-only afterwards.
    """
    global GPU_AVAILABLE, _xp

    _setup_cuda_path()

    try:
        # CuPy is an optional extra; importing at module scope would make
        # `pip install .` without the GPU extra fail on import.
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise RuntimeError("No CUDA devices found")

        # Round-trip a small buffer to confirm the device is actually usable
        # without requiring the nvrtc JIT compiler.
        test_arr = cp.zeros(10, dtype=cp.float32)
        test_arr += 1.0
        if float(cp.sum(test_arr)) != _GPU_SELFTEST_SUM:
            raise RuntimeError("GPU computation test failed")
        del test_arr

        _xp = cp
        GPU_AVAILABLE = True
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        logger.info("GPU detected: %s - using CuPy acceleration", device_name)
    except Exception as e:
        _xp = np
        GPU_AVAILABLE = False
        # Only surface the specific error when CuPy was installed but failed.
        try:
            import cupy  # noqa: F401

            logger.info(
                "GPU not available (CuPy error: %s) - using NumPy (CPU)",
                type(e).__name__,
            )
        except ImportError:
            logger.info(
                "GPU not available (CuPy not installed); using NumPy (CPU). "
                "If you have an NVIDIA GPU, enable it with "
                "`uv sync --extra gpu-cuda12` (CUDA 12.x) or "
                "`uv sync --extra gpu-cuda11` (CUDA 11.x), then restart."
            )


# Side-effect: decide backend at import time so the rest of the package can
# treat ``_xp`` as a stable module-level reference.
_detect_gpu()


def get_array_module():
    """Return the active array module (``numpy`` or ``cupy``)."""
    return _xp


def to_cpu(arr):
    """Move a (possibly GPU) array to host memory. No-op on NumPy."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return arr


def to_gpu(arr):
    """Move an array onto the active backend. No-op if CPU-only."""
    if GPU_AVAILABLE:
        return _xp.asarray(arr)
    return arr


def is_gpu_enabled() -> bool:
    """True iff a working CUDA device and CuPy install were detected."""
    return GPU_AVAILABLE
