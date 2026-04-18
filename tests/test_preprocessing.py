"""Tests for the MNIST-style canvas preprocessing."""

from __future__ import annotations

import numpy as np
from PIL import Image

from mnist_ann.preprocessing import preprocess_drawing


def _canvas_png(
    bg: int = 26,
    fg_regions: list[tuple[int, int, int, int]] | None = None,
):
    """Build a synthetic canvas image: dark bg with optional bright rectangles."""
    arr = np.full((224, 224), bg, dtype=np.uint8)
    for (r0, r1, c0, c1) in fg_regions or []:
        arr[r0:r1, c0:c1] = 232
    return Image.fromarray(arr)


class TestPreprocessDrawing:
    """Canvas PNG -> 28x28 MNIST-style input (polarity, centering, aspect)."""

    def test_empty_canvas_is_zero(self):
        out = preprocess_drawing(_canvas_png())
        assert out.shape == (28, 28)
        assert out.dtype == np.float32
        assert out.sum() == 0

    def test_output_polarity_matches_mnist(self):
        # A bright rectangle in the middle of a dark canvas
        out = preprocess_drawing(_canvas_png(fg_regions=[(80, 140, 80, 140)]))
        assert out[0, 0] == 0.0              # background stays 0
        assert out.max() > 0.8               # digit region stays bright (~0.9-1.0)

    def test_center_of_mass_centered(self):
        # Off-center bright region should be pulled to the canvas center.
        out = preprocess_drawing(_canvas_png(fg_regions=[(90, 134, 100, 124)]))
        total = out.sum()
        assert total > 0
        ys, xs = np.indices(out.shape)
        cy = (out * ys).sum() / total
        cx = (out * xs).sum() / total
        assert abs(cy - 13.5) < 1.0
        assert abs(cx - 13.5) < 1.0

    def test_aspect_ratio_preserved(self):
        # A very tall stroke should fit within a 20-row band; width scales down.
        out = preprocess_drawing(_canvas_png(fg_regions=[(10, 210, 100, 124)]))
        rows_with_signal = np.any(out > 0, axis=1)
        cols_with_signal = np.any(out > 0, axis=0)
        assert rows_with_signal.sum() <= 20
        assert cols_with_signal.sum() <= 20
