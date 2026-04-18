"""Canvas-drawing -> MNIST-style network input.

Typical usage example::

    from PIL import Image
    from mnist_ann.preprocessing import preprocess_drawing

    img = Image.open("user_drawing.png")
    processed = preprocess_drawing(img)   # (28, 28) float32 in [0, 1]
    X = processed.flatten()               # shape (784,) for NeuralNetwork.predict
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Any pixel below this is treated as background. The canvas paints a
# ``#1a1a1a`` (~0.10) baseline; thresholding at 0.15 kills that without
# nibbling the drawn stroke (which clamps near 0.91).
_BG_THRESHOLD = 0.15


def preprocess_drawing(pil_image: Image.Image) -> np.ndarray:
    """Turn a raw canvas PNG into an MNIST-style 28x28 input.

    The canvas draws a light stroke (``#e8e8e8`` ~= 232) on a dark background
    (``#1a1a1a`` ~= 26), which matches MNIST's convention of bright digit on
    dark background -- no polarity inversion needed.

    Steps:
      1. Greyscale -> float in ``[0, 1]``.
      2. Threshold away dim background noise.
      3. Crop to the digit's bounding box.
      4. Scale the longer side to 20 pixels, preserving aspect ratio.
      5. Paste onto a 28x28 canvas and shift so centre of mass sits at
         ``(13.5, 13.5)`` -- the same centring convention the original
         MNIST pipeline uses.

    Returns:
        ``(28, 28)`` ``float32`` array in ``[0, 1]``.
    """
    arr = np.array(pil_image.convert("L"), dtype=np.float32) / 255.0

    arr[arr < _BG_THRESHOLD] = 0.0

    nonzero_rows = np.any(arr > 0, axis=1)
    nonzero_cols = np.any(arr > 0, axis=0)
    if not nonzero_rows.any() or not nonzero_cols.any():
        return np.zeros((28, 28), dtype=np.float32)

    r0, r1 = np.where(nonzero_rows)[0][[0, -1]]
    c0, c1 = np.where(nonzero_cols)[0][[0, -1]]
    digit = arr[r0:r1 + 1, c0:c1 + 1]

    h, w = digit.shape
    if h > w:
        new_h, new_w = 20, max(1, round(w * 20 / h))
    else:
        new_h, new_w = max(1, round(h * 20 / w)), 20

    digit_img = Image.fromarray((digit * 255).astype(np.uint8))
    digit_resized = np.array(
        digit_img.resize((new_w, new_h), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    y0 = (28 - new_h) // 2
    x0 = (28 - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = digit_resized

    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        cy = (canvas * ys).sum() / total
        cx = (canvas * xs).sum() / total
        shift_y = round(13.5 - cy)
        shift_x = round(13.5 - cx)
        if shift_y or shift_x:
            canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
            # np.roll wraps; zero the wrapped strips.
            if shift_y > 0:
                canvas[:shift_y] = 0
            elif shift_y < 0:
                canvas[shift_y:] = 0
            if shift_x > 0:
                canvas[:, :shift_x] = 0
            elif shift_x < 0:
                canvas[:, shift_x:] = 0

    return canvas
