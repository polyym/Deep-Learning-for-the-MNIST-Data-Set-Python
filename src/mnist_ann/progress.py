"""Console progress bar used while training from the CLI.

Typical usage example::

    from mnist_ann.progress import ProgressBar

    bar = ProgressBar(total=50, prefix="Training: ")
    for epoch in range(50):
        # ... train one epoch ...
        bar.update(epoch + 1, loss=0.42, accuracy=87.5)
    bar.finish()
"""

from __future__ import annotations

import sys
import time


class ProgressBar:
    """Simple progress bar for training visualization."""

    def __init__(self, total: int, width: int = 40, prefix: str = ""):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()

    def update(
        self,
        current: int,
        loss: float | None = None,
        accuracy: float | None = None,
    ) -> None:
        """Render the current progress line to stdout."""
        self.current = current
        percent = current / self.total
        filled = int(self.width * percent)
        bar = "=" * filled + ">" + "." * (self.width - filled - 1)

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

        line = (
            f"\r{self.prefix}[{bar}] {current}/{self.total} "
            f"({percent * 100:.1f}%) {eta_str}{metrics}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def finish(self) -> None:
        """Print a newline and final wall-clock summary."""
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\n{self.prefix}Training completed in {elapsed:.1f}s\n")
        sys.stdout.flush()
