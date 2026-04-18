"""Thread-safe in-memory training state shared across request handlers.

This is per-worker (per-Gunicorn-process). Scaling beyond one worker would
require moving this into a shared store (e.g. Redis).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class TrainingState:
    """Holds the currently-active model, the latest results, and a cancel flag.

    All mutations go through an :class:`~threading.RLock` so the background
    training thread and Flask request threads don't race.
    """

    _PROGRESS_CAP = 100

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._is_training: bool = False
        self._progress: deque[dict[str, Any]] = deque(maxlen=self._PROGRESS_CAP)
        self._network: Any | None = None
        self._results: dict[str, Any] | None = None
        self._cancel_event = threading.Event()

    # --- flags ----------------------------------------------------------
    @property
    def is_training(self) -> bool:
        with self._lock:
            return self._is_training

    @is_training.setter
    def is_training(self, value: bool) -> None:
        with self._lock:
            self._is_training = value

    # --- network --------------------------------------------------------
    @property
    def network(self) -> Any | None:
        with self._lock:
            return self._network

    @network.setter
    def network(self, value: Any | None) -> None:
        with self._lock:
            self._network = value

    # --- results --------------------------------------------------------
    @property
    def results(self) -> dict[str, Any] | None:
        with self._lock:
            return self._results

    @results.setter
    def results(self, value: dict[str, Any] | None) -> None:
        with self._lock:
            self._results = value

    # --- progress log ---------------------------------------------------
    def add_progress(self, info: dict[str, Any]) -> None:
        """Append a progress entry; the deque self-caps at ``_PROGRESS_CAP``.

        Args:
            info: Arbitrary JSON-serialisable payload. The training worker
                publishes ``{"epoch", "total_epochs", "loss", "accuracy"}``
                dicts, plus sentinels like ``{"cancelled": True}`` or
                ``{"error": "..."}`` on early exit.
        """
        with self._lock:
            self._progress.append(info)

    def get_progress(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent progress entries.

        Args:
            limit: Max number of entries to return, counted from the tail
                (newest). Defaults to 20 to match the UI's status-panel
                tail.

        Returns:
            A fresh list of the last ``limit`` entries (possibly empty).
            The caller may mutate the list safely; the underlying deque
            isn't exposed.
        """
        with self._lock:
            if not self._progress:
                return []
            return list(self._progress)[-limit:]

    # --- cancellation ---------------------------------------------------
    def request_cancel(self) -> bool:
        """Signal the training loop to stop.

        Returns:
            ``True`` iff a run was active when the call arrived (i.e. the
            cancel signal will actually be observed by the worker);
            ``False`` when idle, so the caller can surface a "no training
            in progress" error instead of pretending it succeeded.
        """
        with self._lock:
            if not self._is_training:
                return False
            self._cancel_event.set()
            return True

    def should_cancel(self) -> bool:
        """Thread-safe cancel check (Event is lock-free).

        Returns:
            ``True`` iff a cancel has been requested since the last
            successful :meth:`start_if_idle` / :meth:`reset`.
        """
        return self._cancel_event.is_set()

    # --- lifecycle ------------------------------------------------------
    def reset(self) -> None:
        """Clear progress/results and the cancel flag (keep network).

        Called when starting a new run so stale progress doesn't leak into
        the UI, but the last-trained network stays available for
        ``/api/predict`` and ``/api/sample_images`` until the new run
        completes.
        """
        with self._lock:
            self._progress.clear()
            self._results = None
            self._cancel_event.clear()

    def start_if_idle(self) -> bool:
        """Atomically claim the training slot.

        Combines the ``is_training`` check, transient-state reset, and
        flag flip into a single lock-held section so two concurrent
        ``/api/train`` requests can't both pass the gate.

        Returns:
            True iff the caller now owns the run (and should start a
            worker thread); False if a run was already in progress.
        """
        with self._lock:
            if self._is_training:
                return False
            self._progress.clear()
            self._results = None
            self._cancel_event.clear()
            self._is_training = True
            return True


# Module-level singleton. Importers get the same instance.
training_state = TrainingState()
