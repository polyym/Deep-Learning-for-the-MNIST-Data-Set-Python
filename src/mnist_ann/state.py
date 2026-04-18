"""Thread-safe in-memory training state shared across request handlers.

This is per-worker (per-Gunicorn-process). Scaling beyond one worker would
require moving this into a shared store (e.g. Redis).
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional


class TrainingState:
    """Holds the currently-active model, the latest results, and a cancel flag.

    All mutations go through an :class:`~threading.RLock` so the background
    training thread and Flask request threads don't race.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._is_training: bool = False
        self._progress: List[Dict[str, Any]] = []
        self._network: Optional[Any] = None
        self._results: Optional[Dict[str, Any]] = None
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
    def network(self) -> Optional[Any]:
        with self._lock:
            return self._network

    @network.setter
    def network(self, value: Optional[Any]) -> None:
        with self._lock:
            self._network = value

    # --- results --------------------------------------------------------
    @property
    def results(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._results

    @results.setter
    def results(self, value: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self._results = value

    # --- progress log ---------------------------------------------------
    def add_progress(self, info: Dict[str, Any]) -> None:
        """Append a progress entry, capped at the most recent 100 entries."""
        with self._lock:
            self._progress.append(info)
            if len(self._progress) > 100:
                self._progress = self._progress[-100:]

    def get_progress(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return self._progress[-limit:] if self._progress else []

    # --- cancellation ---------------------------------------------------
    def request_cancel(self) -> bool:
        """Signal the training loop to stop. Returns True iff a run was active."""
        with self._lock:
            if not self._is_training:
                return False
            self._cancel_event.set()
            return True

    def should_cancel(self) -> bool:
        """Thread-safe cancel check (Event is lock-free)."""
        return self._cancel_event.is_set()

    # --- lifecycle ------------------------------------------------------
    def reset(self) -> None:
        """Clear progress/results and the cancel flag (keep network)."""
        with self._lock:
            self._progress = []
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
            self._progress = []
            self._results = None
            self._cancel_event.clear()
            self._is_training = True
            return True


# Module-level singleton. Importers get the same instance.
training_state = TrainingState()
