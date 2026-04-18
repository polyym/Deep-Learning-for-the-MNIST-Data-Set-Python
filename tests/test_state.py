"""Tests for the thread-safe TrainingState."""

from __future__ import annotations

from mnist_ann.state import training_state


class TestTrainingState:
    """Thread-safe flags, progress log cap, reset, and cancellation."""

    def test_is_training_property(self):
        assert training_state.is_training is False
        training_state.is_training = True
        assert training_state.is_training is True

    def test_progress_management(self):
        training_state.add_progress({"epoch": 1})
        training_state.add_progress({"epoch": 2})

        progress = training_state.get_progress(10)
        assert len(progress) == 2
        assert progress[0]["epoch"] == 1
        assert progress[1]["epoch"] == 2

    def test_progress_limit(self):
        for i in range(150):
            training_state.add_progress({"epoch": i})

        assert len(training_state._progress) == 100
        assert training_state._progress[0]["epoch"] == 50

    def test_reset_clears_progress_and_results_and_cancel(self):
        training_state.add_progress({"epoch": 1})
        training_state._results = {"test": "data"}
        training_state._cancel_event.set()

        training_state.reset()

        assert len(training_state._progress) == 0
        assert training_state._results is None
        assert not training_state._cancel_event.is_set()

    def test_request_cancel_noop_when_idle(self):
        assert not training_state.is_training
        assert training_state.request_cancel() is False
        assert not training_state.should_cancel()

    def test_request_cancel_signals_when_training(self):
        training_state.is_training = True
        try:
            assert training_state.request_cancel() is True
            assert training_state.should_cancel() is True
        finally:
            training_state.is_training = False
            training_state._cancel_event.clear()


class TestStartIfIdle:
    """Atomic training-slot claim used by /api/train to avoid a TOCTOU race."""

    def test_claims_slot_when_idle(self):
        assert training_state.is_training is False
        assert training_state.start_if_idle() is True
        assert training_state.is_training is True

    def test_rejects_when_already_training(self):
        training_state.start_if_idle()
        assert training_state.start_if_idle() is False

    def test_claim_clears_transient_state(self):
        training_state.add_progress({"epoch": 1})
        training_state._results = {"old": "data"}
        training_state._cancel_event.set()

        assert training_state.start_if_idle() is True
        assert training_state._progress == []
        assert training_state._results is None
        assert not training_state._cancel_event.is_set()

    def test_concurrent_claims_have_single_winner(self):
        # Widen the check-then-set window to confirm atomicity under
        # contention; pre-fix this returned >1 winner.
        import threading
        import time

        winners = 0
        lock = threading.Lock()

        def racer():
            nonlocal winners
            if not training_state.start_if_idle():
                return
            time.sleep(0.001)
            with lock:
                winners += 1

        threads = [threading.Thread(target=racer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert winners == 1
