"""End-to-end HTTP tests.

These run a real training worker (no :func:`mock_train_thread`) and exercise
the full train -> status -> results -> predict flow. They use the 100-sample
shipped dataset and a deliberately small network so a pass takes <5 s.
"""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
import pytest
from PIL import Image

from mnist_ann.state import training_state


def _png_base64(pixels: np.ndarray) -> str:
    """Encode a uint8 2D pixel array as a ``data:image/png;base64,...`` URL."""
    img = Image.fromarray(pixels, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _wait_for_training(client, timeout_s: float = 15.0) -> dict:
    """Poll ``/api/status`` until training finishes; return the final payload."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        response = client.get("/api/status")
        last = json.loads(response.data)
        if not last["is_training"]:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Training did not finish within {timeout_s}s; last status={last}"
    )


class TestTrainPredictFlow:
    """Full POST /api/train -> poll /api/status -> POST /api/predict cycle.

    Guards against regressions that slip past the per-module unit tests,
    e.g. a bad nn.train() signature that breaks the worker thread or a
    serialisation bug that only shows up when the results dict is jsonified.
    """

    @pytest.fixture(autouse=True)
    def _cleanup_training(self):
        """Make sure any lingering worker is drained before the next test."""
        yield
        if training_state.is_training:
            training_state.request_cancel()
            # Best-effort drain; the daemon thread dies with the process anyway.
            for _ in range(60):
                if not training_state.is_training:
                    break
                time.sleep(0.05)

    def test_full_train_predict_cycle(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({
                "epochs": 2,
                "batchSize": 16,
                "useSmallData": True,
                "useGpu": False,
                "hiddenU": 16,
                "hiddenV": 8,
                "hiddenW": 8,
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert json.loads(response.data)["message"] == "Training started"

        status = _wait_for_training(client)
        assert status["has_results"] is True

        results = client.get("/api/results")
        assert results.status_code == 200
        body = json.loads(results.data)
        for key in ("training", "testing", "history", "config"):
            assert key in body
        assert body["config"]["batchSize"] == 16
        assert 0 <= body["training"]["accuracy"] <= 100
        assert 0 <= body["testing"]["accuracy"] <= 100
        assert len(body["history"]["loss_history"]) == 2

        # Synthetic 28x28 "digit" image (a centred blob). We don't assert the
        # prediction *value* -- the toy net isn't accurate after 2 epochs -- just
        # that the pipeline returns a well-formed response.
        pixels = np.zeros((28, 28), dtype=np.uint8)
        pixels[10:18, 10:18] = 255
        image_b64 = _png_base64(pixels)

        predict = client.post(
            "/api/predict",
            data=json.dumps({"image": image_b64}),
            content_type="application/json",
        )
        assert predict.status_code == 200
        pred_body = json.loads(predict.data)
        assert "prediction" in pred_body
        assert "class_index" in pred_body
        assert "probabilities" in pred_body
        probs = pred_body["probabilities"]
        assert len(probs) == 5
        assert abs(sum(probs.values()) - 1.0) < 1e-4

    def test_predict_rejects_invalid_image_payload(self, client):
        # Train first so the predict handler gets past the 'no model' gate.
        client.post(
            "/api/train",
            data=json.dumps({
                "epochs": 1,
                "batchSize": 32,
                "useSmallData": True,
                "useGpu": False,
                "hiddenU": 8,
                "hiddenV": 8,
                "hiddenW": 8,
            }),
            content_type="application/json",
        )
        _wait_for_training(client)

        # Valid base64, but not an image; must surface as a 400, not a 500.
        garbage = (
            "data:image/png;base64,"
            + base64.b64encode(b"not an image").decode("ascii")
        )
        response = client.post(
            "/api/predict",
            data=json.dumps({"image": garbage}),
            content_type="application/json",
        )
        assert response.status_code == 400
