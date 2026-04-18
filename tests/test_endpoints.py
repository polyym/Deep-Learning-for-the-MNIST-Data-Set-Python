"""Tests for the HTTP endpoints."""

from __future__ import annotations

import json

from mnist_ann.state import training_state


class TestHealthEndpoint:
    """``GET /api/health``, liveness + backend + state flags."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["is_training"] is False
        assert data["has_model"] is False
        assert data["has_results"] is False
        assert "gpu_available" in data


class TestStatusEndpoint:
    """``GET /api/status``, training flag + recent progress tail."""

    def test_status_initial(self, client):
        response = client.get("/api/status")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["is_training"] is False
        assert data["progress"] == []
        assert data["has_results"] is False


class TestTrainEndpoint:
    """``POST /api/train``, validation, dispatch, concurrency guard."""

    def test_train_requires_json(self, client):
        response = client.post("/api/train", data="not json")
        assert response.status_code == 400

    def test_train_validation_error_epochs(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({"epochs": -1}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "epochs" in json.loads(response.data)["error"].lower()

    def test_train_validation_error_learning_rate(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({"learningRate": "invalid"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "learningrate" in json.loads(response.data)["error"].lower()

    def test_train_validation_error_backprop_method(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({"backpropMethod": "invalid"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "backpropmethod" in json.loads(response.data)["error"].lower()

    def test_train_validation_error_duplicate_digits(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({"digitA": 1, "digitB": 1, "digitC": 2, "digitD": 3}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "unique" in json.loads(response.data)["error"].lower()

    def test_train_validation_error_invalid_digit(self, client):
        response = client.post(
            "/api/train",
            data=json.dumps({"digitA": 15}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_train_starts_with_defaults(self, client, mock_train_thread):
        response = client.post(
            "/api/train",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert json.loads(response.data)["message"] == "Training started"
        mock_train_thread.return_value.start.assert_called_once()

    def test_train_rejects_when_training(self, client, mock_train_thread):
        # First call flips is_training to True (thread is mocked out)
        client.post("/api/train", data=json.dumps({}), content_type="application/json")
        # Second call must be rejected
        response = client.post(
            "/api/train",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "already in progress" in json.loads(response.data)["error"].lower()


class TestCancelEndpoint:
    """``POST /api/cancel``, signals an active run, 400 when idle."""

    def test_cancel_without_training(self, client):
        response = client.post("/api/cancel")
        assert response.status_code == 400
        assert "no training in progress" in json.loads(response.data)["error"].lower()

    def test_cancel_while_training(self, client):
        training_state.is_training = True
        try:
            response = client.post("/api/cancel")
            assert response.status_code == 200
            assert json.loads(response.data)["message"] == "Cancellation requested"
            assert training_state.should_cancel() is True
        finally:
            training_state.is_training = False
            training_state.reset()


class TestResultsEndpoint:
    """``GET /api/results``, returns training/testing metrics once published."""

    def test_results_not_available(self, client):
        response = client.get("/api/results")
        assert response.status_code == 404
        assert "no results" in json.loads(response.data)["error"].lower()


class TestPredictEndpoint:
    """``POST /api/predict``, input validation, gated on a trained model."""

    def test_predict_no_model(self, client):
        response = client.post(
            "/api/predict",
            data=json.dumps({"image": "base64data"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "no trained model" in json.loads(response.data)["error"].lower()

    def test_predict_requires_json(self, client):
        response = client.post("/api/predict", data="not json")
        assert response.status_code == 400

    def test_predict_requires_image(self, client):
        training_state.network = True  # mock: bypass the "no model" branch
        response = client.post(
            "/api/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400
        assert "no image" in json.loads(response.data)["error"].lower()


class TestSampleImagesEndpoint:
    """``GET /api/sample_images``, requires a trained model."""

    def test_sample_images_no_model(self, client):
        response = client.get("/api/sample_images")
        assert response.status_code == 400
        assert "no trained model" in json.loads(response.data)["error"].lower()
