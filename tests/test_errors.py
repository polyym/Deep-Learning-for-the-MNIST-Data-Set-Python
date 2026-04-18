"""Tests for Flask error handlers."""

from __future__ import annotations


class TestErrorHandlers:
    """Flask error handlers, JSON shape + status codes."""

    def test_404_error(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_request_too_large(self, app, client):
        # Tighten the limit for this test, then restore.
        original = app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)
        app.config["MAX_CONTENT_LENGTH"] = 100
        try:
            large_data = "x" * 200
            response = client.post(
                "/api/train",
                data=large_data,
                content_type="application/json",
            )
            assert response.status_code == 413
        finally:
            app.config["MAX_CONTENT_LENGTH"] = original
