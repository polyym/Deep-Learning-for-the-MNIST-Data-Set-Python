"""Tests for the Flask application factory and wiring.

The other test modules already exercise routing through the test client, but
these cover the factory-level wiring directly: returned object type, static
folder, security headers, and JSON error responses.
"""

from __future__ import annotations

from flask import Flask

from mnist_ann.app import create_app


class TestCreateApp:
    """``create_app()`` returns an independent, ready-to-serve Flask instance."""

    def test_returns_flask_app(self):
        app = create_app()
        assert isinstance(app, Flask)

    def test_independent_calls_produce_independent_apps(self):
        assert create_app() is not create_app()

    def test_static_folder_points_at_repo_static(self, app):
        # The fixture app has already been through create_app; static_folder
        # should resolve to a readable directory containing index.html.
        from pathlib import Path

        static_root = Path(app.static_folder)
        assert (static_root / "index.html").is_file()


class TestSecurityHeaders:
    """Every response (including errors) carries the defensive headers."""

    def test_headers_on_health_response(self, client):
        resp = client.get("/api/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


class TestErrorHandlerShape:
    """Global handlers return JSON + correct status for bad input / missing routes."""

    def test_404_returns_json_error(self, client):
        resp = client.get("/definitely-not-a-route")
        assert resp.status_code == 404
        assert resp.is_json
        assert "error" in resp.get_json()

    def test_non_dict_body_rejected(self, client):
        # Tightened require_json should refuse JSON arrays/scalars.
        import json

        resp = client.post(
            "/api/train",
            data=json.dumps([1, 2, 3]),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "json object" in resp.get_json()["error"].lower()

    def test_null_body_rejected(self, client):
        import json

        resp = client.post(
            "/api/train",
            data=json.dumps(None),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "json object" in resp.get_json()["error"].lower()
