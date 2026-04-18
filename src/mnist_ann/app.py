"""Flask application factory.

Usage:
    # Programmatic
    from mnist_ann.app import create_app
    app = create_app()

    # Gunicorn
    gunicorn "mnist_ann.app:create_app()"
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import (
    ALLOWED_ORIGINS,
    MAX_CONTENT_LENGTH,
    RATE_LIMIT_ENABLED,
    STATIC_DIR,
    configure_logging,
)
from .extensions import limiter
from .routes import bp as api_bp

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Construct and wire a Flask application."""
    configure_logging()

    app = Flask(__name__, static_folder=str(STATIC_DIR))
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    _configure_cors(app)
    _configure_rate_limiter(app)
    _register_security_headers(app)
    _register_error_handlers(app)

    app.register_blueprint(api_bp)

    return app


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------
def _configure_cors(app: Flask) -> None:
    """Enable CORS for ``ALLOWED_ORIGINS`` (``*`` or a comma-separated list)."""
    if ALLOWED_ORIGINS == "*":
        CORS(app)
        logger.warning("CORS is set to allow all origins. Consider restricting in production.")
    else:
        origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",")]
        CORS(app, origins=origins)
        logger.info(f"CORS configured for origins: {origins}")


def _configure_rate_limiter(app: Flask) -> None:
    """Bind the module-level :data:`limiter` to ``app`` and honour ``RATE_LIMIT_ENABLED``."""
    limiter.enabled = RATE_LIMIT_ENABLED
    limiter.init_app(app)


def _register_security_headers(app: Flask) -> None:
    """Attach conservative defensive headers to every response (errors included)."""
    @app.after_request
    def _add_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def _register_error_handlers(app: Flask) -> None:
    """Convert the common HTTP errors (400/404/413/429/500) into JSON payloads."""
    @app.errorhandler(400)
    def _bad_request(e):
        return jsonify({"error": "Bad request"}), 400

    @app.errorhandler(404)
    def _not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(413)
    def _request_too_large(e):
        return jsonify({"error": "Request too large"}), 413

    @app.errorhandler(429)
    def _rate_limit_exceeded(e):
        logger.warning(f"Rate limit exceeded: {request.remote_addr}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    @app.errorhandler(500)
    def _internal_error(e):
        logger.exception("Internal server error")
        return jsonify({"error": "Internal server error"}), 500
