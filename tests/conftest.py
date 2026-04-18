"""Shared pytest fixtures.

Sets test-friendly environment variables *before* importing the application,
then creates a fresh Flask test client per test and resets the global
training state between tests.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Must be set before importing the app so config.py reads them.
os.environ["FLASK_DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["RATE_LIMIT_ENABLED"] = "false"

from mnist_ann import routes as routes_module
from mnist_ann.app import create_app
from mnist_ann.state import training_state


@pytest.fixture
def app():
    """A Flask app instance in testing mode."""
    flask_app = create_app()
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    """Flask test client."""
    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def reset_training_state():
    """Reset the in-memory training state between tests."""
    training_state.is_training = False
    training_state.network = None
    training_state.reset()
    yield


@pytest.fixture
def mock_train_thread():
    """Prevent /api/train from spawning real training threads in tests.

    Replaces ``threading.Thread`` inside the routes module with a MagicMock,
    so the request handler still flips ``is_training`` to True but no real
    worker starts.
    """
    dummy = MagicMock()
    dummy.start = MagicMock()
    with patch.object(routes_module.threading, "Thread", return_value=dummy) as m:
        yield m
