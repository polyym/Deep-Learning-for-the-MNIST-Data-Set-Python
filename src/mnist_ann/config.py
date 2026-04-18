"""Environment-variable configuration, filesystem paths, and logging setup.

All runtime settings that used to live at the top of the old monolithic
``app.py`` now come from here. Importing this module has no side effects;
call :func:`configure_logging` once at process start to initialise logging.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent              # e.g. src/mnist_ann/
# Only valid when running from a source checkout or editable install. For a
# regular ``pip install .`` the package lives in site-packages and this would
# point at site-packages/..  -- use the resolver below which falls back.
_SOURCE_ROOT_CANDIDATE = _PACKAGE_DIR.parent.parent


def _resolve_resource_dir(env_var: str, dirname: str) -> Path:
    """Locate a sibling resource directory (``static`` / ``data``).

    Resolution order:
      1. Explicit override via environment variable (``STATIC_DIR``, ``DATA_DIR``)
      2. ``<repo>/{dirname}`` when running from source or an editable install
      3. ``<cwd>/{dirname}`` (works for ``pip install .`` deployments that
         run from the repo as their working directory, e.g. Render)
    """
    override = os.environ.get(env_var)
    if override:
        return Path(override).resolve()

    candidate = _SOURCE_ROOT_CANDIDATE / dirname
    if candidate.is_dir():
        return candidate.resolve()

    return (Path.cwd() / dirname).resolve()


PROJECT_ROOT = _SOURCE_ROOT_CANDIDATE  # kept for backwards-compat exports
STATIC_DIR: Path = _resolve_resource_dir("STATIC_DIR", "static")
DEFAULT_DATA_DIR: Path = _resolve_resource_dir("DATA_DIR", "data")

# ---------------------------------------------------------------------------
# Runtime settings (env-var overridable)
# ---------------------------------------------------------------------------
DEBUG: bool = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
PORT: int = int(os.environ.get("PORT", 5000))
HOST: str = os.environ.get("HOST", "0.0.0.0")
ALLOWED_ORIGINS: str = os.environ.get("ALLOWED_ORIGINS", "*")
MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
DATA_DIR: str = str(DEFAULT_DATA_DIR)
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
RATE_LIMIT_ENABLED: bool = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"


def configure_logging() -> None:
    """Initialise root logging. Safe to call more than once."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
