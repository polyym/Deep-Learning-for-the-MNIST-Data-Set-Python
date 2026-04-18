"""Module-level Flask extensions.

Kept in their own module so route decorators can import ``limiter`` without
creating a circular dependency with :mod:`mnist_ann.app`.
"""

from __future__ import annotations

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)
