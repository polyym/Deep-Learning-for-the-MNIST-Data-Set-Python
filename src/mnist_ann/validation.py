"""Request-input validation helpers used by the training/predict endpoints."""

from __future__ import annotations

from functools import wraps
from typing import Any, Iterable, Optional

from flask import jsonify, request


class ValidationError(Exception):
    """Raised when a request parameter fails a bounds or type check."""


def validate_int(
    value: Any,
    name: str,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    default: Optional[int] = None,
) -> int:
    """Coerce ``value`` to int and range-check it."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be a valid integer")

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_float(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Coerce ``value`` to float and range-check it."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"{name} must be a valid number")

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_choice(
    value: Any,
    name: str,
    choices: Iterable[Any],
    default: Optional[Any] = None,
) -> Any:
    """Ensure ``value`` is a member of ``choices``."""
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    choices = list(choices)
    if value not in choices:
        raise ValidationError(f"{name} must be one of: {', '.join(map(str, choices))}")

    return value


def validate_digit(value: Any, name: str, default: Optional[int] = None) -> int:
    """Convenience wrapper: an int in [0, 9]."""
    return validate_int(value, name, min_val=0, max_val=9, default=default)


def require_json(f):
    """Reject requests whose body isn't a JSON object.

    Guards against missing ``Content-Type: application/json`` headers as
    well as valid-JSON-but-non-object bodies (``null``, arrays, strings,
    numbers) which would otherwise crash ``data.get(...)`` calls inside
    the route handler.
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        return f(*args, **kwargs)

    return decorated
