"""Request-input validation helpers used by the training/predict endpoints."""

from __future__ import annotations

from functools import wraps
from typing import Any
from collections.abc import Iterable

from flask import jsonify, request


class ValidationError(Exception):
    """Raised when a request parameter fails a bounds or type check."""


def validate_int(
    value: Any,
    name: str,
    min_val: int | None = None,
    max_val: int | None = None,
    default: int | None = None,
) -> int:
    """Coerce ``value`` to int and range-check it.

    Args:
        value: The raw request value (often a JSON number or string).
        name: Human-readable parameter name, used in error messages.
        min_val: Optional inclusive lower bound.
        max_val: Optional inclusive upper bound.
        default: Returned verbatim if ``value`` is ``None``. If omitted and
            ``value`` is ``None``, the call fails.

    Returns:
        The coerced and bounds-checked integer.

    Raises:
        ValidationError: If ``value`` is ``None`` with no ``default``, can't
            be parsed as an int, or falls outside ``[min_val, max_val]``.
    """
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{name} must be a valid integer") from exc

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_float(
    value: Any,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    default: float | None = None,
) -> float:
    """Coerce ``value`` to float and range-check it.

    Args:
        value: The raw request value (often a JSON number or string).
        name: Human-readable parameter name, used in error messages.
        min_val: Optional inclusive lower bound.
        max_val: Optional inclusive upper bound.
        default: Returned verbatim if ``value`` is ``None``. If omitted and
            ``value`` is ``None``, the call fails.

    Returns:
        The coerced and bounds-checked float.

    Raises:
        ValidationError: If ``value`` is ``None`` with no ``default``, can't
            be parsed as a float, or falls outside ``[min_val, max_val]``.
    """
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{name} must be a valid number") from exc

    if min_val is not None and result < min_val:
        raise ValidationError(f"{name} must be at least {min_val}")
    if max_val is not None and result > max_val:
        raise ValidationError(f"{name} must be at most {max_val}")

    return result


def validate_choice(
    value: Any,
    name: str,
    choices: Iterable[Any],
    default: Any | None = None,
) -> Any:
    """Ensure ``value`` is a member of ``choices``.

    Args:
        value: The raw request value.
        name: Human-readable parameter name, used in error messages.
        choices: Allowed values, compared via ``in`` (equality).
        default: Returned verbatim if ``value`` is ``None``. If omitted and
            ``value`` is ``None``, the call fails.

    Returns:
        ``value`` itself (not a copy) when membership holds.

    Raises:
        ValidationError: If ``value`` is ``None`` with no ``default`` or is
            not a member of ``choices``.
    """
    if value is None:
        if default is not None:
            return default
        raise ValidationError(f"{name} is required")

    choices = list(choices)
    if value not in choices:
        allowed = ", ".join(map(str, choices))
        raise ValidationError(f"{name} must be one of: {allowed}")

    return value


def validate_digit(value: Any, name: str, default: int | None = None) -> int:
    """Convenience wrapper: an int in ``[0, 9]``.

    Args:
        value: The raw request value.
        name: Human-readable parameter name, used in error messages.
        default: Returned verbatim if ``value`` is ``None``.

    Returns:
        The coerced integer, guaranteed in ``[0, 9]``.

    Raises:
        ValidationError: Same cases as :func:`validate_int`.
    """
    return validate_int(value, name, min_val=0, max_val=9, default=default)


def require_json(f):
    """Reject requests whose body isn't a JSON object.

    Guards against missing ``Content-Type: application/json`` headers as
    well as valid-JSON-but-non-object bodies (``null``, arrays, strings,
    numbers) which would otherwise crash ``data.get(...)`` calls inside
    the route handler.

    Args:
        f: The Flask view function to wrap.

    Returns:
        A wrapper that short-circuits with a ``400`` JSON response if the
        request body isn't a JSON object, and otherwise calls ``f``.
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
