"""Tests for the standalone validation helpers."""

from __future__ import annotations

import pytest

from mnist_ann.validation import (
    ValidationError,
    validate_choice,
    validate_digit,
    validate_float,
    validate_int,
)


class TestValidateInt:
    """Integer coercion, bounds, defaults, and failure cases."""

    def test_valid_int(self):
        assert validate_int(5, "test") == 5

    def test_string_int(self):
        assert validate_int("10", "test") == 10

    def test_default_value(self):
        assert validate_int(None, "test", default=42) == 42

    def test_min_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(5, "test", min_val=10)
        assert "at least 10" in str(exc_info.value)

    def test_max_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(100, "test", max_val=50)
        assert "at most 50" in str(exc_info.value)

    def test_invalid_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int("not a number", "test")
        assert "valid integer" in str(exc_info.value)

    def test_required(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_int(None, "test")
        assert "required" in str(exc_info.value)


class TestValidateFloat:
    """Float coercion, bounds, and defaults."""

    def test_valid_float(self):
        assert validate_float(3.14, "test") == 3.14

    def test_string_float(self):
        assert validate_float("2.5", "test") == 2.5

    def test_int_to_float(self):
        assert validate_float(5, "test") == 5.0

    def test_default_value(self):
        assert validate_float(None, "test", default=1.0) == 1.0

    def test_min_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_float(0.5, "test", min_val=1.0)
        assert "at least 1.0" in str(exc_info.value)

    def test_max_value(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_float(100.0, "test", max_val=50.0)
        assert "at most 50.0" in str(exc_info.value)


class TestValidateChoice:
    """Membership validation against an allowed set."""

    def test_valid_choice(self):
        assert validate_choice("cb", "method", ["cb", "uhb"]) == "cb"

    def test_default_value(self):
        assert validate_choice(None, "method", ["cb", "uhb"], default="cb") == "cb"

    def test_invalid_choice(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_choice("invalid", "method", ["cb", "uhb"])
        assert "must be one of" in str(exc_info.value)


class TestValidateDigit:
    """Digit (0-9) convenience wrapper around validate_int."""

    def test_valid_digit(self):
        for d in range(10):
            assert validate_digit(d, "digit") == d

    def test_invalid_digit_negative(self):
        with pytest.raises(ValidationError):
            validate_digit(-1, "digit")

    def test_invalid_digit_too_large(self):
        with pytest.raises(ValidationError):
            validate_digit(10, "digit")
