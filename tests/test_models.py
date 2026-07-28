"""Tests for validated trace models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tracebench.models import Trace


def valid_trace_data() -> dict[str, object]:
    """Return a minimal valid trace payload."""
    return {
        "trace_id": "trace-001",
        "timestamp": "2026-07-28T14:00:00Z",
        "task_type": "question-answering",
        "prompt": "What is the capital of Canada?",
    }


def test_valid_trace_is_accepted() -> None:
    """A complete valid trace passes validation."""
    trace = Trace.model_validate(
        {
            **valid_trace_data(),
            "response": "Ottawa",
            "context": {"locale": "en-CA"},
            "metadata": {"source": "test"},
        }
    )

    assert trace.timestamp == datetime(2026, 7, 28, 14, tzinfo=UTC)
    assert trace.response == "Ottawa"


@pytest.mark.parametrize("field", ["trace_id", "task_type", "prompt"])
def test_required_text_fields_reject_blank_values(field: str) -> None:
    """Required text fields cannot contain only whitespace."""
    payload = valid_trace_data()
    payload[field] = "   "

    with pytest.raises(ValidationError, match="must not be blank"):
        Trace.model_validate(payload)


def test_timezone_naive_timestamp_is_rejected() -> None:
    """A timestamp without a timezone offset is invalid."""
    payload = valid_trace_data()
    payload["timestamp"] = "2026-07-28T14:00:00"

    with pytest.raises(ValidationError, match="timezone"):
        Trace.model_validate(payload)


def test_missing_optional_fields_receive_independent_defaults() -> None:
    """Optional fields default without sharing mutable dictionaries."""
    first = Trace.model_validate(valid_trace_data())
    second = Trace.model_validate({**valid_trace_data(), "trace_id": "trace-002"})

    assert first.response is None
    assert first.context == {}
    assert first.metadata == {}
    first.context["value"] = 1
    assert second.context == {}


@pytest.mark.parametrize("field", ["context", "metadata"])
def test_dictionary_fields_reject_other_json_types(field: str) -> None:
    """Context and metadata must be JSON objects."""
    payload = valid_trace_data()
    payload[field] = []

    with pytest.raises(ValidationError):
        Trace.model_validate(payload)


def test_unknown_fields_are_rejected() -> None:
    """Trace records cannot silently include unknown schema fields."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Trace.model_validate({**valid_trace_data(), "unknown": True})
