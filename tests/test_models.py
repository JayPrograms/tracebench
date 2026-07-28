"""Tests for validated trace models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tracebench.models import (
    EvalCase,
    EvalDataset,
    Priority,
    ReviewStatus,
    ScorerConfig,
    Trace,
)


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


def valid_eval_case_data() -> dict[str, object]:
    """Return a valid deterministic evaluation case payload."""
    return {
        "eval_id": "eval-001",
        "dataset_id": "dataset-001",
        "source_trace_id": "trace-001",
        "source_timestamp": "2026-07-28T13:59:00Z",
        "source_task_type": "question-answering",
        "source_response": None,
        "source_metadata": {"request_id": "request-001"},
        "input": "What is the capital of Canada?",
        "context": {"locale": "en-CA"},
        "evaluation_mode": "deterministic",
        "scorers": [{"name": "exact_match"}],
        "created_at": "2026-07-28T14:00:00Z",
    }


def test_eval_dataset_normalizes_selector_fields() -> None:
    """Dataset names and versions are normalized for stable references."""
    dataset = EvalDataset.model_validate(
        {
            "dataset_id": "dataset-001",
            "name": " support-eval ",
            "version": " 0.1 ",
            "created_at": "2026-07-28T14:00:00Z",
        }
    )

    assert dataset.name == "support-eval"
    assert dataset.version == "0.1"
    assert dataset.description == ""


@pytest.mark.parametrize("field", ["name", "version"])
def test_eval_dataset_rejects_colons(field: str) -> None:
    """Selector components cannot contain the selector separator."""
    payload: dict[str, object] = {
        "dataset_id": "dataset-001",
        "name": "support-eval",
        "version": "0.1",
        "created_at": "2026-07-28T14:00:00Z",
    }
    payload[field] = "invalid:value"

    with pytest.raises(ValidationError, match="must not contain ':'"):
        EvalDataset.model_validate(payload)


def test_eval_case_defaults_priority_and_review_status() -> None:
    """New cases begin at medium priority in draft review state."""
    case = EvalCase.model_validate(valid_eval_case_data())

    assert case.priority is Priority.MEDIUM
    assert case.review_status is ReviewStatus.DRAFT


def test_deterministic_mode_requires_a_scorer_configuration() -> None:
    """Deterministic evaluation has at least one configured scorer."""
    payload = {**valid_eval_case_data(), "scorers": []}

    with pytest.raises(ValidationError, match="requires at least one scorer"):
        EvalCase.model_validate(payload)


def test_reference_mode_requires_nonblank_reference_answer() -> None:
    """Reference evaluation cannot omit its expected answer."""
    payload = {
        **valid_eval_case_data(),
        "evaluation_mode": "reference",
        "scorers": [],
        "reference_answer": "   ",
    }

    with pytest.raises(ValidationError, match="must not be blank"):
        EvalCase.model_validate(payload)


def test_rubric_mode_requires_criteria() -> None:
    """Rubric evaluation cannot have an empty criteria list."""
    payload = {
        **valid_eval_case_data(),
        "evaluation_mode": "rubric",
        "scorers": [],
        "rubric": [],
    }

    with pytest.raises(ValidationError, match="requires at least one rubric"):
        EvalCase.model_validate(payload)


@pytest.mark.parametrize("rubric", [["valid", "  "], ["same", " same "]])
def test_rubric_criteria_reject_blanks_and_duplicates(rubric: list[str]) -> None:
    """Rubric criteria remain meaningful and unique after normalization."""
    payload = {
        **valid_eval_case_data(),
        "evaluation_mode": "rubric",
        "scorers": [],
        "rubric": rubric,
    }

    with pytest.raises(ValidationError):
        EvalCase.model_validate(payload)


@pytest.mark.parametrize(
    ("mode", "field", "value", "message"),
    [
        ("deterministic", "reference_answer", "Ottawa", "forbids a reference"),
        ("deterministic", "source_response", "Ottawa", "forbids a source"),
        ("deterministic", "rubric", ["Correct"], "forbids rubric"),
        ("reference", "scorers", [{"name": "exact"}], "forbids deterministic"),
        ("reference", "rubric", ["Correct"], "forbids rubric"),
        ("rubric", "reference_answer", "Ottawa", "forbids a reference"),
        ("rubric", "source_response", "Ottawa", "forbids a source"),
        ("rubric", "scorers", [{"name": "exact"}], "forbids deterministic"),
    ],
)
def test_eval_modes_reject_fields_from_other_modes(
    mode: str,
    field: str,
    value: object,
    message: str,
) -> None:
    """Each mode rejects configuration owned by another evaluation strategy."""
    payload = {
        **valid_eval_case_data(),
        "evaluation_mode": mode,
        "scorers": [{"name": "exact"}] if mode == "deterministic" else [],
        "reference_answer": "Ottawa" if mode == "reference" else None,
        "rubric": ["Correct"] if mode == "rubric" else [],
        field: value,
    }

    with pytest.raises(ValidationError, match=message):
        EvalCase.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rubric", [None]),
        ("scorers", [1]),
        ("scorers", [{"config": {}}]),
    ],
)
def test_eval_case_rejects_invalid_array_item_shapes(field: str, value: object) -> None:
    """Null rubric items and scalar scorer entries fail model validation."""
    payload = {**valid_eval_case_data(), field: value}

    with pytest.raises(ValidationError):
        EvalCase.model_validate(payload)


@pytest.mark.parametrize("field", ["context", "source_metadata"])
def test_eval_case_requires_json_object_snapshots(field: str) -> None:
    """Context and source metadata remain finite JSON objects."""
    with pytest.raises(ValidationError):
        EvalCase.model_validate(
            {**valid_eval_case_data(), field: ["not", "an", "object"]}
        )
    with pytest.raises(ValidationError, match="finite JSON"):
        EvalCase.model_validate(
            {**valid_eval_case_data(), field: {"score": float("nan")}}
        )


def test_scorer_configuration_is_strict_json() -> None:
    """Scorer names and configuration values must be usable JSON."""
    with pytest.raises(ValidationError, match="must not be blank"):
        ScorerConfig.model_validate({"name": "  "})
    with pytest.raises(ValidationError, match="finite JSON"):
        ScorerConfig.model_validate({"name": "numeric", "config": {"x": float("nan")}})
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScorerConfig.model_validate({"name": "exact", "unknown": True})


def test_eval_case_accepts_critical_and_rejected_values() -> None:
    """The full priority and review-status vocabularies are supported."""
    case = EvalCase.model_validate(
        {
            **valid_eval_case_data(),
            "priority": "critical",
            "review_status": "rejected",
        }
    )

    assert case.priority is Priority.CRITICAL
    assert case.review_status is ReviewStatus.REJECTED
