"""Tests for JSONL ingestion and SQLite persistence."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tracebench.ingestion import IngestionSummary, ingest_file
from tracebench.models import Trace
from tracebench.storage import connect_database, insert_trace, list_traces


def trace_record(
    trace_id: str,
    *,
    timestamp: str = "2026-07-28T14:00:00Z",
    prompt: str = "A valid prompt",
) -> dict[str, object]:
    """Build a valid JSON-compatible trace record."""
    return {
        "trace_id": trace_id,
        "timestamp": timestamp,
        "task_type": "test-task",
        "prompt": prompt,
    }


def write_lines(path: Path, lines: list[str]) -> None:
    """Write physical JSONL records with a trailing newline."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_valid_records_are_stored_and_ordered_newest_first(tmp_path: Path) -> None:
    """Valid records persist completely in timestamp order."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    records = [
        {
            **trace_record("older"),
            "response": "response",
            "context": {"nested": {"language": "français"}},
            "metadata": {"sequence": 1},
        },
        trace_record("newer", timestamp="2026-07-28T11:00:00-04:00"),
    ]
    write_lines(input_path, [json.dumps(record) for record in records])

    summary = ingest_file(input_path, database_path)

    assert summary == IngestionSummary(2, 2, 0, 0, 2)
    with closing(connect_database(database_path)) as connection:
        traces = list_traces(connection)
    assert [trace.trace_id for trace in traces] == ["newer", "older"]
    assert traces[1].context == {"nested": {"language": "français"}}
    assert traces[1].metadata == {"sequence": 1}
    assert traces[1].response == "response"
    assert traces[0].response is None
    assert traces[0].context == {}
    assert traces[0].metadata == {}


def test_subsecond_timestamps_are_ordered_newest_first(tmp_path: Path) -> None:
    """Fixed-width UTC timestamps preserve subsecond chronological order."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    records = [
        trace_record("whole", timestamp="2026-07-20T14:00:00Z"),
        trace_record("milliseconds", timestamp="2026-07-20T10:00:00.500-04:00"),
        trace_record("microseconds", timestamp="2026-07-20T14:00:00.500001Z"),
    ]
    write_lines(input_path, [json.dumps(record) for record in records])

    summary = ingest_file(input_path, database_path)

    assert summary == IngestionSummary(3, 3, 0, 0, 3)
    with closing(connect_database(database_path)) as connection:
        traces = list_traces(connection)
        stored_timestamps = {
            row["trace_id"]: row["timestamp"]
            for row in connection.execute("SELECT trace_id, timestamp FROM traces")
        }
    assert [trace.trace_id for trace in traces] == [
        "microseconds",
        "milliseconds",
        "whole",
    ]
    assert stored_timestamps == {
        "whole": "2026-07-20T14:00:00.000000Z",
        "milliseconds": "2026-07-20T14:00:00.500000Z",
        "microseconds": "2026-07-20T14:00:00.500001Z",
    }


def test_malformed_json_is_reported_and_processing_continues(
    tmp_path: Path,
) -> None:
    """Malformed JSON is counted while subsequent records are stored."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    write_lines(
        input_path,
        ["{not-json", json.dumps(trace_record("valid-after-error"))],
    )
    errors: list[str] = []

    summary = ingest_file(input_path, database_path, errors.append)

    assert summary == IngestionSummary(2, 1, 1, 0, 1)
    assert len(errors) == 1
    assert "line 1" in errors[0]
    assert "malformed JSON" in errors[0]


@pytest.mark.parametrize(
    ("invalid_fragment", "case_name"),
    [
        ('"context":{"value":NaN}', "NaN"),
        ('"context":{"value":Infinity}', "Infinity"),
        ('"context":{"value":-Infinity}', "-Infinity"),
        ('"context":{"value":1e400}', "numeric overflow"),
        ('"context":{"nested":{"value":NaN}}', "nested context"),
        ('"metadata":{"nested":[Infinity]}', "nested metadata"),
    ],
    ids=lambda value: value if "{" not in value else None,
)
def test_non_finite_numbers_are_invalid_and_processing_continues(
    tmp_path: Path,
    invalid_fragment: str,
    case_name: str,
) -> None:
    """Non-finite numbers are skipped without blocking later valid records."""
    input_path = tmp_path / f"{case_name}.jsonl"
    database_path = tmp_path / f"{case_name}.sqlite3"
    invalid_line = (
        '{"trace_id":"invalid","timestamp":"2026-07-28T14:00:00Z",'
        '"task_type":"test-task","prompt":"invalid",'
        f"{invalid_fragment}}}"
    )
    write_lines(
        input_path,
        [invalid_line, json.dumps(trace_record("valid-after-error"))],
    )
    errors: list[str] = []

    summary = ingest_file(input_path, database_path, errors.append)

    assert summary == IngestionSummary(2, 1, 1, 0, 1)
    assert len(errors) == 1
    assert "line 1" in errors[0]
    assert "malformed JSON" in errors[0]
    with closing(connect_database(database_path)) as connection:
        traces = list_traces(connection)
    assert [trace.trace_id for trace in traces] == ["valid-after-error"]


def test_storage_refuses_non_finite_json_values(tmp_path: Path) -> None:
    """Persistence rejects non-finite values even outside file ingestion."""
    database_path = tmp_path / "tracebench.sqlite3"
    trace = Trace.model_validate(
        {
            **trace_record("non-finite"),
            "context": {"nested": {"value": float("nan")}},
        }
    )

    with closing(connect_database(database_path)) as connection:
        with pytest.raises(ValueError, match="Out of range float values"):
            insert_trace(connection, trace)
        stored_count = connection.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    assert stored_count == 0


def test_schema_failure_reports_field_and_line_number(tmp_path: Path) -> None:
    """Schema failures include both their record line and failing field."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    invalid_record = trace_record("invalid")
    invalid_record.pop("task_type")
    write_lines(input_path, [json.dumps(invalid_record)])
    errors: list[str] = []

    summary = ingest_file(input_path, database_path, errors.append)

    assert summary == IngestionSummary(1, 0, 1, 0, 0)
    assert "line 1" in errors[0]
    assert "task_type" in errors[0]
    assert "schema error" in errors[0]


def test_blank_prompt_is_invalid(tmp_path: Path) -> None:
    """Whitespace-only prompts are reported as schema failures."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    write_lines(input_path, [json.dumps(trace_record("blank", prompt="  "))])
    errors: list[str] = []

    summary = ingest_file(input_path, database_path, errors.append)

    assert summary.invalid_records == 1
    assert summary.records_stored == 0
    assert "prompt" in errors[0]
    assert "line 1" in errors[0]


def test_timezone_naive_timestamp_is_invalid(tmp_path: Path) -> None:
    """Naive timestamps are reported and skipped during ingestion."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    record = trace_record("naive", timestamp="2026-07-28T14:00:00")
    write_lines(input_path, [json.dumps(record)])
    errors: list[str] = []

    summary = ingest_file(input_path, database_path, errors.append)

    assert summary == IngestionSummary(1, 0, 1, 0, 0)
    assert "timestamp" in errors[0]
    assert "line 1" in errors[0]


def test_duplicate_trace_ids_within_file_are_skipped(tmp_path: Path) -> None:
    """The first occurrence of an ID wins within one input file."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    write_lines(
        input_path,
        [
            json.dumps(trace_record("duplicate", prompt="first")),
            json.dumps(trace_record("duplicate", prompt="second")),
        ],
    )

    summary = ingest_file(input_path, database_path)

    assert summary == IngestionSummary(2, 2, 0, 1, 1)
    with closing(connect_database(database_path)) as connection:
        traces = list_traces(connection)
    assert traces[0].prompt == "first"


def test_trace_ids_already_in_database_are_skipped(tmp_path: Path) -> None:
    """Stored IDs remain unchanged when later inputs repeat them."""
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    database_path = tmp_path / "tracebench.sqlite3"
    write_lines(first_path, [json.dumps(trace_record("stored", prompt="original"))])
    write_lines(second_path, [json.dumps(trace_record("stored", prompt="replacement"))])
    ingest_file(first_path, database_path)

    summary = ingest_file(second_path, database_path)

    assert summary == IngestionSummary(1, 1, 0, 1, 0)
    with closing(connect_database(database_path)) as connection:
        traces = list_traces(connection)
    assert traces[0].prompt == "original"


def test_schema_initialization_failure_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema error preserves its exception and closes the opened database."""
    database_path = tmp_path / "incompatible.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE traces (trace_id TEXT PRIMARY KEY)")

    real_connect = sqlite3.connect
    opened_connections: list[sqlite3.Connection] = []

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        opened_connections.append(connection)
        return connection

    monkeypatch.setattr("tracebench.storage.sqlite3.connect", tracked_connect)

    with pytest.raises(sqlite3.OperationalError, match="no such column: timestamp"):
        connect_database(database_path)

    assert len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened_connections[0].execute("SELECT 1")
