"""Streaming ingestion of newline-delimited JSON traces."""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tracebench.models import Trace
from tracebench.storage import connect_database, insert_trace

ErrorReporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Counters collected while ingesting one JSONL file."""

    records_read: int
    records_accepted: int
    invalid_records: int
    duplicates_skipped: int
    records_stored: int


def ingest_file(
    input_path: Path,
    database_path: Path,
    error_reporter: ErrorReporter | None = None,
) -> IngestionSummary:
    """Validate and persist traces from a JSONL file."""
    records_read = 0
    records_accepted = 0
    invalid_records = 0
    duplicates_skipped = 0
    records_stored = 0

    connection = connect_database(database_path)
    try:
        with connection, input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                records_read += 1
                try:
                    payload: Any = json.loads(
                        line,
                        parse_constant=_reject_non_finite_constant,
                        parse_float=_parse_finite_float,
                    )
                except ValueError as error:
                    invalid_records += 1
                    message = (
                        error.msg
                        if isinstance(error, json.JSONDecodeError)
                        else str(error)
                    )
                    _report(
                        error_reporter,
                        f"{input_path}: line {line_number}: malformed JSON: {message}",
                    )
                    continue

                if not isinstance(payload, dict):
                    invalid_records += 1
                    _report(
                        error_reporter,
                        f"{input_path}: line {line_number}: "
                        "schema error: record must be a JSON object",
                    )
                    continue

                try:
                    trace = Trace.model_validate(payload)
                except ValidationError as error:
                    invalid_records += 1
                    _report(
                        error_reporter,
                        f"{input_path}: line {line_number}: "
                        f"schema error: {_format_validation_error(error)}",
                    )
                    continue

                records_accepted += 1
                if insert_trace(connection, trace):
                    records_stored += 1
                else:
                    duplicates_skipped += 1
    finally:
        connection.close()

    return IngestionSummary(
        records_read=records_read,
        records_accepted=records_accepted,
        invalid_records=invalid_records,
        duplicates_skipped=duplicates_skipped,
        records_stored=records_stored,
    )


def _report(error_reporter: ErrorReporter | None, message: str) -> None:
    if error_reporter is not None:
        error_reporter(message)


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"number {value} is outside the finite JSON number range")
    return parsed


def _format_validation_error(error: ValidationError) -> str:
    messages: list[str] = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in detail["loc"])
        prefix = f"{location}: " if location else ""
        messages.append(f"{prefix}{detail['msg']}")
    return "; ".join(messages)
