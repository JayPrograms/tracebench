"""SQLite persistence for traces."""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tracebench.models import Trace

DATABASE_PATH_ENV = "TRACEBENCH_DB_PATH"
DEFAULT_DATABASE_PATH = Path(".tracebench") / "tracebench.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY CHECK (length(trim(trace_id)) > 0),
    timestamp TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (length(trim(task_type)) > 0),
    prompt TEXT NOT NULL CHECK (length(trim(prompt)) > 0),
    response TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_traces_timestamp_trace_id
ON traces (timestamp DESC, trace_id ASC);
"""


def resolve_database_path() -> Path:
    """Resolve the configured database path against the current directory."""
    configured_path = os.environ.get(DATABASE_PATH_ENV)
    path = (
        Path(configured_path)
        if configured_path and configured_path.strip()
        else DEFAULT_DATABASE_PATH
    )
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a database connection and ensure its schema exists."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.row_factory = sqlite3.Row
        connection.executescript(SCHEMA)
    except BaseException:
        try:
            connection.close()
        except BaseException:
            pass
        raise
    return connection


def timestamp_to_text(timestamp: datetime) -> str:
    """Convert an aware timestamp to canonical UTC text."""
    return (
        timestamp.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def insert_trace(connection: sqlite3.Connection, trace: Trace) -> bool:
    """Insert a trace, returning whether a new row was stored."""
    cursor = connection.execute(
        """
        INSERT INTO traces (
            trace_id,
            timestamp,
            task_type,
            prompt,
            response,
            context_json,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id) DO NOTHING
        """,
        (
            trace.trace_id,
            timestamp_to_text(trace.timestamp),
            trace.task_type,
            trace.prompt,
            trace.response,
            _encode_object(trace.context),
            _encode_object(trace.metadata),
        ),
    )
    return cursor.rowcount == 1


def list_traces(connection: sqlite3.Connection) -> list[Trace]:
    """Return all stored traces in deterministic newest-first order."""
    rows = connection.execute(
        """
        SELECT
            trace_id,
            timestamp,
            task_type,
            prompt,
            response,
            context_json,
            metadata_json
        FROM traces
        ORDER BY timestamp DESC, trace_id ASC
        """
    ).fetchall()
    return [
        Trace.model_validate(
            {
                "trace_id": row["trace_id"],
                "timestamp": row["timestamp"],
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "context": _decode_object(row["context_json"]),
                "metadata": _decode_object(row["metadata_json"]),
            }
        )
        for row in rows
    ]


def _encode_object(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored trace data is not a JSON object")
    return cast(dict[str, Any], decoded)
