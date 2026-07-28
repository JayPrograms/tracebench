"""Tests for the TraceBench command-line interface."""

import json
from importlib.metadata import version as package_version
from pathlib import Path

from typer.testing import CliRunner

from tracebench.cli import app

runner = CliRunner()


def test_version_command_succeeds() -> None:
    """The version command exits successfully."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0, result.output


def test_version_command_reports_installed_version() -> None:
    """The version command prints the installed distribution version."""
    result = runner.invoke(app, ["version"])

    assert result.stdout == f"{package_version('tracebench')}\n"


def test_missing_input_file_is_a_usage_error(tmp_path: Path) -> None:
    """Ingestion rejects a path that does not exist."""
    result = runner.invoke(app, ["ingest", str(tmp_path / "missing.jsonl")])

    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_ingestion_and_listing_use_database_override(tmp_path: Path) -> None:
    """Separate CLI invocations share the configured SQLite database."""
    input_path = tmp_path / "traces.jsonl"
    database_path = tmp_path / "local.sqlite3"
    records = [
        {
            "trace_id": "older",
            "timestamp": "2026-07-28T14:00:00Z",
            "task_type": "question-answering",
            "prompt": "First prompt",
        },
        {"invalid": True},
        {
            "trace_id": "newer",
            "timestamp": "2026-07-28T15:00:00Z",
            "task_type": "summarization",
            "prompt": "Second prompt",
            "response": "Second response",
        },
        {
            "trace_id": "newer",
            "timestamp": "2026-07-28T15:00:00Z",
            "task_type": "summarization",
            "prompt": "Duplicate prompt",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}

    ingest_result = runner.invoke(
        app,
        ["ingest", str(input_path)],
        env=environment,
    )

    assert ingest_result.exit_code == 0, ingest_result.output
    assert ingest_result.stdout == (
        "Records read: 4\n"
        "Records accepted: 3\n"
        "Invalid records: 1\n"
        "Duplicates skipped: 1\n"
        "Records stored: 2\n"
    )
    assert "line 2" in ingest_result.stderr
    assert "schema error" in ingest_result.stderr
    assert database_path.exists()

    list_result = runner.invoke(app, ["traces", "list"], env=environment)

    assert list_result.exit_code == 0, list_result.output
    assert "TRACE_ID" in list_result.stdout
    assert "TIMESTAMP" in list_result.stdout
    assert "TASK_TYPE" in list_result.stdout
    assert "HAS_RESPONSE" in list_result.stdout
    assert list_result.stdout.index("newer") < list_result.stdout.index("older")
    assert "yes" in list_result.stdout
    assert "no" in list_result.stdout


def test_empty_trace_list_has_clear_message(tmp_path: Path) -> None:
    """Listing an empty database succeeds with a concise message."""
    result = runner.invoke(
        app,
        ["traces", "list"],
        env={"TRACEBENCH_DB_PATH": str(tmp_path / "empty.sqlite3")},
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == "No traces found.\n"


def test_sample_dataset_can_be_ingested(tmp_path: Path) -> None:
    """The checked-in sample file is valid and usable from the CLI."""
    sample_path = Path(__file__).parents[1] / "datasets" / "traces.sample.jsonl"
    result = runner.invoke(
        app,
        ["ingest", str(sample_path)],
        env={"TRACEBENCH_DB_PATH": str(tmp_path / "sample.sqlite3")},
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == (
        "Records read: 3\n"
        "Records accepted: 3\n"
        "Invalid records: 0\n"
        "Duplicates skipped: 0\n"
        "Records stored: 3\n"
    )
