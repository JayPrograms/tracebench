"""Tests for the TraceBench command-line interface."""

import json
from importlib.metadata import version as package_version
from pathlib import Path

import pytest
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


def test_dataset_cli_complete_workflow(tmp_path: Path) -> None:
    """The CLI creates, promotes, displays, and exports a dataset."""
    database_path = tmp_path / "local.sqlite3"
    input_path = tmp_path / "traces.jsonl"
    output_path = tmp_path / "support.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "trace_id": "trace-001",
                "timestamp": "2026-07-28T14:00:00Z",
                "task_type": "question-answering",
                "prompt": "What is the capital of Canada?",
                "response": "Ottawa",
                "context": {"locale": "en-CA"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}
    assert (
        runner.invoke(app, ["ingest", str(input_path)], env=environment).exit_code == 0
    )

    create_result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--name",
            "support-eval",
            "--version",
            "0.1",
            "--description",
            "Support regression benchmark",
        ],
        env=environment,
    )
    assert create_result.exit_code == 0, create_result.output
    assert "Created dataset support-eval:0.1" in create_result.stdout

    duplicate_result = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--name",
            "support-eval",
            "--version",
            "0.1",
        ],
        env=environment,
    )
    assert duplicate_result.exit_code == 1
    assert "already exists" in duplicate_result.stderr

    list_result = runner.invoke(app, ["dataset", "list"], env=environment)
    assert list_result.exit_code == 0, list_result.output
    assert "DATASET_ID" in list_result.stdout
    assert "support-eval" in list_result.stdout
    assert "0.1" in list_result.stdout

    add_result = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support-eval:0.1",
            "--trace-id",
            "trace-001",
            "--mode",
            "reference",
            "--use-source-response",
        ],
        env=environment,
    )
    assert add_result.exit_code == 0, add_result.output
    assert "Added trace trace-001" in add_result.stdout

    show_result = runner.invoke(
        app,
        ["dataset", "show", "support-eval:0.1"],
        env=environment,
    )
    assert show_result.exit_code == 0, show_result.output
    assert "Dataset: support-eval:0.1" in show_result.stdout
    assert "Support regression benchmark" in show_result.stdout
    assert "trace-001" in show_result.stdout
    assert "reference" in show_result.stdout
    assert "medium" in show_result.stdout
    assert "draft" in show_result.stdout

    export_result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "support-eval:0.1",
            "--output",
            str(output_path),
        ],
        env=environment,
    )
    assert export_result.exit_code == 0, export_result.output
    assert "Exported 1 evaluation cases" in export_result.stdout
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["record_type"] == "dataset"
    assert records[0]["dataset"]["dataset_ref"] == "support-eval:0.1"
    assert records[1]["case"]["source_trace_id"] == "trace-001"

    protected_result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "support-eval:0.1",
            "--output",
            str(output_path),
        ],
        env=environment,
    )
    assert protected_result.exit_code == 1
    assert "already exists" in protected_result.stderr

    overwrite_result = runner.invoke(
        app,
        [
            "dataset",
            "export",
            "support-eval:0.1",
            "--output",
            str(output_path),
            "--overwrite",
        ],
        env=environment,
    )
    assert overwrite_result.exit_code == 0, overwrite_result.output


def test_dataset_cli_rejects_missing_resources_and_invalid_mode_data(
    tmp_path: Path,
) -> None:
    """Dataset commands expose expected domain and usage failures."""
    database_path = tmp_path / "local.sqlite3"
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}
    create_result = runner.invoke(
        app,
        ["dataset", "create", "--name", "support", "--version", "1"],
        env=environment,
    )
    assert create_result.exit_code == 0

    missing_trace = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "missing",
            "--mode",
            "reference",
            "--reference-answer",
            "answer",
        ],
        env=environment,
    )
    assert missing_trace.exit_code == 1
    assert "trace 'missing' was not found" in missing_trace.stderr

    missing_dataset = runner.invoke(
        app,
        ["dataset", "show", "missing:1"],
        env=environment,
    )
    assert missing_dataset.exit_code == 1
    assert "dataset 'missing:1' was not found" in missing_dataset.stderr

    missing_scorer = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "missing",
            "--mode",
            "deterministic",
        ],
        env=environment,
    )
    assert missing_scorer.exit_code == 2
    assert "requires at least one --scorer" in missing_scorer.output

    invalid_scorer = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "missing",
            "--mode",
            "deterministic",
            "--scorer",
            "not-json",
        ],
        env=environment,
    )
    assert invalid_scorer.exit_code == 2
    assert "scorer must be valid JSON" in invalid_scorer.output

    missing_reference = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "missing",
            "--mode",
            "reference",
        ],
        env=environment,
    )
    assert missing_reference.exit_code == 2
    assert "--reference-answer or --use-source-response" in missing_reference.output

    conflicting_reference = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "missing",
            "--mode",
            "reference",
            "--reference-answer",
            "answer",
            "--use-source-response",
        ],
        env=environment,
    )
    assert conflicting_reference.exit_code == 2
    assert "either --reference-answer or --use-source-response" in (
        conflicting_reference.output
    )


@pytest.mark.parametrize(
    ("mode", "wrong_options", "message"),
    [
        ("deterministic", ["--reference-answer", "answer"], "forbids"),
        ("deterministic", ["--rubric", "Correct"], "forbids"),
        ("deterministic", ["--use-source-response"], "forbids"),
        ("reference", ["--scorer", '{"name":"exact"}'], "forbids"),
        ("reference", ["--rubric", "Correct"], "forbids"),
        ("rubric", ["--reference-answer", "answer"], "forbids"),
        ("rubric", ["--scorer", '{"name":"exact"}'], "forbids"),
        ("rubric", ["--use-source-response"], "forbids"),
    ],
)
def test_dataset_cli_rejects_wrong_mode_options(
    tmp_path: Path,
    mode: str,
    wrong_options: list[str],
    message: str,
) -> None:
    """Mode-specific CLI options cannot be combined with another strategy."""
    result = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "support:1",
            "--trace-id",
            "trace-001",
            "--mode",
            mode,
            *wrong_options,
        ],
        env={"TRACEBENCH_DB_PATH": str(tmp_path / "local.sqlite3")},
    )

    assert result.exit_code == 2
    assert message in result.output


def test_dataset_cli_accepts_deterministic_and_rubric_options(
    tmp_path: Path,
) -> None:
    """Repeatable mode-specific options reach stored evaluation cases."""
    database_path = tmp_path / "local.sqlite3"
    input_path = tmp_path / "traces.jsonl"
    records = [
        {
            "trace_id": trace_id,
            "timestamp": "2026-07-28T14:00:00Z",
            "task_type": "test-task",
            "prompt": f"Prompt for {trace_id}",
        }
        for trace_id in ("deterministic", "rubric", "file-scorer")
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}
    scorer_path = tmp_path / "exact-match.json"
    scorer_path.write_text(
        '\ufeff{"name":"exact_match","config":{"case_sensitive":false}}',
        encoding="utf-8",
    )
    assert (
        runner.invoke(app, ["ingest", str(input_path)], env=environment).exit_code == 0
    )
    assert (
        runner.invoke(
            app,
            ["dataset", "create", "--name", "modes", "--version", "1"],
            env=environment,
        ).exit_code
        == 0
    )

    deterministic = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "modes:1",
            "--trace-id",
            "deterministic",
            "--mode",
            "deterministic",
            "--scorer",
            '{"name":"exact_match","config":{"case_sensitive":false}}',
            "--priority",
            "critical",
            "--review-status",
            "approved",
        ],
        env=environment,
    )
    rubric = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "modes:1",
            "--trace-id",
            "rubric",
            "--mode",
            "rubric",
            "--rubric",
            "Correct",
            "--rubric",
            "Relevant",
        ],
        env=environment,
    )
    file_scorer = runner.invoke(
        app,
        [
            "dataset",
            "add-trace",
            "--dataset",
            "modes:1",
            "--trace-id",
            "file-scorer",
            "--mode",
            "deterministic",
            "--scorer-file",
            str(scorer_path),
        ],
        env=environment,
    )

    assert deterministic.exit_code == 0, deterministic.output
    assert rubric.exit_code == 0, rubric.output
    assert file_scorer.exit_code == 0, file_scorer.output
    show = runner.invoke(app, ["dataset", "show", "modes:1"], env=environment)
    assert show.exit_code == 0, show.output
    assert "critical" in show.stdout
    assert "approved" in show.stdout
    assert "deterministic" in show.stdout
    assert "rubric" in show.stdout
