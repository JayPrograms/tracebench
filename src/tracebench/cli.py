"""Command-line interface for TraceBench."""

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any, Never

import typer
from pydantic import ValidationError

from tracebench.datasets import (
    DatasetAlreadyExistsError,
    DatasetError,
    create_dataset,
    export_dataset,
    get_dataset_and_cases,
    get_datasets,
    promote_trace,
)
from tracebench.ingestion import IngestionSummary, ingest_file
from tracebench.models import (
    EvalCase,
    EvalDataset,
    EvaluationMode,
    Priority,
    ReviewStatus,
    ScorerConfig,
    Trace,
)
from tracebench.storage import (
    connect_database,
    list_traces,
    resolve_database_path,
    timestamp_to_text,
)

app = typer.Typer(
    add_completion=False,
    help="Local-first regression testing for LLM applications.",
)
traces_app = typer.Typer(help="Inspect stored traces.")
dataset_app = typer.Typer(help="Manage versioned evaluation datasets.")
app.add_typer(traces_app, name="traces")
app.add_typer(dataset_app, name="dataset")


@app.callback()
def main() -> None:
    """Run TraceBench commands."""


@app.command("version")
def show_version() -> None:
    """Print the installed TraceBench version."""
    typer.echo(package_version("tracebench"))


@app.command()
def ingest(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a newline-delimited JSON trace file.",
        ),
    ],
) -> None:
    """Validate and store traces from a JSONL file."""
    try:
        summary = ingest_file(
            path,
            resolve_database_path(),
            error_reporter=lambda message: typer.echo(message, err=True),
        )
    except (OSError, UnicodeError, sqlite3.Error) as error:
        typer.echo(f"Error: could not ingest traces: {error}", err=True)
        raise typer.Exit(code=1) from error

    _print_ingestion_summary(summary)


@traces_app.command("list")
def show_traces() -> None:
    """List stored traces in newest-first order."""
    try:
        with closing(connect_database(resolve_database_path())) as connection:
            traces = list_traces(connection)
    except (OSError, sqlite3.Error, ValueError) as error:
        typer.echo(f"Error: could not list traces: {error}", err=True)
        raise typer.Exit(code=1) from error

    if not traces:
        typer.echo("No traces found.")
        return

    _print_trace_table(traces)


@dataset_app.command("create")
def create_eval_dataset(
    name: Annotated[str, typer.Option(help="Dataset name.")],
    version: Annotated[str, typer.Option(help="Dataset version.")],
    description: Annotated[
        str, typer.Option(help="Optional dataset description.")
    ] = "",
) -> None:
    """Create an explicitly versioned evaluation dataset."""
    try:
        dataset = create_dataset(
            resolve_database_path(),
            name=name,
            version=version,
            description=description,
        )
    except ValidationError as error:
        raise typer.BadParameter(_format_validation_error(error)) from error
    except DatasetAlreadyExistsError as error:
        _exit_dataset_error(error)
    except (OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    typer.echo(
        f"Created dataset {dataset.name}:{dataset.version} ({dataset.dataset_id})."
    )


@dataset_app.command("list")
def show_eval_datasets() -> None:
    """List stored evaluation datasets."""
    try:
        datasets = get_datasets(resolve_database_path())
    except (OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    if not datasets:
        typer.echo("No datasets found.")
        return
    _print_dataset_table(datasets)


@dataset_app.command("add-trace")
def add_trace_to_dataset(
    dataset_reference: Annotated[
        str,
        typer.Option("--dataset", help="Dataset reference in name:version form."),
    ],
    trace_id: Annotated[str, typer.Option(help="Stored source trace identifier.")],
    mode: Annotated[EvaluationMode, typer.Option(help="Evaluation mode.")],
    scorer: Annotated[
        list[str] | None,
        typer.Option(
            "--scorer",
            help="Repeatable scorer JSON object with name and optional config.",
        ),
    ] = None,
    scorer_file: Annotated[
        list[Path] | None,
        typer.Option(
            "--scorer-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Repeatable path to a scorer JSON object.",
        ),
    ] = None,
    rubric: Annotated[
        list[str] | None,
        typer.Option("--rubric", help="Repeatable rubric criterion."),
    ] = None,
    reference_answer: Annotated[
        str | None,
        typer.Option(help="Explicit answer for reference mode."),
    ] = None,
    use_source_response: Annotated[
        bool,
        typer.Option(
            help="Use the stored trace response as the reference-mode answer."
        ),
    ] = False,
    priority: Annotated[
        Priority, typer.Option(help="Case priority.")
    ] = Priority.MEDIUM,
    review_status: Annotated[
        ReviewStatus, typer.Option(help="Manual review status.")
    ] = ReviewStatus.DRAFT,
) -> None:
    """Promote a stored trace into an evaluation dataset."""
    scorer_values = scorer or []
    scorer_paths = scorer_file or []
    rubric_criteria = rubric or []
    _validate_mode_options(
        mode,
        has_scorers=bool(scorer_values or scorer_paths),
        rubric_criteria=rubric_criteria,
        reference_answer=reference_answer,
        use_source_response=use_source_response,
    )
    scorer_configs = _load_scorers(scorer_values, scorer_paths)

    try:
        case = promote_trace(
            resolve_database_path(),
            dataset_reference=dataset_reference,
            trace_id=trace_id,
            mode=mode,
            reference_answer=reference_answer,
            rubric=rubric_criteria,
            scorers=scorer_configs,
            use_source_response=use_source_response,
            priority=priority,
            review_status=review_status,
        )
    except ValidationError as error:
        _exit_dataset_error(ValueError(_format_validation_error(error)))
    except (DatasetError, OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    typer.echo(
        f"Added trace {case.source_trace_id} to {dataset_reference} as {case.eval_id}."
    )


@dataset_app.command("show")
def show_eval_dataset(
    reference: Annotated[str, typer.Argument(help="Dataset name:version reference.")],
) -> None:
    """Show dataset metadata and evaluation cases."""
    try:
        dataset, cases = get_dataset_and_cases(resolve_database_path(), reference)
    except (DatasetError, OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    typer.echo(f"Dataset: {dataset.name}:{dataset.version}")
    typer.echo(f"Dataset ID: {dataset.dataset_id}")
    typer.echo(f"Description: {dataset.description}")
    typer.echo(f"Created at: {timestamp_to_text(dataset.created_at)}")
    typer.echo(f"Cases: {len(cases)}")
    if cases:
        _print_eval_case_table(cases)


@dataset_app.command("export")
def export_eval_dataset(
    reference: Annotated[str, typer.Argument(help="Dataset name:version reference.")],
    output: Annotated[
        Path, typer.Option(help="Destination newline-delimited JSON file.")
    ],
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing output file.")
    ] = False,
) -> None:
    """Export a dataset as newline-delimited JSON."""
    try:
        count = export_dataset(
            resolve_database_path(),
            reference,
            output,
            overwrite=overwrite,
        )
    except (DatasetError, OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    typer.echo(f"Exported {count} evaluation cases to {output}.")


def _print_ingestion_summary(summary: IngestionSummary) -> None:
    typer.echo(f"Records read: {summary.records_read}")
    typer.echo(f"Records accepted: {summary.records_accepted}")
    typer.echo(f"Invalid records: {summary.invalid_records}")
    typer.echo(f"Duplicates skipped: {summary.duplicates_skipped}")
    typer.echo(f"Records stored: {summary.records_stored}")


def _print_trace_table(traces: list[Trace]) -> None:
    headers = ("TRACE_ID", "TIMESTAMP", "TASK_TYPE", "HAS_RESPONSE")
    rows = [
        (
            trace.trace_id,
            timestamp_to_text(trace.timestamp),
            trace.task_type,
            "yes" if trace.response is not None else "no",
        )
        for trace in traces
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    typer.echo(_format_table_row(headers, widths))
    for row in rows:
        typer.echo(_format_table_row(row, widths))


def _print_dataset_table(datasets: list[tuple[EvalDataset, int]]) -> None:
    headers = ("DATASET_ID", "NAME", "VERSION", "CASES", "CREATED_AT")
    rows = [
        (
            dataset.dataset_id,
            dataset.name,
            dataset.version,
            str(case_count),
            timestamp_to_text(dataset.created_at),
        )
        for dataset, case_count in datasets
    ]
    _print_table(headers, rows)


def _print_eval_case_table(cases: list[EvalCase]) -> None:
    headers = (
        "EVAL_ID",
        "SOURCE_TRACE_ID",
        "MODE",
        "PRIORITY",
        "REVIEW_STATUS",
        "SCORERS",
        "RUBRIC_CRITERIA",
    )
    rows = [
        (
            case.eval_id,
            case.source_trace_id,
            case.evaluation_mode.value,
            case.priority.value,
            case.review_status.value,
            str(len(case.scorers)),
            str(len(case.rubric)),
        )
        for case in cases
    ]
    _print_table(headers, rows)


def _print_table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> None:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    typer.echo(_format_table_row(headers, widths))
    for row in rows:
        typer.echo(_format_table_row(row, widths))


def _parse_scorers(values: list[str]) -> list[ScorerConfig]:
    scorers: list[ScorerConfig] = []
    for value in values:
        try:
            payload: Any = json.loads(value)
        except json.JSONDecodeError as error:
            _exit_usage_error(f"scorer must be valid JSON: {error.msg}")
        if not isinstance(payload, dict):
            _exit_usage_error("scorer must be a JSON object")
        try:
            scorers.append(ScorerConfig.model_validate(payload))
        except ValidationError as error:
            _exit_usage_error(_format_validation_error(error))
    return scorers


def _load_scorers(values: list[str], paths: list[Path]) -> list[ScorerConfig]:
    file_values: list[str] = []
    for path in paths:
        try:
            file_values.append(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError) as error:
            _exit_usage_error(f"could not read scorer file {path}: {error}")
    return _parse_scorers([*values, *file_values])


def _validate_mode_options(
    mode: EvaluationMode,
    *,
    has_scorers: bool,
    rubric_criteria: list[str],
    reference_answer: str | None,
    use_source_response: bool,
) -> None:
    if reference_answer is not None and not reference_answer.strip():
        _exit_usage_error("--reference-answer must not be blank")

    if mode is EvaluationMode.DETERMINISTIC:
        if reference_answer is not None:
            _exit_usage_error("deterministic mode forbids --reference-answer")
        if use_source_response:
            _exit_usage_error("deterministic mode forbids --use-source-response")
        if rubric_criteria:
            _exit_usage_error("deterministic mode forbids --rubric")
        if not has_scorers:
            _exit_usage_error(
                "deterministic mode requires at least one --scorer or --scorer-file"
            )
    elif mode is EvaluationMode.REFERENCE:
        if has_scorers:
            _exit_usage_error("reference mode forbids --scorer and --scorer-file")
        if rubric_criteria:
            _exit_usage_error("reference mode forbids --rubric")
        if reference_answer is not None and use_source_response:
            _exit_usage_error(
                "reference mode accepts either --reference-answer "
                "or --use-source-response"
            )
        if reference_answer is None and not use_source_response:
            _exit_usage_error(
                "reference mode requires --reference-answer or --use-source-response"
            )
    else:
        if reference_answer is not None:
            _exit_usage_error("rubric mode forbids --reference-answer")
        if use_source_response:
            _exit_usage_error("rubric mode forbids --use-source-response")
        if has_scorers:
            _exit_usage_error("rubric mode forbids --scorer and --scorer-file")
        if not rubric_criteria:
            _exit_usage_error("rubric mode requires at least one --rubric")


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


def _exit_dataset_error(error: BaseException) -> Never:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


def _exit_usage_error(message: str) -> Never:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _format_table_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(width) for value, width in zip(row, widths, strict=True)
    ).rstrip()
