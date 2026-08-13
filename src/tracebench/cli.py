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
from tracebench.experiment_config import ExperimentPreflightError
from tracebench.experiment_models import (
    ExperimentReport,
    ExperimentVerdict,
    JudgeReviewStatus,
)
from tracebench.experiments import ExperimentOperationalError, execute_experiment
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
experiment_app = typer.Typer(help="Run persisted baseline/candidate experiments.")
app.add_typer(traces_app, name="traces")
app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")


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


@experiment_app.command("run")
def run_experiment_command(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a versioned experiment YAML file.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print only the machine-readable JSON result."),
    ] = False,
) -> None:
    """Run, score, compare, gate, and persist an experiment attempt."""
    try:
        report = execute_experiment(config, resolve_database_path())
    except ExperimentPreflightError as error:
        typer.echo(f"Error: {error}; no experiment attempt was persisted", err=True)
        raise typer.Exit(code=2) from error
    except ExperimentOperationalError as error:
        _print_operational_failure(error, json_output=json_output)
        raise typer.Exit(code=3) from error
    except (OSError, sqlite3.Error, ValueError) as error:
        typer.echo(f"Operational failure before attempt creation: {error}", err=True)
        raise typer.Exit(code=3) from error

    if json_output:
        typer.echo(
            json.dumps(
                report.model_dump(mode="json", by_alias=True),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        _print_experiment_report(report)
    if report.verdict is ExperimentVerdict.FAIL:
        raise typer.Exit(code=1)


def _print_ingestion_summary(summary: IngestionSummary) -> None:
    typer.echo(f"Records read: {summary.records_read}")
    typer.echo(f"Records accepted: {summary.records_accepted}")
    typer.echo(f"Invalid records: {summary.invalid_records}")
    typer.echo(f"Duplicates skipped: {summary.duplicates_skipped}")
    typer.echo(f"Records stored: {summary.records_stored}")


def _print_experiment_report(report: ExperimentReport) -> None:
    """Print the concise human representation of a completed experiment."""
    if report.verdict is None or report.comparison is None or report.gate is None:
        raise ValueError("completed experiment report is missing a verdict")
    baseline = report.runs[
        next(role for role in report.runs if role.value == "baseline")
    ]
    candidate = report.runs[
        next(role for role in report.runs if role.value == "candidate")
    ]
    typer.echo(report.verdict.value)
    typer.echo(f"Experiment: {report.experiment_id}")
    typer.echo(f"Configuration hash: {report.configuration_hash}")
    typer.echo(
        f"Dataset: {report.dataset.name}:{report.dataset.version} "
        f"({report.comparison.global_.case_count} cases)"
    )
    typer.echo(f"Baseline score: {baseline.global_.score:.6f}")
    typer.echo(f"Candidate score: {candidate.global_.score:.6f}")
    typer.echo(f"Newly passed: {len(report.comparison.newly_passed)}")
    typer.echo(f"Newly failed: {len(report.comparison.newly_failed)}")
    rubric_results = [
        case
        for run in (baseline, candidate)
        for case in run.cases
        if case.judge is not None
    ]
    if rubric_results:
        for label, run in (("Baseline", baseline), ("Candidate", candidate)):
            judge_results = [case.judge for case in run.cases if case.judge is not None]
            low_confidence = sum(
                judge.below_confidence_threshold for judge in judge_results
            )
            typer.echo(f"{label} low-confidence rubric cases: {low_confidence}")
            cache_hits = sum(judge.cache_hit is True for judge in judge_results)
            cache_misses = sum(judge.cache_hit is False for judge in judge_results)
            typer.echo(
                f"{label} rubric judge cache: "
                f"{cache_hits} hit(s), {cache_misses} miss(es)"
            )
            needs_review = [
                case.eval_id
                for case in run.cases
                if case.judge is not None
                and case.judge.review.status is JudgeReviewStatus.NEEDS_REVIEW
            ]
            typer.echo(f"{label} rubric cases needing review: {len(needs_review)}")
            for eval_id in needs_review:
                typer.echo(f"Needs review: {label.lower()}:{eval_id}")
    for violation in report.gate.violations:
        typer.echo(f"Violation: {violation.message}")


def _print_operational_failure(
    error: ExperimentOperationalError,
    *,
    json_output: bool,
) -> None:
    """Render an operational failure separately from a regression verdict."""
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": error.experiment_id,
                    "status": "failed",
                    "verdict": None,
                    "failure_stage": error.stage,
                    "failure_message": error.message,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    else:
        typer.echo(
            f"Operational failure: experiment {error.experiment_id} failed during "
            f"{error.stage}: {error.message}",
            err=True,
        )


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
