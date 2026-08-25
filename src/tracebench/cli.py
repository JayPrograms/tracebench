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

from tracebench.clustering import (
    ClusteringError,
    ClusteringValidationError,
    create_clustering_run,
    list_slices,
    rename_slice,
)
from tracebench.datasets import (
    DatasetAlreadyExistsError,
    DatasetBuildValidationError,
    DatasetError,
    build_dataset_from_slices,
    create_dataset,
    export_dataset,
    get_dataset_details,
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
from tracebench.reporting import ExperimentDetailError, export_experiment_detail
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
slices_app = typer.Typer(help="Inspect and label persistent trace slices.")
app.add_typer(traces_app, name="traces")
app.add_typer(dataset_app, name="dataset")
app.add_typer(experiment_app, name="experiment")
app.add_typer(slices_app, name="slices")


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


@traces_app.command("cluster")
def cluster_traces(
    name: Annotated[str, typer.Option(help="Unique versioned clustering run name.")],
    clusters: Annotated[int, typer.Option(help="Number of clusters.")],
    include_context: Annotated[
        bool, typer.Option(help="Append canonical trace context to each prompt.")
    ] = False,
    svd_components: Annotated[
        int | None,
        typer.Option(help="Enable Truncated SVD with this dimension count."),
    ] = None,
) -> None:
    """Cluster the current deterministic trace snapshot."""
    try:
        result = create_clustering_run(
            resolve_database_path(),
            name=name,
            clusters=clusters,
            include_context=include_context,
            svd_components=svd_components,
        )
    except ClusteringValidationError as error:
        typer.echo(
            f"Error: {error}; no clustering run was persisted",
            err=True,
        )
        raise typer.Exit(code=2) from error
    except (ClusteringError, OSError, sqlite3.Error, ValueError) as error:
        _exit_clustering_error(error)

    run = result.run
    typer.echo(f"Created clustering run {run.name} ({run.clustering_run_id}).")
    typer.echo(f"Configuration hash: {run.configuration_hash}")
    typer.echo(f"Source manifest hash: {run.source_manifest_hash}")
    typer.echo(f"Traces clustered: {run.trace_count}")
    typer.echo(f"Clusters: {run.cluster_count}")
    typer.echo(
        "SVD components: "
        + ("disabled" if run.svd_components is None else str(run.svd_components))
    )


@slices_app.command("list")
def show_slices(
    name: Annotated[str, typer.Argument(help="Named clustering run.")],
) -> None:
    """List numeric slices and their editable labels."""
    try:
        run, slices = list_slices(resolve_database_path(), name)
    except (ClusteringError, OSError, sqlite3.Error, ValueError) as error:
        _exit_clustering_error(error)
    typer.echo(f"Clustering run: {run.name} ({run.clustering_run_id})")
    _print_table(
        ("CLUSTER", "LABEL", "TRACES"),
        [
            (
                str(item.cluster_number),
                "-" if item.label is None else item.label,
                str(item.trace_count),
            )
            for item in slices
        ],
    )


@slices_app.command("rename")
def rename_slice_command(
    name: Annotated[str, typer.Argument(help="Named clustering run.")],
    cluster_number: Annotated[int, typer.Argument(help="Numeric cluster identifier.")],
    label: Annotated[str, typer.Argument(help="Human-readable slice label.")],
) -> None:
    """Rename a numeric slice without changing its assignments."""
    try:
        rename_slice(resolve_database_path(), name, cluster_number, label)
    except (ClusteringError, OSError, sqlite3.Error, ValueError) as error:
        _exit_clustering_error(error)
    typer.echo(f"Renamed slice {cluster_number} in {name.strip()} to {label.strip()}.")


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


@dataset_app.command("build")
def build_eval_dataset(
    name: Annotated[str, typer.Option(help="Dataset name.")],
    version: Annotated[str, typer.Option(help="Dataset version.")],
    from_slices: Annotated[
        str,
        typer.Option("--from-slices", help="Immutable clustering run name."),
    ],
    size: Annotated[int, typer.Option(min=1, help="Exact number of evaluation cases.")],
    case_file: Annotated[
        Path | None,
        typer.Option(
            "--case-file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Optional strict schema-v1 JSON overrides for selected cases.",
        ),
    ] = None,
) -> None:
    """Build an exact-size balanced dataset from numeric slices."""
    try:
        result = build_dataset_from_slices(
            resolve_database_path(),
            name=name,
            version=version,
            clustering_run_name=from_slices,
            size=size,
            case_file=case_file,
        )
    except DatasetBuildValidationError as error:
        typer.echo(f"Error: {error}; no dataset was persisted", err=True)
        raise typer.Exit(code=2) from error
    except (DatasetError, OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    dataset = result.dataset
    source = result.slice_source
    typer.echo(
        f"Created dataset {dataset.name}:{dataset.version} ({dataset.dataset_id})."
    )
    typer.echo(
        f"Source clustering run: {source.clustering_run_name} "
        f"({source.clustering_run_id})"
    )
    typer.echo(f"Sampling algorithm: {source.sampling_algorithm}")
    typer.echo(f"Requested size: {source.requested_size}")
    typer.echo(f"Eligible traces: {source.eligible_trace_count}")
    typer.echo(f"Cases created: {len(result.cases)}")
    _print_table(
        ("SLICE", "CLUSTER", "LABEL", "ELIGIBLE", "SELECTED"),
        [
            (
                item.selector,
                str(item.cluster_number),
                "-" if item.label_snapshot is None else item.label_snapshot,
                str(item.eligible_count),
                str(item.selected_count),
            )
            for item in result.slices
        ],
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
        dataset, cases, slice_source = get_dataset_details(
            resolve_database_path(), reference
        )
    except (DatasetError, OSError, sqlite3.Error, ValueError) as error:
        _exit_dataset_error(error)

    typer.echo(f"Dataset: {dataset.name}:{dataset.version}")
    typer.echo(f"Dataset ID: {dataset.dataset_id}")
    typer.echo(f"Description: {dataset.description}")
    typer.echo(f"Created at: {timestamp_to_text(dataset.created_at)}")
    typer.echo(f"Cases: {len(cases)}")
    if slice_source is not None:
        typer.echo(
            f"Slice source: {slice_source.clustering_run_name} "
            f"({slice_source.clustering_run_id})"
        )
        typer.echo(
            f"Sampling: {slice_source.sampling_algorithm}; "
            f"requested={slice_source.requested_size}; "
            f"eligible={slice_source.eligible_trace_count}"
        )
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


@experiment_app.command("export")
def export_experiment_command(
    experiment_id: Annotated[str, typer.Argument(help="Persisted experiment ID.")],
    output: Annotated[Path, typer.Option(help="Destination UTF-8 JSON detail file.")],
    overwrite: Annotated[
        bool, typer.Option(help="Replace an existing output file.")
    ] = False,
) -> None:
    """Export a persisted experiment detail for dashboards and automation."""
    try:
        export_experiment_detail(
            resolve_database_path(),
            experiment_id,
            output,
            overwrite=overwrite,
        )
    except (ExperimentDetailError, OSError, sqlite3.Error, ValueError) as error:
        _exit_experiment_export_error(error)
    typer.echo(f"Exported experiment detail {experiment_id} to {output}.")


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
    if report.comparison.by_slice:
        _print_table(
            (
                "SLICE",
                "LABEL",
                "CASES",
                "BASE_SCORE",
                "CAND_SCORE",
                "BASE_PASS_RATE",
                "CAND_PASS_RATE",
                "NEW_PASS",
                "NEW_FAIL",
            ),
            [
                (
                    item.selector,
                    "-" if item.label_snapshot is None else item.label_snapshot,
                    str(item.case_count),
                    f"{item.baseline_score:.6f}",
                    f"{item.candidate_score:.6f}",
                    f"{item.baseline_pass_rate:.6f}",
                    f"{item.candidate_pass_rate:.6f}",
                    str(item.newly_passed_count),
                    str(item.newly_failed_count),
                )
                for item in sorted(
                    report.comparison.by_slice.values(),
                    key=lambda aggregate: aggregate.cluster_number,
                )
            ],
        )
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
    slice_aware = any(case.slice_provenance is not None for case in cases)
    headers = (
        "EVAL_ID",
        "SOURCE_TRACE_ID",
        *(("SLICE",) if slice_aware else ()),
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
            *(
                (
                    case.slice_provenance.selector
                    if case.slice_provenance is not None
                    else "-",
                )
                if slice_aware
                else ()
            ),
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


def _exit_clustering_error(error: BaseException) -> Never:
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=1) from error


def _exit_experiment_export_error(error: BaseException) -> Never:
    """Render export IDs and paths as usage/preflight failures."""
    typer.echo(f"Error: {error}", err=True)
    raise typer.Exit(code=2) from error


def _exit_usage_error(message: str) -> Never:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=2)


def _format_table_row(row: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(
        value.ljust(width) for value, width in zip(row, widths, strict=True)
    ).rstrip()
