"""Pure presentation helpers for the read-only Streamlit result explorer.

The dashboard is intentionally a consumer of :class:`ExperimentDetail`.  The
detail document is the authority for scores, transitions, review metadata,
slice aggregates, gate decisions, and generation observations.  Functions in
this module only select, count, sort, and format those persisted values; they
never connect to SQLite or re-run evaluation logic.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from pydantic import ValidationError

from tracebench.experiment_models import (
    ComparisonTransition,
    JudgeReviewStatus,
    RunRole,
    RunStatus,
)
from tracebench.models import EvaluationMode, Priority
from tracebench.reporting import CaseDetail, ExperimentDetail


class DashboardLoadError(ValueError):
    """A user-facing error while loading a strict detail export."""

    def __init__(self, message: str, *, kind: str = "invalid") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class OverviewMetrics:
    """Persisted top-level metrics suitable for summary cards."""

    verdict: str
    status: str
    baseline_score: float | None
    candidate_score: float | None
    score_delta: float | None
    newly_passed_count: int
    newly_failed_count: int
    critical_regression_count: int
    review_count: int
    gate_passed: bool | None
    gate_violation_count: int
    failure_stage: str | None
    failure_message: str | None


@dataclass(frozen=True, slots=True)
class SliceRow:
    """One persisted slice comparison row in numeric cluster order."""

    selector: str
    cluster_number: int
    label: str
    case_count: int
    baseline_score: float
    candidate_score: float
    score_delta: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    newly_passed_count: int
    newly_failed_count: int
    gate_configured: bool
    gate_status: str


@dataclass(frozen=True, slots=True)
class ReviewRow:
    """A persisted candidate review item with both run states visible."""

    case: CaseDetail
    reasons: tuple[str, ...]
    candidate_review_status: str
    baseline_review_status: str


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Meaningful latency comparison, or a reason it is unavailable."""

    available: bool
    message: str
    baseline_mean_ms: float | None = None
    candidate_mean_ms: float | None = None
    delta_ms: float | None = None
    baseline_observations: int = 0
    candidate_observations: int = 0


def resolve_report_path(
    report_argument: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """Resolve an explicit, environment, or conventional detail path.

    An explicit argument wins over ``TRACEBENCH_REPORT_PATH``.  The conventional
    ``.tracebench/experiment-detail.json`` is used only when it exists, so an
    empty app can explain how to load a report instead of displaying an error.
    """

    env = os.environ if environment is None else environment
    candidate = report_argument or env.get("TRACEBENCH_REPORT_PATH")
    if candidate:
        return Path(candidate).expanduser()
    base = Path.cwd() if cwd is None else cwd
    default = base / ".tracebench" / "experiment-detail.json"
    return default if default.is_file() else None


def load_detail_json(payload: str | bytes) -> ExperimentDetail:
    """Validate a strict versioned detail document from JSON text or bytes."""

    try:
        json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise DashboardLoadError(
            "The selected file is not valid JSON. Choose a strict experiment "
            "detail export.",
            kind="invalid",
        ) from error
    try:
        return ExperimentDetail.model_validate_json(payload)
    except ValidationError as error:
        error_text = str(error)
        if "unsupported experiment detail schema version" in error_text:
            raise DashboardLoadError(
                "This experiment detail uses an unsupported schema version. "
                "Export it again with the installed TraceBench version.",
                kind="unsupported_version",
            ) from error
        raise DashboardLoadError(
            "The selected file is not a valid TraceBench experiment detail JSON "
            "document. Check that it was exported with `tracebench experiment "
            "export`.",
            kind="invalid",
        ) from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise DashboardLoadError(
            "The selected file is not valid JSON. Choose a strict experiment "
            "detail export.",
            kind="invalid",
        ) from error


def load_detail_file(path: Path) -> ExperimentDetail:
    """Read and validate one UTF-8 strict detail export."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DashboardLoadError(
            f"Could not read report file `{path}`: {error}", kind="unreadable"
        ) from error
    return load_detail_json(payload)


def overview_metrics(detail: ExperimentDetail) -> OverviewMetrics:
    """Return summary values exactly as persisted in the detail document."""

    comparison = detail.report.comparison
    baseline = detail.report.runs.get(RunRole.BASELINE)
    candidate = detail.report.runs.get(RunRole.CANDIDATE)
    critical_ids = set() if comparison is None else set(comparison.newly_failed)
    critical_count = sum(
        1
        for case in detail.cases
        if case.eval_id in critical_ids and case.priority is Priority.CRITICAL
    )
    review_count = sum(
        1
        for case in detail.cases
        if case.candidate is not None
        and case.candidate.judge is not None
        and case.candidate.judge.review.status is JudgeReviewStatus.NEEDS_REVIEW
    )
    return OverviewMetrics(
        verdict="—" if detail.verdict is None else detail.verdict.value,
        status=detail.status.value,
        baseline_score=None if baseline is None else baseline.global_.score,
        candidate_score=None if candidate is None else candidate.global_.score,
        score_delta=None if comparison is None else comparison.global_.score_delta,
        newly_passed_count=0 if comparison is None else len(comparison.newly_passed),
        newly_failed_count=0 if comparison is None else len(comparison.newly_failed),
        critical_regression_count=critical_count,
        review_count=review_count,
        gate_passed=None if detail.gate is None else detail.gate.passed,
        gate_violation_count=len(detail.gate.violations),
        failure_stage=detail.failure_stage,
        failure_message=detail.failure_message,
    )


def slice_rows(detail: ExperimentDetail) -> tuple[SliceRow, ...]:
    """Arrange persisted slice comparison aggregates by numeric cluster order."""

    comparison = detail.report.comparison
    if comparison is None:
        return ()
    configured = detail.gate.thresholds.by_slice
    rows = []
    for aggregate in comparison.by_slice.values():
        label = aggregate.label_snapshot or aggregate.selector
        selector = aggregate.selector
        has_violation = any(
            violation.scope == selector or violation.scope == label
            for violation in detail.gate.violations
        )
        rows.append(
            SliceRow(
                selector=selector,
                cluster_number=aggregate.cluster_number,
                label=label,
                case_count=aggregate.case_count,
                baseline_score=aggregate.baseline_score,
                candidate_score=aggregate.candidate_score,
                score_delta=aggregate.score_delta,
                baseline_pass_rate=aggregate.baseline_pass_rate,
                candidate_pass_rate=aggregate.candidate_pass_rate,
                newly_passed_count=aggregate.newly_passed_count,
                newly_failed_count=aggregate.newly_failed_count,
                gate_configured=selector in configured or label in configured,
                gate_status=(
                    "FAIL"
                    if has_violation
                    else "PASS"
                    if selector in configured or label in configured
                    else "Not configured"
                ),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.cluster_number, row.selector)))


def slice_options(detail: ExperimentDetail) -> tuple[str, ...]:
    """Return stable human-readable slice labels for filter controls."""

    return tuple(row.label for row in slice_rows(detail))


def case_slice_label(case: CaseDetail) -> str:
    """Choose the persisted label snapshot, falling back to its selector."""

    if case.slice_provenance is None:
        return "Unassigned"
    return case.slice_provenance.label_snapshot or case.slice_provenance.selector


def filter_cases(
    detail: ExperimentDetail,
    *,
    slice_label: str | None = None,
    priority: Priority | str | None = None,
    evaluation_mode: EvaluationMode | str | None = None,
    transition: ComparisonTransition | str | None = None,
    review_state: JudgeReviewStatus | str | None = None,
    search: str = "",
    newly_failed_only: bool = True,
) -> tuple[CaseDetail, ...]:
    """Filter persisted case rows without changing their values or ordering."""

    normalized_priority = None if priority is None else Priority(priority)
    normalized_mode = (
        None if evaluation_mode is None else EvaluationMode(evaluation_mode)
    )
    normalized_transition = (
        None if transition is None else ComparisonTransition(transition)
    )
    normalized_review = (
        None if review_state is None else JudgeReviewStatus(review_state)
    )
    query = search.strip().casefold()
    selected: list[CaseDetail] = []
    for case in detail.cases:
        if (
            newly_failed_only
            and case.transition is not ComparisonTransition.NEWLY_FAILED
        ):
            continue
        if slice_label is not None and case_slice_label(case) != slice_label:
            continue
        if normalized_priority is not None and case.priority is not normalized_priority:
            continue
        if normalized_mode is not None and case.evaluation_mode is not normalized_mode:
            continue
        if (
            normalized_transition is not None
            and case.transition is not normalized_transition
        ):
            continue
        if normalized_review is not None:
            judge = None if case.candidate is None else case.candidate.judge
            if judge is None or judge.review.status is not normalized_review:
                continue
        if query:
            haystack = " ".join(
                (
                    case.eval_id,
                    case.source_trace.trace_id,
                    case.input,
                    case.source_trace.task_type,
                    case_slice_label(case),
                )
            ).casefold()
            if query not in haystack:
                continue
        selected.append(case)
    return tuple(sorted(selected, key=lambda case: case.eval_id))


def review_queue(detail: ExperimentDetail) -> tuple[ReviewRow, ...]:
    """Return cases whose persisted candidate judge state needs review."""

    rows: list[ReviewRow] = []
    for case in detail.cases:
        candidate_judge = None if case.candidate is None else case.candidate.judge
        if (
            candidate_judge is None
            or candidate_judge.review.status is not JudgeReviewStatus.NEEDS_REVIEW
        ):
            continue
        baseline_judge = None if case.baseline is None else case.baseline.judge
        rows.append(
            ReviewRow(
                case=case,
                reasons=tuple(
                    reason.value for reason in candidate_judge.review.reasons
                ),
                candidate_review_status=candidate_judge.review.status.value,
                baseline_review_status=(
                    "unavailable"
                    if baseline_judge is None
                    else baseline_judge.review.status.value
                ),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                0 if row.case.priority is Priority.CRITICAL else 1,
                row.case.eval_id,
            ),
        )
    )


def meaningful_latency(detail: ExperimentDetail) -> LatencySummary:
    """Summarize real-provider observations, or explicitly explain unavailability."""

    providers = {
        snapshot.provider.casefold() for snapshot in detail.provider_snapshots.values()
    }
    if "fixture" in providers:
        return LatencySummary(
            available=False,
            message="Unavailable for fixture providers",
        )
    run_status = {run.role: run.status for run in detail.runs}
    if (
        run_status.get(RunRole.BASELINE) is not RunStatus.COMPLETED
        or run_status.get(RunRole.CANDIDATE) is not RunStatus.COMPLETED
    ):
        return LatencySummary(
            available=False,
            message="Unavailable until both runs complete",
        )

    baseline_values = [
        case.baseline.generation.latency_ms
        for case in detail.cases
        if case.baseline is not None
        and _meaningful_number(case.baseline.generation.latency_ms)
    ]
    candidate_values = [
        case.candidate.generation.latency_ms
        for case in detail.cases
        if case.candidate is not None
        and _meaningful_number(case.candidate.generation.latency_ms)
    ]
    if len(baseline_values) < 2 or len(candidate_values) < 2:
        return LatencySummary(
            available=False,
            message="Unavailable: not enough meaningful provider observations",
            baseline_observations=len(baseline_values),
            candidate_observations=len(candidate_values),
        )
    baseline_mean = fmean(baseline_values)
    candidate_mean = fmean(candidate_values)
    return LatencySummary(
        available=True,
        message="Observed provider latency",
        baseline_mean_ms=baseline_mean,
        candidate_mean_ms=candidate_mean,
        delta_ms=candidate_mean - baseline_mean,
        baseline_observations=len(baseline_values),
        candidate_observations=len(candidate_values),
    )


def safe_context_text(context: Mapping[str, Any]) -> str:
    """Format persisted context without leaking Python repr or failing on values."""

    try:
        return json.dumps(dict(context), ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return "Context is not available as JSON."


def safe_metadata_text(metadata: Mapping[str, Any]) -> str:
    """Format sanitized persisted metadata for an expandable provenance view."""

    return safe_context_text(metadata)


def _meaningful_number(value: float) -> bool:
    return math.isfinite(value) and value > 0
