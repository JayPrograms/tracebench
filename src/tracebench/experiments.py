"""End-to-end experiment execution, comparison, and regression gating."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Never
from uuid import uuid4

from tracebench.experiment_config import PreparedExperiment, prepare_experiment
from tracebench.experiment_models import (
    CaseComparison,
    CaseResult,
    ComparisonAggregate,
    ComparisonTransition,
    EffectiveThresholds,
    ExperimentReport,
    ExperimentVerdict,
    GateMetric,
    GateViolation,
    GenerationDetails,
    RunAggregate,
    RunRole,
)
from tracebench.experiment_storage import (
    fail_attempt,
    insert_attempt,
    load_experiment_report,
    persist_completed_comparison,
    persist_completed_run,
    start_run,
)
from tracebench.scorers import score_case
from tracebench.storage import connect_database


class ExperimentOperationalError(Exception):
    """Raised when an accepted experiment fails after preflight."""

    def __init__(self, experiment_id: str, stage: str, message: str) -> None:
        super().__init__(message)
        self.experiment_id = experiment_id
        self.stage = stage
        self.message = message


def execute_experiment(config_path: Path, database_path: Path) -> ExperimentReport:
    """Run one independently identified baseline/candidate experiment attempt."""
    prepared = prepare_experiment(config_path, database_path)
    experiment_id = f"experiment_{uuid4().hex}"
    run_ids = {
        RunRole.BASELINE: f"run_{uuid4().hex}",
        RunRole.CANDIDATE: f"run_{uuid4().hex}",
    }
    with closing(connect_database(database_path)) as connection:
        try:
            with connection:
                insert_attempt(
                    connection,
                    experiment_id=experiment_id,
                    name=prepared.config.name,
                    dataset_id=prepared.dataset.dataset_id,
                    configuration_hash=prepared.configuration_hash,
                    configuration_json=prepared.configuration_json,
                    run_ids=run_ids,
                    provider_snapshots=prepared.provider_snapshots,
                    timestamp=datetime.now(UTC),
                )
        except Exception as error:
            raise ExperimentOperationalError(
                experiment_id,
                "attempt_creation",
                _error_message(error),
            ) from error

        completed_results: dict[RunRole, list[CaseResult]] = {}
        for role in RunRole:
            completed_results[role] = _execute_and_persist_run(
                connection=connection,
                prepared=prepared,
                experiment_id=experiment_id,
                run_id=run_ids[role],
                role=role,
            )

        try:
            comparisons, comparison_aggregates = compare_runs(
                completed_results[RunRole.BASELINE],
                completed_results[RunRole.CANDIDATE],
            )
            violations = apply_regression_gate(
                comparison_aggregates,
                prepared.effective_thresholds,
            )
            verdict = (
                ExperimentVerdict.PASS if not violations else ExperimentVerdict.FAIL
            )
            with connection:
                persist_completed_comparison(
                    connection,
                    experiment_id=experiment_id,
                    comparisons=comparisons,
                    aggregates=comparison_aggregates,
                    violations=violations,
                    verdict=verdict,
                    timestamp=datetime.now(UTC),
                )
                report = load_experiment_report(connection, experiment_id)
        except Exception as error:
            _record_operational_failure(
                connection=connection,
                experiment_id=experiment_id,
                active_run_id=None,
                stage="comparison",
                error=error,
            )
        return report


def aggregate_run(results: list[CaseResult]) -> list[RunAggregate]:
    """Calculate the global and present-mode aggregates for one run."""
    if not results:
        raise ValueError("cannot aggregate an empty run")
    aggregates = [_aggregate_scope("global", results)]
    modes = sorted({result.evaluation_mode for result in results}, key=str)
    for mode in modes:
        mode_results = [result for result in results if result.evaluation_mode is mode]
        aggregates.append(_aggregate_scope(mode.value, mode_results))
    return aggregates


def compare_runs(
    baseline: list[CaseResult],
    candidate: list[CaseResult],
) -> tuple[list[CaseComparison], list[ComparisonAggregate]]:
    """Compare matching result sets globally and by present evaluation mode."""
    baseline_by_id = {result.eval_id: result for result in baseline}
    candidate_by_id = {result.eval_id: result for result in candidate}
    if set(baseline_by_id) != set(candidate_by_id):
        raise ValueError("baseline and candidate result sets do not match")
    comparisons: list[CaseComparison] = []
    for eval_id in sorted(baseline_by_id):
        baseline_result = baseline_by_id[eval_id]
        candidate_result = candidate_by_id[eval_id]
        if baseline_result.evaluation_mode is not candidate_result.evaluation_mode:
            raise ValueError(f"evaluation mode differs for case '{eval_id}'")
        comparisons.append(
            CaseComparison(
                eval_id=eval_id,
                evaluation_mode=baseline_result.evaluation_mode,
                baseline_score=baseline_result.score,
                candidate_score=candidate_result.score,
                score_delta=candidate_result.score - baseline_result.score,
                baseline_passed=baseline_result.passed,
                candidate_passed=candidate_result.passed,
                transition=_transition(
                    baseline_result.passed,
                    candidate_result.passed,
                ),
            )
        )
    aggregates = [_comparison_scope("global", comparisons)]
    modes = sorted({item.evaluation_mode for item in comparisons}, key=str)
    for mode in modes:
        mode_comparisons = [
            item for item in comparisons if item.evaluation_mode is mode
        ]
        aggregates.append(_comparison_scope(mode.value, mode_comparisons))
    return comparisons, aggregates


def apply_regression_gate(
    aggregates: list[ComparisonAggregate],
    thresholds: dict[str, EffectiveThresholds],
) -> list[GateViolation]:
    """Return deterministic violations for every exceeded configured limit."""
    by_scope = {aggregate.scope: aggregate for aggregate in aggregates}
    violations: list[GateViolation] = []
    for scope, effective_thresholds in thresholds.items():
        aggregate = by_scope[scope]
        score_drop = max(0.0, -aggregate.score_delta)
        if score_drop > effective_thresholds.max_score_drop:
            violations.append(
                GateViolation(
                    scope=scope,
                    metric=GateMetric.SCORE_DROP,
                    actual=score_drop,
                    allowed=effective_thresholds.max_score_drop,
                    message=(
                        f"{scope} score drop {score_drop:.6f} exceeds "
                        f"{effective_thresholds.max_score_drop:.6f}"
                    ),
                )
            )
        if aggregate.newly_failed_count > effective_thresholds.max_new_failures:
            violations.append(
                GateViolation(
                    scope=scope,
                    metric=GateMetric.NEW_FAILURES,
                    actual=float(aggregate.newly_failed_count),
                    allowed=float(effective_thresholds.max_new_failures),
                    message=(
                        f"{scope} newly failed count "
                        f"{aggregate.newly_failed_count} exceeds "
                        f"{effective_thresholds.max_new_failures}"
                    ),
                )
            )
    return violations


def _execute_and_persist_run(
    *,
    connection: sqlite3.Connection,
    prepared: PreparedExperiment,
    experiment_id: str,
    run_id: str,
    role: RunRole,
) -> list[CaseResult]:
    stage = role.value
    try:
        with connection:
            start_run(connection, run_id, datetime.now(UTC))
        provider = prepared.providers[role]
        results: list[CaseResult] = []
        generation_details: dict[str, GenerationDetails] = {}
        for case in prepared.cases:
            started_at = perf_counter_ns()
            response = provider.generate(case)
            latency_ms = max(0.0, (perf_counter_ns() - started_at) / 1_000_000)
            results.append(score_case(case, response.output))
            generation_details[case.eval_id] = GenerationDetails(
                latency_ms=latency_ms,
                provider_metadata=dict(response.metadata),
            )
        aggregates = aggregate_run(results)
        with connection:
            persist_completed_run(
                connection,
                run_id=run_id,
                results=results,
                generation_details=generation_details,
                aggregates=aggregates,
                timestamp=datetime.now(UTC),
            )
        return results
    except Exception as error:
        _record_operational_failure(
            connection=connection,
            experiment_id=experiment_id,
            active_run_id=run_id,
            stage=stage,
            error=error,
        )


def _record_operational_failure(
    *,
    connection: sqlite3.Connection,
    experiment_id: str,
    active_run_id: str | None,
    stage: str,
    error: Exception,
) -> Never:
    message = _error_message(error)
    try:
        with connection:
            fail_attempt(
                connection,
                experiment_id=experiment_id,
                active_run_id=active_run_id,
                stage=stage,
                message=message,
                timestamp=datetime.now(UTC),
            )
    except Exception as persistence_error:
        message = (
            f"{message}; could not persist failure: {_error_message(persistence_error)}"
        )
    raise ExperimentOperationalError(experiment_id, stage, message) from error


def _aggregate_scope(scope: str, results: list[CaseResult]) -> RunAggregate:
    passed_count = sum(result.passed for result in results)
    case_count = len(results)
    return RunAggregate(
        scope=scope,
        case_count=case_count,
        passed_count=passed_count,
        failed_count=case_count - passed_count,
        score=sum(result.score for result in results) / case_count,
        pass_rate=passed_count / case_count,
    )


def _comparison_scope(
    scope: str,
    comparisons: list[CaseComparison],
) -> ComparisonAggregate:
    case_count = len(comparisons)
    baseline_score = sum(item.baseline_score for item in comparisons) / case_count
    candidate_score = sum(item.candidate_score for item in comparisons) / case_count
    return ComparisonAggregate(
        scope=scope,
        case_count=case_count,
        baseline_score=baseline_score,
        candidate_score=candidate_score,
        score_delta=candidate_score - baseline_score,
        newly_passed_count=sum(
            item.transition is ComparisonTransition.NEWLY_PASSED for item in comparisons
        ),
        newly_failed_count=sum(
            item.transition is ComparisonTransition.NEWLY_FAILED for item in comparisons
        ),
    )


def _transition(
    baseline_passed: bool,
    candidate_passed: bool,
) -> ComparisonTransition:
    if baseline_passed and candidate_passed:
        return ComparisonTransition.PASSED_BOTH
    if not baseline_passed and not candidate_passed:
        return ComparisonTransition.FAILED_BOTH
    if candidate_passed:
        return ComparisonTransition.NEWLY_PASSED
    return ComparisonTransition.NEWLY_FAILED


def _error_message(error: Exception) -> str:
    return str(error).strip() or type(error).__name__
