"""Normalized SQLite persistence for experiment attempts and results."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from tracebench.experiment_models import (
    CaseComparison,
    CaseResult,
    ComparisonAggregate,
    ComparisonReport,
    ComparisonTransition,
    DatasetIdentity,
    ExperimentReport,
    ExperimentStatus,
    ExperimentVerdict,
    GateMetric,
    GateReport,
    GateViolation,
    GenerationDetails,
    JudgeCacheMetadata,
    JudgeCacheStatus,
    JudgeCaseResult,
    JudgeReviewMetadata,
    JudgeReviewReason,
    JudgeReviewStatus,
    RunAggregate,
    RunReport,
    RunRole,
    ScorerResult,
    SliceComparisonAggregate,
    SliceRunAggregate,
)
from tracebench.providers import JsonValue
from tracebench.storage import get_dataset_slice_build, timestamp_to_text


@dataclass(frozen=True, slots=True)
class JudgeCacheEntry:
    """One immutable cached raw and canonical successful judge response."""

    cache_key: str
    key_version: int
    identity_json: str
    response_schema_version: int
    response_json: str
    response_hash: str
    raw_output: str


def insert_attempt(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    name: str,
    dataset_id: str,
    configuration_hash: str,
    configuration_json: str,
    run_ids: dict[RunRole, str],
    provider_snapshots: dict[RunRole, dict[str, JsonValue]],
    timestamp: datetime,
    judge_snapshot: dict[str, JsonValue] | None = None,
) -> None:
    """Insert one running attempt and its two pending run records."""
    timestamp_text = timestamp_to_text(timestamp)
    connection.execute(
        """
        INSERT INTO experiments (
            experiment_id, name, dataset_id, configuration_hash,
            configuration_json, status, verdict, failure_stage,
            failure_message, created_at, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, 'running', NULL, NULL, NULL, ?, ?, NULL)
        """,
        (
            experiment_id,
            name,
            dataset_id,
            configuration_hash,
            configuration_json,
            timestamp_text,
            timestamp_text,
        ),
    )
    for role in RunRole:
        snapshot = provider_snapshots[role]
        provider_name = snapshot.get("provider")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError(f"provider snapshot for {role.value} has no provider")
        connection.execute(
            """
            INSERT INTO experiment_runs (
                run_id, experiment_id, role, provider_name,
                provider_config_json, status, error_message,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, NULL)
            """,
            (
                run_ids[role],
                experiment_id,
                role.value,
                provider_name,
                _encode_json(snapshot),
            ),
        )
    if judge_snapshot is not None:
        provider_name = judge_snapshot.get("provider")
        prompt_version = judge_snapshot.get("prompt_version")
        prompt_hash = judge_snapshot.get("prompt_hash")
        retry_prompt_hash = judge_snapshot.get("retry_prompt_hash")
        response_schema_version = judge_snapshot.get("response_schema_version")
        max_malformed_retries = judge_snapshot.get("max_malformed_retries")
        confidence_threshold = judge_snapshot.get("confidence_threshold")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("judge snapshot has no provider")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("judge snapshot has no prompt version")
        connection.execute(
            """
            INSERT INTO experiment_judges (
                experiment_id, provider_name, provider_config_json,
                prompt_version, prompt_hash, retry_prompt_hash,
                response_schema_version, max_malformed_retries,
                confidence_threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                provider_name,
                _encode_json(judge_snapshot),
                prompt_version,
                prompt_hash,
                retry_prompt_hash,
                response_schema_version,
                max_malformed_retries,
                confidence_threshold,
            ),
        )


def persist_judge_attempt(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    eval_id: str,
    attempt_number: int,
    request_hash: str,
    raw_output: str,
    parse_status: str,
    validation_error: str | None,
    latency_ms: float,
    provider_metadata: dict[str, JsonValue],
    timestamp: datetime,
) -> None:
    """Persist one received raw judge response independently of run results."""
    connection.execute(
        """
        INSERT INTO experiment_judge_attempts (
            run_id, eval_id, attempt_number, request_hash, raw_output,
            parse_status, validation_error, latency_ms,
            provider_metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            eval_id,
            attempt_number,
            request_hash,
            raw_output,
            parse_status,
            validation_error,
            latency_ms,
            _encode_json(provider_metadata),
            timestamp_to_text(timestamp),
        ),
    )


def load_judge_cache_entry(
    connection: sqlite3.Connection,
    cache_key: str,
) -> JudgeCacheEntry | None:
    """Load one immutable cached judge response by request identity."""
    row = connection.execute(
        """
        SELECT
            cache_key, key_version, identity_json,
            response_schema_version, response_json,
            response_hash, raw_output
        FROM judge_result_cache
        WHERE cache_key = ?
        """,
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    return JudgeCacheEntry(
        cache_key=row["cache_key"],
        key_version=row["key_version"],
        identity_json=row["identity_json"],
        response_schema_version=row["response_schema_version"],
        response_json=row["response_json"],
        response_hash=row["response_hash"],
        raw_output=row["raw_output"],
    )


def insert_judge_cache_entry(
    connection: sqlite3.Connection,
    *,
    cache_key: str,
    key_version: int,
    identity_json: str,
    response_schema_version: int,
    response_json: str,
    response_hash: str,
    raw_output: str,
    timestamp: datetime,
) -> bool:
    """Insert a validated cache entry without replacing an existing winner."""
    cursor = connection.execute(
        """
        INSERT INTO judge_result_cache (
            cache_key, key_version, identity_json,
            response_schema_version, response_json,
            response_hash, raw_output, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT DO NOTHING
        """,
        (
            cache_key,
            key_version,
            identity_json,
            response_schema_version,
            response_json,
            response_hash,
            raw_output,
            timestamp_to_text(timestamp),
        ),
    )
    return cursor.rowcount == 1


def persist_judge_cache_lookup(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    eval_id: str,
    cache_key: str,
    cache_status: JudgeCacheStatus,
    timestamp: datetime,
) -> None:
    """Persist the cache lookup outcome before any possible provider call."""
    if cache_status is JudgeCacheStatus.NOT_RECORDED:
        raise ValueError("new judge cache lookups must be hits or misses")
    connection.execute(
        """
        INSERT INTO experiment_judge_cache_lookups (
            run_id, eval_id, cache_key, cache_status, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            run_id,
            eval_id,
            cache_key,
            cache_status.value,
            timestamp_to_text(timestamp),
        ),
    )


def start_run(
    connection: sqlite3.Connection,
    run_id: str,
    timestamp: datetime,
) -> None:
    """Transition a pending run to running."""
    cursor = connection.execute(
        """
        UPDATE experiment_runs
        SET status = 'running', started_at = ?
        WHERE run_id = ? AND status = 'pending'
        """,
        (timestamp_to_text(timestamp), run_id),
    )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError(f"run '{run_id}' is not pending")


def persist_completed_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    results: list[CaseResult],
    generation_details: dict[str, GenerationDetails],
    aggregates: list[RunAggregate],
    slice_aggregates: list[SliceRunAggregate],
    timestamp: datetime,
) -> None:
    """Atomically insert all normalized run rows and complete the run."""
    result_ids = {result.eval_id for result in results}
    if result_ids != set(generation_details):
        raise ValueError("generation details do not match run results")
    created_at = timestamp_to_text(timestamp)
    for result in results:
        details = generation_details[result.eval_id]
        connection.execute(
            """
            INSERT INTO experiment_case_results (
                run_id, eval_id, evaluation_mode, output,
                score, passed, generation_latency_ms,
                provider_metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result.eval_id,
                result.evaluation_mode.value,
                result.output,
                result.score,
                int(result.passed),
                details.latency_ms,
                _encode_json(details.provider_metadata),
                created_at,
            ),
        )
        for scorer_index, scorer in enumerate(result.scorers):
            connection.execute(
                """
                INSERT INTO experiment_scorer_results (
                    run_id, eval_id, scorer_index, scorer_name,
                    score, passed, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.eval_id,
                    scorer_index,
                    scorer.name,
                    scorer.score,
                    int(scorer.passed),
                    _encode_json(scorer.details),
                ),
            )
        if result.judge is not None:
            judge = result.judge
            connection.execute(
                """
                INSERT INTO experiment_judge_results (
                    run_id, eval_id, final_attempt_number,
                    response_schema_version, overall_score, overall_passed,
                    confidence, confidence_threshold,
                    below_confidence_threshold, cache_hit,
                    critical_priority_failure, review_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    result.eval_id,
                    (None if judge.cache_hit is True else judge.attempt_count),
                    judge.response_schema_version,
                    judge.overall_score,
                    int(judge.overall_passed),
                    judge.confidence,
                    judge.confidence_threshold,
                    int(judge.below_confidence_threshold),
                    (None if judge.cache_hit is None else int(judge.cache_hit)),
                    int(JudgeReviewReason.CRITICAL_FAILURE in judge.review.reasons),
                    judge.review.status.value,
                ),
            )
    for aggregate in aggregates:
        connection.execute(
            """
            INSERT INTO experiment_run_aggregates (
                run_id, scope, case_count, passed_count,
                failed_count, score, pass_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                aggregate.scope,
                aggregate.case_count,
                aggregate.passed_count,
                aggregate.failed_count,
                aggregate.score,
                aggregate.pass_rate,
            ),
        )
    for slice_aggregate in slice_aggregates:
        connection.execute(
            """
            INSERT INTO experiment_run_slice_aggregates (
                run_id, cluster_number, selector, label_snapshot,
                case_count, passed_count, failed_count, score, pass_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                slice_aggregate.cluster_number,
                slice_aggregate.selector,
                slice_aggregate.label_snapshot,
                slice_aggregate.case_count,
                slice_aggregate.passed_count,
                slice_aggregate.failed_count,
                slice_aggregate.score,
                slice_aggregate.pass_rate,
            ),
        )
    cursor = connection.execute(
        """
        UPDATE experiment_runs
        SET status = 'completed', completed_at = ?
        WHERE run_id = ? AND status = 'running'
        """,
        (created_at, run_id),
    )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError(f"run '{run_id}' is not running")


def persist_completed_comparison(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    comparisons: list[CaseComparison],
    aggregates: list[ComparisonAggregate],
    slice_aggregates: list[SliceComparisonAggregate],
    violations: list[GateViolation],
    verdict: ExperimentVerdict,
    timestamp: datetime,
) -> None:
    """Atomically persist comparison, gate, and completed verdict."""
    for comparison in comparisons:
        connection.execute(
            """
            INSERT INTO experiment_case_comparisons (
                experiment_id, eval_id, evaluation_mode,
                baseline_score, candidate_score, score_delta,
                baseline_passed, candidate_passed, transition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                comparison.eval_id,
                comparison.evaluation_mode.value,
                comparison.baseline_score,
                comparison.candidate_score,
                comparison.score_delta,
                int(comparison.baseline_passed),
                int(comparison.candidate_passed),
                comparison.transition.value,
            ),
        )
    for aggregate in aggregates:
        connection.execute(
            """
            INSERT INTO experiment_comparison_aggregates (
                experiment_id, scope, case_count,
                baseline_score, candidate_score, score_delta,
                newly_passed_count, newly_failed_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                aggregate.scope,
                aggregate.case_count,
                aggregate.baseline_score,
                aggregate.candidate_score,
                aggregate.score_delta,
                aggregate.newly_passed_count,
                aggregate.newly_failed_count,
            ),
        )
    for slice_aggregate in slice_aggregates:
        connection.execute(
            """
            INSERT INTO experiment_comparison_slice_aggregates (
                experiment_id, cluster_number, selector, label_snapshot,
                case_count, baseline_score, candidate_score, score_delta,
                baseline_pass_rate, candidate_pass_rate,
                newly_passed_count, newly_failed_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                slice_aggregate.cluster_number,
                slice_aggregate.selector,
                slice_aggregate.label_snapshot,
                slice_aggregate.case_count,
                slice_aggregate.baseline_score,
                slice_aggregate.candidate_score,
                slice_aggregate.score_delta,
                slice_aggregate.baseline_pass_rate,
                slice_aggregate.candidate_pass_rate,
                slice_aggregate.newly_passed_count,
                slice_aggregate.newly_failed_count,
            ),
        )
    for violation_index, violation in enumerate(violations):
        connection.execute(
            """
            INSERT INTO experiment_gate_violations (
                experiment_id, violation_index, scope, scope_kind,
                cluster_number, label_snapshot, metric,
                actual, allowed, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                violation_index,
                violation.scope,
                violation.scope_kind,
                violation.cluster_number,
                violation.label_snapshot,
                violation.metric.value,
                violation.actual,
                violation.allowed,
                violation.message,
            ),
        )
    cursor = connection.execute(
        """
        UPDATE experiments
        SET status = 'completed', verdict = ?, completed_at = ?
        WHERE experiment_id = ? AND status = 'running'
        """,
        (verdict.value, timestamp_to_text(timestamp), experiment_id),
    )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError(f"experiment '{experiment_id}' is not running")


def fail_attempt(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    active_run_id: str | None,
    stage: str,
    message: str,
    timestamp: datetime,
) -> None:
    """Persist an operational failure after the failed stage rolled back."""
    timestamp_text = timestamp_to_text(timestamp)
    if active_run_id is not None:
        connection.execute(
            """
            UPDATE experiment_runs
            SET status = 'failed', error_message = ?, completed_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (message, timestamp_text, active_run_id),
        )
    connection.execute(
        """
        UPDATE experiment_runs
        SET status = 'skipped', error_message = ?, completed_at = ?
        WHERE experiment_id = ? AND status = 'pending'
        """,
        (f"skipped after {stage} failure", timestamp_text, experiment_id),
    )
    cursor = connection.execute(
        """
        UPDATE experiments
        SET status = 'failed', verdict = NULL, failure_stage = ?,
            failure_message = ?, completed_at = ?
        WHERE experiment_id = ? AND status = 'running'
        """,
        (stage, message, timestamp_text, experiment_id),
    )
    if cursor.rowcount != 1:
        raise sqlite3.IntegrityError(f"experiment '{experiment_id}' is not running")


def load_experiment_report(
    connection: sqlite3.Connection,
    experiment_id: str,
) -> ExperimentReport:
    """Reconstruct a report exclusively from normalized database records."""
    attempt = connection.execute(
        """
        SELECT
            e.experiment_id, e.name, e.configuration_hash,
            e.status, e.verdict, e.failure_stage, e.failure_message,
            d.dataset_id, d.name AS dataset_name, d.version AS dataset_version
        FROM experiments AS e
        JOIN eval_datasets AS d ON d.dataset_id = e.dataset_id
        WHERE e.experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    if attempt is None:
        raise ValueError(f"experiment '{experiment_id}' was not found")

    runs: dict[RunRole, RunReport] = {}
    run_rows = connection.execute(
        """
        SELECT run_id, role
        FROM experiment_runs
        WHERE experiment_id = ? AND status = 'completed'
        ORDER BY role ASC
        """,
        (experiment_id,),
    ).fetchall()
    for run_row in run_rows:
        role = RunRole(run_row["role"])
        runs[role] = _load_run_report(connection, run_row["run_id"], role)

    comparison_rows = connection.execute(
        """
        SELECT
            eval_id, evaluation_mode, baseline_score, candidate_score,
            score_delta, baseline_passed, candidate_passed, transition
        FROM experiment_case_comparisons
        WHERE experiment_id = ?
        ORDER BY eval_id ASC
        """,
        (experiment_id,),
    ).fetchall()
    comparison: ComparisonReport | None = None
    gate: GateReport | None = None
    if comparison_rows:
        case_comparisons = [
            CaseComparison(
                eval_id=row["eval_id"],
                evaluation_mode=row["evaluation_mode"],
                baseline_score=row["baseline_score"],
                candidate_score=row["candidate_score"],
                score_delta=row["score_delta"],
                baseline_passed=bool(row["baseline_passed"]),
                candidate_passed=bool(row["candidate_passed"]),
                transition=ComparisonTransition(row["transition"]),
            )
            for row in comparison_rows
        ]
        aggregate_rows = connection.execute(
            """
            SELECT
                scope, case_count, baseline_score, candidate_score,
                score_delta, newly_passed_count, newly_failed_count
            FROM experiment_comparison_aggregates
            WHERE experiment_id = ?
            ORDER BY CASE scope WHEN 'global' THEN 0 ELSE 1 END, scope ASC
            """,
            (experiment_id,),
        ).fetchall()
        comparison_aggregates = {
            row["scope"]: ComparisonAggregate.model_validate(dict(row))
            for row in aggregate_rows
        }
        global_comparison = comparison_aggregates.pop("global")
        membership_rows = connection.execute(
            """
            SELECT provenance.eval_id, provenance.cluster_number
            FROM eval_case_slice_provenance AS provenance
            JOIN experiments AS experiment
              ON experiment.dataset_id = provenance.dataset_id
            WHERE experiment.experiment_id = ?
            """,
            (experiment_id,),
        ).fetchall()
        membership = {
            str(row["eval_id"]): int(row["cluster_number"]) for row in membership_rows
        }
        slice_rows = connection.execute(
            """
            SELECT * FROM experiment_comparison_slice_aggregates
            WHERE experiment_id = ? ORDER BY cluster_number ASC
            """,
            (experiment_id,),
        ).fetchall()
        slice_aggregates: dict[str, SliceComparisonAggregate] = {}
        for row in slice_rows:
            cluster_number = int(row["cluster_number"])
            items = [
                item
                for item in case_comparisons
                if membership.get(item.eval_id) == cluster_number
            ]
            newly_passed = sorted(
                item.eval_id
                for item in items
                if item.transition is ComparisonTransition.NEWLY_PASSED
            )
            newly_failed = sorted(
                item.eval_id
                for item in items
                if item.transition is ComparisonTransition.NEWLY_FAILED
            )
            aggregate = SliceComparisonAggregate(
                selector=row["selector"],
                cluster_number=cluster_number,
                label_snapshot=row["label_snapshot"],
                case_count=row["case_count"],
                baseline_score=row["baseline_score"],
                candidate_score=row["candidate_score"],
                score_delta=row["score_delta"],
                baseline_pass_rate=row["baseline_pass_rate"],
                candidate_pass_rate=row["candidate_pass_rate"],
                newly_passed_count=row["newly_passed_count"],
                newly_failed_count=row["newly_failed_count"],
                newly_passed=newly_passed,
                newly_failed=newly_failed,
            )
            slice_aggregates[aggregate.selector] = aggregate
        comparison = ComparisonReport.model_validate(
            {
                "global": global_comparison,
                "by_mode": comparison_aggregates,
                "cases": case_comparisons,
                "newly_passed": [
                    case.eval_id
                    for case in case_comparisons
                    if case.transition is ComparisonTransition.NEWLY_PASSED
                ],
                "newly_failed": [
                    case.eval_id
                    for case in case_comparisons
                    if case.transition is ComparisonTransition.NEWLY_FAILED
                ],
                "by_slice": slice_aggregates,
            }
        )
        violation_rows = connection.execute(
            """
            SELECT scope, scope_kind, cluster_number, label_snapshot,
                   metric, actual, allowed, message
            FROM experiment_gate_violations
            WHERE experiment_id = ?
            ORDER BY violation_index ASC
            """,
            (experiment_id,),
        ).fetchall()
        violations = [
            GateViolation(
                scope=row["scope"],
                metric=GateMetric(row["metric"]),
                actual=row["actual"],
                allowed=row["allowed"],
                message=row["message"],
                scope_kind=row["scope_kind"],
                cluster_number=row["cluster_number"],
                label_snapshot=row["label_snapshot"],
            )
            for row in violation_rows
        ]
        gate = GateReport(passed=not violations, violations=violations)

    slice_source = get_dataset_slice_build(connection, attempt["dataset_id"])
    return ExperimentReport(
        schema_version=2 if slice_source is not None else 1,
        experiment_id=attempt["experiment_id"],
        name=attempt["name"],
        configuration_hash=attempt["configuration_hash"],
        dataset=DatasetIdentity(
            dataset_id=attempt["dataset_id"],
            name=attempt["dataset_name"],
            version=attempt["dataset_version"],
            slice_source=slice_source,
        ),
        status=ExperimentStatus(attempt["status"]),
        verdict=(
            ExperimentVerdict(attempt["verdict"])
            if attempt["verdict"] is not None
            else None
        ),
        runs=runs,
        comparison=comparison,
        gate=gate,
        failure_stage=attempt["failure_stage"],
        failure_message=attempt["failure_message"],
    )


def _load_run_report(
    connection: sqlite3.Connection,
    run_id: str,
    role: RunRole,
) -> RunReport:
    result_rows = connection.execute(
        """
        SELECT eval_id, evaluation_mode, output, score, passed
        FROM experiment_case_results
        WHERE run_id = ?
        ORDER BY eval_id ASC
        """,
        (run_id,),
    ).fetchall()
    results: list[CaseResult] = []
    for result_row in result_rows:
        scorer_rows = connection.execute(
            """
            SELECT scorer_name, score, passed, details_json
            FROM experiment_scorer_results
            WHERE run_id = ? AND eval_id = ?
            ORDER BY scorer_index ASC
            """,
            (run_id, result_row["eval_id"]),
        ).fetchall()
        judge_row = connection.execute(
            """
            SELECT
                result.final_attempt_number,
                result.response_schema_version,
                result.overall_score,
                result.overall_passed,
                result.confidence,
                result.confidence_threshold,
                result.below_confidence_threshold,
                result.cache_hit,
                result.critical_priority_failure,
                result.review_status,
                lookup.cache_key,
                lookup.cache_status
            FROM experiment_judge_results AS result
            JOIN experiment_judge_cache_lookups AS lookup
                ON lookup.run_id = result.run_id
                AND lookup.eval_id = result.eval_id
            WHERE result.run_id = ? AND result.eval_id = ?
            """,
            (run_id, result_row["eval_id"]),
        ).fetchone()
        results.append(
            CaseResult(
                eval_id=result_row["eval_id"],
                evaluation_mode=result_row["evaluation_mode"],
                output=result_row["output"],
                score=result_row["score"],
                passed=bool(result_row["passed"]),
                scorers=[
                    ScorerResult(
                        name=scorer_row["scorer_name"],
                        score=scorer_row["score"],
                        passed=bool(scorer_row["passed"]),
                        details=json.loads(scorer_row["details_json"]),
                    )
                    for scorer_row in scorer_rows
                ],
                judge=(
                    JudgeCaseResult(
                        response_schema_version=judge_row["response_schema_version"],
                        attempt_count=(
                            0
                            if judge_row["final_attempt_number"] is None
                            else judge_row["final_attempt_number"]
                        ),
                        overall_score=judge_row["overall_score"],
                        overall_passed=bool(judge_row["overall_passed"]),
                        confidence=judge_row["confidence"],
                        confidence_threshold=judge_row["confidence_threshold"],
                        below_confidence_threshold=bool(
                            judge_row["below_confidence_threshold"]
                        ),
                        cache_hit=(
                            None
                            if judge_row["cache_hit"] is None
                            else bool(judge_row["cache_hit"])
                        ),
                        cache=JudgeCacheMetadata(
                            key=judge_row["cache_key"],
                            status=JudgeCacheStatus(judge_row["cache_status"]),
                        ),
                        review=JudgeReviewMetadata(
                            status=JudgeReviewStatus(judge_row["review_status"]),
                            reasons=[
                                reason
                                for reason, applies in (
                                    (
                                        JudgeReviewReason.LOW_CONFIDENCE,
                                        bool(judge_row["below_confidence_threshold"]),
                                    ),
                                    (
                                        JudgeReviewReason.CRITICAL_FAILURE,
                                        bool(judge_row["critical_priority_failure"]),
                                    ),
                                )
                                if applies
                            ],
                        ),
                    )
                    if judge_row is not None
                    else None
                ),
            )
        )
    aggregate_rows = connection.execute(
        """
        SELECT
            scope, case_count, passed_count, failed_count, score, pass_rate
        FROM experiment_run_aggregates
        WHERE run_id = ?
        ORDER BY CASE scope WHEN 'global' THEN 0 ELSE 1 END, scope ASC
        """,
        (run_id,),
    ).fetchall()
    aggregates = {
        row["scope"]: RunAggregate.model_validate(dict(row)) for row in aggregate_rows
    }
    global_aggregate = aggregates.pop("global")
    slice_rows = connection.execute(
        """
        SELECT * FROM experiment_run_slice_aggregates
        WHERE run_id = ? ORDER BY cluster_number ASC
        """,
        (run_id,),
    ).fetchall()
    slice_aggregates = {
        str(row["selector"]): SliceRunAggregate(
            selector=row["selector"],
            cluster_number=row["cluster_number"],
            label_snapshot=row["label_snapshot"],
            case_count=row["case_count"],
            passed_count=row["passed_count"],
            failed_count=row["failed_count"],
            score=row["score"],
            pass_rate=row["pass_rate"],
        )
        for row in slice_rows
    }
    return RunReport.model_validate(
        {
            "run_id": run_id,
            "role": role,
            "global": global_aggregate,
            "by_mode": aggregates,
            "cases": results,
            "by_slice": slice_aggregates,
        }
    )


def _encode_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
