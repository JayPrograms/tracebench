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
    JudgeCacheStatus,
    RunAggregate,
    RunRole,
)
from tracebench.experiment_storage import (
    JudgeCacheEntry,
    fail_attempt,
    insert_attempt,
    insert_judge_cache_entry,
    load_experiment_report,
    load_judge_cache_entry,
    persist_completed_comparison,
    persist_completed_run,
    persist_judge_attempt,
    persist_judge_cache_lookup,
    start_run,
)
from tracebench.judges import (
    JUDGE_CACHE_KEY_VERSION,
    JUDGE_RESPONSE_SCHEMA_VERSION,
    JudgeOutputError,
    build_judge_cache_identity,
    build_judge_prompt,
    build_judge_retry_prompt,
    judge_request_id,
    materialize_judge_result,
    request_hash,
    validate_cached_judge_output,
    validated_judge_output,
)
from tracebench.models import EvalCase, EvaluationMode
from tracebench.providers import ProviderError, ProviderRequest, build_prompt
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
                    judge_snapshot=(
                        prepared.judge.snapshot if prepared.judge is not None else None
                    ),
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
            system_prompt = prepared.provider_prompts[role]
            prompt = "" if system_prompt is None else build_prompt(case, system_prompt)
            response = provider.generate(
                ProviderRequest(request_id=case.eval_id, prompt=prompt)
            )
            latency_ms = max(0.0, (perf_counter_ns() - started_at) / 1_000_000)
            if case.evaluation_mode is EvaluationMode.RUBRIC:
                results.append(
                    _judge_case(
                        connection=connection,
                        prepared=prepared,
                        run_id=run_id,
                        role=role,
                        case=case,
                        output=response.output,
                    )
                )
            else:
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


def _judge_case(
    *,
    connection: sqlite3.Connection,
    prepared: PreparedExperiment,
    run_id: str,
    role: RunRole,
    case: EvalCase,
    output: str,
) -> CaseResult:
    judge = prepared.judge
    if judge is None:
        raise ValueError("rubric case has no prepared judge")
    judge_request = judge_request_id(role, case.eval_id)
    cache_identity = build_judge_cache_identity(
        case,
        output,
        prompt_version=judge.prompt_version,
        prompt_hash=judge.prompt_hash,
        retry_prompt_hash=judge.retry_prompt_hash,
        provider_identity=judge.provider_identity_for(judge_request),
    )
    cache_entry = _lookup_and_record_judge_cache(
        connection=connection,
        run_id=run_id,
        case=case,
        cache_key=cache_identity.cache_key,
    )
    if cache_entry is not None:
        cached_response = validate_cached_judge_output(
            case,
            cache_identity,
            cache_key=cache_entry.cache_key,
            key_version=cache_entry.key_version,
            identity_json=cache_entry.identity_json,
            response_schema_version=cache_entry.response_schema_version,
            response_json=cache_entry.response_json,
            response_hash=cache_entry.response_hash,
            raw_output=cache_entry.raw_output,
        )
        return materialize_judge_result(
            case,
            cached_response,
            evaluated_output=output,
            confidence_threshold=judge.confidence_threshold,
            attempt_count=0,
            cache_hit=True,
            cache_key=cache_identity.cache_key,
        )
    previous_output: str | None = None
    previous_error: JudgeOutputError | None = None
    for attempt_number in (1, 2):
        if attempt_number == 1:
            prompt = build_judge_prompt(case, output, judge.prompt)
        else:
            if previous_output is None or previous_error is None:
                raise AssertionError(
                    "judge retry is missing the prior malformed output"
                )
            prompt = build_judge_retry_prompt(
                case,
                output,
                judge.prompt,
                judge.retry_prompt,
                previous_output,
                previous_error.code,
            )
        started_at = perf_counter_ns()
        try:
            response = judge.provider.generate(
                ProviderRequest(
                    request_id=judge_request,
                    prompt=prompt,
                    response_format="json",
                )
            )
        except ProviderError as error:
            raise ProviderError(
                f"judge request for evaluation case '{case.eval_id}' failed: {error}"
            ) from error
        latency_ms = max(0.0, (perf_counter_ns() - started_at) / 1_000_000)
        try:
            validated = validated_judge_output(case, response.output)
        except JudgeOutputError as error:
            with connection:
                persist_judge_attempt(
                    connection,
                    run_id=run_id,
                    eval_id=case.eval_id,
                    attempt_number=attempt_number,
                    request_hash=request_hash(prompt),
                    raw_output=response.output,
                    parse_status="malformed",
                    validation_error=error.diagnostic(),
                    latency_ms=latency_ms,
                    provider_metadata=dict(response.metadata),
                    timestamp=datetime.now(UTC),
                )
            if attempt_number == 2:
                raise ProviderError(
                    f"judge output for evaluation case '{case.eval_id}' remained "
                    f"malformed after 2 attempts: {error.code}"
                ) from error
            previous_output = response.output
            previous_error = error
            continue
        with connection:
            persist_judge_attempt(
                connection,
                run_id=run_id,
                eval_id=case.eval_id,
                attempt_number=attempt_number,
                request_hash=request_hash(prompt),
                raw_output=response.output,
                parse_status="parsed",
                validation_error=None,
                latency_ms=latency_ms,
                provider_metadata=dict(response.metadata),
                timestamp=datetime.now(UTC),
            )
        with connection:
            insert_judge_cache_entry(
                connection,
                cache_key=cache_identity.cache_key,
                key_version=JUDGE_CACHE_KEY_VERSION,
                identity_json=cache_identity.identity_json,
                response_schema_version=JUDGE_RESPONSE_SCHEMA_VERSION,
                response_json=validated.response_json,
                response_hash=validated.response_hash,
                raw_output=validated.raw_output,
                timestamp=datetime.now(UTC),
            )
        winning_entry = load_judge_cache_entry(
            connection,
            cache_identity.cache_key,
        )
        if winning_entry is None:
            raise sqlite3.IntegrityError(
                "judge cache insertion conflict has no winning cache entry"
            )
        validate_cached_judge_output(
            case,
            cache_identity,
            cache_key=winning_entry.cache_key,
            key_version=winning_entry.key_version,
            identity_json=winning_entry.identity_json,
            response_schema_version=winning_entry.response_schema_version,
            response_json=winning_entry.response_json,
            response_hash=winning_entry.response_hash,
            raw_output=winning_entry.raw_output,
        )
        return materialize_judge_result(
            case,
            validated.response,
            evaluated_output=output,
            confidence_threshold=judge.confidence_threshold,
            attempt_count=attempt_number,
            cache_hit=False,
            cache_key=cache_identity.cache_key,
        )
    raise AssertionError("judge attempts exhausted without a result")


def _lookup_and_record_judge_cache(
    *,
    connection: sqlite3.Connection,
    run_id: str,
    case: EvalCase,
    cache_key: str,
) -> JudgeCacheEntry | None:
    """Atomically record a short cache lookup without spanning provider work."""
    try:
        connection.execute("BEGIN IMMEDIATE")
        entry = load_judge_cache_entry(connection, cache_key)
        persist_judge_cache_lookup(
            connection,
            run_id=run_id,
            eval_id=case.eval_id,
            cache_key=cache_key,
            cache_status=(
                JudgeCacheStatus.HIT if entry is not None else JudgeCacheStatus.MISS
            ),
            timestamp=datetime.now(UTC),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return entry


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
