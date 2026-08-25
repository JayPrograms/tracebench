"""Read-only, dashboard-ready detail exports for persisted experiments.

The machine report remains the authority for scores, transitions, aggregates,
and gate decisions.  This module only joins that report with the normalized
records needed by a future dashboard; it deliberately never re-evaluates a
case or exposes raw judge attempts or prompt contents.
"""

import json
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from tracebench.experiment_models import (
    CaseComparison,
    CaseResult,
    ComparisonTransition,
    ExperimentReport,
    ExperimentStatus,
    ExperimentVerdict,
    GateViolation,
    JudgeCaseResult,
    RunRole,
    RunStatus,
    ScorerResult,
)
from tracebench.experiment_storage import load_experiment_report
from tracebench.models import (
    EvalCase,
    EvaluationMode,
    Priority,
    ReviewStatus,
    SliceBuildSource,
    SliceCaseProvenance,
)
from tracebench.storage import connect_database, list_eval_cases


class ExperimentDetailError(ValueError):
    """Raised when an experiment detail cannot be reconstructed safely."""


class ExperimentDetailTimestamps(BaseModel):
    """Persisted lifecycle timestamps for an experiment attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    created_at: AwareDatetime
    started_at: AwareDatetime
    completed_at: AwareDatetime | None


class ProviderSnapshot(BaseModel):
    """A sanitized, persisted provider configuration snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provider: str
    configuration: dict[str, Any] = Field(default_factory=dict)

    @field_validator("configuration")
    @classmethod
    def require_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_finite_json(value, "provider configuration")
        return value


class JudgeSnapshot(BaseModel):
    """Persisted judge configuration without prompt text or raw responses."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provider: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str
    prompt_hash: str
    retry_prompt_hash: str
    response_schema_version: int
    max_malformed_retries: int
    confidence_threshold: float

    @field_validator("configuration")
    @classmethod
    def require_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_finite_json(value, "judge configuration")
        return value


class DatasetDetail(BaseModel):
    """Dataset metadata and immutable slice-build provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dataset_id: str
    name: str
    version: str
    description: str
    created_at: AwareDatetime
    case_count: int = Field(ge=0)
    sealed: bool
    slice_source: SliceBuildSource | None


class GateThresholdOverride(BaseModel):
    """One optional persisted mode or slice threshold override."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_score_drop: float | None
    max_new_failures: int | None


class GateThresholds(BaseModel):
    """The configured global and scoped regression limits."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    max_score_drop: float
    max_new_failures: int
    by_mode: dict[str, GateThresholdOverride]
    by_slice: dict[str, GateThresholdOverride]


class GateDetail(BaseModel):
    """Configured gate limits alongside the authoritative persisted result."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    thresholds: GateThresholds
    passed: bool | None
    violations: list[GateViolation]


class GenerationObservation(BaseModel):
    """Persisted client-observed generation details for one output."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    latency_ms: float = Field(ge=0.0)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    observed_at: AwareDatetime

    @field_validator("provider_metadata")
    @classmethod
    def require_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_finite_json(value, "provider metadata")
        return value


class RunCaseDetail(BaseModel):
    """One persisted baseline or candidate result joined with observations."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    run_id: str
    output: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    scorers: list[ScorerResult]
    judge: JudgeCaseResult | None
    generation: GenerationObservation


class SourceTraceDetail(BaseModel):
    """The trace snapshot retained by an evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    timestamp: AwareDatetime
    task_type: str
    metadata: dict[str, Any]
    response: str | None = None

    @field_validator("metadata")
    @classmethod
    def require_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_finite_json(value, "trace metadata")
        return value

    @model_serializer(mode="wrap")
    def omit_non_reference_response(self, handler: Any) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("source trace must serialize as an object")
        if self.response is None:
            serialized.pop("response", None)
        return serialized


class CaseDetail(BaseModel):
    """One combined case view, ordered by stable evaluation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    eval_id: str
    source_trace: SourceTraceDetail
    input: str
    context: dict[str, Any]
    evaluation_mode: EvaluationMode
    priority: Priority
    review_status: ReviewStatus
    reference_answer: str | None = None
    scorers: list[dict[str, Any]]
    rubric: list[str]
    slice_provenance: SliceCaseProvenance | None
    baseline: RunCaseDetail | None
    candidate: RunCaseDetail | None
    comparison: CaseComparison | None
    transition: ComparisonTransition | None
    score_delta: float | None

    @field_validator("context")
    @classmethod
    def require_finite_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_finite_json(value, "case context")
        return value

    @field_validator("scorers")
    @classmethod
    def require_finite_scorer_json(
        cls, value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        _require_finite_json(value, "case scorers")
        return value

    @model_validator(mode="after")
    def validate_answer_exposure(self) -> Self:
        if self.evaluation_mode is not EvaluationMode.REFERENCE:
            if (
                self.reference_answer is not None
                or self.source_trace.response is not None
            ):
                raise ValueError("non-reference detail case exposes an answer")
        elif self.reference_answer is None:
            raise ValueError("reference detail case is missing its answer snapshot")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_answer(self, handler: Any) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("case detail must serialize as an object")
        if self.reference_answer is None:
            serialized.pop("reference_answer", None)
        return serialized


class ExperimentDetail(BaseModel):
    """Strict, versioned, read-only reconstruction of one experiment."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, populate_by_name=True
    )

    schema_version: int = Field(default=1, strict=True)
    experiment_id: str
    name: str
    status: ExperimentStatus
    verdict: ExperimentVerdict | None
    configuration_hash: str
    timestamps: ExperimentDetailTimestamps
    failure_stage: str | None
    failure_message: str | None
    dataset: DatasetDetail
    provider_snapshots: dict[str, ProviderSnapshot]
    judge_snapshot: JudgeSnapshot | None
    gate: GateDetail
    runs: list["RunDetail"]
    cases: list[CaseDetail]
    report: ExperimentReport

    @field_validator("schema_version")
    @classmethod
    def require_current_schema(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported experiment detail schema version")
        return value


class RunDetail(BaseModel):
    """Persisted lifecycle metadata for one experiment role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    role: RunRole
    provider_name: str
    status: RunStatus
    error_message: str | None
    started_at: AwareDatetime | None
    completed_at: AwareDatetime | None


ExperimentDetail.model_rebuild()


def get_experiment_detail(database_path: Path, experiment_id: str) -> ExperimentDetail:
    """Reconstruct a persisted experiment detail without changing the database."""
    normalized_id = experiment_id.strip()
    if not normalized_id:
        raise ExperimentDetailError("experiment id must not be blank")
    with closing(connect_database(database_path)) as connection:
        return _load_detail(connection, normalized_id)


def load_experiment_detail(database_path: Path, experiment_id: str) -> ExperimentDetail:
    """Compatibility alias for callers that prefer a loader verb."""
    return get_experiment_detail(database_path, experiment_id)


def export_experiment_detail(
    database_path: Path,
    experiment_id: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> ExperimentDetail:
    """Write one deterministic UTF-8 JSON detail export atomically."""
    detail = get_experiment_detail(database_path, experiment_id)
    parent = output_path.parent
    if not parent.exists():
        raise OSError(f"output directory does not exist: {parent}")
    if not parent.is_dir():
        raise OSError(f"output parent is not a directory: {parent}")
    if output_path.is_dir():
        raise OSError(f"output path is a directory: {output_path}")
    payload = json.dumps(
        detail.model_dump(mode="json", by_alias=True),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if overwrite:
        _write_replacing(output_path, payload)
    else:
        _write_exclusive(output_path, payload)
    return detail


def _load_detail(
    connection: sqlite3.Connection, experiment_id: str
) -> ExperimentDetail:
    """Join normalized records while checking relationships before validation."""
    attempt = connection.execute(
        """
        SELECT e.*, d.name AS dataset_name, d.version AS dataset_version,
               d.description AS dataset_description, d.created_at AS dataset_created_at
        FROM experiments AS e
        JOIN eval_datasets AS d ON d.dataset_id = e.dataset_id
        WHERE e.experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    if attempt is None:
        raise ExperimentDetailError(f"experiment '{experiment_id}' was not found")
    try:
        report = load_experiment_report(connection, experiment_id)
        runs = _load_runs(connection, experiment_id)
        cases = _load_cases(connection, attempt["dataset_id"], report)
        provider_snapshots = _load_provider_snapshots(connection, experiment_id)
        judge_snapshot = _load_judge_snapshot(connection, experiment_id)
        gate = _load_gate_detail(attempt["configuration_json"], report)
        source = report.dataset.slice_source
        dataset = DatasetDetail(
            dataset_id=attempt["dataset_id"],
            name=attempt["dataset_name"],
            version=attempt["dataset_version"],
            description=attempt["dataset_description"],
            created_at=attempt["dataset_created_at"],
            case_count=len(cases),
            sealed=source is not None,
            slice_source=source,
        )
        timestamps = ExperimentDetailTimestamps(
            created_at=attempt["created_at"],
            started_at=attempt["started_at"],
            completed_at=attempt["completed_at"],
        )
        return ExperimentDetail(
            experiment_id=attempt["experiment_id"],
            name=attempt["name"],
            status=attempt["status"],
            verdict=attempt["verdict"],
            configuration_hash=attempt["configuration_hash"],
            timestamps=timestamps,
            failure_stage=attempt["failure_stage"],
            failure_message=attempt["failure_message"],
            dataset=dataset,
            provider_snapshots=provider_snapshots,
            judge_snapshot=judge_snapshot,
            gate=gate,
            runs=runs,
            cases=cases,
            report=report,
        )
    except ExperimentDetailError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        sqlite3.Error,
    ) as error:
        raise ExperimentDetailError(
            f"experiment '{experiment_id}' contains inconsistent persisted data: "
            f"{error}"
        ) from error


def _load_runs(connection: sqlite3.Connection, experiment_id: str) -> list[RunDetail]:
    rows = connection.execute(
        """
        SELECT run_id, role, provider_name, status, error_message,
               started_at, completed_at
        FROM experiment_runs
        WHERE experiment_id = ?
        ORDER BY CASE role WHEN 'baseline' THEN 0 ELSE 1 END, run_id ASC
        """,
        (experiment_id,),
    ).fetchall()
    seen: set[str] = set()
    result: list[RunDetail] = []
    for row in rows:
        role = str(row["role"])
        if role in seen:
            raise ExperimentDetailError(f"duplicate persisted run role '{role}'")
        seen.add(role)
        result.append(
            RunDetail(
                run_id=row["run_id"],
                role=RunRole(role),
                provider_name=row["provider_name"],
                status=RunStatus(row["status"]),
                error_message=row["error_message"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
            )
        )
    if seen != {role.value for role in RunRole}:
        raise ExperimentDetailError(
            "experiment run records must contain exactly baseline and candidate roles"
        )
    return result


def _load_provider_snapshots(
    connection: sqlite3.Connection, experiment_id: str
) -> dict[str, ProviderSnapshot]:
    rows = connection.execute(
        """
        SELECT role, provider_name, provider_config_json
        FROM experiment_runs
        WHERE experiment_id = ?
        ORDER BY CASE role WHEN 'baseline' THEN 0 ELSE 1 END, role ASC
        """,
        (experiment_id,),
    ).fetchall()
    snapshots: dict[str, ProviderSnapshot] = {}
    for row in rows:
        configuration = _sanitize_snapshot(_decode_object(row["provider_config_json"]))
        snapshots[str(row["role"])] = ProviderSnapshot(
            provider=row["provider_name"], configuration=configuration
        )
    return snapshots


def _load_judge_snapshot(
    connection: sqlite3.Connection, experiment_id: str
) -> JudgeSnapshot | None:
    row = connection.execute(
        """
        SELECT provider_name, provider_config_json, prompt_version, prompt_hash,
               retry_prompt_hash, response_schema_version, max_malformed_retries,
               confidence_threshold
        FROM experiment_judges WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchone()
    if row is None:
        return None
    return JudgeSnapshot(
        provider=row["provider_name"],
        configuration=_sanitize_snapshot(_decode_object(row["provider_config_json"])),
        prompt_version=row["prompt_version"],
        prompt_hash=row["prompt_hash"],
        retry_prompt_hash=row["retry_prompt_hash"],
        response_schema_version=row["response_schema_version"],
        max_malformed_retries=row["max_malformed_retries"],
        confidence_threshold=row["confidence_threshold"],
    )


def _load_cases(
    connection: sqlite3.Connection,
    dataset_id: str,
    report: ExperimentReport,
) -> list[CaseDetail]:
    cases = list_eval_cases(connection, dataset_id)
    expected_ids = {case.eval_id for case in cases}
    report_ids = {
        result.eval_id for run in report.runs.values() for result in run.cases
    }
    if not report_ids.issubset(expected_ids):
        raise ExperimentDetailError("experiment result references an unknown eval case")
    comparison_ids = (
        set()
        if report.comparison is None
        else {item.eval_id for item in report.comparison.cases}
    )
    if not comparison_ids.issubset(expected_ids):
        raise ExperimentDetailError(
            "experiment comparison references an unknown eval case"
        )
    if report.status is ExperimentStatus.COMPLETED:
        if set(report.runs) != {RunRole.BASELINE, RunRole.CANDIDATE}:
            raise ExperimentDetailError(
                "completed experiment is missing a baseline or candidate report"
            )
        if report_ids != expected_ids or comparison_ids != expected_ids:
            raise ExperimentDetailError(
                "completed experiment does not cover every dataset case"
            )
    baseline = report.runs.get(RunRole.BASELINE)
    candidate = report.runs.get(RunRole.CANDIDATE)
    baseline_by_id = (
        {} if baseline is None else {item.eval_id: item for item in baseline.cases}
    )
    candidate_by_id = (
        {} if candidate is None else {item.eval_id: item for item in candidate.cases}
    )
    comparison_by_id = (
        {}
        if report.comparison is None
        else {item.eval_id: item for item in report.comparison.cases}
    )
    result_details = _load_generation_details(connection, report)
    return [
        _case_detail(
            case,
            baseline_by_id.get(case.eval_id),
            candidate_by_id.get(case.eval_id),
            comparison_by_id.get(case.eval_id),
            result_details,
        )
        for case in cases
    ]


def _case_detail(
    case: EvalCase,
    baseline: CaseResult | None,
    candidate: CaseResult | None,
    comparison: CaseComparison | None,
    generation: dict[tuple[str, str], tuple[str, GenerationObservation]],
) -> CaseDetail:
    baseline_detail = _run_case_detail(RunRole.BASELINE, baseline, generation)
    candidate_detail = _run_case_detail(RunRole.CANDIDATE, candidate, generation)
    for result in (baseline, candidate):
        if result is not None and result.evaluation_mode is not case.evaluation_mode:
            raise ExperimentDetailError(
                f"evaluation mode mismatch for case '{case.eval_id}'"
            )
    return CaseDetail(
        eval_id=case.eval_id,
        source_trace=SourceTraceDetail(
            trace_id=case.source_trace_id,
            timestamp=case.source_timestamp,
            task_type=case.source_task_type,
            metadata=_sanitize_snapshot(case.source_metadata),
            response=case.source_response,
        ),
        input=case.input,
        context=_sanitize_snapshot(case.context),
        evaluation_mode=case.evaluation_mode,
        priority=case.priority,
        review_status=case.review_status,
        reference_answer=case.reference_answer,
        scorers=[
            _sanitize_snapshot(scorer.model_dump(mode="json"))
            for scorer in case.scorers
        ],
        rubric=list(case.rubric),
        slice_provenance=case.slice_provenance,
        baseline=baseline_detail,
        candidate=candidate_detail,
        comparison=comparison,
        transition=None if comparison is None else comparison.transition,
        score_delta=None if comparison is None else comparison.score_delta,
    )


def _run_case_detail(
    role: RunRole,
    result: CaseResult | None,
    generation: dict[tuple[str, str], tuple[str, GenerationObservation]],
) -> RunCaseDetail | None:
    if result is None:
        return None
    persisted = generation.get((role.value, result.eval_id))
    if persisted is None:
        raise ExperimentDetailError(
            f"missing generation observation for {role.value}:{result.eval_id}"
        )
    run_id, observation = persisted
    return RunCaseDetail(
        run_id=run_id,
        output=result.output,
        score=result.score,
        passed=result.passed,
        scorers=result.scorers,
        judge=result.judge,
        generation=observation,
    )


def _load_generation_details(
    connection: sqlite3.Connection, report: ExperimentReport
) -> dict[tuple[str, str], tuple[str, GenerationObservation]]:
    run_ids = [run.run_id for run in report.runs.values()]
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    rows = connection.execute(
        f"""
        SELECT run.role, result.eval_id, result.generation_latency_ms,
               result.provider_metadata_json, result.created_at, run.run_id
        FROM experiment_case_results AS result
        JOIN experiment_runs AS run ON run.run_id = result.run_id
        WHERE result.run_id IN ({placeholders})
        ORDER BY CASE run.role WHEN 'baseline' THEN 0 ELSE 1 END,
                 result.eval_id ASC
        """,
        tuple(run_ids),
    ).fetchall()
    details: dict[tuple[str, str], tuple[str, GenerationObservation]] = {}
    for row in rows:
        if row["generation_latency_ms"] is None:
            raise ExperimentDetailError(
                f"missing generation latency for {row['role']}:{row['eval_id']}"
            )
        observation = GenerationObservation(
            latency_ms=row["generation_latency_ms"],
            provider_metadata=_sanitize_snapshot(
                _decode_object(row["provider_metadata_json"])
            ),
            observed_at=row["created_at"],
        )
        details[(str(row["role"]), str(row["eval_id"]))] = (
            str(row["run_id"]),
            observation,
        )
    return details


def _decode_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ExperimentDetailError("stored JSON value is not an object")
    return decoded


def _load_gate_detail(configuration_json: str, report: ExperimentReport) -> GateDetail:
    configuration = _decode_object(configuration_json)
    raw_gate = configuration.get("gate", {})
    if not isinstance(raw_gate, dict):
        raise ExperimentDetailError("persisted experiment gate is not an object")

    raw_global = raw_gate.get("global", {})
    if not isinstance(raw_global, dict):
        raise ExperimentDetailError("persisted global gate is not an object")

    def override(value: Any) -> GateThresholdOverride:
        if not isinstance(value, dict):
            raise ExperimentDetailError("persisted gate override is not an object")
        return GateThresholdOverride(
            max_score_drop=value.get("max_score_drop"),
            max_new_failures=value.get("max_new_failures"),
        )

    by_mode_raw = {
        key: value
        for key, value in raw_gate.items()
        if key not in {"global", "by_slice"}
    }
    by_slice_raw = raw_gate.get("by_slice", {})
    if not isinstance(by_mode_raw, dict) or not isinstance(by_slice_raw, dict):
        raise ExperimentDetailError("persisted gate scopes are not objects")
    thresholds = GateThresholds(
        max_score_drop=raw_global.get("max_score_drop", 0.0),
        max_new_failures=raw_global.get("max_new_failures", 0),
        by_mode={
            str(key): override(value) for key, value in sorted(by_mode_raw.items())
        },
        by_slice={
            str(key): override(value) for key, value in sorted(by_slice_raw.items())
        },
    )
    return GateDetail(
        thresholds=thresholds,
        passed=None if report.gate is None else report.gate.passed,
        violations=[] if report.gate is None else report.gate.violations,
    )


def _sanitize_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove credential-like keys while retaining reproducibility metadata."""
    secret_markers = ("secret", "password", "token", "api_key", "apikey", "auth")

    def clean(item: Any, key: str | None = None) -> Any:
        if key is not None and any(marker in key.lower() for marker in secret_markers):
            return "[redacted]"
        if isinstance(item, dict):
            return {str(k): clean(v, str(k)) for k, v in item.items()}
        if isinstance(item, list):
            return [clean(v) for v in item]
        return item

    result = clean(dict(value))
    if not isinstance(result, dict):
        raise ExperimentDetailError("provider snapshot is not an object")
    return result


def _require_finite_json(value: object, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON values") from error


def _write_exclusive(path: Path, payload: str) -> None:
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            created = True
            output.write(payload)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise ExperimentDetailError(f"output file already exists: {path}") from error
    except BaseException:
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _write_replacing(path: Path, payload: str) -> None:
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(payload)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
