"""Validated configuration and result models for experiment execution."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tracebench.models import EvaluationMode


class ExperimentStatus(StrEnum):
    """Lifecycle state for an experiment attempt."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentVerdict(StrEnum):
    """Regression decision for a completed experiment."""

    PASS = "PASS"
    FAIL = "FAIL"


class RunRole(StrEnum):
    """The configuration represented by an experiment run."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class RunStatus(StrEnum):
    """Lifecycle state for one side of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ComparisonTransition(StrEnum):
    """Pass/fail transition between baseline and candidate."""

    PASSED_BOTH = "passed_both"
    FAILED_BOTH = "failed_both"
    NEWLY_PASSED = "newly_passed"
    NEWLY_FAILED = "newly_failed"


class GateMetric(StrEnum):
    """Regression metrics supported by the gate."""

    SCORE_DROP = "score_drop"
    NEW_FAILURES = "new_failures"


class FixtureProviderConfig(BaseModel):
    """Configuration for the local fixture provider."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["fixture"]
    path: Path


class ModeThresholdOverride(BaseModel):
    """Optional threshold replacements for one present evaluation mode."""

    model_config = ConfigDict(extra="forbid")

    max_score_drop: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    max_new_failures: Annotated[int | None, Field(ge=0)] = None

    @model_validator(mode="after")
    def require_an_override(self) -> Self:
        """Reject a mode entry that changes no threshold."""
        if self.max_score_drop is None and self.max_new_failures is None:
            raise ValueError("must override at least one threshold")
        return self


class RegressionGateConfig(BaseModel):
    """Global regression limits and optional mode-specific overrides."""

    model_config = ConfigDict(extra="forbid")

    max_score_drop: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    max_new_failures: Annotated[int, Field(ge=0)] = 0
    by_mode: dict[EvaluationMode, ModeThresholdOverride] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level versioned experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    name: str
    dataset: str
    baseline: FixtureProviderConfig
    candidate: FixtureProviderConfig
    gate: RegressionGateConfig = Field(default_factory=RegressionGateConfig)

    @field_validator("name", "dataset")
    @classmethod
    def normalize_nonblank_text(cls, value: str) -> str:
        """Normalize labels and reject blank values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class EffectiveThresholds(BaseModel):
    """Fully materialized thresholds for one gate scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_score_drop: Annotated[float, Field(ge=0.0, le=1.0)]
    max_new_failures: Annotated[int, Field(ge=0)]


class ScorerResult(BaseModel):
    """One deterministic scorer outcome."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    """Provider output and combined scoring outcome for one case."""

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    evaluation_mode: EvaluationMode
    output: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    passed: bool
    scorers: Annotated[list[ScorerResult], Field(min_length=1)]


class RunAggregate(BaseModel):
    """Authoritative aggregate for a run and one scope."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    case_count: Annotated[int, Field(gt=0)]
    passed_count: Annotated[int, Field(ge=0)]
    failed_count: Annotated[int, Field(ge=0)]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    pass_rate: Annotated[float, Field(ge=0.0, le=1.0)]


class CaseComparison(BaseModel):
    """Baseline-to-candidate comparison for one evaluation case."""

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    evaluation_mode: EvaluationMode
    baseline_score: Annotated[float, Field(ge=0.0, le=1.0)]
    candidate_score: Annotated[float, Field(ge=0.0, le=1.0)]
    score_delta: Annotated[float, Field(ge=-1.0, le=1.0)]
    baseline_passed: bool
    candidate_passed: bool
    transition: ComparisonTransition


class ComparisonAggregate(BaseModel):
    """Aggregate comparison for the global or a mode scope."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    case_count: Annotated[int, Field(gt=0)]
    baseline_score: Annotated[float, Field(ge=0.0, le=1.0)]
    candidate_score: Annotated[float, Field(ge=0.0, le=1.0)]
    score_delta: Annotated[float, Field(ge=-1.0, le=1.0)]
    newly_passed_count: Annotated[int, Field(ge=0)]
    newly_failed_count: Annotated[int, Field(ge=0)]


class GateViolation(BaseModel):
    """One exceeded regression threshold."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    metric: GateMetric
    actual: Annotated[float, Field(ge=0.0)]
    allowed: Annotated[float, Field(ge=0.0)]
    message: str


class RunReport(BaseModel):
    """Machine-readable report for one completed run."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    run_id: str
    role: RunRole
    global_: RunAggregate = Field(alias="global", serialization_alias="global")
    by_mode: dict[str, RunAggregate]
    cases: list[CaseResult]


class ComparisonReport(BaseModel):
    """Machine-readable baseline-to-candidate comparison."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    global_: ComparisonAggregate = Field(alias="global", serialization_alias="global")
    by_mode: dict[str, ComparisonAggregate]
    cases: list[CaseComparison]
    newly_passed: list[str]
    newly_failed: list[str]


class GateReport(BaseModel):
    """Machine-readable regression gate result."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    violations: list[GateViolation]


class DatasetIdentity(BaseModel):
    """Dataset identity recorded in an experiment report."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    name: str
    version: str


class ExperimentReport(BaseModel):
    """Complete result reconstructed from normalized experiment records."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    experiment_id: str
    name: str
    configuration_hash: str
    dataset: DatasetIdentity
    status: ExperimentStatus
    verdict: ExperimentVerdict | None
    runs: dict[RunRole, RunReport]
    comparison: ComparisonReport | None
    gate: GateReport | None
    failure_stage: str | None = None
    failure_message: str | None = None
