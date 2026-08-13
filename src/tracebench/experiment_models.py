"""Validated configuration and result models for experiment execution."""

import re
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

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


class JudgeCacheStatus(StrEnum):
    """Cache lookup outcome for one rubric judge result."""

    HIT = "hit"
    MISS = "miss"
    NOT_RECORDED = "not_recorded"


class JudgeReviewStatus(StrEnum):
    """Human-review state for one rubric judge result."""

    NOT_REQUIRED = "not_required"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"


class JudgeReviewReason(StrEnum):
    """Reason a rubric judge result requires or received review."""

    LOW_CONFIDENCE = "low_confidence"
    CRITICAL_FAILURE = "critical_failure"


@dataclass(frozen=True, slots=True)
class GenerationDetails:
    """Persistable observations from one provider call."""

    latency_ms: float
    provider_metadata: dict[str, Any]


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


class OllamaProviderConfig(BaseModel):
    """Configuration for an Ollama-compatible local provider."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["ollama"]
    base_url: str = "http://localhost:11434"
    model: str
    prompt_version: str
    system_prompt_file: Path
    temperature: Annotated[float, Field(ge=0.0)] = 0.0
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 120.0
    seed: Annotated[int | None, Field(ge=0)] = None

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        """Require an unauthenticated HTTP server root."""
        candidate = value.strip()
        try:
            parsed = urlsplit(candidate)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError as error:
            raise ValueError("must contain a valid hostname and port") from error
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("must not contain a query string or fragment")
        if hostname is None or not _is_valid_hostname(hostname):
            raise ValueError("must contain a valid hostname")
        if port is None or port == 0:
            raise ValueError("must contain a valid port between 1 and 65535")
        if parsed.path not in {"", "/"}:
            raise ValueError("path must be empty or '/'")
        return candidate[:-1] if parsed.path == "/" else candidate

    @field_validator("model", "prompt_version")
    @classmethod
    def normalize_nonblank_provider_text(cls, value: str) -> str:
        """Normalize required provider labels."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("system_prompt_file", mode="before")
    @classmethod
    def reject_blank_prompt_path(cls, value: object) -> object:
        """Reject values that Path would otherwise coerce to the current directory."""
        if isinstance(value, (str, Path)) and not str(value).strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("temperature", "timeout_seconds", "seed", mode="before")
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        """Keep YAML booleans from silently becoming numeric options."""
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value


ProviderConfig = Annotated[
    FixtureProviderConfig | OllamaProviderConfig,
    Field(discriminator="provider"),
]


class FixtureJudgeConfig(BaseModel):
    """Configuration for deterministic fixture-based rubric judging."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["fixture"]
    path: Path
    prompt_version: str
    prompt_file: Path
    retry_prompt_file: Path
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("prompt_version")
    @classmethod
    def normalize_prompt_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("path", "prompt_file", "retry_prompt_file", mode="before")
    @classmethod
    def reject_blank_paths(cls, value: object) -> object:
        if isinstance(value, (str, Path)) and not str(value).strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("confidence_threshold", mode="before")
    @classmethod
    def validate_confidence_threshold(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number, not a boolean or string")
        return value


class OllamaJudgeConfig(BaseModel):
    """Configuration for rubric judging through an Ollama-compatible server."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["ollama"]
    base_url: str = "http://localhost:11434"
    model: str
    prompt_version: str
    prompt_file: Path
    retry_prompt_file: Path
    temperature: Annotated[float, Field(ge=0.0)] = 0.0
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 120.0
    seed: Annotated[int | None, Field(ge=0)] = None
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return OllamaProviderConfig.normalize_base_url(value)

    @field_validator("model", "prompt_version")
    @classmethod
    def normalize_nonblank_text(cls, value: str) -> str:
        return OllamaProviderConfig.normalize_nonblank_provider_text(value)

    @field_validator("prompt_file", "retry_prompt_file", mode="before")
    @classmethod
    def reject_blank_prompt_paths(cls, value: object) -> object:
        if isinstance(value, (str, Path)) and not str(value).strip():
            raise ValueError("must not be blank")
        return value

    @field_validator(
        "temperature",
        "timeout_seconds",
        "seed",
        mode="before",
    )
    @classmethod
    def reject_boolean_numbers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value

    @field_validator("confidence_threshold", mode="before")
    @classmethod
    def validate_confidence_threshold(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number, not a boolean or string")
        return value


JudgeConfig = Annotated[
    FixtureJudgeConfig | OllamaJudgeConfig,
    Field(discriminator="provider"),
]


def _is_valid_hostname(hostname: str) -> bool:
    try:
        ip_address(hostname)
    except ValueError:
        if "." in hostname and all(part.isdigit() for part in hostname.split(".")):
            return False
        try:
            ascii_hostname = hostname.encode("idna").decode("ascii").removesuffix(".")
        except UnicodeError:
            return False
        if not ascii_hostname or len(ascii_hostname) > 253:
            return False
        return all(
            1 <= len(label) <= 63
            and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            is not None
            for label in ascii_hostname.split(".")
        )
    return True


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
    baseline: ProviderConfig
    candidate: ProviderConfig
    judge: JudgeConfig | None = None
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


class JudgeCacheMetadata(BaseModel):
    """Persisted cache lookup metadata exposed for one rubric result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str | None
    status: JudgeCacheStatus

    @model_validator(mode="after")
    def require_status_specific_key(self) -> Self:
        if self.status is JudgeCacheStatus.NOT_RECORDED:
            if self.key is not None:
                raise ValueError("not-recorded cache metadata forbids a key")
            return self
        if self.key is None or re.fullmatch(r"[0-9a-f]{64}", self.key) is None:
            raise ValueError("cache hits and misses require a lowercase SHA-256 key")
        return self


class JudgeReviewMetadata(BaseModel):
    """Human-review state and its deterministic escalation reasons."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: JudgeReviewStatus
    reasons: list[JudgeReviewReason]

    @model_validator(mode="after")
    def require_status_specific_reasons(self) -> Self:
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("review reasons must be unique")
        if self.status is JudgeReviewStatus.NOT_REQUIRED and self.reasons:
            raise ValueError("not-required review metadata forbids reasons")
        if self.status is not JudgeReviewStatus.NOT_REQUIRED and not self.reasons:
            raise ValueError("reviewed results require at least one reason")
        return self


class JudgeCaseResult(BaseModel):
    """Validated judge-level fields for one rubric case."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    response_schema_version: Literal[1] = 1
    attempt_count: Annotated[int, Field(ge=0, le=2)]
    overall_score: Annotated[float, Field(ge=0.0, le=1.0)]
    overall_passed: bool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    below_confidence_threshold: bool
    cache_hit: bool | None = None
    cache: JudgeCacheMetadata = Field(
        default_factory=lambda: JudgeCacheMetadata(
            key=None,
            status=JudgeCacheStatus.NOT_RECORDED,
        )
    )
    review: JudgeReviewMetadata

    @model_validator(mode="before")
    @classmethod
    def default_additive_review_metadata(cls, value: Any) -> Any:
        """Keep A2 constructors valid while deriving the new additive field."""
        if not isinstance(value, dict) or "review" in value:
            return value
        prepared = dict(value)
        reasons = (
            [JudgeReviewReason.LOW_CONFIDENCE]
            if prepared.get("below_confidence_threshold") is True
            else []
        )
        prepared["review"] = {
            "status": (
                JudgeReviewStatus.NEEDS_REVIEW
                if reasons
                else JudgeReviewStatus.NOT_REQUIRED
            ),
            "reasons": reasons,
        }
        return prepared

    @model_validator(mode="after")
    def require_consistent_cache_metadata(self) -> Self:
        if self.cache_hit is True:
            if self.cache.status is not JudgeCacheStatus.HIT or self.attempt_count != 0:
                raise ValueError("cache hits require hit metadata and zero attempts")
        elif self.cache_hit is False:
            if self.cache.status is not JudgeCacheStatus.MISS:
                raise ValueError("cache misses require miss metadata")
            if self.attempt_count not in (1, 2):
                raise ValueError("cache misses require one or two attempts")
        else:
            if self.cache.status is not JudgeCacheStatus.NOT_RECORDED:
                raise ValueError("unknown cache state requires not-recorded metadata")
            if self.attempt_count not in (1, 2):
                raise ValueError("historical results require one or two attempts")
        has_low_confidence_reason = (
            JudgeReviewReason.LOW_CONFIDENCE in self.review.reasons
        )
        if has_low_confidence_reason is not self.below_confidence_threshold:
            raise ValueError("low-confidence review metadata is inconsistent")
        if (
            JudgeReviewReason.CRITICAL_FAILURE in self.review.reasons
            and self.overall_passed
        ):
            raise ValueError("critical-failure review metadata requires failure")
        return self


class CaseResult(BaseModel):
    """Provider output and combined scoring outcome for one case."""

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    evaluation_mode: EvaluationMode
    output: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    passed: bool
    scorers: Annotated[list[ScorerResult], Field(min_length=1)]
    judge: JudgeCaseResult | None = None

    @model_validator(mode="after")
    def require_mode_specific_result(self) -> Self:
        if self.evaluation_mode is EvaluationMode.RUBRIC and self.judge is None:
            raise ValueError("rubric result requires judge metadata")
        if self.evaluation_mode is not EvaluationMode.RUBRIC and self.judge is not None:
            raise ValueError("non-rubric result forbids judge metadata")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_judge(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("case result must serialize as an object")
        if self.judge is None:
            serialized.pop("judge", None)
        return serialized


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
