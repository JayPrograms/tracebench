"""Validated trace and evaluation dataset models."""

import json
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)


class Trace(BaseModel):
    """A single application trace."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    timestamp: AwareDatetime
    task_type: str
    prompt: str
    response: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("trace_id", "task_type", "prompt")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject values that contain only whitespace."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class EvaluationMode(StrEnum):
    """Supported evaluation strategies."""

    DETERMINISTIC = "deterministic"
    REFERENCE = "reference"
    RUBRIC = "rubric"


class Priority(StrEnum):
    """Evaluation case importance."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(StrEnum):
    """Manual review state for an evaluation case."""

    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class ScorerConfig(BaseModel):
    """Named configuration for a future deterministic scorer."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        """Normalize and reject blank scorer names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("config")
    @classmethod
    def validate_config_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require scorer configuration to contain valid finite JSON."""
        _validate_json_value(value)
        return value


class EvalDataset(BaseModel):
    """One explicitly versioned evaluation dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    name: str
    version: str
    description: str = ""
    created_at: AwareDatetime

    @field_validator("dataset_id")
    @classmethod
    def reject_blank_id(cls, value: str) -> str:
        """Reject a blank dataset identifier."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("name", "version")
    @classmethod
    def normalize_selector_part(cls, value: str) -> str:
        """Normalize values used in a name:version dataset reference."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        if ":" in normalized:
            raise ValueError("must not contain ':'")
        return normalized


class SliceBuildSource(BaseModel):
    """Immutable dataset-level provenance for a slice-built dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clustering_run_id: str
    clustering_run_name: str
    clustering_schema_version: int
    clustering_configuration_hash: str
    clustering_source_manifest_hash: str
    cluster_count: int
    sampling_schema_version: int
    sampling_algorithm: str
    requested_size: int
    sampled_size: int
    eligible_trace_count: int
    slice_manifest_hash: str
    slice_manifest: list[dict[str, Any]]
    built_at: AwareDatetime


class SliceCaseProvenance(BaseModel):
    """Clustering assignment and sampling snapshot for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selector: str
    cluster_number: int
    label_snapshot: str | None
    label_key_snapshot: str | None
    clustering_run_id: str
    clustering_run_name: str
    clustering_schema_version: int
    clustering_configuration_hash: str
    clustering_source_manifest_hash: str
    cluster_count: int
    source_trace_id: str
    source_timestamp: AwareDatetime
    source_trace_hash: str
    document_index: int
    document_hash: str
    sampling_schema_version: int
    sampling_algorithm: str
    requested_size: int
    sampled_size: int
    eligible_trace_count: int
    slice_availability: int
    slice_quota: int
    rank_within_slice: int
    selection_key: str
    allocation_key: str
    slice_manifest_hash: str


class EvalCase(BaseModel):
    """A stored trace promoted into an evaluation dataset."""

    model_config = ConfigDict(extra="forbid")

    eval_id: str
    dataset_id: str
    source_trace_id: str
    source_timestamp: AwareDatetime
    source_task_type: str
    source_response: str | None
    source_metadata: dict[str, Any]
    input: str
    context: dict[str, Any] = Field(default_factory=dict)
    evaluation_mode: EvaluationMode
    reference_answer: str | None = None
    rubric: list[str] = Field(default_factory=list)
    scorers: list[ScorerConfig] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    review_status: ReviewStatus = ReviewStatus.DRAFT
    created_at: AwareDatetime
    slice_provenance: SliceCaseProvenance | None = None

    @field_validator(
        "eval_id",
        "dataset_id",
        "source_trace_id",
        "source_task_type",
        "input",
    )
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject required text containing only whitespace."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("context", "source_metadata")
    @classmethod
    def validate_context_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Require snapshot JSON objects to contain finite values."""
        _validate_json_value(value)
        return value

    @field_validator("reference_answer")
    @classmethod
    def reject_blank_reference(cls, value: str | None) -> str | None:
        """Reject a supplied reference answer containing only whitespace."""
        if value is not None and not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("rubric")
    @classmethod
    def normalize_rubric(cls, value: list[str]) -> list[str]:
        """Normalize rubric criteria and reject blanks or duplicates."""
        normalized = [criterion.strip() for criterion in value]
        if any(not criterion for criterion in normalized):
            raise ValueError("criteria must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("criteria must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_evaluation_mode(self) -> Self:
        """Enforce the required data for the selected evaluation mode."""
        if self.evaluation_mode is EvaluationMode.DETERMINISTIC:
            if not self.scorers:
                raise ValueError("deterministic mode requires at least one scorer")
            if self.source_response is not None:
                raise ValueError("deterministic mode forbids a source response")
            if self.reference_answer is not None:
                raise ValueError("deterministic mode forbids a reference answer")
            if self.rubric:
                raise ValueError("deterministic mode forbids rubric criteria")
        elif self.evaluation_mode is EvaluationMode.REFERENCE:
            if self.reference_answer is None:
                raise ValueError("reference mode requires a reference answer")
            if self.scorers:
                raise ValueError("reference mode forbids deterministic scorers")
            if self.rubric:
                raise ValueError("reference mode forbids rubric criteria")
        else:
            if not self.rubric:
                raise ValueError("rubric mode requires at least one rubric criterion")
            if self.source_response is not None:
                raise ValueError("rubric mode forbids a source response")
            if self.reference_answer is not None:
                raise ValueError("rubric mode forbids a reference answer")
            if self.scorers:
                raise ValueError("rubric mode forbids deterministic scorers")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_slice_provenance(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        serialized = handler(self)
        if not isinstance(serialized, dict):
            raise TypeError("evaluation case must serialize as an object")
        if self.slice_provenance is None:
            serialized.pop("slice_provenance", None)
        return serialized


def _validate_json_value(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("must contain only finite JSON values") from error
