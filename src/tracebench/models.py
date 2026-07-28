"""Validated trace data models."""

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


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
