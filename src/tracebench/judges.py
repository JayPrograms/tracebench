"""Strict rubric-judge prompting, fixtures, validation, and scoring."""

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
)

from tracebench.experiment_models import (
    CaseResult,
    JudgeCaseResult,
    RunRole,
    ScorerResult,
)
from tracebench.models import EvalCase, EvaluationMode
from tracebench.providers import ProviderError

OVERALL_SCORE_ABS_TOLERANCE = 1e-6
JUDGE_RESPONSE_SCHEMA_VERSION = 1
MAX_MALFORMED_RETRIES = 1


class JudgeOutputError(Exception):
    """A received judge string did not satisfy the structured contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def diagnostic(self) -> str:
        return f"{self.code}: {self.message}"


class CriterionJudgment(BaseModel):
    """One criterion outcome returned by a judge."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    criterion: str
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    passed: StrictBool
    reason: str

    @field_validator("score", mode="before")
    @classmethod
    def require_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a finite JSON number")
        return value

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class StructuredJudgeResponse(BaseModel):
    """Strict response envelope required from every rubric judge."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1]
    criteria: Annotated[list[CriterionJudgment], Field(min_length=1)]
    overall_score: Annotated[float, Field(ge=0.0, le=1.0)]
    overall_passed: StrictBool
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("must be the integer 1")
        return value

    @field_validator("overall_score", "confidence", mode="before")
    @classmethod
    def require_numbers(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a finite JSON number")
        return value


def build_judge_prompt(case: EvalCase, output: str, prompt: str) -> str:
    """Append canonical untrusted judge data to the versioned primary prompt."""
    payload = _judge_input(case, output)
    return (
        f"{prompt}\n\n"
        "<TRACEBENCH_JUDGE_INPUT_JSON>\n"
        f"{_encode_canonical(payload)}\n"
        "</TRACEBENCH_JUDGE_INPUT_JSON>"
    )


def build_judge_retry_prompt(
    case: EvalCase,
    output: str,
    prompt: str,
    retry_prompt: str,
    previous_output: str,
    validation_error_code: str,
) -> str:
    """Build a repair request without interpolating untrusted values into text."""
    payload = {
        "judge_input": _judge_input(case, output),
        "previous_output": previous_output,
        "validation_error_code": validation_error_code,
    }
    return (
        f"{prompt}\n\n{retry_prompt}\n\n"
        "<TRACEBENCH_JUDGE_RETRY_JSON>\n"
        f"{_encode_canonical(payload)}\n"
        "</TRACEBENCH_JUDGE_RETRY_JSON>"
    )


def parse_judge_output(
    case: EvalCase,
    raw_output: str,
    *,
    evaluated_output: str,
    confidence_threshold: float,
    attempt_count: int,
) -> CaseResult:
    """Validate a judge response and convert it into the existing result shape."""
    if case.evaluation_mode is not EvaluationMode.RUBRIC:
        raise ValueError("only rubric cases can be judged")
    payload = _decode_strict_json(raw_output)
    if not isinstance(payload, dict):
        raise JudgeOutputError("root_type", "response root must be a JSON object")
    try:
        response = StructuredJudgeResponse.model_validate(payload)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise JudgeOutputError("schema_validation", details) from error

    expected = set(case.rubric)
    by_criterion: dict[str, CriterionJudgment] = {}
    duplicates: list[str] = []
    for result in response.criteria:
        if result.criterion in by_criterion:
            duplicates.append(result.criterion)
        else:
            by_criterion[result.criterion] = result
    actual = set(by_criterion)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if duplicates or missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(
                "missing criteria: " + ", ".join(repr(item) for item in missing)
            )
        if unknown:
            parts.append(
                "unknown criteria: " + ", ".join(repr(item) for item in unknown)
            )
        if duplicates:
            parts.append(
                "duplicate criteria: "
                + ", ".join(repr(item) for item in sorted(set(duplicates)))
            )
        raise JudgeOutputError("criterion_coverage", "; ".join(parts))

    ordered = [by_criterion[criterion] for criterion in case.rubric]
    computed_score = sum(item.score for item in ordered) / len(ordered)
    if not math.isclose(
        response.overall_score,
        computed_score,
        rel_tol=0.0,
        abs_tol=OVERALL_SCORE_ABS_TOLERANCE,
    ):
        raise JudgeOutputError(
            "overall_score_mismatch",
            f"overall_score {response.overall_score} does not match criterion mean "
            f"{computed_score}",
        )
    computed_passed = all(item.passed for item in ordered)
    if response.overall_passed is not computed_passed:
        raise JudgeOutputError(
            "overall_pass_mismatch",
            "overall_passed does not match all criterion pass values",
        )

    return CaseResult(
        eval_id=case.eval_id,
        evaluation_mode=case.evaluation_mode,
        output=evaluated_output,
        score=computed_score,
        passed=computed_passed,
        scorers=[
            ScorerResult(
                name="rubric",
                score=item.score,
                passed=item.passed,
                details={"criterion": item.criterion, "reason": item.reason},
            )
            for item in ordered
        ],
        judge=JudgeCaseResult(
            attempt_count=attempt_count,
            overall_score=response.overall_score,
            overall_passed=response.overall_passed,
            confidence=response.confidence,
            confidence_threshold=confidence_threshold,
            below_confidence_threshold=response.confidence < confidence_threshold,
        ),
    )


def load_judge_fixture(
    path: Path,
    expected_eval_ids: set[str],
) -> dict[str, tuple[str, ...]]:
    """Load strict baseline/candidate judge-response sequences."""
    outputs: dict[str, tuple[str, ...]] = {}
    try:
        with path.open(encoding="utf-8-sig") as fixture_file:
            for line_number, line in enumerate(fixture_file, start=1):
                payload = _decode_fixture_line(path, line_number, line)
                if set(payload) != {"eval_id", "role", "outputs"}:
                    raise ProviderError(
                        f"{path}: line {line_number}: record must contain only "
                        "eval_id, role, and outputs"
                    )
                eval_id = payload["eval_id"]
                role = payload["role"]
                values = payload["outputs"]
                if not isinstance(eval_id, str) or not eval_id.strip():
                    raise ProviderError(
                        f"{path}: line {line_number}: eval_id must be a nonblank string"
                    )
                if role not in {item.value for item in RunRole}:
                    raise ProviderError(
                        f"{path}: line {line_number}: role must be baseline or "
                        "candidate"
                    )
                if (
                    not isinstance(values, list)
                    or not 1 <= len(values) <= 2
                    or any(not isinstance(value, str) for value in values)
                ):
                    raise ProviderError(
                        f"{path}: line {line_number}: outputs must contain one or "
                        "two strings"
                    )
                request_id = judge_request_id(RunRole(role), eval_id)
                if request_id in outputs:
                    raise ProviderError(
                        f"{path}: line {line_number}: duplicate judge request "
                        f"'{request_id}'"
                    )
                outputs[request_id] = tuple(values)
    except (OSError, UnicodeError) as error:
        raise ProviderError(f"could not read judge fixture {path}: {error}") from error

    expected = {
        judge_request_id(role, eval_id)
        for role in RunRole
        for eval_id in expected_eval_ids
    }
    actual = set(outputs)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ProviderError(f"{path}: missing judge requests: {', '.join(missing)}")
    if unknown:
        raise ProviderError(f"{path}: unknown judge requests: {', '.join(unknown)}")
    return outputs


def judge_request_id(role: RunRole, eval_id: str) -> str:
    return f"{role.value}:{eval_id}"


def request_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _judge_input(case: EvalCase, output: str) -> dict[str, object]:
    return {
        "context": case.context,
        "criteria": list(case.rubric),
        "input": case.input,
        "output": output,
        "priority": case.priority.value,
    }


def _decode_strict_json(raw_output: str) -> object:
    try:
        return json.loads(
            raw_output,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        message = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise JudgeOutputError("invalid_json", message) from error


def _decode_fixture_line(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        message = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise ProviderError(
            f"{path}: line {line_number}: malformed JSON: {message}"
        ) from error
    if not isinstance(payload, dict):
        raise ProviderError(f"{path}: line {line_number}: record must be a JSON object")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"number {value} is outside the finite JSON number range")
    return parsed


def _encode_canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
