"""Deterministic scorer validation and execution."""

import json
import math
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tracebench.experiment_models import CaseResult, ScorerResult
from tracebench.models import EvalCase, EvaluationMode, ScorerConfig


class ScorerConfigurationError(Exception):
    """Raised when a stored deterministic scorer is not executable."""


class ExactMatchConfig(BaseModel):
    """Configuration for exact string equality."""

    model_config = ConfigDict(extra="forbid")

    expected: str
    case_sensitive: bool = True


class ContainsConfig(BaseModel):
    """Configuration for substring matching."""

    model_config = ConfigDict(extra="forbid")

    substring: str
    case_sensitive: bool = True

    @field_validator("substring")
    @classmethod
    def reject_blank_substring(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class RegexConfig(BaseModel):
    """Configuration for regular-expression search."""

    model_config = ConfigDict(extra="forbid")

    pattern: str
    case_sensitive: bool = True

    @field_validator("pattern")
    @classmethod
    def compile_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error
        return value


class JsonValidityConfig(BaseModel):
    """Empty configuration for strict JSON validity."""

    model_config = ConfigDict(extra="forbid")


class RequiredKeysConfig(BaseModel):
    """Configuration for required top-level JSON object keys."""

    model_config = ConfigDict(extra="forbid")

    keys: list[str] = Field(min_length=1)

    @field_validator("keys")
    @classmethod
    def normalize_keys(cls, value: list[str]) -> list[str]:
        normalized = [key.strip() for key in value]
        if any(not key for key in normalized):
            raise ValueError("keys must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("keys must not contain duplicates")
        return normalized


ScorerConfigModel = (
    ExactMatchConfig
    | ContainsConfig
    | RegexConfig
    | JsonValidityConfig
    | RequiredKeysConfig
)
ScorerEvaluator = Callable[[str, ScorerConfigModel], ScorerResult]

_CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "exact_match": ExactMatchConfig,
    "contains": ContainsConfig,
    "regex": RegexConfig,
    "json_validity": JsonValidityConfig,
    "required_keys": RequiredKeysConfig,
}


def validate_case_scorers(case: EvalCase) -> None:
    """Validate every scorer required to execute one case."""
    if case.evaluation_mode is EvaluationMode.RUBRIC:
        return
    if case.evaluation_mode is EvaluationMode.REFERENCE:
        return
    for scorer in case.scorers:
        _validated_config(scorer)


def score_case(case: EvalCase, output: str) -> CaseResult:
    """Apply all configured deterministic scorers to one provider output."""
    if case.evaluation_mode is EvaluationMode.REFERENCE:
        expected = case.reference_answer
        if expected is None:
            raise ScorerConfigurationError("reference case has no reference answer")
        scorer_results = [
            _score_exact_match(
                output,
                ExactMatchConfig(expected=expected, case_sensitive=True),
            )
        ]
    elif case.evaluation_mode is EvaluationMode.DETERMINISTIC:
        scorer_results = [_score_configured(output, scorer) for scorer in case.scorers]
    else:
        raise ScorerConfigurationError("rubric evaluation is not deterministic")

    score = sum(result.score for result in scorer_results) / len(scorer_results)
    return CaseResult(
        eval_id=case.eval_id,
        evaluation_mode=case.evaluation_mode,
        output=output,
        score=score,
        passed=all(result.passed for result in scorer_results),
        scorers=scorer_results,
    )


def _score_configured(output: str, scorer: ScorerConfig) -> ScorerResult:
    config = _validated_config(scorer)
    if isinstance(config, ExactMatchConfig):
        return _score_exact_match(output, config)
    if isinstance(config, ContainsConfig):
        expected = config.substring
        actual = output
        if not config.case_sensitive:
            expected = expected.casefold()
            actual = actual.casefold()
        passed = expected in actual
        return _boolean_result(
            "contains",
            passed,
            {"substring": config.substring, "case_sensitive": config.case_sensitive},
        )
    if isinstance(config, RegexConfig):
        flags = 0 if config.case_sensitive else re.IGNORECASE
        passed = re.search(config.pattern, output, flags=flags) is not None
        return _boolean_result(
            "regex",
            passed,
            {"pattern": config.pattern, "case_sensitive": config.case_sensitive},
        )
    if isinstance(config, JsonValidityConfig):
        try:
            _load_strict_json(output)
        except ValueError as error:
            return _boolean_result("json_validity", False, {"error": str(error)})
        return _boolean_result("json_validity", True, {})
    if isinstance(config, RequiredKeysConfig):
        try:
            parsed = _load_strict_json(output)
        except ValueError as error:
            return _boolean_result(
                "required_keys",
                False,
                {"required_keys": config.keys, "error": str(error)},
            )
        if not isinstance(parsed, dict):
            return _boolean_result(
                "required_keys",
                False,
                {"required_keys": config.keys, "error": "output is not a JSON object"},
            )
        missing = [key for key in config.keys if key not in parsed]
        return _boolean_result(
            "required_keys",
            not missing,
            {"required_keys": config.keys, "missing_keys": missing},
        )
    raise AssertionError("unreachable scorer configuration type")


def _validated_config(scorer: ScorerConfig) -> ScorerConfigModel:
    model = _CONFIG_MODELS.get(scorer.name)
    if model is None:
        raise ScorerConfigurationError(f"unknown scorer '{scorer.name}'")
    try:
        validated = model.model_validate(scorer.config)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise ScorerConfigurationError(
            f"invalid {scorer.name} configuration: {details}"
        ) from error
    if isinstance(
        validated,
        (
            ExactMatchConfig,
            ContainsConfig,
            RegexConfig,
            JsonValidityConfig,
            RequiredKeysConfig,
        ),
    ):
        return validated
    raise AssertionError("unreachable scorer configuration model")


def _score_exact_match(output: str, config: ExactMatchConfig) -> ScorerResult:
    expected = config.expected
    actual = output
    if not config.case_sensitive:
        expected = expected.casefold()
        actual = actual.casefold()
    return _boolean_result(
        "exact_match",
        actual == expected,
        {"expected": config.expected, "case_sensitive": config.case_sensitive},
    )


def _boolean_result(
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> ScorerResult:
    return ScorerResult(
        name=name,
        score=1.0 if passed else 0.0,
        passed=passed,
        details=details,
    )


def _load_strict_json(output: str) -> object:
    try:
        return json.loads(
            output,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(str(error)) from error


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"number {value} is outside the finite JSON number range")
    return parsed
