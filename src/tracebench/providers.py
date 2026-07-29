"""Provider protocol and local fixture implementation."""

import json
import math
from pathlib import Path
from typing import Protocol

from tracebench.models import EvalCase


class ProviderError(Exception):
    """Raised when provider configuration or execution is invalid."""


class Provider(Protocol):
    """Minimal synchronous output provider."""

    def generate(self, case: EvalCase) -> str:
        """Return an output for one evaluation case."""
        ...


class FixtureProvider:
    """Return prevalidated outputs addressed by stable evaluation ID."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self._outputs = dict(outputs)

    def generate(self, case: EvalCase) -> str:
        """Return the fixture output, including an intentionally empty string."""
        try:
            return self._outputs[case.eval_id]
        except KeyError as error:
            raise ProviderError(
                f"fixture has no output for evaluation case '{case.eval_id}'"
            ) from error


def load_fixture(path: Path, expected_eval_ids: set[str]) -> dict[str, str]:
    """Load and validate exact fixture coverage for a dataset."""
    outputs: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8-sig") as fixture_file:
            for line_number, line in enumerate(fixture_file, start=1):
                try:
                    payload = json.loads(
                        line,
                        parse_constant=_reject_non_finite_constant,
                        parse_float=_parse_finite_float,
                    )
                except ValueError as error:
                    message = (
                        error.msg
                        if isinstance(error, json.JSONDecodeError)
                        else str(error)
                    )
                    raise ProviderError(
                        f"{path}: line {line_number}: malformed JSON: {message}"
                    ) from error
                if not isinstance(payload, dict):
                    raise ProviderError(
                        f"{path}: line {line_number}: record must be a JSON object"
                    )
                if set(payload) != {"eval_id", "output"}:
                    raise ProviderError(
                        f"{path}: line {line_number}: record must contain only "
                        "eval_id and output"
                    )
                eval_id = payload["eval_id"]
                output = payload["output"]
                if not isinstance(eval_id, str) or not eval_id.strip():
                    raise ProviderError(
                        f"{path}: line {line_number}: eval_id must be a nonblank string"
                    )
                if not isinstance(output, str):
                    raise ProviderError(
                        f"{path}: line {line_number}: output must be a string"
                    )
                if eval_id in outputs:
                    raise ProviderError(
                        f"{path}: line {line_number}: duplicate eval_id '{eval_id}'"
                    )
                outputs[eval_id] = output
    except (OSError, UnicodeError) as error:
        raise ProviderError(f"could not read fixture {path}: {error}") from error

    actual_eval_ids = set(outputs)
    missing = sorted(expected_eval_ids - actual_eval_ids)
    unknown = sorted(actual_eval_ids - expected_eval_ids)
    if missing:
        raise ProviderError(f"{path}: missing evaluation IDs: {', '.join(missing)}")
    if unknown:
        raise ProviderError(f"{path}: unknown evaluation IDs: {', '.join(unknown)}")
    return outputs


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value} is not valid JSON")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"number {value} is outside the finite JSON number range")
    return parsed
