"""Provider protocol and local fixture implementation."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tracebench.models import EvalCase

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


class ProviderError(Exception):
    """Raised when provider configuration or execution is invalid."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Output and provider-specific metadata from one generation."""

    output: str
    metadata: dict[str, JsonValue]


class Provider(Protocol):
    """Minimal synchronous output provider."""

    def generate(self, case: EvalCase) -> ProviderResponse:
        """Return an output and metadata for one evaluation case."""
        ...


class FixtureProvider:
    """Return prevalidated outputs addressed by stable evaluation ID."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self._outputs = dict(outputs)

    def generate(self, case: EvalCase) -> ProviderResponse:
        """Return the fixture output, including an intentionally empty string."""
        try:
            return ProviderResponse(output=self._outputs[case.eval_id], metadata={})
        except KeyError as error:
            raise ProviderError(
                f"fixture has no output for evaluation case '{case.eval_id}'"
            ) from error


class OllamaProvider:
    """Generate outputs through an Ollama-compatible local HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        system_prompt: str,
        temperature: float,
        timeout_seconds: float,
        seed: int | None,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/api/generate"
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._seed = seed

    def generate(self, case: EvalCase) -> ProviderResponse:
        """Send one non-streaming generation request."""
        options: dict[str, JsonValue] = {"temperature": self._temperature}
        if self._seed is not None:
            options["seed"] = self._seed
        payload: dict[str, JsonValue] = {
            "model": self._model,
            "prompt": build_prompt(case, self._system_prompt),
            "stream": False,
            "options": options,
        }
        request = Request(
            self._endpoint,
            data=json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
        except HTTPError as error:
            detail = _http_error_detail(error)
            raise ProviderError(
                f"Ollama request for model '{self._model}' failed with HTTP "
                f"{error.code}: {detail}"
            ) from error
        except URLError as error:
            if _is_timeout(error.reason):
                raise self._timeout_error() from error
            raise ProviderError(
                f"could not connect to Ollama at {self._endpoint} for model "
                f"'{self._model}': {error.reason}"
            ) from error
        except TimeoutError as error:
            raise self._timeout_error() from error
        except OSError as error:
            raise ProviderError(
                f"could not connect to Ollama at {self._endpoint} for model "
                f"'{self._model}': {error}"
            ) from error

        response_payload = _decode_response(body, self._endpoint)
        return _provider_response(response_payload, self._endpoint)

    def _timeout_error(self) -> ProviderError:
        return ProviderError(
            f"Ollama request for model '{self._model}' timed out after "
            f"{self._timeout_seconds:g} seconds"
        )


def build_prompt(case: EvalCase, system_prompt: str) -> str:
    """Construct the stable prompt sent to local generation providers."""
    context = json.dumps(
        case.context,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        f"{system_prompt}\n\n"
        "<TRACEBENCH_EVALUATION_INPUT>\n"
        f"{case.input}\n"
        "</TRACEBENCH_EVALUATION_INPUT>\n\n"
        "<TRACEBENCH_CONTEXT_JSON>\n"
        f"{context}\n"
        "</TRACEBENCH_CONTEXT_JSON>"
    )


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


def _decode_response(body: bytes, endpoint: str) -> dict[str, object]:
    try:
        decoded = body.decode("utf-8")
        payload = json.loads(
            decoded,
            parse_constant=_reject_non_finite_constant,
            parse_float=_parse_finite_float,
        )
    except (UnicodeError, ValueError) as error:
        raise ProviderError(
            f"invalid JSON response from Ollama at {endpoint}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ProviderError(
            f"invalid response from Ollama at {endpoint}: expected a JSON object"
        )
    return payload


def _provider_response(payload: dict[str, object], endpoint: str) -> ProviderResponse:
    server_error = payload.get("error")
    if isinstance(server_error, str) and server_error.strip():
        raise ProviderError(f"Ollama at {endpoint} returned an error: {server_error}")
    output = payload.get("response")
    if not isinstance(output, str):
        raise ProviderError(
            f"invalid response from Ollama at {endpoint}: response must be a string"
        )
    if payload.get("done") is not True:
        raise ProviderError(
            f"invalid response from Ollama at {endpoint}: done must be true"
        )

    metadata: dict[str, JsonValue] = {"done": True}
    for field in ("model", "created_at", "done_reason"):
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ProviderError(
                f"invalid response from Ollama at {endpoint}: {field} must be a string"
            )
        metadata[field] = value
    for field in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProviderError(
                f"invalid response from Ollama at {endpoint}: "
                f"{field} must be a nonnegative integer"
            )
        metadata[field] = value
    return ProviderResponse(output=output, metadata=metadata)


def _http_error_detail(error: HTTPError) -> str:
    try:
        body = error.read()
        payload = json.loads(body.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return str(error.reason)
    if isinstance(payload, dict):
        message = payload.get("error")
        if isinstance(message, str) and message.strip():
            return message
    return str(error.reason)


def _is_timeout(reason: object) -> bool:
    return isinstance(reason, TimeoutError)
