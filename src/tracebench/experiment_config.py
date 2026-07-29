"""Strict YAML loading and complete experiment preflight."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from tracebench.datasets import DatasetError, get_dataset_and_cases
from tracebench.experiment_models import (
    EffectiveThresholds,
    ExperimentConfig,
    FixtureProviderConfig,
    OllamaProviderConfig,
    RunRole,
)
from tracebench.models import EvalCase, EvalDataset, EvaluationMode
from tracebench.providers import (
    FixtureProvider,
    JsonValue,
    OllamaProvider,
    Provider,
    ProviderError,
    load_fixture,
)
from tracebench.scorers import ScorerConfigurationError, validate_case_scorers


class ExperimentPreflightError(Exception):
    """Raised when an experiment cannot safely create an attempt."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    """Fully validated, in-memory inputs ready to persist as an attempt."""

    config: ExperimentConfig
    dataset: EvalDataset
    cases: tuple[EvalCase, ...]
    providers: dict[RunRole, Provider]
    effective_thresholds: dict[str, EffectiveThresholds]
    configuration_hash: str
    configuration_json: str
    provider_snapshots: dict[RunRole, dict[str, JsonValue]]


def prepare_experiment(
    config_path: Path,
    database_path: Path,
) -> PreparedExperiment:
    """Validate every input without persisting an experiment attempt."""
    config = load_experiment_config(config_path)
    try:
        dataset, loaded_cases = get_dataset_and_cases(database_path, config.dataset)
    except (DatasetError, OSError, ValueError) as error:
        raise ExperimentPreflightError(str(error)) from error

    if not loaded_cases:
        raise ExperimentPreflightError(
            f"dataset '{config.dataset}' contains no evaluation cases"
        )
    cases = tuple(sorted(loaded_cases, key=lambda case: case.eval_id))
    rubric_cases = [
        case.eval_id for case in cases if case.evaluation_mode is EvaluationMode.RUBRIC
    ]
    if rubric_cases:
        raise ExperimentPreflightError(
            "rubric evaluation is not supported by the deterministic core loop: "
            + ", ".join(rubric_cases)
        )

    present_modes = {case.evaluation_mode for case in cases}
    absent_overrides = sorted(
        mode.value for mode in set(config.gate.by_mode) - present_modes
    )
    if absent_overrides:
        raise ExperimentPreflightError(
            "per-mode gate overrides require modes present in the dataset: "
            + ", ".join(absent_overrides)
        )

    for case in cases:
        try:
            validate_case_scorers(case)
        except ScorerConfigurationError as error:
            raise ExperimentPreflightError(
                f"evaluation case '{case.eval_id}': {error}"
            ) from error

    expected_ids = {case.eval_id for case in cases}
    config_parent = config_path.resolve().parent
    providers: dict[RunRole, Provider] = {}
    provider_snapshots: dict[RunRole, dict[str, JsonValue]] = {}
    canonical_providers: dict[RunRole, dict[str, JsonValue]] = {}
    for role, provider_config in (
        (RunRole.BASELINE, config.baseline),
        (RunRole.CANDIDATE, config.candidate),
    ):
        if isinstance(provider_config, FixtureProviderConfig):
            path = provider_config.path
            if not path.is_absolute():
                path = config_parent / path
            try:
                outputs = load_fixture(path.resolve(), expected_ids)
            except ProviderError as error:
                raise ExperimentPreflightError(str(error)) from error
            providers[role] = FixtureProvider(outputs)
            provider_snapshots[role] = {
                "provider": "fixture",
                "fixture_hash": hashlib.sha256(
                    _encode_canonical(outputs).encode("utf-8")
                ).hexdigest(),
            }
            canonical_outputs: dict[str, JsonValue] = {
                eval_id: output for eval_id, output in outputs.items()
            }
            canonical_providers[role] = {
                "provider": "fixture",
                "outputs": canonical_outputs,
            }
        elif isinstance(provider_config, OllamaProviderConfig):
            prompt_path = provider_config.system_prompt_file
            if not prompt_path.is_absolute():
                prompt_path = config_parent / prompt_path
            system_prompt, system_prompt_hash = _load_system_prompt(
                prompt_path.resolve()
            )
            providers[role] = OllamaProvider(
                base_url=provider_config.base_url,
                model=provider_config.model,
                system_prompt=system_prompt,
                temperature=provider_config.temperature,
                timeout_seconds=provider_config.timeout_seconds,
                seed=provider_config.seed,
            )
            snapshot: dict[str, JsonValue] = {
                "provider": "ollama",
                "base_url": provider_config.base_url,
                "model": provider_config.model,
                "prompt_version": provider_config.prompt_version,
                "system_prompt_hash": system_prompt_hash,
                "temperature": provider_config.temperature,
                "timeout_seconds": provider_config.timeout_seconds,
                "seed": provider_config.seed,
            }
            provider_snapshots[role] = snapshot
            canonical_providers[role] = dict(snapshot)
        else:
            raise AssertionError("unreachable provider configuration")

    effective_thresholds = _effective_thresholds(config)
    canonical = _canonical_configuration(
        config=config,
        dataset=dataset,
        cases=cases,
        canonical_providers=canonical_providers,
        effective_thresholds=effective_thresholds,
    )
    configuration_json = _encode_canonical(canonical)
    return PreparedExperiment(
        config=config,
        dataset=dataset,
        cases=cases,
        providers=providers,
        effective_thresholds=effective_thresholds,
        configuration_hash=hashlib.sha256(
            configuration_json.encode("utf-8")
        ).hexdigest(),
        configuration_json=configuration_json,
        provider_snapshots=provider_snapshots,
    )


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Parse one strict, versioned YAML experiment configuration."""
    try:
        text = config_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ExperimentPreflightError(
            f"could not read experiment configuration {config_path}: {error}"
        ) from error
    try:
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ExperimentPreflightError(
            f"invalid YAML in {config_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ExperimentPreflightError(
            f"invalid experiment configuration {config_path}: root must be a mapping"
        )
    try:
        return ExperimentConfig.model_validate(payload)
    except ValidationError as error:
        messages = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])}: {detail['msg']}"
            for detail in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )
        )
        raise ExperimentPreflightError(
            f"invalid experiment configuration {config_path}: {messages}"
        ) from error


def _effective_thresholds(
    config: ExperimentConfig,
) -> dict[str, EffectiveThresholds]:
    global_thresholds = EffectiveThresholds(
        max_score_drop=config.gate.max_score_drop,
        max_new_failures=config.gate.max_new_failures,
    )
    effective = {"global": global_thresholds}
    for mode, override in sorted(
        config.gate.by_mode.items(), key=lambda item: item[0].value
    ):
        effective[mode.value] = EffectiveThresholds(
            max_score_drop=(
                global_thresholds.max_score_drop
                if override.max_score_drop is None
                else override.max_score_drop
            ),
            max_new_failures=(
                global_thresholds.max_new_failures
                if override.max_new_failures is None
                else override.max_new_failures
            ),
        )
    return effective


def _canonical_configuration(
    *,
    config: ExperimentConfig,
    dataset: EvalDataset,
    cases: tuple[EvalCase, ...],
    canonical_providers: dict[RunRole, dict[str, JsonValue]],
    effective_thresholds: dict[str, EffectiveThresholds],
) -> dict[str, object]:
    case_definitions = [
        {
            "eval_id": case.eval_id,
            "input": case.input,
            "context": case.context,
            "evaluation_mode": case.evaluation_mode.value,
            "reference_answer": case.reference_answer,
            "scorers": [scorer.model_dump(mode="json") for scorer in case.scorers],
        }
        for case in cases
    ]
    return {
        "schema_version": config.schema_version,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "version": dataset.version,
            "cases": case_definitions,
        },
        "baseline": canonical_providers[RunRole.BASELINE],
        "candidate": canonical_providers[RunRole.CANDIDATE],
        "gate": {
            scope: thresholds.model_dump(mode="json")
            for scope, thresholds in sorted(effective_thresholds.items())
        },
    }


def _encode_canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_system_prompt(path: Path) -> tuple[str, str]:
    try:
        contents = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ExperimentPreflightError(
            f"could not read system prompt {path}: {error}"
        ) from error
    if not contents.strip():
        raise ExperimentPreflightError(f"system prompt {path} must not be blank")
    content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return contents, content_hash
