"""Strict YAML loading and complete experiment preflight."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from tracebench.clustering import canonical_label
from tracebench.datasets import DatasetError, get_dataset_details
from tracebench.experiment_models import (
    EffectiveThresholds,
    ExperimentConfig,
    FixtureJudgeConfig,
    FixtureProviderConfig,
    OllamaJudgeConfig,
    OllamaProviderConfig,
    RunRole,
)
from tracebench.judges import (
    JUDGE_RESPONSE_SCHEMA_VERSION,
    MAX_MALFORMED_RETRIES,
    load_judge_fixture,
)
from tracebench.models import (
    EvalCase,
    EvalDataset,
    EvaluationMode,
    SliceBuildSource,
    SliceCaseProvenance,
)
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
    provider_prompts: dict[RunRole, str | None]
    judge: PreparedJudge | None
    effective_thresholds: dict[str, EffectiveThresholds]
    configuration_hash: str
    configuration_json: str
    provider_snapshots: dict[RunRole, dict[str, JsonValue]]
    slice_source: SliceBuildSource | None
    slice_membership: dict[str, SliceCaseProvenance]
    effective_slice_thresholds: dict[int, EffectiveThresholds]


@dataclass(frozen=True, slots=True)
class PreparedJudge:
    """Fully materialized judge configuration and provider."""

    provider: Provider
    prompt: str
    retry_prompt: str
    prompt_version: str
    prompt_hash: str
    retry_prompt_hash: str
    confidence_threshold: float
    cache_provider_identity: dict[str, JsonValue]
    fixture_request_hashes: dict[str, str] | None
    snapshot: dict[str, JsonValue]

    def provider_identity_for(self, request_id: str) -> dict[str, object]:
        """Return only provider inputs that affect this judge generation."""
        identity: dict[str, object] = dict(self.cache_provider_identity)
        if self.fixture_request_hashes is not None:
            identity["request_id"] = request_id
            identity["response_sequence_hash"] = self.fixture_request_hashes[request_id]
        return identity


def prepare_experiment(
    config_path: Path,
    database_path: Path,
) -> PreparedExperiment:
    """Validate every input without persisting an experiment attempt."""
    config = load_experiment_config(config_path)
    try:
        dataset, loaded_cases, slice_source = get_dataset_details(
            database_path, config.dataset
        )
    except (DatasetError, OSError, ValueError) as error:
        raise ExperimentPreflightError(str(error)) from error

    if not loaded_cases:
        raise ExperimentPreflightError(
            f"dataset '{config.dataset}' contains no evaluation cases"
        )
    cases = tuple(sorted(loaded_cases, key=lambda case: case.eval_id))
    slice_membership = _validate_slice_provenance(cases, slice_source)
    effective_slice_thresholds = _effective_slice_thresholds(
        config, slice_membership, slice_source
    )
    rubric_cases = [
        case.eval_id for case in cases if case.evaluation_mode is EvaluationMode.RUBRIC
    ]
    if rubric_cases and config.judge is None:
        raise ExperimentPreflightError(
            "rubric evaluation requires a configured judge: " + ", ".join(rubric_cases)
        )
    if not rubric_cases and config.judge is not None:
        raise ExperimentPreflightError(
            "judge configuration requires rubric cases in the dataset"
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
    provider_prompts: dict[RunRole, str | None] = {}
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
            provider_prompts[role] = None
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
                temperature=provider_config.temperature,
                timeout_seconds=provider_config.timeout_seconds,
                seed=provider_config.seed,
            )
            provider_prompts[role] = system_prompt
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

    prepared_judge: PreparedJudge | None = None
    canonical_judge: dict[str, JsonValue] | None = None
    if config.judge is not None:
        judge_config = config.judge
        prompt_path = judge_config.prompt_file
        if not prompt_path.is_absolute():
            prompt_path = config_parent / prompt_path
        retry_prompt_path = judge_config.retry_prompt_file
        if not retry_prompt_path.is_absolute():
            retry_prompt_path = config_parent / retry_prompt_path
        prompt, prompt_hash = _load_prompt(prompt_path.resolve(), "judge prompt")
        retry_prompt, retry_prompt_hash = _load_prompt(
            retry_prompt_path.resolve(), "judge retry prompt"
        )
        common_snapshot: dict[str, JsonValue] = {
            "prompt_version": judge_config.prompt_version,
            "prompt_hash": prompt_hash,
            "retry_prompt_hash": retry_prompt_hash,
            "response_schema_version": JUDGE_RESPONSE_SCHEMA_VERSION,
            "max_malformed_retries": MAX_MALFORMED_RETRIES,
            "confidence_threshold": judge_config.confidence_threshold,
        }
        rubric_ids = set(rubric_cases)
        if isinstance(judge_config, FixtureJudgeConfig):
            fixture_path = judge_config.path
            if not fixture_path.is_absolute():
                fixture_path = config_parent / fixture_path
            try:
                judge_outputs = load_judge_fixture(fixture_path.resolve(), rubric_ids)
            except ProviderError as error:
                raise ExperimentPreflightError(str(error)) from error
            judge_provider: Provider = FixtureProvider(judge_outputs)
            fixture_request_hashes = {
                request_id: hashlib.sha256(
                    _encode_canonical(list(outputs)).encode("utf-8")
                ).hexdigest()
                for request_id, outputs in judge_outputs.items()
            }
            cache_provider_identity: dict[str, JsonValue] = {
                "provider": "fixture",
            }
            snapshot = {
                "provider": "fixture",
                "fixture_hash": hashlib.sha256(
                    _encode_canonical(judge_outputs).encode("utf-8")
                ).hexdigest(),
                **common_snapshot,
            }
            canonical_judge = {
                "provider": "fixture",
                "outputs": {key: list(value) for key, value in judge_outputs.items()},
                **common_snapshot,
            }
        elif isinstance(judge_config, OllamaJudgeConfig):
            judge_provider = OllamaProvider(
                base_url=judge_config.base_url,
                model=judge_config.model,
                temperature=judge_config.temperature,
                timeout_seconds=judge_config.timeout_seconds,
                seed=judge_config.seed,
            )
            snapshot = {
                "provider": "ollama",
                "base_url": judge_config.base_url,
                "model": judge_config.model,
                "temperature": judge_config.temperature,
                "timeout_seconds": judge_config.timeout_seconds,
                "seed": judge_config.seed,
                **common_snapshot,
            }
            canonical_judge = dict(snapshot)
            fixture_request_hashes = None
            cache_provider_identity = {
                "provider": "ollama",
                "base_url": judge_config.base_url,
                "model": judge_config.model,
                "temperature": judge_config.temperature,
                "seed": judge_config.seed,
            }
        else:
            raise AssertionError("unreachable judge configuration")
        prepared_judge = PreparedJudge(
            provider=judge_provider,
            prompt=prompt,
            retry_prompt=retry_prompt,
            prompt_version=judge_config.prompt_version,
            prompt_hash=prompt_hash,
            retry_prompt_hash=retry_prompt_hash,
            confidence_threshold=judge_config.confidence_threshold,
            cache_provider_identity=cache_provider_identity,
            fixture_request_hashes=fixture_request_hashes,
            snapshot=snapshot,
        )

    effective_thresholds = _effective_thresholds(config)
    canonical = _canonical_configuration(
        config=config,
        dataset=dataset,
        cases=cases,
        canonical_providers=canonical_providers,
        canonical_judge=canonical_judge,
        effective_thresholds=effective_thresholds,
        slice_source=slice_source,
        effective_slice_thresholds=effective_slice_thresholds,
    )
    configuration_json = _encode_canonical(canonical)
    return PreparedExperiment(
        config=config,
        dataset=dataset,
        cases=cases,
        providers=providers,
        provider_prompts=provider_prompts,
        judge=prepared_judge,
        effective_thresholds=effective_thresholds,
        configuration_hash=hashlib.sha256(
            configuration_json.encode("utf-8")
        ).hexdigest(),
        configuration_json=configuration_json,
        provider_snapshots=provider_snapshots,
        slice_source=slice_source,
        slice_membership=slice_membership,
        effective_slice_thresholds=effective_slice_thresholds,
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


def _validate_slice_provenance(
    cases: tuple[EvalCase, ...],
    source: SliceBuildSource | None,
) -> dict[str, SliceCaseProvenance]:
    provenance = {
        case.eval_id: case.slice_provenance
        for case in cases
        if case.slice_provenance is not None
    }
    if source is None:
        if provenance:
            raise ExperimentPreflightError(
                "dataset has case slice provenance without a sealed slice build"
            )
        return {}
    if len(provenance) != len(cases):
        raise ExperimentPreflightError(
            "slice-built dataset has incomplete case provenance"
        )
    for eval_id, item in provenance.items():
        if (
            item.clustering_run_id != source.clustering_run_id
            or item.clustering_configuration_hash
            != source.clustering_configuration_hash
            or item.clustering_source_manifest_hash
            != source.clustering_source_manifest_hash
            or item.slice_manifest_hash != source.slice_manifest_hash
            or item.sampled_size != source.sampled_size
        ):
            raise ExperimentPreflightError(
                f"evaluation case '{eval_id}' has inconsistent slice provenance"
            )
    return {eval_id: item for eval_id, item in provenance.items() if item is not None}


def _effective_slice_thresholds(
    config: ExperimentConfig,
    membership: dict[str, SliceCaseProvenance],
    source: SliceBuildSource | None,
) -> dict[int, EffectiveThresholds]:
    if not config.gate.by_slice:
        return {}
    if source is None or not membership:
        raise ExperimentPreflightError(
            "per-slice gate overrides require a slice-built dataset"
        )
    represented: dict[int, SliceCaseProvenance] = {}
    for item in membership.values():
        represented.setdefault(item.cluster_number, item)
    label_candidates: dict[str, set[int]] = {}
    for cluster_number, item in represented.items():
        if item.label_key_snapshot is not None:
            label_candidates.setdefault(item.label_key_snapshot, set()).add(
                cluster_number
            )
    resolved: dict[int, EffectiveThresholds] = {}
    global_thresholds = EffectiveThresholds(
        max_score_drop=config.gate.max_score_drop,
        max_new_failures=config.gate.max_new_failures,
    )
    for raw_selector, override in config.gate.by_slice.items():
        selector = raw_selector.strip()
        candidates: set[int] = set()
        numeric = re.fullmatch(r"cluster-(0|[1-9][0-9]*)", selector)
        if numeric is not None:
            cluster_number = int(numeric.group(1))
            if cluster_number in represented:
                candidates.add(cluster_number)
        _, key = canonical_label(selector)
        candidates.update(label_candidates.get(key, set()))
        if not candidates:
            raise ExperimentPreflightError(
                f"configured slice '{raw_selector}' is not represented in the dataset"
            )
        if len(candidates) > 1:
            raise ExperimentPreflightError(
                f"configured slice '{raw_selector}' is ambiguous"
            )
        cluster_number = next(iter(candidates))
        if cluster_number in resolved:
            raise ExperimentPreflightError(
                f"multiple slice overrides resolve to cluster-{cluster_number}"
            )
        resolved[cluster_number] = EffectiveThresholds(
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
    return dict(sorted(resolved.items()))


def _canonical_configuration(
    *,
    config: ExperimentConfig,
    dataset: EvalDataset,
    cases: tuple[EvalCase, ...],
    canonical_providers: dict[RunRole, dict[str, JsonValue]],
    canonical_judge: dict[str, JsonValue] | None,
    effective_thresholds: dict[str, EffectiveThresholds],
    slice_source: SliceBuildSource | None = None,
    effective_slice_thresholds: dict[int, EffectiveThresholds] | None = None,
) -> dict[str, object]:
    case_definitions: list[dict[str, object]] = []
    for case in cases:
        definition: dict[str, object] = {
            "eval_id": case.eval_id,
            "input": case.input,
            "context": case.context,
            "evaluation_mode": case.evaluation_mode.value,
            "reference_answer": case.reference_answer,
            "scorers": [scorer.model_dump(mode="json") for scorer in case.scorers],
        }
        if case.evaluation_mode is EvaluationMode.RUBRIC:
            definition["rubric"] = list(case.rubric)
            definition["priority"] = case.priority.value
        if case.slice_provenance is not None:
            definition["slice"] = {
                "selector": case.slice_provenance.selector,
                "cluster_number": case.slice_provenance.cluster_number,
                "slice_manifest_hash": case.slice_provenance.slice_manifest_hash,
                "selection_key": case.slice_provenance.selection_key,
            }
        case_definitions.append(definition)
    dataset_payload: dict[str, object] = {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "version": dataset.version,
        "cases": case_definitions,
    }
    if slice_source is not None:
        dataset_payload["slice_source"] = slice_source.model_dump(mode="json")
    gate_payload: dict[str, object] = {
        scope: thresholds.model_dump(mode="json")
        for scope, thresholds in sorted(effective_thresholds.items())
    }
    if effective_slice_thresholds:
        gate_payload["by_slice"] = {
            f"cluster-{cluster_number}": thresholds.model_dump(mode="json")
            for cluster_number, thresholds in sorted(effective_slice_thresholds.items())
        }
    canonical: dict[str, object] = {
        "schema_version": config.schema_version,
        "dataset": dataset_payload,
        "baseline": canonical_providers[RunRole.BASELINE],
        "candidate": canonical_providers[RunRole.CANDIDATE],
        "gate": gate_payload,
    }
    if canonical_judge is not None:
        canonical["judge"] = canonical_judge
    return canonical


def _encode_canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _load_system_prompt(path: Path) -> tuple[str, str]:
    return _load_prompt(path, "system prompt")


def _load_prompt(path: Path, description: str) -> tuple[str, str]:
    try:
        contents = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ExperimentPreflightError(
            f"could not read {description} {path}: {error}"
        ) from error
    if not contents.strip():
        raise ExperimentPreflightError(f"{description} {path} must not be blank")
    content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return contents, content_hash
