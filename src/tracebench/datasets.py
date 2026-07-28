"""Workflows for versioned evaluation datasets."""

import json
import os
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
from uuid import UUID, uuid5

from tracebench.models import (
    EvalCase,
    EvalDataset,
    EvaluationMode,
    Priority,
    ReviewStatus,
    ScorerConfig,
)
from tracebench.storage import (
    connect_database,
    get_eval_dataset,
    get_trace,
    insert_eval_case,
    insert_eval_dataset,
    list_eval_cases,
    list_eval_datasets,
    timestamp_to_text,
)

# Fixed UUID5 namespaces make logical dataset and case identities reproducible
# across processes, machines, and clean database rebuilds.
DATASET_ID_NAMESPACE = UUID("4dc9af25-7cdb-5d6b-a985-318ecdea4f55")
EVAL_ID_NAMESPACE = UUID("b272f60f-9f27-5152-a2d5-80e6e4e77341")


class DatasetError(Exception):
    """Base class for expected dataset workflow failures."""


class InvalidDatasetReferenceError(DatasetError):
    """Raised when a name:version reference is malformed."""


class DatasetAlreadyExistsError(DatasetError):
    """Raised when a dataset name/version is already stored."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset reference cannot be resolved."""


class TraceNotFoundError(DatasetError):
    """Raised when a source trace cannot be found."""


class DuplicateTraceError(DatasetError):
    """Raised when a trace is already a member of a dataset."""


class ExportFileExistsError(DatasetError):
    """Raised when export would overwrite a file without permission."""


def parse_dataset_reference(reference: str) -> tuple[str, str]:
    """Parse an exact nonblank ``name:version`` dataset reference."""
    if reference.count(":") != 1:
        raise InvalidDatasetReferenceError(
            f"invalid dataset reference {reference!r}; expected name:version"
        )
    name, version = reference.split(":", maxsplit=1)
    try:
        return _canonical_dataset_identity(name, version)
    except ValueError as error:
        raise InvalidDatasetReferenceError(
            f"invalid dataset reference {reference!r}; expected name:version"
        ) from error


def create_dataset(
    database_path: Path,
    *,
    name: str,
    version: str,
    description: str = "",
) -> EvalDataset:
    """Create and persist one explicitly versioned dataset."""
    canonical_name, canonical_version = _canonical_dataset_identity(name, version)
    dataset = EvalDataset.model_validate(
        {
            "dataset_id": generate_dataset_id(canonical_name, canonical_version),
            "name": canonical_name,
            "version": canonical_version,
            "description": description,
            "created_at": datetime.now(UTC),
        }
    )
    with closing(connect_database(database_path)) as connection:
        with connection:
            if not insert_eval_dataset(connection, dataset):
                raise DatasetAlreadyExistsError(
                    f"dataset '{dataset.name}:{dataset.version}' already exists"
                )
    return dataset


def get_dataset_and_cases(
    database_path: Path, reference: str
) -> tuple[EvalDataset, list[EvalCase]]:
    """Resolve a dataset reference and return its ordered cases."""
    name, version = parse_dataset_reference(reference)
    with closing(connect_database(database_path)) as connection:
        dataset = get_eval_dataset(connection, name, version)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset '{reference}' was not found")
        cases = list_eval_cases(connection, dataset.dataset_id)
    return dataset, cases


def get_datasets(database_path: Path) -> list[tuple[EvalDataset, int]]:
    """Return all datasets and their case counts."""
    with closing(connect_database(database_path)) as connection:
        return list_eval_datasets(connection)


def promote_trace(
    database_path: Path,
    *,
    dataset_reference: str,
    trace_id: str,
    mode: EvaluationMode,
    reference_answer: str | None = None,
    rubric: list[str] | None = None,
    scorers: list[ScorerConfig] | None = None,
    use_source_response: bool = False,
    priority: Priority = Priority.MEDIUM,
    review_status: ReviewStatus = ReviewStatus.DRAFT,
) -> EvalCase:
    """Snapshot a stored trace into an evaluation dataset."""
    if use_source_response and mode is not EvaluationMode.REFERENCE:
        raise ValueError("--use-source-response is only valid in reference mode")
    if use_source_response and reference_answer is not None:
        raise ValueError(
            "reference mode accepts either a reference answer or the source response"
        )

    name, version = parse_dataset_reference(dataset_reference)
    with closing(connect_database(database_path)) as connection:
        with connection:
            dataset = get_eval_dataset(connection, name, version)
            if dataset is None:
                raise DatasetNotFoundError(
                    f"dataset '{dataset_reference}' was not found"
                )
            trace = get_trace(connection, trace_id)
            if trace is None:
                raise TraceNotFoundError(f"trace '{trace_id}' was not found")
            if use_source_response and trace.response is None:
                raise ValueError(f"trace '{trace_id}' has no source response")

            case = EvalCase.model_validate(
                {
                    "eval_id": generate_eval_id(dataset.dataset_id, trace.trace_id),
                    "dataset_id": dataset.dataset_id,
                    "source_trace_id": trace.trace_id,
                    "source_timestamp": trace.timestamp,
                    "source_task_type": trace.task_type,
                    "source_response": (
                        trace.response if mode is EvaluationMode.REFERENCE else None
                    ),
                    "source_metadata": trace.metadata,
                    "input": trace.prompt,
                    "context": trace.context,
                    "evaluation_mode": mode,
                    "reference_answer": (
                        trace.response if use_source_response else reference_answer
                    ),
                    "rubric": rubric or [],
                    "scorers": scorers or [],
                    "priority": priority,
                    "review_status": review_status,
                    "created_at": datetime.now(UTC),
                }
            )
            if not insert_eval_case(connection, case):
                raise DuplicateTraceError(
                    f"trace '{trace_id}' is already in dataset '{dataset_reference}'"
                )
    return case


def generate_dataset_id(name: str, version: str) -> str:
    """Generate a stable UUID5 ID from a canonical dataset name and version."""
    canonical_name, canonical_version = _canonical_dataset_identity(name, version)
    identity = _encode_identity([canonical_name, canonical_version])
    return f"dataset_{uuid5(DATASET_ID_NAMESPACE, identity).hex}"


def generate_eval_id(dataset_id: str, source_trace_id: str) -> str:
    """Generate a stable UUID5 ID from a dataset identity and source trace ID."""
    identity = _encode_identity([dataset_id.strip(), source_trace_id])
    return f"eval_{uuid5(EVAL_ID_NAMESPACE, identity).hex}"


def export_dataset(
    database_path: Path,
    reference: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> int:
    """Export a dataset metadata envelope followed by ordered case envelopes."""
    dataset, cases = get_dataset_and_cases(database_path, reference)
    parent = output_path.parent
    if not parent.exists():
        raise OSError(f"output directory does not exist: {parent}")
    if not parent.is_dir():
        raise OSError(f"output parent is not a directory: {parent}")
    if output_path.is_dir():
        raise OSError(f"output path is a directory: {output_path}")
    if overwrite:
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                text=True,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                _write_export(output, dataset, cases)
            os.replace(temporary_path, output_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    else:
        try:
            output = output_path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as error:
            raise ExportFileExistsError(
                f"output file already exists: {output_path}"
            ) from error
        try:
            with output:
                _write_export(output, dataset, cases)
        except BaseException:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    return len(cases)


def _write_export(
    output: TextIO,
    dataset: EvalDataset,
    cases: list[EvalCase],
) -> None:
    output.write(_encode_dataset_export_record(dataset))
    output.write("\n")
    for case in cases:
        output.write(_encode_case_export_record(dataset, case))
        output.write("\n")
    output.flush()
    os.fsync(output.fileno())


def _encode_dataset_export_record(dataset: EvalDataset) -> str:
    payload = {
        **dataset.model_dump(mode="json"),
        "dataset_ref": f"{dataset.name}:{dataset.version}",
        "created_at": timestamp_to_text(dataset.created_at),
    }
    return _encode_export_record({"record_type": "dataset", "dataset": payload})


def _encode_case_export_record(dataset: EvalDataset, case: EvalCase) -> str:
    payload = {
        **case.model_dump(mode="json"),
        "dataset_ref": f"{dataset.name}:{dataset.version}",
        "source_timestamp": timestamp_to_text(case.source_timestamp),
        "created_at": timestamp_to_text(case.created_at),
    }
    return _encode_export_record({"record_type": "eval_case", "case": payload})


def _encode_export_record(record: dict[str, object]) -> str:
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_dataset_identity(name: str, version: str) -> tuple[str, str]:
    canonical_name = name.strip()
    canonical_version = version.strip()
    if not canonical_name or not canonical_version:
        raise ValueError("dataset name and version must not be blank")
    if ":" in canonical_name or ":" in canonical_version:
        raise ValueError("dataset name and version must not contain ':'")
    return canonical_name, canonical_version


def _encode_identity(parts: list[str]) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
