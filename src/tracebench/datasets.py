"""Workflows for versioned evaluation datasets."""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict

from tracebench.clustering import build_document, canonical_trace_payload
from tracebench.models import (
    EvalCase,
    EvalDataset,
    EvaluationMode,
    Priority,
    ReviewStatus,
    ScorerConfig,
    SliceBuildSource,
    SliceCaseProvenance,
    Trace,
)
from tracebench.storage import (
    connect_database,
    get_dataset_slice_build,
    get_eval_dataset,
    get_trace,
    insert_case_slice_provenance,
    insert_dataset_slice_build,
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


class DatasetBuildValidationError(DatasetError):
    """Raised when a requested exact-size slice build is not possible."""


class DatasetBuildIntegrityError(DatasetError):
    """Raised when an immutable clustering snapshot no longer validates."""


class SliceBuildSummary(BaseModel):
    """Availability and selected quota for one numeric slice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selector: str
    cluster_number: int
    label_snapshot: str | None
    eligible_count: int
    selected_count: int
    allocation_key: str


class DatasetBuildResult(BaseModel):
    """Completed deterministic slice-built dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: EvalDataset
    cases: tuple[EvalCase, ...]
    slice_source: SliceBuildSource
    slices: tuple[SliceBuildSummary, ...]


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


def get_dataset_details(
    database_path: Path, reference: str
) -> tuple[EvalDataset, list[EvalCase], SliceBuildSource | None]:
    """Resolve a dataset with its optional sealed slice-build snapshot."""
    name, version = parse_dataset_reference(reference)
    with closing(connect_database(database_path)) as connection:
        dataset = get_eval_dataset(connection, name, version)
        if dataset is None:
            raise DatasetNotFoundError(f"dataset '{reference}' was not found")
        return (
            dataset,
            list_eval_cases(connection, dataset.dataset_id),
            get_dataset_slice_build(connection, dataset.dataset_id),
        )


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


def build_dataset_from_slices(
    database_path: Path,
    *,
    name: str,
    version: str,
    clustering_run_name: str,
    size: int,
) -> DatasetBuildResult:
    """Atomically build an exact-size balanced reference dataset from B1 slices."""
    canonical_name, canonical_version = _canonical_dataset_identity(name, version)
    run_name = clustering_run_name.strip()
    if not run_name:
        raise DatasetBuildValidationError("clustering run name must not be blank")
    if isinstance(size, bool) or size < 1:
        raise DatasetBuildValidationError("dataset size must be at least 1")

    built_at = datetime.now(UTC)
    dataset = EvalDataset(
        dataset_id=generate_dataset_id(canonical_name, canonical_version),
        name=canonical_name,
        version=canonical_version,
        description="",
        created_at=built_at,
    )
    with closing(connect_database(database_path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                get_eval_dataset(connection, canonical_name, canonical_version)
                is not None
            ):
                raise DatasetAlreadyExistsError(
                    f"dataset '{canonical_name}:{canonical_version}' already exists"
                )
            run = connection.execute(
                "SELECT * FROM trace_clustering_runs WHERE name = ?", (run_name,)
            ).fetchone()
            if run is None:
                raise DatasetBuildIntegrityError(
                    f"clustering run '{run_name}' was not found"
                )
            config = json.loads(str(run["configuration_json"]))
            include_context = bool(config["include_context"])
            label_rows = connection.execute(
                "SELECT cluster_number, label, label_key "
                "FROM trace_cluster_labels WHERE clustering_run_id = ? "
                "ORDER BY cluster_number ASC",
                (run["clustering_run_id"],),
            ).fetchall()
            cluster_count = int(run["cluster_count"])
            if [int(row["cluster_number"]) for row in label_rows] != list(
                range(cluster_count)
            ):
                raise DatasetBuildIntegrityError(
                    "clustering run has incomplete numeric label rows"
                )
            labels = {
                int(row["cluster_number"]): (row["label"], row["label_key"])
                for row in label_rows
            }
            assignment_rows = connection.execute(
                """
                SELECT
                    assignment.*, trace.timestamp AS current_timestamp,
                    trace.task_type, trace.prompt, trace.response,
                    trace.context_json, trace.metadata_json
                FROM trace_cluster_assignments AS assignment
                JOIN traces AS trace ON trace.trace_id = assignment.trace_id
                WHERE assignment.clustering_run_id = ?
                ORDER BY assignment.document_index ASC
                """,
                (run["clustering_run_id"],),
            ).fetchall()
            if len(assignment_rows) != int(run["trace_count"]):
                raise DatasetBuildIntegrityError(
                    "clustering run assignment count does not match its trace count"
                )
            if [int(row["document_index"]) for row in assignment_rows] != list(
                range(len(assignment_rows))
            ):
                raise DatasetBuildIntegrityError(
                    "clustering run document indexes are incomplete"
                )

            eligible: dict[int, list[tuple[sqlite3.Row, Trace, str]]] = {
                cluster_number: [] for cluster_number in range(cluster_count)
            }
            for row in assignment_rows:
                cluster_number = int(row["cluster_number"])
                if cluster_number not in eligible:
                    raise DatasetBuildIntegrityError(
                        "assignment cluster number is outside its run"
                    )
                trace = Trace.model_validate(
                    {
                        "trace_id": row["trace_id"],
                        "timestamp": row["current_timestamp"],
                        "task_type": row["task_type"],
                        "prompt": row["prompt"],
                        "response": row["response"],
                        "context": json.loads(row["context_json"]),
                        "metadata": json.loads(row["metadata_json"]),
                    }
                )
                source_hash = _sha256(canonical_trace_payload(trace))
                document_hash = _sha256(
                    build_document(trace, include_context=include_context)
                )
                if (
                    str(row["source_timestamp"]) != timestamp_to_text(trace.timestamp)
                    or str(row["source_trace_hash"]) != source_hash
                    or str(row["document_hash"]) != document_hash
                ):
                    raise DatasetBuildIntegrityError(
                        f"trace '{trace.trace_id}' no longer matches clustering run "
                        f"'{run_name}'"
                    )
                if trace.response is not None and trace.response.strip():
                    selection_key = _sha256(
                        _canonical_json(
                            {
                                "algorithm": "balanced-hash-v1",
                                "cluster_number": cluster_number,
                                "clustering_run_id": run["clustering_run_id"],
                                "document_hash": row["document_hash"],
                                "source_trace_hash": row["source_trace_hash"],
                                "trace_id": trace.trace_id,
                            }
                        )
                    )
                    eligible[cluster_number].append((row, trace, selection_key))

            eligible_count = sum(len(items) for items in eligible.values())
            if eligible_count == 0:
                raise DatasetBuildValidationError(
                    "no traces in the clustering run have a nonblank source response"
                )
            if size > eligible_count:
                raise DatasetBuildValidationError(
                    f"requested dataset size {size} exceeds eligible trace count "
                    f"{eligible_count}"
                )

            allocation_keys = {
                cluster_number: _sha256(
                    _canonical_json(
                        {
                            "algorithm": "balanced-hash-v1",
                            "cluster_number": cluster_number,
                            "clustering_configuration_hash": run["configuration_hash"],
                            "clustering_run_id": run["clustering_run_id"],
                            "source_manifest_hash": run["source_manifest_hash"],
                        }
                    )
                )
                for cluster_number in range(cluster_count)
            }
            allocation_order = sorted(
                range(cluster_count),
                key=lambda number: (allocation_keys[number], number),
            )
            quotas = {cluster_number: 0 for cluster_number in range(cluster_count)}
            allocated = 0
            while allocated < size:
                progressed = False
                for cluster_number in allocation_order:
                    if quotas[cluster_number] >= len(eligible[cluster_number]):
                        continue
                    quotas[cluster_number] += 1
                    allocated += 1
                    progressed = True
                    if allocated == size:
                        break
                if not progressed:
                    raise DatasetBuildIntegrityError(
                        "sampling allocation exhausted before reaching requested size"
                    )

            manifest = [
                {
                    "allocation_key": allocation_keys[cluster_number],
                    "cluster_number": cluster_number,
                    "eligible_count": len(eligible[cluster_number]),
                    "label_key_snapshot": labels[cluster_number][1],
                    "label_snapshot": labels[cluster_number][0],
                    "quota": quotas[cluster_number],
                    "selector": f"cluster-{cluster_number}",
                }
                for cluster_number in range(cluster_count)
            ]
            manifest_hash = _sha256(_canonical_json(manifest))
            source = SliceBuildSource(
                clustering_run_id=run["clustering_run_id"],
                clustering_run_name=run["name"],
                clustering_schema_version=run["schema_version"],
                clustering_configuration_hash=run["configuration_hash"],
                clustering_source_manifest_hash=run["source_manifest_hash"],
                cluster_count=cluster_count,
                sampling_schema_version=1,
                sampling_algorithm="balanced-hash-v1",
                requested_size=size,
                sampled_size=size,
                eligible_trace_count=eligible_count,
                slice_manifest_hash=manifest_hash,
                slice_manifest=manifest,
                built_at=built_at,
            )
            if not insert_eval_dataset(connection, dataset):
                raise DatasetAlreadyExistsError(
                    f"dataset '{canonical_name}:{canonical_version}' already exists"
                )

            built_cases: list[EvalCase] = []
            for cluster_number in range(cluster_count):
                ranked = sorted(
                    eligible[cluster_number],
                    key=lambda item: (item[2], item[1].trace_id),
                )
                for rank, (assignment, trace, selection_key) in enumerate(
                    ranked[: quotas[cluster_number]]
                ):
                    provenance = SliceCaseProvenance(
                        selector=f"cluster-{cluster_number}",
                        cluster_number=cluster_number,
                        label_snapshot=labels[cluster_number][0],
                        label_key_snapshot=labels[cluster_number][1],
                        clustering_run_id=run["clustering_run_id"],
                        clustering_run_name=run["name"],
                        clustering_schema_version=run["schema_version"],
                        clustering_configuration_hash=run["configuration_hash"],
                        clustering_source_manifest_hash=run["source_manifest_hash"],
                        cluster_count=cluster_count,
                        source_trace_id=trace.trace_id,
                        source_timestamp=trace.timestamp,
                        source_trace_hash=assignment["source_trace_hash"],
                        document_index=assignment["document_index"],
                        document_hash=assignment["document_hash"],
                        sampling_schema_version=1,
                        sampling_algorithm="balanced-hash-v1",
                        requested_size=size,
                        sampled_size=size,
                        eligible_trace_count=eligible_count,
                        slice_availability=len(eligible[cluster_number]),
                        slice_quota=quotas[cluster_number],
                        rank_within_slice=rank,
                        selection_key=selection_key,
                        allocation_key=allocation_keys[cluster_number],
                        slice_manifest_hash=manifest_hash,
                    )
                    case = EvalCase(
                        eval_id=generate_eval_id(dataset.dataset_id, trace.trace_id),
                        dataset_id=dataset.dataset_id,
                        source_trace_id=trace.trace_id,
                        source_timestamp=trace.timestamp,
                        source_task_type=trace.task_type,
                        source_response=trace.response,
                        source_metadata=trace.metadata,
                        input=trace.prompt,
                        context=trace.context,
                        evaluation_mode=EvaluationMode.REFERENCE,
                        reference_answer=trace.response,
                        rubric=[],
                        scorers=[],
                        priority=Priority.MEDIUM,
                        review_status=ReviewStatus.DRAFT,
                        created_at=built_at,
                        slice_provenance=provenance,
                    )
                    if not insert_eval_case(connection, case):
                        raise DatasetBuildIntegrityError(
                            f"duplicate source trace '{trace.trace_id}' in build"
                        )
                    insert_case_slice_provenance(
                        connection, case.eval_id, dataset.dataset_id, provenance
                    )
                    built_cases.append(case)
            insert_dataset_slice_build(connection, dataset.dataset_id, source)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    summaries = tuple(
        SliceBuildSummary(
            selector=f"cluster-{cluster_number}",
            cluster_number=cluster_number,
            label_snapshot=labels[cluster_number][0],
            eligible_count=len(eligible[cluster_number]),
            selected_count=quotas[cluster_number],
            allocation_key=allocation_keys[cluster_number],
        )
        for cluster_number in range(cluster_count)
    )
    return DatasetBuildResult(
        dataset=dataset,
        cases=tuple(sorted(built_cases, key=lambda case: case.eval_id)),
        slice_source=source,
        slices=summaries,
    )


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
    dataset, cases, slice_source = get_dataset_details(database_path, reference)
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
                _write_export(output, dataset, cases, slice_source)
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
                _write_export(output, dataset, cases, slice_source)
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
    slice_source: SliceBuildSource | None,
) -> None:
    output.write(_encode_dataset_export_record(dataset, slice_source))
    output.write("\n")
    for case in cases:
        output.write(_encode_case_export_record(dataset, case))
        output.write("\n")
    output.flush()
    os.fsync(output.fileno())


def _encode_dataset_export_record(
    dataset: EvalDataset, slice_source: SliceBuildSource | None
) -> str:
    payload = {
        **dataset.model_dump(mode="json"),
        "dataset_ref": f"{dataset.name}:{dataset.version}",
        "created_at": timestamp_to_text(dataset.created_at),
    }
    if slice_source is not None:
        payload["slice_source"] = slice_source.model_dump(mode="json")
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


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
