"""Deterministic local trace clustering and persistent named slices."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
import warnings
from contextlib import closing
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

import numpy as np
import sklearn  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
)
from scipy.sparse import csr_matrix, issparse  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.decomposition import TruncatedSVD  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)
from sklearn.preprocessing import normalize  # type: ignore[import-untyped]

from tracebench.models import Trace
from tracebench.storage import connect_database, timestamp_to_text

HASH_PATTERN = r"^[0-9a-f]{64}$"


class ClusteringError(Exception):
    """Base class for expected clustering workflow failures."""


class ClusteringValidationError(ClusteringError):
    """Raised before a clustering run can be persisted."""


class ClusteringRunExistsError(ClusteringError):
    """Raised when an immutable clustering run name is reused."""


class ClusteringRunNotFoundError(ClusteringError):
    """Raised when a named clustering run does not exist."""


class SliceNotFoundError(ClusteringError):
    """Raised when a numeric cluster does not exist in a run."""


class DuplicateSliceLabelError(ClusteringError):
    """Raised when a canonical label key is already used in a run."""


class SourceManifestChangedError(ClusteringError):
    """Raised when traces change between computation and persistence."""


class SVDConfig(BaseModel):
    """Fixed Truncated SVD configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    n_components: Annotated[StrictInt, Field(ge=1)]
    algorithm: Literal["randomized"] = "randomized"
    n_iter: Literal[7] = 7
    n_oversamples: Literal[10] = 10
    power_iteration_normalizer: Literal["QR"] = "QR"
    tol: Annotated[float, Field(ge=0.0, le=0.0)] = 0.0
    random_state: Literal[42] = 42


class TfidfConfig(BaseModel):
    """Fixed TF-IDF configuration recorded in the pipeline identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analyzer: Literal["word"] = "word"
    binary: Literal[False] = False
    decode_error: Literal["strict"] = "strict"
    dtype: Literal["float64"] = "float64"
    encoding: Literal["utf-8"] = "utf-8"
    input: Literal["content"] = "content"
    lowercase: Literal[True] = True
    max_df: Annotated[float, Field(ge=1.0, le=1.0)] = 1.0
    max_features: None = None
    min_df: Literal[1] = 1
    ngram_range: tuple[Literal[1], Literal[2]] = (1, 2)
    norm: Literal["l2"] = "l2"
    preprocessor: None = None
    smooth_idf: Literal[True] = True
    stop_words: None = None
    strip_accents: Literal["unicode"] = "unicode"
    sublinear_tf: Literal[True] = True
    token_pattern: Literal[r"(?u)\b\w\w+\b"] = r"(?u)\b\w\w+\b"
    tokenizer: None = None
    use_idf: Literal[True] = True
    vocabulary: None = None


class NormalizationConfig(BaseModel):
    """Fixed normalization applied to Truncated SVD output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    norm: Literal["l2"] = "l2"
    axis: Literal[1] = 1
    copy_output: Literal[False] = Field(False, serialization_alias="copy")
    return_norm: Literal[False] = False


class KMeansConfig(BaseModel):
    """Fixed KMeans settings plus the requested cluster count."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    algorithm: Literal["lloyd"] = "lloyd"
    copy_x: Literal[True] = True
    init: Literal["k-means++"] = "k-means++"
    max_iter: Literal[300] = 300
    n_clusters: Annotated[StrictInt, Field(ge=1)]
    n_init: Literal[20] = 20
    random_state: Literal[42] = 42
    tol: Annotated[float, Field(ge=0.0001, le=0.0001)] = 0.0001
    verbose: Literal[0] = 0


class TraceClusteringConfig(BaseModel):
    """Complete versioned identity of the clustering pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    include_context: StrictBool = False
    text_format: Literal["prompt-v1", "prompt-context-v1"]
    tfidf: TfidfConfig = Field(default_factory=TfidfConfig)
    svd: SVDConfig | None = None
    svd_output_normalization: NormalizationConfig | None = None
    kmeans: KMeansConfig
    runtime: dict[str, str]


class ClusteringRun(BaseModel):
    """Persisted immutable clustering-run metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    clustering_run_id: str
    name: str
    schema_version: Literal[1] = 1
    configuration_hash: Annotated[str, Field(pattern=HASH_PATTERN)]
    source_manifest_hash: Annotated[str, Field(pattern=HASH_PATTERN)]
    trace_count: Annotated[StrictInt, Field(gt=0)]
    feature_count: Annotated[StrictInt, Field(gt=0)]
    cluster_count: Annotated[StrictInt, Field(gt=0)]
    inertia: Annotated[float, Field(ge=0.0)]
    created_at: datetime
    svd_components: int | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class TraceClusterAssignment(BaseModel):
    """Immutable trace provenance and numeric assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    document_index: Annotated[StrictInt, Field(ge=0)]
    source_timestamp: datetime
    source_trace_hash: Annotated[str, Field(pattern=HASH_PATTERN)]
    document_hash: Annotated[str, Field(pattern=HASH_PATTERN)]
    cluster_number: Annotated[StrictInt, Field(ge=0)]


class SliceSummary(BaseModel):
    """One numeric cluster and its editable label."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_number: int
    label: str | None
    trace_count: int


class ClusteringResult(BaseModel):
    """Completed clustering output returned to the CLI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: ClusteringRun
    assignments: tuple[TraceClusterAssignment, ...]


def build_document(trace: Trace, *, include_context: bool) -> str:
    """Build the exact document supplied to TF-IDF."""
    if not include_context:
        return trace.prompt
    return f"PROMPT:\n{trace.prompt}\nCONTEXT_JSON:\n{_canonical_json(trace.context)}"


def canonical_label(label: str) -> tuple[str, str]:
    """Return the preserved stripped label and its uniqueness key."""
    display = label.strip()
    if not display:
        raise ValueError("label must not be blank")
    return display, unicodedata.normalize("NFKC", display).casefold()


def canonical_trace_payload(trace: Trace) -> str:
    """Encode every stored trace field for source provenance hashing."""
    return _canonical_json(
        {
            "context": trace.context,
            "metadata": trace.metadata,
            "prompt": trace.prompt,
            "response": trace.response,
            "task_type": trace.task_type,
            "timestamp": timestamp_to_text(trace.timestamp),
            "trace_id": trace.trace_id,
        }
    )


def create_clustering_run(
    database_path: Path,
    *,
    name: str,
    clusters: int,
    include_context: bool = False,
    svd_components: int | None = None,
) -> ClusteringResult:
    """Compute and atomically persist one immutable clustering run."""
    normalized_name = name.strip()
    if not normalized_name:
        raise ClusteringValidationError("clustering run name must not be blank")
    if isinstance(clusters, bool) or clusters < 1:
        raise ClusteringValidationError("cluster count must be at least 1")
    if isinstance(svd_components, bool) or (
        svd_components is not None and svd_components < 1
    ):
        raise ClusteringValidationError("SVD components must be at least 1")

    with closing(connect_database(database_path)) as connection:
        traces = _load_ordered_traces(connection)
        if not traces:
            raise ClusteringValidationError("no traces are available to cluster")
        if clusters > len(traces):
            raise ClusteringValidationError(
                f"cluster count {clusters} exceeds trace count {len(traces)}"
            )
        documents = [
            build_document(trace, include_context=include_context) for trace in traces
        ]
        source_records = _source_records(traces, documents)
        source_manifest_hash = _manifest_hash(source_records)

        vectorizer = TfidfVectorizer(**_tfidf_estimator_parameters())
        try:
            tfidf_matrix = vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" in str(error).lower():
                raise ClusteringValidationError(
                    "TF-IDF produced an empty vocabulary"
                ) from error
            raise ClusteringValidationError(f"TF-IDF failed: {error}") from error
        feature_count = int(tfidf_matrix.shape[1])
        effective_matrix: Any = tfidf_matrix
        svd_config: SVDConfig | None = None
        normalization_config: NormalizationConfig | None = None
        if svd_components is not None:
            maximum = min(len(traces), feature_count)
            if svd_components >= maximum:
                raise ClusteringValidationError(
                    "SVD components must satisfy 1 <= n_components < "
                    f"min(trace_count, feature_count) ({maximum})"
                )
            svd_config = SVDConfig(n_components=svd_components)
            reduced_matrix = TruncatedSVD(
                **svd_config.model_dump(mode="python"),
            ).fit_transform(tfidf_matrix)
            _validate_effective_matrix(reduced_matrix, trace_count=len(traces))
            normalization_config = NormalizationConfig(copy_output=False)
            try:
                effective_matrix = normalize(
                    reduced_matrix,
                    **normalization_config.model_dump(mode="python", by_alias=True),
                )
            except ValueError as error:
                raise ClusteringValidationError(
                    f"SVD normalization failed: {error}"
                ) from error
        _validate_effective_matrix(
            effective_matrix,
            trace_count=len(traces),
            require_l2_normalized=normalization_config is not None,
        )
        distinct_count = _distinct_vector_count(effective_matrix)
        if clusters > distinct_count:
            raise ClusteringValidationError(
                f"cluster count {clusters} exceeds distinct final vector count "
                f"{distinct_count}"
            )

        config = _build_config(
            clusters=clusters,
            include_context=include_context,
            svd=svd_config,
            normalization=normalization_config,
        )
        configuration_json = _canonical_json(
            config.model_dump(mode="json", by_alias=True)
        )
        configuration_hash = _sha256(configuration_json)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator = KMeans(**config.kmeans.model_dump(mode="python")).fit(
                    effective_matrix
                )
        except (ConvergenceWarning, ValueError) as error:
            raise ClusteringValidationError(f"KMeans failed: {error}") from error
        if not np.isfinite(estimator.inertia_):
            raise ClusteringValidationError("KMeans produced non-finite inertia")
        raw_labels = [int(value) for value in estimator.labels_]
        if len(set(raw_labels)) != clusters:
            raise ClusteringValidationError(
                "KMeans produced fewer clusters than requested"
            )
        labels = _canonicalize_cluster_numbers(traces, raw_labels)
        created_at = datetime.now(UTC)
        run = ClusteringRun(
            clustering_run_id=f"cluster_run_{uuid4().hex}",
            name=normalized_name,
            configuration_hash=configuration_hash,
            source_manifest_hash=source_manifest_hash,
            trace_count=len(traces),
            feature_count=feature_count,
            cluster_count=clusters,
            inertia=float(estimator.inertia_),
            created_at=created_at,
            svd_components=svd_components,
        )
        assignments = tuple(
            TraceClusterAssignment(
                trace_id=trace.trace_id,
                document_index=index,
                source_timestamp=trace.timestamp,
                source_trace_hash=source_records[index]["source_trace_hash"],
                document_hash=source_records[index]["document_hash"],
                cluster_number=labels[index],
            )
            for index, trace in enumerate(traces)
        )

        try:
            connection.execute("BEGIN IMMEDIATE")
            current_traces = _load_ordered_traces(connection)
            current_documents = [
                build_document(trace, include_context=include_context)
                for trace in current_traces
            ]
            if (
                _manifest_hash(_source_records(current_traces, current_documents))
                != source_manifest_hash
            ):
                raise SourceManifestChangedError(
                    "source traces changed while clustering; retry the command"
                )
            _insert_result(connection, run, configuration_json, assignments)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    return ClusteringResult(run=run, assignments=assignments)


def list_slices(
    database_path: Path, name: str
) -> tuple[ClusteringRun, list[SliceSummary]]:
    """Return a named run and its slices in numeric order."""
    with closing(connect_database(database_path)) as connection:
        row = connection.execute(
            "SELECT * FROM trace_clustering_runs WHERE name = ?", (name.strip(),)
        ).fetchone()
        if row is None:
            raise ClusteringRunNotFoundError(f"clustering run '{name}' was not found")
        run = _run_from_row(row)
        rows = connection.execute(
            """
            SELECT
                label.cluster_number,
                label.label,
                COUNT(assignment.trace_id) AS trace_count
            FROM trace_cluster_labels AS label
            LEFT JOIN trace_cluster_assignments AS assignment
              ON assignment.clustering_run_id = label.clustering_run_id
             AND assignment.cluster_number = label.cluster_number
            WHERE label.clustering_run_id = ?
            GROUP BY label.cluster_number, label.label
            ORDER BY label.cluster_number ASC
            """,
            (run.clustering_run_id,),
        ).fetchall()
        slices = [SliceSummary.model_validate(dict(item)) for item in rows]
        if not slices or sum(item.trace_count for item in slices) == 0:
            raise sqlite3.DatabaseError("clustering run has no assignments")
        return run, slices


def rename_slice(
    database_path: Path, name: str, cluster_number: int, label: str
) -> None:
    """Atomically update only one slice's human-readable label metadata."""
    display, label_key = canonical_label(label)
    with closing(connect_database(database_path)) as connection:
        with connection:
            row = connection.execute(
                "SELECT clustering_run_id, cluster_count "
                "FROM trace_clustering_runs WHERE name = ?",
                (name.strip(),),
            ).fetchone()
            if row is None:
                raise ClusteringRunNotFoundError(
                    f"clustering run '{name}' was not found"
                )
            if cluster_number < 0 or cluster_number >= int(row["cluster_count"]):
                raise SliceNotFoundError(
                    f"cluster {cluster_number} was not found in run '{name}'"
                )
            timestamp = timestamp_to_text(datetime.now(UTC))
            try:
                cursor = connection.execute(
                    """
                    UPDATE trace_cluster_labels
                    SET label = ?, label_key = ?, updated_at = ?
                    WHERE clustering_run_id = ? AND cluster_number = ?
                      AND (label IS NOT ? OR label_key IS NOT ?)
                    """,
                    (
                        display,
                        label_key,
                        timestamp,
                        row["clustering_run_id"],
                        cluster_number,
                        display,
                        label_key,
                    ),
                )
                if cursor.rowcount == 0:
                    existing = connection.execute(
                        "SELECT 1 FROM trace_cluster_labels "
                        "WHERE clustering_run_id = ? AND cluster_number = ?",
                        (row["clustering_run_id"], cluster_number),
                    ).fetchone()
                    if existing is None:
                        raise sqlite3.DatabaseError("clustering label row is missing")
            except sqlite3.IntegrityError as error:
                if "UNIQUE" in str(error):
                    raise DuplicateSliceLabelError(
                        f"label '{display}' is already used in run '{name}'"
                    ) from error
                raise


def _build_config(
    *,
    clusters: int,
    include_context: bool,
    svd: SVDConfig | None,
    normalization: NormalizationConfig | None,
) -> TraceClusteringConfig:
    return TraceClusteringConfig(
        include_context=include_context,
        text_format="prompt-context-v1" if include_context else "prompt-v1",
        tfidf=TfidfConfig(),
        svd=svd,
        svd_output_normalization=normalization,
        kmeans=KMeansConfig(n_clusters=clusters),
        runtime={
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": version("scipy"),
            "tracebench": version("tracebench"),
        },
    )


def _tfidf_identity() -> dict[str, object]:
    return TfidfConfig().model_dump(mode="python")


def _tfidf_estimator_parameters() -> dict[str, object]:
    parameters = _tfidf_identity()
    parameters["ngram_range"] = (1, 2)
    parameters["dtype"] = np.float64
    return parameters


def _load_ordered_traces(connection: sqlite3.Connection) -> list[Trace]:
    rows = connection.execute(
        "SELECT trace_id, timestamp, task_type, prompt, response, "
        "context_json, metadata_json "
        "FROM traces ORDER BY trace_id ASC"
    ).fetchall()
    return [
        Trace.model_validate(
            {
                "trace_id": row["trace_id"],
                "timestamp": row["timestamp"],
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "context": json.loads(row["context_json"]),
                "metadata": json.loads(row["metadata_json"]),
            }
        )
        for row in rows
    ]


def _source_records(traces: list[Trace], documents: list[str]) -> list[dict[str, str]]:
    return [
        {
            "trace_id": trace.trace_id,
            "source_trace_hash": _sha256(canonical_trace_payload(trace)),
            "document_hash": _sha256(document),
        }
        for trace, document in zip(traces, documents, strict=True)
    ]


def _manifest_hash(records: list[dict[str, str]]) -> str:
    return _sha256(_canonical_json(records))


def _validate_effective_matrix(
    matrix: Any,
    *,
    trace_count: int,
    require_l2_normalized: bool = False,
) -> None:
    """Validate the exact matrix that will be inspected or passed to KMeans."""
    if issparse(matrix):
        shape = cast(Any, matrix).shape
        values = cast(Any, matrix).data
    else:
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ClusteringValidationError(
                "effective clustering matrix must be two-dimensional"
            )
        shape = array.shape
        values = array
    if len(shape) != 2:
        raise ClusteringValidationError(
            "effective clustering matrix must be two-dimensional"
        )
    if int(shape[0]) != trace_count:
        raise ClusteringValidationError(
            "effective clustering matrix row count must equal trace count"
        )
    if int(shape[1]) < 1:
        raise ClusteringValidationError(
            "effective clustering matrix must have a positive feature dimension"
        )
    if not np.isfinite(values).all():
        raise ClusteringValidationError(
            "effective clustering matrix contains non-finite values"
        )
    if require_l2_normalized:
        dense = cast(Any, matrix).toarray() if issparse(matrix) else np.asarray(matrix)
        row_norms = np.linalg.norm(dense, axis=1)
        valid_norms = np.isclose(row_norms, 0.0, rtol=0.0, atol=1e-12) | np.isclose(
            row_norms, 1.0, rtol=1e-12, atol=1e-12
        )
        if not valid_norms.all():
            raise ClusteringValidationError(
                "normalized SVD output rows must have L2 norm 1 or be zero vectors"
            )


def _distinct_vector_count(matrix: Any) -> int:
    if issparse(matrix):
        csr = csr_matrix(matrix)
        keys = {
            (
                tuple(csr.indices[csr.indptr[i] : csr.indptr[i + 1]]),
                tuple(csr.data[csr.indptr[i] : csr.indptr[i + 1]]),
            )
            for i in range(csr.shape[0])
        }
        return len(keys)
    return int(np.unique(np.asarray(matrix), axis=0).shape[0])


def _canonicalize_cluster_numbers(traces: list[Trace], labels: list[int]) -> list[int]:
    members: dict[int, list[str]] = {}
    for trace, label in zip(traces, labels, strict=True):
        members.setdefault(label, []).append(trace.trace_id)
    mapping = {
        old: new
        for new, (old, _) in enumerate(
            sorted(members.items(), key=lambda item: tuple(sorted(item[1])))
        )
    }
    return [mapping[label] for label in labels]


def _insert_result(
    connection: sqlite3.Connection,
    run: ClusteringRun,
    configuration_json: str,
    assignments: tuple[TraceClusterAssignment, ...],
) -> None:
    try:
        connection.execute(
            """
            INSERT INTO trace_clustering_runs (
                clustering_run_id, name, schema_version, configuration_hash,
                configuration_json, source_manifest_hash, trace_count,
                feature_count, cluster_count, inertia, created_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.clustering_run_id,
                run.name,
                run.configuration_hash,
                configuration_json,
                run.source_manifest_hash,
                run.trace_count,
                run.feature_count,
                run.cluster_count,
                run.inertia,
                timestamp_to_text(run.created_at),
            ),
        )
    except sqlite3.IntegrityError as error:
        if "UNIQUE" in str(error):
            raise ClusteringRunExistsError(
                f"clustering run '{run.name}' already exists"
            ) from error
        raise
    for assignment in assignments:
        connection.execute(
            """
            INSERT INTO trace_cluster_assignments (
                clustering_run_id, trace_id, document_index, source_timestamp,
                source_trace_hash, document_hash, cluster_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.clustering_run_id,
                assignment.trace_id,
                assignment.document_index,
                timestamp_to_text(assignment.source_timestamp),
                assignment.source_trace_hash,
                assignment.document_hash,
                assignment.cluster_number,
            ),
        )
    created_at = timestamp_to_text(run.created_at)
    for cluster_number in range(run.cluster_count):
        connection.execute(
            "INSERT INTO trace_cluster_labels "
            "(clustering_run_id, cluster_number, label, label_key, "
            "created_at, updated_at) "
            "VALUES (?, ?, NULL, NULL, ?, ?)",
            (run.clustering_run_id, cluster_number, created_at, created_at),
        )


def _run_from_row(row: sqlite3.Row) -> ClusteringRun:
    config = json.loads(row["configuration_json"])
    svd = config.get("svd")
    return ClusteringRun(
        clustering_run_id=row["clustering_run_id"],
        name=row["name"],
        schema_version=row["schema_version"],
        configuration_hash=row["configuration_hash"],
        source_manifest_hash=row["source_manifest_hash"],
        trace_count=row["trace_count"],
        feature_count=row["feature_count"],
        cluster_count=row["cluster_count"],
        inertia=row["inertia"],
        created_at=row["created_at"],
        svd_components=None if svd is None else svd["n_components"],
    )


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
