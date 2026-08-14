"""Tests for reproducible trace clustering and persistent named slices."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import tracebench.clustering as clustering_module
import tracebench.storage as storage_module
from tracebench.cli import app
from tracebench.clustering import (
    ClusteringRunExistsError,
    ClusteringValidationError,
    KMeansConfig,
    NormalizationConfig,
    SVDConfig,
    TraceClusteringConfig,
    build_document,
    canonical_label,
    canonical_trace_payload,
    create_clustering_run,
    list_slices,
    rename_slice,
)
from tracebench.models import Trace
from tracebench.storage import connect_database, insert_trace

runner = CliRunner()


def store_traces(database_path: Path, prompts: list[str]) -> None:
    """Store traces in reverse identity order to exercise deterministic loading."""
    with closing(connect_database(database_path)) as connection:
        with connection:
            for index in reversed(range(len(prompts))):
                insert_trace(
                    connection,
                    Trace.model_validate(
                        {
                            "trace_id": f"trace-{index}",
                            "timestamp": f"2026-08-13T12:00:0{index}Z",
                            "task_type": "support",
                            "prompt": prompts[index],
                            "response": f"response-{index}",
                            "context": {"nested": {"z": 1, "a": "é"}},
                            "metadata": {"index": index},
                        }
                    ),
                )


def test_document_composition_is_exact_and_excludes_unselected_fields() -> None:
    trace = Trace.model_validate(
        {
            "trace_id": "secret-id",
            "timestamp": "2026-08-13T08:00:00-04:00",
            "task_type": "secret-task",
            "prompt": "  Preserve prompt whitespace  ",
            "response": "secret-response",
            "context": {"z": 2, "a": {"é": True}},
            "metadata": {"secret": True},
        }
    )

    assert (
        build_document(trace, include_context=False) == "  Preserve prompt whitespace  "
    )
    assert build_document(trace, include_context=True) == (
        'PROMPT:\n  Preserve prompt whitespace  \nCONTEXT_JSON:\n{"a":{"é":true},"z":2}'
    )
    assert "secret-response" not in build_document(trace, include_context=True)
    assert "secret-task" not in build_document(trace, include_context=True)
    assert "secret-id" not in build_document(trace, include_context=True)


def test_canonical_source_hash_uses_all_fields_and_utc_timestamp() -> None:
    trace = Trace.model_validate(
        {
            "trace_id": "trace",
            "timestamp": "2026-08-13T08:00:00-04:00",
            "task_type": "support",
            "prompt": "Prompt",
            "response": "Response",
            "context": {"b": 2, "a": 1},
            "metadata": {"z": "last", "a": "first"},
        }
    )
    expected = (
        '{"context":{"a":1,"b":2},"metadata":{"a":"first","z":"last"},'
        '"prompt":"Prompt","response":"Response","task_type":"support",'
        '"timestamp":"2026-08-13T12:00:00.000000Z","trace_id":"trace"}'
    )

    assert canonical_trace_payload(trace) == expected
    assert (
        hashlib.sha256(expected.encode()).hexdigest()
        != hashlib.sha256(
            canonical_trace_payload(
                trace.model_copy(update={"response": "changed"})
            ).encode()
        ).hexdigest()
    )


def test_configuration_models_are_strict_and_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SVDConfig.model_validate({"n_components": True})
    with pytest.raises(ValidationError):
        KMeansConfig.model_validate({"n_clusters": True})
    with pytest.raises(ValidationError):
        TraceClusteringConfig.model_validate(
            {
                "include_context": False,
                "text_format": "prompt-v1",
                "kmeans": {"n_clusters": 2},
                "runtime": {},
                "unknown": True,
            }
        )


def test_repeated_snapshot_has_stable_hashes_and_assignments(tmp_path: Path) -> None:
    database_path = tmp_path / "clustering.sqlite3"
    store_traces(
        database_path,
        ["refund payment card", "refund card charge", "password login reset"],
    )

    first = create_clustering_run(database_path, name="slices-v1", clusters=2)
    second = create_clustering_run(database_path, name="slices-v2", clusters=2)

    assert first.run.clustering_run_id != second.run.clustering_run_id
    assert first.run.configuration_hash == second.run.configuration_hash
    assert first.run.source_manifest_hash == second.run.source_manifest_hash
    assert [item.trace_id for item in first.assignments] == [
        "trace-0",
        "trace-1",
        "trace-2",
    ]
    assert [item.cluster_number for item in first.assignments] == [
        item.cluster_number for item in second.assignments
    ]
    with closing(connect_database(database_path)) as connection:
        config = json.loads(
            connection.execute(
                "SELECT configuration_json FROM trace_clustering_runs WHERE name = ?",
                ("slices-v1",),
            ).fetchone()[0]
        )
    assert config["schema_version"] == 1
    assert config["tfidf"]["token_pattern"] == r"(?u)\b\w\w+\b"
    assert config["kmeans"]["n_init"] == 20
    assert config["kmeans"]["verbose"] == 0
    assert config["tfidf"]["input"] == "content"
    assert config["tfidf"]["preprocessor"] is None
    assert config["svd_output_normalization"] is None
    assert set(config["runtime"]) == {"numpy", "scikit-learn", "scipy", "tracebench"}


def test_default_pipeline_does_not_construct_svd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "no-svd.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("SVD must not be constructed")

    monkeypatch.setattr(clustering_module, "TruncatedSVD", forbidden)
    result = create_clustering_run(database_path, name="no-svd", clusters=2)

    assert result.run.svd_components is None


@pytest.mark.parametrize("components", [0, 2, 3])
def test_svd_strict_boundaries_are_rejected(tmp_path: Path, components: int) -> None:
    database_path = tmp_path / f"svd-{components}.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta"])

    with pytest.raises(ClusteringValidationError, match="SVD"):
        create_clustering_run(
            database_path,
            name=f"svd-{components}",
            clusters=1,
            svd_components=components,
        )


def test_distinct_tfidf_rows_collapsing_after_svd_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "collapsed.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta", "epsilon zeta"])

    class CollapsingSVD:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, matrix: object) -> np.ndarray:
            return np.ones((3, 1), dtype=np.float64)

    monkeypatch.setattr(clustering_module, "TruncatedSVD", CollapsingSVD)

    with pytest.raises(
        ClusteringValidationError, match="distinct final vector count 1"
    ):
        create_clustering_run(
            database_path,
            name="collapsed",
            clusters=2,
            svd_components=1,
        )


def test_non_finite_effective_svd_matrix_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "non-finite.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta", "epsilon zeta"])

    class NonFiniteSVD:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, matrix: object) -> np.ndarray:
            return np.array([[np.nan], [1.0], [2.0]])

    monkeypatch.setattr(clustering_module, "TruncatedSVD", NonFiniteSVD)

    with pytest.raises(ClusteringValidationError, match="non-finite"):
        create_clustering_run(
            database_path,
            name="non-finite",
            clusters=1,
            svd_components=1,
        )


def test_persisted_configuration_matches_effective_invocation_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "parameters.sqlite3"
    store_traces(
        database_path,
        ["refund payment card", "password login reset", "shipping order delay"],
    )
    captured: dict[str, dict[str, object]] = {}
    real_tfidf = clustering_module.TfidfVectorizer
    real_svd = clustering_module.TruncatedSVD
    real_normalize = clustering_module.normalize
    real_kmeans = clustering_module.KMeans

    class CapturingTfidf:
        def __init__(self, **kwargs: object) -> None:
            captured["tfidf"] = kwargs
            self.delegate = real_tfidf(**kwargs)

        def fit_transform(self, documents: list[str]) -> object:
            return self.delegate.fit_transform(documents)

    class CapturingSVD:
        def __init__(self, **kwargs: object) -> None:
            captured["svd"] = kwargs
            self.delegate = real_svd(**kwargs)

        def fit_transform(self, matrix: object) -> np.ndarray:
            return self.delegate.fit_transform(matrix)

    class CapturingKMeans:
        def __init__(self, **kwargs: object) -> None:
            captured["kmeans"] = kwargs
            self.delegate = real_kmeans(**kwargs)

        def fit(self, matrix: object) -> "CapturingKMeans":
            self.delegate.fit(matrix)
            self.inertia_ = self.delegate.inertia_
            self.labels_ = self.delegate.labels_
            return self

    def capturing_normalize(matrix: object, **kwargs: object) -> np.ndarray:
        captured["normalization"] = kwargs
        return real_normalize(matrix, **kwargs)

    monkeypatch.setattr(clustering_module, "TfidfVectorizer", CapturingTfidf)
    monkeypatch.setattr(clustering_module, "TruncatedSVD", CapturingSVD)
    monkeypatch.setattr(clustering_module, "normalize", capturing_normalize)
    monkeypatch.setattr(clustering_module, "KMeans", CapturingKMeans)

    create_clustering_run(
        database_path, name="parameters", clusters=2, svd_components=2
    )
    with closing(connect_database(database_path)) as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT configuration_json FROM trace_clustering_runs"
            ).fetchone()[0]
        )

    expected_tfidf = dict(persisted["tfidf"])
    expected_tfidf["dtype"] = np.float64
    expected_tfidf["ngram_range"] = (1, 2)
    assert captured["tfidf"] == expected_tfidf
    assert captured["svd"] == persisted["svd"]
    assert captured["normalization"] == persisted["svd_output_normalization"]
    assert captured["kmeans"] == persisted["kmeans"]
    assert persisted["svd_output_normalization"] == NormalizationConfig(
        copy_output=False
    ).model_dump(mode="json", by_alias=True)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones(3), "two-dimensional"),
        (np.ones((2, 1)), "row count"),
        (np.ones((4, 1)), "row count"),
        (np.empty((3, 0)), "positive feature dimension"),
        (np.array([[np.nan], [1.0], [2.0]]), "non-finite"),
    ],
    ids=["one-dimensional", "truncated-rows", "extra-rows", "zero-columns", "nan"],
)
def test_invalid_effective_matrix_is_cli_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: np.ndarray,
    message: str,
) -> None:
    database_path = tmp_path / f"matrix-{message.replace(' ', '-')}.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta", "epsilon zeta"])

    class InvalidSVD:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, matrix: object) -> np.ndarray:
            return output

    monkeypatch.setattr(clustering_module, "TruncatedSVD", InvalidSVD)
    result = runner.invoke(
        app,
        [
            "traces",
            "cluster",
            "--name",
            "invalid-matrix",
            "--clusters",
            "1",
            "--svd-components",
            "1",
        ],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert message in result.stderr
    assert "no clustering run was persisted" in result.stderr
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_clustering_runs").fetchone()[
                0
            ]
            == 0
        )


def test_invalid_normalized_svd_output_is_cli_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "invalid-normalization.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta", "epsilon zeta"])
    monkeypatch.setattr(
        clustering_module,
        "normalize",
        lambda matrix, **kwargs: np.array([[2.0], [3.0], [4.0]]),
    )

    result = runner.invoke(
        app,
        [
            "traces",
            "cluster",
            "--name",
            "invalid-normalization",
            "--clusters",
            "1",
            "--svd-components",
            "1",
        ],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert "L2 norm 1 or be zero vectors" in result.stderr
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_clustering_runs").fetchone()[
                0
            ]
            == 0
        )


def test_normalized_svd_output_allows_legitimate_zero_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "zero-row.sqlite3"
    store_traces(database_path, ["alpha beta", "gamma delta", "epsilon zeta"])

    class SVDWithZeroRow:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, matrix: object) -> np.ndarray:
            return np.array([[0.0], [1.0], [-1.0]])

    monkeypatch.setattr(clustering_module, "TruncatedSVD", SVDWithZeroRow)
    result = create_clustering_run(
        database_path,
        name="zero-row",
        clusters=3,
        svd_components=1,
    )

    assert len(result.assignments) == 3


def test_empty_duplicate_and_invalid_cluster_counts_persist_nothing(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "empty.sqlite3"
    with pytest.raises(ClusteringValidationError, match="no traces"):
        create_clustering_run(empty_path, name="empty", clusters=1)

    duplicate_path = tmp_path / "duplicates.sqlite3"
    store_traces(duplicate_path, ["same document", "same document"])
    with pytest.raises(ClusteringValidationError, match="distinct final vector"):
        create_clustering_run(duplicate_path, name="duplicates", clusters=2)
    with closing(connect_database(duplicate_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_clustering_runs").fetchone()[
                0
            ]
            == 0
        )

    vocabulary_path = tmp_path / "empty-vocabulary.sqlite3"
    store_traces(vocabulary_path, ["a"])
    with pytest.raises(ClusteringValidationError, match="empty vocabulary"):
        create_clustering_run(vocabulary_path, name="vocabulary", clusters=1)


def test_unique_run_name_and_immutable_assignments(tmp_path: Path) -> None:
    database_path = tmp_path / "immutable.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    result = create_clustering_run(database_path, name="immutable", clusters=2)

    with pytest.raises(ClusteringRunExistsError):
        create_clustering_run(database_path, name="immutable", clusters=2)
    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE trace_cluster_assignments SET cluster_number = 0 "
                "WHERE clustering_run_id = ?",
                (result.run.clustering_run_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE trace_clustering_runs SET name = 'changed' "
                "WHERE clustering_run_id = ?",
                (result.run.clustering_run_id,),
            )


def test_completed_run_rejects_new_assignment_and_preserves_trace_count(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "assignment-insert.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    result = create_clustering_run(database_path, name="immutable", clusters=2)

    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": "later-trace",
                        "timestamp": "2026-08-13T12:00:09Z",
                        "task_type": "support",
                        "prompt": "shipping delay",
                    }
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(
                    "INSERT INTO trace_cluster_assignments VALUES "
                    "(?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.run.clustering_run_id,
                        "later-trace",
                        result.run.trace_count - 1,
                        "2026-08-13T12:00:09.000000Z",
                        "0" * 64,
                        "1" * 64,
                        0,
                    ),
                )
        stored_count = connection.execute(
            "SELECT COUNT(*) FROM trace_cluster_assignments "
            "WHERE clustering_run_id = ?",
            (result.run.clustering_run_id,),
        ).fetchone()[0]

    assert stored_count == result.run.trace_count


def test_assignment_document_index_must_be_inside_run_snapshot(tmp_path: Path) -> None:
    database_path = tmp_path / "assignment-index.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    with closing(connect_database(database_path)) as connection:
        connection.execute("BEGIN")
        connection.execute(
            "INSERT INTO trace_clustering_runs VALUES "
            "(?, ?, 1, ?, ?, ?, 2, 2, 1, 0, ?)",
            (
                "cluster_run_" + "0" * 32,
                "unfinished-test-run",
                "0" * 64,
                "{}",
                "1" * 64,
                "2026-08-13T12:00:00.000000Z",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="outside its run"):
            connection.execute(
                "INSERT INTO trace_cluster_assignments VALUES (?, ?, 2, ?, ?, ?, 0)",
                (
                    "cluster_run_" + "0" * 32,
                    "trace-0",
                    "2026-08-13T12:00:00.000000Z",
                    "2" * 64,
                    "3" * 64,
                ),
            )
        connection.rollback()


def test_labels_use_nfkc_casefold_key_and_do_not_change_assignments(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "labels.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    result = create_clustering_run(database_path, name="labels", clusters=2)
    before = [(item.trace_id, item.cluster_number) for item in result.assignments]
    kelvin_label = "\u212aelvin"

    rename_slice(database_path, "labels", 0, f"  {kelvin_label}  ")
    assert canonical_label(f"  {kelvin_label}  ") == (kelvin_label, "kelvin")
    with pytest.raises(clustering_module.DuplicateSliceLabelError):
        rename_slice(database_path, "labels", 1, "KELVIN")
    rename_slice(database_path, "labels", 0, kelvin_label)

    _, slices = list_slices(database_path, "labels")
    assert slices[0].label == kelvin_label
    with closing(connect_database(database_path)) as connection:
        row = connection.execute(
            "SELECT label, label_key FROM trace_cluster_labels "
            "WHERE clustering_run_id = ? AND cluster_number = 0",
            (result.run.clustering_run_id,),
        ).fetchone()
        after = [
            (item["trace_id"], item["cluster_number"])
            for item in connection.execute(
                "SELECT trace_id, cluster_number FROM trace_cluster_assignments "
                "WHERE clustering_run_id = ? ORDER BY trace_id",
                (result.run.clustering_run_id,),
            )
        ]
        with pytest.raises(sqlite3.IntegrityError, match="atomically"):
            connection.execute(
                "UPDATE trace_cluster_labels SET label = 'changed' "
                "WHERE clustering_run_id = ? AND cluster_number = 0",
                (result.run.clustering_run_id,),
            )
    assert dict(row) == {"label": kelvin_label, "label_key": "kelvin"}
    assert after == before


def test_partial_preexisting_clustering_schema_is_rejected(tmp_path: Path) -> None:
    database_path = tmp_path / "partial.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("CREATE TABLE trace_clustering_runs (id TEXT)")

    with pytest.raises(sqlite3.DatabaseError, match="incomplete or incompatible"):
        connect_database(database_path)


@pytest.mark.parametrize(
    "defect",
    [
        "table-metadata",
        "primary-key",
        "foreign-key",
        "index",
        "check",
        "trigger",
        "missing-object",
        "extra-object",
    ],
)
def test_every_malformed_clustering_schema_category_is_rejected(
    tmp_path: Path, defect: str
) -> None:
    database_path = tmp_path / f"malformed-{defect}.sqlite3"
    statements = [
        statement
        for statement in storage_module.SCHEMA_STATEMENTS
        if any(
            table_name in statement
            for table_name in storage_module._CLUSTERING_TABLE_NAMES
        )
    ] + list(storage_module.CLUSTERING_TRIGGER_STATEMENTS)

    if defect == "table-metadata":
        statements = [
            statement.replace(
                "trace_count INTEGER NOT NULL",
                "trace_count TEXT DEFAULT 0",
            )
            for statement in statements
        ]
    elif defect == "primary-key":
        statements = [
            statement.replace(
                "PRIMARY KEY (clustering_run_id, trace_id)",
                "UNIQUE (clustering_run_id, trace_id)",
            )
            for statement in statements
        ]
    elif defect == "foreign-key":
        statements = [
            statement.replace(
                "REFERENCES traces(trace_id) ON DELETE RESTRICT",
                "REFERENCES traces(trace_id) ON DELETE CASCADE",
            )
            for statement in statements
        ]
    elif defect == "index":
        statements = [
            statement.replace(
                "(clustering_run_id, cluster_number, trace_id)",
                "(clustering_run_id, trace_id, cluster_number)",
            )
            for statement in statements
        ]
    elif defect == "check":
        statements = [
            statement.replace(
                "trace_count > 0",
                "trace_count >= 0",
            )
            for statement in statements
        ]
    elif defect == "trigger":
        statements = [
            statement.replace(
                "cluster assignments are immutable",
                "cluster assignments may change",
            )
            for statement in statements
        ]
    elif defect == "missing-object":
        statements = [
            statement
            for statement in statements
            if "prevent_trace_cluster_assignment_delete" not in statement
        ]
    elif defect == "extra-object":
        statements.append(
            "CREATE INDEX trace_cluster_assignments_extra "
            "ON trace_cluster_assignments (trace_id)"
        )

    with closing(sqlite3.connect(database_path)) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()

    with pytest.raises(sqlite3.DatabaseError, match="incomplete or incompatible"):
        connect_database(database_path)


def test_source_manifest_change_rolls_back_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "changed.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    original = clustering_module._load_ordered_traces
    calls = 0

    def changed_snapshot(connection: sqlite3.Connection) -> list[Trace]:
        nonlocal calls
        calls += 1
        traces = original(connection)
        if calls == 2:
            traces[0] = traces[0].model_copy(update={"prompt": "changed prompt"})
        return traces

    monkeypatch.setattr(clustering_module, "_load_ordered_traces", changed_snapshot)

    with pytest.raises(clustering_module.SourceManifestChangedError):
        create_clustering_run(database_path, name="changed", clusters=2)
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_clustering_runs").fetchone()[
                0
            ]
            == 0
        )


def test_run_persistence_failure_rolls_back_every_b1_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "rollback.sqlite3"
    store_traces(database_path, ["refund payment", "password reset"])
    original = clustering_module._insert_result

    def fail_after_inserts(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(clustering_module, "_insert_result", fail_after_inserts)

    with pytest.raises(RuntimeError, match="injected"):
        create_clustering_run(database_path, name="rollback", clusters=2)
    with closing(connect_database(database_path)) as connection:
        assert [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "trace_clustering_runs",
                "trace_cluster_assignments",
                "trace_cluster_labels",
            )
        ] == [0, 0, 0]


def test_cli_cluster_list_and_rename(tmp_path: Path) -> None:
    database_path = tmp_path / "cli.sqlite3"
    store_traces(database_path, ["single support trace"])
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}

    clustered = runner.invoke(
        app,
        ["traces", "cluster", "--name", "support-v1", "--clusters", "1"],
        env=environment,
    )
    assert clustered.exit_code == 0, clustered.output
    assert "Created clustering run support-v1 (cluster_run_" in clustered.stdout
    assert "SVD components: disabled" in clustered.stdout

    renamed = runner.invoke(
        app,
        ["slices", "rename", "support-v1", "0", "refunds"],
        env=environment,
    )
    assert renamed.exit_code == 0, renamed.output
    assert renamed.stdout == "Renamed slice 0 in support-v1 to refunds.\n"

    listed = runner.invoke(app, ["slices", "list", "support-v1"], env=environment)
    assert listed.exit_code == 0, listed.output
    assert "CLUSTER" in listed.stdout
    assert "refunds" in listed.stdout
    assert "1" in listed.stdout


def test_cli_validation_error_is_exit_two_without_a_run(tmp_path: Path) -> None:
    database_path = tmp_path / "cli-empty.sqlite3"
    result = runner.invoke(
        app,
        ["traces", "cluster", "--name", "empty", "--clusters", "1"],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert "no clustering run was persisted" in result.stderr
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM trace_clustering_runs").fetchone()[
                0
            ]
            == 0
        )
