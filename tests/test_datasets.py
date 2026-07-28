"""Tests for versioned evaluation dataset workflows and persistence."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import tracebench.datasets as datasets_module
import tracebench.storage as storage_module
from tracebench.datasets import (
    DatasetAlreadyExistsError,
    DatasetNotFoundError,
    DuplicateTraceError,
    ExportFileExistsError,
    InvalidDatasetReferenceError,
    TraceNotFoundError,
    create_dataset,
    export_dataset,
    generate_dataset_id,
    generate_eval_id,
    get_dataset_and_cases,
    get_datasets,
    parse_dataset_reference,
    promote_trace,
)
from tracebench.models import (
    EvaluationMode,
    Priority,
    ReviewStatus,
    ScorerConfig,
    Trace,
)
from tracebench.storage import connect_database, insert_trace


def store_trace(
    database_path: Path,
    trace_id: str = "trace-001",
    *,
    response: str | None = "Ottawa",
) -> None:
    """Store one source trace for promotion tests."""
    trace = Trace.model_validate(
        {
            "trace_id": trace_id,
            "timestamp": "2026-07-28T14:00:00Z",
            "task_type": "question-answering",
            "prompt": "What is the capital of Canada?",
            "response": response,
            "context": {"locale": "en-CA"},
            "metadata": {"request_id": f"request-{trace_id}"},
        }
    )
    with closing(connect_database(database_path)) as connection, connection:
        assert insert_trace(connection, trace)


def test_dataset_creation_and_listing(tmp_path: Path) -> None:
    """Created datasets persist and list by name/version with case counts."""
    database_path = tmp_path / "tracebench.sqlite3"
    later = create_dataset(database_path, name="zeta", version="1")
    earlier = create_dataset(database_path, name="alpha", version="2")

    datasets = get_datasets(database_path)

    assert [dataset.dataset_id for dataset, _ in datasets] == [
        earlier.dataset_id,
        later.dataset_id,
    ]
    assert [count for _, count in datasets] == [0, 0]
    assert later.dataset_id.startswith("dataset_")


def test_duplicate_name_and_version_are_rejected(tmp_path: Path) -> None:
    """A dataset version can only be created once."""
    database_path = tmp_path / "tracebench.sqlite3"
    create_dataset(database_path, name="support", version="0.1")

    with pytest.raises(DatasetAlreadyExistsError, match="already exists"):
        create_dataset(database_path, name="support", version="0.1")

    assert len(get_datasets(database_path)) == 1


@pytest.mark.parametrize("padding", [" ", "\t", "\u00a0"])
def test_canonical_whitespace_duplicates_are_rejected(
    tmp_path: Path, padding: str
) -> None:
    """Stored dataset selectors are trimmed before uniqueness is applied."""
    database_path = tmp_path / "tracebench.sqlite3"
    padded_name = f"{padding}support{padding}"
    padded_version = f"{padding}0.1{padding}"
    first = create_dataset(database_path, name=padded_name, version=padded_version)

    with pytest.raises(DatasetAlreadyExistsError, match="already exists"):
        create_dataset(database_path, name="support", version="0.1")

    assert first.name == "support"
    assert first.version == "0.1"
    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            with connection:
                connection.execute(
                    """
                    INSERT INTO eval_datasets (
                        dataset_id, name, version, description, created_at
                    ) VALUES ('raw-id', ?, '0.1', '', ?)
                    """,
                    (padded_name, "2026-07-28T14:00:00.000000Z"),
                )


@pytest.mark.parametrize(
    "reference",
    ["support", "support:", ":1", "support:1:extra", " : "],
)
def test_malformed_dataset_references_are_rejected(reference: str) -> None:
    """References require exactly one nonblank name/version separator."""
    with pytest.raises(InvalidDatasetReferenceError, match="expected name:version"):
        parse_dataset_reference(reference)


def test_dataset_reference_is_canonicalized() -> None:
    """Reference lookup identity strips surrounding selector whitespace."""
    assert parse_dataset_reference(" support : 0.1 ") == ("support", "0.1")


def test_dataset_ids_are_stable_across_clean_databases(tmp_path: Path) -> None:
    """Canonical dataset identity produces the same UUID5 in every database."""
    first = create_dataset(tmp_path / "first.sqlite3", name=" support ", version=" 1 ")
    second = create_dataset(tmp_path / "second.sqlite3", name="support", version="1")
    different_version = create_dataset(
        tmp_path / "third.sqlite3", name="support", version="2"
    )

    assert first.dataset_id == second.dataset_id
    assert first.dataset_id == generate_dataset_id(" support ", " 1 ")
    assert different_version.dataset_id != first.dataset_id


def test_eval_ids_are_stable_across_clean_databases(tmp_path: Path) -> None:
    """The same canonical dataset and trace produce the same case UUID5."""
    cases = []
    for filename in ("first.sqlite3", "second.sqlite3"):
        database_path = tmp_path / filename
        store_trace(database_path)
        create_dataset(database_path, name="support", version="1")
        cases.append(
            promote_trace(
                database_path,
                dataset_reference="support:1",
                trace_id="trace-001",
                mode=EvaluationMode.DETERMINISTIC,
                scorers=[ScorerConfig(name="exact_match")],
            )
        )

    assert cases[0].eval_id == cases[1].eval_id
    assert cases[0].dataset_id == cases[1].dataset_id


def test_different_dataset_versions_produce_different_case_ids(tmp_path: Path) -> None:
    """A version change alters both the dataset and evaluation-case identity."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    first_dataset = create_dataset(database_path, name="support", version="1")
    second_dataset = create_dataset(database_path, name="support", version="2")

    cases = [
        promote_trace(
            database_path,
            dataset_reference=f"support:{version}",
            trace_id="trace-001",
            mode=EvaluationMode.DETERMINISTIC,
            scorers=[ScorerConfig(name="exact_match")],
        )
        for version in ("1", "2")
    ]

    assert first_dataset.dataset_id != second_dataset.dataset_id
    assert cases[0].eval_id != cases[1].eval_id


def test_existing_trace_is_promoted_as_a_snapshot(tmp_path: Path) -> None:
    """Promotion snapshots source fields and applies workflow defaults."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    dataset = create_dataset(database_path, name="support", version="0.1")

    case = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="trace-001",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )

    assert case.dataset_id == dataset.dataset_id
    assert case.source_trace_id == "trace-001"
    assert case.source_timestamp.isoformat() == "2026-07-28T14:00:00+00:00"
    assert case.source_task_type == "question-answering"
    assert case.source_response == "Ottawa"
    assert case.source_metadata == {"request_id": "request-trace-001"}
    assert case.input == "What is the capital of Canada?"
    assert case.context == {"locale": "en-CA"}
    assert case.reference_answer == "Ottawa"
    assert case.eval_id == generate_eval_id(dataset.dataset_id, "trace-001")
    assert case.priority is Priority.MEDIUM
    assert case.review_status is ReviewStatus.DRAFT
    _, stored_cases = get_dataset_and_cases(database_path, "support:0.1")
    assert stored_cases == [case]


def test_reference_answer_override_and_explicit_metadata(tmp_path: Path) -> None:
    """Promotion accepts an override and explicit workflow values."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    create_dataset(database_path, name="support", version="0.1")

    case = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="trace-001",
        mode=EvaluationMode.REFERENCE,
        reference_answer="Toronto",
        priority=Priority.CRITICAL,
        review_status=ReviewStatus.APPROVED,
    )

    assert case.reference_answer == "Toronto"
    assert case.priority is Priority.CRITICAL
    assert case.review_status is ReviewStatus.APPROVED


def test_source_response_does_not_leak_into_other_modes(tmp_path: Path) -> None:
    """Non-reference modes cannot retain or use the source response."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path, "deterministic")
    store_trace(database_path, "rubric")
    create_dataset(database_path, name="support", version="0.1")

    deterministic = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="deterministic",
        mode=EvaluationMode.DETERMINISTIC,
        scorers=[ScorerConfig(name="exact_match")],
    )
    rubric = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="rubric",
        mode=EvaluationMode.RUBRIC,
        rubric=["Correct"],
    )

    assert deterministic.source_response is None
    assert deterministic.reference_answer is None
    assert rubric.source_response is None
    assert rubric.reference_answer is None
    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            with connection:
                connection.execute(
                    "UPDATE eval_cases SET source_response = ? WHERE eval_id = ?",
                    ("Ottawa", deterministic.eval_id),
                )


def test_missing_trace_and_dataset_are_rejected_atomically(tmp_path: Path) -> None:
    """Failed promotion cannot leave a partial evaluation case."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    dataset = create_dataset(database_path, name="support", version="0.1")

    with pytest.raises(TraceNotFoundError, match="missing"):
        promote_trace(
            database_path,
            dataset_reference="support:0.1",
            trace_id="missing",
            mode=EvaluationMode.REFERENCE,
            reference_answer="answer",
        )
    with pytest.raises(DatasetNotFoundError, match=r"missing:0\.1"):
        promote_trace(
            database_path,
            dataset_reference="missing:0.1",
            trace_id="trace-001",
            mode=EvaluationMode.REFERENCE,
        )

    with closing(connect_database(database_path)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM eval_cases WHERE dataset_id = ?",
            (dataset.dataset_id,),
        ).fetchone()[0]
    assert count == 0


def test_duplicate_trace_promotion_is_rejected(tmp_path: Path) -> None:
    """One source trace can appear only once in a dataset version."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    create_dataset(database_path, name="support", version="0.1")
    arguments = {
        "dataset_reference": "support:0.1",
        "trace_id": "trace-001",
        "mode": EvaluationMode.REFERENCE,
        "use_source_response": True,
    }
    promote_trace(database_path, **arguments)

    with pytest.raises(DuplicateTraceError, match="already in dataset"):
        promote_trace(database_path, **arguments)


def test_foreign_keys_are_enabled_and_enforced(tmp_path: Path) -> None:
    """Raw invalid evaluation membership is refused by SQLite."""
    database_path = tmp_path / "tracebench.sqlite3"
    dataset = create_dataset(database_path, name="support", version="0.1")

    with closing(connect_database(database_path)) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            with connection:
                connection.execute(
                    """
                    INSERT INTO eval_cases (
                        eval_id, dataset_id, source_trace_id, source_timestamp,
                        source_task_type, source_response, source_metadata_json,
                        input, context_json, evaluation_mode, reference_answer,
                        rubric_json, scorers_json, priority, review_status, created_at
                    ) VALUES (?, ?, ?, ?, 'question-answering', NULL, '{}', ?, '{}',
                              'reference', ?, '[]', '[]', 'medium', 'draft', ?)
                    """,
                    (
                        "eval-invalid",
                        dataset.dataset_id,
                        "missing-trace",
                        "2026-07-28T14:00:00.000000Z",
                        "input",
                        "answer",
                        "2026-07-28T14:00:00.000000Z",
                    ),
                )


@pytest.mark.parametrize(
    ("column", "invalid_json"),
    [
        ("context_json", "[]"),
        ("source_metadata_json", "[]"),
    ],
)
def test_database_rejects_non_object_json_fields(
    tmp_path: Path, column: str, invalid_json: str
) -> None:
    """SQLite protects the object shape required for case reconstruction."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    dataset = create_dataset(database_path, name="support", version="1")

    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            with connection:
                connection.execute(
                    f"""
                    INSERT INTO eval_cases (
                        eval_id, dataset_id, source_trace_id, source_timestamp,
                        source_task_type, source_response, source_metadata_json,
                        input, context_json, evaluation_mode, reference_answer,
                        rubric_json, scorers_json, priority, review_status, created_at
                    ) VALUES (?, ?, ?, ?, 'question-answering', 'Ottawa',
                              {"?" if column == "source_metadata_json" else "'{}'"},
                              'input', {"?" if column == "context_json" else "'{}'"},
                              'reference', 'Ottawa', '[]', '[]', 'medium', 'draft', ?)
                    """,
                    (
                        "eval-invalid-json",
                        dataset.dataset_id,
                        "trace-001",
                        "2026-07-28T14:00:00.000000Z",
                        invalid_json,
                        "2026-07-28T14:00:00.000000Z",
                    ),
                )


@pytest.mark.parametrize(
    "invalid_json",
    ["[null]", '[" "]', '["same"," same "]'],
)
def test_database_rejects_invalid_rubric_items(
    tmp_path: Path, invalid_json: str
) -> None:
    """SQLite refuses rubric arrays that cannot reconstruct as model criteria."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    create_dataset(database_path, name="support", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="support:1",
        trace_id="trace-001",
        mode=EvaluationMode.RUBRIC,
        rubric=["Correct"],
    )

    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="rubric_json"):
            with connection:
                connection.execute(
                    "UPDATE eval_cases SET rubric_json = ? WHERE eval_id = ?",
                    (invalid_json, case.eval_id),
                )


@pytest.mark.parametrize(
    "invalid_json",
    [
        "[1]",
        '[{"config":{}}]',
        '[{"name":" ","config":{}}]',
        '[{"name":"exact","config":[]}]',
        '[{"name":"exact","unknown":true}]',
    ],
)
def test_database_rejects_invalid_scorer_items(
    tmp_path: Path, invalid_json: str
) -> None:
    """SQLite refuses scorer arrays that cannot reconstruct as model configs."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    create_dataset(database_path, name="support", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="support:1",
        trace_id="trace-001",
        mode=EvaluationMode.DETERMINISTIC,
        scorers=[ScorerConfig(name="exact")],
    )

    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="scorers_json"):
            with connection:
                connection.execute(
                    "UPDATE eval_cases SET scorers_json = ? WHERE eval_id = ?",
                    (invalid_json, case.eval_id),
                )


def test_schema_initialization_rolls_back_all_new_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late schema error rolls back tables created earlier in the attempt."""
    database_path = tmp_path / "broken.sqlite3"
    monkeypatch.setattr(
        storage_module,
        "SCHEMA_STATEMENTS",
        (
            "CREATE TABLE first_schema_object (id INTEGER)",
            "CREATE TABLE second_schema_object (id INTEGER)",
            "THIS IS NOT VALID SQL",
        ),
    )

    with pytest.raises(sqlite3.OperationalError):
        connect_database(database_path)

    with closing(sqlite3.connect(database_path)) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "first_schema_object" not in names
    assert "second_schema_object" not in names


def test_legacy_unmerged_eval_schema_requires_recreation(tmp_path: Path) -> None:
    """Old development eval tables fail with the documented reset guidance."""
    database_path = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute(
            """
            CREATE TABLE traces (
                trace_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                context_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE eval_datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE eval_cases (
                eval_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                source_trace_id TEXT NOT NULL,
                source_timestamp TEXT NOT NULL,
                source_task_type TEXT NOT NULL,
                source_response TEXT,
                source_metadata_json TEXT NOT NULL,
                input TEXT NOT NULL,
                context_json TEXT NOT NULL,
                evaluation_mode TEXT NOT NULL,
                reference_answer TEXT,
                rubric_json TEXT NOT NULL,
                scorers_json TEXT NOT NULL,
                priority TEXT NOT NULL,
                review_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    with pytest.raises(sqlite3.DatabaseError, match=r"recreate.*development database"):
        connect_database(database_path)


def test_export_writes_metadata_and_auditable_case_envelopes(tmp_path: Path) -> None:
    """Export writes dataset metadata before self-contained ordered cases."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path, "trace-001")
    store_trace(database_path, "trace-002", response=None)
    create_dataset(database_path, name="support", version="0.1")
    first = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="trace-001",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )
    second = promote_trace(
        database_path,
        dataset_reference="support:0.1",
        trace_id="trace-002",
        mode=EvaluationMode.DETERMINISTIC,
        scorers=[ScorerConfig(name="exact_match", config={"case_sensitive": False})],
    )
    output_path = tmp_path / "support.jsonl"

    count = export_dataset(database_path, "support:0.1", output_path)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert count == 2
    assert records[0]["record_type"] == "dataset"
    assert records[0]["dataset"]["dataset_ref"] == "support:0.1"
    assert records[0]["dataset"]["dataset_id"] == first.dataset_id
    assert [record["record_type"] for record in records[1:]] == [
        "eval_case",
        "eval_case",
    ]
    cases_by_id = {record["case"]["eval_id"]: record["case"] for record in records[1:]}
    first_record = cases_by_id[first.eval_id]
    second_record = cases_by_id[second.eval_id]
    assert [record["case"]["eval_id"] for record in records[1:]] == sorted(cases_by_id)
    assert first_record["dataset_ref"] == "support:0.1"
    assert first_record["source_trace_id"] == "trace-001"
    assert first_record["source_timestamp"] == "2026-07-28T14:00:00.000000Z"
    assert first_record["source_task_type"] == "question-answering"
    assert first_record["source_response"] == "Ottawa"
    assert first_record["source_metadata"] == {"request_id": "request-trace-001"}
    assert first_record["reference_answer"] == "Ottawa"
    assert second_record["reference_answer"] is None
    assert second_record["source_response"] is None
    assert second_record["scorers"] == [
        {"name": "exact_match", "config": {"case_sensitive": False}}
    ]
    assert second_record["priority"] == "medium"
    assert second_record["review_status"] == "draft"
    assert output_path.read_bytes().endswith(b"\n")


def test_export_protects_existing_file_unless_overwritten(tmp_path: Path) -> None:
    """Export never silently destroys an existing destination."""
    database_path = tmp_path / "tracebench.sqlite3"
    create_dataset(database_path, name="empty", version="1")
    output_path = tmp_path / "empty.jsonl"
    output_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(ExportFileExistsError, match="already exists"):
        export_dataset(database_path, "empty:1", output_path)
    assert output_path.read_text(encoding="utf-8") == "keep me"

    count = export_dataset(
        database_path,
        "empty:1",
        output_path,
        overwrite=True,
    )
    assert count == 0
    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "dataset": {
                "created_at": records[0]["dataset"]["created_at"],
                "dataset_id": generate_dataset_id("empty", "1"),
                "dataset_ref": "empty:1",
                "description": "",
                "name": "empty",
                "version": "1",
            },
            "record_type": "dataset",
        }
    ]


def test_non_overwrite_export_does_not_require_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Safe exclusive creation remains usable when hard links are unavailable."""
    database_path = tmp_path / "tracebench.sqlite3"
    create_dataset(database_path, name="empty", version="1")
    output_path = tmp_path / "empty.jsonl"

    def fail_hard_link(*args: object, **kwargs: object) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(datasets_module.os, "link", fail_hard_link)

    assert export_dataset(database_path, "empty:1", output_path) == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["record_type"] == (
        "dataset"
    )


def test_failed_serialization_removes_partial_exclusive_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed no-overwrite export does not leave a partial destination."""
    database_path = tmp_path / "tracebench.sqlite3"
    store_trace(database_path)
    create_dataset(database_path, name="support", version="1")
    promote_trace(
        database_path,
        dataset_reference="support:1",
        trace_id="trace-001",
        mode=EvaluationMode.REFERENCE,
        reference_answer="Ottawa",
    )
    output_path = tmp_path / "partial.jsonl"

    def fail_case_encoding(dataset: object, case: object) -> str:
        raise ValueError("serialization failed")

    monkeypatch.setattr(
        datasets_module, "_encode_case_export_record", fail_case_encoding
    )

    with pytest.raises(ValueError, match="serialization failed"):
        export_dataset(database_path, "support:1", output_path)
    assert not output_path.exists()
