"""Checkpoint B2 slice-built dataset and regression-gate tests."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

import tracebench.storage as storage_module
from tracebench.cli import app
from tracebench.clustering import create_clustering_run, rename_slice
from tracebench.datasets import (
    DatasetBuildValidationError,
    build_dataset_from_slices,
    get_dataset_details,
    promote_trace,
)
from tracebench.experiment_config import ExperimentPreflightError, prepare_experiment
from tracebench.experiment_models import ExperimentVerdict
from tracebench.experiments import execute_experiment
from tracebench.models import EvaluationMode, Trace
from tracebench.storage import connect_database, insert_trace

runner = CliRunner()


def _store_traces(database_path: Path, count: int = 6) -> None:
    prompts = [
        "refund card payment",
        "refund duplicate charge",
        "password account login",
        "password reset access",
        "shipping package delay",
        "shipping tracking order",
    ]
    with closing(connect_database(database_path)) as connection, connection:
        for index, prompt in enumerate(prompts[:count]):
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": f"trace-{index}",
                        "timestamp": f"2026-08-13T12:00:0{index}Z",
                        "task_type": "support",
                        "prompt": prompt,
                        "response": f" answer-{index} ",
                        "metadata": {"priority": "must-not-be-used"},
                    }
                ),
            )


def test_slice_build_is_exact_balanced_reproducible_and_sealed(tmp_path: Path) -> None:
    database_path = tmp_path / "slice-build.sqlite3"
    _store_traces(database_path)
    clustered = create_clustering_run(
        database_path, name="support-slices-v1", clusters=3
    )
    rename_slice(database_path, "support-slices-v1", 0, "Refunds")

    first = build_dataset_from_slices(
        database_path,
        name="support-eval",
        version="0.2",
        clustering_run_name="support-slices-v1",
        size=4,
    )
    second = build_dataset_from_slices(
        database_path,
        name="support-eval",
        version="0.3",
        clustering_run_name="support-slices-v1",
        size=4,
    )

    assert len(first.cases) == 4
    assert {case.source_trace_id for case in first.cases} == {
        case.source_trace_id for case in second.cases
    }
    assert sum(item.selected_count for item in first.slices) == 4
    assert max(item.selected_count for item in first.slices) <= 2
    assert all(case.evaluation_mode is EvaluationMode.REFERENCE for case in first.cases)
    assert all(case.reference_answer == case.source_response for case in first.cases)
    assert all(case.priority.value == "medium" for case in first.cases)
    assert all(case.review_status.value == "draft" for case in first.cases)
    assert all(case.slice_provenance is not None for case in first.cases)
    assert first.slice_source.clustering_run_id == clustered.run.clustering_run_id

    rename_slice(database_path, "support-slices-v1", 0, "Renamed")
    _, loaded, source = get_dataset_details(database_path, "support-eval:0.2")
    assert source is not None
    cluster_zero = [
        case.slice_provenance
        for case in loaded
        if case.slice_provenance is not None
        and case.slice_provenance.cluster_number == 0
    ]
    assert all(item.label_snapshot == "Refunds" for item in cluster_zero)

    with pytest.raises(sqlite3.IntegrityError, match="sealed"):
        promote_trace(
            database_path,
            dataset_reference="support-eval:0.2",
            trace_id="trace-5",
            mode=EvaluationMode.REFERENCE,
            use_source_response=True,
        )


def test_slice_build_shortfall_rolls_back_every_dataset_row(tmp_path: Path) -> None:
    database_path = tmp_path / "shortfall.sqlite3"
    _store_traces(database_path, count=3)
    create_clustering_run(database_path, name="slices", clusters=2)

    with pytest.raises(DatasetBuildValidationError, match="eligible trace count 3"):
        build_dataset_from_slices(
            database_path,
            name="too-large",
            version="1",
            clustering_run_name="slices",
            size=4,
        )

    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM eval_datasets WHERE name = 'too-large'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM eval_case_slice_provenance"
            ).fetchone()[0]
            == 0
        )


def test_slice_aggregation_label_gate_and_schema_two_report(tmp_path: Path) -> None:
    database_path = tmp_path / "slice-gate.sqlite3"
    _store_traces(database_path, count=4)
    create_clustering_run(database_path, name="slices", clusters=2)
    rename_slice(database_path, "slices", 0, "refunds")
    rename_slice(database_path, "slices", 1, "accounts")
    built = build_dataset_from_slices(
        database_path,
        name="support",
        version="2",
        clustering_run_name="slices",
        size=4,
    )
    regressed = next(
        case
        for case in built.cases
        if case.slice_provenance is not None
        and case.slice_provenance.cluster_number == 0
    )
    for filename, candidate in (
        ("baseline.jsonl", False),
        ("candidate.jsonl", True),
    ):
        records = [
            {
                "eval_id": case.eval_id,
                "output": (
                    "wrong"
                    if candidate and case.eval_id == regressed.eval_id
                    else case.reference_answer
                ),
            }
            for case in built.cases
        ]
        (tmp_path / filename).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """schema_version: 1
name: slice-check
dataset: support:2
baseline: {provider: fixture, path: baseline.jsonl}
candidate: {provider: fixture, path: candidate.jsonl}
gate:
  max_score_drop: 1
  max_new_failures: 4
  by_slice:
    refunds:
      max_score_drop: 0
      max_new_failures: 0
""",
        encoding="utf-8",
    )

    rename_slice(database_path, "slices", 0, "renamed-after-build")
    report = execute_experiment(config_path, database_path)

    assert report.schema_version == 2
    assert report.verdict is ExperimentVerdict.FAIL
    assert report.dataset.slice_source is not None
    assert report.comparison is not None
    assert report.gate is not None
    assert set(report.comparison.by_slice) == {"cluster-0", "cluster-1"}
    refund_comparison = report.comparison.by_slice["cluster-0"]
    assert refund_comparison.label_snapshot == "refunds"
    assert refund_comparison.newly_failed == [regressed.eval_id]
    assert [item.scope for item in report.gate.violations] == [
        "cluster-0",
        "cluster-0",
    ]
    assert [item.metric.value for item in report.gate.violations] == [
        "score_drop",
        "new_failures",
    ]
    payload = report.model_dump(mode="json", by_alias=True)
    assert payload["schema_version"] == 2
    assert "by_slice" in payload["runs"]["baseline"]


def test_missing_slice_gate_is_preflight_error_without_attempt(tmp_path: Path) -> None:
    database_path = tmp_path / "missing-slice.sqlite3"
    _store_traces(database_path, count=3)
    create_clustering_run(database_path, name="slices", clusters=2)
    built = build_dataset_from_slices(
        database_path,
        name="support",
        version="2",
        clustering_run_name="slices",
        size=2,
    )
    for filename in ("baseline.jsonl", "candidate.jsonl"):
        (tmp_path / filename).write_text(
            "\n".join(
                json.dumps({"eval_id": case.eval_id, "output": case.reference_answer})
                for case in built.cases
            )
            + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "missing.yaml"
    config_path.write_text(
        """schema_version: 1
name: missing
dataset: support:2
baseline: {provider: fixture, path: baseline.jsonl}
candidate: {provider: fixture, path: candidate.jsonl}
gate:
  by_slice:
    absent-label: {max_new_failures: 0}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPreflightError, match="not represented"):
        prepare_experiment(config_path, database_path)
    with closing(connect_database(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0


def test_dataset_build_cli_reports_sampling_and_exact_shortfall(tmp_path: Path) -> None:
    database_path = tmp_path / "cli.sqlite3"
    _store_traces(database_path, count=3)
    create_clustering_run(database_path, name="cli-slices", clusters=2)
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}

    success = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--name",
            "cli-eval",
            "--version",
            "1",
            "--from-slices",
            "cli-slices",
            "--size",
            "2",
        ],
        env=environment,
    )

    assert success.exit_code == 0, success.output
    assert "Created dataset cli-eval:1" in success.stdout
    assert "Sampling algorithm: balanced-hash-v1" in success.stdout
    assert "Eligible traces: 3" in success.stdout
    assert "Cases created: 2" in success.stdout
    assert "cluster-0" in success.stdout

    shortfall = runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--name",
            "cli-shortfall",
            "--version",
            "1",
            "--from-slices",
            "cli-slices",
            "--size",
            "4",
        ],
        env=environment,
    )
    assert shortfall.exit_code == 2
    assert "eligible trace count 3" in shortfall.stderr
    assert "no dataset was persisted" in shortfall.stderr


def test_pre_b2_database_migrates_in_place_and_partial_b2_is_rejected(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pre-b2.sqlite3"
    with closing(connect_database(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        for statement in storage_module.B2_TRIGGER_STATEMENTS:
            trigger_name = statement.split("CREATE TRIGGER IF NOT EXISTS ", 1)[
                1
            ].splitlines()[0]
            connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
        connection.execute("DROP INDEX idx_eval_case_slice_dataset_cluster")
        for table_name in (
            "experiment_comparison_slice_aggregates",
            "experiment_run_slice_aggregates",
            "eval_case_slice_provenance",
            "eval_dataset_slice_builds",
        ):
            connection.execute(f'DROP TABLE "{table_name}"')
        connection.execute("DROP TABLE experiment_gate_violations")
        connection.execute(
            """
            CREATE TABLE experiment_gate_violations (
                experiment_id TEXT NOT NULL
                    REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                violation_index INTEGER NOT NULL CHECK (violation_index >= 0),
                scope TEXT NOT NULL
                    CHECK (
                        scope IN (
                            'global', 'deterministic', 'reference', 'rubric'
                        )
                    ),
                metric TEXT NOT NULL
                    CHECK (metric IN ('score_drop', 'new_failures')),
                actual REAL NOT NULL CHECK (actual >= 0.0),
                allowed REAL NOT NULL CHECK (allowed >= 0.0),
                message TEXT NOT NULL CHECK (length(trim(message)) > 0),
                PRIMARY KEY (experiment_id, violation_index)
            )
            """
        )
        connection.commit()

    with closing(connect_database(database_path)) as connection:
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(experiment_gate_violations)"
            ).fetchall()
        }
        assert {"scope_kind", "cluster_number", "label_snapshot"}.issubset(columns)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'eval_dataset_slice_builds'"
            ).fetchone()[0]
            == 1
        )

    partial_path = tmp_path / "partial-b2.sqlite3"
    with closing(sqlite3.connect(partial_path)) as connection:
        connection.execute("CREATE TABLE eval_dataset_slice_builds (id INTEGER)")
    with pytest.raises(sqlite3.DatabaseError, match="partial or incompatible"):
        connect_database(partial_path)
