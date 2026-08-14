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
    create_dataset,
    get_dataset_details,
    promote_trace,
)
from tracebench.experiment_config import ExperimentPreflightError, prepare_experiment
from tracebench.experiment_models import (
    ComparisonAggregate,
    EffectiveThresholds,
    ExperimentStatus,
    ExperimentVerdict,
    SliceComparisonAggregate,
)
from tracebench.experiment_storage import load_experiment_report
from tracebench.experiments import (
    ExperimentOperationalError,
    apply_regression_gate,
    execute_experiment,
)
from tracebench.models import EvaluationMode, Priority, Trace
from tracebench.providers import OllamaProvider, ProviderError
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
    assert first.slice_source.sampling_schema_version == 2
    assert first.slice_source.sampling_algorithm == "balanced-preference-hash-v1"

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


def test_sampling_prefers_snapshotted_critical_and_completed_failure_signals(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "preference.sqlite3"
    _store_traces(database_path)
    create_dataset(database_path, name="history", version="1")
    critical = promote_trace(
        database_path,
        dataset_reference="history:1",
        trace_id="trace-0",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
        priority=Priority.CRITICAL,
    )
    failed = promote_trace(
        database_path,
        dataset_reference="history:1",
        trace_id="trace-1",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )
    records = [
        {"eval_id": critical.eval_id, "output": critical.reference_answer},
        {"eval_id": failed.eval_id, "output": "wrong"},
    ]
    for filename in ("history-baseline.jsonl", "history-candidate.jsonl"):
        (tmp_path / filename).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "history.yaml"
    config_path.write_text(
        """schema_version: 1
name: completed-history
dataset: history:1
baseline: {provider: fixture, path: history-baseline.jsonl}
candidate: {provider: fixture, path: history-candidate.jsonl}
gate: {max_score_drop: 0, max_new_failures: 0}
""",
        encoding="utf-8",
    )
    completed = execute_experiment(config_path, database_path)
    assert completed.status is ExperimentStatus.COMPLETED

    create_clustering_run(database_path, name="preference-slices", clusters=1)
    built = build_dataset_from_slices(
        database_path,
        name="preferred",
        version="1",
        clustering_run_name="preference-slices",
        size=2,
    )

    assert {case.source_trace_id for case in built.cases} == {"trace-0", "trace-1"}
    snapshots = {case.source_trace_id: case.slice_provenance for case in built.cases}
    critical_snapshot = snapshots["trace-0"]
    failed_snapshot = snapshots["trace-1"]
    assert critical_snapshot is not None
    assert failed_snapshot is not None
    assert critical_snapshot.critical_priority_signal is True
    assert critical_snapshot.prior_failure_signal is False
    assert critical_snapshot.preference_tier == 1
    assert failed_snapshot.critical_priority_signal is False
    assert failed_snapshot.prior_failure_signal is True
    assert failed_snapshot.preference_tier == 1

    _, loaded, source = get_dataset_details(database_path, "preferred:1")
    assert source is not None
    assert source.sampling_algorithm == "balanced-preference-hash-v1"
    assert {case.source_trace_id: case.slice_provenance for case in loaded} == snapshots


def test_sampling_handles_size_below_slice_count_and_capacity_redistribution(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "allocation-boundaries.sqlite3"
    _store_traces(database_path)
    create_clustering_run(database_path, name="discovery", clusters=3)

    below_slice_count = build_dataset_from_slices(
        database_path,
        name="small",
        version="1",
        clustering_run_name="discovery",
        size=2,
    )
    selected = [item.selected_count for item in below_slice_count.slices]
    assert sum(selected) == 2
    assert sorted(selected) == [0, 1, 1]

    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            "SELECT trace_id, cluster_number FROM trace_cluster_assignments "
            "WHERE clustering_run_id = ? ORDER BY trace_id",
            (below_slice_count.slice_source.clustering_run_id,),
        ).fetchall()
        target_cluster = int(rows[0]["cluster_number"])
        target_ids = [
            str(row["trace_id"])
            for row in rows
            if int(row["cluster_number"]) == target_cluster
        ]
        with connection:
            for trace_id in target_ids[1:]:
                connection.execute(
                    "UPDATE traces SET response = NULL WHERE trace_id = ?",
                    (trace_id,),
                )

    create_clustering_run(database_path, name="skewed", clusters=3)
    skewed = build_dataset_from_slices(
        database_path,
        name="redistributed",
        version="1",
        clustering_run_name="skewed",
        size=4,
    )
    by_cluster = {item.cluster_number: item for item in skewed.slices}
    assert by_cluster[target_cluster].eligible_count == 1
    assert by_cluster[target_cluster].selected_count == 1
    assert sum(item.selected_count for item in skewed.slices) == 4


def test_slice_selector_ambiguity_and_duplicate_resolution_fail_preflight(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "selector-boundaries.sqlite3"
    _store_traces(database_path, count=4)

    create_clustering_run(database_path, name="ambiguous", clusters=2)
    rename_slice(database_path, "ambiguous", 1, "cluster-0")
    build_dataset_from_slices(
        database_path,
        name="ambiguous-data",
        version="1",
        clustering_run_name="ambiguous",
        size=4,
    )
    ambiguous_path = tmp_path / "ambiguous.yaml"
    ambiguous_path.write_text(
        """schema_version: 1
name: ambiguous
dataset: ambiguous-data:1
baseline: {provider: fixture, path: unused-baseline.jsonl}
candidate: {provider: fixture, path: unused-candidate.jsonl}
gate:
  by_slice:
    cluster-0: {max_new_failures: 0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentPreflightError, match="ambiguous"):
        prepare_experiment(ambiguous_path, database_path)

    create_clustering_run(database_path, name="duplicate", clusters=2)
    rename_slice(database_path, "duplicate", 0, "refunds")
    build_dataset_from_slices(
        database_path,
        name="duplicate-data",
        version="1",
        clustering_run_name="duplicate",
        size=4,
    )
    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text(
        """schema_version: 1
name: duplicate
dataset: duplicate-data:1
baseline: {provider: fixture, path: unused-baseline.jsonl}
candidate: {provider: fixture, path: unused-candidate.jsonl}
gate:
  by_slice:
    cluster-0: {max_score_drop: 0}
    refunds: {max_new_failures: 0}
""",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentPreflightError, match="multiple slice overrides"):
        prepare_experiment(duplicate_path, database_path)


def test_slice_gate_passes_at_exact_threshold_equality() -> None:
    global_aggregate = ComparisonAggregate(
        scope="global",
        case_count=2,
        baseline_score=1.0,
        candidate_score=1.0,
        score_delta=0.0,
        newly_passed_count=0,
        newly_failed_count=0,
    )
    slice_aggregate = SliceComparisonAggregate(
        selector="cluster-0",
        cluster_number=0,
        label_snapshot=None,
        case_count=2,
        baseline_score=1.0,
        candidate_score=0.5,
        score_delta=-0.5,
        baseline_pass_rate=1.0,
        candidate_pass_rate=0.5,
        newly_passed_count=0,
        newly_failed_count=1,
        newly_passed=[],
        newly_failed=["eval-regressed"],
    )

    assert (
        apply_regression_gate(
            [global_aggregate],
            {"global": EffectiveThresholds(max_score_drop=0.0, max_new_failures=0)},
            slice_aggregates=[slice_aggregate],
            slice_thresholds={
                0: EffectiveThresholds(max_score_drop=0.5, max_new_failures=1)
            },
            slice_aware=True,
        )
        == []
    )


def test_slice_aware_operational_failure_reconstructs_schema_two_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database_path = tmp_path / "slice-operational.sqlite3"
    _store_traces(database_path, count=2)
    create_clustering_run(database_path, name="operational", clusters=1)
    built = build_dataset_from_slices(
        database_path,
        name="operational-data",
        version="1",
        clustering_run_name="operational",
        size=2,
    )
    (tmp_path / "baseline.jsonl").write_text(
        "\n".join(
            json.dumps({"eval_id": case.eval_id, "output": case.reference_answer})
            for case in built.cases
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "system.txt").write_text("Answer locally.", encoding="utf-8")
    config_path = tmp_path / "operational.yaml"
    config_path.write_text(
        """schema_version: 1
name: slice-operational
dataset: operational-data:1
baseline: {provider: fixture, path: baseline.jsonl}
candidate:
  provider: ollama
  base_url: http://localhost:11434
  model: local-test
  prompt_version: answer-v1
  system_prompt_file: system.txt
  temperature: 0
  timeout_seconds: 1
""",
        encoding="utf-8",
    )

    def fail_generation(*args: object, **kwargs: object) -> object:
        raise ProviderError("simulated local provider failure")

    monkeypatch.setattr(OllamaProvider, "generate", fail_generation)
    with pytest.raises(ExperimentOperationalError) as captured:
        execute_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        report = load_experiment_report(connection, captured.value.experiment_id)
    assert report.schema_version == 2
    assert report.status is ExperimentStatus.FAILED
    assert report.verdict is None
    assert report.dataset.slice_source is not None
    assert report.failure_stage == "candidate"


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
    assert "Sampling algorithm: balanced-preference-hash-v1" in success.stdout
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
