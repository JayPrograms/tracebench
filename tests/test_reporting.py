"""Tests for the shared persisted experiment detail export."""

import json
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

import tracebench.experiments as experiment_module
from test_customer_support_demo import (
    CI_PASS_CONFIG,
    FIXTURE_CONFIG,
    _build_demo,
)
from test_experiment_loop import build_mixed_experiment
from tracebench.cli import app
from tracebench.experiment_models import ExperimentStatus, ExperimentVerdict
from tracebench.experiments import ExperimentOperationalError, execute_experiment
from tracebench.reporting import (
    ExperimentDetail,
    ExperimentDetailError,
    export_experiment_detail,
    get_experiment_detail,
)
from tracebench.storage import connect_database

runner = CliRunner()


def _run_failing_demo(database_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    environment = _build_demo(database_path)
    first = runner.invoke(
        app,
        ["experiment", "run", str(FIXTURE_CONFIG), "--json"],
        env=environment,
    )
    second = runner.invoke(
        app,
        ["experiment", "run", str(FIXTURE_CONFIG), "--json"],
        env=environment,
    )
    assert first.exit_code == second.exit_code == 1
    return json.loads(first.stdout), json.loads(second.stdout)


def test_customer_support_detail_joins_authoritative_report_and_observations(
    tmp_path: Path,
) -> None:
    """A failed slice-aware demo retains stable cases, judges, and cache state."""
    first_report, second_report = _run_failing_demo(tmp_path / "northstar.sqlite3")
    first_detail = get_experiment_detail(
        tmp_path / "northstar.sqlite3", str(first_report["experiment_id"])
    )
    first_rubric = [
        case for case in first_detail.cases if case.evaluation_mode.value == "rubric"
    ]
    assert all(
        case.candidate is not None
        and case.candidate.judge is not None
        and case.candidate.judge.cache_hit is False
        for case in first_rubric
    )
    detail = get_experiment_detail(
        tmp_path / "northstar.sqlite3", str(second_report["experiment_id"])
    )

    assert detail.schema_version == 1
    assert detail.status is ExperimentStatus.COMPLETED
    assert detail.verdict is ExperimentVerdict.FAIL
    assert detail.dataset.sealed is True
    assert detail.dataset.case_count == 21
    assert detail.report.model_dump(mode="json", by_alias=True) == second_report
    assert len(detail.cases) == 21
    assert [case.eval_id for case in detail.cases] == sorted(
        case.eval_id for case in detail.cases
    )
    assert {case.evaluation_mode.value for case in detail.cases} == {
        "reference",
        "deterministic",
        "rubric",
    }
    assert all(case.slice_provenance is not None for case in detail.cases)
    assert {
        case.slice_provenance.selector
        for case in detail.cases
        if case.slice_provenance is not None
    } == {
        "cluster-0",
        "cluster-1",
        "cluster-2",
        "cluster-3",
        "cluster-4",
        "cluster-5",
        "cluster-6",
    }
    assert all(
        sum(
            case.slice_provenance is not None
            and case.slice_provenance.selector == selector
            for case in detail.cases
        )
        == 3
        for selector in {f"cluster-{index}" for index in range(7)}
    )
    assert all(
        case.baseline is not None and case.candidate is not None
        for case in detail.cases
    )
    assert all(
        case.baseline.generation.latency_ms >= 0
        for case in detail.cases
        if case.baseline
    )
    assert any(
        case.transition.value == "newly_failed"
        for case in detail.cases
        if case.transition is not None
    )
    rubric_cases = [
        case for case in detail.cases if case.evaluation_mode.value == "rubric"
    ]
    assert len(rubric_cases) == 7
    assert all(
        case.candidate is not None and case.candidate.judge is not None
        for case in rubric_cases
    )
    assert all(
        case.candidate.judge.cache_hit is True
        for case in rubric_cases
        if case.candidate and case.candidate.judge
    )
    assert set(detail.provider_snapshots) == {"baseline", "candidate"}
    assert detail.judge_snapshot is not None
    assert detail.gate.thresholds.max_new_failures == 2
    assert detail.gate.passed is False
    assert len(detail.gate.violations) == len(second_report["gate"]["violations"])
    exported = json.dumps(detail.model_dump(mode="json"), ensure_ascii=False)
    assert "raw_output" not in exported
    assert "validation_error" not in exported
    assert "You are Northstar Shop's customer-support assistant." not in exported
    assert "You are a strict evaluation judge." not in exported
    assert first_report["experiment_id"] != second_report["experiment_id"]


def test_customer_support_passing_detail_has_pass_gate_and_two_new_passes(
    tmp_path: Path,
) -> None:
    """The CI-safe fixture exports the same detail schema with a PASS gate."""
    database_path = tmp_path / "northstar-pass.sqlite3"
    environment = _build_demo(database_path)
    result = runner.invoke(
        app,
        ["experiment", "run", str(CI_PASS_CONFIG), "--json"],
        env=environment,
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    detail = get_experiment_detail(database_path, report["experiment_id"])
    assert detail.verdict is ExperimentVerdict.PASS
    assert detail.gate.passed is True
    assert detail.gate.violations == []
    assert detail.report.comparison is not None
    assert detail.report.comparison.newly_failed == []
    assert len(detail.report.comparison.newly_passed) == 2


def test_detail_prevents_non_reference_answer_leakage_and_supports_export(
    tmp_path: Path,
) -> None:
    """Deterministic/rubric cases omit answers while the export round-trips."""
    report, _ = _run_failing_demo(tmp_path / "northstar.sqlite3")
    output_path = tmp_path / "detail.json"
    export_experiment_detail(
        tmp_path / "northstar.sqlite3",
        str(report["experiment_id"]),
        output_path,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validated = ExperimentDetail.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert validated.experiment_id == report["experiment_id"]
    assert "global" in payload["report"]["runs"]["baseline"]
    assert "global_" not in payload["report"]["runs"]["baseline"]
    assert "NaN" not in output_path.read_text(encoding="utf-8")
    assert "Infinity" not in output_path.read_text(encoding="utf-8")
    assert all(
        "reference_answer" not in case and "response" not in case["source_trace"]
        for case in payload["cases"]
        if case["evaluation_mode"] != "reference"
    )
    assert all(
        "reference_answer" in case
        for case in payload["cases"]
        if case["evaluation_mode"] == "reference"
    )

    cli_output = tmp_path / "cli-detail.json"
    result = runner.invoke(
        app,
        [
            "experiment",
            "export",
            str(report["experiment_id"]),
            "--output",
            str(cli_output),
        ],
        env={"TRACEBENCH_DB_PATH": str(tmp_path / "northstar.sqlite3")},
    )
    assert result.exit_code == 0, result.output
    assert ExperimentDetail.model_validate_json(cli_output.read_text(encoding="utf-8"))


def test_ordinary_and_operational_details_are_reconstructible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary schema-1 and partially persisted failed attempts export safely."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path / "ordinary", candidate_regresses=False
    )
    report = execute_experiment(config_path, database_path)
    ordinary = get_experiment_detail(database_path, report.experiment_id)
    assert ordinary.report.schema_version == 1
    assert ordinary.dataset.sealed is False
    assert ordinary.dataset.slice_source is None
    assert len(ordinary.cases) == 2

    operational_path = tmp_path / "operational.sqlite3"
    operational_config_db, operational_config, _ = build_mixed_experiment(
        tmp_path / "operational", candidate_regresses=False
    )

    def fail_candidate(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected detail failure")

    monkeypatch.setattr(experiment_module, "persist_completed_run", fail_candidate)
    with pytest.raises(ExperimentOperationalError) as caught:
        execute_experiment(operational_config, operational_config_db)
    partial = get_experiment_detail(operational_config_db, caught.value.experiment_id)
    assert partial.status is ExperimentStatus.FAILED
    assert partial.verdict is None
    assert partial.failure_stage == "baseline"
    assert any(run.status.value == "failed" for run in partial.runs)
    assert any(
        case.baseline is None and case.candidate is None for case in partial.cases
    )
    partial_export = tmp_path / "operational-detail.json"
    exported = export_experiment_detail(
        operational_config_db,
        caught.value.experiment_id,
        partial_export,
    )
    assert exported.status is ExperimentStatus.FAILED
    assert (
        ExperimentDetail.model_validate_json(
            partial_export.read_text(encoding="utf-8")
        ).failure_stage
        == "baseline"
    )
    assert not operational_path.exists()


def test_export_preflight_overwrite_and_atomic_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid IDs, protected files, and replacement failures are non-mutating."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path / "export", candidate_regresses=False
    )
    report = execute_experiment(config_path, database_path)
    output_path = tmp_path / "export.json"
    output_path.write_text("keep", encoding="utf-8")
    with pytest.raises(ExperimentDetailError, match="already exists"):
        export_experiment_detail(database_path, report.experiment_id, output_path)
    assert output_path.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ExperimentDetailError, match="was not found"):
        get_experiment_detail(database_path, "missing-experiment")
    invalid_id = runner.invoke(
        app,
        ["experiment", "export", "missing-experiment", "--output", str(output_path)],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )
    assert invalid_id.exit_code == 2
    protected = runner.invoke(
        app,
        [
            "experiment",
            "export",
            report.experiment_id,
            "--output",
            str(output_path),
        ],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )
    assert protected.exit_code == 2

    import tracebench.reporting as reporting_module

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(reporting_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        export_experiment_detail(
            database_path, report.experiment_id, output_path, overwrite=True
        )
    assert output_path.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".export.json.*.tmp"))


def test_export_does_not_change_database(tmp_path: Path) -> None:
    """Export is read-only, including for its connection lifecycle."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path / "readonly", candidate_regresses=False
    )
    report = execute_experiment(config_path, database_path)
    before = database_path.read_bytes()
    export_experiment_detail(database_path, report.experiment_id, tmp_path / "out.json")
    after = database_path.read_bytes()
    assert after == before
    with closing(connect_database(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
