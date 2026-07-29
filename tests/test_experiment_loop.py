"""Integration tests for lifecycle, persistence, reporting, and the CLI."""

import json
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import tracebench.experiments as experiment_module
from tracebench.cli import app
from tracebench.datasets import create_dataset, promote_trace
from tracebench.experiment_models import ExperimentStatus, ExperimentVerdict
from tracebench.experiments import ExperimentOperationalError, execute_experiment
from tracebench.models import EvaluationMode, ScorerConfig, Trace
from tracebench.storage import connect_database, insert_trace

runner = CliRunner()


def build_mixed_experiment(
    tmp_path: Path,
    *,
    candidate_regresses: bool,
) -> tuple[Path, Path, dict[str, str]]:
    """Create a deterministic/reference dataset and complete fixture config."""
    database_path = tmp_path / "experiment.sqlite3"
    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": "deterministic",
                        "timestamp": "2026-07-28T14:00:00Z",
                        "task_type": "classification",
                        "prompt": "Return yes.",
                    }
                ),
            )
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": "reference",
                        "timestamp": "2026-07-28T14:01:00Z",
                        "task_type": "question-answering",
                        "prompt": "What is the capital of Canada?",
                        "response": "Ottawa",
                    }
                ),
            )
    create_dataset(database_path, name="mixed", version="1")
    deterministic = promote_trace(
        database_path,
        dataset_reference="mixed:1",
        trace_id="deterministic",
        mode=EvaluationMode.DETERMINISTIC,
        scorers=[ScorerConfig(name="exact_match", config={"expected": "yes"})],
    )
    reference = promote_trace(
        database_path,
        dataset_reference="mixed:1",
        trace_id="reference",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )
    case_ids = {
        "deterministic": deterministic.eval_id,
        "reference": reference.eval_id,
    }
    baseline = {
        deterministic.eval_id: "yes",
        reference.eval_id: "wrong",
    }
    candidate = {
        deterministic.eval_id: "no" if candidate_regresses else "yes",
        reference.eval_id: "Ottawa",
    }
    for filename, outputs in (
        ("baseline.jsonl", baseline),
        ("candidate.jsonl", candidate),
    ):
        (tmp_path / filename).write_text(
            "".join(
                json.dumps({"eval_id": eval_id, "output": output}) + "\n"
                for eval_id, output in reversed(list(outputs.items()))
            ),
            encoding="utf-8",
        )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """schema_version: 1
name: mixed-regression
dataset: mixed:1
baseline:
  provider: fixture
  path: baseline.jsonl
candidate:
  provider: fixture
  path: candidate.jsonl
gate:
  max_score_drop: 0
  max_new_failures: 0
  by_mode:
    deterministic:
      max_score_drop: 0
""",
        encoding="utf-8",
    )
    return database_path, config_path, case_ids


def test_complete_attempt_persists_normalized_fail_report(tmp_path: Path) -> None:
    """A regression is completed with FAIL and reconstructible normalized data."""
    database_path, config_path, case_ids = build_mixed_experiment(
        tmp_path,
        candidate_regresses=True,
    )

    report = execute_experiment(config_path, database_path)

    assert report.status is ExperimentStatus.COMPLETED
    assert report.verdict is ExperimentVerdict.FAIL
    assert report.comparison is not None
    assert report.gate is not None
    assert report.comparison.newly_passed == [case_ids["reference"]]
    assert report.comparison.newly_failed == [case_ids["deterministic"]]
    assert report.comparison.global_.score_delta == 0.0
    assert set(report.comparison.by_mode) == {"deterministic", "reference"}
    assert len(report.gate.violations) == 3
    with closing(connect_database(database_path)) as connection:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "experiment_case_results",
                "experiment_scorer_results",
                "experiment_run_aggregates",
                "experiment_case_comparisons",
                "experiment_comparison_aggregates",
                "experiment_gate_violations",
            )
        }
    assert counts == {
        "experiment_case_results": 4,
        "experiment_scorer_results": 4,
        "experiment_run_aggregates": 6,
        "experiment_case_comparisons": 2,
        "experiment_comparison_aggregates": 3,
        "experiment_gate_violations": 3,
    }


def test_rerun_has_new_attempt_id_and_same_configuration_hash(tmp_path: Path) -> None:
    """Every accepted invocation is independent while behavioral identity is stable."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=False,
    )

    first = execute_experiment(config_path, database_path)
    second = execute_experiment(config_path, database_path)

    assert first.verdict is ExperimentVerdict.PASS
    assert second.verdict is ExperimentVerdict.PASS
    assert first.experiment_id != second.experiment_id
    assert first.configuration_hash == second.configuration_hash


def test_candidate_stage_rollback_preserves_baseline_and_records_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed candidate transaction leaves no partial candidate result rows."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=False,
    )
    original = experiment_module.persist_completed_run
    calls = 0

    def fail_after_candidate_inserts(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("injected candidate persistence failure")

    monkeypatch.setattr(
        experiment_module,
        "persist_completed_run",
        fail_after_candidate_inserts,
    )

    with pytest.raises(ExperimentOperationalError) as caught:
        execute_experiment(config_path, database_path)

    assert caught.value.stage == "candidate"
    experiment_id = caught.value.experiment_id
    with closing(connect_database(database_path)) as connection:
        attempt = connection.execute(
            "SELECT status, verdict, failure_stage FROM experiments"
        ).fetchone()
        runs = connection.execute(
            """
            SELECT run_id, role, status
            FROM experiment_runs
            WHERE experiment_id = ?
            ORDER BY role
            """,
            (experiment_id,),
        ).fetchall()
        result_counts = {
            row["role"]: connection.execute(
                "SELECT COUNT(*) FROM experiment_case_results WHERE run_id = ?",
                (row["run_id"],),
            ).fetchone()[0]
            for row in runs
        }
        comparison_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_case_comparisons"
        ).fetchone()[0]

    assert dict(attempt) == {
        "status": "failed",
        "verdict": None,
        "failure_stage": "candidate",
    }
    assert {row["role"]: row["status"] for row in runs} == {
        "baseline": "completed",
        "candidate": "failed",
    }
    assert result_counts == {"baseline": 2, "candidate": 0}
    assert comparison_count == 0


def test_comparison_stage_rolls_back_all_gate_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison, aggregate, violation, and verdict writes share one boundary."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=True,
    )
    original = experiment_module.persist_completed_comparison

    def fail_after_comparison_inserts(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        raise RuntimeError("injected comparison persistence failure")

    monkeypatch.setattr(
        experiment_module,
        "persist_completed_comparison",
        fail_after_comparison_inserts,
    )

    with pytest.raises(ExperimentOperationalError) as caught:
        execute_experiment(config_path, database_path)

    assert caught.value.stage == "comparison"
    with closing(connect_database(database_path)) as connection:
        attempt = connection.execute(
            "SELECT status, verdict, failure_stage FROM experiments"
        ).fetchone()
        run_statuses = [
            row["status"]
            for row in connection.execute(
                "SELECT status FROM experiment_runs ORDER BY role"
            ).fetchall()
        ]
        comparison_counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "experiment_case_comparisons",
                "experiment_comparison_aggregates",
                "experiment_gate_violations",
            )
        ]

    assert dict(attempt) == {
        "status": "failed",
        "verdict": None,
        "failure_stage": "comparison",
    }
    assert run_statuses == ["completed", "completed"]
    assert comparison_counts == [0, 0, 0]


@pytest.mark.parametrize(
    ("candidate_regresses", "exit_code", "verdict"),
    [(False, 0, "PASS"), (True, 1, "FAIL")],
)
def test_cli_json_distinguishes_pass_and_regression_failure(
    tmp_path: Path,
    candidate_regresses: bool,
    exit_code: int,
    verdict: str,
) -> None:
    """Completed JSON reports retain their verdict even when the gate exits nonzero."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=candidate_regresses,
    )

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path), "--json"],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == exit_code, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["verdict"] == verdict
    assert payload["comparison"]["global"]["case_count"] == 2
    assert set(payload["comparison"]["by_mode"]) == {
        "deterministic",
        "reference",
    }


def test_cli_preflight_error_is_exit_two_and_persists_no_attempt(
    tmp_path: Path,
) -> None:
    """Invalid fixture coverage is a preflight error rather than a failed attempt."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=False,
    )
    (tmp_path / "candidate.jsonl").write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path)],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert "no experiment attempt was persisted" in result.stderr
    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 0


def test_cli_operational_failure_is_exit_three_with_null_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational failure has a distinct exit and never masquerades as regression."""
    database_path, config_path, _ = build_mixed_experiment(
        tmp_path,
        candidate_regresses=False,
    )

    def fail_run(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected operational failure")

    monkeypatch.setattr(experiment_module, "persist_completed_run", fail_run)

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path), "--json"],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 3, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert payload["verdict"] is None
    assert payload["failure_stage"] == "baseline"


def test_checked_in_sample_runs_complete_readme_workflow(tmp_path: Path) -> None:
    """The documented trace-to-gate sample executes successfully end to end."""
    repository = Path(__file__).parents[1]
    datasets = repository / "datasets"
    database_path = tmp_path / "readme.sqlite3"
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}

    ingest = runner.invoke(
        app,
        ["ingest", str(datasets / "traces.sample.jsonl")],
        env=environment,
    )
    create = runner.invoke(
        app,
        [
            "dataset",
            "create",
            "--name",
            "core-eval",
            "--version",
            "0.1",
            "--description",
            "Core evaluation-loop sample",
        ],
        env=environment,
    )
    add_commands = [
        [
            "dataset",
            "add-trace",
            "--dataset",
            "core-eval:0.1",
            "--trace-id",
            "sample-001",
            "--mode",
            "deterministic",
            "--scorer-file",
            str(datasets / "scorer.exact-match.json"),
            "--scorer-file",
            str(datasets / "scorer.contains.json"),
        ],
        [
            "dataset",
            "add-trace",
            "--dataset",
            "core-eval:0.1",
            "--trace-id",
            "sample-002",
            "--mode",
            "deterministic",
            "--scorer-file",
            str(datasets / "scorer.regex.json"),
        ],
        [
            "dataset",
            "add-trace",
            "--dataset",
            "core-eval:0.1",
            "--trace-id",
            "sample-003",
            "--mode",
            "deterministic",
            "--scorer-file",
            str(datasets / "scorer.json-validity.json"),
            "--scorer-file",
            str(datasets / "scorer.required-keys.json"),
        ],
    ]
    additions = [
        runner.invoke(app, command, env=environment) for command in add_commands
    ]

    run = runner.invoke(
        app,
        [
            "experiment",
            "run",
            str(datasets / "core-evaluation.sample.yaml"),
            "--json",
        ],
        env=environment,
    )

    assert ingest.exit_code == 0, ingest.output
    assert create.exit_code == 0, create.output
    assert all(result.exit_code == 0 for result in additions), [
        result.output for result in additions
    ]
    assert run.exit_code == 0, run.output
    payload = json.loads(run.stdout)
    assert payload["verdict"] == "PASS"
    assert payload["runs"]["baseline"]["global"]["score"] == pytest.approx(5 / 6)
    assert payload["runs"]["candidate"]["global"]["score"] == 1.0
    assert payload["comparison"]["newly_passed"] == [
        "eval_c72dcd43c7cb5ad39cc447e26af78a85"
    ]
    assert payload["comparison"]["newly_failed"] == []
