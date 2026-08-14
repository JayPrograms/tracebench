"""Tests for experiment preflight, identity, schema, and providers."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tracebench.datasets import create_dataset, promote_trace
from tracebench.experiment_config import (
    ExperimentPreflightError,
    PreparedExperiment,
    prepare_experiment,
)
from tracebench.experiment_models import RunRole
from tracebench.experiment_storage import insert_attempt
from tracebench.models import EvaluationMode, ScorerConfig, Trace
from tracebench.providers import FixtureProvider, ProviderError, ProviderRequest
from tracebench.storage import connect_database, insert_trace


def prepare_reference_experiment(
    tmp_path: Path,
    *,
    name: str = "reference-check",
    baseline_output: str = "Ottawa",
    candidate_output: str = "Ottawa",
    gate_yaml: str = "",
) -> tuple[Path, Path, PreparedExperiment]:
    """Create a one-case reference dataset and its experiment files."""
    database_path = tmp_path / "tracebench.sqlite3"
    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": "trace-001",
                        "timestamp": "2026-07-28T14:00:00Z",
                        "task_type": "question-answering",
                        "prompt": "What is the capital of Canada?",
                        "response": "Ottawa",
                    }
                ),
            )
    create_dataset(database_path, name="support", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="support:1",
        trace_id="trace-001",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_path.write_text(
        json.dumps({"eval_id": case.eval_id, "output": baseline_output}) + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps({"eval_id": case.eval_id, "output": candidate_output}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        (
            "schema_version: 1\n"
            f"name: {name}\n"
            "dataset: support:1\n"
            "baseline:\n"
            "  provider: fixture\n"
            "  path: baseline.jsonl\n"
            "candidate:\n"
            "  provider: fixture\n"
            "  path: candidate.jsonl\n"
            f"{gate_yaml}"
        ),
        encoding="utf-8",
    )
    return database_path, config_path, prepare_experiment(config_path, database_path)


def test_preflight_accepts_empty_fixture_output(tmp_path: Path) -> None:
    """An empty string is a real provider output, not a missing record."""
    _, _, prepared = prepare_reference_experiment(
        tmp_path,
        baseline_output="",
        candidate_output="",
    )

    request = ProviderRequest(request_id=prepared.cases[0].eval_id, prompt="")
    assert prepared.providers[RunRole.BASELINE].generate(request).output == ""
    assert prepared.providers[RunRole.CANDIDATE].generate(request).output == ""


def test_string_fixtures_repeat_while_tuple_sequences_are_consumed() -> None:
    """Ordinary A1 fixtures are reusable while judge retry sequences advance."""
    request = ProviderRequest(request_id="request", prompt="")
    repeating = FixtureProvider({"request": "stable"})
    sequenced = FixtureProvider({"request": ("first", "second")})

    assert repeating.generate(request).output == "stable"
    assert repeating.generate(request).output == "stable"
    assert [sequenced.generate(request).output for _ in range(2)] == [
        "first",
        "second",
    ]
    with pytest.raises(ProviderError, match="attempt 3"):
        sequenced.generate(request)


def test_configuration_hash_ignores_name_and_file_location(tmp_path: Path) -> None:
    """Behaviorally identical inputs share a deterministic configuration hash."""
    database_path, config_path, first = prepare_reference_experiment(tmp_path)
    copies = tmp_path / "copies"
    copies.mkdir()
    for filename in ("baseline-copy.jsonl", "candidate-copy.jsonl"):
        (copies / filename).write_text(
            (tmp_path / filename.replace("-copy", "")).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    second_path = copies / "second.yaml"
    second_path.write_text(
        """schema_version: 1
name: renamed-experiment
dataset: support:1
baseline:
  provider: fixture
  path: baseline-copy.jsonl
candidate:
  provider: fixture
  path: candidate-copy.jsonl
""",
        encoding="utf-8",
    )

    second = prepare_experiment(second_path, database_path)

    assert first.configuration_hash == second.configuration_hash
    assert config_path != second_path


def test_fixture_content_changes_configuration_hash(tmp_path: Path) -> None:
    """Changing a materialized provider output changes behavioral identity."""
    database_path, config_path, first = prepare_reference_experiment(tmp_path)
    candidate_path = tmp_path / "candidate.jsonl"
    record = json.loads(candidate_path.read_text(encoding="utf-8"))
    record["output"] = "Toronto"
    candidate_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    second = prepare_experiment(config_path, database_path)

    assert first.configuration_hash != second.configuration_hash


def test_duplicate_yaml_key_is_rejected_without_attempt(tmp_path: Path) -> None:
    """Ambiguous YAML fails before an experiment row can be persisted."""
    database_path, config_path, _ = prepare_reference_experiment(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "name: duplicate\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPreflightError, match="duplicate key"):
        prepare_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 0


def test_missing_fixture_ids_are_rejected_without_attempt(tmp_path: Path) -> None:
    """Both fixtures require exact evaluation-case coverage before persistence."""
    database_path, config_path, _ = prepare_reference_experiment(tmp_path)
    (tmp_path / "candidate.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ExperimentPreflightError, match="missing evaluation IDs"):
        prepare_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 0


def test_absent_mode_override_is_rejected(tmp_path: Path) -> None:
    """A mode-specific threshold cannot silently target an absent mode."""
    with pytest.raises(ExperimentPreflightError, match="modes present"):
        prepare_reference_experiment(
            tmp_path,
            gate_yaml=(
                "gate:\n  by_mode:\n    deterministic:\n      max_new_failures: 1\n"
            ),
        )


def test_invalid_deterministic_scorer_is_rejected_before_attempt(
    tmp_path: Path,
) -> None:
    """Stored generic scorer data is validated before experiment persistence."""
    database_path = tmp_path / "invalid-scorer.sqlite3"
    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_trace(
                connection,
                Trace.model_validate(
                    {
                        "trace_id": "trace-001",
                        "timestamp": "2026-07-28T14:00:00Z",
                        "task_type": "test",
                        "prompt": "Prompt",
                    }
                ),
            )
    create_dataset(database_path, name="deterministic", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="deterministic:1",
        trace_id="trace-001",
        mode=EvaluationMode.DETERMINISTIC,
        scorers=[ScorerConfig(name="regex", config={"pattern": "["})],
    )
    for filename in ("baseline.jsonl", "candidate.jsonl"):
        (tmp_path / filename).write_text(
            json.dumps({"eval_id": case.eval_id, "output": "anything"}) + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "invalid-scorer.yaml"
    config_path.write_text(
        """schema_version: 1
name: invalid-scorer
dataset: deterministic:1
baseline: {provider: fixture, path: baseline.jsonl}
candidate: {provider: fixture, path: candidate.jsonl}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPreflightError, match="invalid regular expression"):
        prepare_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 0


def test_repeated_name_and_hash_create_independent_attempts(tmp_path: Path) -> None:
    """Names and hashes are indexed metadata, never uniqueness constraints."""
    database_path, _, prepared = prepare_reference_experiment(tmp_path)
    timestamp = datetime(2026, 7, 29, 12, tzinfo=UTC)
    with closing(connect_database(database_path)) as connection:
        with connection:
            for suffix in ("one", "two"):
                insert_attempt(
                    connection,
                    experiment_id=f"experiment_{suffix}",
                    name=prepared.config.name,
                    dataset_id=prepared.dataset.dataset_id,
                    configuration_hash=prepared.configuration_hash,
                    configuration_json=prepared.configuration_json,
                    run_ids={
                        RunRole.BASELINE: f"run_baseline_{suffix}",
                        RunRole.CANDIDATE: f"run_candidate_{suffix}",
                    },
                    provider_snapshots=prepared.provider_snapshots,
                    timestamp=timestamp,
                )
        rows = connection.execute(
            """
            SELECT experiment_id, configuration_hash
            FROM experiments
            ORDER BY experiment_id
            """
        ).fetchall()

    assert [row["experiment_id"] for row in rows] == [
        "experiment_one",
        "experiment_two",
    ]
    assert {row["configuration_hash"] for row in rows} == {prepared.configuration_hash}


def test_database_distinguishes_completed_verdict_from_failed_status(
    tmp_path: Path,
) -> None:
    """A lifecycle failure cannot be stored with a regression verdict."""
    database_path, _, prepared = prepare_reference_experiment(tmp_path)
    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_attempt(
                connection,
                experiment_id="experiment_status",
                name=prepared.config.name,
                dataset_id=prepared.dataset.dataset_id,
                configuration_hash=prepared.configuration_hash,
                configuration_json=prepared.configuration_json,
                run_ids={
                    RunRole.BASELINE: "run_baseline_status",
                    RunRole.CANDIDATE: "run_candidate_status",
                },
                provider_snapshots=prepared.provider_snapshots,
                timestamp=datetime(2026, 7, 29, 12, tzinfo=UTC),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE experiments
                SET status = 'failed', verdict = 'FAIL',
                    failure_stage = 'baseline', failure_message = 'boom',
                    completed_at = '2026-07-29T12:01:00Z'
                WHERE experiment_id = 'experiment_status'
                """
            )


def test_database_accepts_empty_output_and_enforces_attempt_case_mode(
    tmp_path: Path,
) -> None:
    """Result rows permit empty text but cannot misstate dataset membership or mode."""
    database_path, _, prepared = prepare_reference_experiment(tmp_path)
    timestamp = datetime(2026, 7, 29, 12, tzinfo=UTC)
    with closing(connect_database(database_path)) as connection:
        with connection:
            insert_attempt(
                connection,
                experiment_id="experiment_relation",
                name=prepared.config.name,
                dataset_id=prepared.dataset.dataset_id,
                configuration_hash=prepared.configuration_hash,
                configuration_json=prepared.configuration_json,
                run_ids={
                    RunRole.BASELINE: "run_baseline_relation",
                    RunRole.CANDIDATE: "run_candidate_relation",
                },
                provider_snapshots=prepared.provider_snapshots,
                timestamp=timestamp,
            )
            connection.execute(
                """
                INSERT INTO experiment_case_results (
                    run_id, eval_id, evaluation_mode, output,
                    score, passed, created_at
                ) VALUES (?, ?, 'reference', '', 0.0, 0, ?)
                """,
                (
                    "run_baseline_relation",
                    prepared.cases[0].eval_id,
                    timestamp.isoformat(),
                ),
            )
        stored_output = connection.execute(
            "SELECT output FROM experiment_case_results"
        ).fetchone()[0]
        assert stored_output == ""
        with pytest.raises(sqlite3.IntegrityError, match="attempt dataset and mode"):
            connection.execute(
                """
                INSERT INTO experiment_case_results (
                    run_id, eval_id, evaluation_mode, output,
                    score, passed, created_at
                ) VALUES (?, ?, 'deterministic', 'value', 0.0, 0, ?)
                """,
                (
                    "run_candidate_relation",
                    prepared.cases[0].eval_id,
                    timestamp.isoformat(),
                ),
            )
