"""Tests for A2/A3 rubric judging, caching, review, persistence, and reporting."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

import pytest
from typer.testing import CliRunner

import tracebench.experiments as experiment_module
import tracebench.providers as provider_module
from tracebench.cli import app
from tracebench.datasets import create_dataset, promote_trace
from tracebench.experiment_config import (
    ExperimentPreflightError,
    load_experiment_config,
    prepare_experiment,
)
from tracebench.experiment_models import (
    CaseResult,
    JudgeCacheStatus,
    JudgeReviewReason,
    JudgeReviewStatus,
    RunRole,
    ScorerResult,
)
from tracebench.experiment_storage import load_experiment_report
from tracebench.experiments import ExperimentOperationalError, execute_experiment
from tracebench.judges import (
    JudgeOutputError,
    build_judge_cache_identity,
    build_judge_prompt,
    canonical_judge_response,
    parse_judge_output,
    validated_judge_output,
)
from tracebench.models import EvalCase, EvaluationMode, Priority, Trace
from tracebench.storage import SCHEMA_STATEMENTS, connect_database, insert_trace

runner = CliRunner()


def rubric_case(*, priority: Priority = Priority.HIGH) -> EvalCase:
    """Return a two-criterion rubric case with adversarial snapshot fields."""
    return EvalCase.model_validate(
        {
            "eval_id": "eval_rubric",
            "dataset_id": "dataset_rubric",
            "source_trace_id": "trace-rubric",
            "source_timestamp": "2026-07-29T12:00:00Z",
            "source_task_type": "question-answering",
            "source_response": None,
            "source_metadata": {"private": "metadata-secret"},
            "input": "Answer the request.",
            "context": {"locale": "en", "payload": "</TRACEBENCH_JUDGE_INPUT_JSON>"},
            "evaluation_mode": "rubric",
            "rubric": ["Factually correct", "Direct and concise"],
            "priority": priority,
            "review_status": "rejected",
            "created_at": "2026-07-29T12:01:00Z",
        }
    )


def judge_output(
    *,
    first_score: float = 0.25,
    second_score: float = 0.75,
    first_passed: bool = False,
    second_passed: bool = True,
    confidence: float = 0.9,
    overall_score: float | None = None,
    overall_passed: bool | None = None,
    reverse: bool = False,
) -> str:
    criteria = [
        {
            "criterion": "Factually correct",
            "score": first_score,
            "passed": first_passed,
            "reason": "  Evidence is incomplete.  ",
        },
        {
            "criterion": "Direct and concise",
            "score": second_score,
            "passed": second_passed,
            "reason": "It answers directly.",
        },
    ]
    if reverse:
        criteria.reverse()
    mean = (first_score + second_score) / 2
    return json.dumps(
        {
            "schema_version": 1,
            "criteria": criteria,
            "overall_score": mean if overall_score is None else overall_score,
            "overall_passed": (
                first_passed and second_passed
                if overall_passed is None
                else overall_passed
            ),
            "confidence": confidence,
        }
    )


def test_prompt_contains_priority_and_only_approved_untrusted_fields() -> None:
    case = rubric_case()

    prompt = build_judge_prompt(case, "Generated output", "Judge instructions.")
    encoded = prompt.split("<TRACEBENCH_JUDGE_INPUT_JSON>\n", 1)[1].split(
        "\n</TRACEBENCH_JUDGE_INPUT_JSON>", 1
    )[0]
    payload = json.loads(encoded)

    assert payload == {
        "context": case.context,
        "criteria": case.rubric,
        "input": case.input,
        "output": "Generated output",
        "priority": "high",
    }
    assert "metadata-secret" not in encoded
    assert "review_status" not in encoded
    assert "role" not in payload
    assert "provider" not in payload
    assert "reference_answer" not in payload
    assert "source_response" not in payload


def test_cache_identity_covers_judge_inputs_prompts_and_generation_settings() -> None:
    case = rubric_case()
    provider_identity: dict[str, object] = {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "judge-model:v1",
        "temperature": 0.0,
        "seed": 7,
    }

    def identity(
        candidate: EvalCase = case,
        output: str = "Generated output",
        prompt_version: str = "rubric-v1",
        prompt_hash: str = "a" * 64,
        retry_prompt_hash: str = "b" * 64,
        provider: dict[str, object] | None = None,
    ) -> str:
        return build_judge_cache_identity(
            candidate,
            output,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            retry_prompt_hash=retry_prompt_hash,
            provider_identity=provider_identity if provider is None else provider,
        ).cache_key

    original = identity()
    variants = [
        identity(output="Different output"),
        identity(case.model_copy(update={"rubric": list(reversed(case.rubric))})),
        identity(case.model_copy(update={"input": "Different input"})),
        identity(case.model_copy(update={"context": {"locale": "fr"}})),
        identity(case.model_copy(update={"priority": Priority.CRITICAL})),
        identity(prompt_version="rubric-v2"),
        identity(prompt_hash="c" * 64),
        identity(retry_prompt_hash="d" * 64),
        identity(provider={**provider_identity, "model": "judge-model:v2"}),
        identity(provider={**provider_identity, "temperature": 0.5}),
        identity(provider={**provider_identity, "seed": 8}),
        identity(
            provider={
                **provider_identity,
                "base_url": "http://localhost:22434",
            }
        ),
    ]

    assert all(candidate != original for candidate in variants)
    assert identity() == original


def test_canonical_response_hashes_parsed_json_not_exact_raw_output() -> None:
    case = rubric_case()
    raw_output = "\n  " + judge_output() + "  \n"

    validated = validated_judge_output(case, raw_output)

    assert validated.raw_output == raw_output
    assert validated.response_json == canonical_judge_response(validated.response)
    assert (
        validated.response_hash
        == hashlib.sha256(validated.response_json.encode("utf-8")).hexdigest()
    )
    assert (
        validated.response_hash
        != hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    )


def test_continuous_results_preserve_judge_fields_and_existing_scorers() -> None:
    result = parse_judge_output(
        rubric_case(),
        judge_output(reverse=True, confidence=0.4),
        evaluated_output="Generated output",
        confidence_threshold=0.7,
        attempt_count=2,
    )

    assert result.output == "Generated output"
    assert result.score == 0.5
    assert result.passed is False
    assert [scorer.name for scorer in result.scorers] == ["rubric", "rubric"]
    assert [scorer.score for scorer in result.scorers] == [0.25, 0.75]
    assert [scorer.passed for scorer in result.scorers] == [False, True]
    assert result.scorers[0].details == {
        "criterion": "Factually correct",
        "reason": "Evidence is incomplete.",
    }
    assert result.judge is not None
    assert result.judge.model_dump() == {
        "response_schema_version": 1,
        "attempt_count": 2,
        "overall_score": 0.5,
        "overall_passed": False,
        "confidence": 0.4,
        "confidence_threshold": 0.7,
        "below_confidence_threshold": True,
        "cache_hit": None,
        "cache": {
            "key": None,
            "status": JudgeCacheStatus.NOT_RECORDED,
        },
        "review": {
            "status": JudgeReviewStatus.NEEDS_REVIEW,
            "reasons": [JudgeReviewReason.LOW_CONFIDENCE],
        },
    }


@pytest.mark.parametrize(
    (
        "priority",
        "confidence",
        "passed",
        "expected_status",
        "expected_reasons",
    ),
    [
        (
            Priority.HIGH,
            0.7,
            False,
            JudgeReviewStatus.NOT_REQUIRED,
            [],
        ),
        (
            Priority.HIGH,
            0.69,
            False,
            JudgeReviewStatus.NEEDS_REVIEW,
            [JudgeReviewReason.LOW_CONFIDENCE],
        ),
        (
            Priority.CRITICAL,
            0.9,
            False,
            JudgeReviewStatus.NEEDS_REVIEW,
            [JudgeReviewReason.CRITICAL_FAILURE],
        ),
        (
            Priority.CRITICAL,
            0.4,
            False,
            JudgeReviewStatus.NEEDS_REVIEW,
            [
                JudgeReviewReason.LOW_CONFIDENCE,
                JudgeReviewReason.CRITICAL_FAILURE,
            ],
        ),
        (
            Priority.CRITICAL,
            0.9,
            True,
            JudgeReviewStatus.NOT_REQUIRED,
            [],
        ),
    ],
)
def test_review_state_derives_from_confidence_and_critical_failure(
    priority: Priority,
    confidence: float,
    passed: bool,
    expected_status: JudgeReviewStatus,
    expected_reasons: list[JudgeReviewReason],
) -> None:
    raw_output = (
        judge_output(
            first_score=1.0,
            second_score=1.0,
            first_passed=True,
            second_passed=True,
            confidence=confidence,
        )
        if passed
        else judge_output(confidence=confidence)
    )

    result = parse_judge_output(
        rubric_case(priority=priority),
        raw_output,
        evaluated_output="output",
        confidence_threshold=0.7,
        attempt_count=1,
    )

    assert result.judge is not None
    assert result.judge.review.status is expected_status
    assert result.judge.review.reasons == expected_reasons


@pytest.mark.parametrize(
    ("overall_score", "accepted"),
    [(0.5000005, True), (0.500002, False)],
)
def test_overall_score_uses_fixed_absolute_tolerance(
    overall_score: float,
    accepted: bool,
) -> None:
    raw = judge_output(overall_score=overall_score)
    if accepted:
        result = parse_judge_output(
            rubric_case(),
            raw,
            evaluated_output="output",
            confidence_threshold=0.5,
            attempt_count=1,
        )
        assert result.score == 0.5
        assert result.judge is not None
        assert result.judge.overall_score == overall_score
    else:
        with pytest.raises(JudgeOutputError) as caught:
            parse_judge_output(
                rubric_case(),
                raw,
                evaluated_output="output",
                confidence_threshold=0.5,
                attempt_count=1,
            )
        assert caught.value.code == "overall_score_mismatch"


def test_overall_pass_must_match_all_criterion_passes() -> None:
    with pytest.raises(JudgeOutputError) as caught:
        parse_judge_output(
            rubric_case(),
            judge_output(overall_passed=True),
            evaluated_output="output",
            confidence_threshold=0.5,
            attempt_count=1,
        )
    assert caught.value.code == "overall_pass_mismatch"


@pytest.mark.parametrize(
    ("criteria", "message"),
    [
        (
            [
                {
                    "criterion": "Factually correct",
                    "score": 1.0,
                    "passed": True,
                    "reason": "ok",
                }
            ],
            "missing criteria",
        ),
        (
            [
                {
                    "criterion": "Factually correct ",
                    "score": 1.0,
                    "passed": True,
                    "reason": "ok",
                },
                {
                    "criterion": "Direct and concise",
                    "score": 1.0,
                    "passed": True,
                    "reason": "ok",
                },
            ],
            "unknown criteria",
        ),
        (
            [
                {
                    "criterion": "Factually correct",
                    "score": 1.0,
                    "passed": True,
                    "reason": "ok",
                },
                {
                    "criterion": "Factually correct",
                    "score": 1.0,
                    "passed": True,
                    "reason": "again",
                },
            ],
            "duplicate criteria",
        ),
    ],
)
def test_criterion_text_requires_exact_unique_coverage(
    criteria: list[dict[str, object]],
    message: str,
) -> None:
    raw = json.dumps(
        {
            "schema_version": 1,
            "criteria": criteria,
            "overall_score": 1.0,
            "overall_passed": True,
            "confidence": 0.9,
        }
    )
    with pytest.raises(JudgeOutputError, match=message):
        parse_judge_output(
            rubric_case(),
            raw,
            evaluated_output="output",
            confidence_threshold=0.5,
            attempt_count=1,
        )


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1,"criteria":[],"overall_score":1.0,"overall_passed":true,"confidence":0.9}',
        judge_output(confidence=1.1),
        judge_output(confidence=-0.1),
        judge_output().replace('"confidence": 0.9', '"confidence": true'),
        judge_output().replace('"score": 0.25', '"score": "0.25"'),
        judge_output().replace('"score": 0.25', '"score": NaN'),
        judge_output().replace('"confidence": 0.9', '"confidence": 1e999'),
        judge_output().replace(
            '"reason": "  Evidence is incomplete.  "', '"reason": "  "'
        ),
        judge_output()[:-1] + ', "extra": 1}',
        "```json\n" + judge_output() + "\n```",
    ],
)
def test_structured_response_rejects_non_strict_values(raw: str) -> None:
    with pytest.raises(JudgeOutputError):
        parse_judge_output(
            rubric_case(),
            raw,
            evaluated_output="output",
            confidence_threshold=0.5,
            attempt_count=1,
        )


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_judge_schema_version_requires_an_integer_token(
    schema_version: object,
) -> None:
    payload = json.loads(judge_output())
    payload["schema_version"] = schema_version

    with pytest.raises(JudgeOutputError) as caught:
        parse_judge_output(
            rubric_case(),
            json.dumps(payload),
            evaluated_output="output",
            confidence_threshold=0.5,
            attempt_count=1,
        )

    assert caught.value.code == "schema_validation"


def build_rubric_experiment(
    tmp_path: Path,
    *,
    baseline_judge_outputs: list[str],
    candidate_judge_outputs: list[str],
    confidence_threshold: object = 0.7,
    include_judge: bool = True,
    priority: Priority = Priority.HIGH,
) -> tuple[Path, Path, EvalCase]:
    database_path = tmp_path / "rubric.sqlite3"
    with closing(connect_database(database_path)) as connection, connection:
        insert_trace(
            connection,
            Trace.model_validate(
                {
                    "trace_id": "trace-rubric",
                    "timestamp": "2026-07-29T12:00:00Z",
                    "task_type": "support",
                    "prompt": "Answer the customer.",
                    "context": {"locale": "en"},
                }
            ),
        )
    create_dataset(database_path, name="rubric", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="rubric:1",
        trace_id="trace-rubric",
        mode=EvaluationMode.RUBRIC,
        rubric=["Factually correct", "Direct and concise"],
        priority=priority,
    )
    for filename, output in (
        ("baseline.jsonl", "baseline answer"),
        ("candidate.jsonl", "candidate answer"),
    ):
        (tmp_path / filename).write_text(
            json.dumps({"eval_id": case.eval_id, "output": output}) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "judge.txt").write_text("Judge prompt.", encoding="utf-8")
    (tmp_path / "retry.txt").write_text("Retry prompt.", encoding="utf-8")
    (tmp_path / "judge.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "eval_id": case.eval_id,
                    "role": role,
                    "outputs": outputs,
                }
            )
            + "\n"
            for role, outputs in (
                ("baseline", baseline_judge_outputs),
                ("candidate", candidate_judge_outputs),
            )
        ),
        encoding="utf-8",
    )
    judge_yaml = ""
    if include_judge:
        judge_yaml = (
            "judge:\n"
            "  provider: fixture\n"
            "  path: judge.jsonl\n"
            "  prompt_version: rubric-v1\n"
            "  prompt_file: judge.txt\n"
            "  retry_prompt_file: retry.txt\n"
            f"  confidence_threshold: {json.dumps(confidence_threshold)}\n"
        )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        (
            "schema_version: 1\n"
            "name: rubric-check\n"
            "dataset: rubric:1\n"
            "baseline: {provider: fixture, path: baseline.jsonl}\n"
            "candidate: {provider: fixture, path: candidate.jsonl}\n"
            f"{judge_yaml}"
            "gate:\n"
            "  max_score_drop: 1\n"
            "  max_new_failures: 1\n"
        ),
        encoding="utf-8",
    )
    return database_path, config_path, case


def test_fixture_judge_runs_scores_persists_and_reports_additively(
    tmp_path: Path,
) -> None:
    database_path, config_path, case = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output(confidence=0.4)],
        candidate_judge_outputs=[
            judge_output(
                first_score=0.8,
                second_score=1.0,
                first_passed=True,
                second_passed=True,
                confidence=0.7,
            )
        ],
    )

    prepared = prepare_experiment(config_path, database_path)
    assert '"priority":"high"' in prepared.configuration_json
    assert '"rubric":["Factually correct","Direct and concise"]' in (
        prepared.configuration_json
    )
    assert "judge.jsonl" not in prepared.configuration_json
    report = execute_experiment(config_path, database_path)

    baseline = report.runs[RunRole.BASELINE].cases[0]
    candidate = report.runs[RunRole.CANDIDATE].cases[0]
    assert baseline.eval_id == case.eval_id
    assert baseline.score == 0.5
    assert candidate.score == 0.9
    assert report.runs[RunRole.BASELINE].by_mode["rubric"].score == 0.5
    assert report.comparison is not None
    assert report.comparison.by_mode["rubric"].score_delta == 0.4
    public = report.model_dump(mode="json", by_alias=True)
    public_case = public["runs"]["baseline"]["cases"][0]
    assert public["schema_version"] == 1
    assert len(public_case["scorers"]) == 2
    assert public_case["judge"]["confidence"] == 0.4
    assert public_case["judge"]["below_confidence_threshold"] is True
    assert public_case["judge"]["cache_hit"] is False
    assert public_case["judge"]["cache"]["status"] == "miss"
    assert public_case["judge"]["review"] == {
        "status": "needs_review",
        "reasons": ["low_confidence"],
    }
    assert "raw_output" not in json.dumps(public)

    with closing(connect_database(database_path)) as connection:
        judge_config = connection.execute(
            "SELECT confidence_threshold, provider_config_json FROM experiment_judges"
        ).fetchone()
        attempts = connection.execute(
            "SELECT parse_status, raw_output FROM experiment_judge_attempts "
            "ORDER BY run_id"
        ).fetchall()
        scorer_rows = connection.execute(
            "SELECT scorer_name, score, details_json "
            "FROM experiment_scorer_results ORDER BY run_id, scorer_index"
        ).fetchall()
        judge_results = connection.execute(
            "SELECT overall_score, overall_passed, confidence, "
            "below_confidence_threshold FROM experiment_judge_results "
            "ORDER BY run_id"
        ).fetchall()
    assert judge_config["confidence_threshold"] == 0.7
    assert (
        json.loads(judge_config["provider_config_json"])["confidence_threshold"] == 0.7
    )
    assert len(attempts) == 2
    assert all(row["parse_status"] == "parsed" for row in attempts)
    assert len(scorer_rows) == 4
    assert all(row["scorer_name"] == "rubric" for row in scorer_rows)
    assert json.loads(scorer_rows[0]["details_json"])["criterion"] in case.rubric
    assert len(judge_results) == 2


def test_repeated_identical_judging_uses_immutable_cached_raw_result(
    tmp_path: Path,
) -> None:
    baseline_raw = "\n " + judge_output(confidence=0.8) + "\n"
    candidate_raw = judge_output(
        first_score=1.0,
        second_score=1.0,
        first_passed=True,
        second_passed=True,
        confidence=0.9,
    )
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[baseline_raw],
        candidate_judge_outputs=[candidate_raw],
    )

    first = execute_experiment(config_path, database_path)
    second = execute_experiment(config_path, database_path)

    for role in RunRole:
        first_judge = first.runs[role].cases[0].judge
        second_judge = second.runs[role].cases[0].judge
        assert first_judge is not None
        assert second_judge is not None
        assert first_judge.cache_hit is False
        assert first_judge.cache.status is JudgeCacheStatus.MISS
        assert first_judge.attempt_count == 1
        assert second_judge.cache_hit is True
        assert second_judge.cache.status is JudgeCacheStatus.HIT
        assert second_judge.attempt_count == 0
        assert second_judge.cache.key == first_judge.cache.key

    with closing(connect_database(database_path)) as connection:
        cache_rows = connection.execute(
            """
            SELECT
                raw_output, response_json, response_hash
            FROM judge_result_cache
            ORDER BY raw_output
            """
        ).fetchall()
        lookup_statuses = [
            row["cache_status"]
            for row in connection.execute(
                """
                SELECT lookup.cache_status
                FROM experiment_judge_cache_lookups AS lookup
                JOIN experiment_runs AS run ON run.run_id = lookup.run_id
                JOIN experiments AS experiment
                    ON experiment.experiment_id = run.experiment_id
                ORDER BY experiment.created_at, run.role
                """
            ).fetchall()
        ]
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_judge_attempts"
        ).fetchone()[0]

    assert {row["raw_output"] for row in cache_rows} == {
        baseline_raw,
        candidate_raw,
    }
    assert all(
        row["response_hash"]
        == hashlib.sha256(row["response_json"].encode("utf-8")).hexdigest()
        for row in cache_rows
    )
    assert lookup_statuses == ["miss", "miss", "hit", "hit"]
    assert attempt_count == 2


def test_cache_lookup_transaction_closes_before_waiting_for_judge_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
    )
    original_lookup = experiment_module._lookup_and_record_judge_cache
    original_generate = provider_module.FixtureProvider.generate
    active_connection: sqlite3.Connection | None = None
    provider_transaction_states: list[bool] = []

    def recording_lookup(
        *,
        connection: sqlite3.Connection,
        run_id: str,
        case: EvalCase,
        cache_key: str,
    ) -> object:
        nonlocal active_connection
        entry = original_lookup(
            connection=connection,
            run_id=run_id,
            case=case,
            cache_key=cache_key,
        )
        active_connection = connection
        assert not connection.in_transaction
        return entry

    def observing_generate(
        provider: provider_module.FixtureProvider,
        request: provider_module.ProviderRequest,
    ) -> provider_module.ProviderResponse:
        if request.response_format == "json":
            assert active_connection is not None
            provider_transaction_states.append(active_connection.in_transaction)
        return original_generate(provider, request)

    monkeypatch.setattr(
        experiment_module,
        "_lookup_and_record_judge_cache",
        recording_lookup,
    )
    monkeypatch.setattr(
        provider_module.FixtureProvider,
        "generate",
        observing_generate,
    )

    execute_experiment(config_path, database_path)

    assert provider_transaction_states == [False, False]


def test_cache_entries_reject_updates_and_deletes(tmp_path: Path) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
    )
    report = execute_experiment(config_path, database_path)
    judge = report.runs[RunRole.BASELINE].cases[0].judge
    assert judge is not None
    assert judge.cache.key is not None

    with closing(connect_database(database_path)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(
                    "UPDATE judge_result_cache SET raw_output = ? WHERE cache_key = ?",
                    ("changed", judge.cache.key),
                )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(
                    "DELETE FROM judge_result_cache WHERE cache_key = ?",
                    (judge.cache.key,),
                )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO judge_result_cache (
                        cache_key, key_version, identity_json,
                        response_schema_version, response_json,
                        response_hash, raw_output, created_at
                    )
                    SELECT
                        cache_key, key_version, identity_json,
                        response_schema_version, response_json,
                        response_hash, raw_output, created_at
                    FROM judge_result_cache
                    WHERE cache_key = ?
                    """,
                    (judge.cache.key,),
                )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            with connection:
                connection.execute(
                    """
                    UPDATE experiment_judge_cache_lookups
                    SET cache_status = 'hit'
                    WHERE run_id = ?
                    """,
                    (report.runs[RunRole.BASELINE].run_id,),
                )


@pytest.mark.parametrize(
    ("corruption", "error_code"),
    [
        ("identity", "identity_mismatch"),
        ("raw_output", "raw_output_invalid"),
        ("raw_mismatch", "raw_output_response_mismatch"),
        ("response_json", "response_json_invalid"),
        ("response_hash", "response_hash_mismatch"),
    ],
)
def test_cache_hit_corruption_is_operational_without_provider_fallback(
    tmp_path: Path,
    corruption: str,
    error_code: str,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
    )
    first = execute_experiment(config_path, database_path)
    judge = first.runs[RunRole.BASELINE].cases[0].judge
    assert judge is not None
    assert judge.cache.key is not None

    with closing(connect_database(database_path)) as connection, connection:
        connection.execute("DROP TRIGGER prevent_judge_cache_update")
        if corruption == "identity":
            connection.execute(
                "UPDATE judge_result_cache SET identity_json = ? WHERE cache_key = ?",
                ('{"key_version":1}', judge.cache.key),
            )
        elif corruption == "raw_output":
            connection.execute(
                "UPDATE judge_result_cache SET raw_output = 'not json' "
                "WHERE cache_key = ?",
                (judge.cache.key,),
            )
        elif corruption == "raw_mismatch":
            connection.execute(
                "UPDATE judge_result_cache SET raw_output = ? WHERE cache_key = ?",
                (
                    judge_output(
                        first_score=1.0,
                        second_score=1.0,
                        first_passed=True,
                        second_passed=True,
                    ),
                    judge.cache.key,
                ),
            )
        elif corruption == "response_json":
            malformed_response = '{"schema_version":1}'
            connection.execute(
                """
                UPDATE judge_result_cache
                SET response_json = ?, response_hash = ?
                WHERE cache_key = ?
                """,
                (
                    malformed_response,
                    hashlib.sha256(malformed_response.encode("utf-8")).hexdigest(),
                    judge.cache.key,
                ),
            )
        else:
            connection.execute(
                "UPDATE judge_result_cache SET response_hash = ? WHERE cache_key = ?",
                ("0" * 64, judge.cache.key),
            )

    with pytest.raises(ExperimentOperationalError, match=error_code) as caught:
        execute_experiment(config_path, database_path)

    assert caught.value.stage == "baseline"
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM experiment_judge_attempts"
            ).fetchone()[0]
            == 2
        )
        statuses = [
            row["cache_status"]
            for row in connection.execute(
                "SELECT cache_status FROM experiment_judge_cache_lookups "
                "ORDER BY created_at"
            ).fetchall()
        ]
    assert statuses == ["miss", "miss", "hit"]


def test_concurrent_miss_loser_uses_own_response_while_first_cache_entry_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losing_raw = judge_output(confidence=0.9)
    winning_raw = judge_output(
        first_score=1.0,
        second_score=1.0,
        first_passed=True,
        second_passed=True,
        confidence=0.95,
    )
    database_path, config_path, case = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[losing_raw],
        candidate_judge_outputs=[judge_output(confidence=0.8)],
    )
    original_insert = experiment_module.insert_judge_cache_entry
    original_validate = experiment_module.validate_cached_judge_output
    injected_winner = False
    validated_cache_raws: list[str] = []

    def insert_after_concurrent_winner(
        connection: sqlite3.Connection,
        *,
        cache_key: str,
        key_version: int,
        identity_json: str,
        response_schema_version: int,
        response_json: str,
        response_hash: str,
        raw_output: str,
        timestamp: datetime,
    ) -> bool:
        nonlocal injected_winner
        if not injected_winner:
            injected_winner = True
            winner = validated_judge_output(case, winning_raw)
            assert original_insert(
                connection,
                cache_key=cache_key,
                key_version=key_version,
                identity_json=identity_json,
                response_schema_version=response_schema_version,
                response_json=winner.response_json,
                response_hash=winner.response_hash,
                raw_output=winner.raw_output,
                timestamp=timestamp,
            )
        return original_insert(
            connection,
            cache_key=cache_key,
            key_version=key_version,
            identity_json=identity_json,
            response_schema_version=response_schema_version,
            response_json=response_json,
            response_hash=response_hash,
            raw_output=raw_output,
            timestamp=timestamp,
        )

    def record_cache_validation(*args: object, **kwargs: object) -> object:
        raw_output = kwargs["raw_output"]
        assert isinstance(raw_output, str)
        validated_cache_raws.append(raw_output)
        return original_validate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        experiment_module,
        "insert_judge_cache_entry",
        insert_after_concurrent_winner,
    )
    monkeypatch.setattr(
        experiment_module,
        "validate_cached_judge_output",
        record_cache_validation,
    )

    losing_run = execute_experiment(config_path, database_path)
    assert validated_cache_raws[0] == winning_raw
    later_hit = execute_experiment(config_path, database_path)

    losing_result = losing_run.runs[RunRole.BASELINE].cases[0]
    hit_result = later_hit.runs[RunRole.BASELINE].cases[0]
    assert losing_result.score == 0.5
    assert losing_result.judge is not None
    assert losing_result.judge.cache_hit is False
    assert losing_result.judge.attempt_count == 1
    assert hit_result.score == 1.0
    assert hit_result.judge is not None
    assert hit_result.judge.cache_hit is True
    assert hit_result.judge.attempt_count == 0

    cache_key = losing_result.judge.cache.key
    with closing(connect_database(database_path)) as connection:
        cache_raw = connection.execute(
            "SELECT raw_output FROM judge_result_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()[0]
        attempt_raw = connection.execute(
            """
            SELECT attempt.raw_output
            FROM experiment_judge_attempts AS attempt
            JOIN experiment_runs AS run ON run.run_id = attempt.run_id
            WHERE run.role = 'baseline'
            ORDER BY attempt.created_at
            LIMIT 1
            """
        ).fetchone()[0]
    assert cache_raw == winning_raw
    assert attempt_raw == losing_raw


def test_malformed_then_valid_retries_once_and_persists_both_attempts(
    tmp_path: Path,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=["not json", judge_output(confidence=0.6)],
        candidate_judge_outputs=[judge_output(confidence=0.8)],
    )

    report = execute_experiment(config_path, database_path)

    baseline = report.runs[RunRole.BASELINE].cases[0]
    assert baseline.judge is not None
    assert baseline.judge.attempt_count == 2
    with closing(connect_database(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT run.role, attempt.attempt_number, attempt.parse_status,
                   attempt.validation_error, attempt.request_hash
            FROM experiment_judge_attempts AS attempt
            JOIN experiment_runs AS run ON run.run_id = attempt.run_id
            ORDER BY run.role, attempt.attempt_number
            """
        ).fetchall()
    assert [
        (row["role"], row["attempt_number"], row["parse_status"]) for row in rows
    ] == [
        ("baseline", 1, "malformed"),
        ("baseline", 2, "parsed"),
        ("candidate", 1, "parsed"),
    ]
    assert rows[0]["validation_error"].startswith("invalid_json:")
    assert all(len(row["request_hash"]) == 64 for row in rows)
    with closing(connect_database(database_path)) as connection:
        cached_raw_outputs = {
            row["raw_output"]
            for row in connection.execute(
                "SELECT raw_output FROM judge_result_cache"
            ).fetchall()
        }
    assert "not json" not in cached_raw_outputs
    assert judge_output(confidence=0.6) in cached_raw_outputs


def test_two_malformed_outputs_fail_but_preserve_audit_rows(tmp_path: Path) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=["bad one", "bad two"],
        candidate_judge_outputs=[judge_output()],
    )

    with pytest.raises(ExperimentOperationalError, match="after 2 attempts") as caught:
        execute_experiment(config_path, database_path)

    assert caught.value.stage == "baseline"
    with closing(connect_database(database_path)) as connection:
        attempts = connection.execute(
            "SELECT attempt_number, parse_status, raw_output "
            "FROM experiment_judge_attempts ORDER BY attempt_number"
        ).fetchall()
        result_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_case_results"
        ).fetchone()[0]
        scorer_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_scorer_results"
        ).fetchone()[0]
        judge_result_count = connection.execute(
            "SELECT COUNT(*) FROM experiment_judge_results"
        ).fetchone()[0]
        cache_count = connection.execute(
            "SELECT COUNT(*) FROM judge_result_cache"
        ).fetchone()[0]
        lookup_status = connection.execute(
            "SELECT cache_status FROM experiment_judge_cache_lookups"
        ).fetchone()[0]
    assert [tuple(row) for row in attempts] == [
        (1, "malformed", "bad one"),
        (2, "malformed", "bad two"),
    ]
    assert (result_count, scorer_count, judge_result_count) == (0, 0, 0)
    assert cache_count == 0
    assert lookup_status == "miss"


def test_detailed_malformed_diagnostics_remain_database_only(tmp_path: Path) -> None:
    secret = "JUDGE_SUPPLIED_SECRET"
    malformed = judge_output().replace("Factually correct", secret)
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[malformed, malformed],
        candidate_judge_outputs=[judge_output()],
    )

    with pytest.raises(ExperimentOperationalError) as caught:
        execute_experiment(config_path, database_path)
    assert caught.value.message.endswith("criterion_coverage")
    assert secret not in caught.value.message
    assert "missing criteria" not in caught.value.message

    human = runner.invoke(
        app,
        ["experiment", "run", str(config_path)],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )
    assert human.exit_code == 3
    assert "criterion_coverage" in human.output
    assert secret not in human.output
    assert "missing criteria" not in human.output

    machine = runner.invoke(
        app,
        ["experiment", "run", str(config_path), "--json"],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )
    assert machine.exit_code == 3
    payload = json.loads(machine.stdout)
    assert payload["schema_version"] == 1
    assert payload["failure_message"].endswith("criterion_coverage")
    assert secret not in machine.stdout
    assert "missing criteria" not in machine.stdout

    with closing(connect_database(database_path)) as connection:
        diagnostics = [
            row["validation_error"]
            for row in connection.execute(
                "SELECT validation_error FROM experiment_judge_attempts "
                "ORDER BY created_at, attempt_number"
            ).fetchall()
        ]
        failure_messages = [
            row["failure_message"]
            for row in connection.execute(
                "SELECT failure_message FROM experiments ORDER BY created_at"
            ).fetchall()
        ]
    assert len(diagnostics) == 6
    assert all(secret in diagnostic for diagnostic in diagnostics)
    assert all(message.endswith("criterion_coverage") for message in failure_messages)
    assert all(secret not in message for message in failure_messages)


def test_judge_transport_failure_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  provider: fixture\n  path: judge.jsonl\n",
            "  provider: ollama\n"
            "  base_url: http://localhost:11434\n"
            "  model: judge-model\n"
            "  temperature: 0\n"
            "  timeout_seconds: 5\n"
            "  seed: 7\n",
        ),
        encoding="utf-8",
    )
    calls = 0

    def fail_connection(request: object, timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise URLError("offline")

    monkeypatch.setattr(provider_module, "urlopen", fail_connection)

    with pytest.raises(ExperimentOperationalError, match="judge request"):
        execute_experiment(config_path, database_path)

    assert calls == 1
    with closing(connect_database(database_path)) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM experiment_judge_attempts"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM judge_result_cache").fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT cache_status FROM experiment_judge_cache_lookups"
            ).fetchone()[0]
            == "miss"
        )


def test_missing_judge_is_preflight_without_attempt(tmp_path: Path) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
        include_judge=False,
    )

    with pytest.raises(ExperimentPreflightError, match="requires a configured judge"):
        prepare_experiment(config_path, database_path)
    with closing(connect_database(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 0


@pytest.mark.parametrize("value", ["true", "'0.5'", ".nan", "-0.1", "1.1"])
def test_confidence_threshold_is_strict_finite_unit_float(
    tmp_path: Path,
    value: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        f"""schema_version: 1
name: invalid-threshold
dataset: rubric:1
baseline: {{provider: fixture, path: baseline.jsonl}}
candidate: {{provider: fixture, path: candidate.jsonl}}
judge:
  provider: fixture
  path: judge.jsonl
  prompt_version: rubric-v1
  prompt_file: judge.txt
  retry_prompt_file: retry.txt
  confidence_threshold: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentPreflightError):
        load_experiment_config(config_path)


def test_threshold_and_prompt_content_change_configuration_hash(tmp_path: Path) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output()],
        candidate_judge_outputs=[judge_output()],
    )
    first = prepare_experiment(config_path, database_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("confidence_threshold: 0.7", "confidence_threshold: 0.6"),
        encoding="utf-8",
    )
    threshold_changed = prepare_experiment(config_path, database_path)
    (tmp_path / "judge.txt").write_text("Changed judge prompt.", encoding="utf-8")
    prompt_changed = prepare_experiment(config_path, database_path)

    assert first.configuration_hash != threshold_changed.configuration_hash
    assert threshold_changed.configuration_hash != prompt_changed.configuration_hash


def test_threshold_change_reuses_cache_and_rederives_review_state(
    tmp_path: Path,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output(confidence=0.6)],
        candidate_judge_outputs=[judge_output(confidence=0.8)],
    )

    first = execute_experiment(config_path, database_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "confidence_threshold: 0.7",
            "confidence_threshold: 0.5",
        ),
        encoding="utf-8",
    )
    second = execute_experiment(config_path, database_path)

    first_judge = first.runs[RunRole.BASELINE].cases[0].judge
    second_judge = second.runs[RunRole.BASELINE].cases[0].judge
    assert first_judge is not None
    assert second_judge is not None
    assert first.configuration_hash != second.configuration_hash
    assert first_judge.cache_hit is False
    assert second_judge.cache_hit is True
    assert second_judge.cache.key == first_judge.cache.key
    assert first_judge.review.status is JudgeReviewStatus.NEEDS_REVIEW
    assert second_judge.review.status is JudgeReviewStatus.NOT_REQUIRED


def test_reviewed_status_round_trips_without_an_editing_interface(
    tmp_path: Path,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output(confidence=0.4)],
        candidate_judge_outputs=[judge_output(confidence=0.8)],
    )
    report = execute_experiment(config_path, database_path)
    baseline_run_id = report.runs[RunRole.BASELINE].run_id
    eval_id = report.runs[RunRole.BASELINE].cases[0].eval_id

    with closing(connect_database(database_path)) as connection, connection:
        connection.execute(
            """
            UPDATE experiment_judge_results
            SET review_status = 'reviewed'
            WHERE run_id = ? AND eval_id = ?
            """,
            (baseline_run_id, eval_id),
        )
        reloaded = load_experiment_report(connection, report.experiment_id)

    judge = reloaded.runs[RunRole.BASELINE].cases[0].judge
    assert judge is not None
    assert judge.review.status is JudgeReviewStatus.REVIEWED
    assert judge.review.reasons == [JudgeReviewReason.LOW_CONFIDENCE]


def test_non_rubric_case_serialization_omits_judge_field() -> None:
    result = CaseResult(
        eval_id="eval_reference",
        evaluation_mode=EvaluationMode.REFERENCE,
        output="Ottawa",
        score=1.0,
        passed=True,
        scorers=[ScorerResult(name="exact_match", score=1.0, passed=True)],
    )

    assert result.model_dump(mode="json") == {
        "eval_id": "eval_reference",
        "evaluation_mode": "reference",
        "output": "Ottawa",
        "score": 1.0,
        "passed": True,
        "scorers": [
            {
                "name": "exact_match",
                "score": 1.0,
                "passed": True,
                "details": {},
            }
        ],
    }


def test_checked_in_primary_prompt_exists_and_is_nonblank() -> None:
    prompt = Path(__file__).parents[1] / "prompts" / "judges" / "rubric-v1.txt"
    assert prompt.read_text(encoding="utf-8").strip()


def test_a2_judge_results_migrate_with_unknown_cache_and_derived_review(
    tmp_path: Path,
) -> None:
    database_path, config_path, _ = build_rubric_experiment(
        tmp_path,
        baseline_judge_outputs=[judge_output(confidence=0.9)],
        candidate_judge_outputs=[judge_output(confidence=0.9)],
        priority=Priority.CRITICAL,
    )
    before = execute_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        for trigger in (
            "validate_judge_result_relation_insert",
            "validate_judge_result_relation_update",
            "validate_judge_cache_lookup_relation_insert",
            "validate_judge_cache_lookup_relation_update",
            "prevent_judge_cache_update",
            "prevent_judge_cache_delete",
        ):
            connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        connection.execute(
            """
            CREATE TABLE experiment_judge_results_a2 (
                run_id TEXT NOT NULL,
                eval_id TEXT NOT NULL,
                final_attempt_number INTEGER NOT NULL
                    CHECK (final_attempt_number IN (1, 2)),
                response_schema_version INTEGER NOT NULL
                    CHECK (response_schema_version = 1),
                overall_score REAL NOT NULL
                    CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
                overall_passed INTEGER NOT NULL
                    CHECK (overall_passed IN (0, 1)),
                confidence REAL NOT NULL
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
                confidence_threshold REAL NOT NULL
                    CHECK (
                        confidence_threshold >= 0.0
                        AND confidence_threshold <= 1.0
                    ),
                below_confidence_threshold INTEGER NOT NULL
                    CHECK (below_confidence_threshold IN (0, 1)),
                PRIMARY KEY (run_id, eval_id),
                FOREIGN KEY (run_id, eval_id)
                    REFERENCES experiment_case_results(
                        run_id, eval_id
                    ) ON DELETE CASCADE,
                FOREIGN KEY (run_id, eval_id, final_attempt_number)
                    REFERENCES experiment_judge_attempts(
                        run_id, eval_id, attempt_number
                    ) ON DELETE RESTRICT,
                CHECK (
                    below_confidence_threshold
                    = (confidence < confidence_threshold)
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO experiment_judge_results_a2 (
                run_id, eval_id, final_attempt_number,
                response_schema_version, overall_score, overall_passed,
                confidence, confidence_threshold, below_confidence_threshold
            )
            SELECT
                run_id, eval_id, final_attempt_number,
                response_schema_version, overall_score, overall_passed,
                confidence, confidence_threshold, below_confidence_threshold
            FROM experiment_judge_results
            """
        )
        connection.execute("DROP TABLE experiment_judge_results")
        connection.execute(
            "ALTER TABLE experiment_judge_results_a2 RENAME TO experiment_judge_results"
        )
        connection.execute("DROP TABLE experiment_judge_cache_lookups")
        connection.execute("DROP TABLE judge_result_cache")
        connection.commit()

    with closing(connect_database(database_path)) as connection:
        after = load_experiment_report(connection, before.experiment_id)
        lookup_rows = connection.execute(
            "SELECT cache_key, cache_status "
            "FROM experiment_judge_cache_lookups ORDER BY run_id"
        ).fetchall()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert after.schema_version == 1
    for role in RunRole:
        before_case = before.runs[role].cases[0]
        after_case = after.runs[role].cases[0]
        assert after_case.score == before_case.score
        assert after_case.passed == before_case.passed
        assert after_case.judge is not None
        assert after_case.judge.cache_hit is None
        assert after_case.judge.cache.status is JudgeCacheStatus.NOT_RECORDED
        assert after_case.judge.cache.key is None
        assert after_case.judge.review.status is JudgeReviewStatus.NEEDS_REVIEW
        assert after_case.judge.review.reasons == [JudgeReviewReason.CRITICAL_FAILURE]
    assert [tuple(row) for row in lookup_rows] == [
        (None, "not_recorded"),
        (None, "not_recorded"),
    ]
    assert foreign_key_errors == []


def test_a1_constrained_schema_migrates_to_rubric_without_report_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.sqlite3"
    with closing(connect_database(database_path)) as connection, connection:
        insert_trace(
            connection,
            Trace.model_validate(
                {
                    "trace_id": "migration-reference",
                    "timestamp": "2026-07-29T12:00:00Z",
                    "task_type": "qa",
                    "prompt": "Capital of Canada?",
                    "response": "Ottawa",
                }
            ),
        )
    create_dataset(database_path, name="migration", version="1")
    case = promote_trace(
        database_path,
        dataset_reference="migration:1",
        trace_id="migration-reference",
        mode=EvaluationMode.REFERENCE,
        use_source_response=True,
    )
    for filename in ("migration-baseline.jsonl", "migration-candidate.jsonl"):
        (tmp_path / filename).write_text(
            json.dumps({"eval_id": case.eval_id, "output": "Ottawa"}) + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "migration.yaml"
    config_path.write_text(
        """schema_version: 1
name: migration
dataset: migration:1
baseline: {provider: fixture, path: migration-baseline.jsonl}
candidate: {provider: fixture, path: migration-candidate.jsonl}
""",
        encoding="utf-8",
    )
    before = execute_experiment(config_path, database_path)

    with closing(connect_database(database_path)) as connection:
        current_tables = (
            "experiment_case_results",
            "experiment_run_aggregates",
            "experiment_case_comparisons",
            "experiment_comparison_aggregates",
            "experiment_gate_violations",
        )
        connection.execute("PRAGMA foreign_keys = OFF")
        for trigger in (
            "validate_experiment_result_relation_insert",
            "validate_experiment_result_relation_update",
            "validate_experiment_comparison_relation_insert",
            "validate_experiment_comparison_relation_update",
            "validate_judge_attempt_relation_insert",
            "validate_judge_attempt_relation_update",
            "validate_judge_result_relation_insert",
            "validate_judge_result_relation_update",
        ):
            connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
        for table_name in current_tables:
            marker = f"CREATE TABLE IF NOT EXISTS {table_name} ("
            current = next(
                statement for statement in SCHEMA_STATEMENTS if marker in statement
            )
            old = current.replace(
                "'deterministic', 'reference', 'rubric'", "'deterministic', 'reference'"
            )
            old = old.replace(
                "'global', 'deterministic', 'reference', 'rubric'",
                "'global', 'deterministic', 'reference'",
            )
            old_name = f"{table_name}_a1"
            old = old.replace(marker, f"CREATE TABLE {old_name} (", 1)
            columns = [
                row["name"]
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            encoded = ", ".join(f'"{column}"' for column in columns)
            connection.execute(old)
            connection.execute(
                f'INSERT INTO "{old_name}" ({encoded}) '
                f'SELECT {encoded} FROM "{table_name}"'
            )
            connection.execute(f'DROP TABLE "{table_name}"')
            connection.execute(f'ALTER TABLE "{old_name}" RENAME TO "{table_name}"')
        connection.commit()

    with closing(connect_database(database_path)) as connection:
        after = load_experiment_report(connection, before.experiment_id)
        schemas = {
            table: connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()[0]
            for table in current_tables
        }
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert after.model_dump(mode="json", by_alias=True) == before.model_dump(
        mode="json", by_alias=True
    )
    assert all("'rubric'" in schema for schema in schemas.values())
