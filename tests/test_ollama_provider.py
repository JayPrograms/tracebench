"""Tests for versioned prompts and Ollama-compatible local generation."""

import io
import json
import sqlite3
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from typer.testing import CliRunner

import tracebench.providers as provider_module
from tracebench.cli import app
from tracebench.datasets import create_dataset, promote_trace
from tracebench.experiment_config import (
    ExperimentPreflightError,
    load_experiment_config,
    prepare_experiment,
)
from tracebench.experiment_models import RunRole
from tracebench.experiment_storage import load_experiment_report
from tracebench.experiments import ExperimentOperationalError, execute_experiment
from tracebench.models import EvalCase, EvaluationMode, Trace
from tracebench.providers import (
    OllamaProvider,
    ProviderError,
    ProviderRequest,
    build_prompt,
)
from tracebench.storage import connect_database, insert_trace

runner = CliRunner()


class StubResponse:
    """Minimal context-managed HTTP response for provider tests."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "StubResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def reference_case(tmp_path: Path) -> tuple[Path, EvalCase]:
    """Create one reference evaluation case."""
    database_path = tmp_path / "tracebench.sqlite3"
    with closing(connect_database(database_path)) as connection, connection:
        insert_trace(
            connection,
            Trace.model_validate(
                {
                    "trace_id": "trace-001",
                    "timestamp": "2026-07-28T14:00:00Z",
                    "task_type": "question-answering",
                    "prompt": "What is the capital of Canada?",
                    "response": "Ottawa",
                    "context": {"locale": "français", "facts": {"country": "CA"}},
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
    return database_path, case


def write_ollama_config(
    config_path: Path,
    prompt_file: str,
    *,
    base_url: str = "http://localhost:11434/",
    prompt_version: str = "support-v1",
    extra_candidate: str = "",
) -> None:
    """Write an Ollama/Ollama experiment configuration."""
    config_path.write_text(
        (
            "schema_version: 1\n"
            "name: local-generation\n"
            "dataset: support:1\n"
            "baseline: &local\n"
            "  provider: ollama\n"
            f"  base_url: {json.dumps(base_url)}\n"
            "  model: llama3.2:3b\n"
            f"  prompt_version: {prompt_version}\n"
            f"  system_prompt_file: {prompt_file}\n"
            "  temperature: 0\n"
            "  timeout_seconds: 30\n"
            "  seed: 7\n"
            "candidate:\n"
            "  <<: *local\n"
            f"{extra_candidate}"
        ),
        encoding="utf-8",
    )


def ollama_provider(*, seed: int | None = 7) -> OllamaProvider:
    """Build a compact provider for HTTP adapter tests."""
    return OllamaProvider(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        temperature=0.25,
        timeout_seconds=12,
        seed=seed,
    )


def provider_request(
    case: EvalCase, prompt: str = "Answer directly."
) -> ProviderRequest:
    """Render the request accepted by the shared provider protocol."""
    return ProviderRequest(
        request_id=case.eval_id,
        prompt=build_prompt(case, prompt),
    )


def test_prompt_construction_is_exact_and_excludes_answer_data(tmp_path: Path) -> None:
    """The versioned prompt has stable delimiters and canonical context."""
    _, case = reference_case(tmp_path)

    prompt = build_prompt(case, "System prompt.\n")

    assert prompt == (
        "System prompt.\n\n\n"
        "<TRACEBENCH_EVALUATION_INPUT>\n"
        "What is the capital of Canada?\n"
        "</TRACEBENCH_EVALUATION_INPUT>\n\n"
        "<TRACEBENCH_CONTEXT_JSON>\n"
        '{"facts":{"country":"CA"},"locale":"français"}\n'
        "</TRACEBENCH_CONTEXT_JSON>"
    )
    assert case.reference_answer not in prompt
    assert case.source_response not in prompt
    assert case.source_trace_id not in prompt


def test_ollama_request_and_response_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter sends one non-streaming request and retains scalar metrics."""
    _, case = reference_case(tmp_path)
    captured: dict[str, object] = {}
    response_payload = {
        "model": "llama3.2:3b",
        "created_at": "2026-07-29T15:00:00Z",
        "response": "Ottawa",
        "done": True,
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 30,
        "eval_count": 2,
        "eval_duration": 40,
        "future_field": {"ignored": True},
    }

    def fake_urlopen(request: Request, timeout: float) -> StubResponse:
        captured["request"] = request
        captured["timeout"] = timeout
        return StubResponse(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    response = ollama_provider().generate(provider_request(case))

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "http://localhost:11434/api/generate"
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("Accept") == "application/json"
    assert captured["timeout"] == 12
    assert request.data is not None
    sent = json.loads(request.data)
    assert sent == {
        "model": "llama3.2:3b",
        "options": {"seed": 7, "temperature": 0.25},
        "prompt": build_prompt(case, "Answer directly."),
        "stream": False,
    }
    assert response.output == "Ottawa"
    assert response.metadata == {
        key: response_payload[key]
        for key in (
            "model",
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    }


def test_ollama_json_mode_is_selected_by_provider_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge requests add Ollama JSON mode without changing text requests."""
    captured: list[dict[str, Any]] = []

    def fake_urlopen(request: Request, timeout: float) -> StubResponse:
        assert request.data is not None
        captured.append(json.loads(request.data))
        return StubResponse(b'{"response":"{}","done":true}')

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    response = ollama_provider().generate(
        ProviderRequest(
            request_id="baseline:eval_rubric",
            prompt="judge",
            response_format="json",
        )
    )

    assert response.output == "{}"
    assert captured[0]["format"] == "json"


def test_ollama_omits_unset_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent seed is omitted instead of being sent as null."""
    _, case = reference_case(tmp_path)
    captured: list[dict[str, Any]] = []

    def fake_urlopen(request: Request, timeout: float) -> StubResponse:
        assert request.data is not None
        captured.append(json.loads(request.data))
        return StubResponse(b'{"response":"","done":true}')

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    response = ollama_provider(seed=None).generate(provider_request(case))

    assert response.output == ""
    assert captured[0]["options"] == {"temperature": 0.25}


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "invalid JSON response"),
        (b"[]", "expected a JSON object"),
        (b'{"response":1,"done":true}', "response must be a string"),
        (b'{"response":"value","done":false}', "done must be true"),
        (
            b'{"response":"value","done":true,"eval_count":-1}',
            "eval_count must be a nonnegative integer",
        ),
    ],
)
def test_ollama_rejects_invalid_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    message: str,
) -> None:
    """Malformed or incomplete provider responses fail clearly."""
    _, case = reference_case(tmp_path)
    monkeypatch.setattr(
        provider_module,
        "urlopen",
        lambda request, timeout: StubResponse(body),
    )

    with pytest.raises(ProviderError, match=message):
        ollama_provider().generate(provider_request(case))


def test_ollama_reports_http_connection_and_timeout_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transport failures retain actionable provider context."""
    _, case = reference_case(tmp_path)

    def raise_http(request: Request, timeout: float) -> StubResponse:
        raise HTTPError(
            request.full_url,
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr(provider_module, "urlopen", raise_http)
    with pytest.raises(ProviderError, match="HTTP 404: model not found"):
        ollama_provider().generate(provider_request(case))

    def raise_connection(request: Request, timeout: float) -> StubResponse:
        raise URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(provider_module, "urlopen", raise_connection)
    with pytest.raises(ProviderError, match="could not connect to Ollama"):
        ollama_provider().generate(provider_request(case))

    def raise_timeout(request: Request, timeout: float) -> StubResponse:
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(provider_module, "urlopen", raise_timeout)
    with pytest.raises(ProviderError, match="timed out after 12 seconds"):
        ollama_provider().generate(provider_request(case))


def test_relative_prompt_is_preflighted_and_snapshot_is_path_free(
    tmp_path: Path,
) -> None:
    """Prompt paths use the YAML directory while identity uses decoded content."""
    database_path, _ = reference_case(tmp_path)
    config_directory = tmp_path / "configuration"
    prompt_directory = config_directory / "prompts"
    prompt_directory.mkdir(parents=True)
    prompt_contents = "Answer using only supplied facts."
    (prompt_directory / "system.txt").write_text(
        f"\ufeff{prompt_contents}", encoding="utf-8"
    )
    config_path = config_directory / "experiment.yaml"
    write_ollama_config(config_path, "prompts/system.txt")

    prepared = prepare_experiment(config_path, database_path)

    expected_hash = sha256(prompt_contents.encode("utf-8")).hexdigest()
    for role in RunRole:
        assert prepared.provider_snapshots[role]["prompt_version"] == "support-v1"
        assert prepared.provider_snapshots[role]["system_prompt_hash"] == expected_hash
        assert "system_prompt_file" not in prepared.provider_snapshots[role]
    assert str(config_directory.resolve()) not in prepared.configuration_json
    assert "prompts/system.txt" not in prepared.configuration_json


def test_system_prompt_preserves_crlf_content_and_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Byte decoding preserves CRLF in prompt construction and identity."""
    database_path, case = reference_case(tmp_path)
    decoded_prompt = "First instruction.\r\nSecond instruction.\r\n"
    (tmp_path / "system.txt").write_bytes(
        b"\xef\xbb\xbf" + decoded_prompt.encode("utf-8")
    )
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt")
    captured_prompts: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> StubResponse:
        assert request.data is not None
        captured_prompts.append(json.loads(request.data)["prompt"])
        return StubResponse(b'{"response":"Ottawa","done":true}')

    monkeypatch.setattr(provider_module, "urlopen", fake_urlopen)

    prepared = prepare_experiment(config_path, database_path)
    prepared.providers[RunRole.BASELINE].generate(
        ProviderRequest(
            request_id=case.eval_id,
            prompt=build_prompt(
                case,
                prepared.provider_prompts[RunRole.BASELINE] or "",
            ),
        )
    )

    expected_hash = sha256(decoded_prompt.encode("utf-8")).hexdigest()
    normalized_lf_hash = sha256(
        decoded_prompt.replace("\r\n", "\n").encode("utf-8")
    ).hexdigest()
    assert prepared.provider_snapshots[RunRole.BASELINE]["system_prompt_hash"] == (
        expected_hash
    )
    assert expected_hash != normalized_lf_hash
    assert captured_prompts == [build_prompt(case, decoded_prompt)]
    assert "First instruction.\r\nSecond instruction.\r\n\n\n" in captured_prompts[0]


def test_prompt_hash_depends_on_content_and_not_location(tmp_path: Path) -> None:
    """Identical relocated prompts share identity while content changes do not."""
    database_path, _ = reference_case(tmp_path)
    prepared = []
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "system.txt").write_text("Stable prompt.", encoding="utf-8")
        config_path = directory / "experiment.yaml"
        write_ollama_config(config_path, "system.txt")
        prepared.append(prepare_experiment(config_path, database_path))

    assert prepared[0].configuration_hash == prepared[1].configuration_hash
    assert prepared[0].provider_snapshots == prepared[1].provider_snapshots

    (tmp_path / "second" / "system.txt").write_text("Changed prompt.", encoding="utf-8")
    changed = prepare_experiment(tmp_path / "second" / "experiment.yaml", database_path)
    assert changed.configuration_hash != prepared[0].configuration_hash

    write_ollama_config(
        tmp_path / "second" / "experiment.yaml",
        "system.txt",
        prompt_version="support-v2",
    )
    changed_version = prepare_experiment(
        tmp_path / "second" / "experiment.yaml", database_path
    )
    assert changed_version.configuration_hash != changed.configuration_hash


@pytest.mark.parametrize(
    ("prompt_kind", "message"),
    [
        ("missing", "could not read system prompt"),
        ("blank", "must not be blank"),
        ("invalid-encoding", "could not read system prompt"),
        ("directory", "could not read system prompt"),
    ],
)
def test_invalid_prompt_files_exit_two_without_attempt(
    tmp_path: Path,
    prompt_kind: str,
    message: str,
) -> None:
    """Every prompt-file preflight failure is a usage error with no attempt."""
    database_path, _ = reference_case(tmp_path)
    prompt_path = tmp_path / "system.txt"
    if prompt_kind == "blank":
        prompt_path.write_text("\ufeff \n\t", encoding="utf-8")
    elif prompt_kind == "invalid-encoding":
        prompt_path.write_bytes(b"\xff\xfe\x00")
    elif prompt_kind == "directory":
        prompt_path.mkdir()
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt")

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path)],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert message in result.stderr
    assert "no experiment attempt was persisted" in result.stderr
    with closing(connect_database(database_path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "'   '"),
        ("prompt_version", "'   '"),
        ("system_prompt_file", "''"),
        ("temperature", "true"),
        ("timeout_seconds", "0"),
        ("seed", "false"),
        ("unknown", "value"),
    ],
)
def test_ollama_configuration_is_strict(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """Invalid local-provider values fail strict YAML validation."""
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(
        config_path, "system.txt", extra_candidate=f"  {field}: {value}\n"
    )

    with pytest.raises(ExperimentPreflightError):
        load_experiment_config(config_path)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://:11434",
        "http://localhost",
        "http://localhost:not-a-port",
        "http://localhost:0",
        "http://localhost:65536",
        "http://bad_host:11434",
        "http://999.999.999.999:11434",
        "http://localhost:11434/api",
        "http://localhost:11434//",
    ],
)
def test_invalid_ollama_base_urls_exit_two_without_attempt(
    tmp_path: Path,
    base_url: str,
) -> None:
    """Malformed hosts, ports, and endpoint paths fail during preflight."""
    database_path, _ = reference_case(tmp_path)
    (tmp_path / "system.txt").write_text("Answer directly.", encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt", base_url=base_url)

    result = runner.invoke(
        app,
        ["experiment", "run", str(config_path)],
        env={"TRACEBENCH_DB_PATH": str(database_path)},
    )

    assert result.exit_code == 2
    assert "invalid experiment configuration" in result.stderr
    assert "no experiment attempt was persisted" in result.stderr
    with closing(connect_database(database_path)) as connection:
        attempt_count = connection.execute(
            "SELECT COUNT(*) FROM experiments"
        ).fetchone()[0]
    assert attempt_count == 0


@pytest.mark.parametrize(
    ("base_url", "normalized"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("https://127.0.0.1:443", "https://127.0.0.1:443"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_valid_ollama_server_roots_are_normalized(
    tmp_path: Path,
    base_url: str,
    normalized: str,
) -> None:
    """Hostnames and IP literals with explicit valid ports remain supported."""
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt", base_url=base_url)

    config = load_experiment_config(config_path)

    assert config.baseline.base_url == normalized
    assert config.candidate.base_url == normalized


def test_local_generation_persists_observability_without_changing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Latency and provider metadata stay normalized and out of public JSON."""
    database_path, _ = reference_case(tmp_path)
    prompt_contents = "Answer directly."
    (tmp_path / "system.txt").write_text(prompt_contents, encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt")
    response_payload = {
        "model": "llama3.2:3b",
        "response": "Ottawa",
        "done": True,
        "done_reason": "stop",
        "total_duration": 1234,
        "eval_count": 2,
    }
    monkeypatch.setattr(
        provider_module,
        "urlopen",
        lambda request, timeout: StubResponse(
            json.dumps(response_payload).encode("utf-8")
        ),
    )

    report = execute_experiment(config_path, database_path)

    public = report.model_dump(mode="json", by_alias=True)
    for run in public["runs"].values():
        for case in run["cases"]:
            assert "generation_latency_ms" not in case
            assert "provider_metadata" not in case
    expected_prompt_hash = sha256(prompt_contents.encode("utf-8")).hexdigest()
    with closing(connect_database(database_path)) as connection:
        runs = connection.execute(
            "SELECT provider_name, provider_config_json "
            "FROM experiment_runs ORDER BY role"
        ).fetchall()
        results = connection.execute(
            "SELECT generation_latency_ms, provider_metadata_json "
            "FROM experiment_case_results ORDER BY run_id"
        ).fetchall()
    assert [run["provider_name"] for run in runs] == ["ollama", "ollama"]
    for run in runs:
        snapshot = json.loads(run["provider_config_json"])
        assert snapshot["prompt_version"] == "support-v1"
        assert snapshot["system_prompt_hash"] == expected_prompt_hash
        assert "system_prompt_file" not in snapshot
    for result in results:
        assert result["generation_latency_ms"] >= 0
        assert json.loads(result["provider_metadata_json"]) == {
            "done": True,
            "done_reason": "stop",
            "eval_count": 2,
            "model": "llama3.2:3b",
            "total_duration": 1234,
        }


def test_timeout_is_an_operational_failure_after_attempt_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reachable configuration can still fail during its active run."""
    database_path, _ = reference_case(tmp_path)
    (tmp_path / "system.txt").write_text("Answer directly.", encoding="utf-8")
    config_path = tmp_path / "experiment.yaml"
    write_ollama_config(config_path, "system.txt")

    def raise_timeout(request: Request, timeout: float) -> StubResponse:
        raise URLError(TimeoutError("timed out"))

    monkeypatch.setattr(provider_module, "urlopen", raise_timeout)

    with pytest.raises(ExperimentOperationalError, match="timed out") as caught:
        execute_experiment(config_path, database_path)

    assert caught.value.stage == "baseline"
    with closing(connect_database(database_path)) as connection:
        attempt = connection.execute(
            "SELECT status, verdict, failure_stage FROM experiments"
        ).fetchone()
        run_statuses = connection.execute(
            "SELECT role, status FROM experiment_runs ORDER BY role"
        ).fetchall()
    assert dict(attempt) == {
        "status": "failed",
        "verdict": None,
        "failure_stage": "baseline",
    }
    assert {row["role"]: row["status"] for row in run_statuses} == {
        "baseline": "failed",
        "candidate": "skipped",
    }


def test_current_experiment_schema_migrates_in_place(tmp_path: Path) -> None:
    """Fixture attempts survive the provider and observability schema upgrade."""
    database_path, case = reference_case(tmp_path)
    for filename in ("baseline.jsonl", "candidate.jsonl"):
        (tmp_path / filename).write_text(
            json.dumps({"eval_id": case.eval_id, "output": "Ottawa"}) + "\n",
            encoding="utf-8",
        )
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        """schema_version: 1
name: migration-fixture
dataset: support:1
baseline: {provider: fixture, path: baseline.jsonl}
candidate: {provider: fixture, path: candidate.jsonl}
""",
        encoding="utf-8",
    )
    before = execute_experiment(config_path, database_path)

    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TRIGGER validate_experiment_result_relation_insert")
        connection.execute("DROP TRIGGER validate_experiment_result_relation_update")
        connection.execute("DROP TRIGGER validate_judge_attempt_relation_insert")
        connection.execute("DROP TRIGGER validate_judge_attempt_relation_update")
        connection.execute("DROP TRIGGER validate_judge_cache_lookup_relation_insert")
        connection.execute("DROP TRIGGER validate_judge_cache_lookup_relation_update")
        connection.execute(
            """
            CREATE TABLE experiment_runs_old (
                run_id TEXT PRIMARY KEY CHECK (length(trim(run_id)) > 0),
                experiment_id TEXT NOT NULL
                    REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('baseline', 'candidate')),
                provider_name TEXT NOT NULL CHECK (provider_name = 'fixture'),
                provider_config_json TEXT NOT NULL
                    CHECK (
                        json_valid(provider_config_json)
                        AND json_type(provider_config_json) = 'object'
                    ),
                status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'pending', 'running', 'completed', 'failed', 'skipped'
                        )
                    ),
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                UNIQUE (experiment_id, role),
                CHECK (
                    (status = 'pending' AND error_message IS NULL
                        AND started_at IS NULL AND completed_at IS NULL)
                    OR (status = 'running' AND error_message IS NULL
                        AND started_at IS NOT NULL AND completed_at IS NULL)
                    OR (status = 'completed' AND error_message IS NULL
                        AND started_at IS NOT NULL AND completed_at IS NOT NULL)
                    OR (status = 'failed' AND length(trim(error_message)) > 0
                        AND started_at IS NOT NULL AND completed_at IS NOT NULL)
                    OR (status = 'skipped' AND length(trim(error_message)) > 0
                        AND started_at IS NULL AND completed_at IS NOT NULL)
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO experiment_runs_old
            SELECT * FROM experiment_runs
            """
        )
        connection.execute("DROP TABLE experiment_runs")
        connection.execute("ALTER TABLE experiment_runs_old RENAME TO experiment_runs")
        connection.execute(
            "ALTER TABLE experiment_case_results DROP COLUMN provider_metadata_json"
        )
        connection.execute(
            "ALTER TABLE experiment_case_results DROP COLUMN generation_latency_ms"
        )
        connection.commit()

    with closing(connect_database(database_path)) as connection:
        after = load_experiment_report(connection, before.experiment_id)
        run_schema = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'experiment_runs'"
        ).fetchone()[0]
        migrated_results = connection.execute(
            "SELECT generation_latency_ms, provider_metadata_json "
            "FROM experiment_case_results"
        ).fetchall()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert after.model_dump(mode="json", by_alias=True) == before.model_dump(
        mode="json", by_alias=True
    )
    assert "provider_name IN ('fixture', 'ollama')" in run_schema
    assert all(row["generation_latency_ms"] is None for row in migrated_results)
    assert all(row["provider_metadata_json"] == "{}" for row in migrated_results)
    assert foreign_key_errors == []
