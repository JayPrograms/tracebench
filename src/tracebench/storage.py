"""SQLite persistence for traces."""

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tracebench.models import EvalCase, EvalDataset, ScorerConfig, Trace

DATABASE_PATH_ENV = "TRACEBENCH_DB_PATH"
DEFAULT_DATABASE_PATH = Path(".tracebench") / "tracebench.sqlite3"

# SQLite's one-argument trim() removes only U+0020, while Python's str.strip()
# recognizes this full set. Use the same characters in schema constraints so a
# direct SQL write cannot create selectors that Python later canonicalizes.
_SQLITE_PYTHON_WHITESPACE = (
    "char(9,10,11,12,13,28,29,30,31,32,133,160,5760,"
    "8192,8193,8194,8195,8196,8197,8198,8199,8200,8201,8202,"
    "8232,8233,8239,8287,12288)"
)

SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY CHECK (length(trim(trace_id)) > 0),
    timestamp TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK (length(trim(task_type)) > 0),
    prompt TEXT NOT NULL CHECK (length(trim(prompt)) > 0),
    response TEXT,
    context_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}'
)
""",
    """
CREATE INDEX IF NOT EXISTS idx_traces_timestamp_trace_id
ON traces (timestamp DESC, trace_id ASC)
""",
    f"""
CREATE TABLE IF NOT EXISTS eval_datasets (
    dataset_id TEXT PRIMARY KEY CHECK (length(trim(dataset_id)) > 0),
    name TEXT NOT NULL
        CHECK (
            length(name) > 0
            AND name = trim(name, {_SQLITE_PYTHON_WHITESPACE})
            AND instr(name, ':') = 0
        ),
    version TEXT NOT NULL
        CHECK (
            length(version) > 0
            AND version = trim(version, {_SQLITE_PYTHON_WHITESPACE})
            AND instr(version, ':') = 0
        ),
    description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (name, version)
)
""",
    """
CREATE TABLE IF NOT EXISTS eval_cases (
    eval_id TEXT PRIMARY KEY CHECK (length(trim(eval_id)) > 0),
    dataset_id TEXT NOT NULL
        REFERENCES eval_datasets(dataset_id) ON DELETE CASCADE,
    source_trace_id TEXT NOT NULL
        REFERENCES traces(trace_id) ON DELETE RESTRICT,
    source_timestamp TEXT NOT NULL,
    source_task_type TEXT NOT NULL
        CHECK (length(trim(source_task_type)) > 0),
    source_response TEXT,
    source_metadata_json TEXT NOT NULL
        CHECK (
            json_valid(source_metadata_json)
            AND json_type(source_metadata_json) = 'object'
        ),
    input TEXT NOT NULL CHECK (length(trim(input)) > 0),
    context_json TEXT NOT NULL
        CHECK (json_valid(context_json) AND json_type(context_json) = 'object'),
    evaluation_mode TEXT NOT NULL
        CHECK (evaluation_mode IN ('deterministic', 'reference', 'rubric')),
    reference_answer TEXT,
    rubric_json TEXT NOT NULL
        CHECK (json_valid(rubric_json) AND json_type(rubric_json) = 'array'),
    scorers_json TEXT NOT NULL
        CHECK (json_valid(scorers_json) AND json_type(scorers_json) = 'array'),
    priority TEXT NOT NULL
        CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    review_status TEXT NOT NULL
        CHECK (review_status IN ('draft', 'approved', 'rejected')),
    created_at TEXT NOT NULL,
    UNIQUE (dataset_id, source_trace_id),
    CHECK (
        (
            evaluation_mode = 'deterministic'
            AND source_response IS NULL
            AND reference_answer IS NULL
            AND json_array_length(rubric_json) = 0
            AND json_array_length(scorers_json) > 0
        )
        OR (
            evaluation_mode = 'reference'
            AND
            reference_answer IS NOT NULL
            AND length(trim(reference_answer)) > 0
            AND json_array_length(rubric_json) = 0
            AND json_array_length(scorers_json) = 0
        )
        OR (
            evaluation_mode = 'rubric'
            AND source_response IS NULL
            AND reference_answer IS NULL
            AND json_array_length(rubric_json) > 0
            AND json_array_length(scorers_json) = 0
        )
    )
)
""",
    """
CREATE INDEX IF NOT EXISTS idx_eval_cases_dataset_eval
ON eval_cases (dataset_id, eval_id ASC)
    """,
    """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY CHECK (length(trim(experiment_id)) > 0),
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    dataset_id TEXT NOT NULL
        REFERENCES eval_datasets(dataset_id) ON DELETE RESTRICT,
    configuration_hash TEXT NOT NULL
        CHECK (
            length(configuration_hash) = 64
            AND configuration_hash NOT GLOB '*[^0-9a-f]*'
        ),
    configuration_json TEXT NOT NULL
        CHECK (
            json_valid(configuration_json)
            AND json_type(configuration_json) = 'object'
        ),
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    verdict TEXT CHECK (verdict IN ('PASS', 'FAIL')),
    failure_stage TEXT,
    failure_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (
        (
            status = 'running'
            AND verdict IS NULL
            AND failure_stage IS NULL
            AND failure_message IS NULL
            AND completed_at IS NULL
        )
        OR (
            status = 'completed'
            AND verdict IN ('PASS', 'FAIL')
            AND failure_stage IS NULL
            AND failure_message IS NULL
            AND completed_at IS NOT NULL
        )
        OR (
            status = 'failed'
            AND verdict IS NULL
            AND length(trim(failure_stage)) > 0
            AND length(trim(failure_message)) > 0
            AND completed_at IS NOT NULL
        )
    )
)
""",
    """
CREATE INDEX IF NOT EXISTS idx_experiments_name_created
ON experiments (name ASC, created_at DESC, experiment_id ASC)
""",
    """
CREATE INDEX IF NOT EXISTS idx_experiments_configuration_hash
ON experiments (configuration_hash, created_at ASC)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY CHECK (length(trim(run_id)) > 0),
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('baseline', 'candidate')),
    provider_name TEXT NOT NULL
        CHECK (provider_name IN ('fixture', 'ollama')),
    provider_config_json TEXT NOT NULL
        CHECK (
            json_valid(provider_config_json)
            AND json_type(provider_config_json) = 'object'
        ),
    status TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    error_message TEXT,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE (experiment_id, role),
    CHECK (
        (
            status = 'pending'
            AND error_message IS NULL
            AND started_at IS NULL
            AND completed_at IS NULL
        )
        OR (
            status = 'running'
            AND error_message IS NULL
            AND started_at IS NOT NULL
            AND completed_at IS NULL
        )
        OR (
            status = 'completed'
            AND error_message IS NULL
            AND started_at IS NOT NULL
            AND completed_at IS NOT NULL
        )
        OR (
            status = 'failed'
            AND length(trim(error_message)) > 0
            AND started_at IS NOT NULL
            AND completed_at IS NOT NULL
        )
        OR (
            status = 'skipped'
            AND length(trim(error_message)) > 0
            AND started_at IS NULL
            AND completed_at IS NOT NULL
        )
    )
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_case_results (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL
        REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    evaluation_mode TEXT NOT NULL
        CHECK (evaluation_mode IN ('deterministic', 'reference')),
    output TEXT NOT NULL,
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    generation_latency_ms REAL CHECK (generation_latency_ms >= 0.0),
    provider_metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (
            json_valid(provider_metadata_json)
            AND json_type(provider_metadata_json) = 'object'
        ),
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, eval_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_scorer_results (
    run_id TEXT NOT NULL,
    eval_id TEXT NOT NULL,
    scorer_index INTEGER NOT NULL CHECK (scorer_index >= 0),
    scorer_name TEXT NOT NULL CHECK (length(trim(scorer_name)) > 0),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
    details_json TEXT NOT NULL
        CHECK (json_valid(details_json) AND json_type(details_json) = 'object'),
    PRIMARY KEY (run_id, eval_id, scorer_index),
    FOREIGN KEY (run_id, eval_id)
        REFERENCES experiment_case_results(run_id, eval_id) ON DELETE CASCADE
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_run_aggregates (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    scope TEXT NOT NULL
        CHECK (scope IN ('global', 'deterministic', 'reference')),
    case_count INTEGER NOT NULL CHECK (case_count > 0),
    passed_count INTEGER NOT NULL CHECK (passed_count >= 0),
    failed_count INTEGER NOT NULL CHECK (failed_count >= 0),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    pass_rate REAL NOT NULL CHECK (pass_rate >= 0.0 AND pass_rate <= 1.0),
    PRIMARY KEY (run_id, scope),
    CHECK (passed_count + failed_count = case_count)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_case_comparisons (
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL
        REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    evaluation_mode TEXT NOT NULL
        CHECK (evaluation_mode IN ('deterministic', 'reference')),
    baseline_score REAL NOT NULL
        CHECK (baseline_score >= 0.0 AND baseline_score <= 1.0),
    candidate_score REAL NOT NULL
        CHECK (candidate_score >= 0.0 AND candidate_score <= 1.0),
    score_delta REAL NOT NULL CHECK (score_delta >= -1.0 AND score_delta <= 1.0),
    baseline_passed INTEGER NOT NULL CHECK (baseline_passed IN (0, 1)),
    candidate_passed INTEGER NOT NULL CHECK (candidate_passed IN (0, 1)),
    transition TEXT NOT NULL
        CHECK (
            transition IN (
                'passed_both', 'failed_both', 'newly_passed', 'newly_failed'
            )
        ),
    PRIMARY KEY (experiment_id, eval_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_comparison_aggregates (
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    scope TEXT NOT NULL
        CHECK (scope IN ('global', 'deterministic', 'reference')),
    case_count INTEGER NOT NULL CHECK (case_count > 0),
    baseline_score REAL NOT NULL
        CHECK (baseline_score >= 0.0 AND baseline_score <= 1.0),
    candidate_score REAL NOT NULL
        CHECK (candidate_score >= 0.0 AND candidate_score <= 1.0),
    score_delta REAL NOT NULL CHECK (score_delta >= -1.0 AND score_delta <= 1.0),
    newly_passed_count INTEGER NOT NULL CHECK (newly_passed_count >= 0),
    newly_failed_count INTEGER NOT NULL CHECK (newly_failed_count >= 0),
    PRIMARY KEY (experiment_id, scope),
    CHECK (newly_passed_count <= case_count),
    CHECK (newly_failed_count <= case_count)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_gate_violations (
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    violation_index INTEGER NOT NULL CHECK (violation_index >= 0),
    scope TEXT NOT NULL
        CHECK (scope IN ('global', 'deterministic', 'reference')),
    metric TEXT NOT NULL CHECK (metric IN ('score_drop', 'new_failures')),
    actual REAL NOT NULL CHECK (actual >= 0.0),
    allowed REAL NOT NULL CHECK (allowed >= 0.0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    PRIMARY KEY (experiment_id, violation_index)
)
""",
)

_EVAL_CASE_JSON_VALIDATION = f"""
    SELECT CASE WHEN EXISTS (
        SELECT 1
        FROM json_each(NEW.rubric_json)
        WHERE type != 'text'
            OR length(
                trim(CAST(value AS TEXT), {_SQLITE_PYTHON_WHITESPACE})
            ) = 0
    ) THEN RAISE(ABORT, 'rubric_json must contain nonblank strings') END;
    SELECT CASE WHEN (
        SELECT COUNT(*) FROM json_each(NEW.rubric_json)
    ) != (
        SELECT COUNT(DISTINCT trim(
            CAST(value AS TEXT), {_SQLITE_PYTHON_WHITESPACE}
        ))
        FROM json_each(NEW.rubric_json)
    ) THEN RAISE(ABORT, 'rubric_json must not contain duplicates') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1
        FROM json_each(NEW.scorers_json) AS scorer
        WHERE scorer.type != 'object'
            OR json_type(scorer.value, '$.name') IS NOT 'text'
            OR length(trim(
                json_extract(scorer.value, '$.name'),
                {_SQLITE_PYTHON_WHITESPACE}
            )) = 0
            OR (
                json_type(scorer.value, '$.config') IS NOT NULL
                AND json_type(scorer.value, '$.config') != 'object'
            )
            OR EXISTS (
                SELECT 1
                FROM json_each(scorer.value) AS field
                WHERE field.key NOT IN ('name', 'config')
            )
    ) THEN RAISE(ABORT, 'scorers_json contains an invalid scorer') END;
"""

EVAL_CASE_TRIGGER_STATEMENTS = (
    f"""
CREATE TRIGGER IF NOT EXISTS validate_eval_case_json_insert
BEFORE INSERT ON eval_cases
BEGIN
{_EVAL_CASE_JSON_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_eval_case_json_update
BEFORE UPDATE ON eval_cases
BEGIN
{_EVAL_CASE_JSON_VALIDATION}
END
    """,
)

_EXPERIMENT_RESULT_RELATION_VALIDATION = """
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiment_runs AS run
        JOIN experiments AS experiment
            ON experiment.experiment_id = run.experiment_id
        JOIN eval_cases AS eval_case
            ON eval_case.eval_id = NEW.eval_id
        WHERE run.run_id = NEW.run_id
            AND eval_case.dataset_id = experiment.dataset_id
            AND eval_case.evaluation_mode = NEW.evaluation_mode
    ) THEN RAISE(
        ABORT,
        'experiment result case must belong to the attempt dataset and mode'
    ) END;
"""

_EXPERIMENT_COMPARISON_RELATION_VALIDATION = """
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiments AS experiment
        JOIN eval_cases AS eval_case
            ON eval_case.eval_id = NEW.eval_id
        WHERE experiment.experiment_id = NEW.experiment_id
            AND eval_case.dataset_id = experiment.dataset_id
            AND eval_case.evaluation_mode = NEW.evaluation_mode
    ) THEN RAISE(
        ABORT,
        'experiment comparison case must belong to the attempt dataset and mode'
    ) END;
"""

EXPERIMENT_TRIGGER_STATEMENTS = (
    f"""
CREATE TRIGGER IF NOT EXISTS validate_experiment_result_relation_insert
BEFORE INSERT ON experiment_case_results
BEGIN
{_EXPERIMENT_RESULT_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_experiment_result_relation_update
BEFORE UPDATE ON experiment_case_results
BEGIN
{_EXPERIMENT_RESULT_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_experiment_comparison_relation_insert
BEFORE INSERT ON experiment_case_comparisons
BEGIN
{_EXPERIMENT_COMPARISON_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_experiment_comparison_relation_update
BEFORE UPDATE ON experiment_case_comparisons
BEGIN
{_EXPERIMENT_COMPARISON_RELATION_VALIDATION}
END
""",
)

REQUIRED_EVAL_CASE_COLUMNS = {
    "source_timestamp",
    "source_task_type",
    "source_response",
    "source_metadata_json",
}


def resolve_database_path() -> Path:
    """Resolve the configured database path against the current directory."""
    configured_path = os.environ.get(DATABASE_PATH_ENV)
    path = (
        Path(configured_path)
        if configured_path and configured_path.strip()
        else DEFAULT_DATABASE_PATH
    )
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a database connection and ensure its schema exists."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.row_factory = sqlite3.Row
        # Schema migrations may rebuild a referenced table. A new sqlite3
        # connection starts with foreign keys disabled, so keep them disabled
        # only for this transaction and verify all relationships before commit.
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_eval_schema_compatible(connection)
        _migrate_experiment_schema(connection)
        for statement in (
            *EVAL_CASE_TRIGGER_STATEMENTS,
            *EXPERIMENT_TRIGGER_STATEMENTS,
        ):
            connection.execute(statement)
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            first = foreign_key_errors[0]
            raise sqlite3.DatabaseError(
                "database schema migration failed foreign-key validation: "
                f"table={first[0]}, rowid={first[1]}, parent={first[2]}"
            )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = ON")
    except BaseException:
        try:
            connection.rollback()
        except BaseException:
            pass
        try:
            connection.close()
        except BaseException:
            pass
        raise
    return connection


def timestamp_to_text(timestamp: datetime) -> str:
    """Convert an aware timestamp to canonical UTC text."""
    return (
        timestamp.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def insert_trace(connection: sqlite3.Connection, trace: Trace) -> bool:
    """Insert a trace, returning whether a new row was stored."""
    cursor = connection.execute(
        """
        INSERT INTO traces (
            trace_id,
            timestamp,
            task_type,
            prompt,
            response,
            context_json,
            metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trace_id) DO NOTHING
        """,
        (
            trace.trace_id,
            timestamp_to_text(trace.timestamp),
            trace.task_type,
            trace.prompt,
            trace.response,
            _encode_object(trace.context),
            _encode_object(trace.metadata),
        ),
    )
    return cursor.rowcount == 1


def list_traces(connection: sqlite3.Connection) -> list[Trace]:
    """Return all stored traces in deterministic newest-first order."""
    rows = connection.execute(
        """
        SELECT
            trace_id,
            timestamp,
            task_type,
            prompt,
            response,
            context_json,
            metadata_json
        FROM traces
        ORDER BY timestamp DESC, trace_id ASC
        """
    ).fetchall()
    return [
        Trace.model_validate(
            {
                "trace_id": row["trace_id"],
                "timestamp": row["timestamp"],
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "context": _decode_object(row["context_json"]),
                "metadata": _decode_object(row["metadata_json"]),
            }
        )
        for row in rows
    ]


def get_trace(connection: sqlite3.Connection, trace_id: str) -> Trace | None:
    """Return one trace by identifier, or ``None`` when it is absent."""
    row = connection.execute(
        """
        SELECT
            trace_id,
            timestamp,
            task_type,
            prompt,
            response,
            context_json,
            metadata_json
        FROM traces
        WHERE trace_id = ?
        """,
        (trace_id,),
    ).fetchone()
    if row is None:
        return None
    return Trace.model_validate(
        {
            "trace_id": row["trace_id"],
            "timestamp": row["timestamp"],
            "task_type": row["task_type"],
            "prompt": row["prompt"],
            "response": row["response"],
            "context": _decode_object(row["context_json"]),
            "metadata": _decode_object(row["metadata_json"]),
        }
    )


def insert_eval_dataset(connection: sqlite3.Connection, dataset: EvalDataset) -> bool:
    """Insert a dataset, returning whether its name/version was new."""
    cursor = connection.execute(
        """
        INSERT INTO eval_datasets (
            dataset_id,
            name,
            version,
            description,
            created_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name, version) DO NOTHING
        """,
        (
            dataset.dataset_id,
            dataset.name,
            dataset.version,
            dataset.description,
            timestamp_to_text(dataset.created_at),
        ),
    )
    return cursor.rowcount == 1


def get_eval_dataset(
    connection: sqlite3.Connection, name: str, version: str
) -> EvalDataset | None:
    """Return a dataset by its human-readable name and version."""
    row = connection.execute(
        """
        SELECT dataset_id, name, version, description, created_at
        FROM eval_datasets
        WHERE name = ? AND version = ?
        """,
        (name.strip(), version.strip()),
    ).fetchone()
    return _dataset_from_row(row) if row is not None else None


def list_eval_datasets(
    connection: sqlite3.Connection,
) -> list[tuple[EvalDataset, int]]:
    """Return datasets and their case counts in deterministic order."""
    rows = connection.execute(
        """
        SELECT
            d.dataset_id,
            d.name,
            d.version,
            d.description,
            d.created_at,
            COUNT(c.eval_id) AS case_count
        FROM eval_datasets AS d
        LEFT JOIN eval_cases AS c ON c.dataset_id = d.dataset_id
        GROUP BY d.dataset_id
        ORDER BY d.name ASC, d.version ASC
        """
    ).fetchall()
    return [(_dataset_from_row(row), int(row["case_count"])) for row in rows]


def insert_eval_case(connection: sqlite3.Connection, case: EvalCase) -> bool:
    """Insert a case, returning whether the trace membership was new."""
    cursor = connection.execute(
        """
        INSERT INTO eval_cases (
            eval_id,
            dataset_id,
            source_trace_id,
            source_timestamp,
            source_task_type,
            source_response,
            source_metadata_json,
            input,
            context_json,
            evaluation_mode,
            reference_answer,
            rubric_json,
            scorers_json,
            priority,
            review_status,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_id, source_trace_id) DO NOTHING
        """,
        (
            case.eval_id,
            case.dataset_id,
            case.source_trace_id,
            timestamp_to_text(case.source_timestamp),
            case.source_task_type,
            case.source_response,
            _encode_object(case.source_metadata),
            case.input,
            _encode_object(case.context),
            case.evaluation_mode.value,
            case.reference_answer,
            _encode_json(case.rubric),
            _encode_json([scorer.model_dump(mode="json") for scorer in case.scorers]),
            case.priority.value,
            case.review_status.value,
            timestamp_to_text(case.created_at),
        ),
    )
    return cursor.rowcount == 1


def list_eval_cases(connection: sqlite3.Connection, dataset_id: str) -> list[EvalCase]:
    """Return a dataset's evaluation cases in stable identity order."""
    rows = connection.execute(
        """
        SELECT
            eval_id,
            dataset_id,
            source_trace_id,
            source_timestamp,
            source_task_type,
            source_response,
            source_metadata_json,
            input,
            context_json,
            evaluation_mode,
            reference_answer,
            rubric_json,
            scorers_json,
            priority,
            review_status,
            created_at
        FROM eval_cases
        WHERE dataset_id = ?
        ORDER BY eval_id ASC
        """,
        (dataset_id,),
    ).fetchall()
    return [_case_from_row(row) for row in rows]


def _encode_object(value: dict[str, Any]) -> str:
    return _encode_json(value)


def _encode_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("stored trace data is not a JSON object")
    return cast(dict[str, Any], decoded)


def _decode_list(value: str) -> list[Any]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("stored evaluation data is not a JSON array")
    return decoded


def _dataset_from_row(row: sqlite3.Row) -> EvalDataset:
    return EvalDataset.model_validate(
        {
            "dataset_id": row["dataset_id"],
            "name": row["name"],
            "version": row["version"],
            "description": row["description"],
            "created_at": row["created_at"],
        }
    )


def _case_from_row(row: sqlite3.Row) -> EvalCase:
    return EvalCase.model_validate(
        {
            "eval_id": row["eval_id"],
            "dataset_id": row["dataset_id"],
            "source_trace_id": row["source_trace_id"],
            "source_timestamp": row["source_timestamp"],
            "source_task_type": row["source_task_type"],
            "source_response": row["source_response"],
            "source_metadata": _decode_object(row["source_metadata_json"]),
            "input": row["input"],
            "context": _decode_object(row["context_json"]),
            "evaluation_mode": row["evaluation_mode"],
            "reference_answer": row["reference_answer"],
            "rubric": _decode_list(row["rubric_json"]),
            "scorers": [
                ScorerConfig.model_validate(scorer)
                for scorer in _decode_list(row["scorers_json"])
            ],
            "priority": row["priority"],
            "review_status": row["review_status"],
            "created_at": row["created_at"],
        }
    )


def _ensure_eval_schema_compatible(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(eval_cases)").fetchall()
    }
    dataset_schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'eval_datasets'"
    ).fetchone()
    case_schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'eval_cases'"
    ).fetchone()
    required_dataset_constraint = f"trim(name, {_SQLITE_PYTHON_WHITESPACE})"
    if (
        not REQUIRED_EVAL_CASE_COLUMNS.issubset(columns)
        or dataset_schema is None
        or required_dataset_constraint not in str(dataset_schema["sql"])
        or case_schema is None
        or "source_response IS NULL" not in str(case_schema["sql"])
    ):
        raise sqlite3.DatabaseError(
            "evaluation dataset schema predates provenance snapshots or final "
            "integrity constraints; "
            "recreate this unmerged development database"
        )


def _migrate_experiment_schema(connection: sqlite3.Connection) -> None:
    """Upgrade the pre-local-provider experiment schema without losing rows."""
    run_schema_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' "
        "AND name = 'experiment_runs'"
    ).fetchone()
    if run_schema_row is None:
        raise sqlite3.DatabaseError("experiment_runs schema is missing")
    run_schema = str(run_schema_row["sql"])
    if "provider_name IN ('fixture', 'ollama')" not in run_schema:
        _rebuild_experiment_runs(connection)

    result_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(experiment_case_results)"
        ).fetchall()
    }
    if "generation_latency_ms" not in result_columns:
        connection.execute(
            """
            ALTER TABLE experiment_case_results
            ADD COLUMN generation_latency_ms REAL
                CHECK (generation_latency_ms >= 0.0)
            """
        )
    if "provider_metadata_json" not in result_columns:
        connection.execute(
            """
            ALTER TABLE experiment_case_results
            ADD COLUMN provider_metadata_json TEXT NOT NULL DEFAULT '{}'
                CHECK (
                    json_valid(provider_metadata_json)
                    AND json_type(provider_metadata_json) = 'object'
                )
            """
        )


def _rebuild_experiment_runs(connection: sqlite3.Connection) -> None:
    """Replace the fixture-only provider constraint while preserving runs."""
    # These child-table triggers query experiment_runs. SQLite validates their
    # bodies during the table swap, so recreate them after the migration.
    connection.execute(
        "DROP TRIGGER IF EXISTS validate_experiment_result_relation_insert"
    )
    connection.execute(
        "DROP TRIGGER IF EXISTS validate_experiment_result_relation_update"
    )
    run_statement = next(
        (
            statement
            for statement in SCHEMA_STATEMENTS
            if "CREATE TABLE IF NOT EXISTS experiment_runs (" in statement
        ),
        None,
    )
    if run_statement is None:
        raise sqlite3.DatabaseError("current experiment_runs schema is missing")
    migration_statement = run_statement.replace(
        "CREATE TABLE IF NOT EXISTS experiment_runs (",
        "CREATE TABLE experiment_runs_migration (",
        1,
    )
    connection.execute("DROP TABLE IF EXISTS experiment_runs_migration")
    connection.execute(migration_statement)
    connection.execute(
        """
        INSERT INTO experiment_runs_migration (
            run_id, experiment_id, role, provider_name,
            provider_config_json, status, error_message,
            started_at, completed_at
        )
        SELECT
            run_id, experiment_id, role, provider_name,
            provider_config_json, status, error_message,
            started_at, completed_at
        FROM experiment_runs
        """
    )
    connection.execute("DROP TABLE experiment_runs")
    connection.execute(
        "ALTER TABLE experiment_runs_migration RENAME TO experiment_runs"
    )
