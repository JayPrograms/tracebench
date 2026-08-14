"""SQLite persistence for traces."""

import json
import os
import sqlite3
from contextlib import closing
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
CREATE TABLE IF NOT EXISTS trace_clustering_runs (
    clustering_run_id TEXT PRIMARY KEY
        CHECK (clustering_run_id GLOB 'cluster_run_[0-9a-f]*'
            AND clustering_run_id NOT GLOB 'cluster_run_*[^0-9a-f]*'
            AND length(clustering_run_id) = 44),
    name TEXT NOT NULL UNIQUE
        CHECK (length(name) > 0 AND name = trim(name, {_SQLITE_PYTHON_WHITESPACE})),
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    configuration_hash TEXT NOT NULL
        CHECK (length(configuration_hash) = 64
            AND configuration_hash NOT GLOB '*[^0-9a-f]*'),
    configuration_json TEXT NOT NULL
        CHECK (json_valid(configuration_json)
            AND json_type(configuration_json) = 'object'),
    source_manifest_hash TEXT NOT NULL
        CHECK (length(source_manifest_hash) = 64
            AND source_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    trace_count INTEGER NOT NULL CHECK (trace_count > 0),
    feature_count INTEGER NOT NULL CHECK (feature_count > 0),
    cluster_count INTEGER NOT NULL CHECK (cluster_count > 0),
    inertia REAL NOT NULL CHECK (inertia >= 0.0 AND inertia < 1.0e999),
    created_at TEXT NOT NULL
)
""",
    """
CREATE TABLE IF NOT EXISTS trace_cluster_assignments (
    clustering_run_id TEXT NOT NULL
        REFERENCES trace_clustering_runs(clustering_run_id) ON DELETE RESTRICT,
    trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE RESTRICT,
    document_index INTEGER NOT NULL CHECK (document_index >= 0),
    source_timestamp TEXT NOT NULL,
    source_trace_hash TEXT NOT NULL
        CHECK (length(source_trace_hash) = 64
            AND source_trace_hash NOT GLOB '*[^0-9a-f]*'),
    document_hash TEXT NOT NULL
        CHECK (length(document_hash) = 64
            AND document_hash NOT GLOB '*[^0-9a-f]*'),
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    PRIMARY KEY (clustering_run_id, trace_id),
    UNIQUE (clustering_run_id, document_index)
)
""",
    """
CREATE INDEX IF NOT EXISTS idx_trace_cluster_assignments_cluster
ON trace_cluster_assignments (clustering_run_id, cluster_number, trace_id)
""",
    """
CREATE TABLE IF NOT EXISTS trace_cluster_labels (
    clustering_run_id TEXT NOT NULL
        REFERENCES trace_clustering_runs(clustering_run_id) ON DELETE RESTRICT,
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    label TEXT,
    label_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (clustering_run_id, cluster_number),
    CHECK ((label IS NULL AND label_key IS NULL)
        OR (length(label) > 0 AND length(label_key) > 0))
)
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS idx_trace_cluster_labels_key
ON trace_cluster_labels (clustering_run_id, label_key)
WHERE label_key IS NOT NULL
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
CREATE TABLE IF NOT EXISTS experiment_judges (
    experiment_id TEXT PRIMARY KEY
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    provider_name TEXT NOT NULL
        CHECK (provider_name IN ('fixture', 'ollama')),
    provider_config_json TEXT NOT NULL
        CHECK (
            json_valid(provider_config_json)
            AND json_type(provider_config_json) = 'object'
        ),
    prompt_version TEXT NOT NULL CHECK (length(trim(prompt_version)) > 0),
    prompt_hash TEXT NOT NULL
        CHECK (length(prompt_hash) = 64 AND prompt_hash NOT GLOB '*[^0-9a-f]*'),
    retry_prompt_hash TEXT NOT NULL
        CHECK (
            length(retry_prompt_hash) = 64
            AND retry_prompt_hash NOT GLOB '*[^0-9a-f]*'
        ),
    response_schema_version INTEGER NOT NULL
        CHECK (response_schema_version = 1),
    max_malformed_retries INTEGER NOT NULL
        CHECK (max_malformed_retries = 1),
    confidence_threshold REAL NOT NULL
        CHECK (confidence_threshold >= 0.0 AND confidence_threshold <= 1.0)
)
""",
    """
CREATE TABLE IF NOT EXISTS judge_result_cache (
    cache_key TEXT PRIMARY KEY
        CHECK (
            length(cache_key) = 64
            AND cache_key NOT GLOB '*[^0-9a-f]*'
        ),
    key_version INTEGER NOT NULL CHECK (key_version = 1),
    identity_json TEXT NOT NULL
        CHECK (
            json_valid(identity_json)
            AND json_type(identity_json) = 'object'
        ),
    response_schema_version INTEGER NOT NULL
        CHECK (response_schema_version = 1),
    response_json TEXT NOT NULL
        CHECK (
            json_valid(response_json)
            AND json_type(response_json) = 'object'
        ),
    response_hash TEXT NOT NULL
        CHECK (
            length(response_hash) = 64
            AND response_hash NOT GLOB '*[^0-9a-f]*'
        ),
    raw_output TEXT NOT NULL CHECK (length(raw_output) > 0),
    created_at TEXT NOT NULL,
    UNIQUE (key_version, identity_json)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_case_results (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL
        REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    evaluation_mode TEXT NOT NULL
        CHECK (evaluation_mode IN ('deterministic', 'reference', 'rubric')),
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
CREATE TABLE IF NOT EXISTS experiment_judge_attempts (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL
        REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    attempt_number INTEGER NOT NULL CHECK (attempt_number IN (1, 2)),
    request_hash TEXT NOT NULL
        CHECK (length(request_hash) = 64 AND request_hash NOT GLOB '*[^0-9a-f]*'),
    raw_output TEXT NOT NULL,
    parse_status TEXT NOT NULL CHECK (parse_status IN ('parsed', 'malformed')),
    validation_error TEXT,
    latency_ms REAL NOT NULL CHECK (latency_ms >= 0.0),
    provider_metadata_json TEXT NOT NULL
        CHECK (
            json_valid(provider_metadata_json)
            AND json_type(provider_metadata_json) = 'object'
        ),
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, eval_id, attempt_number),
    CHECK (
        (parse_status = 'parsed' AND validation_error IS NULL)
        OR (
            parse_status = 'malformed'
            AND length(trim(validation_error)) > 0
        )
    )
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_judge_cache_lookups (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    eval_id TEXT NOT NULL
        REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    cache_key TEXT,
    cache_status TEXT NOT NULL
        CHECK (cache_status IN ('hit', 'miss', 'not_recorded')),
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, eval_id),
    CHECK (
        (
            cache_status = 'not_recorded'
            AND cache_key IS NULL
        )
        OR (
            cache_status IN ('hit', 'miss')
            AND length(cache_key) = 64
            AND cache_key NOT GLOB '*[^0-9a-f]*'
        )
    )
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
CREATE TABLE IF NOT EXISTS experiment_judge_results (
    run_id TEXT NOT NULL,
    eval_id TEXT NOT NULL,
    final_attempt_number INTEGER CHECK (final_attempt_number IN (1, 2)),
    response_schema_version INTEGER NOT NULL CHECK (response_schema_version = 1),
    overall_score REAL NOT NULL CHECK (overall_score >= 0.0 AND overall_score <= 1.0),
    overall_passed INTEGER NOT NULL CHECK (overall_passed IN (0, 1)),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    confidence_threshold REAL NOT NULL
        CHECK (confidence_threshold >= 0.0 AND confidence_threshold <= 1.0),
    below_confidence_threshold INTEGER NOT NULL
        CHECK (below_confidence_threshold IN (0, 1)),
    cache_hit INTEGER CHECK (cache_hit IN (0, 1)),
    critical_priority_failure INTEGER NOT NULL
        CHECK (critical_priority_failure IN (0, 1)),
    review_status TEXT NOT NULL
        CHECK (
            review_status IN ('not_required', 'needs_review', 'reviewed')
        ),
    PRIMARY KEY (run_id, eval_id),
    FOREIGN KEY (run_id, eval_id)
        REFERENCES experiment_case_results(run_id, eval_id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, eval_id, final_attempt_number)
        REFERENCES experiment_judge_attempts(
            run_id, eval_id, attempt_number
        ) ON DELETE RESTRICT,
    CHECK (
        below_confidence_threshold = (confidence < confidence_threshold)
    ),
    CHECK (
        (
            cache_hit = 1
            AND final_attempt_number IS NULL
        )
        OR (
            (cache_hit = 0 OR cache_hit IS NULL)
            AND final_attempt_number IN (1, 2)
        )
    ),
    CHECK (
        (
            review_status = 'not_required'
            AND below_confidence_threshold = 0
            AND critical_priority_failure = 0
        )
        OR (
            review_status IN ('needs_review', 'reviewed')
            AND (
                below_confidence_threshold = 1
                OR critical_priority_failure = 1
            )
        )
    )
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_run_aggregates (
    run_id TEXT NOT NULL
        REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    scope TEXT NOT NULL
        CHECK (scope IN ('global', 'deterministic', 'reference', 'rubric')),
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
        CHECK (evaluation_mode IN ('deterministic', 'reference', 'rubric')),
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
        CHECK (scope IN ('global', 'deterministic', 'reference', 'rubric')),
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
        CHECK (scope IN ('global', 'deterministic', 'reference', 'rubric')),
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

CACHE_TRIGGER_STATEMENTS = (
    """
CREATE TRIGGER IF NOT EXISTS prevent_judge_cache_update
BEFORE UPDATE ON judge_result_cache
BEGIN
    SELECT RAISE(ABORT, 'judge result cache entries are immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_judge_cache_delete
BEFORE DELETE ON judge_result_cache
BEGIN
    SELECT RAISE(ABORT, 'judge result cache entries are immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_judge_cache_lookup_update
BEFORE UPDATE ON experiment_judge_cache_lookups
BEGIN
    SELECT RAISE(ABORT, 'judge cache lookup records are immutable');
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

_JUDGE_ATTEMPT_RELATION_VALIDATION = """
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiment_runs AS run
        JOIN experiments AS experiment
            ON experiment.experiment_id = run.experiment_id
        JOIN experiment_judges AS judge
            ON judge.experiment_id = experiment.experiment_id
        JOIN eval_cases AS eval_case
            ON eval_case.eval_id = NEW.eval_id
        WHERE run.run_id = NEW.run_id
            AND eval_case.dataset_id = experiment.dataset_id
            AND eval_case.evaluation_mode = 'rubric'
    ) THEN RAISE(
        ABORT,
        'judge attempt case must be a rubric case in the attempt dataset'
    ) END;
"""

_JUDGE_CACHE_LOOKUP_RELATION_VALIDATION = """
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiment_runs AS run
        JOIN experiments AS experiment
            ON experiment.experiment_id = run.experiment_id
        JOIN experiment_judges AS judge
            ON judge.experiment_id = experiment.experiment_id
        JOIN eval_cases AS eval_case
            ON eval_case.eval_id = NEW.eval_id
        WHERE run.run_id = NEW.run_id
            AND eval_case.dataset_id = experiment.dataset_id
            AND eval_case.evaluation_mode = 'rubric'
    ) THEN RAISE(
        ABORT,
        'judge cache lookup case must be a rubric case in the attempt dataset'
    ) END;
    SELECT CASE WHEN (
        NEW.cache_status = 'hit'
        AND NOT EXISTS (
            SELECT 1
            FROM judge_result_cache AS cache
            WHERE cache.cache_key = NEW.cache_key
        )
    ) THEN RAISE(
        ABORT,
        'judge cache hit must reference an existing cache entry'
    ) END;
"""

_JUDGE_RESULT_RELATION_VALIDATION = """
    SELECT CASE WHEN NEW.final_attempt_number IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM experiment_judge_attempts AS attempt
        WHERE attempt.run_id = NEW.run_id
            AND attempt.eval_id = NEW.eval_id
            AND attempt.attempt_number = NEW.final_attempt_number
            AND attempt.parse_status = 'parsed'
    ) THEN RAISE(
        ABORT,
        'judge result must reference a parsed final attempt'
    ) END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiment_judge_cache_lookups AS lookup
        WHERE lookup.run_id = NEW.run_id
            AND lookup.eval_id = NEW.eval_id
            AND (
                (
                    lookup.cache_status = 'hit'
                    AND NEW.cache_hit = 1
                    AND NEW.final_attempt_number IS NULL
                    AND EXISTS (
                        SELECT 1
                        FROM judge_result_cache AS cache
                        WHERE cache.cache_key = lookup.cache_key
                    )
                )
                OR (
                    lookup.cache_status = 'miss'
                    AND NEW.cache_hit = 0
                    AND NEW.final_attempt_number IS NOT NULL
                    AND EXISTS (
                        SELECT 1
                        FROM judge_result_cache AS cache
                        WHERE cache.cache_key = lookup.cache_key
                    )
                )
                OR (
                    lookup.cache_status = 'not_recorded'
                    AND NEW.cache_hit IS NULL
                    AND NEW.final_attempt_number IS NOT NULL
                )
            )
    ) THEN RAISE(
        ABORT,
        'judge result cache metadata does not match its lookup'
    ) END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM eval_cases AS eval_case
        WHERE eval_case.eval_id = NEW.eval_id
            AND NEW.critical_priority_failure = CASE
                WHEN (
                    eval_case.priority = 'critical'
                    AND NEW.overall_passed = 0
                ) THEN 1
                ELSE 0
            END
    ) THEN RAISE(
        ABORT,
        'judge result critical-failure metadata is inconsistent'
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
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_attempt_relation_insert
BEFORE INSERT ON experiment_judge_attempts
BEGIN
{_JUDGE_ATTEMPT_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_attempt_relation_update
BEFORE UPDATE ON experiment_judge_attempts
BEGIN
{_JUDGE_ATTEMPT_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_cache_lookup_relation_insert
BEFORE INSERT ON experiment_judge_cache_lookups
BEGIN
{_JUDGE_CACHE_LOOKUP_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_cache_lookup_relation_update
BEFORE UPDATE ON experiment_judge_cache_lookups
BEGIN
{_JUDGE_CACHE_LOOKUP_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_result_relation_insert
BEFORE INSERT ON experiment_judge_results
BEGIN
{_JUDGE_RESULT_RELATION_VALIDATION}
END
""",
    f"""
CREATE TRIGGER IF NOT EXISTS validate_judge_result_relation_update
BEFORE UPDATE ON experiment_judge_results
BEGIN
{_JUDGE_RESULT_RELATION_VALIDATION}
END
""",
)

CLUSTERING_TRIGGER_STATEMENTS = (
    """
CREATE TRIGGER IF NOT EXISTS validate_trace_cluster_assignment_insert
BEFORE INSERT ON trace_cluster_assignments
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM trace_clustering_runs AS run
        WHERE run.clustering_run_id = NEW.clustering_run_id
          AND NEW.cluster_number < run.cluster_count
          AND NEW.document_index < run.trace_count
    ) THEN RAISE(ABORT, 'assignment cluster number is outside its run') END;
    SELECT CASE WHEN (
        SELECT COUNT(*) FROM trace_cluster_assignments AS assignment
        WHERE assignment.clustering_run_id = NEW.clustering_run_id
    ) >= (
        SELECT run.trace_count FROM trace_clustering_runs AS run
        WHERE run.clustering_run_id = NEW.clustering_run_id
    ) THEN RAISE(ABORT, 'cluster assignments are immutable') END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_trace_cluster_label_insert
BEFORE INSERT ON trace_cluster_labels
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM trace_clustering_runs AS run
        WHERE run.clustering_run_id = NEW.clustering_run_id
          AND NEW.cluster_number < run.cluster_count
    ) THEN RAISE(ABORT, 'label cluster number is outside its run') END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_trace_clustering_run_update
BEFORE UPDATE ON trace_clustering_runs
BEGIN SELECT RAISE(ABORT, 'clustering runs are immutable'); END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_trace_clustering_run_delete
BEFORE DELETE ON trace_clustering_runs
BEGIN SELECT RAISE(ABORT, 'clustering runs are immutable'); END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_trace_cluster_assignment_update
BEFORE UPDATE ON trace_cluster_assignments
BEGIN SELECT RAISE(ABORT, 'cluster assignments are immutable'); END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_trace_cluster_assignment_delete
BEFORE DELETE ON trace_cluster_assignments
BEGIN SELECT RAISE(ABORT, 'cluster assignments are immutable'); END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_trace_cluster_label_delete
BEFORE DELETE ON trace_cluster_labels
BEGIN SELECT RAISE(ABORT, 'cluster label rows cannot be deleted'); END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_trace_cluster_label_update
BEFORE UPDATE ON trace_cluster_labels
BEGIN
    SELECT CASE WHEN NEW.clustering_run_id != OLD.clustering_run_id
        OR NEW.cluster_number != OLD.cluster_number
        OR NEW.created_at != OLD.created_at
    THEN RAISE(ABORT, 'cluster label identity is immutable') END;
    SELECT CASE WHEN (NEW.label IS NULL) != (NEW.label_key IS NULL)
    THEN RAISE(ABORT, 'cluster label and key must change atomically') END;
    SELECT CASE WHEN
        (
            NEW.label IS NOT OLD.label
            OR NEW.label_key IS NOT OLD.label_key
        ) != (NEW.updated_at IS NOT OLD.updated_at)
    THEN RAISE(ABORT, 'cluster label, key, and timestamp must change atomically') END;
END
""",
)

_CLUSTERING_TABLE_NAMES = (
    "trace_clustering_runs",
    "trace_cluster_assignments",
    "trace_cluster_labels",
)

SchemaRows = tuple[tuple[object, ...], ...]
ClusteringTableMetadata = tuple[SchemaRows, SchemaRows, SchemaRows]
ClusteringSchemaSnapshot = tuple[
    tuple[tuple[str, str, str, str | None], ...],
    dict[str, ClusteringTableMetadata],
]

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
        connection.execute("PRAGMA recursive_triggers = ON")
        # Schema migrations may rebuild a referenced table. A new sqlite3
        # connection starts with foreign keys disabled, so keep them disabled
        # only for this transaction and verify all relationships before commit.
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN")
        _ensure_clustering_schema_compatible(connection)
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_eval_schema_compatible(connection)
        _migrate_experiment_schema(connection)
        for statement in (
            *EVAL_CASE_TRIGGER_STATEMENTS,
            *CACHE_TRIGGER_STATEMENTS,
            *EXPERIMENT_TRIGGER_STATEMENTS,
            *CLUSTERING_TRIGGER_STATEMENTS,
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


def _ensure_clustering_schema_compatible(connection: sqlite3.Connection) -> None:
    """Reject every partial, extra, or incompatible pre-existing B1 object."""
    actual_objects = _clustering_schema_objects(connection)
    if not actual_objects:
        return
    if _clustering_schema_snapshot(connection) == _EXPECTED_CLUSTERING_SCHEMA:
        return
    raise sqlite3.DatabaseError(
        "clustering schema is incomplete or incompatible; recreate the B1 "
        "clustering objects"
    )


def _build_expected_clustering_schema() -> ClusteringSchemaSnapshot:
    """Build the canonical structured B1 schema snapshot once at import time."""
    with closing(sqlite3.connect(":memory:")) as reference:
        reference.row_factory = sqlite3.Row
        for statement in SCHEMA_STATEMENTS:
            if any(name in statement for name in _CLUSTERING_TABLE_NAMES):
                reference.execute(statement)
        for statement in CLUSTERING_TRIGGER_STATEMENTS:
            reference.execute(statement)
        return _clustering_schema_snapshot(reference)


def _clustering_schema_snapshot(
    connection: sqlite3.Connection,
) -> ClusteringSchemaSnapshot:
    return (
        _clustering_schema_objects(connection),
        {
            table_name: (
                _pragma_rows(connection, "table_xinfo", table_name),
                _pragma_rows(connection, "foreign_key_list", table_name),
                _index_metadata(connection, table_name),
            )
            for table_name in _CLUSTERING_TABLE_NAMES
        },
    )


def _clustering_schema_objects(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str, str | None], ...]:
    """Return normalized definitions for all B1-owned schema objects."""
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE tbl_name IN (?, ?, ?) OR name GLOB 'trace_cluster*' "
        "ORDER BY type, name",
        _CLUSTERING_TABLE_NAMES,
    ).fetchall()
    return tuple(
        (
            str(row["type"]),
            str(row["name"]),
            str(row["tbl_name"]),
            _normalize_schema_sql(row["sql"]),
        )
        for row in rows
    )


def _normalize_schema_sql(value: object) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())


def _pragma_rows(
    connection: sqlite3.Connection, pragma: str, object_name: str
) -> tuple[tuple[object, ...], ...]:
    """Return complete ordered PRAGMA metadata for a trusted object name."""
    return tuple(
        tuple(row)
        for row in connection.execute(f'PRAGMA {pragma}("{object_name}")').fetchall()
    )


def _index_metadata(
    connection: sqlite3.Connection, table_name: str
) -> tuple[tuple[object, ...], ...]:
    """Return index uniqueness, origin, partiality, and ordered column metadata."""
    metadata: list[tuple[object, ...]] = []
    for row in connection.execute(f'PRAGMA index_list("{table_name}")').fetchall():
        index_name = str(row[1])
        metadata.append(
            (
                index_name,
                int(row[2]),
                str(row[3]),
                int(row[4]),
                _pragma_rows(connection, "index_xinfo", index_name),
            )
        )
    return tuple(sorted(metadata, key=lambda item: str(item[0])))


_EXPECTED_CLUSTERING_SCHEMA = _build_expected_clustering_schema()


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

    rubric_constrained_tables = (
        "experiment_case_results",
        "experiment_run_aggregates",
        "experiment_case_comparisons",
        "experiment_comparison_aggregates",
        "experiment_gate_violations",
    )
    tables_to_rebuild: list[str] = []
    for table_name in rubric_constrained_tables:
        schema_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if schema_row is None:
            raise sqlite3.DatabaseError(f"{table_name} schema is missing")
        if "'rubric'" not in str(schema_row["sql"]):
            tables_to_rebuild.append(table_name)
    if tables_to_rebuild:
        _rebuild_experiment_tables(connection, tables_to_rebuild)

    judge_result_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(experiment_judge_results)"
        ).fetchall()
    }
    if not {
        "cache_hit",
        "critical_priority_failure",
        "review_status",
    }.issubset(judge_result_columns):
        _migrate_judge_result_metadata(connection)


def _migrate_judge_result_metadata(connection: sqlite3.Connection) -> None:
    """Add A3 cache provenance and review state without losing A2 results."""
    for trigger_name in (
        "validate_judge_cache_lookup_relation_insert",
        "validate_judge_cache_lookup_relation_update",
        "validate_judge_result_relation_insert",
        "validate_judge_result_relation_update",
    ):
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
    connection.execute(
        """
        INSERT OR IGNORE INTO experiment_judge_cache_lookups (
            run_id, eval_id, cache_key, cache_status, created_at
        )
        SELECT
            result.run_id,
            result.eval_id,
            NULL,
            'not_recorded',
            COALESCE(attempt.created_at, case_result.created_at)
        FROM experiment_judge_results AS result
        JOIN experiment_case_results AS case_result
            ON case_result.run_id = result.run_id
            AND case_result.eval_id = result.eval_id
        LEFT JOIN experiment_judge_attempts AS attempt
            ON attempt.run_id = result.run_id
            AND attempt.eval_id = result.eval_id
            AND attempt.attempt_number = result.final_attempt_number
        """
    )
    marker = "CREATE TABLE IF NOT EXISTS experiment_judge_results ("
    current_statement = next(
        statement for statement in SCHEMA_STATEMENTS if marker in statement
    )
    migration_statement = current_statement.replace(
        marker,
        "CREATE TABLE experiment_judge_results_a3 (",
        1,
    )
    connection.execute("DROP TABLE IF EXISTS experiment_judge_results_a3")
    connection.execute(migration_statement)
    connection.execute(
        """
        INSERT INTO experiment_judge_results_a3 (
            run_id, eval_id, final_attempt_number,
            response_schema_version, overall_score, overall_passed,
            confidence, confidence_threshold, below_confidence_threshold,
            cache_hit, critical_priority_failure, review_status
        )
        SELECT
            result.run_id,
            result.eval_id,
            result.final_attempt_number,
            result.response_schema_version,
            result.overall_score,
            result.overall_passed,
            result.confidence,
            result.confidence_threshold,
            result.below_confidence_threshold,
            NULL,
            CASE
                WHEN eval_case.priority = 'critical'
                    AND result.overall_passed = 0
                THEN 1
                ELSE 0
            END,
            CASE
                WHEN result.below_confidence_threshold = 1
                    OR (
                        eval_case.priority = 'critical'
                        AND result.overall_passed = 0
                    )
                THEN 'needs_review'
                ELSE 'not_required'
            END
        FROM experiment_judge_results AS result
        JOIN eval_cases AS eval_case ON eval_case.eval_id = result.eval_id
        """
    )
    connection.execute("DROP TABLE experiment_judge_results")
    connection.execute(
        "ALTER TABLE experiment_judge_results_a3 RENAME TO experiment_judge_results"
    )


def _rebuild_experiment_tables(
    connection: sqlite3.Connection,
    table_names: list[str],
) -> None:
    """Replace constrained experiment tables while preserving their rows."""
    trigger_names = (
        "validate_experiment_result_relation_insert",
        "validate_experiment_result_relation_update",
        "validate_experiment_comparison_relation_insert",
        "validate_experiment_comparison_relation_update",
        "validate_judge_attempt_relation_insert",
        "validate_judge_attempt_relation_update",
        "validate_judge_cache_lookup_relation_insert",
        "validate_judge_cache_lookup_relation_update",
        "validate_judge_result_relation_insert",
        "validate_judge_result_relation_update",
    )
    for trigger_name in trigger_names:
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
    for table_name in table_names:
        marker = f"CREATE TABLE IF NOT EXISTS {table_name} ("
        current_statement = next(
            (statement for statement in SCHEMA_STATEMENTS if marker in statement),
            None,
        )
        if current_statement is None:
            raise sqlite3.DatabaseError(f"current {table_name} schema is missing")
        migration_name = f"{table_name}_migration"
        migration_statement = current_statement.replace(
            marker,
            f"CREATE TABLE {migration_name} (",
            1,
        )
        columns = [
            str(row["name"])
            for row in connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
        ]
        encoded_columns = ", ".join(f'"{column}"' for column in columns)
        connection.execute(f'DROP TABLE IF EXISTS "{migration_name}"')
        connection.execute(migration_statement)
        connection.execute(
            f'INSERT INTO "{migration_name}" ({encoded_columns}) '
            f'SELECT {encoded_columns} FROM "{table_name}"'
        )
        connection.execute(f'DROP TABLE "{table_name}"')
        connection.execute(f'ALTER TABLE "{migration_name}" RENAME TO "{table_name}"')


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
    connection.execute("DROP TRIGGER IF EXISTS validate_judge_attempt_relation_insert")
    connection.execute("DROP TRIGGER IF EXISTS validate_judge_attempt_relation_update")
    connection.execute(
        "DROP TRIGGER IF EXISTS validate_judge_cache_lookup_relation_insert"
    )
    connection.execute(
        "DROP TRIGGER IF EXISTS validate_judge_cache_lookup_relation_update"
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
