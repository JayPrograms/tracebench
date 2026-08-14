"""SQLite persistence for traces."""

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tracebench.models import (
    EvalCase,
    EvalDataset,
    ScorerConfig,
    SliceBuildSource,
    SliceCaseProvenance,
    Trace,
)

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
CREATE TABLE IF NOT EXISTS eval_dataset_slice_builds (
    dataset_id TEXT PRIMARY KEY
        REFERENCES eval_datasets(dataset_id) ON DELETE RESTRICT,
    clustering_run_id TEXT NOT NULL
        REFERENCES trace_clustering_runs(clustering_run_id) ON DELETE RESTRICT,
    clustering_run_name TEXT NOT NULL CHECK (length(trim(clustering_run_name)) > 0),
    clustering_schema_version INTEGER NOT NULL CHECK (clustering_schema_version = 1),
    clustering_configuration_hash TEXT NOT NULL
        CHECK (length(clustering_configuration_hash) = 64
            AND clustering_configuration_hash NOT GLOB '*[^0-9a-f]*'),
    clustering_source_manifest_hash TEXT NOT NULL
        CHECK (length(clustering_source_manifest_hash) = 64
            AND clustering_source_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    cluster_count INTEGER NOT NULL CHECK (cluster_count > 0),
    sampling_schema_version INTEGER NOT NULL CHECK (sampling_schema_version = 1),
    sampling_algorithm TEXT NOT NULL CHECK (sampling_algorithm = 'balanced-hash-v1'),
    requested_size INTEGER NOT NULL CHECK (requested_size > 0),
    sampled_size INTEGER NOT NULL CHECK (sampled_size = requested_size),
    eligible_trace_count INTEGER NOT NULL
        CHECK (eligible_trace_count >= sampled_size),
    slice_manifest_json TEXT NOT NULL
        CHECK (json_valid(slice_manifest_json)
            AND json_type(slice_manifest_json) = 'array'),
    slice_manifest_hash TEXT NOT NULL
        CHECK (length(slice_manifest_hash) = 64
            AND slice_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    built_at TEXT NOT NULL,
    UNIQUE (dataset_id, clustering_run_id)
)
""",
    """
CREATE TABLE IF NOT EXISTS eval_case_slice_provenance (
    eval_id TEXT PRIMARY KEY REFERENCES eval_cases(eval_id) ON DELETE RESTRICT,
    dataset_id TEXT NOT NULL,
    selector TEXT NOT NULL,
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    label_snapshot TEXT,
    label_key_snapshot TEXT,
    clustering_run_id TEXT NOT NULL,
    clustering_run_name TEXT NOT NULL CHECK (length(trim(clustering_run_name)) > 0),
    clustering_schema_version INTEGER NOT NULL CHECK (clustering_schema_version = 1),
    clustering_configuration_hash TEXT NOT NULL
        CHECK (length(clustering_configuration_hash) = 64
            AND clustering_configuration_hash NOT GLOB '*[^0-9a-f]*'),
    clustering_source_manifest_hash TEXT NOT NULL
        CHECK (length(clustering_source_manifest_hash) = 64
            AND clustering_source_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    cluster_count INTEGER NOT NULL CHECK (cluster_count > 0),
    source_trace_id TEXT NOT NULL,
    source_timestamp TEXT NOT NULL,
    source_trace_hash TEXT NOT NULL
        CHECK (length(source_trace_hash) = 64
            AND source_trace_hash NOT GLOB '*[^0-9a-f]*'),
    document_index INTEGER NOT NULL CHECK (document_index >= 0),
    document_hash TEXT NOT NULL
        CHECK (length(document_hash) = 64
            AND document_hash NOT GLOB '*[^0-9a-f]*'),
    sampling_schema_version INTEGER NOT NULL CHECK (sampling_schema_version = 1),
    sampling_algorithm TEXT NOT NULL CHECK (sampling_algorithm = 'balanced-hash-v1'),
    requested_size INTEGER NOT NULL CHECK (requested_size > 0),
    sampled_size INTEGER NOT NULL CHECK (sampled_size = requested_size),
    eligible_trace_count INTEGER NOT NULL CHECK (eligible_trace_count >= sampled_size),
    slice_availability INTEGER NOT NULL CHECK (slice_availability > 0),
    slice_quota INTEGER NOT NULL
        CHECK (slice_quota > 0 AND slice_quota <= slice_availability),
    rank_within_slice INTEGER NOT NULL
        CHECK (rank_within_slice >= 0 AND rank_within_slice < slice_availability),
    selection_key TEXT NOT NULL
        CHECK (length(selection_key) = 64
            AND selection_key NOT GLOB '*[^0-9a-f]*'),
    allocation_key TEXT NOT NULL
        CHECK (length(allocation_key) = 64
            AND allocation_key NOT GLOB '*[^0-9a-f]*'),
    slice_manifest_hash TEXT NOT NULL
        CHECK (length(slice_manifest_hash) = 64
            AND slice_manifest_hash NOT GLOB '*[^0-9a-f]*'),
    CHECK (selector = 'cluster-' || CAST(cluster_number AS TEXT)),
    CHECK ((label_snapshot IS NULL AND label_key_snapshot IS NULL)
        OR (length(label_snapshot) > 0 AND length(label_key_snapshot) > 0)),
    FOREIGN KEY (dataset_id, clustering_run_id)
        REFERENCES eval_dataset_slice_builds(dataset_id, clustering_run_id)
        ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (clustering_run_id, source_trace_id)
        REFERENCES trace_cluster_assignments(clustering_run_id, trace_id)
        ON DELETE RESTRICT
)
""",
    """
CREATE INDEX IF NOT EXISTS idx_eval_case_slice_dataset_cluster
ON eval_case_slice_provenance (dataset_id, cluster_number, eval_id)
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
CREATE TABLE IF NOT EXISTS experiment_run_slice_aggregates (
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    selector TEXT NOT NULL,
    label_snapshot TEXT,
    case_count INTEGER NOT NULL CHECK (case_count > 0),
    passed_count INTEGER NOT NULL CHECK (passed_count >= 0),
    failed_count INTEGER NOT NULL CHECK (failed_count >= 0),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    pass_rate REAL NOT NULL CHECK (pass_rate >= 0.0 AND pass_rate <= 1.0),
    PRIMARY KEY (run_id, cluster_number),
    CHECK (selector = 'cluster-' || CAST(cluster_number AS TEXT)),
    CHECK (passed_count + failed_count = case_count)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_comparison_slice_aggregates (
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    cluster_number INTEGER NOT NULL CHECK (cluster_number >= 0),
    selector TEXT NOT NULL,
    label_snapshot TEXT,
    case_count INTEGER NOT NULL CHECK (case_count > 0),
    baseline_score REAL NOT NULL
        CHECK (baseline_score >= 0.0 AND baseline_score <= 1.0),
    candidate_score REAL NOT NULL
        CHECK (candidate_score >= 0.0 AND candidate_score <= 1.0),
    score_delta REAL NOT NULL CHECK (score_delta >= -1.0 AND score_delta <= 1.0),
    baseline_pass_rate REAL NOT NULL
        CHECK (baseline_pass_rate >= 0.0 AND baseline_pass_rate <= 1.0),
    candidate_pass_rate REAL NOT NULL
        CHECK (candidate_pass_rate >= 0.0 AND candidate_pass_rate <= 1.0),
    newly_passed_count INTEGER NOT NULL CHECK (newly_passed_count >= 0),
    newly_failed_count INTEGER NOT NULL CHECK (newly_failed_count >= 0),
    PRIMARY KEY (experiment_id, cluster_number),
    CHECK (selector = 'cluster-' || CAST(cluster_number AS TEXT)),
    CHECK (newly_passed_count <= case_count),
    CHECK (newly_failed_count <= case_count)
)
""",
    """
CREATE TABLE IF NOT EXISTS experiment_gate_violations (
    experiment_id TEXT NOT NULL
        REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    violation_index INTEGER NOT NULL CHECK (violation_index >= 0),
    scope TEXT NOT NULL CHECK (
        scope IN ('global', 'deterministic', 'reference', 'rubric')
        OR scope GLOB 'cluster-[0-9]*'
    ),
    scope_kind TEXT CHECK (scope_kind IN ('global', 'mode', 'slice')),
    cluster_number INTEGER CHECK (cluster_number >= 0),
    label_snapshot TEXT,
    metric TEXT NOT NULL CHECK (metric IN ('score_drop', 'new_failures')),
    actual REAL NOT NULL CHECK (actual >= 0.0),
    allowed REAL NOT NULL CHECK (allowed >= 0.0),
    message TEXT NOT NULL CHECK (length(trim(message)) > 0),
    PRIMARY KEY (experiment_id, violation_index),
    CHECK (
        (scope_kind IS NULL AND cluster_number IS NULL AND label_snapshot IS NULL)
        OR (scope_kind IN ('global', 'mode') AND cluster_number IS NULL)
        OR (
            scope_kind = 'slice'
            AND cluster_number IS NOT NULL
            AND scope = 'cluster-' || CAST(cluster_number AS TEXT)
        )
    )
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

B2_TRIGGER_STATEMENTS = (
    """
CREATE TRIGGER IF NOT EXISTS prevent_slice_built_dataset_update
BEFORE UPDATE ON eval_datasets
WHEN EXISTS (
    SELECT 1 FROM eval_dataset_slice_builds
    WHERE dataset_id = OLD.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'slice-built datasets are sealed');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_slice_built_dataset_delete
BEFORE DELETE ON eval_datasets
WHEN EXISTS (
    SELECT 1 FROM eval_dataset_slice_builds
    WHERE dataset_id = OLD.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'slice-built datasets are sealed');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_case_insert_into_slice_build
BEFORE INSERT ON eval_cases
WHEN EXISTS (
    SELECT 1 FROM eval_dataset_slice_builds
    WHERE dataset_id = NEW.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'slice-built datasets are sealed');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_slice_built_case_update
BEFORE UPDATE ON eval_cases
WHEN EXISTS (
    SELECT 1 FROM eval_dataset_slice_builds
    WHERE dataset_id = OLD.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'slice-built datasets are sealed');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_slice_built_case_delete
BEFORE DELETE ON eval_cases
WHEN EXISTS (
    SELECT 1 FROM eval_dataset_slice_builds
    WHERE dataset_id = OLD.dataset_id
)
BEGIN
    SELECT RAISE(ABORT, 'slice-built datasets are sealed');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_case_slice_provenance_insert
BEFORE INSERT ON eval_case_slice_provenance
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM eval_cases AS case_row
        JOIN trace_clustering_runs AS run
          ON run.clustering_run_id = NEW.clustering_run_id
        JOIN trace_cluster_assignments AS assignment
          ON assignment.clustering_run_id = NEW.clustering_run_id
         AND assignment.trace_id = NEW.source_trace_id
        JOIN trace_cluster_labels AS label
          ON label.clustering_run_id = NEW.clustering_run_id
         AND label.cluster_number = NEW.cluster_number
        WHERE case_row.eval_id = NEW.eval_id
          AND case_row.dataset_id = NEW.dataset_id
          AND case_row.source_trace_id = NEW.source_trace_id
          AND case_row.source_timestamp = NEW.source_timestamp
          AND case_row.evaluation_mode = 'reference'
          AND case_row.source_response IS NOT NULL
          AND case_row.reference_answer = case_row.source_response
          AND json_array_length(case_row.rubric_json) = 0
          AND json_array_length(case_row.scorers_json) = 0
          AND case_row.priority = 'medium'
          AND case_row.review_status = 'draft'
          AND run.name = NEW.clustering_run_name
          AND run.schema_version = NEW.clustering_schema_version
          AND run.configuration_hash = NEW.clustering_configuration_hash
          AND run.source_manifest_hash = NEW.clustering_source_manifest_hash
          AND run.cluster_count = NEW.cluster_count
          AND assignment.cluster_number = NEW.cluster_number
          AND assignment.source_timestamp = NEW.source_timestamp
          AND assignment.source_trace_hash = NEW.source_trace_hash
          AND assignment.document_index = NEW.document_index
          AND assignment.document_hash = NEW.document_hash
          AND label.label IS NEW.label_snapshot
          AND label.label_key IS NEW.label_key_snapshot
    ) THEN RAISE(ABORT, 'slice provenance does not match its immutable source') END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_case_slice_provenance_update
BEFORE UPDATE ON eval_case_slice_provenance
BEGIN
    SELECT RAISE(ABORT, 'case slice provenance is immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_case_slice_provenance_delete
BEFORE DELETE ON eval_case_slice_provenance
BEGIN
    SELECT RAISE(ABORT, 'case slice provenance is immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_dataset_slice_build_insert
BEFORE INSERT ON eval_dataset_slice_builds
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM trace_clustering_runs AS run
        WHERE run.clustering_run_id = NEW.clustering_run_id
          AND run.name = NEW.clustering_run_name
          AND run.schema_version = NEW.clustering_schema_version
          AND run.configuration_hash = NEW.clustering_configuration_hash
          AND run.source_manifest_hash = NEW.clustering_source_manifest_hash
          AND run.cluster_count = NEW.cluster_count
    ) THEN RAISE(ABORT, 'dataset slice build does not match its clustering run') END;
    SELECT CASE WHEN (
        SELECT COUNT(*) FROM eval_cases WHERE dataset_id = NEW.dataset_id
    ) != NEW.sampled_size THEN RAISE(
        ABORT, 'dataset slice build case count does not match sampled size'
    ) END;
    SELECT CASE WHEN (
        SELECT COUNT(*) FROM eval_case_slice_provenance
        WHERE dataset_id = NEW.dataset_id
          AND clustering_run_id = NEW.clustering_run_id
          AND slice_manifest_hash = NEW.slice_manifest_hash
    ) != NEW.sampled_size THEN RAISE(
        ABORT, 'dataset slice build provenance coverage is incomplete'
    ) END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_dataset_slice_build_update
BEFORE UPDATE ON eval_dataset_slice_builds
BEGIN
    SELECT RAISE(ABORT, 'dataset slice builds are immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS prevent_dataset_slice_build_delete
BEFORE DELETE ON eval_dataset_slice_builds
BEGIN
    SELECT RAISE(ABORT, 'dataset slice builds are immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_comparison_slice_aggregate_insert
BEFORE INSERT ON experiment_comparison_slice_aggregates
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM experiments AS experiment
        JOIN eval_case_slice_provenance AS provenance
          ON provenance.dataset_id = experiment.dataset_id
         AND provenance.cluster_number = NEW.cluster_number
        WHERE experiment.experiment_id = NEW.experiment_id
          AND provenance.selector = NEW.selector
          AND provenance.label_snapshot IS NEW.label_snapshot
        GROUP BY experiment.experiment_id
        HAVING COUNT(provenance.eval_id) = NEW.case_count
    ) THEN RAISE(
        ABORT, 'comparison slice aggregate does not match dataset membership'
    ) END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_comparison_slice_aggregate_update
BEFORE UPDATE ON experiment_comparison_slice_aggregates
BEGIN
    SELECT RAISE(ABORT, 'comparison slice aggregates are immutable');
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_gate_violation_scope_insert
BEFORE INSERT ON experiment_gate_violations
BEGIN
    SELECT CASE WHEN NEW.scope_kind = 'global' AND NEW.scope != 'global'
        THEN RAISE(ABORT, 'global gate violation has an invalid scope') END;
    SELECT CASE WHEN NEW.scope_kind = 'mode'
        AND NEW.scope NOT IN ('deterministic', 'reference', 'rubric')
        THEN RAISE(ABORT, 'mode gate violation has an invalid scope') END;
    SELECT CASE WHEN NEW.scope_kind = 'slice' AND NOT EXISTS (
        SELECT 1
        FROM experiments AS experiment
        JOIN eval_case_slice_provenance AS provenance
          ON provenance.dataset_id = experiment.dataset_id
         AND provenance.cluster_number = NEW.cluster_number
        WHERE experiment.experiment_id = NEW.experiment_id
          AND provenance.selector = NEW.scope
          AND provenance.label_snapshot IS NEW.label_snapshot
    ) THEN RAISE(ABORT, 'slice gate violation is not represented by its dataset') END;
END
""",
    """
CREATE TRIGGER IF NOT EXISTS validate_gate_violation_scope_update
BEFORE UPDATE ON experiment_gate_violations
BEGIN
    SELECT RAISE(ABORT, 'gate violations are immutable');
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

_B2_TABLE_NAMES = (
    "eval_dataset_slice_builds",
    "eval_case_slice_provenance",
    "experiment_run_slice_aggregates",
    "experiment_comparison_slice_aggregates",
)

_B2_TRIGGER_NAMES = tuple(
    statement.split("CREATE TRIGGER IF NOT EXISTS ", 1)[1].splitlines()[0]
    for statement in B2_TRIGGER_STATEMENTS
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
        _ensure_b2_schema_absent_or_complete(connection)
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_eval_schema_compatible(connection)
        _migrate_experiment_schema(connection)
        for statement in (
            *EVAL_CASE_TRIGGER_STATEMENTS,
            *B2_TRIGGER_STATEMENTS,
            *CACHE_TRIGGER_STATEMENTS,
            *EXPERIMENT_TRIGGER_STATEMENTS,
            *CLUSTERING_TRIGGER_STATEMENTS,
        ):
            connection.execute(statement)
        _ensure_b2_schema_compatible(connection)
        _ensure_b2_data_compatible(connection)
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


def _ensure_b2_schema_absent_or_complete(connection: sqlite3.Connection) -> None:
    """Reject interrupted or hand-authored partial B2 schemas before creation."""
    migratable_trigger_names = {
        "validate_gate_violation_scope_insert",
        "validate_gate_violation_scope_update",
    }
    expected = {
        *_B2_TABLE_NAMES,
        *(name for name in _B2_TRIGGER_NAMES if name not in migratable_trigger_names),
        "idx_eval_case_slice_dataset_cluster",
    }
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE name IN ("
        + ",".join("?" for _ in expected)
        + ")",
        tuple(sorted(expected)),
    ).fetchall()
    actual = {str(row["name"]) for row in rows}
    if actual and actual != expected:
        raise sqlite3.DatabaseError(
            "slice-aware B2 schema is partial or incompatible; recreate the B2 "
            "schema objects"
        )


def _ensure_b2_schema_compatible(connection: sqlite3.Connection) -> None:
    """Require exact B2-owned SQL plus its reconstructable model columns."""
    expected_sql: dict[str, str | None] = {}
    for statement in SCHEMA_STATEMENTS:
        normalized = statement.lstrip()
        for table_name in _B2_TABLE_NAMES:
            marker = f"CREATE TABLE IF NOT EXISTS {table_name} "
            if normalized.startswith(marker):
                expected_sql[table_name] = _normalize_schema_sql(
                    normalized.replace(
                        f"CREATE TABLE IF NOT EXISTS {table_name}",
                        f"CREATE TABLE {table_name}",
                        1,
                    )
                )
        if normalized.startswith(
            "CREATE INDEX IF NOT EXISTS idx_eval_case_slice_dataset_cluster"
        ):
            expected_sql["idx_eval_case_slice_dataset_cluster"] = _normalize_schema_sql(
                normalized.replace("CREATE INDEX IF NOT EXISTS", "CREATE INDEX", 1)
            )
    for statement in B2_TRIGGER_STATEMENTS:
        normalized = statement.lstrip()
        name = normalized.split("CREATE TRIGGER IF NOT EXISTS ", 1)[1].splitlines()[0]
        expected_sql[name] = _normalize_schema_sql(
            normalized.replace("CREATE TRIGGER IF NOT EXISTS", "CREATE TRIGGER", 1)
        )
    actual_rows = connection.execute(
        "SELECT name, sql FROM sqlite_master WHERE name IN ("
        + ",".join("?" for _ in expected_sql)
        + ")",
        tuple(sorted(expected_sql)),
    ).fetchall()
    actual_sql = {
        str(row["name"]): _normalize_schema_sql(row["sql"]) for row in actual_rows
    }
    if actual_sql != expected_sql:
        raise sqlite3.DatabaseError(
            "slice-aware B2 schema is incomplete or incompatible; recreate the B2 "
            "schema objects"
        )

    required: dict[str, set[str]] = {
        "eval_dataset_slice_builds": {
            "dataset_id",
            "clustering_run_id",
            "sampling_algorithm",
            "slice_manifest_json",
            "slice_manifest_hash",
        },
        "eval_case_slice_provenance": {
            "eval_id",
            "dataset_id",
            "selector",
            "cluster_number",
            "source_trace_hash",
            "selection_key",
            "slice_manifest_hash",
        },
        "experiment_run_slice_aggregates": {
            "run_id",
            "cluster_number",
            "score",
            "pass_rate",
        },
        "experiment_comparison_slice_aggregates": {
            "experiment_id",
            "cluster_number",
            "baseline_score",
            "candidate_score",
            "baseline_pass_rate",
            "candidate_pass_rate",
        },
    }
    for table_name, columns in required.items():
        actual = {
            str(row["name"])
            for row in connection.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
        }
        if not columns.issubset(actual):
            raise sqlite3.DatabaseError(
                f"slice-aware B2 schema is incompatible: {table_name}"
            )


def _ensure_b2_data_compatible(connection: sqlite3.Connection) -> None:
    """Reject persisted partial builds that cannot be treated as sealed."""
    row = connection.execute(
        """
        SELECT build.dataset_id
        FROM eval_dataset_slice_builds AS build
        LEFT JOIN eval_cases AS case_row ON case_row.dataset_id = build.dataset_id
        LEFT JOIN eval_case_slice_provenance AS provenance
          ON provenance.eval_id = case_row.eval_id
         AND provenance.dataset_id = build.dataset_id
        GROUP BY build.dataset_id, build.sampled_size
        HAVING COUNT(DISTINCT case_row.eval_id) != build.sampled_size
            OR COUNT(DISTINCT provenance.eval_id) != build.sampled_size
        LIMIT 1
        """
    ).fetchone()
    if row is not None:
        raise sqlite3.DatabaseError(
            f"slice-built dataset '{row['dataset_id']}' is incomplete"
        )


def _build_expected_clustering_schema() -> ClusteringSchemaSnapshot:
    """Build the canonical structured B1 schema snapshot once at import time."""
    with closing(sqlite3.connect(":memory:")) as reference:
        reference.row_factory = sqlite3.Row
        for statement in SCHEMA_STATEMENTS:
            normalized = statement.lstrip()
            if any(
                normalized.startswith(f"CREATE TABLE IF NOT EXISTS {name} ")
                or normalized.startswith(f"CREATE INDEX IF NOT EXISTS {name}")
                or normalized.startswith(f"CREATE UNIQUE INDEX IF NOT EXISTS {name}")
                for name in (
                    "trace_clustering_runs",
                    "trace_cluster_assignments",
                    "idx_trace_cluster_assignments_cluster",
                    "trace_cluster_labels",
                    "idx_trace_cluster_labels_key",
                )
            ):
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
    cases = [_case_from_row(row) for row in rows]
    provenance_rows = connection.execute(
        """
        SELECT * FROM eval_case_slice_provenance
        WHERE dataset_id = ?
        ORDER BY eval_id ASC
        """,
        (dataset_id,),
    ).fetchall()
    provenance = {
        str(row["eval_id"]): _slice_case_provenance_from_row(row)
        for row in provenance_rows
    }
    return [
        case.model_copy(update={"slice_provenance": provenance.get(case.eval_id)})
        for case in cases
    ]


def insert_dataset_slice_build(
    connection: sqlite3.Connection,
    dataset_id: str,
    source: SliceBuildSource,
) -> None:
    """Seal one fully populated slice-built dataset."""
    connection.execute(
        """
        INSERT INTO eval_dataset_slice_builds (
            dataset_id, clustering_run_id, clustering_run_name,
            clustering_schema_version, clustering_configuration_hash,
            clustering_source_manifest_hash, cluster_count,
            sampling_schema_version, sampling_algorithm, requested_size,
            sampled_size, eligible_trace_count, slice_manifest_json,
            slice_manifest_hash, built_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            source.clustering_run_id,
            source.clustering_run_name,
            source.clustering_schema_version,
            source.clustering_configuration_hash,
            source.clustering_source_manifest_hash,
            source.cluster_count,
            source.sampling_schema_version,
            source.sampling_algorithm,
            source.requested_size,
            source.sampled_size,
            source.eligible_trace_count,
            _encode_json(source.slice_manifest),
            source.slice_manifest_hash,
            timestamp_to_text(source.built_at),
        ),
    )


def insert_case_slice_provenance(
    connection: sqlite3.Connection,
    eval_id: str,
    dataset_id: str,
    provenance: SliceCaseProvenance,
) -> None:
    """Insert the immutable B1/sampling snapshot for one promoted case."""
    connection.execute(
        """
        INSERT INTO eval_case_slice_provenance (
            eval_id, dataset_id, selector, cluster_number,
            label_snapshot, label_key_snapshot, clustering_run_id,
            clustering_run_name, clustering_schema_version,
            clustering_configuration_hash, clustering_source_manifest_hash,
            cluster_count, source_trace_id, source_timestamp,
            source_trace_hash, document_index, document_hash,
            sampling_schema_version, sampling_algorithm, requested_size,
            sampled_size, eligible_trace_count, slice_availability,
            slice_quota, rank_within_slice, selection_key, allocation_key,
            slice_manifest_hash
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            eval_id,
            dataset_id,
            provenance.selector,
            provenance.cluster_number,
            provenance.label_snapshot,
            provenance.label_key_snapshot,
            provenance.clustering_run_id,
            provenance.clustering_run_name,
            provenance.clustering_schema_version,
            provenance.clustering_configuration_hash,
            provenance.clustering_source_manifest_hash,
            provenance.cluster_count,
            provenance.source_trace_id,
            timestamp_to_text(provenance.source_timestamp),
            provenance.source_trace_hash,
            provenance.document_index,
            provenance.document_hash,
            provenance.sampling_schema_version,
            provenance.sampling_algorithm,
            provenance.requested_size,
            provenance.sampled_size,
            provenance.eligible_trace_count,
            provenance.slice_availability,
            provenance.slice_quota,
            provenance.rank_within_slice,
            provenance.selection_key,
            provenance.allocation_key,
            provenance.slice_manifest_hash,
        ),
    )


def get_dataset_slice_build(
    connection: sqlite3.Connection, dataset_id: str
) -> SliceBuildSource | None:
    """Load a sealed dataset build snapshot, if this is a B2 dataset."""
    row = connection.execute(
        "SELECT * FROM eval_dataset_slice_builds WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    if row is None:
        return None
    return SliceBuildSource.model_validate(
        {
            "clustering_run_id": row["clustering_run_id"],
            "clustering_run_name": row["clustering_run_name"],
            "clustering_schema_version": row["clustering_schema_version"],
            "clustering_configuration_hash": row["clustering_configuration_hash"],
            "clustering_source_manifest_hash": row["clustering_source_manifest_hash"],
            "cluster_count": row["cluster_count"],
            "sampling_schema_version": row["sampling_schema_version"],
            "sampling_algorithm": row["sampling_algorithm"],
            "requested_size": row["requested_size"],
            "sampled_size": row["sampled_size"],
            "eligible_trace_count": row["eligible_trace_count"],
            "slice_manifest_hash": row["slice_manifest_hash"],
            "slice_manifest": _decode_list(row["slice_manifest_json"]),
            "built_at": row["built_at"],
        }
    )


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


def _slice_case_provenance_from_row(row: sqlite3.Row) -> SliceCaseProvenance:
    return SliceCaseProvenance.model_validate(
        {
            key: row[key]
            for key in (
                "selector",
                "cluster_number",
                "label_snapshot",
                "label_key_snapshot",
                "clustering_run_id",
                "clustering_run_name",
                "clustering_schema_version",
                "clustering_configuration_hash",
                "clustering_source_manifest_hash",
                "cluster_count",
                "source_trace_id",
                "source_timestamp",
                "source_trace_hash",
                "document_index",
                "document_hash",
                "sampling_schema_version",
                "sampling_algorithm",
                "requested_size",
                "sampled_size",
                "eligible_trace_count",
                "slice_availability",
                "slice_quota",
                "rank_within_slice",
                "selection_key",
                "allocation_key",
                "slice_manifest_hash",
            )
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

    gate_columns = {
        str(row["name"])
        for row in connection.execute(
            "PRAGMA table_info(experiment_gate_violations)"
        ).fetchall()
    }
    if not {"scope_kind", "cluster_number", "label_snapshot"}.issubset(gate_columns):
        _rebuild_experiment_tables(connection, ["experiment_gate_violations"])

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
