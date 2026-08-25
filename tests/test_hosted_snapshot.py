"""Integrity and read-only tests for the reviewed hosted dashboard snapshot."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tracebench.dashboard import (
    filter_cases,
    load_detail_file,
    meaningful_latency,
    overview_metrics,
    review_queue,
    slice_rows,
)
from tracebench.experiment_models import (
    ComparisonTransition,
    JudgeReviewStatus,
    RunRole,
)
from tracebench.models import EvaluationMode, Priority
from tracebench.reporting import ExperimentDetail

REPOSITORY_ROOT = Path(__file__).parents[1]
SNAPSHOT_PATH = (
    REPOSITORY_ROOT
    / "demos"
    / "customer-support"
    / "hosted"
    / "northstar-fail-detail.json"
)
APP_PATH = REPOSITORY_ROOT / "dashboard" / "app.py"
EXPECTED_SLICES = {
    "account-security": ("account-03", "account-01", "account-05"),
    "billing-invoices": ("billing-04", "billing-05", "billing-03"),
    "cancellation-retention": ("cancel-02", "cancel-05", "cancel-04"),
    "delivery-tracking": ("delivery-04", "delivery-03", "delivery-02"),
    "product-troubleshooting": (
        "troubleshoot-04",
        "troubleshoot-05",
        "troubleshoot-03",
    ),
    "refund-policy": ("refund-04", "refund-01", "refund-03"),
    "returns-exchanges": ("return-05", "return-01", "return-02"),
}
FORBIDDEN_PATTERNS = (
    r"C:\\Users[\\/]",
    r"C:/Users[\\/]",
    r"OneDrive",
    r"/home/runner",
    r"(?i)sqlite(?:3)?",
    r"(?i)api[_-]?key",
    r"(?i)authorization",
    r"(?i)bearer",
    r"(?i)raw[_ -]?output",
    r"(?i)validation[_ -]?error",
    r"You are Northstar Shop's customer-support assistant",
    r"You are a strict evaluation judge",
)


def _snapshot() -> ExperimentDetail:
    return ExperimentDetail.model_validate_json(
        SNAPSHOT_PATH.read_text(encoding="utf-8")
    )


def test_hosted_snapshot_is_strict_and_authoritative() -> None:
    detail = _snapshot()
    assert detail.schema_version == 1
    assert detail.report.schema_version == 2
    assert detail.status.value == "completed"
    assert detail.verdict.value == "FAIL"
    assert detail.dataset.case_count == 21
    assert len(detail.cases) == 21
    assert {case.evaluation_mode for case in detail.cases} == set(EvaluationMode)
    assert {
        mode: sum(case.evaluation_mode is mode for case in detail.cases)
        for mode in EvaluationMode
    } == {mode: 7 for mode in EvaluationMode}
    assert {
        case.slice_provenance.label_snapshot
        for case in detail.cases
        if case.slice_provenance is not None
    } == set(EXPECTED_SLICES)
    assert {
        label: tuple(
            case.source_trace.trace_id
            for case in detail.cases
            if case.slice_provenance is not None
            and case.slice_provenance.label_snapshot == label
        )
        for label in EXPECTED_SLICES
    } == EXPECTED_SLICES

    baseline = detail.report.runs[RunRole.BASELINE].global_
    candidate = detail.report.runs[RunRole.CANDIDATE].global_
    comparison = detail.report.comparison
    assert baseline.score == pytest.approx(0.9333333333333333)
    assert candidate.score == pytest.approx(0.9)
    assert comparison is not None
    assert comparison.global_.score_delta == pytest.approx(-0.03333333333333344)
    assert {
        case.source_trace.trace_id
        for case in detail.cases
        if case.transition is ComparisonTransition.NEWLY_PASSED
    } == {"troubleshoot-04", "delivery-02"}
    assert {
        case.source_trace.trace_id
        for case in detail.cases
        if case.transition is ComparisonTransition.NEWLY_FAILED
    } == {"billing-03", "cancel-02", "refund-01"}
    critical = next(
        case for case in detail.cases if case.source_trace.trace_id == "refund-01"
    )
    assert critical.priority is Priority.CRITICAL
    assert critical.candidate is not None and critical.candidate.judge is not None
    assert [reason.value for reason in critical.candidate.judge.review.reasons] == [
        "critical_failure"
    ]
    low_confidence = next(
        case for case in detail.cases if case.source_trace.trace_id == "account-03"
    )
    assert low_confidence.candidate is not None
    assert low_confidence.candidate.judge is not None
    assert (
        low_confidence.candidate.judge.review.status is JudgeReviewStatus.NEEDS_REVIEW
    )
    assert [
        reason.value for reason in low_confidence.candidate.judge.review.reasons
    ] == ["low_confidence"]
    assert {snapshot.provider for snapshot in detail.provider_snapshots.values()} == {
        "fixture"
    }
    assert detail.judge_snapshot is not None
    assert detail.judge_snapshot.provider == "fixture"


def test_hosted_snapshot_preserves_leak_protections_and_forbidden_data_scan() -> None:
    payload = SNAPSHOT_PATH.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, payload) is None, pattern
    detail = ExperimentDetail.model_validate_json(payload)
    assert all(
        case.reference_answer is not None and case.source_trace.response is not None
        if case.evaluation_mode is EvaluationMode.REFERENCE
        else case.reference_answer is None and case.source_trace.response is None
        for case in detail.cases
    )
    assert '"raw_attempt"' not in payload
    assert '"prompt_file"' not in payload
    assert '"system_prompt"' not in payload


def test_hosted_snapshot_renders_read_only_without_rewriting() -> None:
    before = hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
    detail = load_detail_file(SNAPSHOT_PATH)
    assert overview_metrics(detail).verdict == "FAIL"
    assert len(slice_rows(detail)) == 7
    assert len(filter_cases(detail, newly_failed_only=False)) == 21
    assert len(review_queue(detail)) == 2
    assert meaningful_latency(detail).message == "Unavailable for fixture providers"
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert app.success
    assert any("Northstar hosted demo" in item.value for item in app.success)
    assert any("FAIL" in item.value for item in app.error)
    uploaded = (
        app.file_uploader[0]
        .set_value(
            ("uploaded-detail.json", SNAPSHOT_PATH.read_bytes(), "application/json")
        )
        .run()
    )
    assert any("Loaded uploaded-detail.json" in item.value for item in uploaded.success)
    assert any("FAIL" in item.value for item in uploaded.error)
    assert hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest() == before


def test_snapshot_regeneration_command_requires_explicit_output_and_overwrite() -> None:
    script = (
        REPOSITORY_ROOT / "scripts" / "prepare_customer_support_dashboard.py"
    ).read_text(encoding="utf-8")
    assert "required=True" in script
    assert '"--overwrite"' in script
    assert "reviewed hosted" in (
        REPOSITORY_ROOT / "demos" / "customer-support" / "README.md"
    ).read_text(encoding="utf-8")
