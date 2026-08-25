"""Tests for the pure dashboard helpers and Streamlit smoke states."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from test_customer_support_demo import FIXTURE_CONFIG, _build_demo
from tracebench.cli import app
from tracebench.dashboard import (
    DashboardLoadError,
    filter_cases,
    load_detail_file,
    load_detail_json,
    meaningful_latency,
    overview_metrics,
    resolve_report_path,
    review_queue,
    safe_context_text,
    slice_rows,
)
from tracebench.experiment_models import (
    ComparisonTransition,
    ExperimentStatus,
    RunStatus,
)
from tracebench.models import EvaluationMode, Priority
from tracebench.reporting import (
    ExperimentDetail,
    get_experiment_detail,
)

runner = CliRunner()
REPOSITORY_ROOT = Path(__file__).parents[1]
APP_PATH = REPOSITORY_ROOT / "dashboard" / "app.py"


def _failing_detail(tmp_path: Path) -> ExperimentDetail:
    database_path = tmp_path / "northstar.sqlite3"
    environment = _build_demo(database_path)
    result = runner.invoke(
        app,
        ["experiment", "run", str(FIXTURE_CONFIG), "--json"],
        env=environment,
    )
    assert result.exit_code == 1, result.output
    report = json.loads(result.stdout)
    return get_experiment_detail(database_path, report["experiment_id"])


def test_strict_loader_reports_valid_invalid_and_unsupported_documents(
    tmp_path: Path,
) -> None:
    detail = _failing_detail(tmp_path)
    valid_payload = detail.model_dump_json()
    assert load_detail_json(valid_payload).experiment_id == detail.experiment_id

    with pytest.raises(DashboardLoadError, match="not valid JSON"):
        load_detail_json("not-json")

    unsupported = json.loads(valid_payload)
    unsupported["schema_version"] = 2
    with pytest.raises(
        DashboardLoadError, match="unsupported schema version"
    ) as caught:
        load_detail_json(json.dumps(unsupported))
    assert caught.value.kind == "unsupported_version"

    report_path = tmp_path / "detail.json"
    report_path.write_text(valid_payload, encoding="utf-8")
    assert load_detail_file(report_path).experiment_id == detail.experiment_id
    assert (
        resolve_report_path(environment={"TRACEBENCH_REPORT_PATH": str(report_path)})
        == report_path
    )
    assert resolve_report_path(cwd=tmp_path) is None


def test_persisted_metrics_slices_filters_reviews_and_fixture_latency(
    tmp_path: Path,
) -> None:
    detail = _failing_detail(tmp_path)
    before = detail.model_dump(mode="json")

    metrics = overview_metrics(detail)
    assert metrics.verdict == "FAIL"
    assert metrics.baseline_score == pytest.approx(0.9333333333333333)
    assert metrics.candidate_score == pytest.approx(0.9)
    assert metrics.score_delta == pytest.approx(-0.033333333333333326)
    assert metrics.newly_passed_count == 2
    assert metrics.newly_failed_count == 3
    assert metrics.critical_regression_count == 1
    assert metrics.review_count == 2
    assert metrics.gate_passed is False

    rows = slice_rows(detail)
    assert [row.cluster_number for row in rows] == list(range(7))
    assert rows[0].label == "account-security"
    assert rows[4].gate_status == "FAIL"
    assert rows[0].gate_status == "PASS"
    assert rows[1].gate_status == "Not configured"

    newly_failed = filter_cases(detail)
    assert len(newly_failed) == 3
    assert all(
        case.transition is ComparisonTransition.NEWLY_FAILED for case in newly_failed
    )
    critical = filter_cases(
        detail,
        slice_label="refund-policy",
        priority=Priority.CRITICAL,
        newly_failed_only=False,
    )
    assert [case.source_trace.trace_id for case in critical] == ["refund-01"]
    rubric = filter_cases(
        detail,
        evaluation_mode=EvaluationMode.RUBRIC,
        newly_failed_only=False,
    )
    assert len(rubric) == 7
    assert filter_cases(detail, search="does-not-exist", newly_failed_only=False) == ()

    queue = review_queue(detail)
    assert [row.case.source_trace.trace_id for row in queue] == [
        "refund-01",
        "account-03",
    ]
    assert queue[0].reasons == ("critical_failure",)
    assert queue[1].reasons == ("low_confidence",)

    latency = meaningful_latency(detail)
    assert latency.available is False
    assert latency.message == "Unavailable for fixture providers"
    assert detail.model_dump(mode="json") == before
    assert safe_context_text({"customer": "A", "order": 3}).startswith("{")


def test_real_provider_latency_requires_meaningful_persisted_observations(
    tmp_path: Path,
) -> None:
    detail = _failing_detail(tmp_path)
    providers = {
        role: snapshot.model_copy(update={"provider": "ollama"})
        for role, snapshot in detail.provider_snapshots.items()
    }
    updated_cases = []
    for index, case in enumerate(detail.cases):
        assert case.baseline is not None and case.candidate is not None
        baseline_generation = case.baseline.generation.model_copy(
            update={"latency_ms": float(index + 1)}
        )
        candidate_generation = case.candidate.generation.model_copy(
            update={"latency_ms": float(index + 3)}
        )
        updated_cases.append(
            case.model_copy(
                update={
                    "baseline": case.baseline.model_copy(
                        update={"generation": baseline_generation}
                    ),
                    "candidate": case.candidate.model_copy(
                        update={"generation": candidate_generation}
                    ),
                }
            )
        )
    real_provider_detail = detail.model_copy(
        update={"provider_snapshots": providers, "cases": updated_cases}
    )
    latency = meaningful_latency(real_provider_detail)
    assert latency.available is True
    assert latency.baseline_mean_ms == pytest.approx(11.0)
    assert latency.candidate_mean_ms == pytest.approx(13.0)
    assert latency.delta_ms == pytest.approx(2.0)
    assert latency.baseline_observations == latency.candidate_observations == 21


def test_operational_partial_detail_is_presented_without_inventing_a_verdict(
    tmp_path: Path,
) -> None:
    detail = _failing_detail(tmp_path)
    failed_candidate_run = detail.runs[1].model_copy(
        update={"status": RunStatus.FAILED}
    )
    partial_report = detail.report.model_copy(
        update={
            "status": ExperimentStatus.FAILED,
            "verdict": None,
            "comparison": None,
            "gate": None,
        }
    )
    partial = detail.model_copy(
        update={
            "status": ExperimentStatus.FAILED,
            "verdict": None,
            "failure_stage": "candidate",
            "failure_message": "provider unavailable",
            "runs": [detail.runs[0], failed_candidate_run],
            "report": partial_report,
            "provider_snapshots": {
                role: snapshot.model_copy(update={"provider": "ollama"})
                for role, snapshot in detail.provider_snapshots.items()
            },
        }
    )
    metrics = overview_metrics(partial)
    assert metrics.status == "failed"
    assert metrics.verdict == "—"
    assert metrics.baseline_score == pytest.approx(0.9333333333333333)
    assert metrics.candidate_score == pytest.approx(0.9)
    assert metrics.newly_failed_count == 0
    assert meaningful_latency(partial).message == "Unavailable until both runs complete"


def test_streamlit_smoke_renders_empty_invalid_and_valid_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRACEBENCH_REPORT_PATH", raising=False)
    default_demo = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert default_demo.success
    assert default_demo.title[0].value == "TraceBench result explorer"
    assert any("Northstar hosted demo" in item.value for item in default_demo.success)
    assert any("FAIL" in item.value for item in default_demo.error)
    assert any(
        "Unavailable for fixture providers" in item.value for item in default_demo.info
    )

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TRACEBENCH_REPORT_PATH", str(invalid_path))
    invalid = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert invalid.error
    assert "Invalid experiment detail JSON" in invalid.error[0].value

    detail = _failing_detail(tmp_path)
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(detail.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("TRACEBENCH_REPORT_PATH", str(valid_path))
    valid = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert valid.success
    assert all(str(valid_path) not in item.value for item in valid.success)
    assert any("FAIL" in item.value for item in valid.error)
    assert any("Unavailable for fixture providers" in item.value for item in valid.info)
