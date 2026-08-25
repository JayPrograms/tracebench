"""Streamlit entrypoint for the read-only TraceBench result explorer.

Run from the repository root with ``streamlit run dashboard/app.py``.  A
detail export can be supplied with ``-- --report path`` or the
``TRACEBENCH_REPORT_PATH`` environment variable; the uploader is useful when
reviewing an export produced on another machine.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import streamlit as st

from tracebench.dashboard import (
    DashboardLoadError,
    OverviewMetrics,
    case_slice_label,
    filter_cases,
    load_detail_file,
    load_detail_json,
    meaningful_latency,
    overview_metrics,
    resolve_report_path,
    review_queue,
    safe_context_text,
    safe_metadata_text,
    slice_rows,
)
from tracebench.experiment_models import ComparisonTransition, JudgeReviewStatus
from tracebench.models import EvaluationMode, Priority
from tracebench.reporting import CaseDetail, ExperimentDetail


def main() -> None:
    """Render the dashboard application."""

    st.set_page_config(
        page_title="TraceBench result explorer",
        page_icon="📊",
        layout="wide",
    )
    st.title("TraceBench result explorer")
    st.caption(
        "Read-only inspection of a strict ExperimentDetail export. "
        "This app never opens the experiment database or re-runs scoring."
    )

    detail, source_label = _load_from_ui()
    if detail is None:
        st.info(
            "Load a detail export to begin. Use the uploader below, set "
            "TRACEBENCH_REPORT_PATH, or pass `-- --report path` to Streamlit."
        )
        return

    st.success(f"Loaded {source_label}")
    _render_dashboard(detail)


def _load_from_ui() -> tuple[ExperimentDetail | None, str]:
    uploaded = st.file_uploader(
        "Experiment detail JSON",
        type=["json"],
        help="Only strict TraceBench ExperimentDetail JSON exports are accepted.",
    )
    report_argument = _command_line_report()
    query_report = _query_report()
    selected_path = resolve_report_path(
        report_argument or query_report,
        environment=os.environ,
    )
    if uploaded is not None:
        try:
            return load_detail_json(uploaded.getvalue()), uploaded.name
        except DashboardLoadError as error:
            _render_load_error(error)
            return None, uploaded.name
    if selected_path is None:
        return None, ""
    try:
        return load_detail_file(selected_path), str(selected_path)
    except DashboardLoadError as error:
        _render_load_error(error)
        return None, str(selected_path)


def _render_load_error(error: DashboardLoadError) -> None:
    if error.kind == "unsupported_version":
        st.error("Unsupported report version")
    elif error.kind == "unreadable":
        st.error("Could not read the selected report")
    else:
        st.error("Invalid experiment detail JSON")
    st.write(str(error))


def _render_dashboard(detail: ExperimentDetail) -> None:
    overview, slices, explorer, reviews, provenance = st.tabs(
        [
            "Overview",
            "Slice analysis",
            "Regression explorer",
            "Review queue",
            "Provenance",
        ]
    )
    with overview:
        _render_overview(detail)
    with slices:
        _render_slices(detail)
    with explorer:
        _render_explorer(detail)
    with reviews:
        _render_reviews(detail)
    with provenance:
        _render_provenance(detail)


def _render_overview(detail: ExperimentDetail) -> None:
    metrics = overview_metrics(detail)
    st.subheader("Release decision")
    decision = _decision_text(metrics)
    if metrics.verdict == "PASS":
        st.success(decision)
    elif metrics.verdict == "FAIL":
        st.error(decision)
    elif metrics.status == "failed":
        st.warning(decision)
    else:
        st.info(decision)

    score_columns = st.columns(3)
    score_columns[0].metric("Baseline score", _format_score(metrics.baseline_score))
    score_columns[1].metric("Candidate score", _format_score(metrics.candidate_score))
    score_columns[2].metric("Candidate delta", _format_score(metrics.score_delta))

    count_columns = st.columns(4)
    count_columns[0].metric("Newly passed", metrics.newly_passed_count)
    count_columns[1].metric("Newly failed", metrics.newly_failed_count)
    count_columns[2].metric("Critical regressions", metrics.critical_regression_count)
    count_columns[3].metric("Cases needing review", metrics.review_count)

    gate_label = (
        "Not recorded"
        if metrics.gate_passed is None
        else ("PASS" if metrics.gate_passed else "FAIL")
    )
    st.write(f"**Gate:** {gate_label} · {metrics.gate_violation_count} violation(s)")
    if detail.failure_stage or detail.failure_message:
        st.warning(
            "Operational failure: "
            + " · ".join(
                part for part in (detail.failure_stage, detail.failure_message) if part
            )
        )

    latency = meaningful_latency(detail)
    st.subheader("Generation latency")
    if latency.available:
        latency_columns = st.columns(3)
        latency_columns[0].metric("Baseline mean", f"{latency.baseline_mean_ms:.1f} ms")
        latency_columns[1].metric(
            "Candidate mean", f"{latency.candidate_mean_ms:.1f} ms"
        )
        latency_columns[2].metric("Candidate delta", f"{latency.delta_ms:+.1f} ms")
        st.caption(
            f"{latency.baseline_observations} baseline and "
            f"{latency.candidate_observations} candidate observations"
        )
    else:
        st.info(latency.message)

    st.subheader("Experiment identity")
    st.json(
        {
            "experiment_id": detail.experiment_id,
            "name": detail.name,
            "dataset": f"{detail.dataset.name}:{detail.dataset.version}",
            "dataset_id": detail.dataset.dataset_id,
            "configuration_hash": detail.configuration_hash,
            "case_count": detail.dataset.case_count,
            "sealed": detail.dataset.sealed,
        }
    )

    critical_cases = [
        case
        for case in detail.cases
        if case.transition is ComparisonTransition.NEWLY_FAILED
        and case.priority is Priority.CRITICAL
    ]
    st.subheader("Critical regressions")
    if critical_cases:
        for case in critical_cases:
            st.write(f"**FAIL · {case.eval_id}** · {case_slice_label(case)}")
            st.caption(case.input)
    else:
        st.write("None persisted in this report.")

    st.subheader("Gate violations")
    if detail.gate.violations:
        st.dataframe(
            [
                {
                    "scope": violation.scope,
                    "metric": violation.metric.value,
                    "actual": violation.actual,
                    "allowed": violation.allowed,
                    "message": violation.message,
                }
                for violation in detail.gate.violations
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("None persisted in this report.")


def _render_slices(detail: ExperimentDetail) -> None:
    st.subheader("Slice analysis")
    rows = slice_rows(detail)
    if not rows:
        st.info("This report has no persisted slice comparison.")
        return
    st.caption("Rows use persisted aggregates and retain numeric cluster order.")
    st.dataframe(
        [
            {
                "cluster": row.cluster_number,
                "slice": row.label,
                "cases": row.case_count,
                "baseline score": row.baseline_score,
                "candidate score": row.candidate_score,
                "delta": row.score_delta,
                "newly passed": row.newly_passed_count,
                "newly failed": row.newly_failed_count,
                "slice gate": row.gate_status,
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )
    chart_data = {
        row.label: {
            "Baseline": row.baseline_score,
            "Candidate": row.candidate_score,
        }
        for row in rows
    }
    st.bar_chart(chart_data)


def _render_explorer(detail: ExperimentDetail) -> None:
    st.subheader("Regression explorer")
    controls = st.columns(4)
    slice_value = controls[0].selectbox(
        "Slice", ["All"] + [row.label for row in slice_rows(detail)]
    )
    priority_value = controls[1].selectbox(
        "Priority", ["All"] + [priority.value for priority in Priority]
    )
    mode_value = controls[2].selectbox(
        "Evaluation mode", ["All"] + [mode.value for mode in EvaluationMode]
    )
    transition_value = controls[3].selectbox(
        "Transition",
        ["All"] + [transition.value for transition in ComparisonTransition],
    )
    second_row = st.columns(3)
    review_value = second_row[0].selectbox(
        "Candidate review", ["All"] + [status.value for status in JudgeReviewStatus]
    )
    newly_failed_only = second_row[1].checkbox(
        "Focus on newly failed cases", value=True
    )
    search = second_row[2].text_input("Search case ID, trace, task, or prompt")
    selected = filter_cases(
        detail,
        slice_label=None if slice_value == "All" else slice_value,
        priority=None if priority_value == "All" else priority_value,
        evaluation_mode=None if mode_value == "All" else mode_value,
        transition=None if transition_value == "All" else transition_value,
        review_state=None if review_value == "All" else review_value,
        search=search,
        newly_failed_only=newly_failed_only,
    )
    st.caption(f"Showing {len(selected)} persisted case(s).")
    if not selected:
        st.info("No cases match these filters.")
        return
    for case in selected:
        _render_case(case)


def _render_case(case: CaseDetail) -> None:
    transition = "not available" if case.transition is None else case.transition.value
    title = f"{transition.upper()} · {case.eval_id} · {case_slice_label(case)}"
    with st.expander(title):
        st.write(
            f"**Priority:** {case.priority.value} · "
            f"**Mode:** {case.evaluation_mode.value}"
        )
        st.write(f"**Input:** {case.input}")
        st.write("**Context**")
        st.code(safe_context_text(case.context), language="json")
        run_columns = st.columns(2)
        _render_run_case(run_columns[0], "Baseline", case.baseline)
        _render_run_case(run_columns[1], "Candidate", case.candidate)
        if (
            case.evaluation_mode is EvaluationMode.REFERENCE
            and case.reference_answer is not None
        ):
            st.write("**Reference answer snapshot**")
            st.code(case.reference_answer)
        if case.scorers:
            st.write("**Persisted deterministic scorers**")
            st.json(case.scorers)
        if case.rubric:
            st.write("**Persisted rubric criteria**")
            for criterion in case.rubric:
                st.write(f"- {criterion}")
            st.caption(
                "Raw judge prompts and attempts are intentionally omitted from "
                "detail exports; the persisted score, confidence, and review state "
                "above are authoritative."
            )
        st.write("**Slice provenance**")
        st.json(
            None
            if case.slice_provenance is None
            else case.slice_provenance.model_dump(mode="json")
        )


def _render_run_case(container: Any, label: str, result: Any) -> None:
    with container:
        st.write(f"**{label} output**")
        if result is None:
            st.info("No persisted result")
            return
        st.write(
            f"**{'PASS' if result.passed else 'FAIL'}** · score {result.score:.6f}"
        )
        st.code(result.output)
        if result.judge is not None:
            st.write(
                f"Judge: {'PASS' if result.judge.overall_passed else 'FAIL'} · "
                f"confidence {result.judge.confidence:.2f} · "
                f"review {result.judge.review.status.value}"
            )
            if result.judge.review.reasons:
                st.write(
                    "Review reasons: "
                    + ", ".join(reason.value for reason in result.judge.review.reasons)
                )
            st.caption(f"Judge cache: {result.judge.cache.status.value}")


def _render_reviews(detail: ExperimentDetail) -> None:
    st.subheader("Review queue")
    queue = review_queue(detail)
    st.caption("Read-only queue from persisted candidate judge review metadata.")
    if not queue:
        st.success("No persisted cases require review.")
        return
    st.dataframe(
        [
            {
                "case": row.case.eval_id,
                "slice": case_slice_label(row.case),
                "priority": row.case.priority.value,
                "reasons": ", ".join(row.reasons),
                "baseline review": row.baseline_review_status,
                "candidate review": row.candidate_review_status,
            }
            for row in queue
        ],
        width="stretch",
        hide_index=True,
    )
    for row in queue:
        with st.expander(f"{row.case.eval_id} · {', '.join(row.reasons)}"):
            st.write(row.case.input)
            st.write(f"Candidate state: **{row.candidate_review_status}**")
            st.write(f"Baseline state: **{row.baseline_review_status}**")


def _render_provenance(detail: ExperimentDetail) -> None:
    st.subheader("Provenance and audit metadata")
    st.json(
        {
            "schema_version": detail.schema_version,
            "experiment_id": detail.experiment_id,
            "configuration_hash": detail.configuration_hash,
            "timestamps": detail.timestamps.model_dump(mode="json"),
            "dataset": detail.dataset.model_dump(mode="json"),
            "providers": {
                role: snapshot.model_dump(mode="json")
                for role, snapshot in detail.provider_snapshots.items()
            },
            "judge": None
            if detail.judge_snapshot is None
            else detail.judge_snapshot.model_dump(mode="json"),
            "gate": detail.gate.model_dump(mode="json"),
        }
    )
    st.write("**Run lifecycle**")
    st.dataframe(
        [run.model_dump(mode="json") for run in detail.runs],
        width="stretch",
        hide_index=True,
    )
    st.write("**Generation metadata sample**")
    sample = next((case for case in detail.cases if case.candidate is not None), None)
    if sample is None or sample.candidate is None:
        st.write("No candidate generation observation is persisted.")
    else:
        st.code(
            safe_metadata_text(sample.candidate.generation.provider_metadata),
            language="json",
        )
    if detail.dataset.slice_source is not None:
        st.write("**Slice-build provenance**")
        st.json(detail.dataset.slice_source.model_dump(mode="json"))
    st.caption(
        "Exported prompt text, raw judge attempts, credential-like metadata, and "
        "database writes are deliberately outside this explorer's authority."
    )


def _decision_text(metrics: OverviewMetrics) -> str:
    if metrics.status == "failed":
        return "Operational failure — this report has no release verdict."
    return f"{metrics.verdict} · {metrics.status}"


def _format_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def _command_line_report() -> str | None:
    args = sys.argv[1:]
    for index, argument in enumerate(args):
        if argument in {"--report", "--report-path"} and index + 1 < len(args):
            return args[index + 1]
    return None


def _query_report() -> str | None:
    try:
        value = st.query_params.get("report")
    except AttributeError:
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


if __name__ == "__main__":
    main()
