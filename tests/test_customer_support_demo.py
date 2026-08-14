"""End-to-end acceptance test for the Northstar Shop customer-support demo."""

import json
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tracebench.cli import app
from tracebench.datasets import get_dataset_details
from tracebench.experiment_config import prepare_experiment
from tracebench.experiment_models import RunRole
from tracebench.models import EvaluationMode
from tracebench.storage import connect_database

REPOSITORY_ROOT = Path(__file__).parents[1]
DEMO_ROOT = REPOSITORY_ROOT / "demos" / "customer-support"
FIXTURE_CONFIG = DEMO_ROOT / "experiment.fixture.yaml"
OLLAMA_CONFIG = DEMO_ROOT / "experiment.ollama.yaml"

EXPECTED_ASSIGNMENTS = {
    0: [f"account-{index:02d}" for index in range(1, 6)],
    1: [f"billing-{index:02d}" for index in range(1, 6)],
    2: [f"cancel-{index:02d}" for index in range(1, 6)],
    3: [f"delivery-{index:02d}" for index in range(1, 6)],
    4: [f"refund-{index:02d}" for index in range(1, 6)],
    5: [f"return-{index:02d}" for index in range(1, 6)],
    6: [f"troubleshoot-{index:02d}" for index in range(1, 6)],
}
EXPECTED_LABELS = [
    "account-security",
    "billing-invoices",
    "cancellation-retention",
    "delivery-tracking",
    "refund-policy",
    "returns-exchanges",
    "product-troubleshooting",
]
EXPECTED_SELECTED = {
    0: ["account-01", "account-03", "account-05"],
    1: ["billing-03", "billing-04", "billing-05"],
    2: ["cancel-02", "cancel-04", "cancel-05"],
    3: ["delivery-02", "delivery-03", "delivery-04"],
    4: ["refund-01", "refund-03", "refund-04"],
    5: ["return-01", "return-02", "return-05"],
    6: ["troubleshoot-03", "troubleshoot-04", "troubleshoot-05"],
}
EXPECTED_EVAL_IDS = {
    "account-01": "eval_4c34135438605873a9ba76451cb3597e",
    "account-03": "eval_0c91ce01bee5539e931198ae8d5f5051",
    "account-05": "eval_ab900eceea0f5a60a52700a14f0c78dd",
    "billing-03": "eval_d0d7407eabac5847aafa8bcee6c7f9f9",
    "billing-04": "eval_13d50af32fab52658fce1bb9590d5a23",
    "billing-05": "eval_7228433be39a54cab42f289757e67bf1",
    "cancel-02": "eval_6158f44064dd518ebf596464dfe3a428",
    "cancel-04": "eval_8a7062bc266b5fd899855e7da9b0439b",
    "cancel-05": "eval_7e886a2f1c1d534fb94b6a629c4dcb6c",
    "delivery-02": "eval_7f776d7ef4b55a7092858d67730fd387",
    "delivery-03": "eval_7409de8607ad54e3a921eebea09dfac8",
    "delivery-04": "eval_60cc0c056ab75a24839ad8dc01027b3e",
    "refund-01": "eval_dffb5784255f5777a33bcffcec495b55",
    "refund-03": "eval_f974a78c5a225bd08cd49d4f02583498",
    "refund-04": "eval_122f850dffb55d068b5e9f3262b68797",
    "return-01": "eval_c064405281835a23b04b7a784d8bd91c",
    "return-02": "eval_c8c37b75c3a25fba83aae21ba036d3c4",
    "return-05": "eval_a6ea5536ddd45f3ea99daa576b327231",
    "troubleshoot-03": "eval_ef121816a3a657819dc8ee80a45ce99f",
    "troubleshoot-04": "eval_731fc2ecbf255436a7af0608d9fa3f5b",
    "troubleshoot-05": "eval_c9919f2c476a5cc1840cccff4719aa6a",
}
EXPECTED_MANIFEST_HASH = (
    "05933fc8b290d4af0ad82e433d4656b2cec3010159c92b9f0357474f4aac6dee"
)

runner = CliRunner()


def _invoke(arguments: list[str], environment: dict[str, str]) -> object:
    result = runner.invoke(app, arguments, env=environment)
    assert result.exit_code == 0, result.output
    return result


def _build_demo(database_path: Path) -> dict[str, str]:
    environment = {"TRACEBENCH_DB_PATH": str(database_path)}
    _invoke(["ingest", str(DEMO_ROOT / "traces.jsonl")], environment)
    _invoke(
        [
            "traces",
            "cluster",
            "--name",
            "northstar-support-slices-v1",
            "--clusters",
            "7",
        ],
        environment,
    )
    for cluster_number, label in enumerate(EXPECTED_LABELS):
        _invoke(
            [
                "slices",
                "rename",
                "northstar-support-slices-v1",
                str(cluster_number),
                label,
            ],
            environment,
        )
    _invoke(
        [
            "dataset",
            "build",
            "--name",
            "northstar-support-eval",
            "--version",
            "1.0",
            "--from-slices",
            "northstar-support-slices-v1",
            "--size",
            "21",
            "--case-file",
            str(DEMO_ROOT / "cases.json"),
        ],
        environment,
    )
    return environment


def _assert_report(report: dict[str, object], *, cache_hit: bool) -> None:
    assert report["schema_version"] == 2
    assert report["verdict"] == "FAIL"
    runs = report["runs"]
    assert isinstance(runs, dict)
    baseline = runs["baseline"]
    candidate = runs["candidate"]
    assert baseline["global"]["score"] == pytest.approx(0.9333333333333333)
    assert candidate["global"]["score"] == pytest.approx(0.9)
    assert baseline["global"]["passed_count"] == 19
    assert candidate["global"]["passed_count"] == 18

    comparison = report["comparison"]
    assert comparison["newly_passed"] == [
        EXPECTED_EVAL_IDS["troubleshoot-04"],
        EXPECTED_EVAL_IDS["delivery-02"],
    ]
    assert comparison["newly_failed"] == [
        EXPECTED_EVAL_IDS["cancel-02"],
        EXPECTED_EVAL_IDS["billing-03"],
        EXPECTED_EVAL_IDS["refund-01"],
    ]
    assert comparison["by_mode"]["reference"]["case_count"] == 7
    assert comparison["by_mode"]["deterministic"]["newly_failed_count"] == 1
    assert comparison["by_mode"]["rubric"]["score_delta"] == pytest.approx(-0.1)

    judged = [
        case
        for role in (baseline, candidate)
        for case in role["cases"]
        if "judge" in case
    ]
    assert len(judged) == 14
    assert all(case["judge"]["cache_hit"] is cache_hit for case in judged)
    candidate_judged = {
        case["eval_id"]: case["judge"] for case in candidate["cases"] if "judge" in case
    }
    assert candidate_judged[EXPECTED_EVAL_IDS["account-03"]]["review"] == {
        "status": "needs_review",
        "reasons": ["low_confidence"],
    }
    assert candidate_judged[EXPECTED_EVAL_IDS["refund-01"]]["review"] == {
        "status": "needs_review",
        "reasons": ["critical_failure"],
    }


def test_customer_support_demo_is_reproducible_and_fails_the_release_gate(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "northstar.sqlite3"
    environment = _build_demo(database_path)

    with closing(connect_database(database_path)) as connection:
        assignment_rows = connection.execute(
            "SELECT cluster_number, trace_id FROM trace_cluster_assignments "
            "ORDER BY cluster_number, trace_id"
        ).fetchall()
        assignments = {
            cluster_number: [
                row["trace_id"]
                for row in assignment_rows
                if row["cluster_number"] == cluster_number
            ]
            for cluster_number in range(7)
        }
    assert assignments == EXPECTED_ASSIGNMENTS

    _, cases, source = get_dataset_details(database_path, "northstar-support-eval:1.0")
    assert source is not None
    assert source.slice_manifest_hash == EXPECTED_MANIFEST_HASH
    selected = {
        cluster_number: sorted(
            case.source_trace_id
            for case in cases
            if case.slice_provenance is not None
            and case.slice_provenance.cluster_number == cluster_number
        )
        for cluster_number in range(7)
    }
    assert selected == EXPECTED_SELECTED
    assert {case.source_trace_id: case.eval_id for case in cases} == EXPECTED_EVAL_IDS
    assert Counter(case.evaluation_mode for case in cases) == {
        EvaluationMode.REFERENCE: 7,
        EvaluationMode.DETERMINISTIC: 7,
        EvaluationMode.RUBRIC: 7,
    }
    assert all(case.slice_provenance is not None for case in cases)
    assert all(
        case.source_response is None and case.reference_answer is None
        for case in cases
        if case.evaluation_mode is not EvaluationMode.REFERENCE
    )

    first = runner.invoke(
        app,
        ["experiment", "run", str(FIXTURE_CONFIG), "--json"],
        env=environment,
    )
    assert first.exit_code == 1, first.output
    _assert_report(json.loads(first.stdout), cache_hit=False)

    second = runner.invoke(
        app,
        ["experiment", "run", str(FIXTURE_CONFIG), "--json"],
        env=environment,
    )
    assert second.exit_code == 1, second.output
    _assert_report(json.loads(second.stdout), cache_hit=True)

    ollama = prepare_experiment(OLLAMA_CONFIG, database_path)
    assert ollama.provider_snapshots[RunRole.BASELINE]["prompt_version"] == (
        "northstar-support-v1"
    )
    assert ollama.provider_snapshots[RunRole.CANDIDATE]["prompt_version"] == (
        "northstar-support-v2"
    )
    assert ollama.judge is not None
    assert ollama.judge.snapshot["provider"] == "ollama"
