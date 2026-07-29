"""Tests for deterministic scorer behavior and aggregation semantics."""

import pytest

from tracebench.experiment_models import (
    CaseResult,
    ComparisonTransition,
    EffectiveThresholds,
    GateMetric,
    ScorerResult,
)
from tracebench.experiments import aggregate_run, apply_regression_gate, compare_runs
from tracebench.models import EvalCase, ScorerConfig
from tracebench.scorers import ScorerConfigurationError, score_case


def deterministic_case(*scorers: ScorerConfig) -> EvalCase:
    """Return a deterministic case with the supplied scorer configurations."""
    return EvalCase.model_validate(
        {
            "eval_id": "eval-001",
            "dataset_id": "dataset-001",
            "source_trace_id": "trace-001",
            "source_timestamp": "2026-07-28T14:00:00Z",
            "source_task_type": "test",
            "source_response": None,
            "source_metadata": {},
            "input": "Prompt",
            "evaluation_mode": "deterministic",
            "scorers": list(scorers),
            "created_at": "2026-07-28T14:01:00Z",
        }
    )


def reference_case() -> EvalCase:
    """Return a reference case using implicit exact matching."""
    return EvalCase.model_validate(
        {
            "eval_id": "eval-reference",
            "dataset_id": "dataset-001",
            "source_trace_id": "trace-reference",
            "source_timestamp": "2026-07-28T14:00:00Z",
            "source_task_type": "test",
            "source_response": "Ottawa",
            "source_metadata": {},
            "input": "Prompt",
            "evaluation_mode": "reference",
            "reference_answer": "Ottawa",
            "created_at": "2026-07-28T14:01:00Z",
        }
    )


@pytest.mark.parametrize(
    ("scorer", "output", "passed"),
    [
        (
            ScorerConfig(
                name="exact_match",
                config={"expected": "Ottawa", "case_sensitive": False},
            ),
            "OTTAWA",
            True,
        ),
        (
            ScorerConfig(name="exact_match", config={"expected": "Ottawa"}),
            "ottawa",
            False,
        ),
        (
            ScorerConfig(
                name="contains",
                config={"substring": "trace ingestion", "case_sensitive": False},
            ),
            "The release improves TRACE INGESTION.",
            True,
        ),
        (
            ScorerConfig(name="regex", config={"pattern": r"^ID-\d+$"}),
            "ID-42",
            True,
        ),
        (ScorerConfig(name="json_validity"), '{"valid":true}', True),
        (ScorerConfig(name="json_validity"), '{"value":NaN}', False),
        (ScorerConfig(name="json_validity"), '{"value":1e999}', False),
        (
            ScorerConfig(name="required_keys", config={"keys": ["label", "score"]}),
            '{"label":"positive","score":1}',
            True,
        ),
        (
            ScorerConfig(name="required_keys", config={"keys": ["label", "score"]}),
            '{"label":"positive"}',
            False,
        ),
        (
            ScorerConfig(name="required_keys", config={"keys": ["label"]}),
            '[{"label":"positive"}]',
            False,
        ),
    ],
)
def test_deterministic_scorers(
    scorer: ScorerConfig,
    output: str,
    passed: bool,
) -> None:
    """Each built-in scorer returns deterministic Boolean scores."""
    result = score_case(deterministic_case(scorer), output)

    assert result.passed is passed
    assert result.score == float(passed)


def test_multiple_scorers_use_and_for_pass_and_mean_for_score() -> None:
    """A partially successful scorer set receives partial score but fails the case."""
    case = deterministic_case(
        ScorerConfig(name="json_validity"),
        ScorerConfig(name="required_keys", config={"keys": ["missing"]}),
    )

    result = score_case(case, '{"present":true}')

    assert result.score == 0.5
    assert result.passed is False


def test_reference_mode_is_case_sensitive_exact_match() -> None:
    """Reference cases use the documented implicit scorer."""
    assert score_case(reference_case(), "Ottawa").passed is True
    assert score_case(reference_case(), "ottawa").passed is False


@pytest.mark.parametrize(
    "scorer",
    [
        ScorerConfig(name="unknown"),
        ScorerConfig(name="contains", config={"substring": " "}),
        ScorerConfig(name="regex", config={"pattern": "["}),
        ScorerConfig(name="required_keys", config={"keys": ["same", " same "]}),
    ],
)
def test_invalid_scorer_configuration_is_rejected(scorer: ScorerConfig) -> None:
    """Generic stored scorer JSON is strictly validated before use."""
    with pytest.raises(ScorerConfigurationError):
        score_case(deterministic_case(scorer), "output")


def result(
    eval_id: str,
    *,
    mode: str,
    score: float,
    passed: bool,
) -> CaseResult:
    """Return a compact result for aggregation and comparison tests."""
    return CaseResult.model_validate(
        {
            "eval_id": eval_id,
            "evaluation_mode": mode,
            "output": "output",
            "score": score,
            "passed": passed,
            "scorers": [
                ScorerResult(
                    name="test",
                    score=score,
                    passed=passed,
                )
            ],
        }
    )


def test_aggregation_emits_only_global_and_present_modes() -> None:
    """Absent modes have no synthetic aggregate."""
    aggregates = aggregate_run(
        [
            result("a", mode="deterministic", score=1.0, passed=True),
            result("b", mode="deterministic", score=0.5, passed=False),
        ]
    )

    assert [aggregate.scope for aggregate in aggregates] == [
        "global",
        "deterministic",
    ]
    assert aggregates[0].score == 0.75
    assert aggregates[0].pass_rate == 0.5


def test_comparison_identifies_all_transitions_and_mode_aggregates() -> None:
    """Comparisons pair stable IDs and classify each pass/fail transition."""
    baseline = [
        result("a", mode="deterministic", score=1.0, passed=True),
        result("b", mode="deterministic", score=0.0, passed=False),
        result("c", mode="reference", score=0.0, passed=False),
        result("d", mode="reference", score=1.0, passed=True),
    ]
    candidate = [
        result("a", mode="deterministic", score=1.0, passed=True),
        result("b", mode="deterministic", score=0.0, passed=False),
        result("c", mode="reference", score=1.0, passed=True),
        result("d", mode="reference", score=0.0, passed=False),
    ]

    comparisons, aggregates = compare_runs(baseline, candidate)

    assert [comparison.transition for comparison in comparisons] == [
        ComparisonTransition.PASSED_BOTH,
        ComparisonTransition.FAILED_BOTH,
        ComparisonTransition.NEWLY_PASSED,
        ComparisonTransition.NEWLY_FAILED,
    ]
    assert [aggregate.scope for aggregate in aggregates] == [
        "global",
        "deterministic",
        "reference",
    ]
    assert aggregates[0].newly_passed_count == 1
    assert aggregates[0].newly_failed_count == 1


def test_gate_threshold_equality_passes_and_excess_fails() -> None:
    """Regression thresholds are inclusive maxima and mode overrides are scoped."""
    baseline = [
        result("a", mode="deterministic", score=1.0, passed=True),
        result("b", mode="reference", score=1.0, passed=True),
    ]
    candidate = [
        result("a", mode="deterministic", score=0.0, passed=False),
        result("b", mode="reference", score=1.0, passed=True),
    ]
    _, aggregates = compare_runs(baseline, candidate)

    equality = apply_regression_gate(
        aggregates,
        {
            "global": EffectiveThresholds(
                max_score_drop=0.5,
                max_new_failures=1,
            ),
            "deterministic": EffectiveThresholds(
                max_score_drop=1.0,
                max_new_failures=1,
            ),
        },
    )
    exceeded = apply_regression_gate(
        aggregates,
        {
            "global": EffectiveThresholds(
                max_score_drop=0.49,
                max_new_failures=0,
            )
        },
    )

    assert equality == []
    assert [violation.metric for violation in exceeded] == [
        GateMetric.SCORE_DROP,
        GateMetric.NEW_FAILURES,
    ]
