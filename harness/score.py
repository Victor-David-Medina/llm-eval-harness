"""Weighted composite scorer and tiered regression detection.

This module turns per-dimension evaluator output into one number per record,
one number for the whole run, and a verdict on whether the run regressed
against a saved baseline. The verdict is what the CI gate acts on.

Two ideas do the work:

    Composite score. Each dimension carries a weight. The composite is the
    weighted mean of the four dimension scores. Weights are explicit and
    checked in, so anyone can see and argue with how much each dimension
    counts. exact_keyword and faithfulness are weighted heavily because a
    missing required disclaimer or an invented number is worse for a real
    business answer than slightly lower keyword overlap.

    Tiered regression. We compare the new run's mean composite to a baseline.
    A small dip is INFO (noise). A larger dip is WARN (look at it). A dip past
    the critical line, or any record that drops below an absolute floor, is
    CRITICAL and fails the build. Three tiers keep the gate from being either
    a hair trigger or a rubber stamp.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import evaluators
from .golden import GoldenRecord

# Default dimension weights. They sum to 1.0 so the composite stays in
# [0.0, 1.0]. Tuned for business answers: the hard floor (exact_keyword) and
# the contradiction check (faithfulness) matter most, grounding next, raw
# keyword relevance last because an answer can be relevant yet wrong.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "grounding": 0.30,
    "faithfulness": 0.30,
    "relevance": 0.15,
    "exact_keyword": 0.25,
}

# Regression tiers, expressed as a drop in mean composite vs baseline.
# A drop of 0.02 or less is treated as noise.
INFO_DROP = 0.02
WARN_DROP = 0.05
CRITICAL_DROP = 0.10

# Absolute per-record floor. Any single record whose composite falls below
# this is CRITICAL on its own, even if the run average looks fine. One badly
# broken answer should not be averaged away by nine good ones.
RECORD_FLOOR = 0.50

# Absolute run floor. Even with no baseline (first run), a mean composite
# below this fails, so a brand new pipeline cannot ship green while broken.
RUN_FLOOR = 0.70

# Severity ordering for comparisons and exit codes.
_SEVERITY_ORDER = {"info": 0, "warn": 1, "critical": 2}


def composite_score(
    dimensions: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted mean of dimension scores. Returns a float in [0.0, 1.0].

    Only the dimensions named in weights are counted, and the result is
    normalized by the weights actually used, so a missing dimension cannot
    silently deflate or inflate the score.
    """
    weights = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    accumulated = 0.0
    for name, weight in weights.items():
        if name in dimensions:
            accumulated += dimensions[name] * weight
            total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return round(accumulated / total_weight, 4)


@dataclass
class RecordScore:
    """Scored result for a single golden record."""

    record_id: str
    composite: float
    dimensions: Dict[str, float]
    missing_keywords: List[str] = field(default_factory=list)
    faithful: bool = True

    def below_floor(self) -> bool:
        """True when this record trips the absolute per-record floor."""
        return self.composite < RECORD_FLOOR


@dataclass
class RunResult:
    """Aggregate result for a full evaluation run plus the gate verdict.

    Attributes:
        mean_composite  Average composite across all records.
        records         Per-record scores, in dataset order.
        severity        info | warn | critical. Drives the CI exit code.
        passed          True when the run is allowed to ship.
        reasons         Human-readable lines explaining the verdict.
        baseline        The baseline mean compared against, if any.
        dimension_means Average of each dimension across the run.
    """

    mean_composite: float
    records: List[RecordScore]
    severity: str
    passed: bool
    reasons: List[str]
    baseline: Optional[float]
    dimension_means: Dict[str, float]

    def exit_code(self) -> int:
        """0 when the gate passes, 1 when it fails (critical)."""
        return 0 if self.passed else 1


def score_records(
    records: Sequence[GoldenRecord],
    weights: Optional[Dict[str, float]] = None,
) -> List[RecordScore]:
    """Evaluate and composite-score every record. Pure, deterministic."""
    weights = weights or DEFAULT_WEIGHTS
    scored: List[RecordScore] = []
    for record in records:
        answer = record.answer_under_test()
        dims = evaluators.evaluate_record(
            answer=answer,
            context=record.context,
            user_input=record.input,
            must_include=record.must_include,
        )
        detail = dims.pop("detail")
        scored.append(
            RecordScore(
                record_id=record.record_id,
                composite=composite_score(dims, weights),
                dimensions=dims,
                missing_keywords=detail["missing_keywords"],
                faithful=detail["faithful"],
            )
        )
    return scored


def _dimension_means(scored: Sequence[RecordScore]) -> Dict[str, float]:
    """Average each dimension across all scored records."""
    means: Dict[str, float] = {}
    if not scored:
        return {name: 0.0 for name in evaluators.all_dimensions()}
    for name in evaluators.all_dimensions():
        total = sum(s.dimensions.get(name, 0.0) for s in scored)
        means[name] = round(total / len(scored), 4)
    return means


def _classify(drop: float) -> str:
    """Map a composite drop vs baseline to a severity tier."""
    if drop >= CRITICAL_DROP:
        return "critical"
    if drop >= WARN_DROP:
        return "warn"
    return "info"


def detect_regression(
    scored: Sequence[RecordScore],
    baseline_mean: Optional[float] = None,
) -> RunResult:
    """Aggregate scored records and decide whether the run may ship.

    Decision logic, evaluated together so the worst condition wins:

        1. Any record below RECORD_FLOOR -> CRITICAL (one broken answer).
        2. Run mean below RUN_FLOOR -> CRITICAL (absolute quality floor).
        3. If a baseline is given, classify the drop into info/warn/critical
           using the tier thresholds.

    The run passes only when the final severity is not critical.
    """
    if not scored:
        return RunResult(
            mean_composite=0.0,
            records=[],
            severity="critical",
            passed=False,
            reasons=["no records scored"],
            baseline=baseline_mean,
            dimension_means=_dimension_means(scored),
        )

    mean_composite = round(sum(s.composite for s in scored) / len(scored), 4)
    reasons: List[str] = []
    severity = "info"

    # Condition 1: per-record floor.
    broken = [s for s in scored if s.below_floor()]
    if broken:
        severity = "critical"
        for s in broken:
            reasons.append(
                f"record {s.record_id} composite {s.composite:.2f} "
                f"below per-record floor {RECORD_FLOOR:.2f}"
            )

    # Condition 2: absolute run floor.
    if mean_composite < RUN_FLOOR:
        severity = _max_severity(severity, "critical")
        reasons.append(
            f"run mean {mean_composite:.2f} below run floor {RUN_FLOOR:.2f}"
        )

    # Condition 3: regression vs baseline.
    if baseline_mean is not None:
        drop = round(baseline_mean - mean_composite, 4)
        if drop > INFO_DROP:
            tier = _classify(drop)
            severity = _max_severity(severity, tier)
            reasons.append(
                f"composite dropped {drop:.3f} vs baseline "
                f"{baseline_mean:.3f} ({tier})"
            )
        else:
            reasons.append(
                f"composite within noise of baseline "
                f"(drop {drop:.3f} <= {INFO_DROP:.3f})"
            )
    else:
        reasons.append("no baseline supplied; gating on absolute floors only")

    if not reasons:
        reasons.append("all checks passed")

    passed = _SEVERITY_ORDER[severity] < _SEVERITY_ORDER["critical"]

    return RunResult(
        mean_composite=mean_composite,
        records=list(scored),
        severity=severity,
        passed=passed,
        reasons=reasons,
        baseline=baseline_mean,
        dimension_means=_dimension_means(scored),
    )


def _max_severity(current: str, candidate: str) -> str:
    """Return the more severe of two severity labels."""
    if _SEVERITY_ORDER[candidate] > _SEVERITY_ORDER[current]:
        return candidate
    return current


def evaluate(
    records: Sequence[GoldenRecord],
    baseline_mean: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> RunResult:
    """End to end: score every record, then return the gate verdict."""
    scored = score_records(records, weights)
    return detect_regression(scored, baseline_mean)
