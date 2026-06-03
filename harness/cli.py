"""Command line entry point. This is the CI gate.

Run it with:

    python -m harness.cli eval --dataset datasets/golden_sample.jsonl

It scores the dataset, prints a scorecard, and exits non-zero on a CRITICAL
regression. A non-zero exit fails the GitHub Actions job, which blocks the
merge. That single property, a failing eval stops the build, is the entire
reason this harness exists. Everything else is plumbing around it.

A second subcommand, "baseline", writes the current run's mean composite to
a JSON file so later runs can be compared against a known-good point.

Stdlib only: argparse, json, sys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import evaluators
from .golden import GoldenError, load_golden
from .score import RunResult, evaluate

# Exit codes. 0 = gate open. 1 = critical regression, build blocked.
# 2 = usage or data error (dataset missing, malformed JSON).
EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2

_BAR_WIDTH = 20


def _bar(value: float) -> str:
    """Render a 0.0 to 1.0 value as a small ASCII meter for the scorecard."""
    filled = int(round(max(0.0, min(1.0, value)) * _BAR_WIDTH))
    return "#" * filled + "." * (_BAR_WIDTH - filled)


def _load_baseline(path: Optional[str]) -> Optional[float]:
    """Read a baseline mean composite from a JSON file, if one is given.

    The file is expected to contain {"mean_composite": <float>}. A missing
    file is treated as "no baseline" rather than an error, so the gate still
    runs on absolute floors the first time it is wired up.
    """
    if not path:
        return None
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None
    try:
        data = json.loads(baseline_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("mean_composite")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _print_scorecard(result: RunResult, dataset: str) -> None:
    """Print a human-readable scorecard to stdout."""
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  llm-eval-harness scorecard")
    lines.append("=" * 60)
    lines.append(f"  dataset      : {dataset}")
    lines.append(f"  records      : {len(result.records)}")
    lines.append(f"  mean compos. : {result.mean_composite:.3f}  [{_bar(result.mean_composite)}]")
    if result.baseline is not None:
        lines.append(f"  baseline     : {result.baseline:.3f}")
    lines.append("")
    lines.append("  dimension averages")
    for name in evaluators.all_dimensions():
        mean = result.dimension_means.get(name, 0.0)
        lines.append(f"    {name:<14}: {mean:.3f}  [{_bar(mean)}]")
    lines.append("")
    lines.append("  per record")
    for record in result.records:
        marker = "ok " if not record.below_floor() else "LOW"
        lines.append(
            f"    [{marker}] {record.record_id:<22} "
            f"composite {record.composite:.3f}"
        )
        if record.missing_keywords:
            lines.append(
                f"           missing required: {', '.join(record.missing_keywords)}"
            )
        if not record.faithful:
            lines.append("           faithfulness flag tripped (possible contradiction)")
    lines.append("")
    lines.append("  verdict")
    lines.append(f"    severity   : {result.severity.upper()}")
    for reason in result.reasons:
        lines.append(f"    - {reason}")
    status = "PASS (gate open)" if result.passed else "FAIL (gate closed, build blocked)"
    lines.append("")
    lines.append(f"  result       : {status}")
    lines.append("=" * 60)
    lines.append("")
    sys.stdout.write("\n".join(lines) + "\n")


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the harness over a dataset and return the process exit code."""
    try:
        records = load_golden(args.dataset)
    except GoldenError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_USAGE

    baseline = _load_baseline(args.baseline)
    result = evaluate(records, baseline_mean=baseline)
    _print_scorecard(result, args.dataset)

    if args.json:
        payload = {
            "dataset": args.dataset,
            "mean_composite": result.mean_composite,
            "baseline": result.baseline,
            "severity": result.severity,
            "passed": result.passed,
            "dimension_means": result.dimension_means,
            "records": [
                {
                    "id": r.record_id,
                    "composite": r.composite,
                    "dimensions": r.dimensions,
                    "missing_keywords": r.missing_keywords,
                    "faithful": r.faithful,
                }
                for r in result.records
            ],
            "reasons": result.reasons,
        }
        Path(args.json).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        sys.stdout.write(f"wrote machine-readable report to {args.json}\n")

    return EXIT_OK if result.passed else EXIT_GATE_FAILED


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Score a dataset and write its mean composite as the new baseline."""
    try:
        records = load_golden(args.dataset)
    except GoldenError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_USAGE

    result = evaluate(records, baseline_mean=None)
    payload = {
        "dataset": args.dataset,
        "mean_composite": result.mean_composite,
        "dimension_means": result.dimension_means,
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    sys.stdout.write(
        f"baseline written to {args.out} "
        f"(mean composite {result.mean_composite:.3f})\n"
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="harness.cli",
        description="Honest, stdlib-only LLM evaluation gate for CI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    eval_parser = sub.add_parser(
        "eval",
        help="Score a golden dataset and exit non-zero on a critical regression.",
    )
    eval_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL golden dataset.",
    )
    eval_parser.add_argument(
        "--baseline",
        default=None,
        help="Optional path to a baseline JSON file for regression comparison.",
    )
    eval_parser.add_argument(
        "--json",
        default=None,
        help="Optional path to write a machine-readable JSON report.",
    )
    eval_parser.set_defaults(func=_cmd_eval)

    baseline_parser = sub.add_parser(
        "baseline",
        help="Score a dataset and save its mean composite as a baseline file.",
    )
    baseline_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL golden dataset.",
    )
    baseline_parser.add_argument(
        "--out",
        required=True,
        help="Path to write the baseline JSON file.",
    )
    baseline_parser.set_defaults(func=_cmd_baseline)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
