"""Minimal library-usage example for llm-eval-harness.

Run from the repository root:

    python examples/run_eval_gate.py

The example uses the same primitives as the CLI: load a golden JSONL dataset,
score it, print a compact summary, and exit non-zero if the gate closes.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.golden import load_golden  # noqa: E402
from harness.score import evaluate  # noqa: E402


def main() -> int:
    records = load_golden(ROOT / "datasets" / "golden_sample.jsonl")
    result = evaluate(records)
    print(f"records={len(result.records)} mean={result.mean_composite:.3f}")
    print(f"severity={result.severity} passed={result.passed}")
    for reason in result.reasons:
        print(f"- {reason}")
    return result.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
