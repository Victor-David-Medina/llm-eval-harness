"""llm-eval-harness: a small, honest LLM evaluation harness.

This package is a clean-room, standalone extraction of the eval-gate
pattern Victor David Medina runs inside his AI operations platform. It is
NOT the production system and copies no production code. It exists to show
the shape of the idea in a form anyone can read and run with the standard
library alone.

Public surface:
    evaluators  - deterministic scoring functions over (answer, context).
    golden      - typed loader for JSONL golden datasets.
    score       - weighted composite scorer + tiered regression detection.
    cli         - argparse entry point that gates CI on evaluation results.
"""

__all__ = ["evaluators", "golden", "score", "cli"]
__version__ = "0.1.0"
