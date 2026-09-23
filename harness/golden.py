"""Typed loader for JSONL golden datasets.

A golden dataset is the spine of an honest eval gate. Each line is one
frozen example: an input, the context the model was given, the expected
answer a reviewer signed off on, and the phrases the answer must contain.
The dataset is checked into the repo and reviewed like code, so a change to
what "correct" means is a visible diff, not a silent drift.

JSONL (one JSON object per line) is used on purpose: it streams, it diffs
cleanly in code review, and a single malformed line does not poison the
whole file. This module is stdlib only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Sequence, Union


class GoldenError(ValueError):
    """Raised when a golden dataset is malformed or missing required fields."""


@dataclass(frozen=True)
class GoldenRecord:
    """One frozen evaluation example.

    Fields:
        input        The user question or task the model received.
        context      The grounding material supplied to the model. Grounding
                     and faithfulness are scored against this.
        expected     A reference answer a human reviewer accepted. Kept for
                     audit and for future similarity scoring; the stdlib
                     evaluators here score the produced answer, not expected
                     directly, so a fresh model run can be graded against the
                     same context and must_include rules.
        must_include Phrases the answer is required to contain. The hard floor.
        record_id    Stable identifier for scorecards and CI logs.
        produced     The answer under test. Defaults to expected so the
                     dataset doubles as a self-consistency check out of the
                     box; in a live pipeline you overwrite this with the
                     model's fresh output before scoring.
        difficulty   A coarse label for the case: "standard", "edge", or
                     "adversarial". Used to slice the scorecard and to let CI
                     run a fast smoke subset and a full release eval from the
                     same file.
        tags         Free-form labels (e.g. "winback", "billing",
                     "red-team", "smoke"). Reports can filter on them without
                     a schema change.
    """

    input: str
    context: str
    expected: str
    must_include: Sequence[str] = field(default_factory=tuple)
    record_id: str = ""
    produced: str = ""
    difficulty: str = "standard"
    tags: Sequence[str] = field(default_factory=tuple)

    def answer_under_test(self) -> str:
        """The text to score. Falls back to expected when produced is empty."""
        return self.produced if self.produced.strip() else self.expected


_REQUIRED_FIELDS = ("input", "context", "expected")


def _coerce_record(obj: dict, line_no: int) -> GoldenRecord:
    """Validate a parsed JSON object and build a GoldenRecord.

    Raises GoldenError with the offending line number so a bad dataset is
    easy to fix instead of producing a cryptic stack trace deep in scoring.
    """
    if not isinstance(obj, dict):
        raise GoldenError(f"line {line_no}: expected a JSON object, got {type(obj).__name__}")

    for field_name in _REQUIRED_FIELDS:
        if field_name not in obj:
            raise GoldenError(f"line {line_no}: missing required field '{field_name}'")
        if not isinstance(obj[field_name], str):
            raise GoldenError(f"line {line_no}: field '{field_name}' must be a string")

    must_include_raw = obj.get("must_include", [])
    if must_include_raw is None:
        must_include_raw = []
    if not isinstance(must_include_raw, list) or not all(isinstance(x, str) for x in must_include_raw):
        raise GoldenError(f"line {line_no}: 'must_include' must be a list of strings")

    record_id = obj.get("id") or obj.get("record_id") or f"record-{line_no}"
    produced = obj.get("produced", "")
    if not isinstance(produced, str):
        raise GoldenError(f"line {line_no}: 'produced' must be a string when present")

    difficulty = obj.get("difficulty", "standard")
    if difficulty not in ("standard", "edge", "adversarial"):
        raise GoldenError(
            f"line {line_no}: 'difficulty' must be one of "
            "'standard', 'edge', or 'adversarial'"
        )

    tags_raw = obj.get("tags", [])
    if tags_raw is None:
        tags_raw = []
    if not isinstance(tags_raw, list) or not all(isinstance(x, str) for x in tags_raw):
        raise GoldenError(f"line {line_no}: 'tags' must be a list of strings")

    return GoldenRecord(
        input=obj["input"],
        context=obj["context"],
        expected=obj["expected"],
        must_include=tuple(must_include_raw),
        record_id=str(record_id),
        produced=produced,
        difficulty=difficulty,
        tags=tuple(tags_raw),
    )


def iter_golden(path: Union[str, Path]) -> Iterator[GoldenRecord]:
    """Yield GoldenRecord objects from a JSONL file, one per non-blank line.

    Blank lines and lines beginning with '#' are skipped so a dataset can
    carry comments. Streaming keeps memory flat on large datasets.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise GoldenError(f"golden dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise GoldenError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            yield _coerce_record(obj, line_no)


def load_golden(path: Union[str, Path]) -> List[GoldenRecord]:
    """Read an entire JSONL golden dataset into a list of GoldenRecord.

    Raises GoldenError if the file is missing, empty, or malformed.
    """
    records = list(iter_golden(path))
    if not records:
        raise GoldenError(f"golden dataset is empty: {path}")
    return records
