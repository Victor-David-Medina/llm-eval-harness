"""Tests for the evaluators and the scorer.

Written with the standard library unittest module so the suite runs with no
pip install:

    python -m unittest discover -s tests

pytest can also run this file unchanged if it is present:

    pytest tests/test_evaluators.py

Every assertion checks a concrete, deterministic property. These tests are
the proof that the gate measures what it claims to measure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# Make the package importable when the suite runs from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import evaluators  # noqa: E402
from harness import golden  # noqa: E402
from harness import score  # noqa: E402


class TestGroundingScore(unittest.TestCase):
    def test_fully_grounded_answer_scores_one(self):
        context = "The deep tissue massage costs 95 dollars on Saturday."
        answer = "deep tissue massage 95 Saturday"
        self.assertEqual(evaluators.grounding_score(answer, context), 1.0)

    def test_ungrounded_answer_scores_low(self):
        context = "The massage costs 95 dollars."
        answer = "free helicopter ride included today"
        self.assertLess(evaluators.grounding_score(answer, context), 0.25)

    def test_empty_answer_scores_zero(self):
        self.assertEqual(evaluators.grounding_score("", "some context here"), 0.0)

    def test_partial_grounding_is_a_fraction(self):
        context = "Saturday morning slots are open."
        # Two content tokens: "saturday" (grounded), "helicopter" (not).
        answer = "Saturday helicopter"
        self.assertEqual(evaluators.grounding_score(answer, context), 0.5)

    def test_is_deterministic(self):
        context = "client Dana booked deep tissue on Saturday"
        answer = "Dana deep tissue Saturday"
        first = evaluators.grounding_score(answer, context)
        second = evaluators.grounding_score(answer, context)
        self.assertEqual(first, second)


class TestFaithfulnessFlag(unittest.TestCase):
    def test_consistent_answer_is_faithful(self):
        context = "The session is 60 minutes and costs 95 dollars."
        answer = "Your session is 60 minutes for 95 dollars."
        self.assertTrue(evaluators.faithfulness_flag(answer, context))

    def test_invented_number_is_unfaithful(self):
        context = "The session costs 95 dollars."
        answer = "The session costs 250 dollars."
        self.assertFalse(evaluators.faithfulness_flag(answer, context))

    def test_polarity_flip_is_unfaithful(self):
        context = "Refunds are available within 7 days for unused packages."
        answer = "Refunds are not available for unused packages."
        self.assertFalse(evaluators.faithfulness_flag(answer, context))

    def test_empty_context_is_treated_as_faithful(self):
        self.assertTrue(evaluators.faithfulness_flag("anything at all", ""))

    def test_number_present_in_context_is_faithful(self):
        context = "The plan renews on May 1 at 79 dollars."
        answer = "Your plan renews May 1 for 79 dollars."
        self.assertTrue(evaluators.faithfulness_flag(answer, context))

    def test_zero_padded_date_number_is_faithful(self):
        # Context stores the day zero padded as "02"; answer writes "April 2".
        # Normalization must treat these as the same number, not invented.
        context = "Estimate of 420 dollars on 2025 04 02."
        answer = "Your 420 dollar estimate from April 2 is still valid."
        self.assertTrue(evaluators.faithfulness_flag(answer, context))

    def test_decimal_trailing_zero_is_faithful(self):
        context = "The session costs 95 dollars."
        answer = "The session costs 95.0 dollars."
        self.assertTrue(evaluators.faithfulness_flag(answer, context))


class TestRelevanceScore(unittest.TestCase):
    def test_on_topic_answer_scores_high(self):
        user_input = "When does the Saturday massage slot open?"
        answer = "The Saturday massage slot opens at 9am."
        self.assertGreaterEqual(evaluators.relevance_score(answer, user_input), 0.6)

    def test_off_topic_answer_scores_low(self):
        user_input = "When does the Saturday massage slot open?"
        answer = "Our parking lot has twelve spaces."
        self.assertLess(evaluators.relevance_score(answer, user_input), 0.25)

    def test_empty_input_scores_zero(self):
        self.assertEqual(evaluators.relevance_score("a full answer", ""), 0.0)


class TestExactKeyword(unittest.TestCase):
    def test_all_present_scores_one(self):
        answer = "Hi Dana, your deep tissue session is Saturday."
        self.assertEqual(
            evaluators.exact_keyword(answer, ["Dana", "deep tissue", "Saturday"]),
            1.0,
        )

    def test_missing_one_is_a_fraction(self):
        answer = "Hi Dana, your session is Saturday."
        self.assertEqual(
            evaluators.exact_keyword(answer, ["Dana", "deep tissue", "Saturday"]),
            round(2 / 3, 4),
        )

    def test_empty_requirements_pass(self):
        self.assertEqual(evaluators.exact_keyword("anything", []), 1.0)

    def test_case_insensitive_match(self):
        self.assertEqual(evaluators.exact_keyword("HELLO DANA", ["dana"]), 1.0)

    def test_missing_keywords_helper(self):
        answer = "Hi Dana on Saturday"
        missing = evaluators.missing_keywords(answer, ["Dana", "deep tissue"])
        self.assertEqual(missing, ["deep tissue"])

    def test_phone_number_phrase_survives(self):
        answer = "Call us at 555-0100 to rebook."
        self.assertEqual(evaluators.exact_keyword(answer, ["555-0100"]), 1.0)


class TestCompositeScore(unittest.TestCase):
    def test_all_perfect_dimensions_score_one(self):
        dims = {
            "grounding": 1.0,
            "faithfulness": 1.0,
            "relevance": 1.0,
            "exact_keyword": 1.0,
        }
        self.assertEqual(score.composite_score(dims), 1.0)

    def test_weights_normalize_when_dimension_missing(self):
        # Only grounding present; result should equal grounding value.
        dims = {"grounding": 0.8}
        self.assertEqual(score.composite_score(dims), 0.8)

    def test_weighted_mean_is_between_dimensions(self):
        dims = {
            "grounding": 0.0,
            "faithfulness": 1.0,
            "relevance": 0.0,
            "exact_keyword": 0.0,
        }
        result = score.composite_score(dims)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)


class TestRegressionDetection(unittest.TestCase):
    def _scored(self, composite: float, record_id: str = "r") -> score.RecordScore:
        return score.RecordScore(
            record_id=record_id,
            composite=composite,
            dimensions={
                "grounding": composite,
                "faithfulness": 1.0,
                "relevance": composite,
                "exact_keyword": composite,
            },
        )

    def test_passing_run_with_no_baseline(self):
        scored = [self._scored(0.9), self._scored(0.85)]
        result = score.detect_regression(scored, baseline_mean=None)
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, "info")
        self.assertEqual(result.exit_code(), 0)

    def test_record_below_floor_is_critical(self):
        scored = [self._scored(0.95), self._scored(0.10, "broken")]
        result = score.detect_regression(scored, baseline_mean=None)
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "critical")
        self.assertEqual(result.exit_code(), 1)

    def test_run_below_floor_is_critical(self):
        # Both records pass the per-record floor but the mean is below RUN_FLOOR.
        scored = [self._scored(0.55), self._scored(0.55)]
        result = score.detect_regression(scored, baseline_mean=None)
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "critical")

    def test_small_drop_is_info_and_passes(self):
        scored = [self._scored(0.90), self._scored(0.90)]
        result = score.detect_regression(scored, baseline_mean=0.91)
        self.assertTrue(result.passed)
        self.assertEqual(result.severity, "info")

    def test_medium_drop_is_warn_and_passes(self):
        scored = [self._scored(0.84), self._scored(0.84)]
        result = score.detect_regression(scored, baseline_mean=0.90)
        self.assertEqual(result.severity, "warn")
        self.assertTrue(result.passed)

    def test_large_drop_is_critical_and_fails(self):
        scored = [self._scored(0.78), self._scored(0.78)]
        result = score.detect_regression(scored, baseline_mean=0.90)
        self.assertEqual(result.severity, "critical")
        self.assertFalse(result.passed)

    def test_empty_run_is_critical(self):
        result = score.detect_regression([], baseline_mean=None)
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "critical")


class TestGoldenLoader(unittest.TestCase):
    def _write_jsonl(self, lines):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        handle.write("\n".join(lines))
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_loads_valid_records(self):
        path = self._write_jsonl([
            json.dumps({
                "id": "a",
                "input": "q",
                "context": "c",
                "expected": "e",
                "must_include": ["e"],
            }),
        ])
        records = golden.load_golden(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "a")
        self.assertEqual(records[0].must_include, ("e",))

    def test_skips_comments_and_blank_lines(self):
        path = self._write_jsonl([
            "# a comment",
            "",
            json.dumps({"input": "q", "context": "c", "expected": "e"}),
        ])
        records = golden.load_golden(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "record-3")

    def test_missing_required_field_raises(self):
        path = self._write_jsonl([
            json.dumps({"input": "q", "context": "c"}),
        ])
        with self.assertRaises(golden.GoldenError):
            golden.load_golden(path)

    def test_invalid_json_raises(self):
        path = self._write_jsonl(["{not valid json}"])
        with self.assertRaises(golden.GoldenError):
            golden.load_golden(path)

    def test_empty_dataset_raises(self):
        path = self._write_jsonl(["# only a comment"])
        with self.assertRaises(golden.GoldenError):
            golden.load_golden(path)

    def test_answer_under_test_falls_back_to_expected(self):
        record = golden.GoldenRecord(
            input="q", context="c", expected="the expected answer"
        )
        self.assertEqual(record.answer_under_test(), "the expected answer")


class TestEndToEndOnShippedDataset(unittest.TestCase):
    """The shipped golden dataset must pass its own gate.

    This is the proof that the harness is honest: the curated expected
    answers, scored as the answer under test, clear the gate. If a future
    edit to the dataset or the evaluators breaks that, this test fails.
    """

    def test_shipped_dataset_passes_gate(self):
        dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "datasets",
            "golden_sample.jsonl",
        )
        records = golden.load_golden(dataset_path)
        self.assertGreaterEqual(len(records), 8)
        result = score.evaluate(records, baseline_mean=None)
        self.assertTrue(
            result.passed,
            msg=f"shipped dataset failed its own gate: {result.reasons}",
        )
        self.assertGreaterEqual(result.mean_composite, score.RUN_FLOOR)


if __name__ == "__main__":
    unittest.main()
