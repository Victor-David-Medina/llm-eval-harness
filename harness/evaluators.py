"""Deterministic, stdlib-only evaluators for LLM output.

Every function here is pure: same inputs always produce the same float or
bool. No network, no model calls, no randomness. That property is the whole
point. An eval gate that is itself non-deterministic cannot tell you whether
a model regressed, because you cannot separate model drift from harness
noise. These evaluators give the CI gate a stable ruler.

The four dimensions, in plain terms:

    grounding_score   How much of the answer is actually backed by the
                      provided context. Catches confident text that wandered
                      away from the source material.
    faithfulness_flag A cheap contradiction heuristic. Returns True when the
                      answer looks faithful to the context, False when it
                      trips a negation or number mismatch.
    relevance_score   How well the answer addresses the user input, measured
                      by keyword overlap. Catches on-topic-but-off-question.
    exact_keyword     A hard must-include check. Some answers MUST contain a
                      specific phone number, policy name, or disclaimer. This
                      is the non-negotiable floor.

These are intentionally simple. In the production platform the same four
dimensions are computed with stronger signals (embeddings on Qdrant and
pgvector, an LLM-as-judge pass, Langfuse traces). The contract is identical:
a number per dimension, fed into a weighted composite, compared to a baseline.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

# Words that carry no topical signal. Removing them keeps overlap scores
# honest, so "the a of and" does not inflate grounding or relevance.
_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "i", "if", "in", "into", "is", "it", "its", "may", "me", "my", "no",
    "nor", "not", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "to",
    "up", "was", "we", "were", "will", "with", "would", "you", "your",
}

# Negation cues used by the faithfulness contradiction heuristic.
_NEGATIONS: Set[str] = {
    "not", "no", "never", "cannot", "cant", "wont", "isnt", "arent",
    "dont", "doesnt", "didnt", "none", "neither", "without",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _normalize_number(raw: str) -> str:
    """Canonicalize a numeric string so equivalent values compare equal.

    Strips leading zeros from the integer part and any trailing zeros from a
    decimal fraction. This makes "2" and "02" the same number, and "95.0"
    the same as "95". Without it the contradiction heuristic would raise a
    false alarm whenever an answer writes a day as "April 2" while the source
    stored it zero padded as "02", which is a formatting difference, not an
    invented figure.
    """
    raw = raw.strip()
    if "." in raw:
        integer_part, _, fraction_part = raw.partition(".")
        integer_part = integer_part.lstrip("0") or "0"
        fraction_part = fraction_part.rstrip("0")
        return f"{integer_part}.{fraction_part}" if fraction_part else integer_part
    return raw.lstrip("0") or "0"


def _number_set(text: str) -> Set[str]:
    """Extract numbers from text as a set of normalized canonical strings."""
    return {_normalize_number(n) for n in _NUMBER_RE.findall(text)}


def tokenize(text: str) -> List[str]:
    """Lowercase a string and split it into alphanumeric tokens.

    Punctuation is dropped. This is the shared tokenizer used by every
    evaluator so that all dimensions agree on what a token is.
    """
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(text: str) -> List[str]:
    """Tokens with stopwords removed. Used for overlap-based scores."""
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Divide, returning 0.0 when the denominator is zero.

    A zero denominator means there was nothing meaningful to measure (for
    example an empty answer). We treat that as a zero score rather than an
    error so a single empty record cannot crash the whole CI run.
    """
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def grounding_score(answer: str, context: str) -> float:
    """Fraction of answer content tokens that appear in the context.

    Returns a float in [0.0, 1.0]. A score of 1.0 means every meaningful
    word in the answer is supported by the supplied context. A low score
    means the model introduced material the context never mentioned, which
    is the classic shape of a hallucination.

    Stopwords are excluded on both sides so grammar does not earn credit.
    An empty answer scores 0.0 (nothing to ground).
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_set: Set[str] = set(_content_tokens(context))
    supported = sum(1 for t in answer_tokens if t in context_set)
    return round(_safe_ratio(supported, len(answer_tokens)), 4)


def faithfulness_flag(answer: str, context: str) -> bool:
    """Cheap contradiction heuristic. True means "looks faithful".

    This is deliberately a flag, not a score, because it answers a yes or no
    question: does the answer contradict the context in a way we can detect
    without a model? Two signals trip it to False:

        1. Negation mismatch. The answer asserts a negation about a key
           context term that the context does not negate (for example the
           answer says "we do not offer refunds" while the context describes
           a refund policy). This catches polarity flips.
        2. Number conflict. The answer states a number that does not appear
           anywhere in the context. Invented figures (a price, a percentage,
           a count) are a common and costly failure for business answers.

    When the context is empty we cannot judge faithfulness, so we return
    True (we do not punish a record that supplied no grounding material).
    The function is conservative: it only flags clear, mechanical conflicts,
    which keeps false alarms low at the cost of missing subtle ones.
    """
    if not context.strip():
        return True

    answer_tokens = tokenize(answer)
    context_tokens = tokenize(context)
    context_set: Set[str] = set(context_tokens)

    # Signal 1: negation pointed at a shared content term.
    answer_has_negation = any(t in _NEGATIONS for t in answer_tokens)
    context_has_negation = any(t in _NEGATIONS for t in context_tokens)
    if answer_has_negation and not context_has_negation:
        shared_content = [
            t for t in answer_tokens
            if t in context_set and t not in _STOPWORDS and t not in _NEGATIONS
        ]
        # A negation aimed at material the context states positively is a
        # likely polarity flip.
        if shared_content:
            return False

    # Signal 2: a number in the answer that the context never states.
    # Numbers are normalized first so "April 2" matches a context "02" and a
    # formatting difference is not mistaken for an invented figure.
    answer_numbers: Set[str] = _number_set(answer)
    if answer_numbers:
        context_numbers: Set[str] = _number_set(context)
        invented = answer_numbers - context_numbers
        if invented:
            return False

    return True


def relevance_score(answer: str, user_input: str) -> float:
    """How well the answer covers the question, by keyword overlap.

    Returns a float in [0.0, 1.0]. We measure the fraction of the user
    input's content tokens that show up in the answer. The intuition: a
    relevant answer addresses the words the user actually asked about.

    This is overlap from the question side on purpose. Grounding already
    measures the answer side against context. Relevance asks the opposite
    direction: did the answer engage with the prompt at all, or did it drift
    to a related but unasked topic.

    An empty user input scores 0.0 (no question to be relevant to).
    """
    input_tokens = _content_tokens(user_input)
    if not input_tokens:
        return 0.0
    answer_set: Set[str] = set(_content_tokens(answer))
    covered = sum(1 for t in set(input_tokens) if t in answer_set)
    return round(_safe_ratio(covered, len(set(input_tokens))), 4)


def exact_keyword(answer: str, must_include: Sequence[str]) -> float:
    """Fraction of required phrases present in the answer.

    Returns a float in [0.0, 1.0]. Each entry in must_include is matched as
    a case-insensitive substring against the raw answer text (not the
    tokenized form, so multi-word phrases and formatted values like a phone
    number survive intact).

    When must_include is empty there is nothing to require, which is a pass,
    so this returns 1.0. When entries are present, a 1.0 means every required
    phrase appears and anything below 1.0 means at least one required phrase
    is missing. This dimension is the hard floor: some answers must contain a
    specific disclaimer or contact detail no matter how good the prose is.
    """
    requirements = [r for r in must_include if r and r.strip()]
    if not requirements:
        return 1.0
    haystack = answer.lower()
    present = sum(1 for r in requirements if r.lower() in haystack)
    return round(_safe_ratio(present, len(requirements)), 4)


def missing_keywords(answer: str, must_include: Sequence[str]) -> List[str]:
    """Return the required phrases that are absent from the answer.

    Helper for human-readable scorecards: tells a reviewer exactly which
    must-include phrase was dropped, instead of just a number.
    """
    haystack = answer.lower()
    return [
        r for r in must_include
        if r and r.strip() and r.lower() not in haystack
    ]


def evaluate_record(
    answer: str,
    context: str,
    user_input: str,
    must_include: Sequence[str],
) -> dict:
    """Run all four evaluators over one record and return a dimension map.

    The returned dict has stable keys consumed by the scorer:
        grounding, faithfulness, relevance, exact_keyword
    plus a small "detail" block for human-facing scorecards. faithfulness
    is reported as 1.0 / 0.0 so it composes cleanly with the float scores.
    """
    faithful = faithfulness_flag(answer, context)
    return {
        "grounding": grounding_score(answer, context),
        "faithfulness": 1.0 if faithful else 0.0,
        "relevance": relevance_score(answer, user_input),
        "exact_keyword": exact_keyword(answer, must_include),
        "detail": {
            "faithful": faithful,
            "missing_keywords": missing_keywords(answer, must_include),
        },
    }


def all_dimensions() -> Iterable[str]:
    """Names of the scored dimensions, in display order."""
    return ("grounding", "faithfulness", "relevance", "exact_keyword")
