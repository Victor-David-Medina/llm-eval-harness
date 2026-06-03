# ADR-001: Gate CI on LLM evaluations

Status: Accepted
Date: 2026-06-01
Author: Victor David Medina

## Context

The code that ships an LLM feature is deterministic. The behavior it ships is
not. A prompt change, a model version bump, a retrieval tweak, or a new system
message can quietly make answers worse while every unit test stays green,
because unit tests check the plumbing, not the quality of the generated text.
The failure mode is slow and expensive: answers degrade a little at a time
until a real user gets a wrong number, an invented policy, or a confident
hallucination, and only then does anyone notice.

The standard engineering answer to "behavior regressed silently" is a test in
CI that fails the build. The pipeline did not have one for model output. This
ADR records the decision to build that test.

## Decision

Gate continuous integration on an automated evaluation of model output against
a curated golden dataset. Concretely:

1. Keep a golden dataset of frozen examples (`input`, `context`, `expected`,
   `must_include`) in the repo, reviewed like code. A change to what "correct"
   means is a visible diff.
2. Score each example on four dimensions: grounding (answer supported by
   context), faithfulness (no contradictions or invented numbers), relevance
   (answer covers the question), and exact keyword coverage (required phrases
   present). Each is deterministic and pure.
3. Combine them into a weighted composite, average across the dataset, and
   compare the result to a saved baseline.
4. Classify the comparison into three tiers and act on the tier:
   info passes, warn passes with a note, critical fails the build.
5. Wire the gate into GitHub Actions so a critical result blocks the merge.

## Why gate, and not just report

A dashboard nobody is forced to read does not change behavior. The production
trust argument is simple: if an eval result can be ignored, it will be ignored
under deadline pressure, which is exactly when regressions ship. A gate that
fails the build converts "quality" from an aspiration into a precondition for
merge. The same logic that makes us refuse to merge on a failing unit test or a
broken type check applies to a measurable drop in answer quality. The cost of a
false block (a developer investigates and re-baselines) is far lower than the
cost of a silent regression reaching a customer.

## The hard-block vs threshold tradeoff

The central tension is between a gate that is too strict and one that is too
loose.

- **Hard block on any drop** is a hair trigger. LLM scores have natural noise.
  A gate that fails on a 0.5 percent dip trains the team to bypass it, which is
  worse than having no gate, because now the gate exists and is routinely
  overridden.
- **Loose threshold** rubber-stamps real regressions. If only a catastrophic
  drop fails, the slow one-percent-at-a-time decay this whole effort exists to
  catch sails straight through.

The resolution is three tiers plus two absolute floors, all explicit and
checked in:

- A drop within noise (<= 0.02 vs baseline) is INFO and passes.
- A meaningful but survivable drop (0.05 to 0.10) is WARN: it passes but is
  logged loudly so a human looks.
- A large drop (>= 0.10) is CRITICAL and fails.
- Independent of any baseline, a single record below an absolute per-record
  floor (0.50) is CRITICAL, so one badly broken answer cannot be averaged away
  by nine good ones.
- Independent of any baseline, a run mean below an absolute run floor (0.70) is
  CRITICAL, so a brand new pipeline cannot ship green while broken.

Tiers are calibration knobs, not law. They live in `harness/score.py` as named
constants so changing the strictness of the gate is a reviewed, visible change,
not a buried magic number.

## What was deliberately left out

Honesty about scope is part of the design.

- **No model-graded scoring (LLM-as-judge).** The evaluators here are
  string and token heuristics. That keeps the harness deterministic, free,
  offline, and fully readable. The tradeoff is real: heuristics miss subtle
  semantic errors a judge model would catch. In a fuller pipeline an
  LLM-as-judge pass and embedding similarity (over a vector store such as
  Qdrant or pgvector) would augment these dimensions. They are out of scope
  here on purpose, so the core stays installable with nothing and auditable by
  eye.
- **No live model calls.** The harness scores text it is given. Generating that
  text from a model and capturing traces (for example in Langfuse) is the job
  of the surrounding pipeline, not this gate.
- **No statistical significance testing.** With small golden sets the gate uses
  fixed thresholds rather than confidence intervals. A larger dataset would
  justify a proper significance test on the drop before failing.
- **No automatic re-baselining.** Moving the baseline is a deliberate human act
  (`harness.cli baseline`), because automatically adopting today's numbers as
  tomorrow's bar would let slow decay redefine "good" downward.

## Consequences

- A measurable quality drop now blocks a merge, the same as a failing test.
- The definition of "correct" is a reviewed artifact (the golden dataset), not
  tribal knowledge.
- The gate adds a small maintenance cost: someone must curate the dataset and
  re-baseline intentionally when an improvement legitimately changes scores.
- The heuristic evaluators will have false negatives on subtle semantic errors.
  That is an accepted limit of the stdlib-only scope and the documented place
  where a model-graded layer would slot in next.
