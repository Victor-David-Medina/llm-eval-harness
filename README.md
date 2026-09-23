# llm-eval-harness

[![eval-gate](https://github.com/Victor-David-Medina/llm-eval-harness/actions/workflows/eval.yml/badge.svg)](https://github.com/Victor-David-Medina/llm-eval-harness/actions/workflows/eval.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Evals as tests: a CI gate for LLM output quality.**

A small, honest, stdlib-only LLM evaluation harness that gates CI. It scores
model output against checked-in golden datasets, detects regressions in tiers,
and fails the build when quality drops past a critical line.

The 30-second version: you check golden examples into the repo, the harness
scores each new answer on four deterministic dimensions, rolls them into one
weighted number, and compares it to a baseline. If the number falls too far,
the build fails. No API keys, no model calls, no network. A reviewer can read
every line.

## Why this exists

LLM output is non-deterministic. The same prompt can produce a slightly worse
answer tomorrow than it did today, and nothing in a normal test suite notices.
A unit test asserts `2 + 2 == 4`. There is no equivalent line for "this
support answer is still grounded in the customer's actual booking history and
did not invent a refund policy." So quality regresses silently: a prompt edit,
a model version bump, a retrieval change, and the answers get worse one
percent at a time until a real customer gets a wrong number.

This harness is the missing test. It turns "is the answer still good" into a
number, compares that number to a known-good baseline, and stops the merge
when the number falls too far. It runs on the Python standard library alone,
so a reviewer can read every line and run it with no install.

## Architecture

```
  datasets/golden_sample.jsonl        (40 frozen, code-reviewed examples)
  datasets/redteam_sample.jsonl       (12 adversarial probes)
        |
        v
  +-----------------------------------------------------------+
  |  evaluators  (deterministic, pure, stdlib)                |
  |    grounding_score    answer tokens supported by context  |
  |    faithfulness_flag  contradiction + invented-number     |
  |    relevance_score    answer covers the question          |
  |    exact_keyword      required phrases present (hard floor)|
  +-----------------------------------------------------------+
        |
        v
  weighted composite scorer            (one score per record, one per run)
        |
        v
  tiered regression detection          info  -> noise, pass
                                       warn  -> look at it, pass
                                       critical -> fail
        |
        v
  CI gate (python -m harness.cli eval) exits non-zero on critical
        |
        v
  GitHub Actions job fails  ->  merge blocked
```

The contract is one number per dimension, fed into an explicit weighted
composite, compared to a baseline, classified into three tiers. The tier
drives the process exit code, and the exit code drives the build.

## Datasets

The corpus is the product. Every record is a frozen, code-reviewed example in
the voice of a real service business (booking, billing, winback, scheduling,
memberships, refunds), each with a human-signed-off expected answer and the
phrases it must contain.

**`datasets/golden_sample.jsonl`: 40 golden records.** The "known good" set
the gate protects. Twelve categories:

| Category | Records | Category | Records |
|---|---|---|---|
| winback | 5 | billing | 4 |
| scheduling | 4 | membership | 4 |
| slot-rescue | 3 | reviews | 3 |
| no-show | 3 | refund | 3 |
| auto-repair | 3 | consult | 3 |
| faq | 3 | tone | 2 |

Each record carries two labels:

- `difficulty`: `standard` (33) or `edge` (7). The edge cases need date math,
  partial refunds, or a second no-show.
- `tags`: the category plus `smoke` on 15 records. `--tag smoke` runs the
  fast subset in CI; the full 40 run on releases.

**`datasets/redteam_sample.jsonl`: 12 adversarial records.** Each one
carries a deliberately *bad* `produced` answer: prompt injection, invented
prices and discounts, fabricated guarantees, a data-exfiltration probe, a
contradicted policy, a missing required disclaimer, a wrong client name, a
hostile tone, a confabulated service, and an indirect injection hidden inside
retrieved notes. The whole suite scores a mean composite of **0.20** and the
gate closes on it: 12 out of 12 records trip the per-record floor. If this
fixture ever passes, the evaluators have gone blind; a test asserts exactly
that.

**How `produced` and `expected` work.** Each record's `expected` is the
reviewer-approved reference answer. `produced` is the answer under test and
defaults to `expected`, so the dataset doubles as a self-consistency check out
of the box. In a live pipeline you overwrite `produced` with the model's fresh
output before scoring. The golden file never moves.

## Metrics, in standard vocabulary

The four dimensions map to the vocabulary senior eval teams use, split into
retrieval quality and generation quality:

- `grounding_score`   answer supported by retrieved context (Ragas
  ContextRelevance / ResponseGroundedness).
- `faithfulness_flag` no contradiction or invented facts (Ragas Faithfulness;
  Phoenix hallucination eval).
- `relevance_score`   answer covers the question (Ragas / Phoenix
  AnswerRelevancy and QA-correctness).
- `exact_keyword`     required phrases present, a deterministic hard floor
  (Ragas StringPresence; promptfoo `contains` / `regex`).

Deliberate design: the deterministic checks (`exact_keyword`, contradiction and
invented-number flags) run first and are reproducible; LLM-as-judge is reserved
for the free-form dimensions where it earns its cost and non-determinism. The
gate fails the build on a critical regression, the same shape as promptfoo
`fail-on-threshold` or DeepEval `assert_test(...)` under `deepeval test run`.

## How to run

No install. Python 3.9 or newer.

```bash
python -m harness.cli eval --dataset datasets/golden_sample.jsonl
```

Fast smoke subset for every commit (15 tagged records):

```bash
python -m harness.cli eval --dataset datasets/golden_sample.jsonl --tag smoke
```

Filter by difficulty, or combine both filters:

```bash
python -m harness.cli eval --dataset datasets/golden_sample.jsonl --difficulty edge
```

Run the adversarial suite and watch the gate close:

```bash
python -m harness.cli eval --dataset datasets/redteam_sample.jsonl
```

Run the tests (standard library, no pytest required):

```bash
python -m unittest discover -s tests
```

Save a baseline, then gate future runs against it:

```bash
python -m harness.cli baseline --dataset datasets/golden_sample.jsonl --out baseline.json
python -m harness.cli eval --dataset datasets/golden_sample.jsonl --baseline baseline.json
```

Write a machine-readable report alongside the scorecard:

```bash
python -m harness.cli eval --dataset datasets/golden_sample.jsonl --json report.json
```

Run the minimal example script:

```bash
python examples/run_eval_gate.py
```

The `eval` command exits `0` when the gate is open and `1` on a critical
regression. That exit code is the whole point: a CI job that runs this command
fails the build when the gate closes.

## Use it as a GitHub Action

The repo ships a reusable composite action (`action.yml`), so any repo can
gate on it without copying code:

```yaml
- uses: actions/checkout@v4
- uses: Victor-David-Medina/llm-eval-harness@v1
  with:
    dataset: datasets/golden_sample.jsonl
    baseline: baseline.json        # optional
    report-path: eval-report.json  # optional
```

Inputs are `dataset` (required), `baseline`, `report-path`, and
`python-version` (default 3.11). It needs nothing but Python: no API keys, no
install step, no network.

## Sample output

This is the real scorecard from running the shipped dataset:

```
============================================================
  llm-eval-harness scorecard
============================================================
  dataset      : datasets/golden_sample.jsonl
  records      : 40
  mean compos. : 0.757  [###############.....]

  dimension averages
    grounding     : 0.528  [###########.........]
    faithfulness  : 1.000  [####################]
    relevance     : 0.326  [#######.............]
    exact_keyword : 1.000  [####################]

  per record
    [ok ] lapsed-client-winback-tone composite 0.672
    [ok ] slot-rescue-same-day   composite 0.762
    [ok ] billing-question-grounded composite 0.833
    [ok ] refund-policy-faithful composite 0.844
    [ok ] auto-repair-estimate-recovery composite 0.735
    ...

  verdict
    severity   : INFO
    - no baseline supplied; gating on absolute floors only

  result       : PASS (gate open)
============================================================
```

And the real scorecard from the red-team suite, where every record carries a
bad answer and the gate closes:

```
  per record
    [LOW] r01  composite 0.169   (prompt injection obeyed)
    [LOW] r02  composite 0.142   (invented $40 discount)
    [LOW] r03  composite 0.037   (hallucinated 11pm closing)
    [LOW] r04  composite 0.150   (fabricated guarantee)
    [LOW] r05  composite 0.063   (exfiltration probe answered)
    [LOW] r06  composite 0.135   (policy contradicted)
    ...

  verdict
    severity   : CRITICAL
    - record r01 composite 0.17 below per-record floor 0.50
    - ...
    - run mean 0.20 below run floor 0.70

  result       : FAIL (gate closed, build blocked)
```

## Sample reports

Checked-in, machine-readable reports a reviewer can inspect without running
anything:

- `reports/eval-report-sample.json`: the full 40-record golden run (passes).
- `reports/eval-report-smoke-sample.json`: the 15-record smoke subset.
- `reports/redteam-report-sample.json`: the 12-record adversarial run (fails,
  as designed).

## Limitations

Stated plainly, because an eval harness that hides its blind spots is worse
than none:

- **Heuristics, not understanding.** Token overlap and negation checks catch
  mechanical failures (invented numbers, missing phrases, polarity flips).
  They do not catch a fluent answer that is wrong in a subtle way.
- **Small datasets.** Forty golden records and twelve adversarial ones are
  enough to gate a demo, not to claim statistical significance. The roadmap
  adds paired significance testing before any claim about "real" regressions.
- **No live model calls.** The harness scores answers you hand it; it does
  not call a model, run retrieval, or measure latency and cost. Those are
  adapters around this same contract, not replacements for it.
- **English tokenization.** The evaluators assume English word boundaries.
- **Relevance is the weakest dimension.** It is a simple overlap measure and
  scores low on good conversational answers. It is weighted lightly (0.15)
  for exactly that reason.

## Milestones

| Milestone | Status | What it is |
|---|---|---|
| M1: Deterministic gate | Shipped | stdlib evaluators, golden datasets, tiered regression, CI gate |
| M2: Corpus and presentation | This release | 40-record corpus, 12-record red-team suite, smoke/release splits, sample reports, reusable Action, Limitations |
| M3: Significance gating | Planned | per-record baseline diffs, paired significance before blocking, "which cases regressed" report |
| M4: Judgment with humility | Planned | optional LLM-as-judge adapters, two-judge agreement stats, human calibration set, cost and latency tracking |
| M5: Evidence over time | Planned | run history, trend dashboard, synthetic dataset generation |

The direction: keep the deterministic stdlib layer as the zero-dependency
core, and add optional adapters (judges, embeddings, history) around the same
stable contract: one score per dimension, one composite, one gate.

## How this maps to a real interview

Every serious LLM team runs some version of this loop. The pattern here is the
small, readable core of it:

- **Anthropic take-home and eval work.** A common prompt is "build something
  that catches model regressions." The grounding and faithfulness checks, the
  golden dataset, and the pass or fail gate are exactly that shape: measure,
  baseline, block on regression.
- **Arize Phoenix evaluators.** Phoenix ships evaluators for relevance,
  hallucination, and Q and A correctness over a dataset. This repo implements
  the same dimensions (grounding, faithfulness, relevance, keyword coverage)
  in pure Python so the logic is legible rather than hidden behind a service.
- **Glean and similar eval frameworks.** The production discipline is a
  curated golden set, a composite score, and a regression check wired into CI.
  This is that discipline, stripped to its frame so it fits in one reading.

The point of the repo in an interview is not that the evaluators are
state of the art. It is that I understand why a CI eval gate matters, where it
belongs in the pipeline, and what the honest tradeoffs are (see
`ADR-001-eval-gate.md`).

## Honest note

This is a standalone, clean-room extraction of a pattern, not a product. I am
Victor David Medina, a veteran and AI engineer. I run a fuller version of
this eval gate inside my own AI operations platform, where the same four
dimensions are computed with stronger signals: embeddings on Qdrant and
pgvector for grounding, an LLM-as-judge pass for faithfulness, traces in
Langfuse, golden datasets and regression detection wired into the build. That
platform is pre-revenue. It has one early unpaid spa pilot and a second family
pilot planned, and any recovery-rate figure it produces is a projected target,
not a measured result, until a pilot actually measures one.

This repo copies none of that platform's code. It re-implements the idea from
scratch with the standard library so the reasoning is fully visible and anyone
can run it. MIT licensed.

## Project structure

```
llm-eval-harness/
  action.yml                 reusable GitHub Action (this repo, as a gate)
  harness/
    cli.py                   the CI gate: eval, baseline, scorecard, --json
    evaluators.py            grounding, faithfulness, relevance, exact_keyword
    golden.py                typed JSONL loader (difficulty + tags labels)
    score.py                 weighted composite, tiered regression, verdict
  datasets/
    golden_sample.jsonl      40 golden records (15 smoke-tagged)
    redteam_sample.jsonl     12 adversarial records (bad produced answers)
  reports/
    eval-report-sample.json  checked-in golden run (passes)
    eval-report-smoke-sample.json  checked-in smoke run (passes)
    redteam-report-sample.json     checked-in adversarial run (fails)
  tests/test_evaluators.py   47 stdlib tests, incl. gate self-checks
  examples/run_eval_gate.py  minimal end-to-end example
  ADR-001-eval-gate.md       the design decision, with tradeoffs
  .github/workflows/eval.yml  CI: tests on 3.9/3.11/3.12, then the eval gate
```

## Links

- Proof repo (Terraform, AWS): https://github.com/Victor-David-Medina/aws-terraform-portfolio
- GitHub: https://github.com/Victor-David-Medina
- LinkedIn: https://linkedin.com/in/victor-david-medina
- Email: v.davidmedina@gmail.com
