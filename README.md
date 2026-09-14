# llm-eval-harness

[![eval-gate](https://github.com/Victor-David-Medina/llm-eval-harness/actions/workflows/eval.yml/badge.svg)](https://github.com/Victor-David-Medina/llm-eval-harness/actions/workflows/eval.yml) [![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/License-MIT-44403C)](LICENSE)

A small, honest, stdlib-only LLM evaluation harness that gates CI. It scores
model output against checked-in golden datasets, detects regressions in tiers,
and fails the build when quality drops past a critical line.

If you are here from a job application: this repo is my public proof of eval
discipline. The same pattern runs in production inside my AI operations
platform (see Honest note below).

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
  datasets/golden_sample.jsonl        (frozen, code-reviewed examples)
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

## Sample output

This is the real scorecard from running the shipped dataset:

```
============================================================
  llm-eval-harness scorecard
============================================================
  dataset      : datasets/golden_sample.jsonl
  records      : 11
  mean compos. : 0.757  [###############.....]

  dimension averages
    grounding     : 0.590  [############........]
    faithfulness  : 1.000  [####################]
    relevance     : 0.203  [####................]
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

And when a dataset regresses (an invented price, a missing required phrase),
the gate closes and the build stops:

```
  verdict
    severity   : CRITICAL
    - record broken-invented-price composite 0.15 below per-record floor 0.50
    - run mean 0.23 below run floor 0.70
    - composite dropped 0.532 vs baseline 0.757 (critical)

  result       : FAIL (gate closed, build blocked)
```

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
- **Forward-deployed debugging.** An engineer embedded with a customer often
  hears "the answers got worse after Tuesday's deploy." The per-record floors
  and the baseline diff turn that into a five-minute answer: which records
  dropped, by how much, and whether the gate should have blocked it.

The point of the repo in an interview is not that the evaluators are
state of the art. It is that I understand why a CI eval gate matters, where it
belongs in the pipeline, and what the honest tradeoffs are (see
`ADR-001-eval-gate.md`).

## Honest note

This is a standalone, clean-room extraction of a pattern, not a product. I am
Victor David Medina, a veteran engineer. I run a fuller version of
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

## Links

- Proof repo (Terraform, AWS): https://github.com/Victor-David-Medina/aws-terraform-portfolio
- GitHub: https://github.com/Victor-David-Medina
- LinkedIn: https://linkedin.com/in/victor-david-medina
- Email: v.davidmedina@gmail.com
