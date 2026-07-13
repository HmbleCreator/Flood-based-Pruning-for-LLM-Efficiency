---
name: llm-evaluation
description: Evaluate LLM/generative outputs rigorously - rubric design, LLM-as-judge setup and its known biases, academic harness usage, and reporting metrics with eval-set size and variance rather than a single vibes-based number. Use whenever a claim rests on "the model is better now."
category: ml-training
---

# LLM Evaluation

## Vibes are not evidence
"It feels better" is not a result. Every quality claim about generative output needs
either a documented rubric with explicit criteria, an academic benchmark/harness run, or
an LLM-as-judge setup with a fixed, inspectable prompt - and the eval-set size reported
alongside the score.

## LLM-as-judge: known failure modes
- **Position bias** - judges tend to favor whichever response is shown first/second
  consistently; randomize order across trials and check the effect isn't driving the
  result.
- **Verbosity bias** - judges often rate longer answers as better independent of quality;
  control for length or explicitly instruct the judge to ignore it.
- **Self-preference** - a judge model tends to rate outputs from its own model family more
  favorably; where possible, use a judge from a different family than the system under
  test, or report this as a caveat if not feasible.
- **Judge prompt matters as much as the metric** - two judge setups with different rubric
  wording can produce meaningfully different rankings on the same outputs; treat the judge
  prompt as part of the experimental method, document it, don't just say "we used
  LLM-as-judge."

## Baselines are mandatory
A metric with no baseline is not evidence of improvement. Always report: the previous
system version, a naive baseline (e.g. simple retrieval, majority answer, un-fine-tuned
base model), and, where relevant, a frontier-API reference point.

## Reporting standard
- State eval-set size (N) - a 20-example eval and a 2,000-example eval should not be
  presented with the same confidence.
- Report variance across seeds/runs where feasible, not a single point estimate, especially
  for anything involving sampling temperature > 0.
- Include a handful of real transcripts/examples alongside the aggregate score - an
  aggregate number can hide a system that's failing in one specific, important way.

## Metric gaming
A system that scores well on the chosen metric but fails spot-check error analysis on real
examples is not done - go read actual outputs, not just the aggregate. This is especially
important for LLM-as-judge scores, which are easier to game (via verbosity, hedging
language, format matching what a judge expects) than a hard benchmark.
