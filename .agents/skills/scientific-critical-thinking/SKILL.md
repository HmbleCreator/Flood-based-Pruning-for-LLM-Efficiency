---
name: scientific-critical-thinking
description: Evaluate research rigor - spot unsupported claims, confounds, statistical misuse, and logical leaps in your own or someone else's work. Use before finalizing SYNTHESIZE, before submitting a paper, or whenever a result seems surprisingly clean.
category: research
---

# Scientific Critical Thinking

## Core stance
Default to skepticism about your own results, especially ones that confirm what you
expected or that look unusually clean. The question isn't "does this support my
hypothesis" but "what else could explain this."

## Checklist
**Claims vs. evidence**
- Does every claim have a specific number, test, or citation behind it - not just "clearly"
  or "it is well known that"?
- Is causal language ("causes", "leads to") justified by the actual study design, or is
  this really a correlation being oversold?
- Are comparative claims ("better than", "outperforms") naming both the baseline and the
  metric, not just asserting superiority?

**Confounds & alternative explanations**
- What else varies between your conditions besides the thing you're claiming causes the
  effect?
- If the result is surprising, what boring explanation (bug, leakage, small N, selection
  effect) should be ruled out first, before a novel one is entertained?

**Statistical hygiene**
- Was the analysis plan set before seeing the data, or chosen after (p-hacking risk)?
- Are effect sizes and uncertainty reported, not just a point estimate or a p-value alone?
- Does the test match the data (paired vs unpaired, parametric vs not)?

**Generalization**
- Is the claim scoped to the data/domain actually studied, or does the language imply
  broader generality than what was tested?
- Are limitations stated honestly, not as boilerplate at the very end and then ignored in
  the abstract?

## How to use this
Apply it as a self-check before writing SYNTHESIZE, and again as the mindset behind the
`agents/critique.md` and `agents/reviewer.md` personas in this package - both are built to
apply exactly this skepticism to an artifact, either during the process (critique) or
after the fact against the evidence trail (reviewer).
