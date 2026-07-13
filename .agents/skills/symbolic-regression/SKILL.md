---
name: symbolic-regression
description: Fit closed-form symbolic expressions to data using genetic-programming-based search (e.g. PySR) - operating on the Pareto front between accuracy and expression complexity, physics-constrained search (dimensional consistency, known symmetries), and avoiding overfit "accurate but meaningless" expressions. Use when you want an interpretable equation, not a black-box fit.
category: physics
---

# Symbolic Regression

## Core idea
Search the space of mathematical expressions (not a fixed parametric family) for ones
that fit the data, using genetic programming or similar evolutionary search over
operators (+, -, *, /, sin, exp, ...) and terminals (variables, constants). The output is
a human-readable equation, not a black-box model — that's the entire point of reaching
for this over a neural fit.

## The Pareto front is the actual result, not a single equation
Symbolic regression naturally produces a family of candidate expressions trading off
accuracy against complexity (number of operators/terms). Report the *whole front*, not
just the single lowest-error expression — the lowest-error one is very often needlessly
complex and overfit, while a slightly-higher-error, much-simpler expression a few points
down the front is usually the more scientifically meaningful result. Let a human (or the
downstream physics interpretation) pick the point on the front, don't auto-select purely
on error.

## Physics-constrained search
Where possible, constrain the search rather than searching blind:
- **Dimensional consistency** — restrict candidate expressions to ones that are
  dimensionally valid given the units of the input variables; this eliminates a huge swath
  of the search space that's guaranteed to be non-physical, and dramatically improves
  sample efficiency.
- **Known symmetries** — if the target quantity is known to be invariant under some
  transformation (e.g. symmetric in two variables, invariant under a sign flip), restrict
  or bias the search toward expressions respecting that symmetry rather than discovering it
  the hard way from data alone.
- **Known limiting behavior** — if the true relationship is known to have a specific limit
  (e.g. approaches zero as x→0, or approaches a known asymptote), penalize candidates that
  violate the known limit even if they fit the bulk of the data well.

## Avoiding a fit that's accurate but meaningless
A high-degree polynomial or an expression stacking many operators can achieve very low
error on finite noisy data without capturing anything real about the underlying process —
this is the classic overfitting failure mode, just in symbolic form instead of parameter
count. Guard against it by: validating on held-out data spanning a different regime than
training data, preferring the simpler expression on the Pareto front unless the more
complex one earns its keep with a real accuracy jump, and checking that the discovered
expression's behavior outside the training data range is physically sane (doesn't blow up,
doesn't do something absurd at a boundary).

## Workflow
1. Prepare clean input/target data; note the physical units of every variable.
2. Constrain the operator set to physically plausible ones for the domain (e.g. no
   arbitrary exponentials for a problem known to be polynomial).
3. Run the search, producing the Pareto front of accuracy vs. complexity.
4. Validate the front's candidates on held-out data spanning a distinct regime.
5. Report the front, not a single equation — state which point you'd recommend and why
   (usually: simplest expression within some acceptable error margin of the best).

## Related skills
Often the final step after `sindy-identification` or `conservation-law-discovery` have
identified *which* terms matter — symbolic regression then finds the precise closed form.
For validating a discovered relationship physically, hand off to `physics-critique.md`.
