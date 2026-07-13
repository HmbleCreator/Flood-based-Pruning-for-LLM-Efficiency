---
name: sindy-identification
description: Recover governing differential equations from time-series data using Sparse Identification of Nonlinear Dynamics (SINDy/PySINDy) - library construction, sparsity threshold sweeps, handling noisy data, and validating discovered equations against held-out trajectories. Use when the goal is "find the ODE/PDE that produced this data" rather than assuming a known form.
category: physics
---

# SINDy Identification (Sparse Identification of Nonlinear Dynamics)

## Core idea
Given time-series state data x(t) and (numerically estimated or measured) derivatives
dx/dt, build a library of candidate nonlinear functions of the state (polynomials,
trigonometric terms, etc.), then solve for a *sparse* linear combination of library terms
that reproduces the observed derivative. Sparsity is the key regularizer — it's what
turns "some combination fits" into "this specific compact equation governs the system."

## Library construction
- Start with a low-degree polynomial library (degree 2-3) plus trig terms if periodicity
  is suspected; richer libraries increase overfitting risk and computational cost without
  necessarily improving discovery.
- If domain knowledge suggests specific functional forms (e.g. an inverse-square term for
  a suspected gravitational system), include them explicitly rather than hoping a
  polynomial expansion approximates them — SINDy recovers exact sparse combinations of
  what's *in* the library, not the shape it needs but doesn't have.
- Normalize/rescale state variables with very different magnitudes before fitting — an
  unscaled library biases the sparsity threshold toward whichever variable happens to have
  larger numeric range.

## Threshold sweep — do not pick one threshold and stop
The sparsity threshold controls the tradeoff between model complexity and fit quality.
Sweep across a range of thresholds and plot the resulting model complexity vs. fit error
(a Pareto front) rather than reporting a single arbitrarily-chosen threshold. The "elbow"
of that curve is usually the more defensible choice than either extreme.

## Handling noisy data
Raw finite-difference derivatives amplify noise badly. Options, roughly in order of how
much noise they tolerate:
- Smooth the state trajectory (Savitzky-Golay or similar) before differentiating.
- Use SINDy variants built for noise directly (e.g. weak-form SINDy, which avoids explicit
  pointwise derivative estimation by integrating against test functions).
- If derivatives are measured directly (not finite-differenced), noise handling is much
  less of a concern — state this explicitly in the report either way.

## Validation
- **Held-out trajectories, not held-out timepoints from the same trajectory** — timepoints
  within one trajectory are highly correlated; a real test is a trajectory from a different
  initial condition the model never saw during fitting.
- **Simulate forward from the discovered equations** and compare against the true
  trajectory, not just compare instantaneous derivative predictions — a model can match
  derivatives locally while diverging badly when integrated forward.
- **Check coefficient stability across noise realizations/bootstrap resamples** — a
  genuinely-discovered term should survive resampling; a term that appears only in one fit
  is likely noise-fit, even if its coefficient looked significant in that one run.

## Common pitfalls
- Reporting a discovered equation without ever simulating it forward to check it's
  actually stable/bounded — a fitted model can have small pointwise derivative error but
  be numerically unstable when integrated.
- Treating discovery on a single, short, or narrow-regime trajectory as generalizable —
  see the same trajectory-coverage caveat as `conservation-law-discovery`.
- Forgetting to report which terms survived across a threshold sweep vs. which appeared
  only at one specific setting — the former is a real finding, the latter is likely a
  sparsity-threshold artifact.

## Related skills
Pairs with `conservation-law-discovery` (verify the discovered equation actually implies
the conserved quantities you'd expect) and `dynamical-systems` (analyze the discovered
equation's fixed points, stability, and phase portrait once you have it).
