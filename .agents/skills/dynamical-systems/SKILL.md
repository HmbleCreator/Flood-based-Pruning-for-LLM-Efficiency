---
name: dynamical-systems
description: Analyze the qualitative behavior of a dynamical system - fixed points, stability classification via eigenvalues, phase portraits, bifurcation diagrams, Lyapunov exponents for chaos detection. Use once you have a governing equation (discovered or hand-derived) and want to understand its long-term/qualitative behavior, not just simulate one trajectory.
category: physics
---

# Dynamical Systems Analysis

## Fixed points and stability
1. Find fixed points: solve dx/dt = 0 (symbolically with `sympy` if tractable, numerically
   otherwise).
2. Linearize around each fixed point (compute the Jacobian) and classify by eigenvalues:
   - All eigenvalues negative real part → stable node/focus.
   - Any eigenvalue positive real part → unstable (saddle if mixed signs, unstable
     node/focus if all positive).
   - Purely imaginary eigenvalues → center (linear analysis is inconclusive here; the
     nonlinear terms determine actual stability — don't report "stable" from linearization
     alone in this case).
3. Report the full classification for every fixed point found, not just the "interesting"
   one — a complete phase-space picture requires all of them.

## Phase portraits
Generate a phase portrait (2D: direct plot; higher-D: projections or Poincaré sections) by
integrating trajectories from a grid of initial conditions and plotting them together with
fixed points and their classifications marked. This is usually far more informative than
any single trajectory plot for understanding qualitative system behavior.

## Bifurcation analysis
When a system depends on a parameter, a bifurcation diagram (system's long-term behavior —
fixed point locations, periodic orbit amplitude, or chaotic attractor — plotted against the
parameter) reveals where qualitative behavior changes. Common types to recognize:
saddle-node (fixed points appear/disappear in pairs), pitchfork (symmetric splitting),
Hopf (fixed point stability changes as a limit cycle emerges). Sweep the parameter finely
near suspected transition points, coarsely elsewhere, to resolve the transition without
excessive compute.

## Chaos detection — Lyapunov exponents
A positive largest Lyapunov exponent indicates sensitive dependence on initial conditions
(chaos): two initially close trajectories diverge exponentially. Estimate numerically by
tracking the separation of two nearby trajectories over time (renormalizing periodically to
avoid numerical overflow) and fitting the exponential growth rate. Report the estimate with
its dependence on integration time and number of renormalization steps — a poorly-converged
Lyapunov exponent estimate is a common source of false chaos claims.

## Common pitfalls
- Classifying a fixed point as stable/unstable from eigenvalues with zero or purely
  imaginary real parts — linearization is inconclusive there; say so rather than guessing.
- Under-sampling phase space when generating a portrait, missing an attractor or a second
  set of fixed points entirely outside the sampled region.
- Reporting a Lyapunov exponent from too short an integration window to have converged.

## Related skills
Downstream of `sindy-identification` or `symbolic-regression` once you have a governing
equation to analyze. Feeds into `hamiltonian-mechanics` specifically for conservative
(energy-preserving) systems where symplectic structure matters for both analysis and
integration.
