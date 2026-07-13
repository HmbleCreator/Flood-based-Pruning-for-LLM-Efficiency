---
name: conservation-law-discovery
description: Discover conserved quantities and symmetries in a dynamical system from simulation or observation data - polynomial null-space search, Lyapunov function candidates, Noether-style symmetry analysis. Use when trying to recover a physical law (like F=ma) from raw data rather than assuming it, or when validating that a simulation conserves what it should.
category: physics
---

# Conservation Law Discovery

Directly relevant to any "learn the physics from data instead of hand-coding the PDE"
project — recovering F=ma, energy conservation, or an unknown invariant from
particle-level simulation traces without assuming the functional form up front.

## Core approaches

**1. Polynomial null-space search (most common starting point)**
Build a candidate library of polynomial terms in the state variables (positions,
velocities, momenta, and low-order products/powers of them), evaluate d/dt of each
candidate along observed trajectories, and search for a linear combination whose time
derivative is ~0 across all trajectories. That combination is a conserved quantity
candidate. This is fast, interpretable, and a good first pass before reaching for
anything more complex.

**2. Lyapunov function candidates**
For systems where you suspect a monotonic (not strictly constant) quantity — e.g. energy
dissipating toward equilibrium rather than being exactly conserved — search instead for a
candidate function whose derivative has a *consistent sign* along trajectories, not
exactly zero. This catches near-conservation and dissipative structure that a strict
null-space search would miss entirely.

**3. Noether-style symmetry analysis**
If the system has an identifiable continuous symmetry (translation invariance, rotation
invariance, time invariance), the corresponding conserved quantity follows directly
(momentum, angular momentum, energy respectively) — check for symmetry in the
data-generating process itself before searching numerically; it's a much stronger and
more interpretable result than a fitted null-space combination.

## Validation — do not skip
A "discovered" conserved quantity must be checked against:
- **Multiple independent trajectories/seeds**, not just one run — a quantity that's
  constant on one trajectory by coincidence will vary on another.
- **Different initial conditions spanning the state space**, not just nearby ones —
  narrow initial-condition coverage is the most common source of a false-positive
  invariant.
- **Held-out data never used in the search** — fit the candidate combination on a subset,
  verify constancy on a disjoint subset.
- **Numerical near-zero, not exactly zero**: report the residual variance of the candidate
  quantity along held-out trajectories, and state the tolerance used to call it "conserved."

## Common failure modes
- **Noisy single-trajectory R² fooling the search** — a candidate can look conserved on
  noisy short trajectories purely because the trajectory didn't explore enough of the state
  space to reveal variation. Require validation across trajectory *sets*, and consider
  bootstrap resampling across trajectories to estimate how stable the discovered candidate
  is.
- **Evicting a real joint invariant because of a noisy univariate pre-filter** — if
  screening candidate terms one at a time before searching combinations, a term that only
  matters jointly with another can get filtered out early. Validate joint candidate sets,
  not just individually-screened terms, before discarding.
- **Overfitting the polynomial library** — an overly rich candidate library (high-degree
  terms, many cross-products) will always find *something* with near-zero derivative on
  finite data; prefer the simplest library that captures the suspected physics, and
  penalize complexity explicitly (e.g. via a sparsity term) rather than allowing arbitrary
  polynomial degree.

## Workflow
1. Simulate/collect multiple trajectories spanning a broad region of the state space.
2. Build a candidate library (start simple: linear + quadratic terms in state variables).
3. Search null-space (or Lyapunov-sign, or symmetry) for the combination(s) with smallest
   time-derivative variance.
4. Validate on held-out trajectories, report residual variance and confidence.
5. If a real physical name/interpretation exists for the discovered quantity (energy,
   angular momentum), state it; if not, report it as a discovered but uninterpreted
   invariant rather than forcing an interpretation.

## Related skills
Pairs naturally with `sindy-identification` (discover the governing equations themselves,
not just an invariant of them) and `symbolic-regression` (fit a closed-form expression to
the discovered relationship). Hand results to `agents/ml-engineer.md` if the next step is
folding the discovered law into a learned model (e.g. a Graph Network Simulator or an
equivariant architecture that respects the discovered symmetry by construction).
