---
name: hamiltonian-mechanics
description: Simulate Hamiltonian (energy-conserving) systems correctly using symplectic integrators - leapfrog, Yoshida higher-order methods, N-body simulation, and why a standard non-symplectic integrator (RK4, Euler) silently leaks or gains energy over long integrations even when it looks accurate short-term. Use for orbital mechanics, N-body, or any conservative system where long-term energy behavior matters.
category: physics
---

# Hamiltonian Mechanics & Symplectic Integration

## Why this needs a dedicated integrator family
A general-purpose ODE integrator (RK4, adaptive-step methods) can look highly accurate
over a short integration but will systematically drift in total energy over long
integrations of a Hamiltonian system — the error doesn't stay bounded, it accumulates
directionally. Symplectic integrators are specifically constructed to preserve the
phase-space structure of Hamiltonian systems, keeping energy error *bounded* (oscillating
around the true value) rather than drifting, even though their per-step local error is not
necessarily smaller than a high-order Runge-Kutta method.

**Rule of thumb: if you're integrating a conservative system for many periods and care
about long-term qualitative behavior (is the orbit stable? does it stay bounded?), reach
for a symplectic method, not RK4 — regardless of RK4's higher formal order.**

## Method choices
- **Leapfrog (velocity Verlet)** — 2nd order, extremely simple to implement, the standard
  default for N-body and molecular dynamics. Good starting point for almost any Hamiltonian
  simulation task.
- **Yoshida 4th order** — composes leapfrog steps with carefully chosen sub-step sizes to
  achieve 4th-order accuracy while remaining symplectic. Worth the added complexity when
  leapfrog's 2nd-order accuracy isn't sufficient and long-term energy conservation still
  matters (a naive 4th-order RK would abandon the symplectic property entirely).

## N-body specifics
- Direct pairwise force summation is O(N²) — fine for small N (up to a few thousand
  bodies), but reach for a tree-based method (Barnes-Hut) or particle-mesh method for
  larger N, since the physics doesn't change but the compute budget does.
- Close encounters (two bodies passing very near each other) cause the force to spike and
  can blow up a fixed-timestep integrator; either use adaptive timestepping (carefully, to
  preserve the symplectic property — not all adaptive schemes remain symplectic) or
  regularize the force at short range if the physical problem tolerates it.

## Mandatory validation
1. **Energy conservation over the full integration window** — plot total energy vs. time;
   for a correctly-implemented symplectic integrator, it should oscillate around a constant
   value with bounded amplitude, not drift monotonically. A monotonic drift indicates either
   a bug or a timestep too large for the dynamics.
2. **Angular momentum conservation** where the system has rotational symmetry — an
   independent check from energy, catches different classes of bugs.
3. **Timestep convergence** — halve the timestep, confirm the qualitative trajectory and
   energy-oscillation amplitude improve (shrink) rather than staying fixed or growing.
4. Compare against a known analytical solution where one exists (two-body Kepler orbits
   have closed-form solutions — an excellent sanity check before trusting an N-body code on
   a genuinely unsolvable configuration).

## Related skills
`dynamical-systems` for phase-space/stability analysis once you have the equations of
motion. `conservation-law-discovery` if the goal is to verify or discover *which*
quantities are conserved rather than assuming energy/momentum a priori. `astropy` for
astronomical coordinate systems and constants when the N-body system is a real astronomical
one.
