---
name: dimensional-analysis
description: Validate physical/engineering results with dimensional consistency checks and derive scaling relationships via the Buckingham Pi theorem - non-dimensionalization, catching unit errors before they become silent numerical bugs, and characterizing a problem by its dimensionless numbers before choosing a numerical method. Use as an early and a final check on any physics/engineering computation.
category: physics
---

# Dimensional Analysis

## Use this both early and late
Early: characterize a new problem by computing its relevant dimensionless numbers before
choosing a numerical method — the regime (e.g. Reynolds number range) determines which
methods and approximations are even valid. Late: as a final sanity check on any result —
every reported quantity should have dimensions that make sense for what it claims to be,
checked explicitly, not just assumed correct because the code ran without error.

## Buckingham Pi theorem
Given a physical relationship among n variables involving k independent fundamental
dimensions (mass, length, time, temperature, etc.), the relationship can be rewritten in
terms of exactly n−k independent dimensionless groups. This is the formal tool for: (a)
reducing the number of independent variables in an experiment/simulation design, (b)
deriving scaling laws without solving the full governing equations, (c) checking that a
proposed relationship isn't missing a variable or including a redundant one.

## Common dimensionless numbers and what they characterize
- **Reynolds number** Re = ρvL/μ — inertial vs. viscous forces; determines laminar vs.
  turbulent flow regime.
- **Mach number** Ma = v/c — flow speed vs. sound speed; determines compressibility
  importance.
- **Prandtl number** Pr = ν/α — momentum vs. thermal diffusivity; relevant for any coupled
  fluid-thermal problem.
- **Rayleigh number** Ra = gβΔTL³/(να) — buoyancy-driven vs. diffusive transport;
  determines whether natural convection is significant.
- **Knudsen number** Kn = λ/L — mean free path vs. system size; determines whether a
  continuum (PDE) description is even valid, or whether a particle-based/kinetic
  description is needed instead.
- **Péclet number** Pe = vL/D — advection vs. diffusion; determines which transport
  mechanism dominates.

Computing the relevant dimensionless numbers for a new problem *before* choosing a
numerical method is usually the single fastest way to avoid picking an inappropriate
method (e.g. attempting a continuum PDE solve in a regime where Kn indicates continuum
assumptions break down).

## Practical validation workflow
1. Before writing simulation code: list every input/output quantity with its units;
   confirm the governing equation is dimensionally consistent term-by-term (every additive
   term in an equation must have the same dimensions — this catches transcription errors
   immediately, before any numerics run).
2. Use `pint` (or manual tracking) to carry units through a calculation programmatically
   rather than trusting that unit consistency was maintained by hand.
3. After computing a result: check its dimensions match what's claimed, and sanity-check
   its order of magnitude against a rough physical estimate — a result off by many orders
   of magnitude from a back-of-envelope estimate usually indicates a unit conversion bug
   (a very common one: forgetting a unit-system conversion between SI and CGS, or between
   degrees and radians).

## Related skills
Feeds directly into method selection in `physics.md`'s methodology routing table. Pairs
with `symbolic-regression` (constrain candidate expressions to be dimensionally valid,
dramatically shrinking the search space) and `physics-fitting` (report fitted parameters
with correct units, not just bare numbers).
