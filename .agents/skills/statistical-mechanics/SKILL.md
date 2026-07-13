---
name: statistical-mechanics
description: Simulate and analyze statistical-mechanical systems - Monte Carlo methods (Metropolis, Wolff cluster), phase transitions, finite-size scaling, critical exponents, and thermodynamic ensemble consistency (NVE/NVT/NPT). Relevant beyond physics proper for anything modeling emergent macroscopic behavior from local microscopic rules, including heat transfer and material-property learning.
category: physics
---

# Statistical Mechanics

## Monte Carlo methods
- **Metropolis algorithm** — the standard local-update MCMC method for sampling from a
  Boltzmann distribution: propose a local change (e.g. flip one spin), accept with
  probability min(1, exp(-ΔE/kT)). Simple, general, but suffers from critical slowing down
  near a phase transition (autocorrelation time diverges, so naive Metropolis becomes very
  inefficient exactly where you most want good statistics).
- **Wolff cluster algorithm** — updates whole clusters of aligned spins at once rather than
  single sites, dramatically reducing autocorrelation time near criticality. Worth the
  added implementation complexity specifically when studying behavior near a phase
  transition; unnecessary overhead far from it.

## Ensemble consistency — state which one explicitly
NVE (fixed particle number, volume, energy — microcanonical), NVT (fixed N, V, temperature
— canonical), NPT (fixed N, pressure, temperature — isothermal-isobaric) are genuinely
different thermodynamic setups with different fluctuation statistics; a result computed
under one ensemble's statistics reported as if it were another is a real correctness bug,
not a technicality. State the ensemble explicitly and make sure the simulation's actual
update rule matches it (e.g. NVT requires a thermostat; a plain energy-conserving
integrator alone gives you NVE, not NVT).

## Phase transitions and finite-size scaling
Real phase transitions are sharp only in the thermodynamic limit (infinite system size);
any finite simulation shows a *rounded* transition. Finite-size scaling theory relates how
observables (e.g. specific heat peak height/width, order parameter) depend on system size
near the transition, and lets you extrapolate finite-size simulation results toward the
true thermodynamic-limit critical point and critical exponents. Simulating only one system
size and reporting its transition point as *the* critical point (without extrapolation) is
a common and avoidable error.

## Critical exponents
Near a continuous phase transition, thermodynamic quantities follow power laws in
(T − T_c) with characteristic exponents (α, β, γ, ν, ...) that are often *universal* —
shared across microscopically different systems in the same universality class. Fitting
these requires simulating multiple system sizes, extracting the scaling behavior, and
being honest about the fit's uncertainty — critical exponent estimation from finite,
noisy Monte Carlo data is genuinely hard and easy to overstate the precision of.

## Relevance beyond textbook physics
The core idea — macroscopic, emergent behavior arising from simple local microscopic
update rules, studied via ensemble sampling rather than solving exact equations of motion
— generalizes directly to: heat/thermal transport modeled at a particle level rather than
via a hand-coded heat-diffusion PDE, material-property learning where a bulk property
emerges from local interaction rules, and any agent-based or cellular-automaton-style
model where you care about statistical/ensemble behavior rather than one exact trajectory.

## Common pitfalls
- Reporting Metropolis results near a phase transition without addressing critical slowing
  down (either via cluster updates or an explicit autocorrelation-time analysis and
  correspondingly long runs).
- Extracting critical exponents from a single system size.
- Mismatched ensemble between the simulation's actual dynamics and the statistics being
  reported.

## Related skills
`fluid-dynamics`/`fluidsim` for continuum-level (as opposed to particle/ensemble-level)
transport. `conservation-law-discovery` for verifying which quantities are actually
conserved by a given update rule before assuming an ensemble is correctly implemented.
