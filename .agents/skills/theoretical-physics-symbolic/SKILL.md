---
name: theoretical-physics-symbolic
description: Symbolic tensor calculus for general relativity and string-theory-adjacent work - metric tensors, Christoffel symbols, curvature tensors, geodesics via SymPy's diffgeom/tensor modules or EinsteinPy, and an honest note on how thin the mainstream tooling actually is for string theory specifically compared to numerical-relativity/GR. Use for relativity, differential geometry, or theoretical high-energy-adjacent symbolic derivations.
category: research
source: authored-for-this-package
---

# Theoretical Physics Symbolic Computation

**Note on origin**: OpenScience's skill library doesn't include a dedicated GR/string-
theory skill — this one was written to fill that gap, since it's a real gap in the
mainstream open-source ecosystem too, not just in OpenScience specifically (see the honesty
note below). Treat this as a starting map, not a mature, battle-tested skill the way the
physics ones ported from OpenScience are.

## General relativity / differential geometry — tooling that actually exists
- **`sympy.diffgeom`** — SymPy's differential geometry module: manifolds, coordinate
  patches, tensor fields, covariant derivatives. Low-level but fully symbolic and free of
  external dependencies beyond SymPy itself; good for deriving results by hand-verification
  or for small, specific calculations.
- **EinsteinPy** — purpose-built for GR: symbolic computation of Christoffel symbols,
  Riemann/Ricci/Einstein tensors from a metric, geodesic integration, and some standard
  spacetimes (Schwarzschild, Kerr) pre-defined. The more practical starting point than raw
  `sympy.diffgeom` for anything GR-specific — verify current API against its documentation
  with a fresh search before use, since symbolic-GR libraries are a smaller ecosystem with
  less API stability guarantee than something like NumPy.
- **`cadabra2`** — a computer-algebra system built specifically for tensor/field-theory
  calculations in high-energy physics and GR (index gymnastics, tensor simplification under
  symmetries). More specialized and less commonly maintained than the two above; worth
  knowing exists if `sympy`/EinsteinPy prove insufficient for a specific tensor
  manipulation.

## Core workflow for a GR calculation
1. Define the metric tensor (as a `sympy` matrix in a chosen coordinate system).
2. Compute Christoffel symbols (from the metric and its derivatives).
3. Compute curvature (Riemann tensor, then contract for Ricci tensor and scalar
   curvature).
4. Verify symbolically before trusting downstream results: contract the Bianchi identity,
   check known limiting cases (e.g. the metric should reduce to flat Minkowski space in an
   appropriate limit — verify this symbolically, don't just assume the metric was entered
   correctly).
5. For geodesics: derive the geodesic equation from the Christoffel symbols, then
   numerically integrate (hand off to `ode-solver`) for actual trajectories.

## Honesty note: string theory specifically
There is no mainstream, actively-maintained Python equivalent to Qiskit-for-quantum-
computing or EinsteinPy-for-GR that covers string theory broadly (worldsheet CFT,
compactification geometry, string amplitude computation) — this is a genuine gap in the
open tooling ecosystem, not a gap specific to OpenScience or this package. Most real
string-theory computation is done with specialized, often not-publicly-released research
code, or with general computer-algebra systems (Mathematica packages like `SUGRA` or
`Cadabra`) rather than a Python-first open-source stack. If a string-theory-adjacent task
comes up:
- For anything reducible to differential geometry / tensor calculus on a target-space
  metric (compactification geometry, moduli space computations), the GR tooling above
  still applies directly.
- For worldsheet CFT / amplitude-level string calculations specifically, expect to do more
  hand-derivation with `sympy` as a symbolic-algebra assistant rather than finding an
  existing high-level library — verify this assessment with a fresh search before
  committing to an approach, since tooling changes faster than this note will stay current.

## Common pitfalls
- Entering a metric with a sign-convention or coordinate-order error that propagates
  silently through Christoffel/Riemann computation — always verify against a known
  limiting case before trusting a novel result.
- Confusing coordinate singularities (an artifact of the chosen coordinate system, like the
  Schwarzschild coordinate singularity at the event horizon) with genuine physical
  (curvature) singularities — check a coordinate-invariant curvature scalar, not just
  whether a metric component blows up, before claiming a physical singularity.
- Treating index conventions (upper vs. lower, summation convention) inconsistently across
  a derivation — this is the single most common source of sign errors in tensor
  calculations, symbolic or by hand.

## Related skills
`sympy` (standard library skill) for the underlying symbolic-algebra engine.
`hamiltonian-mechanics` for the classical-mechanics side of a relativistic system in the
non-relativistic or weak-field limit, where it's often useful as a sanity-check comparison.
`dimensional-analysis` for verifying a derived expression's dimensions before trusting it.
