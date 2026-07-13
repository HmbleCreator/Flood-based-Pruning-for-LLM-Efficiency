---
name: neural-operator
description: Train neural operators (Fourier Neural Operators, DeepONet, Graph Network Simulators) to learn fast surrogates across a family of PDEs or parameter settings, rather than solving one instance - resolution invariance, generalization across the parameter/geometry distribution, and evaluation against the traditional solver they're meant to replace. Use for particle-based or mesh-based physics surrogates, including learning physics from emergent particle-level interactions.
category: physics
---

# Neural Operator

## Core idea, and how it differs from a PINN
A PINN solves *one* instance of a PDE (one set of boundary/initial conditions,
parameters). A neural operator learns a mapping from a whole *family* of inputs
(parameters, initial conditions, geometries) to their corresponding solutions, so that
after training, evaluating a new instance is a fast forward pass instead of solving from
scratch. This is the right tool when you need many solves across varying conditions, not
one precise solve.

## Architecture families
- **Fourier Neural Operator (FNO)** — operates in Fourier space, naturally resolution-
  invariant (can train at one grid resolution and evaluate at another). Strong default for
  regular-grid PDE problems (fluid flow, weather-style fields).
- **DeepONet** — a branch/trunk architecture: one network encodes the input function, one
  encodes the query location, combined to predict the output function pointwise. More
  flexible with irregular geometries than a pure-FNO approach.
- **Graph Network Simulator (GNS)** — represents the system as a particle/node graph with
  message-passing between neighbors, learning local interaction rules that compose into
  global dynamics. This is the natural architecture for learning physics from
  particle-level interactions (granular flow, fluids, deformable materials) rather than
  hand-coding the governing PDE — the physics emerges from learned local update rules
  rather than being specified globally.
- **Equivariant architectures** (E(3)-equivariant GNNs, tensor-field networks) — worth the
  added complexity specifically when the physics has known symmetries (rotation/translation
  equivariance for 3D particle systems) that you want the architecture to respect *by
  construction* rather than hoping the model learns from data alone. Baking in a known
  symmetry generally improves sample efficiency and generalization over learning it from
  scratch.

## Generalization is the entire point — test it directly
Report accuracy specifically as a function of *distance from the training distribution* in
parameter/geometry space, not just an aggregate test-set number drawn from the same
distribution as training. A neural operator that performs well only very close to training
conditions has limited value over just running the traditional solver more times.

## Evaluation
- Compare wall-clock time and accuracy against the traditional solver being replaced — the
  entire value proposition is a speed/accuracy tradeoff, so both numbers need reporting
  together, not accuracy alone.
- Check rollout stability for time-dependent systems: does repeated autoregressive
  application (predict next state, feed back in, repeat) stay bounded and physical, or does
  error accumulate and blow up? This is a distinct and common failure mode from single-step
  accuracy looking fine.
- For particle/graph-based simulators, validate conservation properties (mass, momentum,
  energy as applicable) hold over long rollouts, not just early timesteps — see
  `conservation-law-discovery` for how to check this rigorously.

## Common pitfalls
- Training only on narrow initial-condition coverage, then reporting broad claims about the
  learned surrogate's generality — same caveat as `sindy-identification` and
  `conservation-law-discovery`.
- No baseline comparison against the traditional solver's own runtime — a neural operator
  that's not meaningfully faster than the thing it replaces isn't earning its added
  complexity (see `AGENTS.md` prime directive 3, simplicity).
- Reporting only single-step prediction error for what's actually meant to be used
  autoregressively — always report rollout error over the intended usage horizon.

## Related skills
`pinn-training` for the single-instance-solve alternative. `conservation-law-discovery` for
validating physical properties of the learned model. Hand off to `agents/ml-engineer.md`
for the broader model-training loop (data pipeline, evaluation harness, reproducibility)
once the architecture choice is made.
