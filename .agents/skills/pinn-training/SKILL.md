---
name: pinn-training
description: Train Physics-Informed Neural Networks (PINNs) - loss weighting between PDE residual, boundary, and initial conditions, collocation point sampling strategy, and diagnosing the specific failure modes (loss imbalance, spectral bias, gradient pathology) that make PINNs notoriously hard to train well. Use for forward or inverse PDE problems where you want a neural surrogate that respects the governing equation.
category: physics
---

# PINN Training

## Core idea
Train a neural network to satisfy a PDE by including the PDE residual (computed via
automatic differentiation), boundary conditions, and initial conditions all as loss terms,
rather than training on labeled solution data alone. This lets the network solve forward
problems (given the PDE, find the solution) or inverse problems (given some solution data,
find unknown PDE parameters) without a traditional mesh-based solver.

## The central failure mode: loss term imbalance
PINNs are notorious for one loss term (usually the PDE residual) dominating the others
during training, so the network satisfies the PDE almost everywhere while failing badly at
the boundary or initial condition — because a naive sum of loss terms with very different
natural scales lets the optimizer ignore whichever term has smaller gradients. Always:
- Track PDE loss, BC loss, and IC loss **separately** during training, not just the summed
  total — a decreasing total loss can hide one term stuck flat.
- Use an adaptive loss-weighting scheme (e.g. gradient-norm balancing, or periodically
  reweighting based on each term's relative magnitude) rather than fixed equal weights,
  unless a fixed weighting has already been tuned and validated for this exact problem.

## Collocation point sampling
- Uniform random sampling across the domain is a reasonable default, but under-resolves
  regions with sharp gradients (shocks, boundary layers, steep initial transients).
- For problems with known difficult regions, bias sampling density toward them (adaptive/
  residual-based resampling, where points are added where the PDE residual is currently
  largest, works well and is worth the extra implementation complexity for anything with a
  shock or steep front).
- Report the number of collocation points relative to problem complexity — an
  under-sampled domain is one of the most common causes of a PINN that "trains" (loss goes
  down) but doesn't actually solve the PDE well everywhere.

## Other known pathologies
- **Spectral bias** — neural networks tend to learn low-frequency components of a function
  much faster than high-frequency ones, which is a real problem for PDEs with fine
  structure. Consider Fourier feature embeddings of the input if the true solution has
  high-frequency content.
- **Gradient pathology from higher-order derivatives** — PDEs requiring second or higher
  derivatives (diffusion, wave equations) can produce ill-conditioned gradients during
  backprop-through-autodiff; if training stalls with no clear loss-imbalance explanation,
  this is worth checking via gradient statistics per layer.
- **Stopping too early** — a PINN's loss curve can plateau temporarily then continue
  improving; don't call training "converged" from a short flat stretch alone, check the
  trend over a longer window.

## Validation (mandatory, same standard as any physics result)
Compare against a traditional solver (finite difference/element/spectral) on the same
problem wherever one is tractable — a PINN result with no traditional-solver comparison
point is much harder to trust. Report PDE residual, BC loss, and IC loss all as final small
values, not just the summed loss. For inverse problems, report uncertainty on the
recovered parameters, not just a point estimate.

## Related skills
`neural-operator` for the related-but-distinct problem of learning a fast surrogate across
a *family* of PDEs/parameters rather than solving one specific instance. `pde-solver` for
the traditional-method baseline to validate against.
