---
name: quantum-computing
description: Build and simulate quantum circuits and quantum algorithms using Qiskit (general-purpose, gate-based, strong for algorithm design and hardware backends) or PennyLane (differentiable, built for hybrid classical-quantum ML). Covers when to reach for which, common simulation-vs-hardware pitfalls, and noise/decoherence considerations. Use for quantum algorithm work or quantum machine learning.
category: quantum
---

# Quantum Computing (Qiskit / PennyLane)

## Framework choice
- **Qiskit** — the standard for gate-based circuit design, algorithm implementation (Shor,
  Grover, VQE, QAOA), and running on real IBM quantum hardware or its noise-model
  simulators. Reach for this when the goal is the quantum algorithm/circuit itself, or
  when hardware execution (or realistic hardware-noise simulation) is part of the point.
- **PennyLane** — built around automatic differentiation of quantum circuits, designed for
  hybrid classical-quantum machine learning (variational quantum circuits as a
  differentiable layer inside a larger, otherwise-classical training loop, interoperable
  with PyTorch/JAX/TensorFlow autodiff). Reach for this when the quantum circuit is a
  component being trained end-to-end with gradient descent, not a standalone algorithm.
- Both can simulate the same circuits; the real choice driver is whether you need
  circuit-level differentiability integrated into a classical training loop (PennyLane) or
  algorithm design / hardware-realistic execution (Qiskit).

## Simulation vs. real hardware — know which you're reporting
A noiseless statevector simulation gives you the "ideal" quantum result — useful for
algorithm correctness verification, not representative of what current noisy hardware
would actually produce. Always state explicitly whether a result is from ideal simulation,
noise-model simulation, or real hardware execution — these are three different claims with
very different caveats, and conflating them (e.g. presenting an ideal-simulation result as
if it demonstrated hardware feasibility) overstates what was actually shown.

## Circuit design basics that are easy to get subtly wrong
- **Qubit ordering convention** differs between frameworks (and even between different
  parts of the same framework's documentation) — verify which end of the register is
  qubit 0 before interpreting measurement results, this is a very common source of
  "correct circuit, wrong answer" bugs.
- **Barrier/measurement placement** — measurements collapse superposition; if a circuit
  needs further coherent operations after a measurement (mid-circuit measurement, quantum
  error correction patterns), confirm the framework/backend actually supports that, not all
  simulators or hardware do.
- **Gate decomposition cost** — an algorithm expressed in terms of an idealized gate set may
  decompose into a much deeper circuit on real hardware's native gate set; report circuit
  depth *after* transpilation to the target backend's gate set, not just the abstract
  circuit's depth, if hardware feasibility is being claimed.

## Noise and decoherence
Real quantum hardware is noisy — decoherence, gate errors, readout errors all degrade
results, and error rates get meaningfully worse as circuit depth increases (more gates =
more accumulated error). For any claim about hardware feasibility, either run on a
realistic noise-model simulator or report actual hardware results — a noiseless-simulation
result presented without this caveat is a common way quantum computing claims get
overstated.

## Quantum machine learning specifics (PennyLane-typical)
- Variational quantum circuits used as ML layers face their own optimization
  pathology — **barren plateaus**, where gradients vanish exponentially with circuit
  width/depth for many circuit architectures and initialization schemes, making training
  effectively stall. If gradients are near-zero early in training with no clear bug, check
  circuit architecture and initialization against known mitigations (e.g. layer-wise
  training, specific initialization strategies) before assuming a data or implementation
  problem.
- Compare against a classical baseline of comparable parameter count — a hybrid
  quantum-classical model's advantage claim needs a fair classical comparison point, same
  discipline as any other ML evaluation (see `llm-evaluation`).

## Related skills
`sympy` for the underlying linear-algebra/tensor-product math when deriving a circuit by
hand before implementing it. `llm-evaluation` for baseline-comparison discipline that
applies equally to quantum-ML claims.
