# Flood Research Program — Future Roadmap

This document outlines the planned direction for the **Flood** research program.

---

## Research Progression

Each paper asks a natural follow-on question from the previous one:

| Paper | Title | Core Question | Status |
|---|---|---|---|
| 1 | *Invisible Bridges* | Which heads matter? | ✅ Complete |
| 2 | *Bridge Dependency Networks* | Who depends on whom? | ⬜ Proposal ready |
| 3 | *Flood Pruning* | Can dependency-aware pruning beat existing methods? | ⬜ Future |
| 4 | *Bridge-Regularized Training* | Can we train models with cleaner routing structure? | ⬜ Future |
| 5 | *Scaling Laws of Structural Routing* | How does bridge density evolve with model scale? | ⬜ Future |

---

## Paper 2 — Bridge Dependency Networks

**See:** [`paper2_bridge_dependency_networks.md`](paper2_bridge_dependency_networks.md)

Paper 1 established that invisible bridge heads exist and can be detected with a scalar metric. Paper 2 asks whether these heads participate in **sparse, modular dependency networks** — structured pathways where specific downstream heads depend on specific bridge outputs.

The proposal is structured in two stages:
1. **Stage 1** — Empirical measurement of pairwise influence (no graph assumptions)
2. **Stage 2** — If influence is sparse and structured, model it as a directed dependency network

Key experiments include pairwise influence matrices, SVD spectrum analysis, CKA similarity, community detection, hybrid importance scoring, and pairwise synergy tests.

---

## Paper 3 — Flood Pruning (Bridge-Guided Compression)

Depends on Paper 2 results. Two possible directions:

**If dependency networks exist (Paper 2 hypothesis holds):**
- Design a structured pruning algorithm that respects dependency network topology
- Never prune both endpoints of a critical dependency edge simultaneously
- Identify redundant parallel pathways and prune one
- Predict recovery trajectories from network structure

**If dependencies are diffuse (Paper 2 null holds):**
- Design pruning based on residual stream geometry
- Use influence spectrum (SVD) to identify low-dimensional routing manifolds
- Prune along directions of minimal influence

Benchmarks: Wanda, SparseGPT, LLM-Pruner.
Evaluation: perplexity, downstream tasks, recovery speed, FLOPs, latency.

---

## Paper 4 — Bridge-Regularized Training

Instead of pruning after training, encourage models to develop cleaner routing structure during pretraining by adding a bridge regularization term to the loss function.

---

## Paper 5 — Scaling Laws of Structural Routing

Study how bridge density, dependency network sparsity, and community structure scale across model sizes (125M → 7B+).
