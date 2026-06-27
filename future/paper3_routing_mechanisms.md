# Paper 3 — Mechanisms of Conserved Perturbation Geometry in Transformers

**Author:** Amit Kumar
**Status:** Proposal (Trilogy Part III)

---

## Opening Question

Paper 2 asked: *What geometric structure governs representation routing?*
Paper 3 asks: **How is this globally conserved perturbation geometry implemented at the head level?**

---

## Research Question

> Which attention heads and network structures act as the causal components (injectors, receivers, amplifiers, dampeners) that implement and maintain the globally conserved dominant perturbation direction?

By modeling representation shifts as a directed dependency graph, we can analyze the network's topological properties to explain why bridge layers act as localized high-amplitude injectors into the shared subspace.

---

## Planned Hypotheses & Analyses

### 1. Subspace Injector and Receiver Roles
* **Injector Score:** Identify heads that project representation shifts directly into the dominant perturbation subspace ($v_1$).
* **Receiver Score:** Identify downstream heads that act as exit points, reading signal out of the shared subspace.

### 2. Perturbation Amplifiers and Dampeners
* **Amplifiers:** Identify intermediate heads that multiply or scale the magnitude of incoming perturbations.
* **Dampeners (Stabilizers):** Identify heads that act to suppress or stabilize incoming representation shifts, maintaining model robustness.

### 3. Topological Dependency Graph
* Build a directed, weighted network of head-to-head dependencies where edge weights represent pairwise influence.
* Compute network metrics to explain the routing topology:
  * **PageRank & Centrality:** Do high-centrality nodes correspond to the bridge heads identified in Paper 1?
  * **Community Structure (Leiden/Louvain):** Do structural communities align with functional domains (code, math, language)?
  * **Vertex Cuts & Articulation Points:** What is the smallest set of heads whose removal disconnects the routing subspace?
