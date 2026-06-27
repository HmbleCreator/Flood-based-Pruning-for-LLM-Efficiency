# Paper 2 — Conserved Perturbation Geometry in Transformer Representations

**Author:** Amit Kumar
**Status:** Proposal (pre-code)

---

## Opening Question

Paper 1 asked: *Which heads matter?*

Paper 2 asks: **How does representation influence propagate downstream?**

---

## Research Question

> Do bridge heads define a unique, localized routing network, or does representation influence flow through a **conserved, low-dimensional perturbation geometry** shared by other heads, where bridge heads are simply distinguished by the magnitude of their perturbations?

If a shared perturbation geometry exists, we can characterize the downstream representation shift as a low-dimensional subspace (rather than discrete graph-like circuits) and analyze why bridge heads excite this subspace with substantially larger amplitudes.

## Main Hypothesis

Across the transformer models examined here, downstream perturbations collapse onto a globally conserved dominant direction. Matched control heads exhibit a perturbation geometry with similar dimensionality and dominant direction. Bridge heads do not drive changes in a different direction; instead, they are structurally important because they drive **substantially larger perturbations** along that same shared dominant perturbation subspace.

## Null Hypothesis

Influence is unstructured: downstream representations shift in completely different directions depending on which source head is ablated. There is no shared subspace, and SVD spectrums are indistinguishable from isotropic noise. Bridge heads do not drive significantly larger perturbations than matched controls along a shared subspace.

---

## Stage 1 — Representation Dependency Mapping

*No graph language. Pure empirical measurement.*

### Experiment 1 — Pairwise Influence Matrix

For each head $u$ at layer $l_u$, measure the representation shift at every individual downstream head $v$:

$$I(u, v) = \mathbb{E}_x \left[ \| h_v(x) - h_v^{(\backslash u)}(x) \|_2 \right]$$

This produces a head-to-head influence matrix $I \in \mathbb{R}^{N \times N}$ where $N$ is the total number of heads.

Additionally, compute **CKA similarity** (Kornblith et al., 2019 — already cited in Paper 1) between each downstream head's intact and ablated representations:

$$\text{CKA}(v, v^{\backslash u}) = \frac{\| K_{v}^T K_{v^{\backslash u}} \|_F^2}{\| K_v^T K_v \|_F \cdot \| K_{v^{\backslash u}}^T K_{v^{\backslash u}} \|_F}$$

This gives two complementary views:
- **Representation shift** ($I$): How much does $v$ change?
- **Representation similarity** (CKA): Does $v$'s internal structure survive?

**Falsification:** If $I$ is approximately uniform across downstream heads (low variance across columns), the null holds.

### Experiment 2 — Influence Spectrum (SVD)

Compute the singular value decomposition of $I$.

Plot the singular value spectrum.

**Key question:** Is the influence matrix low-rank?

If the top $k$ singular values (where $k \ll N$) capture most of the variance, then bridge dependencies live on a **low-dimensional routing manifold** — a strong mechanistic result suggesting that influence flows through a small number of independent pathways.

**Falsification:** If singular values decay slowly (effective rank $\approx N$), dependencies are unstructured.

### Experiment 3 — Sparsity and Modularity

Characterize the influence matrix without imposing graph structure:

- **Sparsity:** Compute the Gini coefficient of the influence distribution. High Gini ($> 0.6$) indicates that influence is concentrated on a small number of head pairs.
- **Clustering:** Apply spectral clustering or non-negative matrix factorization to $I$. Do natural clusters emerge?
- **Domain conditioning:** Compute $I_{\text{code}}$, $I_{\text{math}}$, $I_{\text{lang}}$ separately. Do the clusters align with functional domains?

**Falsification:** If Gini is low ($< 0.4$) and no clusters emerge, the influence structure is diffuse.

### Experiment 4 — Conditional Influence (Mediation Analysis)

Evaluate the causal dependency pathways by measuring whether a mediator head $w$ is necessary for the representation shift:
- Empirical $I(u, v)$
- Conditional $I(u, v \mid w) = \mathbb{E}_x \left[ \| h_v^{(\backslash w)}(x) - h_v^{(\backslash u, \backslash w)}(x) \|_2 \right]$

Compare the influence drop $I(u, v) - I(u, v \mid w)$ using the **best, random, and worst** intermediate mediators (ranked by the product $I(u,w) \times I(w,v)$) across the top 20 strongest downstream edges.

> [!WARNING]
> Because mediator rankings are derived from the same influence matrix being validated, this experiment provides **evidence consistent with mediation** rather than an independent, absolute confirmation of discrete routing chains.

**Falsification:** If the best mediators do not show significantly greater drop distributions than random or worst mediators, the observed representation shifts do not support a structured mediation hierarchy.

## Stage 2 — Dependency Network Analysis

*Only entered if Stage 1 establishes sparse, structured dependencies.*

> Having established a shared, low-dimensional routing manifold, Stage 2 investigates how the network's directed topology explains this geometry:
> 1. Which heads inject representations into the dominant 1D subspace, and which heads receive from it?
> 2. Do distinct network communities share the same routing manifold, or do they divide the subspace?
> 3. Does the topological graph connectivity explain *why* bridge heads drive significantly larger perturbations (singular values) than matched control heads?

We model these interactions as a **directed dependency network**, where heads are nodes and edges are weighted by influence magnitude.

### Experiment 5 — Network Topology

Threshold the influence matrix to extract a directed dependency network. Compute:

- **Community structure** (Leiden or Louvain clustering) — Do communities correspond to functional modules (code, math, language)?
- **Betweenness centrality** — Which heads serve as routing bottlenecks?
- **Articulation points** — Whose removal disconnects subnetworks?
- **Minimum vertex cuts** — What is the smallest set of heads whose removal fragments the network?

Community detection comes first because discovering **functional modules** is arguably more scientifically exciting than identifying bottleneck nodes.

**Falsification:** If communities do not correlate with domain-conditional damage patterns from Paper 1, the network structure is not functionally meaningful.

### Experiment 6 — Hybrid Importance Scoring

Compare three ablation orderings:

| Strategy | Ordering criterion |
|---|---|
| **Bridge** (Paper 1) | Scalar bridge score $C(u)$ |
| **Network** | Betweenness centrality in dependency network |
| **Hybrid** | $\alpha \cdot C(u) + (1-\alpha) \cdot \text{betweenness}(u)$ |

Measure Spearman rank correlation between each ordering and actual ablation damage.

This asks: **Does network structure contain information beyond Paper 1?** — not "which metric wins."

**Falsification:** If Hybrid does not improve over Bridge alone ($\Delta\rho < 0.05$), network topology adds no marginal predictive value.

### Experiment 7 — Pairwise Synergy

Ablate *pairs* of heads and measure whether combined damage exceeds individual damages (superadditivity):

$$\text{synergy}(u, v) = \text{damage}(u, v) - \text{damage}(u) - \text{damage}(v)$$

**Key comparison:** Synergy in **dependency-connected pairs** vs. **random pairs** (matched by individual damage).

This is statistically cleaner than a fixed threshold and tests whether the dependency network predicts nonlinear causal interactions.

**Falsification:** If synergy in connected pairs is not significantly greater than in random pairs (permutation test $p > 0.05$), pairwise dependencies are additive.

### Experiment 8 — Cross-Model Replication

Repeat Stage 1 and Stage 2 on GPT-2 Small and GPT-2 Medium:

- Is influence matrix sparsity conserved across model sizes?
- Are community structures topologically similar?
- Does bridge density scale with model depth?

**Falsification:** If dependency structure is completely different across models, the phenomenon is model-specific.

---

## Success Criteria

| Criterion | Measure |
|---|---|
| Influence sparsity | Gini coefficient $> 0.6$ |
| Low-rank structure | Top 5 singular values capture $> 60\%$ of variance |
| Functional communities | Community–domain alignment (NMI $> 0.3$) |
| Betweenness–damage correlation | Spearman $\rho > 0.6$ |
| Hybrid improvement | $\Delta\rho > 0.05$ over scalar bridge score |
| Synergy in connected pairs | Significantly greater than random pairs ($p < 0.05$) |
| Cross-model conservation | At least 1 conserved structural motif |

---

## Outcomes

**If the hypothesis holds:** Bridge heads participate in sparse, modular dependency networks. Paper 3 (Flood Pruning) becomes circuit-aware: never prune both endpoints of a critical dependency edge; identify redundant parallel pathways; predict recovery trajectories from network topology.

**If the null holds:** Bridge importance is an emergent property of residual stream geometry, not discrete routing. This is equally publishable and redirects Paper 3 toward geometry-based pruning. The influence spectrum analysis (Experiment 2) would characterize the geometric structure even under the null.

---

## Timeline

| Phase | Duration | Description |
|---|---|---|
| A | 1 week | Build pairwise influence + CKA computation infrastructure |
| B | 1 week | Run influence matrix experiments, SVD, sparsity analysis (Stage 1) |
| C | 1 week | Network extraction, community detection, synergy tests (Stage 2) |
| D | 1 week | Cross-model replication, analysis, visualization, writing |

---

## Revised Research Roadmap

```
Paper 1: Invisible Bridges                          ✅ Complete
    "Which heads matter?"

Paper 2: Bridge Dependency Networks                 ⬜ This proposal
    "Who depends on whom?"

Paper 3: Flood Pruning                              ⬜ Future
    "Can dependency-aware pruning beat existing methods?"

Paper 4: Bridge-Regularized Training                ⬜ Future
    "Can we train models with cleaner routing structure?"

Paper 5: Scaling Laws of Structural Routing         ⬜ Future
    "How does bridge density evolve with model scale?"
```
