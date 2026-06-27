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

## Revised Research Roadmap

```
Paper 1: Invisible Bridges                          ✅ Complete
    "Which heads matter?"

Paper 2: Conserved Perturbation Geometry            ⬜ This proposal (Empirical Geometry)
    "How does representation influence propagate downstream?"

Paper 3: Routing Mechanisms                         ⬜ Future (Topological Graph Analysis)
    "How is this geometry implemented?"

Paper 4: Flood Pruning                              ⬜ Future (Pruning Application)
    "Can geometry-aware pruning beat existing methods?"
```
