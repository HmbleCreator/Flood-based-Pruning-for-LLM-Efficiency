# Towards a Theory of Routing in Transformers

This document serves as the master conceptual synthesis and scientific roadmap for the investigation into **Information Routing in Autoregressive Transformers**. It unifies the empirical results, structural findings, and theoretical formulations from **Research Program 1 (Routing Discovery & FLOOD)**, catalogs unresolved anomalies, and outlines the hypothesis-driven direction for **Research Program 2 (Emergence & Theory)**.

---

## 1. Executive Summary & The Routing Hypothesis

### 1.1 The Core Paradigm Shift
Traditional mechanistic interpretability and pruning frameworks view transformers through the lens of **Feature Representation**:
* Individual attention heads are analyzed as "detectors" of semantic features (e.g., induction, indirect objects, syntax).
* Pruning algorithms rank heads by weight magnitude (Magnitude) or local gradient activation (Wanda, Taylor-expansion) under the assumption that activation size corresponds to causal importance.

The **Routing Hypothesis** proposes a fundamental paradigm shift from feature representation to **Information Topology**:
> **The Routing Hypothesis**: Autoregressive transformers do not merely compute static token features; they optimize a dynamic, sparse routing network—a communication backbone—over attention heads. Individual heads act as routers that coordinate the flow of information across layers. Causal damage (perplexity increase) from head ablation is not a function of the local feature magnitude computed by that head, but rather the topological sensitivity of the downstream routing network to the head's removal.

```
Feature Paradigm:       [Input] → [Head (Feature Detector)] → [Output]
                                  (Ranked by Weight/Activation Size)

Routing Paradigm:       [Input] → [Head (Router Node)] ────┐
                                        ↓                  │
                                  [Conserved Subspace] ────┼→ [Downstream Hubs] → [Output]
                                        ↓                  │
                                  [Graph Centrality] ──────┘
                                  (Ranked by Topological Flow)
```

---

## 2. Pillars of Program 1 (The Trilogy)

Research Program 1 focused on the **observation, measurement, validation, and exploitation** of the routing network. It is structured around three core scientific pillars, each corresponding to a major manuscript.

### Pillar 1: Invisible Bridges (Paper 1)
* **Question**: *Can low-weight attention heads be causally indispensable to a transformer's generation capability?*
* **Discovery**: We identified "Invisible Bridge Heads"—primarily in Layer 0 of GPT-2 and Pythia models—that possess near-minimal weight magnitudes (completely invisible to magnitude-based and activation-based pruning like Wanda) but cause catastrophic, model-wide perplexity increases when ablated.
* **Key Finding**: In GPT-2 Small, ablating Group A (invisible bridges) caused a **213×** larger perplexity increase than ablating Group B (Wanda-matched controls):
  $$\text{damage}(A) = 32.7 \text{ PPL} \quad \text{vs.} \quad \text{damage}(B) = 0.15 \text{ PPL}$$
* **Confounders Ruled Out**:
  1. *Layer-0 Position Confound*: Layer-0 heads with low bridge scores were ablated and showed near-zero damage (mean **0.4%** perplexity change on GPT-2 Medium), proving that layer placement alone does not explain the damage.
  2. *Attention Sink Confound*: Some bridge heads exhibit high attention to the BOS token (position 0), but their representation sensitivity profiles remain highly task-dependent, distinguishing them from rigid structural sinks.

### Pillar 2: Conserved Perturbation Geometry (Paper 2)
* **Question**: *Do head ablations cause random, model-wide feature corruption, or do their downstream perturbations lie on a shared, structured manifold?*
* **Discovery**: Downstream representations under ablation do not degrade chaotically. Instead, the representation shifts caused by ablating different bridge heads project onto a shared, low-rank, highly aligned subspace (conserved perturbation geometry).
* **Takeaway**: Bridge heads act as amplitude injectors that drive signals along a conserved routing manifold. When a bridge is removed, the signal falls off this manifold, causing downstream layers to receive out-of-distribution inputs that cascade into generation failure.

### Pillar 3: Estimating Routing Importance (Paper 3)
* **Question**: *Can the causal importance of attention heads be predicted prospectively from the network topology of the attention graph?*
* **Discovery**: We formalize the transformer as a directed, weighted graph where attention heads are nodes and attention patterns act as adjacency matrices.
* **Centrality Mapping**: Graph centrality metrics explain up to **80.2%** of the variance in causal head damage:
  * **Broadcast Centrality** measures a node's capacity to distribute information to downstream layers.
  * **Receiver Centrality** measures a node's capacity to aggregate upstream information.
* **FLOOD Framework**: By combining these topological centralities, the **FLOOD** pruning framework preserves model perplexity significantly better than standard pruning. At 30% pruning budget on GPT-2 Medium, FLOOD preserves perplexity **31× better** than magnitude pruning:
  $$\text{PPL}_{\text{FLOOD}} = 51.28 \quad \text{vs.} \quad \text{PPL}_{\text{Magnitude}} = 1594.18$$

---

## 3. Scientific Grounding & Evidence Matrix

To establish a clear baseline of what has been empirically verified versus what remains theoretical, we categorize the core claims of our research program below by their evidence level and scientific confidence.

| Claim | Evidence Level | Confidence | Key Supporting Evidence / Limitations |
| :--- | :--- | :--- | :--- |
| **Bridge heads exist** | Strong | High | Ablation of low-weight Layer-0 heads causes catastrophic perplexity spikes (e.g., 213× Group A/B damage ratio in GPT-2 Small), which cannot be explained by layer position alone. |
| **Perturbations occupy a shared routing subspace** | Strong | High | Downstream representation shift vectors under different bridge ablations project onto a highly aligned, low-rank conserved manifold (Paper 2). |
| **Centrality predicts damage in several small/medium decoder models** | Strong | High | Ordinary Least Squares (OLS) centrality regression explains up to 80.2% of damage variance on Pythia-160M, 71.5% on Pythia-70M, and 65.4% on OPT-125M. |
| **Routing graph approximates computation** | Moderate | Medium | Graph centrality metrics correlate significantly with causal damage. However, the model is highly scale-dependent, failing on GPT-2 Medium ($R^2 = 7.9\%$). |
| **Routing is the optimization objective** | Speculative | Low | Theoretical conjecture that network sparsification and path stabilization are natural attractors during SGD optimization. Requires validation in Program 2. |

---

## 4. Unresolved Anomalies & Limitations Registry

### 4.1 The Attention Sink Ambiguity
* **Description**: True bridge heads route semantic information. However, some candidate heads assign $>90\%$ attention to position 0 (BOS). These heads may function as structural attention sinks (routing garbage or overflow attention) rather than active routers.
* **Status**: Unresolved. While task-dependent representation variance suggests semantic routing, a subset of bridge heads may be structural side-effects of Softmax normalization.

### 4.2 The GPT-2 Medium $R^2$ Anomaly
* **Description**: While graph centrality regression predicts up to 80% of damage variance on Pythia and OPT, it explains only **7.9%** on GPT-2 Medium, yielding a negative coefficient for Broadcast Centrality.
* **Implications**: The linear centrality-to-damage mapping is scale-dependent. In larger models with more redundant layers (24 layers, 384 heads), information routing becomes highly non-linear or multi-path, violating the single-node centrality assumption. This points to the need for path-based or flow-based graph metrics in Paper 4+.

### 4.3 Predictiveness vs. Pruning Performance
* **Description**: Broadcast Centrality dominates the regression coefficients ( $\beta_{\text{broadcast}} \approx 0.86$ to $0.91$ ) across all models, but OPT's best pruning performance comes from Betweenness-Only pruning.
* **Theoretical Resolution**:
  * Regression $\beta$ answers: *"Which individual heads are most important?"* (Broadcast hubs).
  * Pruning answers: *"Which set of heads can be removed without collapsing the network?"*
  * In highly bottlenecked architectures (like OPT), Broadcast hubs are so critical that removing them collapses the network. Pruning must therefore protect Broadcast hubs and instead remove redundant intermediate routers (Betweenness paths).

### 4.4 Graph Extraction Identifiability & Sensitivity
* **Description**: The directed attention graph constructed in Paper 3 is one mathematical representation of computation flow, but its uniqueness and robustness are untested.
* **Critical Limitations**:
  * *Perturbation Magnitude*: We define graph edges based on linear representation sensitivity to full ablation. If we perturb heads partially (e.g., scaling weights by 0.5 rather than 0.0), the resulting sensitivity graph may shift, indicating that the graph topology is state-dependent.
  * *Metric Sensitivity*: Using L2 representation shift to construct edge weights is intuitive, but alternative metrics (such as cosine similarity of token representations or gradient-based Jacobians) might yield different PageRank/centrality orderings.
  * *Sparsification Thresholds*: The graph construction relies on thresholding small edge weights to induce sparsity. The stability of centrality rankings under different threshold ranges remains unquantified.
  * *Identifiability*: The extracted routing graph is a functional approximation of model dependency, not a uniquely identifiable physical circuit. It should be presented as a useful diagnostic model rather than the ground-truth physical wiring of the transformer.

---

## 5. Program 2 Roadmap: Emergence & Theory

Research Program 2 shifts the scientific inquiry from *how to measure routing* to **why routing backbones emerge during training**.

```
                   PROGRAM 2: THE ROADMAP TO EMERGENCE
                   
      [Hypothesis]  →  Optimization prefers reusable communication paths
                            ↓
      [Predictions] →  1. Routing backbone stabilizes early in training
                       2. Weight decay penalizes non-routing heads into "bridges"
                       3. Gradient flow concentrates along centrality pathways
                            ↓
      [Experiments] →  1. Track centrality metrics across training epochs
                       2. Ablate checkpoints during training (causal dynamics)
                       3. Vary weight decay / initialization seeds
```

### 5.1 The Core Emergence Hypotheses
We propose three testable hypotheses for why transformers organize into sparse routing graphs:
1. **The Pathway Reuse Hypothesis**: Optimization dynamics naturally prefer routing signals through a fixed, reusable set of communication hubs (Bridges) to minimize representation drift across epochs.
2. **The Norm-Penalization (Bridge-Forming) Hypothesis**: Weight decay actively penalizes attention head projections. Heads that compute redundant features are pushed to zero weight, but heads that form critical routing nodes cannot be discarded, resulting in low-weight, high-sensitivity "bridges."
3. **The Spectral Bias Hypothesis**: Transformers learn low-frequency routing structures (broad connectivity) in the first phase of training, and high-frequency semantic features (local heads) in the second phase.

### 5.2 Proposed Experimental Protocol (Paper 4 Design)
* **Model Training**: Train a series of small transformers (e.g., 50M to 125M parameters) from scratch on a curated corpus (e.g., SlimPajama) under controlled settings.
* **Checkpoint Tracking**: Save high-frequency checkpoints (e.g., every 500 gradient steps) during training.
* **Dynamic Centrality Auditing**:
  * Track the evolution of $\lambda_2$ (algebraic connectivity) and Modularity ($Q$) over training time.
  * Measure at what step the "Invisible Bridge" heads differentiate from standard heads.
  * Perform causal ablations at each checkpoint to map the emergence of downstream representation sensitivity.
* **Ablation Dynamics**: Test if training is disrupted or redirected if bridge paths are dynamically ablated *during* the training run.
