# Methodology: Adaptive RIE — architecture-conditioned pruning weights

## Research Question & Hypothesis
- Question: Can RIE combination weights alpha_i be predicted from a model's global
  routing-graph statistics (algebraic connectivity lambda2, modularity Q) so FLOOD needs no
  per-model tuning?
- Hypothesis (H1): alpha_bcast rises as lambda2 rises; alpha_bet rises as Q rises. A simple
  <=2-parameter rule on (lambda2, Q) reproduces each model's known optimal descriptor.
- Status: **FALSIFIED by cached data (2026-07-09, no GPU spent).** See "Negative Result" below.
- Success criteria (original):
  1. Rule predicts dominant descriptor for all 4 known models. -> FAILED (max 67% on 3 training models).
  2. Adaptive PPL <= FLOOD-Full on held-out. -> not tested (premise failed).
  3. Adaptive never worse than magnitude. -> N/A.

## Negative Result (real, from cached dependency matrices + walkthrough PPL)
- lambda2/Q computed for all 5 models (match Paper 3 Table 1). Aligned with component-ablation
  best descriptor:
  * GPT-2 Medium: lambda2=2.7463, Q=0.0874 -> Broadcast
  * OPT-125M:      lambda2=2.8196, Q=0.0609 -> Betweenness
  * Pythia-160M:   lambda2=2.3169, Q=0.0920 -> Broadcast
- OPT has HIGHER lambda2 than Pythia-160M yet is Betweenness-dominant (opposite of H1).
- Best lambda2-threshold fit = 67% on 3 training models; 2-feature logistic is NON-SEPARABLE
  (OPT and Pythia-160M are opposites in both lambda2 and Q).
- Conclusion: a simple alpha(lambda2, Q) rule CANNOT recover the optimal descriptor. The
  routing->pruning-function mapping is richer than a 2-stat spectral summary.

## Beta-Based Weighting: Full 5-Model OLS Results (2026-07-12)

### Beta Coefficient Table (CPU-only, computed from cached .npy data)

All coefficients are standardized OLS (z-scored predictors and response).

| Model          | N    | beta_broadcast | beta_receiver | beta_injector | beta_betweenness | R2     | Adj R2 |
|----------------|------|----------------|---------------|---------------|------------------|--------|--------|
| GPT-2 Small    | 144  | **0.5040**     | 0.0630        | 0.0194        | 0.0045           | 0.2354 | 0.2134 |
| GPT-2 Medium   | 384  | **-0.2883**    | -0.0558       | -0.0021       | 0.0356           | 0.0786 | 0.0689 |
| OPT-125M       | 144  | **0.8650**     | 0.2399        | 0.1355        | 0.0377           | 0.6540 | 0.6440 |
| Pythia-70M     | 48   | **0.8989**     | 0.1846        | 0.0842        | 0.0408           | 0.7148 | 0.6883 |
| Pythia-160M    | 144  | **0.9155**     | 0.1444        | 0.1147        | 0.0108           | 0.8021 | 0.7964 |

### Key Findings

1. **Broadcast centrality dominates beta in ALL 5 models**, including OPT-125M (beta_bcast=0.865 >>>
   beta_bet=0.038). This means: as a *predictor of which heads are causally important*, Broadcast
   is universally dominant.

2. **Critical distinction: predictiveness != pruning performance.**
   - OPT-125M's component ablation (walkthrough S9) shows Betweenness-Only pruning gives
     best PPL (75.92 vs Broadcast-Only 130.86).
   - But the OLS regression shows Broadcast beta=0.865 dominates Betweenness beta=0.038.
   - Interpretation: Broadcast accurately *identifies* important heads, but *removing* them
     (Broadcast-Only pruning) eliminates global routing sources in OPT's bottleneck-sensitive
     architecture. Betweenness-Only pruning avoids these critical hubs.
   - The pruning optimum is not "which heads are most important" but "which heads can be
     safely removed" — these are inversely related for source-dominant architectures.

3. **GPT-2 Medium is anomalous**: R2=7.9% with *negative* beta_broadcast (-0.2883). This suggests
   the causal damage landscape in GPT-2 Medium is not well-explained by a simple linear model
   of these 4 centralities. The dependency graph structure at 384 heads may be qualitatively
   different (24 layers create longer paths and more complex routing patterns).

4. **VIFs remain healthy**: All VIFs < 3.8 across all models, confirming no multicollinearity.

### Implications for Adaptive RIE

The beta-based rule (alpha_i proportional to |beta_i|) would set Broadcast as dominant for all models — which is
empirically correct for *identifying* important heads but wrong for *pruning* on OPT-125M.

This reveals a deeper issue: the adaptive weighting problem is not about "which centrality
predicts importance" but "which centrality's low-scoring heads are safe to remove." These
require different optimization targets:
- **Regression target**: causal damage ~ centralities (already done, Broadcast wins)
- **Pruning target**: minimize PPL after removing K lowest-scoring heads (architecture-dependent)

A true adaptive rule would need to predict the *pruning-optimal* descriptor, not the
*regression-dominant* descriptor. This requires either:
- (a) Component-ablation data on every new model (defeats zero-tuning goal), or
- (b) A proxy for pruning sensitivity (e.g., graph resilience to hub removal), or
- (c) Accepting FLOOD-Full's fixed combination as a robust default that's never catastrophic.

### Recommendation
Record the negative result and the beta-based analysis as a completed investigation.
The fixed FLOOD-Full formula is a reasonable default: it's never the worst performer,
and its fixed nature is actually a strength (zero hyperparameters, fully deterministic).
The adaptive direction requires fundamentally different optimization targets than OLS beta.

## Data Sources (ALL CACHED — no re-extraction)
- Dependency matrices: `gpt2_dependency_matrix.npy`, `gpt2_medium_dependency_matrix.npy`,
  `facebook_opt_125m_dependency_matrix.npy`, `eleutherai_pythia_70m_dependency_matrix.npy`,
  `eleutherai_pythia_160m_dependency_matrix.npy`.
- Centralities: `pagerank_rev` (Broadcast), `pagerank` (Receiver), `injector_scores`,
  `receiver_scores`; Betweenness computed from D via networkx (CPU, seconds).
- Graph stats lambda2, Q: Paper 3 Table 1 (also recomputable from D: Laplacian eigval, Newman modularity).
- Component-ablation PPL: walkthrough Sec 9 (Betweenness-Only / Broadcast-Only / FLOOD-Simplified / FLOOD-Full).
- **beta_coeffs.csv**: Full 5-model beta coefficient table (generated 2026-07-12).

## Limitations & Assumptions
- n=5 models -> simple functional form mandatory (Occam); complex forms overfit.
- GPT-2 Medium R2=7.9% suggests the linear centrality model is a poor fit for larger models.
- "Bridge-Only" dominant for OPT in Paper 3 Table 1 is treated as Betweenness-based (bottleneck),
  consistent with walkthrough component ablation (Betweenness-Only best on OPT).
- The predictiveness vs. pruning-performance distinction is a genuine research insight.
