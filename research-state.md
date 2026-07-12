# Research State: Adaptive RIE (Bucket B)

## Current Stage
COMPUTE COMPLETE (CPU Steps 1-4 DONE, beta extraction DONE) — RESULT: H1 FALSIFIED; beta-based rule also insufficient. Fixed FLOOD-Full validated as robust default.

## Key Finding #1 (2026-07-09, lambda2/Q rule)
The simple Adaptive-RIE premise (alpha_bcast rises with lambda2) is CONTRADICTED by data:
- OPT-125M (lambda2=2.82, HIGHER) -> Betweenness-dominant in pruning
- Pythia-160M (lambda2=2.32, LOWER) -> Broadcast-dominant in pruning
=> non-monotonic in BOTH lambda2 and Q. Best lambda2-threshold fit = 67% on 3 training
models; 2-feature logistic is non-separable.

## Key Finding #2 (2026-07-12, full 5-model OLS beta extraction)
Ran OLS damage regression for ALL 5 models (CPU-only, cached .npy data). Results:

| Model          | beta_bcast | beta_recv | beta_inj | beta_bet | R2     |
|----------------|------------|-----------|----------|----------|--------|
| GPT-2 Small    | 0.5040     | 0.0630    | 0.0194   | 0.0045   | 0.2354 |
| GPT-2 Medium   | -0.2883    | -0.0558   | -0.0021  | 0.0356   | 0.0786 |
| OPT-125M       | 0.8650     | 0.2399    | 0.1355   | 0.0377   | 0.6540 |
| Pythia-70M     | 0.8989     | 0.1846    | 0.0842   | 0.0408   | 0.7148 |
| Pythia-160M    | 0.9155     | 0.1444    | 0.1147   | 0.0108   | 0.8021 |

**Critical insight**: Broadcast dominates beta in ALL models (including OPT-125M), but OPT's
best *pruning* performance comes from Betweenness-Only. This reveals:
- Regression beta measures "which heads are important" (Broadcast answers this)
- Pruning performance measures "which heads can be safely removed" (different question)
- These are INVERSELY related for bottleneck-sensitive architectures like OPT

**GPT-2 Medium anomaly**: R2=7.9% with negative beta_broadcast. The linear centrality model
is a poor fit at 384 heads / 24 layers. This is a genuine scaling limitation.

## Conclusion on Adaptive RIE
Both the lambda2/Q rule (H1) and the beta-based rule are insufficient for adaptive pruning.
The fixed FLOOD-Full formula is validated as a robust, zero-hyperparameter default that:
- Is never the worst performer on any model
- Requires no per-model tuning
- Achieves 31x better perplexity preservation than magnitude pruning (GPT-2 Medium)

The adaptive direction would require fundamentally different targets (pruning resilience,
not regression coefficients). This is a Paper 4+ direction.

## Program Roadmap (updated 2026-07-12)
- Paper 1 — Invisible Bridges: WHAT heads are important? (DONE)
- Paper 2 — Conserved Perturbation Geometry: WHAT geometric structure? (DONE)
- Paper 3 — Estimating Routing Importance: WHAT graph topology explains importance? (DONE)
- Bucket B (Adaptive RIE): COMPLETE — negative result recorded, beta analysis done.
  Fixed FLOOD-Full validated as robust default.
- **Paper 4 (proposed): "Emergence of Routing Structure During Transformer Optimization"**
  — answers WHY routing backbones emerge. Hypothesis-driven. Deferred until budget is set.

## Artifacts
- beta_coeffs.csv: Full 5-model OLS beta table (generated 2026-07-12)
- methodology.md: Updated with beta results and predictiveness-vs-pruning insight
- literature-review.md: Two 10-paper reviews (adaptive pruning + graph centrality pruning)
- reasoning.md: Deliberation on hypothesis, negative result, and refined direction
