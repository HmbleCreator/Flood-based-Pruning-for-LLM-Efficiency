# Research State: Adaptive RIE (Bucket B — new research)

## Current Stage
COMPUTE (CPU Steps 1-4 DONE) — RESULT: H1 FALSIFIED (negative finding). GPU Step 5 NOT warranted.

## Key Finding (2026-07-09, real cached data)
The simple Adaptive-RIE premise (alpha_bcast rises with lambda2) is CONTRADICTED by data:
- OPT-125M (lambda2=2.82, HIGHER) -> Betweenness-dominant
- Pythia-160M (lambda2=2.32, LOWER) -> Broadcast-dominant
=> non-monotonic in BOTH lambda2 and Q. Best lambda2-threshold fit = 67% on 3 training
models; 2-feature logistic is non-separable. Simple alpha(lambda2,Q) CANNOT predict the
optimal descriptor. This is a legitimate NEGATIVE result (falls under reasoning.md fallback).
Implication: routing->function mapping is richer than a 2-stat spectral summary; adaptive
weighting needs either the full centrality spectrum / rich-club, or the per-model OLS beta
coefficients (Betweenness beta already tracks OPT's betweenness optimum), not lambda2/Q.

## Program Roadmap (5-paper progression — updated per colleague review)
Papers 1-3 answer **what** (empirical discovery). The colleague's framing: the next
papers should answer **why** (hypothesis-driven theory).
- Paper 1 — Invisible Bridges: WHAT heads are important? (done)
- Paper 2 — Conserved Perturbation Geometry: WHAT geometric structure? (done)
- Paper 3 — Estimating Routing Importance: WHAT graph topology explains importance? (done)
- Bucket B (Adaptive RIE): engineering refinement of Paper 3 — makes FLOOD zero-tuning.
  Still a "what works" contribution; does NOT answer "why".
- **Paper 4 (proposed, strategic): "Emergence of Routing Structure During Transformer
  Optimization"** — answers WHY routing backbones emerge. Hypothesis-driven. Central
  hypothesis: gradient-based optimization converges toward reusable routing pathways
  because reuse is more efficient than constructing independent flows. Five falsifiable
  predictions: (1) routing emerges during training; (2) convergence across random seeds
  (attractor); (3) systematic scaling laws (lambda2 up, Q down, rich-club denser with scale);
  (4) perturb-and-recover (backbone is an optimization attractor); (5) architecture
  independence (GPT-2/Pythia/Qwen/Gemma/Llama converge to similar routing).
  NOTE: Paper 4 needs training-from-scratch checkpoints across seeds/architectures —
  far larger compute than Bucket B. Deferred until Bucket B lands and budget is set.

## Research Question (Bucket B — Adaptive RIE, CURRENT TASK)
Can the RIE combination weights alpha_i be predicted from a model's global routing-graph
statistics (algebraic connectivity lambda2, modularity Q) so that FLOOD automatically
selects the right structural descriptor per architecture — without per-model tuning?

## Why this is the right next step (from Bucket B)
Paper 3 already lists this as Future Work (Eq. 7): alpha_i = alpha_i(Q, lambda2). The
walkthrough component-ablation table shows the optimal descriptor is architecture-dependent:
- Broadcast-Only wins on GPT-2 Medium (60.35) and Pythia-160M (74.35)
- Betweenness-Only wins on OPT-125M (75.92)
The fixed full FLOOD (f_bet + f_bcast) is only best on GPT-2 Medium; on OPT/Pythia-160M
a single descriptor beats it. Adaptive weighting turns this limitation into a result.

## Key Decisions
- (SCOPE): Reuse cached dependency matrices + centralities for 5 models — NO 2h/model re-extraction.
- (SCOPE): Fit alpha(lambda2, Q) on the 3 models with full component-ablation PPL
  (OPT, GPT-2 Medium, Pythia-160M); validate on held-out models (Pythia-70M, GPT-2 Small).

## Available cached data (verified)
- Dependency matrices: gpt2, gpt2_medium, facebook_opt_125m, eleutherai_pythia_70m/160m
- Centralities: pagerank_rev (Broadcast), pagerank (Receiver), injector_scores, receiver_scores
- Graph stats (Paper 3 Table 1): lambda2, Q per model
- Component-ablation PPL (walkthrough Sec 9): Betweenness/Broadcast/FLOOD-Simplified/FLOOD-Full

## Component-ablation PPL (cached)
| Metric        | GPT-2 Med | OPT-125M | Pythia-160M | best        |
|---------------|-----------|----------|-------------|-------------|
| Baseline      | 38.23     | 59.03    | 50.52       |             |
| Betweenness-O | 276.46    | 75.92    | 87.27       | OPT best    |
| Broadcast-O   | 60.35     | 130.86   | 74.35       | Med, Pythia |
| FLOOD-Simple  | 128.52    | 86.42    | 82.81       |             |
| FLOOD-Full    | 51.28     | 97.51    | 96.79       | Med best    |

## Graph stats (Paper 3 Table 1)
| Model         | N   | lambda2 | Q      | Dominant      |
|---------------|-----|---------|--------|---------------|
| GPT-2 Small   | 144 | 3.8539  | 0.0495 | Broadcast-Only|
| OPT-125M      | 144 | 2.8196  | 0.0609 | Bridge-Only   |
| Pythia-160M   | 144 | 2.3169  | 0.0920 | Broadcast-Only|
| Pythia-70M    | 48  | 1.7978  | 0.0946 | Mixed         |

## Hypothesis H1
alpha_bcast increases with lambda2 (cohesive graphs -> global sources dominate);
alpha_bet increases with Q (modular graphs -> local bottlenecks matter).
A simple function alpha(lambda2, Q) reproduces each model's known optimal descriptor
and yields <= FLOOD-Full PPL on held-out models.

## Experiment Log
| Attempt | Method | Result | Status |
|---------|--------|--------|--------|

## Critique History
- Pre-COMPUTE: pending
- Post-COMPUTE: pending

## Artifacts
- literature-review.md: pending
- reasoning.md: pending
- methodology.md: pending
- figures/: pending
