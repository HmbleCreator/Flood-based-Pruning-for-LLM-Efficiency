# Research Deliberation: Adaptive RIE (architecture-conditioned pruning weights)

## Knowledge Consolidation
From the completed trilogy + cached data:
- RIE centralities explain up to 80% of causal-head damage variance (OLS, Paper 3).
- FLOOD (f_bet + f_bcast) preserves perplexity 31x better than magnitude at 30% on GPT-2 Medium.
- BUT component ablation (walkthrough Sec 9) shows the *optimal* descriptor is architecture-dependent:
  Broadcast-Only best on GPT-2 Medium (60.35) and Pythia-160M (74.35);
  Betweenness-Only best on OPT-125M (75.92). Fixed full FLOOD only wins on GPT-2 Medium.
- Graph stats (Paper 3 Table 1): lambda2 = 3.85 (GPT-2 S), 2.82 (OPT), 2.32 (Pythia-160M),
  1.80 (Pythia-70M); Q = 0.0495, 0.0609, 0.0920, 0.0946.
  Pattern: cohesive (high lambda2, low Q) -> Broadcast dominates; modular (low lambda2, high Q) -> Betweenness matters.

## Knowledge Gaps & Contradictions
- Gap: no method conditions pruning weights on a routing graph's lambda2/Q (literature review confirmed).
- Contradiction to resolve: is the effect linear in (lambda2, Q), or threshold/interaction?
  Only 3-4 data points exist -> must keep the functional form simple (Occam).
- Risk: with n=3-4 models, fitting alpha(lambda2, Q) is heavily underdetermined; a complex
  form will overfit. Prefer a 1-2 parameter rule + validation on held-out models.

## Candidate Hypotheses
### H1: alpha_bcast = sigmoid(k*(lambda2 - lambda2_0)); alpha_bet = 1 - alpha_bcast_proxy
- Null: optimal descriptor is independent of (lambda2, Q) (i.e., fixed FLOOD is universally best).
- Required evidence: per-model best descriptor predicted from (lambda2, Q); adaptive PPL <= fixed FLOOD on held-out.
- Feasibility: high (cached data; fit is a 1-param logistic on 3-4 points).
- Novelty: fills the confirmed gap (graph-stat-conditioned weights).
- If confirmed: FLOOD becomes zero-tuning, architecture-robust.
- If refuted: descriptor optimum is NOT a smooth function of lambda2/Q -> needs richer features.

### H2: linear alpha_bcast = a*lambda2 + b*Q + c (2-feature linear)
- More parameters; same data; higher overfit risk. Use only if H1 insufficient.

## Structured Deliberation
| Hypothesis | Strengths | Weaknesses | Key Uncertainty | Information Gain |
|------------|-----------|------------|-----------------|-----------------|
| H1 (logistic on lambda2) | simple, 1 param, matches observed monotonic trend | assumes single pivot lambda2_0; ignores Q interaction | exact pivot location | high if validates held-out |
| H2 (linear 2-feat) | uses both stats | overfit on n=3-4 | coefficient sign/stability | medium |

## Selected Direction
- **Chosen**: H1 — a threshold/logistic rule on lambda2 (primary signal), with Q as a
  secondary modifier, keeping total parameters <= 2. Rationale: Occam + the clearly
  monotonic lambda2 trend across 4 models; literature review shows no prior graph-stat conditioning.
- **Rationale**: cohesive graphs (high lambda2) broadcast globally; modular graphs (low lambda2,
  high Q) need bottleneck protection. A smooth interpolation between Broadcast-Only and
  Betweenness-Only as lambda2 falls is the minimal model consistent with all 4 known dominants.
- **Key risks**: (1) n=3-4 fit is weak -> mitigate by validating prediction on Pythia-70M
  (Mixed) and GPT-2 Small (Broadcast) WITHOUT using them in the fit; (2) the held-out GPU
  pruning re-eval may be skipped if user declines -> then validation is prediction-only (cached).
- **Pre-specified success criteria**:
  1. Fitted rule predicts the dominant descriptor for all 4 known models correctly.
  2. (If GPU approved) Adaptive-FLOOD PPL <= cached FLOOD-Full PPL on >=1 held-out model at 30%.
  3. Adaptive never worse than magnitude on any evaluated model (sanity floor).
- **Fallback plan**: if H1 fails (e.g., OPT's "Bridge-Only" dominant is actually betweenness but
  at a different lambda2 regime), switch to H2 or a 2-threshold piecewise rule; if still fails,
  report negative result (lambda2/Q insufficient; need richer graph features) — itself a finding.
