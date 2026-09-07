# Research State: Information Routing & FLOOD Framework

## Current Stage
**FORMALLY FROZEN (v1.0.0-paper)** — Research Program 1 is 100% complete and frozen.
Master Technical Freeze Report: [`FINAL_PROJECT_REPORT.md`](FINAL_PROJECT_REPORT.md).
Theoretical Survey Document: [`towards_a_theory_of_routing.md`](towards_a_theory_of_routing.md).

---

## 1. Key Accomplishments Across Research Program 1

### Paper 1: Invisible Bridges (`paper/invisible_bridges.tex`)
- Discovered low-weight attention heads in Layer 0 that are invisible to Wanda/magnitude metrics but causally indispensable.
- $213\times$ damage ratio in GPT-2 Small (32.7 PPL vs 0.15 PPL).
- Confounders ruled out: Layer-0 depth alone produces only $0.4\%$--$2.7\%$ damage; attention sink confound partially discriminated ($\sigma > 0.09$).
- Downstream perturbation amplification tracks bridge score ($r = +0.875$ Small, $r = +0.715$ Medium).

### Paper 2: Conserved Perturbation Geometry (`paper/conserved_perturbation_geometry.tex`)
- Mapped representation shifts across 6 model families (GPT-2 Small/Medium, Pythia-70M, SmolLM-135M, Qwen2.5-0.5B, Gemma-2B).
- Globally conserved dominant direction: Off-diagonal $v_1$ cosine alignment in $[0.935, 0.976]$ ($p < 1/10,001$ vs. isotropic null).
- Severe low-rank collapse: Participation ratio $\approx 1.1$--$2.5$ (vs. shuffled baseline $4.0$--$9.6$); $s_1$ explains $>90\%$ of variance.
- Bridge layers act as localized amplitude injectors (Frobenius norm spikes at L0 in Small/Gemma, L2 in Medium/Qwen).
- Functional validation on Pythia-70M (KL divergence $1.2\times$, $p < 0.05$).

### Paper 3: Estimating Routing Importance & FLOOD (`paper/estimating_routing_importance.tex`)
- Formulated transformer as directed flow graph $G = (V, E, D)$ using $O(N^2)$ activation patches.
- Defined Routing Importance Estimator ($\rie$) family: Broadcast ($f_{\text{bcast}}$), Receiver ($f_{\text{recv}}$), Injector ($f_{\text{inj}}$), Betweenness ($f_{\text{bet}}$).
- OLS causal damage regressions explain up to $80.21\%$ variance (Pythia-160M) and $65.40\%$ (OPT-125M).
- Layer-controlled nested $F$-tests prove RIE signal is orthogonal to depth ($p < 10^{-5}$ to $10^{-15}$, Cohen's $f^2$ up to $4.56$).
- Shuffled negative controls collapse explained variance to $<1.3\%$.
- Zero-shot out-of-distribution transfer: $\rho = 0.469, p < 0.001$ from GPT-2+OPT to Pythia-70M.
- Pruning via FLOOD ($f_{\text{bet}} + f_{\text{bcast}}$): $31\times$ better perplexity preservation on GPT-2 Medium at 30% budget (51.28 PPL vs 1594.18 PPL) with $>0.99$ subspace alignment.

---

## 2. Post-Trilogy Findings & Boundary Stress-Tests

### Finding #1: Adaptive RIE Hypothesis H1 Falsified
- Simple premise ($\alpha_{\text{bcast}}$ rises with $\lambda_2$; $\alpha_{\text{bet}}$ rises with $Q$) falsified by data:
  - OPT-125M ($\lambda_2 = 2.82$, higher) is Betweenness-dominant in pruning.
  - Pythia-160M ($\lambda_2 = 2.32$, lower) is Broadcast-dominant in pruning.
  - Non-monotonic in both metrics; linear separability fails ($<67\%$).

### Finding #2: Predictiveness != Pruning Resilience
- Full 5-model OLS extraction (`beta_coeffs.csv`):
  - GPT-2 Small: $\beta_{\text{bcast}} = 0.5040, R^2 = 0.2354$
  - GPT-2 Medium: $\beta_{\text{bcast}} = -0.2883, R^2 = 0.0786$
  - OPT-125M: $\beta_{\text{bcast}} = 0.8650, R^2 = 0.6540$
  - Pythia-70M: $\beta_{\text{bcast}} = 0.8989, R^2 = 0.7148$
  - Pythia-160M: $\beta_{\text{bcast}} = 0.9155, R^2 = 0.8021$
- **Critical Insight**: Broadcast dominates $\beta$ in all models (identifying what is load-bearing), but OPT's best *pruning* comes from Betweenness-Only (protecting bottlenecks by removing redundant paths). Causal necessity and safe compressibility are inversely related in source-dominant graphs.

### Finding #3: Scaling Boundary & Multi-Path Flow
- Linear OLS centrality models collapse at 24 layers / 384 heads ($R^2 = 7.9\%$ on GPT-2 Medium), though underlying bridge phenomena replicate prospectively (Phase 0/1.5/2 confirmed).
- Multi-step path metrics (`path_metrics_results.csv`: Katz, communicability, current-flow betweenness) boost $R^2$ to $91.2\%$ on Pythia-160M, $87.5\%$ on Pythia-70M, and $76.9\%$ on OPT-125M.

---

## 3. Conclusion & Transition to Program 2
- **Fixed FLOOD-Full** is validated as a robust, zero-hyperparameter, deterministic default across all tested architectures.
- **Scientific Freeze**: Program 1 is locked. No further heuristic tweaking.
- **Future Direction**: **Research Program 2 (Paper 4+)** will investigate: *"Why do routing backbones emerge during transformer optimization?"* (checkpoint tracking, SlimPajama pretraining, weight-decay dynamics).

---

## 4. Key Artifacts
- Master Freeze Report: [`FINAL_PROJECT_REPORT.md`](FINAL_PROJECT_REPORT.md)
- Survey & Program 2 Roadmap: [`towards_a_theory_of_routing.md`](towards_a_theory_of_routing.md)
- Scientific Self-Critique: [`scientific_critique.md`](scientific_critique.md)
- Release Checklist: [`RELEASE.md`](RELEASE.md)
- Reproducibility Guide: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- OLS Beta Matrix: [`beta_coeffs.csv`](beta_coeffs.csv)
- Path Metrics Matrix: [`path_metrics_results.csv`](path_metrics_results.csv)
- Manuscripts: `paper/invisible_bridges.tex`, `paper/conserved_perturbation_geometry.tex`, `paper/estimating_routing_importance.tex`
