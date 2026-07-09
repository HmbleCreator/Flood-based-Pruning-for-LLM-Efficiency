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

## Refined direction (promising, CPU-only, NOT yet run)
- The OLS beta coefficients (Paper 3 Table 3) track the optimum better: Betweenness beta is
  HIGHEST on OPT (0.0377) — the only Betweenness-pruning-optimum model — vs 0.0045 (GPT-2 S)
  and 0.0108 (Pythia-160M). So alpha_i proportional to the per-model OLS beta_i is a more
  principled adaptive rule (weight each descriptor by its empirically measured predictive power).
- This needs OLS beta for all 5 models (have 3 from Paper 3; GPT-2 Small + Pythia-70M require
  running run_damage_regression.py — CPU, cheap, code exists).
- Decision: report negative result on lambda2/Q form; propose beta-based rule as the refined
  Bucket B direction pending user approval. GPU Step 5 is NOT warranted (would only confirm a
  foregone conclusion).

## Data Sources (ALL CACHED — no re-extraction)
- Dependency matrices: `gpt2_dependency_matrix.npy`, `gpt2_medium_dependency_matrix.npy`,
  `facebook_opt_125m_dependency_matrix.npy`, `eleutherai_pythia_70m_dependency_matrix.npy`,
  `eleutherai_pythia_160m_dependency_matrix.npy`.
- Centralities: `pagerank_rev` (Broadcast), `pagerank` (Receiver), `injector_scores`,
  `receiver_scores`; Betweenness computed from D via networkx (CPU, seconds).
- Graph stats lambda2, Q: Paper 3 Table 1 (also recomputable from D: Laplacian eigval, Newman modularity).
- Component-ablation PPL: walkthrough Sec 9 (Betweenness-Only / Broadcast-Only / FLOOD-Simplified / FLOOD-Full).
- Held-out models: Pythia-70M (Mixed dominant), GPT-2 Small (Broadcast dominant) — NOT used in the fit.

## Analysis Pipeline
### Step 1: Recompute centralities + graph stats (CPU, free)
- For each of 5 models: load D, compute Betweenness centrality (networkx), verify lambda2/Q
  from cached or recompute. Output: per-model (lambda2, Q, best_known_descriptor, best_PPL).

### Step 2: Define candidate alpha(lambda2, Q) forms (CPU, free)
- Form A (H1, 1-param logistic on lambda2): w_bcast = sigmoid(k*(lambda2 - lambda2_0));
  w_bet = 1 - w_bcast (renormalized with tiny w_inj/w_recv). Fit (lambda2_0, k) so the implied
  descriptor ranking matches known best per model.
- Form B (H2, 2-param linear): w_bcast = clip(a*lambda2 + b*Q + c). Fit by least-squares to
  minimize predicted-vs-known descriptor mismatch.
- Selection: prefer Form A unless Form B reduces held-out prediction error materially.

### Step 3: In-sample fit + prediction check (CPU, free)
- Fit on the 3 models with full component-ablation PPL (OPT, GPT-2 Medium, Pythia-160M).
- Predict dominant descriptor for all 5; check correctness on the 4 known (incl. held-out
  Pythia-70M, GPT-2 Small predicted, not fitted).

### Step 4: Surrogate PPL of adaptive weights (CPU, free, from cached ablation)
- For OPT / GPT-2 Medium / Pythia-160M we have 4 descriptor-PPL points each. Fit a cheap
  2-descriptor linear surrogate PPL(w_bcast, w_bet) per model; evaluate adaptive (w_bcast, w_bet)
  from Step 2; compare adaptive surrogate PPL to FLOOD-Full cached PPL. This gives a CPU-only
  estimate of whether adaptive helps, using only real cached numbers.

### Step 5 (GPU, OPTIONAL — needs approval): true adaptive pruning re-eval
- On held-out models (Pythia-70M, GPT-2 Small) AND one fit model (e.g., Pythia-160M):
  build RIE_adaptive = w_bcast*f_bcast + w_bet*f_bet (weights from Step 2), prune lowest-K heads
  at 10%/30% budgets, measure WikiText-2 perplexity (cached `gpt2_local/`, etc.). Compare to
  cached FLOOD-Full and magnitude. This is the only step requiring GPU/model eval.

## Controls & Validation
- Positive control: on GPT-2 Medium, adaptive should recover ~Broadcast-Only (known best 60.35).
- Negative control: random alpha should not beat FLOOD-Full systematically.
- Validation: held-out prediction (Pythia-70M Mixed, GPT-2 Small Broadcast) not in fit.
- Sanity floor: adaptive PPL must beat magnitude pruning on every model.

## Statistical Plan
- Primary test: per-model paired comparison of adaptive vs FLOOD-Full PPL (cached bootstrap CIs
  from walkthrough; for GPU step, 5 seeds as in walkthrough).
- Multiple testing: Holm-Bonferroni across models if GPU step runs.
- Significance: alpha = 0.05. Effect size: delta PPL and % improvement.

## Compute Requirements
- Steps 1-4: CPU only, seconds-minutes. **Cost: $0.**
- Step 5 (optional GPU): 2-3 small models (70M-160M) perplexity eval at 10%/30% budgets,
  5 seeds each. Platform: Modal A10G (single GPU). Estimated duration: ~20-40 min.
  Estimated cost: ~$0.50-2.00 (A10G ~$0.0008-0.0015/sec on Modal; eval is forward-pass only).
  **Requires explicit user approval before launch.**

## Limitations & Assumptions
- n=3-4 fit models -> simple functional form mandatory (Occam); complex forms overfit.
- Surrogate PPL (Step 4) is a linear approximation over 4 real points, not a true prune.
- GPU step uses small models only; 1B/7B scale-up deferred (user decision).
- "Bridge-Only" dominant for OPT in Paper 3 Table 1 is treated as Betweenness-based (bottleneck),
  consistent with walkthrough component ablation (Betweenness-Only best on OPT).
