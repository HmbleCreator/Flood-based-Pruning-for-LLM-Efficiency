# Estimating Routing Importance in Transformer Attention Graphs

This repository contains the official codebase and manuscripts for our research program investigating the global representational routing topology of transformers, culminating in the paper **"Estimating Routing Importance in Transformer Attention Graphs"** (Paper 3).

---

## 1. Research Overview

Our research program bridges **mechanistic interpretability** and **model compression** through a three-stage pipeline:
1. **Discovery:** Conduct targeted causal perturbations (activation patching) to estimate a global directed routing dependency graph $G$ over self-attention heads.
2. **Validation:** Fit OLS regressions using graph-theoretic centrality descriptors (Broadcast, Receiver, Injector, Betweenness) to explain experimentally measured causal head necessity, outperforming simple architectural priors like layer depth ($p < 10^{-5}$).
3. **Application:** Prune models via **FLOOD**, a routing-aware structured head pruning algorithm that targets redundant paths while protecting structural routing bottlenecks, preserving downstream perplexity up to $31\times$ better than magnitude pruning.

---

## 2. Getting Started

### Installation
Clone the repository and install the standard dependencies:
```bash
pip install torch transformers numpy scipy scikit-learn matplotlib
```
*Note: To prevent conflicts with local site-packages during script runs, we recommend running Python scripts with the `-s` flag (e.g., `python -s script_name.py`).*

### Dataset Setup
Downstream evaluations utilize the standard **WikiText-2** corpus, which will be automatically downloaded and cached by the Hugging Face `datasets` library upon running the scripts.

### Cached Arrays
To allow instant replication of our statistical regression results without requiring hours of GPU activation patching, the repository includes pre-computed numpy arrays (`.npy`) in the root directory for:
- Causal dependency matrices (`*_dependency_matrix.npy`)
- Experimentally measured causal perplexity damage (`*_causal_damage.npy`)
- Graph Laplacians (`*_laplacian.npy`)
- Pre-calculated centrality scores (`*_pagerank.npy`, etc.)

---

## 3. Reproducing Causal Damage Regressions & Controls (Section 5)

To reproduce the regression statistics, p-values, variance inflation factors (VIFs), effect sizes, and bootstrap confidence intervals reported in the manuscript:

### 1. Standard Causal Damage Regression (Table 3)
Fits multiple linear regressions predicting head damage from RIE centralities for GPT-2 Small and OPT-125M:
```bash
python -s src/flood/run_damage_regression.py
```
*(For Pythia-160M regressions, run: `python -s src/flood/run_pythia_160m_regression.py`)*

### 2. Nested Model Progression & F-Tests
Performs incremental centrality additions and computes $F$-tests with Cohen's $f^2$ effect sizes:
```bash
python -s src/flood/run_nested_regression.py
```

### 3. Controlling for Layer Index (Table 4 & Figure 6)
Compares a baseline Layer Index model (Model A) to a combined Layer + RIE model (Model B) under $F$-testing:
```bash
python -s src/flood/run_layer_controlled_regression.py
```

### 4. Negative Control Shuffling (Section 5.3)
Verifies that predictive power collapses near zero when shuffling the correspondence between descriptors and heads:
```bash
python -s src/flood/run_negative_control.py
```

### 5. Out-of-Distribution Transferability & Calibration Sensitivity
Computes bootstrap confidence intervals, zero-shot transfer correlations to Pythia-70M, and prompt sensitivity filters:
```bash
python -s src/flood/run_transfer_diagnostics.py
```

---

## 4. Reproducing Downstream Head Pruning via FLOOD (Section 6)

To execute structured attention-head pruning sweeps, fine-tuning recovery, and latency comparisons:

### 1. Downstream Perplexity Budget Sweeps (Table 5 & Figures 7 & 8)
Runs attention head pruning comparisons (Magnitude vs FLOOD-Simplified vs FLOOD-Full vs Gradient-Taylor vs Random) on WikiText-2:
```bash
# Sweep for GPT-2 Small
python -s src/flood/benchmark.py --model gpt2

# Sweep for OPT-125M
python -s src/flood/benchmark.py --model opt

# Sweep for Pythia-160M
python -s src/flood/benchmark.py --model pythia
```

### 2. Fine-Tuning Recovery Sweeps (Figure 9)
Evaluates downstream recovery convergence trajectories under post-pruning AdamW fine-tuning:
```bash
python -s src/flood/evaluate.py
```

### 3. Online Scoring Latency Benchmark (Table 6)
Measures the CPU execution latency of RIE centrality computations vs. backward-pass Gradient-Taylor oracle scoring:
```bash
python -s src/flood/benchmark_resources.py
```

---

## 5. Compiling the Manuscript

The LaTeX source document for Paper 3 is located at `paper/estimating_routing_importance.tex`. To compile the PDF along with citations:
```bash
cd paper
pdflatex estimating_routing_importance.tex
bibtex estimating_routing_importance
pdflatex estimating_routing_importance.tex
pdflatex estimating_routing_importance.tex
```
All figures are stored under `paper/figures/`.