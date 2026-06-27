"""
Paper 2 — Stage 1: Representation Dependency Mapping
=====================================================
Model: GPT-2 Small / GPT-2 Medium

Computes the pairwise influence matrix I(u,v) measuring how much
ablating head u shifts the output representation of downstream head v.

Stage 1 is purely empirical — no graph language.

Experiments:
  1. Pairwise Influence Matrix + CKA similarity
  2. Influence Spectrum (SVD analysis)
  3. Sparsity (Gini coefficient) and Modularity (spectral clustering)

Usage:
  python map_dependencies.py
  python map_dependencies.py --model_path C:/path/to/gpt2_local
  python map_dependencies.py --model gpt2-medium --model_path C:/path/to/gpt2_medium_local
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Probe texts (same domains as Paper 1) ─────────────────────────────────────

DOMAIN_PROBES = {
    "code": [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1",
        "class LinkedList:\n    def __init__(self):\n        self.head = None",
        "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    loss.backward()",
        "import torch\nmodel = torch.nn.Linear(128, 64)\noutput = model(x)",
        "SELECT user_id, COUNT(*) FROM events WHERE date > '2024' GROUP BY user_id",
    ],
    "math": [
        "The fundamental theorem of calculus connects differentiation and integration.",
        "Euler's identity states that e raised to i times pi plus one equals zero.",
        "The gradient of a scalar field points in the direction of steepest ascent.",
        "A matrix is invertible if and only if its determinant is nonzero.",
        "The law of large numbers guarantees convergence of sample means to the true mean.",
    ],
    "language": [
        "The ambassador carefully chose her words before addressing the assembly.",
        "Despite initial setbacks, the expedition eventually reached the summit.",
        "The novel explores themes of identity, memory, and the passage of time.",
        "She noticed the subtle shift in his expression when the name was mentioned.",
        "The committee reached a consensus after hours of careful deliberation.",
    ],
}


# ── Model loading (consistent with Paper 1 scripts) ──────────────────────────

def load_model(path=None, model_name="gpt2"):
    src = path
    if src is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_name = "gpt2_medium_local" if "medium" in model_name else "gpt2_local"
        local_path = os.path.join(script_dir, "..", "..", local_name)
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "..", local_name)
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, local_name)
        if os.path.exists(local_path):
            src = local_path
        else:
            src = model_name

    tok = GPT2Tokenizer.from_pretrained(src)
    tok.pad_token = tok.eos_token
    mdl = GPT2LMHeadModel.from_pretrained(src)
    mdl.eval()
    label = "GPT-2 Medium" if mdl.config.n_layer == 24 else "GPT-2 Small"
    print(f"  Loaded {label} from: {src}")
    print(f"  Layers: {mdl.config.n_layer}, Heads/layer: {mdl.config.n_head}, "
          f"d_model: {mdl.config.n_embd}")
    return mdl, tok


def enc(tok, text, max_length=64):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_length)


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Pairwise Influence Matrix + CKA
# ══════════════════════════════════════════════════════════════════════════════

def _capture_head_outputs(model, tokenizer, text, max_length=64):
    """
    Run the model on `text` and capture each attention head's output
    (the per-head slice of c_proj input) at every layer.

    Returns: dict[(layer, head)] -> tensor of shape (seq_len, d_head), float32
    """
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    captures = {}
    hooks    = []

    def make_hook(layer_idx):
        def hook_fn(module, args):
            # args[0] shape: (batch, seq, n_embd)
            x = args[0].detach().float().squeeze(0)  # (seq, n_embd)
            for h in range(n_heads):
                s, e = h * d_head, (h + 1) * d_head
                captures[(layer_idx, h)] = x[:, s:e].clone()  # (seq, d_head)
        return hook_fn

    for l in range(n_layers):
        hk = model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_hook(l))
        hooks.append(hk)

    inp = enc(tokenizer, text, max_length=max_length)
    with torch.no_grad():
        model(**inp)

    for hk in hooks:
        hk.remove()

    return captures


def _capture_head_outputs_ablated(model, tokenizer, text, ablate_layer, ablate_head,
                                  max_length=64):
    """
    Run the model with head (ablate_layer, ablate_head) zeroed out,
    and capture all other heads' outputs.
    """
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    captures = {}
    hooks    = []

    # Ablation hook: zero out the target head's slice
    a_s, a_e = ablate_head * d_head, (ablate_head + 1) * d_head

    def ablation_hook(module, args):
        x = args[0].clone()
        x[:, :, a_s:a_e] = 0.0
        return (x,)

    abl_hk = model.transformer.h[ablate_layer].attn.c_proj.register_forward_pre_hook(
        ablation_hook
    )
    hooks.append(abl_hk)

    # Capture hooks on all layers (including ablated layer, for completeness)
    def make_capture_hook(layer_idx):
        def hook_fn(module, args):
            x = args[0].detach().float().squeeze(0)
            for h in range(n_heads):
                s, e = h * d_head, (h + 1) * d_head
                captures[(layer_idx, h)] = x[:, s:e].clone()
        return hook_fn

    for l in range(n_layers):
        hk = model.transformer.h[l].attn.c_proj.register_forward_pre_hook(make_capture_hook(l))
        hooks.append(hk)

    inp = enc(tokenizer, text, max_length=max_length)
    with torch.no_grad():
        model(**inp)

    for hk in hooks:
        hk.remove()

    return captures


def linear_cka(X, Y):
    """
    Linear CKA between two representation matrices X, Y of shape (n, d).
    Kornblith et al. 2019.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (X.T @ Y).norm('fro') ** 2
    hsic_xx = (X.T @ X).norm('fro') ** 2
    hsic_yy = (Y.T @ Y).norm('fro') ** 2
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 1.0  # both zero -> identical
    return float(hsic_xy / denom)


def compute_influence_matrix(model, tokenizer, texts, source_layer=0):
    """
    Compute the pairwise influence matrix I(u, v) for all heads u at
    `source_layer` and all downstream heads v.

    Also computes CKA similarity between intact and ablated representations.

    Returns:
        influence: dict[(src_head, (tgt_layer, tgt_head))] -> float (L2 shift)
        cka_sim:   dict[(src_head, (tgt_layer, tgt_head))] -> float (CKA)
    """
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    influence = defaultdict(list)
    cka_sims  = defaultdict(list)

    all_targets = [(l, h) for l in range(source_layer + 1, n_layers)
                   for h in range(n_heads)]

    print(f"\n  Computing influence matrix for Layer {source_layer} "
          f"({n_heads} source heads -> {len(all_targets)} target heads)")
    print(f"  Probe texts: {len(texts)}")

    for ti, text in enumerate(texts):
        # Baseline: capture all head outputs
        base = _capture_head_outputs(model, tokenizer, text)

        for src_h in range(n_heads):
            # Ablated: capture with source head zeroed
            ablated = _capture_head_outputs_ablated(
                model, tokenizer, text, source_layer, src_h
            )

            for tgt in all_targets:
                if tgt in base and tgt in ablated:
                    b = base[tgt]    # (seq, d_head)
                    a = ablated[tgt] # (seq, d_head)

                    # L2 representation shift (mean-pooled across sequence)
                    shift = (b - a).norm(dim=-1).mean().item()
                    influence[(src_h, tgt)].append(shift)

                    # CKA similarity
                    if b.shape[0] >= 2:  # need at least 2 points
                        cka = linear_cka(b, a)
                        cka_sims[(src_h, tgt)].append(cka)

        if (ti + 1) % 5 == 0 or ti == len(texts) - 1:
            print(f"    Text {ti+1}/{len(texts)} done")

    # Average over texts
    I = {}
    C = {}
    for key, vals in influence.items():
        I[key] = float(np.mean(vals))
    for key, vals in cka_sims.items():
        C[key] = float(np.mean(vals))

    return I, C, all_targets


def influence_to_matrix(I, n_src_heads, all_targets):
    """Convert influence dict to a numpy matrix (src_heads x targets)."""
    mat = np.zeros((n_src_heads, len(all_targets)))
    target_idx = {t: i for i, t in enumerate(all_targets)}
    for (src_h, tgt), val in I.items():
        if tgt in target_idx:
            mat[src_h, target_idx[tgt]] = val
    return mat


def cka_to_matrix(C, n_src_heads, all_targets):
    """Convert CKA dict to a numpy matrix (src_heads x targets)."""
    mat = np.ones((n_src_heads, len(all_targets)))  # default 1.0 = identical
    target_idx = {t: i for i, t in enumerate(all_targets)}
    for (src_h, tgt), val in C.items():
        if tgt in target_idx:
            mat[src_h, target_idx[tgt]] = val
    return mat


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Influence Spectrum (SVD)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_influence_spectrum(I_mat, prefix=""):
    """
    SVD of the influence matrix.
    Key question: is the influence matrix low-rank?
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: INFLUENCE SPECTRUM (SVD)")
    print("=" * 65)

    U, S, Vt = np.linalg.svd(I_mat, full_matrices=False)
    total_var = (S ** 2).sum()
    cumulative = np.cumsum(S ** 2) / total_var * 100

    print(f"\n  Matrix shape: {I_mat.shape}")
    print(f"  Top singular values:")
    for i in range(min(10, len(S))):
        print(f"    σ_{i+1} = {S[i]:.4f}  (cumulative variance: {cumulative[i]:.1f}%)")

    # Effective rank (number of singular values needed for 90% variance)
    eff_rank_90 = int(np.searchsorted(cumulative, 90.0)) + 1
    eff_rank_80 = int(np.searchsorted(cumulative, 80.0)) + 1
    eff_rank_60 = int(np.searchsorted(cumulative, 60.0)) + 1
    print(f"\n  Effective rank (60% variance): {eff_rank_60}")
    print(f"  Effective rank (80% variance): {eff_rank_80}")
    print(f"  Effective rank (90% variance): {eff_rank_90}")
    print(f"  Full rank: {len(S)}")

    top5_var = cumulative[min(4, len(cumulative)-1)]
    print(f"\n  Top 5 singular values capture {top5_var:.1f}% of variance")
    if top5_var > 60:
        print("  ✓ LOW-RANK: Dependencies live on a low-dimensional routing manifold")
    else:
        print("  ✗ FULL-RANK: Dependencies are unstructured")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(range(len(S)), S, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("Singular Value Spectrum")

    axes[1].plot(range(1, len(cumulative)+1), cumulative, "o-", color="darkorange",
                 markersize=3)
    axes[1].axhline(60, color="gray", linestyle="--", alpha=0.5, label="60%")
    axes[1].axhline(80, color="gray", linestyle="-.", alpha=0.5, label="80%")
    axes[1].axhline(90, color="gray", linestyle=":", alpha=0.5, label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative Variance Explained")
    axes[1].legend()

    plt.suptitle("Experiment 2: Influence Spectrum (SVD)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp2_influence_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp2_influence_spectrum.png")

    return S, cumulative


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Sparsity (Gini) and Modularity
# ══════════════════════════════════════════════════════════════════════════════

def gini_coefficient(values):
    """Compute the Gini coefficient of a distribution."""
    v = np.sort(np.abs(values).ravel())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)


def analyze_sparsity_and_modularity(I_mat, I_dict, all_targets, n_heads,
                                    source_layer, prefix=""):
    """
    Experiment 3: Characterize the influence matrix.
    - Sparsity via Gini coefficient
    - Clustering via spectral methods
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: SPARSITY & MODULARITY")
    print("=" * 65)

    # ── Gini coefficient ──
    flat = I_mat.ravel()
    gini = gini_coefficient(flat)
    print(f"\n  Gini coefficient of influence distribution: {gini:.4f}")
    if gini > 0.6:
        print("  ✓ HIGH SPARSITY: Influence is concentrated on a small number of pairs")
    elif gini > 0.4:
        print("  ~ MODERATE SPARSITY: Some concentration, not fully diffuse")
    else:
        print("  ✗ LOW SPARSITY: Influence is approximately uniform (null hypothesis)")

    # ── Per-source-head concentration ──
    print(f"\n  Per-source-head influence concentration:")
    for h in range(n_heads):
        row = I_mat[h, :]
        row_gini = gini_coefficient(row)
        top_pct = np.sort(row)[::-1][:max(1, len(row)//10)].sum() / (row.sum() + 1e-12) * 100
        print(f"    Head {h:2d}: Gini={row_gini:.3f}, "
              f"Top 10% targets carry {top_pct:.1f}% of influence")

    # ── Column-wise analysis: which targets are most influenced? ──
    col_totals = I_mat.sum(axis=0)
    top_k = min(10, len(col_totals))
    top_idx = np.argsort(col_totals)[::-1][:top_k]
    print(f"\n  Top {top_k} most-influenced downstream heads:")
    for rank, idx in enumerate(top_idx):
        tgt = all_targets[idx]
        print(f"    {rank+1}. L{tgt[0]:02d}H{tgt[1]:02d}: "
              f"total influence = {col_totals[idx]:.4f}")

    # ── Spectral clustering attempt ──
    try:
        from sklearn.cluster import SpectralClustering
        # Cluster the targets based on which sources influence them similarly
        if I_mat.shape[1] >= 4:
            # Use the transpose: cluster targets by their influence profile
            n_clusters = min(4, I_mat.shape[1] // 2)
            affinity = np.abs(I_mat.T @ I_mat)  # target-target similarity
            np.fill_diagonal(affinity, 0)
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                   random_state=42)
            labels = sc.fit_predict(affinity + 1e-10)

            print(f"\n  Spectral clustering ({n_clusters} clusters):")
            for c in range(n_clusters):
                members = [all_targets[i] for i in range(len(labels)) if labels[i] == c]
                layers = [m[0] for m in members]
                print(f"    Cluster {c}: {len(members)} heads, "
                      f"layers {min(layers)}-{max(layers)}")
    except ImportError:
        print("\n  [sklearn not available — skipping spectral clustering]")

    # ── Visualizations ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Influence matrix heatmap
    im = axes[0].imshow(I_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel("Target head index")
    axes[0].set_ylabel(f"Source head (Layer {source_layer})")
    axes[0].set_title("Pairwise Influence Matrix I(u,v)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Influence distribution
    axes[1].hist(flat[flat > 0], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[1].axvline(np.median(flat[flat > 0]), color="red", linestyle="--",
                    label=f"Median={np.median(flat[flat > 0]):.3f}")
    axes[1].axvline(np.mean(flat[flat > 0]), color="orange", linestyle="--",
                    label=f"Mean={np.mean(flat[flat > 0]):.3f}")
    axes[1].set_xlabel("Influence magnitude")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Influence Distribution (Gini={gini:.3f})")
    axes[1].legend(fontsize=8)

    # Per-source-head total influence
    row_totals = I_mat.sum(axis=1)
    axes[2].bar(range(n_heads), row_totals, color="darkorange", alpha=0.8)
    axes[2].set_xlabel(f"Source head (Layer {source_layer})")
    axes[2].set_ylabel("Total downstream influence")
    axes[2].set_title("Per-Head Total Influence")

    plt.suptitle("Experiment 3: Sparsity & Modularity Analysis",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp3_sparsity_modularity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {prefix}exp3_sparsity_modularity.png")

    return gini


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Visualization (influence + CKA heatmaps)
# ══════════════════════════════════════════════════════════════════════════════

def plot_influence_and_cka(I_mat, C_mat, all_targets, n_heads, source_layer, prefix=""):
    """Plot side-by-side influence and CKA heatmaps."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: INFLUENCE & CKA VISUALIZATION")
    print("=" * 65)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Influence heatmap
    im1 = axes[0].imshow(I_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel("Target head index")
    axes[0].set_ylabel(f"Source head (Layer {source_layer})")
    axes[0].set_title("Representation Shift I(u,v)\n(higher = more influence)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # CKA heatmap (inverted: low CKA = high disruption)
    im2 = axes[1].imshow(1 - C_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_xlabel("Target head index")
    axes[1].set_ylabel(f"Source head (Layer {source_layer})")
    axes[1].set_title("CKA Disruption (1 - CKA)\n(higher = more structural change)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle("Experiment 1: Pairwise Influence Matrix + CKA",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp1_influence_cka.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp1_influence_cka.png")

    # Correlation between shift and CKA disruption
    flat_I = I_mat.ravel()
    flat_C = (1 - C_mat).ravel()
    mask = (flat_I > 0) & np.isfinite(flat_C)
    if mask.sum() > 2:
        r = float(np.corrcoef(flat_I[mask], flat_C[mask])[0, 1])
        print(f"  Pearson r(shift, 1-CKA) = {r:.3f}")
        if r > 0.7:
            print("  ✓ Shift and CKA disruption are strongly correlated")
        elif r > 0.4:
            print("  ~ Moderate correlation — metrics capture partially different signals")
        else:
            print("  ✗ Weak correlation — shift and CKA capture different phenomena")


# ══════════════════════════════════════════════════════════════════════════════
#  Domain-conditional analysis
# ══════════════════════════════════════════════════════════════════════════════

def domain_conditional_analysis(model, tokenizer, source_layer, prefix=""):
    """
    Compute influence matrices separately for code, math, and language.
    Compare domain-specific structure.
    """
    print("\n" + "=" * 65)
    print("  DOMAIN-CONDITIONAL INFLUENCE ANALYSIS")
    print("=" * 65)

    n_heads = model.config.n_head
    domain_matrices = {}

    for domain, texts in DOMAIN_PROBES.items():
        print(f"\n  ── Domain: {domain} ──")
        I_d, _, targets = compute_influence_matrix(model, tokenizer, texts, source_layer)
        I_mat_d = influence_to_matrix(I_d, n_heads, targets)
        domain_matrices[domain] = I_mat_d

    # Cross-domain similarity
    domains = list(domain_matrices.keys())
    print(f"\n  Cross-domain influence similarity (Pearson r):")
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            d1, d2 = domains[i], domains[j]
            r = float(np.corrcoef(domain_matrices[d1].ravel(),
                                  domain_matrices[d2].ravel())[0, 1])
            print(f"    {d1} vs {d2}: r = {r:+.3f}")

    # Plot domain comparison
    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5))
    for i, domain in enumerate(domains):
        im = axes[i].imshow(domain_matrices[domain], aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
        axes[i].set_xlabel("Target head index")
        axes[i].set_ylabel(f"Source head (Layer {source_layer})")
        axes[i].set_title(f"{domain.capitalize()} Domain")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle("Domain-Conditional Influence Matrices",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp3_domain_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp3_domain_comparison.png")

    return domain_matrices


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2 Stage 1: Representation Dependency Mapping"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local model weights")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (gpt2 or gpt2-medium)")
    parser.add_argument("--source_layer", type=int, default=None,
                        help="Source layer to analyze (default: 0 for Small, 2 for Medium)")
    parser.add_argument("--skip_domain", action="store_true",
                        help="Skip domain-conditional analysis (faster)")
    args = parser.parse_args()

    print("=" * 65)
    print("  PAPER 2 — STAGE 1: REPRESENTATION DEPENDENCY MAPPING")
    print("=" * 65)

    model, tokenizer = load_model(args.model_path, args.model)
    n_heads  = model.config.n_head
    n_layers = model.config.n_layer

    # Default source layer: 0 for Small, 2 for Medium (where bridges live)
    source_layer = args.source_layer
    if source_layer is None:
        source_layer = 2 if n_layers == 24 else 0
    print(f"  Source layer: {source_layer}")

    prefix = "medium_" if n_layers == 24 else "small_"

    # ── Experiment 1: Pairwise Influence Matrix + CKA ──
    all_texts = [t for v in DOMAIN_PROBES.values() for t in v]
    print(f"\n  Using {len(all_texts)} probe texts across all domains")

    I_dict, C_dict, all_targets = compute_influence_matrix(
        model, tokenizer, all_texts, source_layer
    )
    I_mat = influence_to_matrix(I_dict, n_heads, all_targets)
    C_mat = cka_to_matrix(C_dict, n_heads, all_targets)

    # Save raw data
    np.save(f"{prefix}influence_matrix.npy", I_mat)
    np.save(f"{prefix}cka_matrix.npy", C_mat)
    print(f"  Saved raw matrices -> {prefix}influence_matrix.npy, {prefix}cka_matrix.npy")

    # Visualize
    plot_influence_and_cka(I_mat, C_mat, all_targets, n_heads, source_layer, prefix)

    # ── Experiment 2: SVD ──
    S, cumvar = analyze_influence_spectrum(I_mat, prefix)

    # ── Experiment 3: Sparsity & Modularity ──
    gini = analyze_sparsity_and_modularity(
        I_mat, I_dict, all_targets, n_heads, source_layer, prefix
    )

    # ── Domain-conditional (optional) ──
    if not args.skip_domain:
        domain_mats = domain_conditional_analysis(model, tokenizer, source_layer, prefix)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  STAGE 1 SUMMARY")
    print("=" * 65)
    top5_var = cumvar[min(4, len(cumvar)-1)]
    print(f"\n  Influence matrix shape:  {I_mat.shape}")
    print(f"  Gini coefficient:       {gini:.4f}")
    print(f"  Top 5 SVs variance:     {top5_var:.1f}%")
    print(f"\n  Stage 1 Decision:")
    if gini > 0.6 and top5_var > 60:
        print("  ✓ PROCEED TO STAGE 2: Influence is sparse AND low-rank")
        print("    Dependencies are structured → extract dependency network")
    elif gini > 0.4 or top5_var > 50:
        print("  ~ BORDERLINE: Some structure detected")
        print("    Consider Stage 2 with caution")
    else:
        print("  ✗ STOP: Influence is diffuse and full-rank (null hypothesis)")
        print("    Bridge importance is geometric, not circuit-like")
        print("    → Redirect Paper 3 toward geometry-based pruning")

    print("\n" + "=" * 65)
    print("  Stage 1 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
