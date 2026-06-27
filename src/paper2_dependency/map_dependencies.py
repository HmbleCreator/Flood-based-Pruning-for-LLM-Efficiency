"""
Paper 2 — Stage 1: Representation Dependency Mapping
=====================================================
Model: GPT-2 Small / GPT-2 Medium

Computes the pairwise influence matrix I(u,v) measuring how much
ablating head u shifts the output representation of downstream head v.
The metric is normalized to represent percentage change:
    I(u,v) = ||h_v - h_v^{\\backslash u}|| / (||h_v|| + \\epsilon)

Stage 1 is purely empirical — reporting observations without graph interpretation.

Experiments:
  1. Pairwise Influence Matrix + CKA similarity (with Random Control Baseline)
  2. Influence Spectrum (SVD analysis with repeated permutation test & PR)
  3. Sparsity (Gini coefficient, Lorenz curve) and Modularity
  4. Domain-Conditional Influence
  5. Conditional Influence (Distribution of drops across top 20 edges)

Usage:
  python map_dependencies.py
  python map_dependencies.py --model_path C:/path/to/gpt2_local
  python map_dependencies.py --source_heads L0H00,L0H07
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
import re
from scipy.stats import ttest_ind

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
#  Helper functions for Hook captures
# ══════════════════════════════════════════════════════════════════════════════

def _capture_head_outputs(model, tokenizer, text, max_length=64):
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    captures = {}
    hooks    = []

    def make_hook(layer_idx):
        def hook_fn(module, args):
            x = args[0].detach().float().squeeze(0)  
            for h in range(n_heads):
                s, e = h * d_head, (h + 1) * d_head
                captures[(layer_idx, h)] = x[:, s:e].clone()  
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
                                  max_length=64, second_ablate_layer=None, second_ablate_head=None):
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    captures = {}
    hooks    = []

    a_s, a_e = ablate_head * d_head, (ablate_head + 1) * d_head
    ablations = [(ablate_layer, a_s, a_e)]
    if second_ablate_layer is not None and second_ablate_head is not None:
        s_s, s_e = second_ablate_head * d_head, (second_ablate_head + 1) * d_head
        ablations.append((second_ablate_layer, s_s, s_e))

    def make_ablation_hook(ablate_start, ablate_end):
        def ablation_hook(module, args):
            x = args[0].clone()
            x[:, :, ablate_start:ablate_end] = 0.0
            return (x,)
        return ablation_hook

    for al, start, end in ablations:
        hk = model.transformer.h[al].attn.c_proj.register_forward_pre_hook(
            make_ablation_hook(start, end)
        )
        hooks.append(hk)

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
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (X.T @ Y).norm('fro') ** 2
    hsic_xx = (X.T @ X).norm('fro') ** 2
    hsic_yy = (Y.T @ Y).norm('fro') ** 2
    denom = (hsic_xx * hsic_yy).sqrt()
    if denom < 1e-12:
        return 1.0  
    return float(hsic_xy / denom)


# ══════════════════════════════════════════════════════════════════════════════
#  Bootstrap & Permutation helpers
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_metric(data_by_text, metric_fn, num_bootstraps=200):
    n_texts = data_by_text.shape[0]
    boot_vals = []
    for _ in range(num_bootstraps):
        indices = np.random.choice(n_texts, size=n_texts, replace=True)
        sample_mean_matrix = data_by_text[indices].mean(axis=0)
        boot_vals.append(metric_fn(sample_mean_matrix))
    boot_vals = np.sort(boot_vals)
    low = np.percentile(boot_vals, 2.5)
    high = np.percentile(boot_vals, 97.5)
    return low, high


def bootstrap_domain_correlations(domain_data, num_bootstraps=200):
    domains = list(domain_data.keys())
    n_texts = 5
    corrs = { (d1, d2): [] for i, d1 in enumerate(domains) for d2 in domains[i+1:] }
    
    for _ in range(num_bootstraps):
        resampled_mats = {}
        for d in domains:
            idx = np.random.choice(n_texts, size=n_texts, replace=True)
            resampled_mats[d] = domain_data[d][idx].mean(axis=0).ravel()
        for (d1, d2) in corrs:
            r = np.corrcoef(resampled_mats[d1], resampled_mats[d2])[0, 1]
            corrs[(d1, d2)].append(r)
            
    ci = {}
    for pair, vals in corrs.items():
        ci[pair] = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return ci


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1 — Pairwise Influence Matrix + CKA + Controls
# ══════════════════════════════════════════════════════════════════════════════

def compute_influence_matrix(model, tokenizer, texts, source_heads):
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    all_targets = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    # Store raw values per text to enable post-hoc bootstrapping
    I_by_text = np.zeros((len(texts), len(source_heads), len(all_targets)))
    C_by_text = np.ones((len(texts), len(source_heads), len(all_targets))) # Default CKA = 1.0

    target_idx = {t: i for i, t in enumerate(all_targets)}

    print(f"\n  Computing influence matrix for {len(source_heads)} source heads "
          f"-> {len(all_targets)} target heads")
    print(f"  Probe texts: {len(texts)}")

    for ti, text in enumerate(texts):
        base = _capture_head_outputs(model, tokenizer, text)

        for src_idx, (src_l, src_h) in enumerate(source_heads):
            ablated = _capture_head_outputs_ablated(
                model, tokenizer, text, src_l, src_h
            )

            for tgt_idx, tgt in enumerate(all_targets):
                if tgt[0] <= src_l:
                    continue  # Target is not downstream
                
                if tgt in base and tgt in ablated:
                    b = base[tgt]    
                    a = ablated[tgt] 

                    # Percentage change in representation norm
                    shift = ((b - a).norm(dim=-1) / (b.norm(dim=-1) + 1e-8)).mean().item()
                    I_by_text[ti, src_idx, tgt_idx] = shift

                    # CKA similarity per sequence
                    if b.shape[0] >= 2:  
                        C_by_text[ti, src_idx, tgt_idx] = linear_cka(b, a)

        if (ti + 1) % 5 == 0 or ti == len(texts) - 1:
            print(f"    Text {ti+1}/{len(texts)} done")

    return I_by_text, C_by_text, all_targets


def select_control_heads(model, source_heads):
    n_heads = model.config.n_head
    control_heads = []
    
    by_layer = defaultdict(set)
    for l, h in source_heads:
        by_layer[l].add(h)
        
    for l, heads in by_layer.items():
        all_heads = set(range(n_heads))
        available = list(all_heads - heads)
        if len(available) >= len(heads):
            chosen = np.random.choice(available, size=len(heads), replace=False)
            for h in chosen:
                control_heads.append((l, int(h)))
        else:
            for h in range(n_heads):
                if (l, h) not in source_heads:
                    control_heads.append((l, h))
                    if len(control_heads) == len(source_heads):
                        break
    return control_heads


def plot_influence_and_cka(I_mat, C_mat, prefix=""):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: INFLUENCE & CKA VISUALIZATION")
    print("=" * 65)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(I_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel("Target head index")
    axes[0].set_ylabel("Source head index")
    axes[0].set_title("Representation Shift I(u,v)\n(higher = more influence)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(1 - C_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_xlabel("Target head index")
    axes[1].set_ylabel("Source head index")
    axes[1].set_title("CKA Disruption (1 - CKA)\n(higher = more structural change)")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.suptitle("Experiment 1: Pairwise Influence Matrix + CKA",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp1_influence_cka.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp1_influence_cka.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 — Influence Spectrum (SVD)
# ══════════════════════════════════════════════════════════════════════════════

def pr_metric(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    return float((S.sum() ** 2) / ((S ** 2).sum() + 1e-12))


def repeated_permutation_test(I_mat, num_permutations=1000):
    shuffled_S = []
    flat = I_mat.copy().ravel()
    for _ in range(num_permutations):
        np.random.shuffle(flat)
        sh_mat = flat.reshape(I_mat.shape)
        _, S_rand, _ = np.linalg.svd(sh_mat, full_matrices=False)
        shuffled_S.append(S_rand)
    shuffled_S = np.array(shuffled_S)
    
    _, S_empirical, _ = np.linalg.svd(I_mat, full_matrices=False)
    p_values = []
    for i in range(len(S_empirical)):
        p = np.sum(shuffled_S[:, i] >= S_empirical[i]) / num_permutations
        p_values.append(p)
        
    return S_empirical, shuffled_S, p_values


def analyze_influence_spectrum(I_mat, I_by_text, prefix=""):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: INFLUENCE SPECTRUM (SVD)")
    print("=" * 65)

    S, cumvar = analyze_svd_properties(I_mat)
    pr = pr_metric(I_mat)

    # Bootstrap CIs
    pr_low, pr_high = bootstrap_metric(I_by_text, pr_metric)
    
    # Repeated Permutation Test
    S_emp, S_rand_dist, p_values = repeated_permutation_test(I_mat, num_permutations=1000)

    print(f"\n  Matrix shape: {I_mat.shape}")
    print(f"  Effective Routing Dimensionality (Participation Ratio): {pr:.2f} (95% CI: [{pr_low:.2f}, {pr_high:.2f}])")
    
    print(f"\n  Top singular values vs. permutation baseline:")
    for i in range(min(10, len(S))):
        rand_mean = S_rand_dist[:, i].mean()
        rand_95_low = np.percentile(S_rand_dist[:, i], 2.5)
        rand_95_high = np.percentile(S_rand_dist[:, i], 97.5)
        print(f"    σ_{i+1:02d} = {S[i]:.4f} | Shuffled mean: {rand_mean:.4f} [95% CI: {rand_95_low:.4f}, {rand_95_high:.4f}] | p = {p_values[i]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot empirical vs shuffled spectrum
    axes[0].bar(range(len(S)), S, color="steelblue", alpha=0.8, label="Empirical")
    # Error band for shuffled spectrum
    rand_low = np.percentile(S_rand_dist, 2.5, axis=0)
    rand_high = np.percentile(S_rand_dist, 97.5, axis=0)
    axes[0].fill_between(range(len(S)), rand_low, rand_high, color="red", alpha=0.2, label="Shuffled (95% CI)")
    axes[0].plot(range(len(S)), S_rand_dist.mean(axis=0), color="red", linestyle="--", label="Shuffled Mean")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("Singular Value Spectrum")
    axes[0].legend()

    # Cumulative variance plot
    total_var = (S ** 2).sum()
    cumulative = np.cumsum(S ** 2) / total_var * 100
    cum_rand_mean = np.cumsum(S_rand_dist.mean(axis=0) ** 2) / (S_rand_dist.mean(axis=0) ** 2).sum() * 100
    axes[1].plot(range(1, len(cumulative)+1), cumulative, "o-", color="darkorange",
                 markersize=3, label="Empirical")
    axes[1].plot(range(1, len(cum_rand_mean)+1), cum_rand_mean, "x--", color="red",
                 markersize=3, label="Shuffled Mean")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative Variance Explained")
    axes[1].legend()

    plt.suptitle(f"Experiment 2: Influence Spectrum (Effective routing dim = {pr:.2f})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp2_influence_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp2_influence_spectrum.png")

    return S, cumulative


def analyze_svd_properties(I_mat):
    U, S, Vt = np.linalg.svd(I_mat, full_matrices=False)
    total_var = (S ** 2).sum()
    cumulative = np.cumsum(S ** 2) / total_var * 100
    return S, cumulative


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3 — Sparsity (Gini) and Modularity
# ══════════════════════════════════════════════════════════════════════════════

def gini_metric(mat):
    return gini_coefficient(mat.ravel())


def gini_coefficient(values):
    v = np.sort(np.abs(values).ravel())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)


def analyze_sparsity_and_modularity(I_mat, I_by_text, source_heads, all_targets, prefix=""):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: SPARSITY & MODULARITY")
    print("=" * 65)

    flat = I_mat.ravel()
    gini = gini_coefficient(flat)
    gini_low, gini_high = bootstrap_metric(I_by_text, gini_metric)
    
    print(f"\n  Gini coefficient of influence distribution: {gini:.4f} (95% CI: [{gini_low:.4f}, {gini_high:.4f}])")

    print(f"\n  Per-source-head influence concentration:")
    for i, (sl, sh) in enumerate(source_heads):
        row = I_mat[i, :]
        row_gini = gini_coefficient(row)
        top_pct = np.sort(row)[::-1][:max(1, len(row)//10)].sum() / (row.sum() + 1e-12) * 100
        print(f"    L{sl:02d}H{sh:02d}: Gini={row_gini:.3f}, "
              f"Top 10% targets carry {top_pct:.1f}% of influence")

    col_totals = I_mat.sum(axis=0)
    top_k = min(10, len(col_totals))
    top_idx = np.argsort(col_totals)[::-1][:top_k]
    print(f"\n  Top {top_k} most-influenced downstream heads:")
    for rank, idx in enumerate(top_idx):
        tgt = all_targets[idx]
        print(f"    {rank+1}. L{tgt[0]:02d}H{tgt[1]:02d}: "
              f"total influence = {col_totals[idx]:.4f}")

    try:
        from sklearn.cluster import SpectralClustering
        if I_mat.shape[1] >= 4:
            n_clusters = min(4, I_mat.shape[1] // 2)
            affinity = np.abs(I_mat.T @ I_mat)  
            np.fill_diagonal(affinity, 0)
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                   random_state=42)
            labels = sc.fit_predict(affinity + 1e-10)

            print(f"\n  Spectral clustering ({n_clusters} clusters):")
            for c in range(n_clusters):
                members = [all_targets[i] for i in range(len(labels)) if labels[i] == c]
                layers = [m[0] for m in members]
                if members:
                    print(f"    Cluster {c}: {len(members)} heads, "
                          f"layers {min(layers)}-{max(layers)}")
    except ImportError:
        print("\n  [sklearn not available — skipping spectral clustering]")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im = axes[0].imshow(I_mat, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel("Target head index")
    axes[0].set_ylabel("Source head index")
    axes[0].set_title("Pairwise Influence Matrix I(u,v)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(flat[flat > 0], bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    axes[1].axvline(np.median(flat[flat > 0]), color="red", linestyle="--",
                    label=f"Median={np.median(flat[flat > 0]):.4f}")
    axes[1].axvline(np.mean(flat[flat > 0]), color="orange", linestyle="--",
                    label=f"Mean={np.mean(flat[flat > 0]):.4f}")
    axes[1].set_xlabel("Influence magnitude (Normalized)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Influence Distribution (Gini={gini:.3f})")
    axes[1].legend(fontsize=8)

    # Lorenz Curve
    v_sorted = np.sort(np.abs(flat[flat > 0]))
    if len(v_sorted) > 0:
        lorenz = np.cumsum(v_sorted) / v_sorted.sum()
        lorenz = np.insert(lorenz, 0, 0)
        axes[2].plot(np.linspace(0, 1, len(lorenz)), lorenz, color="darkorange", lw=2)
        axes[2].plot([0, 1], [0, 1], color="gray", linestyle="--")
        axes[2].set_title("Lorenz Curve")
        axes[2].set_xlabel("Cumulative share of target pairs")
        axes[2].set_ylabel("Cumulative share of influence")

    row_totals = I_mat.sum(axis=1)
    axes[3].bar(range(len(source_heads)), row_totals, color="darkorange", alpha=0.8)
    axes[3].set_xticks(range(len(source_heads)))
    axes[3].set_xticklabels([f"L{sl}H{sh}" for sl, sh in source_heads], rotation=45)
    axes[3].set_ylabel("Total downstream influence")
    axes[3].set_title("Per-Head Total Influence")

    plt.suptitle("Experiment 3: Sparsity & Modularity Analysis",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp3_sparsity_modularity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {prefix}exp3_sparsity_modularity.png")

    return gini


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 4 — Domain-Conditional Influence
# ══════════════════════════════════════════════════════════════════════════════

def experiment4_domain_conditional(model, tokenizer, source_heads, prefix=""):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: DOMAIN-CONDITIONAL INFLUENCE")
    print("=" * 65)

    domain_data = {}
    domain_matrices = {}

    for domain, texts in DOMAIN_PROBES.items():
        print(f"\n  ── Domain: {domain} ──")
        I_by_text_d, _, targets = compute_influence_matrix(model, tokenizer, texts, source_heads)
        domain_data[domain] = I_by_text_d
        domain_matrices[domain] = I_by_text_d.mean(axis=0)

    domains = list(domain_matrices.keys())
    
    # Compute cross-domain Pearson correlations with bootstrap CIs
    print(f"\n  Cross-domain influence similarity (Pearson r with 95% bootstrap CI):")
    domain_cis = bootstrap_domain_correlations(domain_data, num_bootstraps=200)
    
    for i in range(len(domains)):
        for j in range(i+1, len(domains)):
            d1, d2 = domains[i], domains[j]
            r = float(np.corrcoef(domain_matrices[d1].ravel(), domain_matrices[d2].ravel())[0, 1])
            ci_low, ci_high = domain_cis[(d1, d2)]
            print(f"    {d1} vs {d2}: r = {r:+.3f} (95% CI: [{ci_low:+.3f}, {ci_high:+.3f}])")

    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5))
    for i, domain in enumerate(domains):
        im = axes[i].imshow(domain_matrices[domain], aspect='auto', cmap='YlOrRd',
                           interpolation='nearest')
        axes[i].set_xlabel("Target head index")
        axes[i].set_ylabel("Source head index")
        axes[i].set_title(f"{domain.capitalize()} Domain")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle("Experiment 4: Domain-Conditional Influence Matrices",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}exp4_domain_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {prefix}exp4_domain_comparison.png")

    return domain_matrices


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 5 — Conditional Influence
# ══════════════════════════════════════════════════════════════════════════════

def compute_conditional_influence(model, tokenizer, texts, source, mediator, target, verbose=True):
    base_shifts = []
    cond_shifts = []

    for text in texts:
        base = _capture_head_outputs(model, tokenizer, text)
        
        ablated_src = _capture_head_outputs_ablated(
            model, tokenizer, text, source[0], source[1]
        )
        
        ablated_med = _capture_head_outputs_ablated(
            model, tokenizer, text, mediator[0], mediator[1]
        )
        
        ablated_src_med = _capture_head_outputs_ablated(
            model, tokenizer, text, source[0], source[1],
            second_ablate_layer=mediator[0], second_ablate_head=mediator[1]
        )

        if target in base and target in ablated_src and target in ablated_med and target in ablated_src_med:
            # Base influence I(source, target)
            b = base[target]
            a_s = ablated_src[target]
            shift = ((b - a_s).norm(dim=-1) / (b.norm(dim=-1) + 1e-8)).mean().item()
            base_shifts.append(shift)

            # Conditional influence I(source, target | mediator) = || h^{\\w} - h^{\\u,\\w} ||
            a_m = ablated_med[target]
            a_sm = ablated_src_med[target]
            cond_shift = ((a_m - a_sm).norm(dim=-1) / (a_m.norm(dim=-1) + 1e-8)).mean().item()
            cond_shifts.append(cond_shift)

    base_val = float(np.mean(base_shifts))
    cond_val = float(np.mean(cond_shifts))
    drop_pct = (base_val - cond_val) / (base_val + 1e-8) * 100

    if verbose:
        print(f"    Base influence I(u, v):         {base_val:.4f}")
        print(f"    Cond influence I(u, v | w):     {cond_val:.4f}")
        print(f"    Influence drop when conditioned: {drop_pct:.1f}%")

    return base_val, cond_val


def experiment5_conditional_influence(model, tokenizer, texts, I_mat, source_heads, all_targets):
    print("\n" + "=" * 65)
    print("  EXPERIMENT 5: CONDITIONAL INFLUENCE (Dependency Chains)")
    print("=" * 65)
    
    if len(source_heads) == 0:
        return
        
    # Find all pairs (u, v) where target v is at least 2 layers downstream of source u
    pairs = []
    target_idx = {t: i for i, t in enumerate(all_targets)}
    source_idx = {s: i for i, s in enumerate(source_heads)}
    
    for src in source_heads:
        for tgt in all_targets:
            if tgt[0] >= src[0] + 2:
                influence_val = I_mat[source_idx[src], target_idx[tgt]]
                pairs.append((influence_val, src, tgt))
                
    pairs.sort(key=lambda x: x[0], reverse=True)
    top_pairs = pairs[:20]
    
    if not top_pairs:
        print("  No candidate pairs with target at least 2 layers downstream.")
        return
        
    print(f"  Evaluating top {len(top_pairs)} strongest influence pairs for mediation drops...")
    
    all_drops = []
    
    for rank, (val, source, target) in enumerate(top_pairs):
        # Candidates w: layer_u < layer_w < layer_v
        mediators = [t for t in all_targets if source[0] < t[0] < target[0]]
        if not mediators:
            continue
            
        # Select mediator w that has the highest product of empirical influence I(u, w) * I(w, v)
        mediator_scores = []
        for w in mediators:
            if w in source_idx:
                score = I_mat[source_idx[source], target_idx[w]] * I_mat[source_idx[w], target_idx[target]]
            else:
                score = I_mat[source_idx[source], target_idx[w]]
            mediator_scores.append((score, w))
            
        mediator_scores.sort(reverse=True)
        best_mediator = mediator_scores[0][1]
        
        base_val, cond_val = compute_conditional_influence(model, tokenizer, texts, source, best_mediator, target, verbose=False)
        drop_pct = (base_val - cond_val) / (base_val + 1e-8) * 100
        all_drops.append(drop_pct)
        print(f"    Rank {rank+1:02d}: L{source[0]}H{source[1]} -> L{target[0]}H{target[1]} via L{best_mediator[0]}H{best_mediator[1]} | Drop: {drop_pct:.1f}%")
        
    if all_drops:
        median_drop = np.median(all_drops)
        low_drop = np.percentile(all_drops, 25)
        high_drop = np.percentile(all_drops, 75)
        print(f"\n  Mediation drop distribution across top pairs:")
        print(f"    Median drop: {median_drop:.1f}% (IQR: [{low_drop:.1f}%, {high_drop:.1f}%])")


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
    parser.add_argument("--source_heads", type=str, default=None,
                        help="Comma-separated list of heads, e.g., L0H00,L0H07. Defaults to all heads in the bridge layer (L0 for Small, L2 for Medium).")
    parser.add_argument("--skip_domain", action="store_true",
                        help="Skip Exp 4 domain-conditional analysis (faster)")
    parser.add_argument("--skip_conditional", action="store_true",
                        help="Skip Exp 5 conditional influence analysis (faster)")
    args = parser.parse_args()

    print("=" * 65)
    print("  PAPER 2 — STAGE 1: REPRESENTATION DEPENDENCY MAPPING")
    print("=" * 65)

    model, tokenizer = load_model(args.model_path, args.model)
    n_heads  = model.config.n_head
    n_layers = model.config.n_layer

    source_heads = []
    if args.source_heads:
        for item in args.source_heads.split(','):
            match = re.match(r'L(\d+)H(\d+)', item.strip(), re.IGNORECASE)
            if match:
                source_heads.append((int(match.group(1)), int(match.group(2))))
    else:
        default_layer = 2 if n_layers == 24 else 0
        source_heads = [(default_layer, h) for h in range(n_heads)]
        
    print(f"  Source (Bridge) heads: {[f'L{l}H{h}' for l,h in source_heads]}")

    # Select random non-bridge control heads
    control_heads = select_control_heads(model, source_heads)
    print(f"  Control (Random) heads: {[f'L{l}H{h}' for l,h in control_heads]}")

    prefix = "medium_" if n_layers == 24 else "small_"

    # Run influence matrix calculation for both bridge and control heads
    all_sources = source_heads + control_heads
    all_texts = [t for v in DOMAIN_PROBES.values() for t in v]
    print(f"\n  Using {len(all_texts)} probe texts across all domains")

    I_by_text, C_by_text, all_targets = compute_influence_matrix(
        model, tokenizer, all_texts, all_sources
    )

    # Slice the results back into Bridge vs Control
    I_by_text_bridge = I_by_text[:, :len(source_heads), :]
    I_by_text_control = I_by_text[:, len(source_heads):, :]
    
    C_by_text_bridge = C_by_text[:, :len(source_heads), :]
    C_by_text_control = C_by_text[:, len(source_heads):, :]

    I_mat = I_by_text_bridge.mean(axis=0)
    C_mat = C_by_text_bridge.mean(axis=0)

    I_mat_ctrl = I_by_text_control.mean(axis=0)

    # Save raw data for bridge heads
    np.save(f"{prefix}influence_matrix.npy", I_mat)
    np.save(f"{prefix}cka_matrix.npy", C_mat)
    print(f"  Saved raw matrices -> {prefix}influence_matrix.npy, {prefix}cka_matrix.npy")

    # ── Experiment 1 Baseline Comparison ──
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: BASELINE COMPARISON (Bridge vs. Control)")
    print("=" * 65)
    mean_bridge = I_mat.mean()
    mean_ctrl = I_mat_ctrl.mean()
    
    # Bootstrap CI for mean difference
    diff_vals = []
    n_texts = len(all_texts)
    for _ in range(200):
        idx = np.random.choice(n_texts, size=n_texts, replace=True)
        diff_vals.append(I_by_text_bridge[idx].mean() - I_by_text_control[idx].mean())
    diff_vals = np.sort(diff_vals)
    diff_low = np.percentile(diff_vals, 2.5)
    diff_high = np.percentile(diff_vals, 97.5)

    # Simple t-test on flat distributions
    t_stat, p_val = ttest_ind(I_mat.ravel(), I_mat_ctrl.ravel(), equal_var=False)
    print(f"  Mean Bridge Influence:         {mean_bridge:.4f}")
    print(f"  Mean Control Influence:        {mean_ctrl:.4f}")
    print(f"  Mean Difference (Bridge-Ctrl): {mean_bridge - mean_ctrl:+.4f} (95% CI: [{diff_low:+.4f}, {diff_high:+.4f}])")
    print(f"  t-statistic:                   {t_stat:.4f} | p-value: {p_val:.4f}")

    plot_influence_and_cka(I_mat, C_mat, prefix)

    # ── Experiment 2: SVD ──
    S, cumvar = analyze_influence_spectrum(I_mat, I_by_text_bridge, prefix)

    # ── Experiment 3: Sparsity & Modularity ──
    gini = analyze_sparsity_and_modularity(
        I_mat, I_by_text_bridge, source_heads, all_targets, prefix
    )

    # ── Experiment 4: Domain-conditional (optional) ──
    if not args.skip_domain:
        experiment4_domain_conditional(model, tokenizer, source_heads, prefix)

    # ── Experiment 5: Conditional Influence (optional) ──
    if not args.skip_conditional:
        experiment5_conditional_influence(model, tokenizer, all_texts, I_mat, source_heads, all_targets)

    print("\n" + "=" * 65)
    print("  STAGE 1 SUMMARY")
    print("=" * 65)
    top5_var = cumvar[min(4, len(cumvar)-1)]
    pr = pr_metric(I_mat)
    pr_low, pr_high = bootstrap_metric(I_by_text_bridge, pr_metric)
    gini_low, gini_high = bootstrap_metric(I_by_text_bridge, gini_metric)
    
    print(f"  Influence matrix shape:  {I_mat.shape}")
    print(f"  Gini coefficient:        {gini:.4f} (95% CI: [{gini_low:.4f}, {gini_high:.4f}])")
    print(f"  Effective routing dim:   {pr:.2f} (95% CI: [{pr_low:.2f}, {pr_high:.2f}])")
    print(f"  Top 5 SVs variance:      {top5_var:.1f}%")
    print(f"  Effective rank (80%):    {int(np.searchsorted(cumvar, 80.0)) + 1}")
    print("=" * 65)


if __name__ == "__main__":
    main()
