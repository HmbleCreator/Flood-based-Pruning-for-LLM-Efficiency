"""
Compare Bridge and Control Routing Geometry
==========================================
Loads saved bridge and control raw influence observations (from map_dependencies.py)
and runs comparative analyses:
  1. Bootstrapped Participation Ratio (PR) & Gini coefficients for both, with 95% CIs.
  2. Bootstrapped singular value ratio (s1_bridge / s1_control) with 95% CI.
  3. Cumulative explained variance curves plotted side-by-side.
  4. Subspace alignment: Cosine similarity between the first right singular vectors (v1)
     representing the downstream routing directions, with bootstrapped 95% CIs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gini_coefficient(values):
    v = np.sort(np.abs(values).ravel())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)

def get_pr(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    return float((S.sum() ** 2) / ((S ** 2).sum() + 1e-12))

def get_gini(mat):
    return gini_coefficient(mat.ravel())

def get_s1(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    return float(S[0])

def get_v1(mat):
    _, _, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[0, :].copy()

def compute_cosine_similarity(v1, v2):
    return float(np.abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)))

def analyze_geometry(prefix):
    print("\n" + "=" * 70)
    print(f"  ROUTING GEOMETRY COMPARISON: {prefix.upper()} MODEL")
    print("=" * 70)
    
    try:
        I_bridge_raw = np.load(f"{prefix}I_by_text.npy")
        I_ctrl_raw = np.load(f"{prefix}I_by_text_control.npy")
    except FileNotFoundError:
        print(f"  [Error] Raw data files ({prefix}I_by_text.npy / {prefix}I_by_text_control.npy) not found.")
        print("  Please run the map_dependencies.py script again to save these observations.")
        return

    n_texts = I_bridge_raw.shape[0]
    
    # ── 1. Point Estimates ──
    I_bridge_mean = I_bridge_raw.mean(axis=0)
    I_ctrl_mean = I_ctrl_raw.mean(axis=0)
    
    gini_b = get_gini(I_bridge_mean)
    gini_c = get_gini(I_ctrl_mean)
    
    pr_b = get_pr(I_bridge_mean)
    pr_c = get_pr(I_ctrl_mean)
    
    s1_b = get_s1(I_bridge_mean)
    s1_c = get_s1(I_ctrl_mean)
    s1_ratio = s1_b / s1_c
    
    v1_b = get_v1(I_bridge_mean)
    v1_c = get_v1(I_ctrl_mean)
    cos_sim = compute_cosine_similarity(v1_b, v1_c)

    # ── 2. Bootstrapping ──
    num_bootstraps = 200
    boot_gini_b = []
    boot_gini_c = []
    boot_gini_diff = []
    
    boot_pr_b = []
    boot_pr_c = []
    boot_pr_diff = []
    
    boot_s1_ratio = []
    boot_cos_sim = []
    
    for _ in range(num_bootstraps):
        idx = np.random.choice(n_texts, size=n_texts, replace=True)
        mat_b = I_bridge_raw[idx].mean(axis=0)
        mat_c = I_ctrl_raw[idx].mean(axis=0)
        
        # Gini
        gb = get_gini(mat_b)
        gc = get_gini(mat_c)
        boot_gini_b.append(gb)
        boot_gini_c.append(gc)
        boot_gini_diff.append(gb - gc)
        
        # PR
        pb = get_pr(mat_b)
        pc = get_pr(mat_c)
        boot_pr_b.append(pb)
        boot_pr_c.append(pc)
        boot_pr_diff.append(pb - pc)
        
        # s1 ratio
        boot_s1_ratio.append(get_s1(mat_b) / get_s1(mat_c))
        
        # Cosine similarity of first right singular vector
        boot_cos_sim.append(compute_cosine_similarity(get_v1(mat_b), get_v1(mat_c)))

    # Compute 95% CIs
    ci = lambda vals: (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    
    ci_gini_b = ci(boot_gini_b)
    ci_gini_c = ci(boot_gini_c)
    ci_gini_diff = ci(boot_gini_diff)
    
    ci_pr_b = ci(boot_pr_b)
    ci_pr_c = ci(boot_pr_c)
    ci_pr_diff = ci(boot_pr_diff)
    
    ci_s1_ratio = ci(boot_s1_ratio)
    ci_cos_sim = ci(boot_cos_sim)

    # ── 3. Print Results ──
    print(f"\n  Sparsity (Gini Coefficient):")
    print(f"    Bridge:  {gini_b:.4f} [95% CI: {ci_gini_b[0]:.4f}, {ci_gini_b[1]:.4f}]")
    print(f"    Control: {gini_c:.4f} [95% CI: {ci_gini_c[0]:.4f}, {ci_gini_c[1]:.4f}]")
    print(f"    Difference (Bridge - Control): {gini_b - gini_c:+.4f} [95% CI: {ci_gini_diff[0]:+.4f}, {ci_gini_diff[1]:+.4f}]")
    
    print(f"\n  Effective Routing Dimensionality (Participation Ratio):")
    print(f"    Bridge:  {pr_b:.2f} [95% CI: {ci_pr_b[0]:.2f}, {ci_pr_b[1]:.2f}]")
    print(f"    Control: {pr_c:.2f} [95% CI: {ci_pr_c[0]:.2f}, {ci_pr_c[1]:.2f}]")
    print(f"    Difference (Bridge - Control): {pr_b - pr_c:+.2f} [95% CI: {ci_pr_diff[0]:+.2f}, {ci_pr_diff[1]:+.2f}]")

    print(f"\n  Routing Magnitude:")
    print(f"    Bridge s1:  {s1_b:.4f}")
    print(f"    Control s1: {s1_c:.4f}")
    print(f"    Ratio (Bridge / Control s1): {s1_ratio:.2f} [95% CI: {ci_s1_ratio[0]:.2f}, {ci_s1_ratio[1]:.2f}]")

    print(f"\n  Subspace Alignment (Downstream Routing Direction):")
    print(f"    Cosine Similarity |<v1_bridge, v1_control>|:")
    print(f"      Observed: {cos_sim:.4f} [95% CI: {ci_cos_sim[0]:.4f}, {ci_cos_sim[1]:.4f}]")

    # ── 4. Plot Cumulative Variance Curves ──
    _, S_b, _ = np.linalg.svd(I_bridge_mean, full_matrices=False)
    _, S_c, _ = np.linalg.svd(I_ctrl_mean, full_matrices=False)
    
    cum_b = np.cumsum(S_b ** 2) / (S_b ** 2).sum() * 100
    cum_c = np.cumsum(S_c ** 2) / (S_c ** 2).sum() * 100

    plt.figure(figsize=(6, 4.5))
    plt.plot(range(1, len(cum_b)+1), cum_b, "o-", color="darkorange", label="Bridge", markersize=4)
    plt.plot(range(1, len(cum_c)+1), cum_c, "x--", color="steelblue", label="Control", markersize=4)
    plt.xlabel("Number of SVD Components")
    plt.ylabel("Cumulative Variance Explained (%)")
    plt.title(f"Cumulative Variance Explained: {prefix.capitalize()} Model")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_name = f"{prefix}geometry_comparison.png"
    plt.savefig(plot_name, dpi=150)
    plt.close()
    print(f"\n  Saved cumulative SVD variance plot -> {plot_name}")
    print("=" * 70)


if __name__ == "__main__":
    for prefix in ["small_", "medium_"]:
        analyze_geometry(prefix)
