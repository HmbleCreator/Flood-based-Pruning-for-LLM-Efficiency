"""
Post-Hoc Control Analyzer
=========================
Loads saved bridge and control influence matrices and compares their SVD spectra, 
Participation Ratios, and Gini coefficients to establish if the low-dimensional 
routing geometry is specific to bridge heads.
"""

import numpy as np

def gini_coefficient(values):
    v = np.sort(np.abs(values).ravel())
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)

def analyze(prefix):
    print("\n" + "=" * 65)
    print(f"  ANALYZING {prefix.upper()} MODEL CONTROL BASELINES")
    print("=" * 65)
    
    try:
        # Load raw 3D data (num_texts, num_heads, num_targets)
        I_bridge_raw = np.load(f"{prefix}I_by_text.npy")
        I_ctrl_raw = np.load(f"{prefix}I_by_text_control.npy")
    except FileNotFoundError:
        print(f"  [Error] Raw data files ({prefix}I_by_text.npy / {prefix}I_by_text_control.npy) not found.")
        print("  Please run the map_dependencies.py script again to save these observations.")
        return

    # Compute mean matrices over probe texts
    I_bridge = I_bridge_raw.mean(axis=0)
    I_ctrl = I_ctrl_raw.mean(axis=0)

    # 1. Sparsity (Gini)
    gini_bridge = gini_coefficient(I_bridge)
    gini_ctrl = gini_coefficient(I_ctrl)

    # 2. SVD / Routing Dimension
    _, S_bridge, _ = np.linalg.svd(I_bridge, full_matrices=False)
    _, S_ctrl, _ = np.linalg.svd(I_ctrl, full_matrices=False)
    
    pr_bridge = (S_bridge.sum() ** 2) / (S_bridge ** 2).sum()
    pr_ctrl = (S_ctrl.sum() ** 2) / (S_ctrl ** 2).sum()

    print(f"\n  Metric Comparison (Bridge vs. Control):")
    print(f"    Gini Coefficient (Sparsity):")
    print(f"      Bridge:  {gini_bridge:.4f}")
    print(f"      Control: {gini_ctrl:.4f}")
    print(f"    Effective Routing Dimensionality (PR):")
    print(f"      Bridge:  {pr_bridge:.2f}")
    print(f"      Control: {pr_ctrl:.2f}")

    print(f"\n  Singular Value Distribution:")
    print("    Bridge singular values:")
    print("      " + ", ".join([f"{s:.3f}" for s in S_bridge[:5]]))
    print("    Control singular values:")
    print("      " + ", ".join([f"{s:.3f}" for s in S_ctrl[:5]]))

    print("\n" + "=" * 65)

if __name__ == "__main__":
    for prefix in ["small_", "medium_"]:
        analyze(prefix)
