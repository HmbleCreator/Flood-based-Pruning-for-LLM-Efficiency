"""
Map All Layers Routing Geometry (Refined V3)
============================================
Computes the downstream routing geometry (PR, average influence, s1) 
for EVERY layer in the model to characterize global representation routing.

Refinements in V3:
  - Added global subspace alignment: Cosine similarity of first right singular vectors
    between every pair of source layers (slicing to project onto shared downstream target spaces).
  - Plots the layer-by-layer alignment matrix as a 2D heatmap.
  - Reports average alignment across all layer pairs.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Probe texts (same domains as Paper 1 & map_dependencies.py) ─────────────────

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
    return mdl, tok

def enc(tok, text, max_length=64):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_length)

# ── Hook captures ──

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

def _capture_head_outputs_ablated(model, tokenizer, text, ablace_layer, ablate_head, max_length=64):
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    captures = {}
    hooks    = []

    a_s, a_e = ablate_head * d_head, (ablate_head + 1) * d_head
    
    def make_ablation_hook(ablate_start, ablate_end):
        def ablation_hook(module, args):
            x = args[0].clone()
            x[:, :, ablate_start:ablate_end] = 0.0
            return (x,)
        return ablation_hook

    hk = model.transformer.h[ablace_layer].attn.c_proj.register_forward_pre_hook(
        make_ablation_hook(a_s, a_e)
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

# ── Refactored SVD Metric Helper ──

def compute_svd_metrics(mat):
    u, S, Vt = np.linalg.svd(mat, full_matrices=False)
    if len(S) == 0 or (S**2).sum() == 0:
        return 0.0, 0.0, 0.0, 0.0, None
    pr = float((S.sum() ** 2) / ((S ** 2).sum() + 1e-12))
    var_exp = float((S[0] ** 2) / (S ** 2).sum() * 100)
    dominance = float(S[0] / (S.sum() + 1e-12))
    fro = float(np.linalg.norm(mat, "fro"))
    return pr, var_exp, dominance, fro, Vt[0, :].copy()

# ── Statistical Helpers ──

def bootstrap_layer_metrics(I_by_text_l, num_bootstraps=200, seed=42):
    rng = np.random.default_rng(seed)
    n_texts = I_by_text_l.shape[0]
    pr_vals = []
    var_vals = []
    mean_vals = []
    fro_vals = []
    
    for _ in range(num_bootstraps):
        idx = rng.choice(n_texts, size=n_texts, replace=True)
        mat = I_by_text_l[idx].mean(axis=0)
        pr, var_exp, _, fro, _ = compute_svd_metrics(mat)
        
        pr_vals.append(pr)
        var_vals.append(var_exp)
        mean_vals.append(mat.mean())
        fro_vals.append(fro)
        
    ci = lambda vals: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return ci(pr_vals), ci(var_vals), ci(mean_vals), ci(fro_vals)

def compute_permutation_baselines(I_mat, num_perms=100, seed=42):
    rng = np.random.default_rng(seed)
    pr_vals = []
    var_vals = []
    for _ in range(num_perms):
        sh_mat = rng.permutation(I_mat.ravel()).reshape(I_mat.shape)
        pr, var_exp, _, _, _ = compute_svd_metrics(sh_mat)
        pr_vals.append(pr)
        var_vals.append(var_exp)
    return float(np.mean(pr_vals)), float(np.mean(var_vals))

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2: Map downstream routing geometry for all layers (Refined V3)"
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local model weights")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model name (gpt2 or gpt2-medium)")
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("  MAPPING GLOBAL ROUTING GEOMETRY (ALL LAYERS)")
    print("=" * 70)

    model, tokenizer = load_model(args.model_path, args.model)
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head

    all_texts = [t for v in DOMAIN_PROBES.values() for t in v]
    prefix = "medium_" if n_layers == 24 else "small_"

    layers_pr = []
    layers_pr_ci_low = []
    layers_pr_ci_high = []
    
    layers_var_exp = []
    layers_var_exp_ci_low = []
    layers_var_exp_ci_high = []
    
    layers_mean_inf = []
    layers_mean_inf_ci_low = []
    layers_mean_inf_ci_high = []

    layers_fro = []
    layers_fro_ci_low = []
    layers_fro_ci_high = []
    
    layers_pr_perm = []
    layers_var_perm = []

    layers_x = list(range(n_layers - 1))  # Last layer has no downstream targets
    
    # Storage for subspace alignment calculation
    v1_dict = {}
    targets_dict = {}

    # 2D Heatmaps: Source Layer vs. Target Layer (Mean and Max)
    source_target_mean = np.zeros((n_layers, n_layers))
    source_target_max = np.zeros((n_layers, n_layers))

    print(f"\n  Running analysis for layers 0 to {n_layers - 2}...")
    
    for l in layers_x:
        source_heads = [(l, h) for h in range(n_heads)]
        all_targets = [(tgt_l, tgt_h) for tgt_l in range(l + 1, n_layers) for tgt_h in range(n_heads)]
        targets_dict[l] = all_targets
        
        # Temporary storage for text observations
        I_by_text = np.zeros((len(all_texts), n_heads, len(all_targets)))

        for ti, text in enumerate(all_texts):
            base = _capture_head_outputs(model, tokenizer, text)
            
            for src_idx, (src_l, src_h) in enumerate(source_heads):
                ablated = _capture_head_outputs_ablated(model, tokenizer, text, src_l, src_h)
                
                for tgt_idx, tgt in enumerate(all_targets):
                    if tgt in base and tgt in ablated:
                        b = base[tgt]
                        a = ablated[tgt]
                        shift = ((b - a).norm(dim=-1) / (b.norm(dim=-1) + 1e-8)).mean().item()
                        I_by_text[ti, src_idx, tgt_idx] = shift
                        
        I_mat = I_by_text.mean(axis=0)
        
        # Store Source-Target Layer Heatmap values (mean and max)
        for tgt_l in range(l + 1, n_layers):
            tgt_indices = [idx for idx, (tl, th) in enumerate(all_targets) if tl == tgt_l]
            source_target_mean[l, tgt_l] = I_by_text[:, :, tgt_indices].mean()
            source_target_max[l, tgt_l] = float(np.max(I_by_text[:, :, tgt_indices].mean(axis=0)))

        # SVD Point Estimates
        pr, var_exp, _, fro, v1 = compute_svd_metrics(I_mat)
        v1_dict[l] = v1
        mean_inf = I_mat.mean()
        
        # Bootstrap CIs
        ci_pr, ci_var, ci_mean, ci_fro = bootstrap_layer_metrics(I_by_text, num_bootstraps=200)
        
        # Permutation Baselines
        perm_pr, perm_var = compute_permutation_baselines(I_mat, num_perms=100)

        # Append data
        layers_pr.append(pr)
        layers_pr_ci_low.append(ci_pr[0])
        layers_pr_ci_high.append(ci_pr[1])
        
        layers_var_exp.append(var_exp)
        layers_var_exp_ci_low.append(ci_var[0])
        layers_var_exp_ci_high.append(ci_var[1])
        
        layers_mean_inf.append(mean_inf)
        layers_mean_inf_ci_low.append(ci_mean[0])
        layers_mean_inf_ci_high.append(ci_mean[1])

        layers_fro.append(fro)
        layers_fro_ci_low.append(ci_fro[0])
        layers_fro_ci_high.append(ci_fro[1])
        
        layers_pr_perm.append(perm_pr)
        layers_var_perm.append(perm_var)
        
        print(f"    Layer {l:02d}/{n_layers - 2} | "
              f"Mean Inf: {mean_inf:.4f} [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}] | "
              f"Fro Norm: {fro:.3f} [{ci_fro[0]:.3f}, {ci_fro[1]:.3f}] | "
              f"PR: {pr:.2f} [{ci_pr[0]:.2f}, {ci_pr[1]:.2f} (perm: {perm_pr:.2f})] | "
              f"s1 Var Explained: {var_exp:.1f}% [{ci_var[0]:.1f}%, {ci_var[1]:.1f}% (perm: {perm_var:.1f}%)]")

    # ── 5. Global Subspace Alignment Calculation ──
    print("\n  Calculating global subspace alignment (layer-by-layer)...")
    alignment_matrix = np.zeros((n_layers - 1, n_layers - 1))
    
    for i in range(n_layers - 1):
        for j in range(n_layers - 1):
            if i == j:
                alignment_matrix[i, j] = 1.0
                continue
            la, lb = min(i, j), max(i, j)
            targets_a = targets_dict[la]
            targets_b = targets_dict[lb]
            
            # targets_b is a subset of targets_a since lb > la
            shared_t = targets_b
            indices_a = [targets_a.index(t) for t in shared_t]
            
            v1_a_shared = v1_dict[la][indices_a]
            v1_b_shared = v1_dict[lb]
            
            norm_a = np.linalg.norm(v1_a_shared)
            norm_b = np.linalg.norm(v1_b_shared)
            
            if norm_a > 1e-12 and norm_b > 1e-12:
                cos_sim = float(np.abs(np.dot(v1_a_shared, v1_b_shared)) / (norm_a * norm_b))
            else:
                cos_sim = 0.0
            alignment_matrix[i, j] = cos_sim

    # Average alignment across upper triangle
    triu_indices = np.triu_indices(n_layers - 1, k=1)
    mean_alignment = alignment_matrix[triu_indices].mean()
    print(f"    Global Average Perturbation Subspace Alignment (Cosine Similarity): {mean_alignment:.4f}")

    # Save data arrays
    np.save(f"{prefix}all_layers_pr.npy", np.array(layers_pr))
    np.save(f"{prefix}all_layers_var_exp.npy", np.array(layers_var_exp))
    np.save(f"{prefix}all_layers_mean_inf.npy", np.array(layers_mean_inf))
    np.save(f"{prefix}all_layers_fro.npy", np.array(layers_fro))
    np.save(f"{prefix}source_target_heatmap_mean.npy", source_target_mean)
    np.save(f"{prefix}source_target_heatmap_max.npy", source_target_max)
    np.save(f"{prefix}global_subspace_alignment.npy", alignment_matrix)

    # Plot panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Layer vs. Routing Magnitude & Energy
    color_inf = "darkorange"
    color_fro = "purple"
    
    axes[0].plot(layers_x, layers_mean_inf, "o-", color=color_inf, linewidth=2, label="Mean Influence")
    axes[0].fill_between(layers_x, layers_mean_inf_ci_low, layers_mean_inf_ci_high, color=color_inf, alpha=0.15)
    axes[0].set_xlabel("Source Layer Index")
    axes[0].set_ylabel("Average Downstream Influence", color=color_inf)
    axes[0].tick_params(axis='y', labelcolor=color_inf)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(layers_x, layers_fro, "x-", color=color_fro, linewidth=2, label="Frobenius Energy")
    ax0_twin.fill_between(layers_x, layers_fro_ci_low, layers_fro_ci_high, color=color_fro, alpha=0.15)
    ax0_twin.set_ylabel("Frobenius Norm (Routing Energy)", color=color_fro)
    ax0_twin.tick_params(axis='y', labelcolor=color_fro)
    
    axes[0].set_title("Routing Magnitude & Total Energy")

    # Plot 2: Layer vs. Dominance
    axes[1].plot(layers_x, layers_var_exp, "o-", color="red", linewidth=2, label="Observed")
    axes[1].plot(layers_x, layers_var_perm, "x--", color="gray", linewidth=1.5, label="Permuted")
    axes[1].fill_between(layers_x, layers_var_exp_ci_low, layers_var_exp_ci_high, color="red", alpha=0.15)
    axes[1].set_xlabel("Source Layer Index")
    axes[1].set_ylabel("s1 Variance Explained (%)")
    axes[1].set_title("s1 Variance Explained (Routing Dominance)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Plot 3: Layer vs. PR
    axes[2].plot(layers_x, layers_pr, "o-", color="steelblue", linewidth=2, label="Observed")
    axes[2].plot(layers_x, layers_pr_perm, "x--", color="gray", linewidth=1.5, label="Permuted")
    axes[2].fill_between(layers_x, layers_pr_ci_low, layers_pr_ci_high, color="steelblue", alpha=0.15)
    axes[2].set_xlabel("Source Layer Index")
    axes[2].set_ylabel("Effective Routing Dimensionality (PR)")
    axes[2].set_title("Effective Routing Dimensionality Profile")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Global Routing Geometry Analysis: {prefix.capitalize()} Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    panel_plot_name = f"{prefix}global_routing_geometry.png"
    plt.savefig(panel_plot_name, dpi=150)
    plt.close()

    # Mask for heatmaps
    mask_heatmap = np.zeros_like(source_target_mean, dtype=bool)
    for r in range(n_layers):
        for c in range(n_layers):
            if c <= r:
                mask_heatmap[r, c] = True
                
    masked_mean = np.ma.masked_where(mask_heatmap, source_target_mean)
    masked_max = np.ma.masked_where(mask_heatmap, source_target_max)

    # Plot Mean Heatmap
    plt.figure(figsize=(7.5, 6))
    im_mean = plt.imshow(masked_mean, aspect='equal', cmap='YlOrRd', interpolation='nearest')
    plt.xlabel("Target Layer Index")
    plt.ylabel("Source Layer Index")
    plt.title(f"Source-to-Target Layer Mean Influence Heatmap: {prefix.capitalize()} Model")
    plt.colorbar(im_mean, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_mean_name = f"{prefix}source_target_heatmap_mean.png"
    plt.savefig(heatmap_mean_name, dpi=150)
    plt.close()

    # Plot Max Heatmap
    plt.figure(figsize=(7.5, 6))
    im_max = plt.imshow(masked_max, aspect='equal', cmap='YlOrRd', interpolation='nearest')
    plt.xlabel("Target Layer Index")
    plt.ylabel("Source Layer Index")
    plt.title(f"Source-to-Target Layer Max Influence Heatmap: {prefix.capitalize()} Model")
    plt.colorbar(im_max, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_max_name = f"{prefix}source_target_heatmap_max.png"
    plt.savefig(heatmap_max_name, dpi=150)
    plt.close()

    # Plot 2D Subspace Alignment Heatmap
    plt.figure(figsize=(7.5, 6))
    # We display the upper triangle since alignment is symmetric
    im_align = plt.imshow(alignment_matrix, aspect='equal', cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
    plt.xlabel("Source Layer B Index")
    plt.ylabel("Source Layer A Index")
    plt.title(f"Layer-to-Layer Subspace Alignment heatmap: {prefix.capitalize()} Model")
    plt.colorbar(im_align, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_align_name = f"{prefix}subspace_alignment_matrix.png"
    plt.savefig(heatmap_align_name, dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print(f"  Saved global geometry plot -> {panel_plot_name}")
    print(f"  Saved mean heatmap -> {heatmap_mean_name}")
    print(f"  Saved max heatmap -> {heatmap_max_name}")
    print(f"  Saved subspace alignment matrix -> {heatmap_align_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
