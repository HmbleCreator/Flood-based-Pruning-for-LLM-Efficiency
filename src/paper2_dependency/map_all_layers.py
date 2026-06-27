"""
Map All Layers Routing Geometry (Refined)
=========================================
Computes the downstream routing geometry (PR, average influence, s1) 
for EVERY layer in the model to determine if:
  1. The low-dimensional routing geometry is a global architectural property.
  2. The bridge layers act as dominant peaks of information flow (s1 and influence magnitude).

Refinements implemented:
  - Normalized s1 (Dominance s1/sum(s) & Variance Explained s1^2/sum(s^2))
  - Bootstrap uncertainty (95% CI bands for all layer-wise curves)
  - 2D Heatmap of Source Layer vs. Target Layer average influence
  - Repeated permutation baseline comparison across all layers
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

def _capture_head_outputs_ablated(model, tokenizer, text, ablate_layer, ablate_head, max_length=64):
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

    hk = model.transformer.h[ablate_layer].attn.c_proj.register_forward_pre_hook(
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

# ── Geometry Metrics ──

def get_pr(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    return float((S.sum() ** 2) / ((S ** 2).sum() + 1e-12))

def get_s1(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    return float(S[0]) if len(S) > 0 else 0.0

def get_variance_explained(mat):
    _, S, _ = np.linalg.svd(mat, full_matrices=False)
    if len(S) == 0 or (S**2).sum() == 0:
        return 0.0
    return float((S[0] ** 2) / (S ** 2).sum() * 100)

def bootstrap_layer_metrics(I_by_text_l, num_bootstraps=200):
    n_texts = I_by_text_l.shape[0]
    pr_vals = []
    var_vals = []
    mean_vals = []
    
    for _ in range(num_bootstraps):
        idx = np.random.choice(n_texts, size=n_texts, replace=True)
        mat = I_by_text_l[idx].mean(axis=0)
        
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        if len(S) > 0:
            pr = (S.sum() ** 2) / ((S ** 2).sum() + 1e-12)
            var_exp = (S[0] ** 2) / ((S ** 2).sum() + 1e-12) * 100
        else:
            pr, var_exp = 0.0, 0.0
            
        pr_vals.append(pr)
        var_vals.append(var_exp)
        mean_vals.append(mat.mean())
        
    ci = lambda vals: (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    return ci(pr_vals), ci(var_vals), ci(mean_vals)

def compute_permutation_baselines(I_mat, num_perms=100):
    pr_vals = []
    var_vals = []
    flat = I_mat.copy().ravel()
    for _ in range(num_perms):
        np.random.shuffle(flat)
        sh_mat = flat.reshape(I_mat.shape)
        _, S, _ = np.linalg.svd(sh_mat, full_matrices=False)
        if len(S) > 0:
            pr = (S.sum() ** 2) / ((S ** 2).sum() + 1e-12)
            var_exp = (S[0] ** 2) / ((S ** 2).sum() + 1e-12) * 100
        else:
            pr, var_exp = 0.0, 0.0
        pr_vals.append(pr)
        var_vals.append(var_exp)
    return float(np.mean(pr_vals)), float(np.mean(var_vals))

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2: Map downstream routing geometry for all layers (Refined)"
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
    
    layers_pr_perm = []
    layers_var_perm = []

    layers_x = list(range(n_layers - 1))  # Last layer has no downstream targets
    
    # 2D Heatmap: Source Layer vs. Target Layer Influence
    source_target_influence = np.zeros((n_layers, n_layers))

    print(f"\n  Running analysis for layers 0 to {n_layers - 2}...")
    
    for l in layers_x:
        source_heads = [(l, h) for h in range(n_heads)]
        all_targets = [(tgt_l, tgt_h) for tgt_l in range(l + 1, n_layers) for tgt_h in range(n_heads)]
        
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
        
        # 1. Store Source-Target Layer Heatmap values (distance analysis)
        for tgt_l in range(l + 1, n_layers):
            tgt_indices = [idx for idx, (tl, th) in enumerate(all_targets) if tl == tgt_l]
            source_target_influence[l, tgt_l] = I_by_text[:, :, tgt_indices].mean()

        # 2. Point Estimates
        pr = get_pr(I_mat)
        var_exp = get_variance_explained(I_mat)
        mean_inf = I_mat.mean()
        
        # 3. Bootstrap CIs
        ci_pr, ci_var, ci_mean = bootstrap_layer_metrics(I_by_text, num_bootstraps=200)
        
        # 4. Permutation Baselines
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
        
        layers_pr_perm.append(perm_pr)
        layers_var_perm.append(perm_var)
        
        print(f"    Layer {l:02d}/{n_layers - 2} | "
              f"Mean Inf: {mean_inf:.4f} [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}] | "
              f"PR: {pr:.2f} [{ci_pr[0]:.2f}, {ci_pr[1]:.2f} (perm: {perm_pr:.2f})] | "
              f"s1 Var Explained: {var_exp:.1f}% [{ci_var[0]:.1f}%, {ci_var[1]:.1f}% (perm: {perm_var:.1f}%)]")

    # Save data arrays
    np.save(f"{prefix}all_layers_pr.npy", np.array(layers_pr))
    np.save(f"{prefix}all_layers_var_exp.npy", np.array(layers_var_exp))
    np.save(f"{prefix}all_layers_mean_inf.npy", np.array(layers_mean_inf))
    np.save(f"{prefix}source_target_heatmap.npy", source_target_influence)

    # Plot panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Layer vs. Mean Downstream Influence
    axes[0].plot(layers_x, layers_mean_inf, "o-", color="darkorange", linewidth=2, label="Observed")
    axes[0].fill_between(layers_x, layers_mean_inf_ci_low, layers_mean_inf_ci_high, color="darkorange", alpha=0.15)
    axes[0].set_xlabel("Source Layer Index")
    axes[0].set_ylabel("Average Downstream Influence")
    axes[0].set_title("Layer-wise Downstream Influence Magnitude")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Layer vs. Dominance (% Variance Explained by s1)
    axes[1].plot(layers_x, layers_var_exp, "o-", color="red", linewidth=2, label="Observed")
    axes[1].plot(layers_x, layers_var_perm, "x--", color="gray", linewidth=1.5, label="Permuted")
    axes[1].fill_between(layers_x, layers_var_exp_ci_low, layers_var_exp_ci_high, color="red", alpha=0.15)
    axes[1].set_xlabel("Source Layer Index")
    axes[1].set_ylabel("s1 Variance Explained (%)")
    axes[1].set_title("s1 Variance Explained (Dimensionless Routing Strength)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Plot 3: Layer vs. PR (Effective Routing Dimensionality)
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

    # Plot 2D Source Layer vs. Target Layer Heatmap
    plt.figure(figsize=(7.5, 6))
    # Mask out values below or on diagonal since influence flows only downstream
    mask_heatmap = np.zeros_like(source_target_influence, dtype=bool)
    for r in range(n_layers):
        for c in range(n_layers):
            if c <= r:
                mask_heatmap[r, c] = True
    masked_influence = np.ma.masked_where(mask_heatmap, source_target_influence)
    
    im = plt.imshow(masked_influence, aspect='equal', cmap='YlOrRd', interpolation='nearest')
    plt.xlabel("Target Layer Index")
    plt.ylabel("Source Layer Index")
    plt.title(f"Source-to-Target Layer Average Influence Heatmap: {prefix.capitalize()} Model")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    heatmap_plot_name = f"{prefix}source_target_heatmap.png"
    plt.savefig(heatmap_plot_name, dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print(f"  Saved global geometry plot -> {panel_plot_name}")
    print(f"  Saved source-target layer heatmap -> {heatmap_plot_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
