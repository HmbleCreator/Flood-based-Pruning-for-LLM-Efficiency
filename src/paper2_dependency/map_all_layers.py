"""
Map All Layers Routing Geometry
===============================
Computes the downstream routing geometry (PR, average influence, s1) 
for EVERY layer in the model to determine if:
  1. The low-dimensional routing geometry is a global architectural property.
  2. The bridge layers act as dominant peaks of information flow (s1 and influence magnitude).
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

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2: Map downstream routing geometry for all layers"
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
    layers_s1 = []
    layers_mean_influence = []
    layers_x = list(range(n_layers - 1))  # Last layer has no downstream targets

    print(f"\n  Running analysis for layers 0 to {n_layers - 2}...")
    
    for l in layers_x:
        source_heads = [(l, h) for h in range(n_heads)]
        all_targets = [(tgt_l, tgt_h) for tgt_l in range(l + 1, n_layers) for tgt_h in range(n_heads)]
        
        # Matrix size: (n_heads, len(all_targets))
        I_mat = np.zeros((n_heads, len(all_targets)))
        
        # Temporary storage for averaging over texts
        I_by_text = np.zeros((len(all_texts), n_heads, len(all_targets)))
        target_idx = {t: idx for idx, t in enumerate(all_targets)}

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
        
        # Calculate metrics for layer l
        pr = get_pr(I_mat)
        s1 = get_s1(I_mat)
        mean_inf = I_mat.mean()
        
        layers_pr.append(pr)
        layers_s1.append(s1)
        layers_mean_influence.append(mean_inf)
        
        print(f"    Layer {l:02d}/{n_layers - 2} | Mean Inf: {mean_inf:.4f} | PR: {pr:.2f} | s1: {s1:.4f}")

    # Save data arrays
    np.save(f"{prefix}all_layers_pr.npy", np.array(layers_pr))
    np.save(f"{prefix}all_layers_s1.npy", np.array(layers_s1))
    np.save(f"{prefix}all_layers_mean_inf.npy", np.array(layers_mean_influence))

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Layer vs Mean Influence
    axes[0].plot(layers_x, layers_mean_influence, "o-", color="darkorange", linewidth=2)
    axes[0].set_xlabel("Source Layer Index")
    axes[0].set_ylabel("Average Downstream Influence")
    axes[0].set_title("Layer-wise Average Downstream Influence")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Layer vs s1 (Primary Routing Strength)
    axes[1].plot(layers_x, layers_s1, "o-", color="red", linewidth=2)
    axes[1].set_xlabel("Source Layer Index")
    axes[1].set_ylabel("First Singular Value (s1)")
    axes[1].set_title("Layer-wise Primary Routing Strength (s1)")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Plot 3: Layer vs PR (Effective Routing Dimensionality)
    axes[2].plot(layers_x, layers_pr, "o-", color="steelblue", linewidth=2)
    axes[2].set_xlabel("Source Layer Index")
    axes[2].set_ylabel("Effective Routing Dimensionality (PR)")
    axes[2].set_title("Layer-wise Effective Routing Dimensionality")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle(f"Global Routing Geometry Analysis: {prefix.capitalize()} Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_name = f"{prefix}global_routing_geometry.png"
    plt.savefig(plot_name, dpi=150)
    plt.close()
    
    print("\n" + "=" * 70)
    print(f"  Saved global geometry plot -> {plot_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
