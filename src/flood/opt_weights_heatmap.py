"""
src/flood/opt_weights_heatmap.py
================================
Performs a 2D grid sweep over w_bridge and w_broadcast weights
and generates a perplexity sensitivity heatmap at 30% pruning budget.
Uses an evaluation cache to prevent redundant forward passes.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_base_model, calculate_perplexity, EVAL_TEXTS
from score import compute_routing_importance_scores
from prune import prune_attention_heads

def run_grid_sweep(model_name="gpt2"):
    print(f"Loading {model_name} for weight sensitivity grid sweep...")
    model, tokenizer = load_base_model(model_name)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n_total_heads = n_layers * n_heads
    n_to_prune = int(n_total_heads * 0.30)
    
    # 11 x 11 grid from 0.0 to 1.0
    w_vals = np.linspace(0.0, 1.0, 11)
    grid_ppl = np.zeros((11, 11))
    
    # Cache mapping pruned heads tuple to evaluated perplexity
    eval_cache = {}
    
    print("Running 2D weight sensitivity sweep (w_bridge vs. w_broadcast)...")
    for i, w_br in enumerate(w_vals):
        for j, w_bc in enumerate(w_vals):
            # Compute score ranking with absolute weights (no normalization inside loop)
            scores = compute_routing_importance_scores(
                model_name, w_bridge=w_br, w_broadcast=w_bc, w_injector=0.0, w_receiver=0.0, w_backbone=0.0
            )
            # Prune lowest-scoring 30% of heads
            ordered = sorted(scores.keys(), key=lambda x: scores[x])
            pruned_subset = tuple(ordered[:n_to_prune])
            
            if pruned_subset in eval_cache:
                ppl = eval_cache[pruned_subset]
            else:
                m_pruned = prune_attention_heads(model, ordered[:n_to_prune])
                ppl = calculate_perplexity(m_pruned, tokenizer, EVAL_TEXTS)
                eval_cache[pruned_subset] = ppl
                del m_pruned
                
            grid_ppl[i, j] = ppl
            print(f"  w_bridge={w_br:.2f}, w_broadcast={w_bc:.2f} -> PPL: {ppl:.4f}")
            
    # Save sweep grid results
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plotting Heatmap
    plt.figure(figsize=(8.5, 7))
    # We display origin='lower': Y-axis is w_bridge (index i), X-axis is w_broadcast (index j)
    im = plt.imshow(grid_ppl, origin='lower', extent=[0.0, 1.0, 0.0, 1.0], cmap='YlOrRd_r', aspect='auto')
    
    # Add perplexity values inside the cells for readability
    for i in range(11):
        for j in range(11):
            val = grid_ppl[i, j]
            # Use black text for light cells, white for dark cells
            color = "white" if val > np.percentile(grid_ppl, 60) else "black"
            plt.text(j/10.0, i/10.0, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
            
    plt.colorbar(im, label="Next-token Prediction Perplexity")
    plt.xlabel("w_broadcast (Routing Importance Weight)", fontsize=11)
    plt.ylabel("w_bridge (Causal Damage Weight)", fontsize=11)
    plt.title(f"FLOOD Weight Sensitivity Analysis ({model_name})\nNext-token Perplexity at 30% Head Pruning Budget", fontsize=12, fontweight="bold")
    
    out_path = os.path.join(fig_dir, f"{model_name}_flood_weight_sensitivity_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"\nSaved weight sensitivity heatmap -> {out_path}")

if __name__ == "__main__":
    run_grid_sweep("gpt2")
