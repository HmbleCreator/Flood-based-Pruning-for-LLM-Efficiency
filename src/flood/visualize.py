"""
src/flood/visualize.py
======================
Generates comparison plots for the FLOOD pruning perplexity benchmark.
Includes comparative sweeps (FLOOD vs. baselines) and ablation sweeps.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_perplexity_curves(results, safe_name):
    """
    Plots perplexity curves comparing:
      FLOOD vs. Magnitude vs. Attention Entropy vs. Random vs. Bridge-Only
    """
    budgets = [b * 100 for b in results["budgets"]]  # Convert to percent
    
    plt.figure(figsize=(7.5, 6))
    
    plt.plot(budgets, results["FLOOD"], 'o-', color='crimson', linewidth=2.5, markersize=7, label="FLOOD (Full Routing)")
    plt.plot(budgets, results["Bridge-Only"], 'd-.', color='darkorange', linewidth=2.0, markersize=6, label="Bridge-Only (Paper 1)")
    plt.plot(budgets, results["Magnitude"], 's--', color='darkblue', linewidth=1.5, markersize=5, label="Magnitude")
    plt.plot(budgets, results["Entropy"], '^--', color='teal', linewidth=1.5, markersize=5, label="Attention Entropy")
    plt.plot(budgets, results["Random"], 'x--', color='gray', linewidth=1.5, markersize=5, label="Random Pruning")
    
    plt.xlabel("Pruning Budget (% attention heads pruned)", fontsize=11)
    plt.ylabel("Next-token Prediction Perplexity", fontsize=11)
    plt.title(f"Pruning Benchmark Perplexity Sweep ({safe_name})\nFLOOD vs. Pruning Baselines", fontsize=12, fontweight="bold")
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    
    # Cap vertical axis to keep plot readable even if random/magnitude explodes
    all_ppl = [p for m in ["FLOOD", "Bridge-Only", "Magnitude", "Entropy", "Random"] for p in results[m] if p != float('inf')]
    if all_ppl:
        max_valid = max(results["FLOOD"] + results["Bridge-Only"] + results["Magnitude"][:4] + results["Entropy"][:4])
        plt.ylim(min(all_ppl) * 0.9, max_valid * 1.5)
        
    plt.tight_layout()
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f"{safe_name}_flood_perplexity_vs_budget.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"  Saved benchmark perplexity plot -> {out_path}")

def plot_ablation_curves(results, safe_name):
    """
    Plots perplexity curves for the FLOOD component ablation study:
      Full FLOOD vs. No-Bridge vs. No-Broadcast vs. No-Injector vs. No-Receiver
    """
    budgets = [b * 100 for b in results["budgets"]]
    
    plt.figure(figsize=(7.5, 6))
    
    plt.plot(budgets, results["FLOOD"], 'o-', color='crimson', linewidth=2.5, markersize=7, label="Full FLOOD")
    plt.plot(budgets, results["Ablation-No-Bridge"], 's--', color='purple', linewidth=1.8, markersize=5, label="No Bridge Score")
    plt.plot(budgets, results["Ablation-No-Broadcast"], 'x--', color='chocolate', linewidth=1.8, markersize=5, label="No Broadcast Centrality")
    plt.plot(budgets, results["Ablation-No-Injector"], '^--', color='darkcyan', linewidth=1.8, markersize=5, label="No Injector Alignment")
    plt.plot(budgets, results["Ablation-No-Receiver"], 'd--', color='forestgreen', linewidth=1.8, markersize=5, label="No Receiver Centrality")
    
    plt.xlabel("Pruning Budget (% attention heads pruned)", fontsize=11)
    plt.ylabel("Next-token Prediction Perplexity", fontsize=11)
    plt.title(f"FLOOD Component Ablation Study ({safe_name})\nTesting Causal Contribution of Signals", fontsize=12, fontweight="bold")
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    
    all_ppl = [p for m in ["FLOOD", "Ablation-No-Bridge", "Ablation-No-Broadcast", "Ablation-No-Injector", "Ablation-No-Receiver"] for p in results[m] if p != float('inf')]
    if all_ppl:
        max_valid = max(results["FLOOD"] + results["Ablation-No-Bridge"] + results["Ablation-No-Broadcast"][:4] + results["Ablation-No-Injector"][:4])
        plt.ylim(min(all_ppl) * 0.9, max_valid * 1.5)
        
    plt.tight_layout()
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f"{safe_name}_flood_ablation_perplexity.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"  Saved component ablation plot -> {out_path}")

def plot_recovery_curves(steps, flood_rec, mag_rec, rand_rec, safe_name):
    """Plots perplexity recovery curves across fine-tuning updates."""
    plt.figure(figsize=(7, 5.5))
    plt.plot(steps, flood_rec, 'o-', color='crimson', linewidth=2.5, markersize=7, label="FLOOD (Routing-Aware)")
    plt.plot(steps, mag_rec, 's--', color='darkblue', linewidth=1.8, markersize=5, label="Magnitude")
    plt.plot(steps, rand_rec, 'x--', color='gray', linewidth=1.8, markersize=5, label="Random")
    
    plt.xlabel("Fine-tuning Steps", fontsize=11)
    plt.ylabel("Next-token Prediction Perplexity", fontsize=11)
    plt.title(f"Post-Pruning Fine-Tuning Recovery Curve ({safe_name})\n30% Attention Heads Pruned", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    
    plt.tight_layout()
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, f"{safe_name}_flood_finetuning_recovery.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved fine-tuning recovery plot -> {out_path}")
