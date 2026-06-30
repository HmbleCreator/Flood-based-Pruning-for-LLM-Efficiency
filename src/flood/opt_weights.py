"""
src/flood/opt_weights.py
========================
Runs a multi-objective simplex optimization sweep over the weight hyperparameters of FLOOD:
  S(h) = w_bridge*Bridge + w_broadcast*Broadcast + w_injector*Injector + w_receiver*Receiver + w_backbone*Backbone

Objectives:
  1. Minimize Next-Token Perplexity (Capability)
  2. Maximize Topology Preservation (Structure: Align, PR, BC Corr, Hub Overlap, Rich-Club Pres)

Finds the Pareto frontier, rank stability correlation, and saves the optimal weights.
Optimized to run topology metrics in pure numpy/NetworkX using the cached influence matrix.
"""

import sys
import os
import numpy as np
import scipy.stats as stats
import torch
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_base_model, prune_attention_heads, calculate_perplexity, EVAL_TEXTS, compute_magnitude_scores
from score import compute_routing_importance_scores
from layout import get_model_layout

def generate_simplex_samples(n_samples=25, seed=42):
    """Generates random samples on the 5-simplex (summing to 1.0)."""
    np.random.seed(seed)
    samples = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0], size=n_samples)
    return samples

def is_pareto_efficient(costs):
    """
    Finds the pareto-efficient points.
    costs: (n_points, n_objectives) array where we want to MINIMIZE all objectives.
    Returns boolean array of shape (n_points,) indicating Pareto efficiency.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

def compute_numpy_topology_metrics(I_base, pruned_heads, mag_pruned, n_total_heads, n_layers, n_heads):
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    
    pruned_indices = [l * n_heads + h for (l, h) in pruned_heads]
    
    # Mask influence matrix for FLOOD
    I_flood = I_base.copy()
    for idx in pruned_indices:
        I_flood[idx, :] = 0.0
        I_flood[:, idx] = 0.0
        
    # SVD Participation Ratio
    U_base, S_base, Vt_base = np.linalg.svd(I_base)
    pr_base = float((S_base.sum())**2 / ((S_base**2).sum() + 1e-12))
    
    U_flood, S_flood, Vt_flood = np.linalg.svd(I_flood)
    pr_flood = float((S_flood.sum())**2 / ((S_flood**2).sum() + 1e-12))
    
    # Subspace Alignment (u_1)
    align_flood = float(abs(np.dot(U_base[:, 0], U_flood[:, 0])))
    
    def get_graph_metrics(I):
        G = nx.DiGraph()
        for i in range(n_total_heads):
            G.add_node(head_labels[i])
        for u in range(n_total_heads):
            for v in range(n_total_heads):
                if I[u, v] > 1e-8:
                    G.add_edge(head_labels[u], head_labels[v], weight=I[u, v])
        G_und = G.to_undirected()
        
        # Fiedler connectivity of giant component
        try:
            G_active = G_und.copy()
            G_active.remove_nodes_from(list(nx.isolates(G_active)))
            if len(G_active) > 0:
                components = sorted(nx.connected_components(G_active), key=len, reverse=True)
                largest_cc = G_active.subgraph(components[0]).copy()
                if len(largest_cc) >= 2:
                    fiedler = float(nx.algebraic_connectivity(largest_cc, weight='weight'))
                else:
                    fiedler = 0.0
            else:
                fiedler = 0.0
        except Exception:
            fiedler = 0.0
            
        # Louvain Modularity
        try:
            comms = nx.community.louvain_communities(G_und, weight='weight', seed=42)
            modularity = float(nx.community.modularity(G_und, comms, weight='weight'))
        except Exception:
            modularity = 0.0
            
        # Rich-Club coefficient
        try:
            rc_dict = nx.rich_club_coefficient(G_und, normalized=False)
            rc_avg = float(np.mean(list(rc_dict.values()))) if len(rc_dict) > 0 else 0.0
        except Exception:
            rc_avg = 0.0
            
        # Broadcast Centrality (Reverse PageRank)
        G_rev = G.reverse(copy=True)
        try:
            pr_rev_dict = nx.pagerank(G_rev, weight='weight')
            pr_rev_arr = np.array([pr_rev_dict.get(n, 0.0) for n in head_labels])
        except Exception:
            pr_rev_arr = np.zeros(n_total_heads)
            
        return fiedler, modularity, rc_avg, pr_rev_arr
        
    fiedler_base, mod_base, rc_base, pr_rev_base = get_graph_metrics(I_base)
    fiedler_flood, mod_flood, rc_flood, pr_rev_flood = get_graph_metrics(I_flood)
    
    # Broadcast Correlation (Spearman)
    try:
        corr_flood = float(stats.spearmanr(pr_rev_base, pr_rev_flood)[0])
        if np.isnan(corr_flood):
            corr_flood = 0.0
    except Exception:
        corr_flood = 0.0
        
    # Top 10 Hub Overlap
    top_k_base = set(np.argsort(pr_rev_base)[::-1][:10])
    top_k_flood = set(np.argsort(pr_rev_flood)[::-1][:10])
    overlap_flood = len(top_k_base.intersection(top_k_flood)) / 10.0
    
    # Rich-Club preservation ratio
    rc_ratio_flood = rc_flood / (rc_base + 1e-12)
    
    return {
        "align_flood": align_flood,
        "pr_ratio_flood": pr_flood / (pr_base + 1e-12),
        "corr_flood": corr_flood,
        "overlap_flood": overlap_flood,
        "rc_ratio_flood": rc_ratio_flood
    }

def main():
    print("Loading model & tokenizer for multi-objective simplex search...")
    model, tokenizer = load_base_model("gpt2")
    
    layout = get_model_layout("gpt2")
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    n_total_heads = n_layers * n_heads
    n_prune_30 = int(n_total_heads * 0.30)
    
    I_base = np.load("gpt2_dependency_matrix.npy")
    
    eval_subset = EVAL_TEXTS[:10]
    base_ppl = calculate_perplexity(model, tokenizer, eval_subset)
    print(f"Baseline Perplexity: {base_ppl:.4f}")
    
    # Generate weight configurations
    configs = [
        [0.3, 0.3, 0.2, 0.1, 0.1],  # Default V1
        [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal
        [1.0, 0.0, 0.0, 0.0, 0.0],  # Bridge only
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Broadcast only
    ]
    random_configs = generate_simplex_samples(26, seed=42)
    for c in random_configs:
        configs.append(c.tolist())
        
    mag_scores = compute_magnitude_scores(model, n_layers, n_heads)
    mag_ordered = sorted(mag_scores.keys(), key=lambda x: mag_scores[x])
    
    print(f"Starting multi-objective sweep across {len(configs)} configurations...")
    results = []
    
    for idx, w in enumerate(configs):
        w_bridge, w_broadcast, w_injector, w_receiver, w_backbone = w
        try:
            scores = compute_routing_importance_scores(
                "gpt2",
                w_bridge=w_bridge,
                w_broadcast=w_broadcast,
                w_injector=w_injector,
                w_receiver=w_receiver,
                w_backbone=w_backbone
            )
            # Sort heads (lowest score = pruned first)
            ordered_heads = sorted(scores.keys(), key=lambda x: scores[x])
            pruned_heads = ordered_heads[:n_prune_30]
            
            # 1. Capability: Perplexity (requires model prune evaluation)
            m_pruned = prune_attention_heads(model, pruned_heads)
            ppl = calculate_perplexity(m_pruned, tokenizer, eval_subset)
            del m_pruned
            
            # 2. Structure: Topological Preservation (fast numpy/NetworkX)
            res_topo = compute_numpy_topology_metrics(
                I_base,
                pruned_heads,
                mag_ordered[:n_prune_30],
                n_total_heads,
                n_layers,
                n_heads
            )
            
            topo_pres = (
                res_topo["align_flood"] +
                res_topo["pr_ratio_flood"] +
                res_topo["corr_flood"] +
                res_topo["overlap_flood"] +
                res_topo["rc_ratio_flood"]
            ) / 5.0
            
            # Objectives to minimize
            cap_cost = ppl / base_ppl
            struct_cost = 1.0 - topo_pres
            
            joint_score = 0.5 * cap_cost + 0.5 * struct_cost
            
            print(f"Config {idx+1:02d} -> w=[{w_bridge:.2f}, {w_broadcast:.2f}, {w_injector:.2f}, {w_receiver:.2f}, {w_backbone:.2f}] | PPL: {ppl:.2f} | TopoPres: {topo_pres:.2%} | J = {joint_score:.4f}")
            results.append({
                "w": w,
                "ppl": ppl,
                "topo_pres": topo_pres,
                "cap_cost": cap_cost,
                "struct_cost": struct_cost,
                "joint_score": joint_score,
                "scores_dict": scores
            })
        except Exception as e:
            print(f"Config {idx+1:02d} failed: {e}")
            
    # Calculate Pareto frontier
    costs = np.array([[r["cap_cost"], r["struct_cost"]] for r in results])
    pareto_indices = is_pareto_efficient(costs)
    
    # Sort results by joint score
    results.sort(key=lambda x: x["joint_score"])
    best_config = results[0]
    
    print("\n" + "="*80)
    print("TOP WEIGHT CONFIGURATIONS (Sorted by Joint Cost J)")
    print("="*80)
    for rank, r in enumerate(results[:5]):
        w = r["w"]
        print(f"  Rank {rank+1:02d} | J: {r['joint_score']:.4f} | PPL: {r['ppl']:.2f} | TopoPres: {r['topo_pres']:.2%} | w: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}, {w[4]:.3f}]")
        
    print("\n" + "="*80)
    print("PARETO-OPTIMAL CONFIGURATIONS")
    print("="*80)
    pareto_count = 0
    for idx, r in enumerate(results):
        if pareto_indices[results.index(r)]:
            w = r["w"]
            print(f"  Pareto {pareto_count+1:02d} | J: {r['joint_score']:.4f} | PPL: {r['ppl']:.2f} | TopoPres: {r['topo_pres']:.2%} | w: [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}, {w[3]:.3f}, {w[4]:.3f}]")
            pareto_count += 1
            
    # Calculate Spearman Rank Stability Correlation between Default and Best config FLOOD scores
    default_scores = compute_routing_importance_scores("gpt2")
    best_scores = best_config["scores_dict"]
    
    head_labels = sorted(default_scores.keys())
    default_vals = np.array([default_scores[h] for h in head_labels])
    best_vals = np.array([best_scores[h] for h in head_labels])
    
    rank_corr, _ = stats.spearmanr(default_vals, best_vals)
    print("\n" + "="*80)
    print("RANK STABILITY ANALSIS")
    print("="*80)
    print(f"  Spearman rho between Default and Opt weight FLOOD scores: {rank_corr:.4f}")
    if rank_corr >= 0.90:
        print("  Status: FLOOD score ranking is highly robust to weight perturbations!")
    else:
        print("  Status: FLOOD score ranking is sensitive to weights.")
    print("="*80)

if __name__ == "__main__":
    main()
