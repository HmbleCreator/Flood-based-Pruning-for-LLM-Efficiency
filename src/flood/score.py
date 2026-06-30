"""
src/flood/score.py
===================
Loads and normalizes various routing signals from Stage 2 cached files
to compute the combined Routing Importance Score S(h).
Supports custom weights and robust percentile-rank normalization.
"""

import os
import numpy as np

def percentile_rank(arr):
    """Normalize array elements to their rank-percentiles in [0, 1]."""
    if len(arr) <= 1:
        return np.zeros_like(arr)
    # Ranks range from 0 to N-1
    ranks = np.argsort(np.argsort(arr)).astype(float)
    return ranks / (len(arr) - 1.0)

def compute_routing_importance_scores(
    model_name="gpt2",
    w_bridge=0.50,
    w_broadcast=0.30,
    w_injector=0.10,
    w_receiver=0.00,
    w_backbone=0.10
):
    """
    Computes a combined Routing Importance Score S(h) for all heads using
    robust percentile-rank normalized routing signals:
      S(h) = w1*Bridge(h) + w2*Broadcast(h) + w3*Injector(h) + w4*Receiver(h) + w5*Backbone(h)
    """
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    prefix = f"{safe_name}_"
    
    # Load cached arrays
    causal_damage_path = f"{prefix}causal_damage.npy"
    pagerank_rev_path = f"{prefix}pagerank_rev.npy"
    injector_path = f"{prefix}injector_scores.npy"
    pagerank_path = f"{prefix}pagerank.npy"
    
    # Check exists
    for path in [causal_damage_path, pagerank_rev_path, injector_path, pagerank_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required cached file {path} not found. Run Stage 2 extraction first.")
            
    causal_damage = np.load(causal_damage_path)
    pagerank_rev_dict = np.load(pagerank_rev_path, allow_pickle=True).item()
    injector_scores = np.load(injector_path)
    pagerank_dict = np.load(pagerank_path, allow_pickle=True).item()
    
    from layout import get_model_layout
    layout = get_model_layout(len(causal_damage))
    n_layers = layout["layers"]
    n_heads = layout["heads"]
        
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    
    # Align structures to standard array order
    pr_rev = np.array([pagerank_rev_dict[n] for n in head_labels])
    pr_fwd = np.array([pagerank_dict[n] for n in head_labels])
    
    # Normalize components using robust percentile ranking
    norm_bridge = percentile_rank(causal_damage)
    norm_broadcast = percentile_rank(pr_rev)
    norm_injector = percentile_rank(injector_scores)
    norm_receiver = percentile_rank(pr_fwd)
    norm_backbone = percentile_rank(pr_rev)  # Backbone is percentile rank of Broadcast centrality
    
    # Combine scores
    S_arr = (
        w_bridge * norm_bridge +
        w_broadcast * norm_broadcast +
        w_injector * norm_injector +
        w_receiver * norm_receiver +
        w_backbone * norm_backbone
    )
    
    # Map back to head label dict and (l, h) dict
    scores_dict = {}
    for idx, name in enumerate(head_labels):
        l = idx // n_heads
        h = idx % n_heads
        scores_dict[(l, h)] = S_arr[idx]
        
    return scores_dict
