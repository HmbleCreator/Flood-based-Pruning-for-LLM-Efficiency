"""
src/flood/evaluate.py
=====================
Evaluates how different pruning strategies preserve the representation geometry
and dependency network topology of the remaining network.
Includes Modularity, Fiedler value, Broadcast correlation, and Hub preservation.
Also includes a test harness that asserts FLOOD outperforms Magnitude pruning.
"""

import os
import sys
import numpy as np
import torch
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer

def mask_heads_in_model(model, heads_to_prune):
    """
    Sets c_proj weights of pruned heads to 0 in-place.
    This preserves tensor shapes, allowing us to evaluate the remaining routing topology.
    """
    masked_model = copy_model_weights(model)
    d_model = masked_model.config.n_embd
    n_heads = masked_model.config.n_head
    d_head = d_model // n_heads
    
    for l, h in heads_to_prune:
        c_proj = masked_model.transformer.h[l].attn.c_proj
        # Zero out the output projection slice corresponding to head h
        with torch.no_grad():
            c_proj.weight.data[h * d_head : (h + 1) * d_head, :] = 0.0
    return masked_model

def copy_model_weights(model):
    """Safe deepcopy helper for PyTorch models to avoid configuration references."""
    cfg = model.config
    new_model = AutoModelForCausalLM.from_config(cfg)
    new_model.load_state_dict(model.state_dict())
    new_model.eval()
    if torch.cuda.is_available():
        new_model = new_model.cuda()
    return new_model

def find_attn_out_modules(model):
    attn_out_modules = {}
    for name, module in model.named_modules():
        parts = name.split('.')
        layer_idx = None
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        if layer_idx is None:
            continue
        name_lower = name.lower()
        if 'attn' in name_lower or 'attention' in name_lower:
            if any(proj in name_lower for proj in ['c_proj', 'o_proj', 'dense', 'out_proj']):
                if module.__class__.__name__ in ['Linear', 'Conv1D']:
                    attn_out_modules[layer_idx] = (name, module)
    sorted_layers = sorted(attn_out_modules.keys())
    return {i: attn_out_modules[i] for i in sorted_layers}

def compute_post_pruning_influence(model, tokenizer, texts):
    """Computes influence matrix on a subset of texts for evaluation speed."""
    layer_mapping = find_attn_out_modules(model)
    n_layers = len(layer_mapping)
    n_heads = model.config.num_attention_heads
    n_total_heads = n_layers * n_heads
    d_model = model.config.n_embd
    d_head = d_model // n_heads
    I_accum = np.zeros((n_total_heads, n_total_heads))
    
    # Simple inline capture helper
    def capture_baseline(m, tok, txt, l_map, inp):
        captures = {}
        hooks = []
        
        def make_hook(l_idx):
            def hook_fn(module, args):
                x = args[0].detach().squeeze(0)
                if x.dim() > 2:
                    x = x[0]
                for h in range(n_heads):
                    captures[(l_idx, h)] = x[:, h * d_head : (h + 1) * d_head].clone().cpu()
            return hook_fn
        for l_idx, (name, mod) in l_map.items():
            hooks.append(mod.register_forward_pre_hook(make_hook(l_idx)))
            
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            m(**inp)
        for hk in hooks:
            hk.remove()
        return captures

    def capture_ablated(m, tok, txt, l_map, ab_l, ab_h, inp):
        captures = {}
        hooks = []
        a_s, a_e = ab_h * d_head, (ab_h + 1) * d_head
        
        def make_ablation_hook():
            def ablation_hook(module, args):
                x = args[0].clone()
                x[:, :, a_s:a_e] = 0.0
                return (x,)
            return ablation_hook
        _, ablate_mod = l_map[ab_l]
        hooks.append(ablate_mod.register_forward_pre_hook(make_ablation_hook()))
        
        def make_capture_hook(l_idx):
            def hook_fn(module, args):
                x = args[0].detach().squeeze(0)
                if x.dim() > 2:
                    x = x[0]
                for h in range(n_heads):
                    captures[(l_idx, h)] = x[:, h * d_head : (h + 1) * d_head].clone().cpu()
            return hook_fn
        for l_idx, (name, mod) in l_map.items():
            if l_idx >= ab_l:
                hooks.append(mod.register_forward_pre_hook(make_capture_hook(l_idx)))
                
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
        with torch.no_grad():
            m(**inp)
        for hk in hooks:
            hk.remove()
        return captures

    for t_idx, text in enumerate(texts):
        inp = tokenizer(text, return_tensors="pt")
        baselines = capture_baseline(model, tokenizer, text, layer_mapping, inp)
        I_prompt = np.zeros((n_total_heads, n_total_heads))
        
        for l_src in range(n_layers - 1):
            for h_src in range(n_heads):
                src_global_idx = l_src * n_heads + h_src
                c_proj = model.transformer.h[l_src].attn.c_proj
                w_slice = c_proj.weight.data[h_src * d_head : (h_src + 1) * d_head, :]
                if w_slice.norm().item() < 1e-8:
                    continue
                    
                ablated = copy.deepcopy(capture_ablated(model, tokenizer, text, layer_mapping, l_src, h_src, inp))
                for l_tgt in range(l_src + 1, n_layers):
                    for h_tgt in range(n_heads):
                        tgt_global_idx = l_tgt * n_heads + h_tgt
                        try:
                            b = baselines[(l_tgt, h_tgt)]
                            a = ablated[(l_tgt, h_tgt)]
                        except KeyError as ke:
                            print(f"[Debug] KeyError looking up {(l_tgt, h_tgt)} when ablate_l={l_src}, ablate_h={h_src}")
                            print(f"[Debug] baselines keys: {list(baselines.keys())}")
                            print(f"[Debug] ablated keys: {list(ablated.keys())}")
                            raise ke
                        shift = ((b - a).norm(dim=-1) / (b.norm(dim=-1) + 1e-8)).mean().item()
                        I_prompt[src_global_idx, tgt_global_idx] = shift
        I_accum += I_prompt
        
    return I_accum / max(1, len(texts))

def spearman_rank_correlation(x, y):
    """Pure NumPy Spearman rank correlation to avoid package incompatibilities."""
    if len(x) <= 1:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    corr = np.corrcoef(rx, ry)
    if corr.shape == (2, 2):
        return float(corr[0, 1])
    return 0.0

def build_networkx_graph(I, n_total_heads):
    G = nx.DiGraph()
    for i in range(n_total_heads):
        G.add_node(i)
    for u in range(n_total_heads):
        for v in range(n_total_heads):
            if I[u, v] > 1e-8:
                G.add_edge(u, v, weight=I[u, v])
    return G

def get_broadcast_pagerank(I, n_total_heads):
    G = build_networkx_graph(I, n_total_heads)
    try:
        pr_rev = nx.pagerank(G.reverse(copy=True), weight='weight')
        return np.array([pr_rev[n] for n in range(n_total_heads)])
    except:
        return np.zeros(n_total_heads)

def run_topological_evaluation(base_model, tokenizer, flood_heads, mag_heads, rand_heads, eval_texts):
    """
    Evaluates SVD alignment, SVD PR, modularity, connectivity (Fiedler value),
    Broadcast correlation, and hub preservation for FLOOD and Magnitude.
    """
    print("\n" + "="*80)
    print("TOPOLOGY PRESERVATION EVALUATION (At 30% Pruning Budget)")
    print("="*80)
    
    n_layers = base_model.config.n_layer
    n_heads = base_model.config.num_attention_heads
    n_total_heads = n_layers * n_heads
    
    # 1. Base Model Topology
    print("Evaluating Baseline Model Topology...")
    I_base = compute_post_pruning_influence(base_model, tokenizer, eval_texts[:3])
    
    # 2. FLOOD Pruned Model Topology
    print("Evaluating FLOOD Pruned Model Topology...")
    m_flood = mask_heads_in_model(base_model, flood_heads)
    I_flood = compute_post_pruning_influence(m_flood, tokenizer, eval_texts[:3])
    del m_flood
    
    # 3. Magnitude Pruned Model Topology
    print("Evaluating Magnitude Pruned Model Topology...")
    m_mag = mask_heads_in_model(base_model, mag_heads)
    I_mag = compute_post_pruning_influence(m_mag, tokenizer, eval_texts[:3])
    del m_mag
    
    # SVD analysis
    def compute_svd_metrics(I, I_ref=None):
        if (I**2).sum() == 0:
            return 0.0, 0.0
        U, S, Vt = np.linalg.svd(I)
        pr = float((S.sum())**2 / ((S**2).sum() + 1e-12))
        
        alignment = 0.0
        if I_ref is not None:
            U_ref, _, _ = np.linalg.svd(I_ref)
            u1_ref = U_ref[:, 0]
            u1_curr = U[:, 0]
            alignment = float(abs(np.dot(u1_ref, u1_curr)))
        return pr, alignment
        
    pr_base, _ = compute_svd_metrics(I_base)
    pr_flood, align_flood = compute_svd_metrics(I_flood, I_base)
    pr_mag, align_mag = compute_svd_metrics(I_mag, I_base)
    
    # Laplacian & Modularity analysis
    def compute_graph_metrics(I):
        G = build_networkx_graph(I, n_total_heads)
        G_und = G.to_undirected()
        
        # Laplacian Fiedler value (algebraic connectivity)
        try:
            fiedler = float(nx.algebraic_connectivity(G_und, weight='weight'))
        except Exception:
            fiedler = 0.0
            
        # Louvain Modularity
        try:
            comms = nx.community.louvain_communities(G_und, weight='weight', seed=42)
            modularity = float(nx.community.modularity(G_und, comms, weight='weight'))
        except Exception:
            modularity = 0.0
            
        return fiedler, modularity
        
    fiedler_base, mod_base = compute_graph_metrics(I_base)
    fiedler_flood, mod_flood = compute_graph_metrics(I_flood)
    fiedler_mag, mod_mag = compute_graph_metrics(I_mag)
    
    # Rich-Club Preservation
    def get_average_rich_club(I):
        G = build_networkx_graph(I, n_total_heads)
        G_und = G.to_undirected()
        try:
            rc_dict = nx.rich_club_coefficient(G_und, normalized=False)
            if len(rc_dict) > 0:
                return float(np.mean(list(rc_dict.values())))
            return 0.0
        except Exception:
            return 0.0
            
    rc_base = get_average_rich_club(I_base)
    rc_flood = get_average_rich_club(I_flood)
    rc_mag = get_average_rich_club(I_mag)
    rc_ratio_flood = rc_flood / (rc_base + 1e-12)
    rc_ratio_mag = rc_mag / (rc_base + 1e-12)
    
    # Broadcast Centrality Correlation
    pr_rev_base = get_broadcast_pagerank(I_base, n_total_heads)
    pr_rev_flood = get_broadcast_pagerank(I_flood, n_total_heads)
    pr_rev_mag = get_broadcast_pagerank(I_mag, n_total_heads)
    
    corr_flood = spearman_rank_correlation(pr_rev_base, pr_rev_flood)
    corr_mag = spearman_rank_correlation(pr_rev_base, pr_rev_mag)
    
    # Hub Preservation (overlap of top 10 broadcaster hubs)
    top_k_base = set(np.argsort(pr_rev_base)[::-1][:10])
    top_k_flood = set(np.argsort(pr_rev_flood)[::-1][:10])
    top_k_mag = set(np.argsort(pr_rev_mag)[::-1][:10])
    
    overlap_flood = len(top_k_base.intersection(top_k_flood)) / 10.0
    overlap_mag = len(top_k_base.intersection(top_k_mag)) / 10.0
    
    print("\nEvaluation Summary:")
    print(f"  - Baseline:  PR={pr_base:.4f} | Fiedler={fiedler_base:.4f} | Modularity={mod_base:.4f} | Rich-Club={rc_base:.4f}")
    print(f"  - FLOOD:     PR={pr_flood:.4f} | Fiedler={fiedler_flood:.4f} | Modularity={mod_flood:.4f} | Subspace Align={align_flood:.4f} | BC Corr={corr_flood:.4f} | Hub Overlap={overlap_flood:.1%} | Rich-Club Pres={rc_ratio_flood:.1%}")
    print(f"  - Magnitude: PR={pr_mag:.4f} | Fiedler={fiedler_mag:.4f} | Modularity={mod_mag:.4f} | Subspace Align={align_mag:.4f} | BC Corr={corr_mag:.4f} | Hub Overlap={overlap_mag:.1%} | Rich-Club Pres={rc_ratio_mag:.1%}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    metrics = ['Subspace Align', 'SVD PR Ratio', 'BC Rank Corr', 'Hub Overlap', 'Rich-Club Pres']
    flood_vals = [align_flood, pr_flood / pr_base, corr_flood, overlap_flood, rc_ratio_flood]
    mag_vals = [align_mag, pr_mag / pr_base, corr_mag, overlap_mag, rc_ratio_mag]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, flood_vals, width, color='crimson', label='FLOOD')
    plt.bar(x + width/2, mag_vals, width, color='navy', alpha=0.7, label='Magnitude')
    
    plt.ylabel('Score Value / Ratio', fontsize=11)
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.title("Routing Topology Preservation Metrics (30% Pruned)", fontsize=12, fontweight="bold")
    plt.legend(loc='lower left', shadow=True)
    
    plt.tight_layout()
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "gpt2_flood_topology_preservation.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved topology preservation plot -> {out_path}")
    print("="*80)
    
    return {
        "pr_base": pr_base, "pr_flood": pr_flood, "pr_mag": pr_mag,
        "align_flood": align_flood, "align_mag": align_mag,
        "fiedler_base": fiedler_base, "fiedler_flood": fiedler_flood, "fiedler_mag": fiedler_mag,
        "corr_flood": corr_flood, "corr_mag": corr_mag,
        "overlap_flood": overlap_flood, "overlap_mag": overlap_mag,
        "rc_ratio_flood": rc_ratio_flood, "rc_ratio_mag": rc_ratio_mag
    }

if __name__ == "__main__":
    # Regression test suite
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from score import compute_routing_importance_scores
    from benchmark import load_base_model, EVAL_TEXTS, compute_magnitude_scores
    
    print("Running evaluate.py self-contained regression tests...")
    model, tokenizer = load_base_model("gpt2")
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    n_total_heads = n_layers * n_heads
    
    flood_scores = compute_routing_importance_scores("gpt2")
    flood_ordered = sorted(flood_scores.keys(), key=lambda x: flood_scores[x])
    mag_scores = compute_magnitude_scores(model, n_layers, n_heads)
    mag_ordered = sorted(mag_scores.keys(), key=lambda x: mag_scores[x])
    
    n_prune_30 = int(n_total_heads * 0.30)
    
    res = run_topological_evaluation(
        model, tokenizer,
        flood_ordered[:n_prune_30],
        mag_ordered[:n_prune_30],
        None,
        EVAL_TEXTS
    )
    
    # Assertions
    print("\nVerifying regression guard assertions:")
    print(f"  Assertion: FLOOD PR ({res['pr_flood']:.4f}) >= Magnitude PR ({res['pr_mag']:.4f})")
    assert res['pr_flood'] >= res['pr_mag'], "Pruning Regression: Magnitude has better SVD Participation Ratio!"
    
    print(f"  Assertion: FLOOD Align ({res['align_flood']:.4f}) >= Magnitude Align ({res['align_mag']:.4f})")
    assert res['align_flood'] >= res['align_mag'], "Pruning Regression: Magnitude has better Subspace Alignment!"
    
    print(f"  Assertion: FLOOD Hub Overlap ({res['overlap_flood']:.1%}) >= Magnitude Hub Overlap ({res['overlap_mag']:.1%})")
    assert res['overlap_flood'] >= res['overlap_mag'], "Pruning Regression: Magnitude has better Hub Preservation!"
    
    print(f"  Assertion: FLOOD Rich-Club Pres ({res['rc_ratio_flood']:.1%}) >= Magnitude Rich-Club Pres ({res['rc_ratio_mag']:.1%})")
    assert res['rc_ratio_flood'] >= res['rc_ratio_mag'], "Pruning Regression: Magnitude has better Rich-Club Preservation!"
    
    print("\nAll regression tests passed successfully!")
