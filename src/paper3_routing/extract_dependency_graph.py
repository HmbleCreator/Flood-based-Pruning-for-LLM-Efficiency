"""
Paper 3 — Stage 2: Dependency Graph Extraction Pipeline (Revised V4)
======================================================================
Computes the complete pairwise head-to-head dependency network
across all layers of a transformer model.

Refinements in V4:
  1. Compute PageRank on both the forward and reversed graph to identify receivers vs. broadcasters.
  2. Inspect the largest magnitude entries of the Fiedler vector.
  3. Save the full dependency graph in GraphML format for downstream tools.
  4. Perform Edge Betweenness Centrality analysis to identify flow bottlenecks.
  5. Use top 2 outgoing edges per node for the network topology visualization.
  6. Compute Spearman, Pearson, and Kendall tau correlations with causal damage.
  7. Symmetrized Average vs. Max Louvain community detection (with pure ARI comparison).
  8. Symmetric/asymmetric decomposition index with architectural causal ordering caveat.
  9. Bootstrap Uncertainty: 200 resamples to report 95% CIs on PageRank (fwd/rev), correlations, and stability.
"""

import argparse
import os
import pickle
import numpy as np
import scipy.stats as stats
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer

# 15 Domain probes (same as Paper 2)
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

def load_model(model_name="gpt2", path=None):
    src = path
    if src is None:
        if "pythia" in model_name.lower():
            # Pythia models are not stored locally — load from HuggingFace cache
            src = model_name
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_name = "gpt2_medium_local" if "medium" in model_name else "gpt2_local"
            for possible_path in [
                os.path.join(script_dir, "..", "..", local_name),
                os.path.join(script_dir, "..", local_name),
                os.path.join(script_dir, local_name),
                os.path.join("c:/Users/amiku/Downloads/NewArch", local_name)
            ]:
                if os.path.exists(possible_path):
                    src = possible_path
                    break
            if src is None or not os.path.exists(src):
                src = model_name
            
    print(f"Loading model and tokenizer from: {src} ...")
    is_local = not ("pythia" in model_name.lower())
    tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True, local_files_only=is_local
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"  Loaded model successfully. Device: {model.device}")
    return model, tokenizer

def capture_baseline_activations_and_loss(model, tokenizer, text, layer_mapping, inp):
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads
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

    for l_idx, (name, mod) in layer_mapping.items():
        hooks.append(mod.register_forward_pre_hook(make_hook(l_idx)))

    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        outputs = model(**inp, labels=inp["input_ids"])
        loss_val = outputs.loss.item()

    for hk in hooks:
        hk.remove()

    return captures, loss_val

def capture_ablated_activations_and_loss(model, tokenizer, text, layer_mapping, ablate_l, ablate_h, inp):
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    d_head = d_model // n_heads
    captures = {}
    hooks = []

    a_s, a_e = ablate_h * d_head, (ablate_h + 1) * d_head
    def make_ablation_hook():
        def ablation_hook(module, args):
            x = args[0].clone()
            x[:, :, a_s:a_e] = 0.0
            return (x,)
        return ablation_hook

    _, ablate_mod = layer_mapping[ablate_l]
    hooks.append(ablate_mod.register_forward_pre_hook(make_ablation_hook()))

    def make_capture_hook(l_idx):
        def hook_fn(module, args):
            x = args[0].detach().squeeze(0)
            if x.dim() > 2:
                x = x[0]
            for h in range(n_heads):
                captures[(l_idx, h)] = x[:, h * d_head : (h + 1) * d_head].clone().cpu()
        return hook_fn

    for l_idx, (name, mod) in layer_mapping.items():
        if l_idx >= ablate_l:
            hooks.append(mod.register_forward_pre_hook(make_capture_hook(l_idx)))

    if torch.cuda.is_available():
        inp = {k: v.cuda() for k, v in inp.items()}
    with torch.no_grad():
        outputs = model(**inp, labels=inp["input_ids"])
        loss_val = outputs.loss.item()

    for hk in hooks:
        hk.remove()

    return captures, loss_val

def compute_prompt_influence_and_damage(model, tokenizer, layer_mapping, text):
    n_layers = len(layer_mapping)
    n_heads = model.config.num_attention_heads
    n_total_heads = n_layers * n_heads
    I_mat = np.zeros((n_total_heads, n_total_heads))
    causal_damage = np.zeros(n_total_heads)
    
    inp = tokenizer(text, return_tensors="pt")
    baselines, base_loss = capture_baseline_activations_and_loss(model, tokenizer, text, layer_mapping, inp)
    
    for l_src in range(n_layers - 1):
        for h_src in range(n_heads):
            src_global_idx = l_src * n_heads + h_src
            ablated, ablated_loss = capture_ablated_activations_and_loss(model, tokenizer, text, layer_mapping, l_src, h_src, inp)
            
            causal_damage[src_global_idx] = ablated_loss - base_loss
            
            for l_tgt in range(l_src + 1, n_layers):
                for h_tgt in range(n_heads):
                    tgt_global_idx = l_tgt * n_heads + h_tgt
                    b = baselines[(l_tgt, h_tgt)]
                    a = ablated[(l_tgt, h_tgt)]
                    shift = ((b - a).norm(dim=-1) / (b.norm(dim=-1) + 1e-8)).mean().item()
                    I_mat[src_global_idx, tgt_global_idx] = shift
    return I_mat, causal_damage

def compute_weighted_rich_club(G, ranking_dict):
    sorted_nodes = sorted(G.nodes(), key=lambda x: ranking_dict[x], reverse=True)
    all_weights = sorted([d['weight'] for u, v, d in G.edges(data=True)], reverse=True)
    
    r_vals = list(range(2, len(sorted_nodes) // 2))
    r_ratios = []
    for r in r_vals:
        rich_nodes = set(sorted_nodes[:r])
        rich_edges = []
        for u in rich_nodes:
            for v in rich_nodes:
                if G.has_edge(u, v):
                    rich_edges.append(G[u][v]['weight'])
        W_r = sum(rich_edges)
        E_r = len(rich_edges)
        if E_r == 0:
            r_ratios.append(0.0)
            continue
        W_max = sum(all_weights[:E_r])
        ratio = W_r / (W_max + 1e-12)
        r_ratios.append(ratio)
    return r_vals, r_ratios

def compute_weighted_rich_club_normalized(G, ranking_dict, n_randomizations=50):
    """Computes Opsahl rich-club coefficient normalized by a weight-shuffled null model."""
    r_vals, obs_ratios = compute_weighted_rich_club(G, ranking_dict)
    edges = list(G.edges(data=True))
    weights = [d['weight'] for u, v, d in edges]
    
    random_ratios_accum = np.zeros((n_randomizations, len(r_vals)))
    
    for b in range(n_randomizations):
        shuffled_w = np.random.permutation(weights)
        G_rand = nx.DiGraph()
        G_rand.add_nodes_from(G.nodes())
        for idx, (u, v, d) in enumerate(edges):
            G_rand.add_edge(u, v, weight=shuffled_w[idx])
            
        _, rand_ratios = compute_weighted_rich_club(G_rand, ranking_dict)
        random_ratios_accum[b] = rand_ratios
        
    mean_rand_ratios = random_ratios_accum.mean(axis=0) + 1e-12
    normalized_ratios = np.array(obs_ratios) / mean_rand_ratios
    return r_vals, obs_ratios, normalized_ratios

def compute_adjusted_rand_index(labels_a, labels_b):
    """Pure-Python implementation of Adjusted Rand Index (ARI) to avoid pandas/sklearn dependency issues."""
    from collections import Counter
    n = len(labels_a)
    if n <= 1:
        return 1.0
    contingency = Counter(zip(labels_a, labels_b))
    sum_n_ij = sum(nij * (nij - 1) / 2 for nij in contingency.values())
    a_counts = Counter(labels_a)
    b_counts = Counter(labels_b)
    sum_a_i = sum(ai * (ai - 1) / 2 for ai in a_counts.values())
    sum_b_j = sum(bj * (bj - 1) / 2 for bj in b_counts.values())
    expected_index = (sum_a_i * sum_b_j) / (n * (n - 1) / 2)
    max_index = (sum_a_i + sum_b_j) / 2.0
    if max_index == expected_index:
        return 0.0
    ari = (sum_n_ij - expected_index) / (max_index - expected_index)
    return float(ari)

def calculate_topological_metrics(I_mat, head_labels, n_layers, n_heads):
    n_total_heads = I_mat.shape[0]
    G_full = nx.DiGraph()
    for i in range(n_total_heads):
        G_full.add_node(head_labels[i], layer=int(i // n_heads), head=int(i % n_heads))
        
    for u_idx in range(n_total_heads):
        for v_idx in range(n_total_heads):
            weight = I_mat[u_idx, v_idx]
            if weight > 0:
                G_full.add_edge(head_labels[u_idx], head_labels[v_idx], weight=weight)
                
    pagerank_dict = nx.pagerank(G_full, weight='weight')
    
    # Reverse Graph PageRank (flows back from downstream targets to sources)
    G_rev = G_full.reverse(copy=True)
    pagerank_rev_dict = nx.pagerank(G_rev, weight='weight')
    
    in_strength = np.array([G_full.in_degree(n, weight='weight') for n in head_labels])
    out_strength = np.array([G_full.out_degree(n, weight='weight') for n in head_labels])
    return G_full, pagerank_dict, pagerank_rev_dict, in_strength, out_strength

def compute_alignment_scores(I_mat, n_layers, n_heads):
    n_total_heads = I_mat.shape[0]
    outgoing_align = np.zeros(n_total_heads)
    incoming_align = np.zeros(n_total_heads)
    
    for l in range(n_layers - 1):
        src_slice_indices = list(range(l * n_heads, (l + 1) * n_heads))
        tgt_slice_indices = list(range((l + 1) * n_heads, n_total_heads))
        
        I_layer = I_mat[src_slice_indices][:, tgt_slice_indices]
        if I_layer.size == 0 or (I_layer**2).sum() == 0:
            continue
            
        u, S, Vt = np.linalg.svd(I_layer, full_matrices=False)
        u_1 = u[:, 0]
        v_1 = Vt[0, :]
        
        # Outgoing alignment: cosine similarity of head's outgoing influence with dominant routing direction v_1
        for h in range(n_heads):
            g_src_idx = l * n_heads + h
            out_vector = I_layer[h, :]
            norm_out = np.linalg.norm(out_vector)
            norm_v1 = np.linalg.norm(v_1)
            if norm_out > 1e-12 and norm_v1 > 1e-12:
                outgoing_align[g_src_idx] = np.abs(np.dot(out_vector, v_1)) / (norm_out * norm_v1)
                
        # Incoming alignment: cosine similarity of target head's incoming influence with dominant left singular vector u_1
        for tgt_idx, g_tgt_idx in enumerate(tgt_slice_indices):
            in_vector = I_layer[:, tgt_idx]
            norm_in = np.linalg.norm(in_vector)
            norm_u1 = np.linalg.norm(u_1)
            if norm_in > 1e-12 and norm_u1 > 1e-12:
                incoming_align[g_tgt_idx] += np.abs(np.dot(in_vector, u_1)) / (norm_in * norm_u1)
                
    for idx in range(n_total_heads):
        l_idx = idx // n_heads
        if l_idx > 0:
            incoming_align[idx] /= l_idx
            
    return outgoing_align, incoming_align

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Dependency Graph Extraction V4")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (gpt2, pythia, etc.)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local model weights")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of prompt matrices")
    parser.add_argument("--threshold_pct", type=float, default=90.0, help="Percentile threshold for community graph visualization")
    args = parser.parse_args()
    
    np.random.seed(42)
    
    safe_name = args.model.replace("/", "_").replace("-", "_").lower()
    prefix = f"{safe_name}_"
    
    matrix_filename = f"{prefix}dependency_matrix.npy"
    pagerank_filename = f"{prefix}pagerank.npy"
    pagerank_rev_filename = f"{prefix}pagerank_rev.npy"
    communities_filename = f"{prefix}communities.pkl"
    laplacian_filename = f"{prefix}laplacian.npy"
    injector_filename = f"{prefix}injector_scores.npy"
    receiver_filename = f"{prefix}receiver_scores.npy"
    causal_damage_filename = f"{prefix}causal_damage.npy"
    graphml_filename = f"{fig_dir}/{safe_name}_dependency_graph.graphml" if 'fig_dir' in locals() else f"paper/figures/{safe_name}_dependency_graph.graphml"
    
    # Ensure folder paths exist
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    graphml_filename = os.path.join(fig_dir, f"{safe_name}_dependency_graph.graphml")
    
    if "medium" in safe_name or "pythia" in safe_name:
        all_texts = [v[0] for v in DOMAIN_PROBES.values()]
    else:
        all_texts = [t for v in DOMAIN_PROBES.values() for t in v]
    n_prompts = len(all_texts)
    
    prompt_matrices = []
    prompt_damages = []
    
    model_loaded = False
    model, tokenizer, layer_mapping = None, None, None
    
    for i, text in enumerate(all_texts):
        prompt_mat_filename = f"{prefix}prompt_{i}_dependency.npy"
        prompt_dmg_filename = f"{prefix}prompt_{i}_causal_damage.npy"
        if os.path.exists(prompt_mat_filename) and os.path.exists(prompt_dmg_filename) and not args.recompute:
            prompt_matrices.append(np.load(prompt_mat_filename))
            prompt_damages.append(np.load(prompt_dmg_filename))
        else:
            if not model_loaded:
                model, tokenizer = load_model(args.model, args.model_path)
                layer_mapping = find_attn_out_modules(model)
                model_loaded = True
                
            print(f"Computing matrix & loss change for prompt {i+1}/{n_prompts}: '{text[:40].strip()}...'")
            I_prompt, dmg_prompt = compute_prompt_influence_and_damage(model, tokenizer, layer_mapping, text)
            np.save(prompt_mat_filename, I_prompt)
            np.save(prompt_dmg_filename, dmg_prompt)
            prompt_matrices.append(I_prompt)
            prompt_damages.append(dmg_prompt)
            
    prompt_matrices = np.array(prompt_matrices)
    prompt_damages = np.array(prompt_damages)
    n_total_heads = prompt_matrices.shape[1]
    
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from flood.layout import get_model_layout
    layout = get_model_layout(args.model)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
        
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    
    I_mat = prompt_matrices.mean(axis=0)
    causal_damage = prompt_damages.mean(axis=0)
    
    np.save(matrix_filename, I_mat)
    np.save(causal_damage_filename, causal_damage)
    
    # =====================================================================
    # PHASE A: DESCRIPTIVE WEIGHTED NETWORK ANALYSIS
    # =====================================================================
    print("\n" + "="*80)
    print("PHASE A: DESCRIPTIVE WEIGHTED NETWORK ANALYSIS")
    print("="*80)
    
    G_full, pagerank_dict, pagerank_rev_dict, in_strength, out_strength = calculate_topological_metrics(I_mat, head_labels, n_layers, n_heads)
    
    # Save GraphML for interoperability (Gephi/Cytoscape)
    nx.write_graphml(G_full, graphml_filename)
    print(f"  Saved complete dependency graph in GraphML format -> {graphml_filename}")
    
    # Katz Centrality safely
    try:
        katz_dict = nx.katz_centrality(G_full, alpha=0.01, beta=1.0, weight='weight', max_iter=2000)
    except Exception as e:
        print(f"  [Warning] Katz centrality failed to converge: {e}. Falling back to default numpy solver.")
        try:
            katz_dict = nx.katz_centrality_numpy(G_full, alpha=0.01, beta=1.0, weight='weight')
        except Exception as e2:
            print(f"  [Warning] Katz centrality numpy solver also failed: {e2}. Slicing to uniform scores.")
            katz_dict = {n: 1.0 / n_total_heads for n in head_labels}
            
    print("\n  Top 5 Central Heads by Forward PageRank (Receivers):")
    top_pr = sorted(pagerank_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (k, v) in enumerate(top_pr):
        print(f"    {i+1:02d}. Head {k} | PageRank = {v:.4f}")
        
    print("\n  Top 5 Central Heads by Reverse PageRank (Broadcasters):")
    top_pr_rev = sorted(pagerank_rev_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (k, v) in enumerate(top_pr_rev):
        print(f"    {i+1:02d}. Head {k} | PageRank (rev) = {v:.4f}")
        
    print("\n  Top 5 Central Heads by Katz Centrality:")
    top_katz = sorted(katz_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (k, v) in enumerate(top_katz):
        print(f"    {i+1:02d}. Head {k} | Katz = {v:.4f}")
        
    print("\n  Top 5 Central Heads by Out-Strength:")
    top_out = sorted(range(n_total_heads), key=lambda x: out_strength[x], reverse=True)[:5]
    for i, idx in enumerate(top_out):
        print(f"    {i+1:02d}. Head {head_labels[idx]} | Out-Strength = {out_strength[idx]:.4f}")
        
    # Spectral Graph Analysis
    print("\n  Spectral Graph Analysis:")
    W = (I_mat + I_mat.T) / 2.0
    D = np.diag(W.sum(axis=1))
    L = D - W
    np.save(laplacian_filename, L)
    
    eigenvals, eigenvecs = np.linalg.eigh(L)
    idx_sorted = np.argsort(eigenvals)
    eigenvals = eigenvals[idx_sorted]
    eigenvecs = eigenvecs[:, idx_sorted]
    
    fiedler_val = eigenvals[1]
    fiedler_vec = eigenvecs[:, 1]
    print(f"    Algebraic Connectivity (2nd eigenvalue): {fiedler_val:.4f}")
    
    # Inspect the dominant entries of the Fiedler Vector (magnitudes & signs)
    abs_fiedler = np.abs(fiedler_vec)
    top_fiedler_indices = np.argsort(abs_fiedler)[::-1][:5]
    print("    Largest magnitude Fiedler vector entries (dominant spectral partition):")
    for idx in top_fiedler_indices:
        print(f"      Head {head_labels[idx]} | Weight = {fiedler_vec[idx]:+.4f} (abs={abs_fiedler[idx]:.4f})")
        
    part_0 = [head_labels[i] for i in range(n_total_heads) if fiedler_vec[i] < 0]
    part_1 = [head_labels[i] for i in range(n_total_heads) if fiedler_vec[i] >= 0]
    print(f"    Dominant Spectral Partition sign-split: Part 0 (size={len(part_0)}), Part 1 (size={len(part_1)})")
    
    # Louvain Modularity Comparison (Option A Symmetrized Avg vs Option B Max)
    all_weights = I_mat[I_mat > 0].ravel()
    threshold_val = np.percentile(all_weights, args.threshold_pct)
    
    G_avg = nx.Graph()
    G_avg.add_nodes_from(head_labels)
    G_max = nx.Graph()
    G_max.add_nodes_from(head_labels)
    
    for u_idx in range(n_total_heads):
        for v_idx in range(u_idx + 1, n_total_heads):
            w1 = I_mat[u_idx, v_idx]
            w2 = I_mat[v_idx, u_idx]
            w_avg = (w1 + w2) / 2.0
            w_max = max(w1, w2)
            
            if w_avg >= threshold_val:
                G_avg.add_edge(head_labels[u_idx], head_labels[v_idx], weight=w_avg)
            if w_max >= threshold_val:
                G_max.add_edge(head_labels[u_idx], head_labels[v_idx], weight=w_max)
                
    comm_avg = list(nx.community.louvain_communities(G_avg, weight='weight', seed=42))
    comm_max = list(nx.community.louvain_communities(G_max, weight='weight', seed=42))
    
    mod_avg = nx.community.modularity(G_avg, comm_avg, weight='weight')
    mod_max = nx.community.modularity(G_max, comm_max, weight='weight')
    print(f"\n  Louvain Modularity Comparison (threshold weight >= {threshold_val:.5f}):")
    print(f"    - Modularity (Symmetrized Average): {mod_avg:.4f} ({len(comm_avg)} communities)")
    print(f"    - Modularity (Symmetrized Max):     {mod_max:.4f} ({len(comm_max)} communities)")
    
    labels_avg = np.zeros(n_total_heads)
    labels_max = np.zeros(n_total_heads)
    for ci, comm in enumerate(comm_avg):
        for n in comm:
            labels_avg[head_labels.index(n)] = ci
    for ci, comm in enumerate(comm_max):
        for n in comm:
            labels_max[head_labels.index(n)] = ci
    ari_score = compute_adjusted_rand_index(labels_avg, labels_max)
    print(f"    - Adjusted Rand Index (ARI) between partitions: {ari_score:.4f}")

    # Anti-symmetric Component (I - I^T)
    asymmetry_val = np.linalg.norm(I_mat - I_mat.T) / (np.linalg.norm(I_mat) + 1e-12)
    print(f"\n  Routing Reciprocity:")
    print(f"    - Graph Asymmetry Index ||I - I^T||_F / ||I||_F: {asymmetry_val:.4f}")
    print(f"      (Note: This value of 1.414 largely reflects the causal feedforward ordering imposed by the transformer architecture (DAG structure), where I(v, u) = 0 for almost all v < u.)")

    # Flow Bottlenecks & Edge Betweenness Centrality
    print("\n  Weighted Flow Bottleneck & Edge Centrality Analysis:")
    
    # 1. Edge Betweenness Centrality
    G_dist = G_full.copy()
    for u, v, d in G_dist.edges(data=True):
        G_dist[u][v]['distance'] = 1.0 / (d['weight'] + 1e-12)
    edge_betweenness = nx.edge_betweenness_centrality(G_dist, weight='distance')
    top_edges_betweenness = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
    print("    - Top 10 edges by Edge Betweenness Centrality:")
    for i, ((u, v), val) in enumerate(top_edges_betweenness):
        print(f"      {i+1:02d}. {u} -> {v} | Betweenness = {val:.4f} | weight = {I_mat[head_labels.index(u), head_labels.index(v)]:.4f}")
        
    # 2. Minimum s-t Cut
    G_flow = G_full.copy()
    G_flow.add_node("VIRTUAL_SOURCE")
    G_flow.add_node("VIRTUAL_SINK")
    for name in head_labels:
        l_idx = head_labels.index(name) // n_heads
        if l_idx == 0:
            G_flow.add_edge("VIRTUAL_SOURCE", name, weight=999.0)
        if l_idx == n_layers - 1:
            G_flow.add_edge(name, "VIRTUAL_SINK", weight=999.0)
            
    cut_val, partition = nx.minimum_cut(G_flow, "VIRTUAL_SOURCE", "VIRTUAL_SINK", capacity="weight")
    part_source, part_sink = partition
    cut_edges = []
    for u in part_source:
        for v in part_sink:
            if G_flow.has_edge(u, v):
                w = G_flow[u][v]['weight']
                if u != "VIRTUAL_SOURCE" and v != "VIRTUAL_SINK":
                    cut_edges.append((u, v, w))
    cut_edges = sorted(cut_edges, key=lambda x: x[2], reverse=True)
    print(f"    - Weighted Min-Cut Capacity: {cut_val:.4f}")
    print("    - Top bottleneck edges crossing the cut partition:")
    for u, v, w in cut_edges[:5]:
        print(f"      {u} -> {v} | weight = {w:.4f}")

    # =====================================================================
    # PHASE B: HYPOTHESIS TESTING & BOOTSTRAP UNCERTAINTY
    # =====================================================================
    print("\n" + "="*80)
    print("PHASE B: HYPOTHESIS TESTING WITH BOOTSTRAP UNCERTAINTY")
    print("="*80)
    
    num_bootstraps = 20 if n_total_heads > 144 else 200
    
    # Correlations for Forward PageRank
    boot_sp_causal_fwd = []
    boot_pe_causal_fwd = []
    boot_kd_causal_fwd = []
    
    # Correlations for Reverse PageRank (Broadcasters)
    boot_sp_causal_rev = []
    boot_pe_causal_rev = []
    boot_kd_causal_rev = []
    
    # Derive bridge candidates dynamically: top-2 heads by mean causal damage
    # This ensures valid labels for any model (GPT-2, Pythia, etc.)
    causal_damage_mean = causal_damage  # already averaged over prompts above
    top_bridge_indices = np.argsort(causal_damage_mean)[-2:][::-1]
    bridge_candidates = [head_labels[i] for i in top_bridge_indices if i < len(head_labels)]
    if not bridge_candidates:
        bridge_candidates = [head_labels[0]]
    
    boot_bridge_prs_fwd = {bc: [] for bc in bridge_candidates}
    boot_bridge_ranks_fwd = {bc: [] for bc in bridge_candidates}
    boot_bridge_prs_rev = {bc: [] for bc in bridge_candidates}
    boot_bridge_ranks_rev = {bc: [] for bc in bridge_candidates}
    
    boot_mod_avg = []
    boot_mod_max = []
    boot_asymmetry = []
    boot_connectivity = []
    
    # PageRank Rich-Club tracking
    rich_club_vals = [10, 20, 30]
    boot_rich_club_obs = {k: [] for k in rich_club_vals}
    boot_rich_club_norm = {k: [] for k in rich_club_vals}
    
    # PageRank stability tracking (top 10 counts)
    pr_top10_counts_fwd = {n: 0 for n in head_labels}
    pr_top10_counts_rev = {n: 0 for n in head_labels}
    
    print(f"Running {num_bootstraps} bootstrap resamples over prompts...")
    
    for b in range(num_bootstraps):
        idx = np.random.choice(n_prompts, size=n_prompts, replace=True)
        I_mat_boot = prompt_matrices[idx].mean(axis=0)
        causal_boot = prompt_damages[idx].mean(axis=0)
        
        G_boot, pr_boot, pr_rev_boot, in_s_boot, out_s_boot = calculate_topological_metrics(I_mat_boot, head_labels, n_layers, n_heads)
        
        # 1. PageRank Top 10 stability (Forward & Reverse)
        sorted_top10_fwd = sorted(head_labels, key=lambda x: pr_boot[x], reverse=True)[:10]
        sorted_top10_rev = sorted(head_labels, key=lambda x: pr_rev_boot[x], reverse=True)[:10]
        for n in sorted_top10_fwd:
            pr_top10_counts_fwd[n] += 1
        for n in sorted_top10_rev:
            pr_top10_counts_rev[n] += 1
            
        # 2. Correlations with Causal Damage (Forward)
        pr_arr_boot = np.array([pr_boot[n] for n in head_labels])
        sp_c_fwd, _ = stats.spearmanr(pr_arr_boot, causal_boot)
        pe_c_fwd, _ = stats.pearsonr(pr_arr_boot, causal_boot)
        kd_c_fwd, _ = stats.kendalltau(pr_arr_boot, causal_boot)
        boot_sp_causal_fwd.append(sp_c_fwd)
        boot_pe_causal_fwd.append(pe_c_fwd)
        boot_kd_causal_fwd.append(kd_c_fwd)
        
        # 3. Correlations with Causal Damage (Reverse)
        pr_rev_arr_boot = np.array([pr_rev_boot[n] for n in head_labels])
        sp_c_rev, _ = stats.spearmanr(pr_rev_arr_boot, causal_boot)
        pe_c_rev, _ = stats.pearsonr(pr_rev_arr_boot, causal_boot)
        kd_c_rev, _ = stats.kendalltau(pr_rev_arr_boot, causal_boot)
        boot_sp_causal_rev.append(sp_c_rev)
        boot_pe_causal_rev.append(pe_c_rev)
        boot_kd_causal_rev.append(kd_c_rev)
        
        # 4. Bridge placements (Forward)
        sorted_pr_fwd = sorted(pr_boot.values(), reverse=True)
        for bc in bridge_candidates:
            if bc in pr_boot:
                boot_bridge_prs_fwd[bc].append(pr_boot[bc])
                boot_bridge_ranks_fwd[bc].append(sorted_pr_fwd.index(pr_boot[bc]) + 1)
                
        # 5. Bridge placements (Reverse)
        sorted_pr_rev = sorted(pr_rev_boot.values(), reverse=True)
        for bc in bridge_candidates:
            if bc in pr_rev_boot:
                boot_bridge_prs_rev[bc].append(pr_rev_boot[bc])
                boot_bridge_ranks_rev[bc].append(sorted_pr_rev.index(pr_rev_boot[bc]) + 1)
                
        # 6. Asymmetry
        asym = np.linalg.norm(I_mat_boot - I_mat_boot.T) / (np.linalg.norm(I_mat_boot) + 1e-12)
        boot_asymmetry.append(asym)
        
        # 7. Spectral connectivity
        W_b = (I_mat_boot + I_mat_boot.T) / 2.0
        D_b = np.diag(W_b.sum(axis=1))
        L_b = D_b - W_b
        e_vals = np.linalg.eigvalsh(L_b)
        boot_connectivity.append(sorted(e_vals)[1])
        
        # 8. Modularity
        G_avg_b = nx.Graph()
        G_avg_b.add_nodes_from(head_labels)
        for u_idx in range(n_total_heads):
            for v_idx in range(u_idx + 1, n_total_heads):
                w1 = I_mat_boot[u_idx, v_idx]
                w2 = I_mat_boot[v_idx, u_idx]
                w_avg = (w1 + w2) / 2.0
                if w_avg >= threshold_val:
                    G_avg_b.add_edge(head_labels[u_idx], head_labels[v_idx], weight=w_avg)
        if G_avg_b.number_of_edges() > 0:
            c_avg_b = list(nx.community.louvain_communities(G_avg_b, weight='weight'))
            boot_mod_avg.append(nx.community.modularity(G_avg_b, c_avg_b, weight='weight'))
        else:
            boot_mod_avg.append(0.0)
            
        # 9. Rich-Club Analysis (using Reverse PageRank)
        for k_pct in rich_club_vals:
            sorted_nodes_pr = sorted(head_labels, key=lambda x: pr_rev_boot[x], reverse=True)
            top_k_count = int(n_total_heads * k_pct / 100.0)
            top_k_nodes = set(sorted_nodes_pr[:top_k_count])
            
            actual_w = 0.0
            max_w = 0.0
            all_w_sorted = sorted([d['weight'] for u, v, d in G_boot.edges(data=True)], reverse=True)
            for u in top_k_nodes:
                for v in top_k_nodes:
                    if G_boot.has_edge(u, v):
                        actual_w += I_mat_boot[head_labels.index(u), head_labels.index(v)]
                        max_w += 1.0
            E_r = int(max_w)
            W_max = sum(all_w_sorted[:E_r]) if len(all_w_sorted) >= E_r else sum(all_w_sorted)
            obs_ratio = actual_w / (W_max + 1e-12)
            boot_rich_club_obs[k_pct].append(obs_ratio)
            
            # Null model weight permutation
            shuffled_w = np.random.permutation(all_w_sorted)
            edges_list = list(G_boot.edges())
            G_rand = nx.DiGraph()
            G_rand.add_nodes_from(head_labels)
            for idx_e, (u_e, v_e) in enumerate(edges_list):
                G_rand.add_edge(u_e, v_e, weight=shuffled_w[idx_e])
            rand_actual = 0.0
            for u in top_k_nodes:
                for v in top_k_nodes:
                    if G_rand.has_edge(u, v):
                        rand_actual += G_rand[u][v]['weight']
            rand_ratio = rand_actual / (W_max + 1e-12)
            boot_rich_club_norm[k_pct].append(obs_ratio / (rand_ratio + 1e-12))

    outgoing_align, incoming_align = compute_alignment_scores(I_mat, n_layers, n_heads)
    ci_95 = lambda vals: (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
    
    # ── HYPOTHESIS 1: Correlation with Bridge Causal Importance ──
    print("\n  Hypothesis 1: Centrality vs. Causal Importance Correlation")
    boot_r2_fwd = [r**2 for r in boot_pe_causal_fwd]
    boot_r2_rev = [r**2 for r in boot_pe_causal_rev]
    
    print("    Receiver Centrality (Forward PageRank):")
    print(f"      Spearman rho: {np.mean(boot_sp_causal_fwd):.4f} [95% CI: {ci_95(boot_sp_causal_fwd)[0]:.4f}, {ci_95(boot_sp_causal_fwd)[1]:.4f}]")
    print(f"      Pearson r:   {np.mean(boot_pe_causal_fwd):.4f} [95% CI: {ci_95(boot_pe_causal_fwd)[0]:.4f}, {ci_95(boot_pe_causal_fwd)[1]:.4f}]")
    print(f"      Kendall tau: {np.mean(boot_kd_causal_fwd):.4f} [95% CI: {ci_95(boot_kd_causal_fwd)[0]:.4f}, {ci_95(boot_kd_causal_fwd)[1]:.4f}]")
    print(f"      R-squared (explained variance): {np.mean(boot_r2_fwd):.4f} [95% CI: {ci_95(boot_r2_fwd)[0]:.4f}, {ci_95(boot_r2_fwd)[1]:.4f}]")
    
    print("    Broadcast Centrality (Reverse PageRank):")
    print(f"      Spearman rho: {np.mean(boot_sp_causal_rev):.4f} [95% CI: {ci_95(boot_sp_causal_rev)[0]:.4f}, {ci_95(boot_sp_causal_rev)[1]:.4f}]")
    print(f"      Pearson r:   {np.mean(boot_pe_causal_rev):.4f} [95% CI: {ci_95(boot_pe_causal_rev)[0]:.4f}, {ci_95(boot_pe_causal_rev)[1]:.4f}]")
    print(f"      Kendall tau: {np.mean(boot_kd_causal_rev):.4f} [95% CI: {ci_95(boot_kd_causal_rev)[0]:.4f}, {ci_95(boot_kd_causal_rev)[1]:.4f}]")
    print(f"      R-squared (explained variance): {np.mean(boot_r2_rev):.4f} [95% CI: {ci_95(boot_r2_rev)[0]:.4f}, {ci_95(boot_r2_rev)[1]:.4f}]")

    # ── HYPOTHESIS 2: Bridge Heads are Centrality Outliers ──
    print("\n  Hypothesis 2: Placement of Causal Bridge Heads")
    for bc in bridge_candidates:
        print(f"    Bridge Head {bc}:")
        print("      Forward PageRank (Receiver Placement):")
        fwd_prs = boot_bridge_prs_fwd[bc]
        fwd_rks = boot_bridge_ranks_fwd[bc]
        rev_prs = boot_bridge_prs_rev[bc]
        rev_rks = boot_bridge_ranks_rev[bc]
        if len(fwd_prs) > 1:
            print(f"        PageRank: {np.mean(fwd_prs):.5f} [95% CI: {ci_95(fwd_prs)[0]:.5f}, {ci_95(fwd_prs)[1]:.5f}]")
            print(f"        Rank:     #{np.mean(fwd_rks):.1f} [95% CI: #{ci_95(fwd_rks)[0]:.0f}, #{ci_95(fwd_rks)[1]:.0f}]")
        else:
            print(f"        PageRank: {np.mean(fwd_prs) if fwd_prs else float('nan'):.5f} (insufficient bootstrap samples)")
        print("      Reverse PageRank (Broadcaster/Source Placement):")
        if len(rev_prs) > 1:
            print(f"        PageRank: {np.mean(rev_prs):.5f} [95% CI: {ci_95(rev_prs)[0]:.5f}, {ci_95(rev_prs)[1]:.5f}]")
            print(f"        Rank:     #{np.mean(rev_rks):.1f} [95% CI: #{ci_95(rev_rks)[0]:.0f}, #{ci_95(rev_rks)[1]:.0f}]")
        else:
            print(f"        PageRank: {np.mean(rev_prs) if rev_prs else float('nan'):.5f} (insufficient bootstrap samples)")

    # ── HYPOTHESIS 3: Bridge Heads form a Dense Rich-Club Backbone ──
    print("\n  Hypothesis 3: Rich-Club Backbone (using Reverse PageRank)")
    for k_pct in rich_club_vals:
        mean_obs = np.mean(boot_rich_club_obs[k_pct])
        ci_obs = ci_95(boot_rich_club_obs[k_pct])
        mean_norm = np.mean(boot_rich_club_norm[k_pct])
        ci_norm = ci_95(boot_rich_club_norm[k_pct])
        print(f"    Top {k_pct}% PageRank (rev) heads:")
        print(f"      Observed phi:   {mean_obs:.4f} [95% CI: {ci_obs[0]:.4f}, {ci_obs[1]:.4f}]")
        print(f"      Normalized phi: {mean_norm:.4f} [95% CI: {ci_norm[0]:.4f}, {ci_norm[1]:.4f}]")

    # ── HYPOTHESIS 4: Bootstrap PageRank stability ──
    print("\n  Hypothesis 4: Bootstrap PageRank stability (Top 10 heads)")
    print("    Forward PageRank (Top Sinks):")
    top10_stability_fwd = sorted(pr_top10_counts_fwd.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (k, count) in enumerate(top10_stability_fwd):
        print(f"      {i+1:02d}. Head {k} | Top 10 frequency = {count / num_bootstraps * 100:.1f}%")
    print("    Reverse PageRank (Top Sources):")
    top10_stability_rev = sorted(pr_top10_counts_rev.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (k, count) in enumerate(top10_stability_rev):
        print(f"      {i+1:02d}. Head {k} | Top 10 frequency = {count / num_bootstraps * 100:.1f}%")

    # ── HYPOTHESIS 5: Spectral Connectivity CIs ──
    print("\n  Hypothesis 5: Algebraic Connectivity CIs")
    print(f"    lambda_2: {np.mean(boot_connectivity):.4f} [95% CI: {ci_95(boot_connectivity)[0]:.4f}, {ci_95(boot_connectivity)[1]:.4f}]")

    # Save outputs
    np.save(pagerank_filename, pagerank_dict)
    np.save(pagerank_rev_filename, pagerank_rev_dict)
    np.save(injector_filename, outgoing_align)
    np.save(receiver_filename, incoming_align)
    
    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    print("\n=== Generating Visualization Plots ===")
    
    # 1. PageRank vs Causal Damage (2-panel Plot: Forward vs. Reverse)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Forward PageRank
    pr_vals_fwd = [pagerank_dict[n] for n in head_labels]
    axes[0].scatter(causal_damage, pr_vals_fwd, color='purple', alpha=0.7, edgecolors='none', s=45)
    for bc in bridge_candidates:
        idx = head_labels.index(bc)
        axes[0].scatter(causal_damage[idx], pagerank_dict[bc], color='red', marker='*', s=180, edgecolors='black', label=f'Bridge {bc}' if bc == bridge_candidates[0] else "")
    axes[0].set_xlabel("Causal Next-token Loss Damage")
    axes[0].set_ylabel("Forward PageRank Centrality")
    axes[0].set_title(f"Forward PageRank (Sinks) vs Causal Damage\nSpearman rho = {np.mean(boot_sp_causal_fwd):.3f}")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()
    
    # Right: Reverse PageRank
    pr_vals_rev = [pagerank_rev_dict[n] for n in head_labels]
    axes[1].scatter(causal_damage, pr_vals_rev, color='darkorange', alpha=0.7, edgecolors='none', s=45)
    for bc in bridge_candidates:
        idx = head_labels.index(bc)
        axes[1].scatter(causal_damage[idx], pagerank_rev_dict[bc], color='red', marker='*', s=180, edgecolors='black', label=f'Bridge {bc}' if bc == bridge_candidates[0] else "")
    axes[1].set_xlabel("Causal Next-token Loss Damage")
    axes[1].set_ylabel("Reverse PageRank Centrality")
    axes[1].set_title(f"Reverse PageRank (Sources) vs Causal Damage\nSpearman rho = {np.mean(boot_sp_causal_rev):.3f}")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()
    
    plt.suptitle(f"Centrality vs. Causal Damage: Forward vs. Reverse Graph ({args.model})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    scatter_path = os.path.join(fig_dir, f"{safe_name}_pagerank_vs_causal_damage.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"  Saved 2-panel PageRank scatter plot -> {scatter_path}")
    
    # 2. Laplacian spectrum & Fiedler partition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(1, len(eigenvals)+1), eigenvals, 'o-', color='steelblue', markersize=4)
    axes[0].set_xlabel("Eigenvalue Rank")
    axes[0].set_ylabel("Eigenvalue Magnitude")
    axes[0].set_title("Graph Laplacian Spectrum")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    fiedler_sorted_idx = np.argsort(fiedler_vec)
    fiedler_sorted = fiedler_vec[fiedler_sorted_idx]
    
    axes[1].bar(range(len(fiedler_sorted)), fiedler_sorted, color=['teal' if val < 0 else 'orange' for val in fiedler_sorted], alpha=0.8)
    axes[1].set_xlabel("Heads (Sorted by Fiedler Value)")
    axes[1].set_ylabel("Fiedler Vector Weight")
    axes[1].set_title("Dominant Spectral Partition (Fiedler Vector)")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    laplacian_path = os.path.join(fig_dir, f"{safe_name}_spectral_laplacian.png")
    plt.savefig(laplacian_path, dpi=200)
    plt.close()

    # 3. Asymmetric Component Heatmap
    plt.figure(figsize=(7, 6))
    asym_mat = I_mat - I_mat.T
    plt.imshow(asym_mat, cmap='coolwarm', aspect='equal', interpolation='nearest')
    plt.colorbar(label='Asymmetry Weight (I(u, v) - I(v, u))')
    plt.title(f"Antisymmetric Routing Component (Asymmetry = {asymmetry_val:.3f})\n{args.model}")
    plt.xlabel("Target Head Index")
    plt.ylabel("Source Head Index")
    plt.tight_layout()
    asym_path = os.path.join(fig_dir, f"{safe_name}_antisymmetric_component.png")
    plt.savefig(asym_path, dpi=200)
    plt.close()

    # 4. Grouped Layer Visual Layout (Grouped by Layer, node size scaled by Reverse PageRank)
    plt.figure(figsize=(14, 8))
    pos = {}
    for idx, name in enumerate(head_labels):
        l = idx // n_heads
        h = idx % n_heads
        x = l * 2.0
        y = h - (n_heads / 2.0)
        pos[name] = (x, y)
        
    eps = 1e-6
    routing_ratio = np.log10((out_strength + eps) / (in_strength + eps))
    min_rr, max_rr = routing_ratio.min(), routing_ratio.max()
    norm_rr = (routing_ratio - min_rr) / (max_rr - min_rr + 1e-12)
    node_colors_rr = plt.cm.coolwarm(norm_rr)
    
    # Scale node size by REVERSE PageRank (broadcaster importance)
    pr_scale = 30 + (np.array(pr_vals_rev) - min(pr_vals_rev)) / (max(pr_vals_rev) - min(pr_vals_rev) + 1e-12) * 450
    
    # Build sparsified visualization graph using top 2 outgoing edges per node
    G_vis = nx.DiGraph()
    G_vis.add_nodes_from(head_labels)
    for u in head_labels:
        out_edges = []
        for v in G_full.successors(u):
            w = G_full[u][v]['weight']
            out_edges.append((v, w))
        out_edges.sort(key=lambda x: x[1], reverse=True)
        for v, w in out_edges[:2]:
            G_vis.add_edge(u, v, weight=w)
            
    nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors_rr, node_size=pr_scale, alpha=0.85)
    
    edge_weights = [d['weight'] for u, v, d in G_vis.edges(data=True)]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (w / max_w) * 3.0 for w in edge_weights]
    nx.draw_networkx_edges(G_vis, pos, width=edge_widths, edge_color="gray", alpha=0.35, arrowsize=8)
    
    top_pr_rev_names = [k for k, v in top_pr_rev[:8]]
    labels_dict = {n: n for n in top_pr_rev_names}
    nx.draw_networkx_labels(G_vis, pos, labels=labels_dict, font_size=8, font_weight="bold", font_color="black")
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min_rr, vmax=max_rr))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.08, fraction=0.046)
    cbar.set_label("Continuous Routing Role: log10(Out-strength / In-strength)  [<0: Dampener,  >0: Amplifier]", fontsize=10)
    
    plt.title(f"Transformer Dependency Graph Topology ({args.model})\n(Node sizes scaled by Reverse PageRank; top 2 outgoing edges per node shown)", fontsize=13, fontweight="bold")
    plt.xlabel("Layer Index", fontsize=11)
    plt.xticks(np.arange(n_layers) * 2.0, [f"L{l}" for l in range(n_layers)])
    plt.yticks([])
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, f"{safe_name}_dependency_network_grouped.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    
    print(f"  Saved layer-grouped network visualization plot -> {fig_path}")
    print(f"  Saved asymmetric component plot -> {asym_path}")
    print("=" * 80)
    print("=== DEPENDENCY GRAPH EXTRACTION COMPLETE ===")

if __name__ == "__main__":
    main()
