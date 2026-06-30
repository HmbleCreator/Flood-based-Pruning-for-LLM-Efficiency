"""
src/flood/run_statistical_tests.py
==================================
Runs all 10 pairwise paired bootstrap significance tests at the 30% pruning budget,
applying Holm-Bonferroni correction and computing effect sizes and confidence intervals.
"""

import numpy as np
import torch
import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
_FLOOD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _FLOOD_DIR)

from score import compute_routing_importance_scores
from prune import prune_attention_heads
from layout import get_model_layout
from benchmark import load_base_model, FT_TRAIN_TEXTS, calculate_perplexity

def get_ordered_heads(model, tokenizer, model_name, method, seed, n_layers, n_heads):
    if method == "FLOOD":
        scores_dict = compute_routing_importance_scores(model_name)
        return sorted(scores_dict.keys(), key=lambda x: scores_dict[x])

    if method == "FLOOD-Simplified":
        scores_dict = compute_routing_importance_scores(
            model_name, w_bridge=0.67, w_broadcast=0.33, w_injector=0.0, w_receiver=0.0, w_backbone=0.0
        )
        return sorted(scores_dict.keys(), key=lambda x: scores_dict[x])

    if method == "Gradient-Taylor":
        from benchmark import compute_gradient_importance
        # Use seed 42 passages
        from multiseed_validation import load_wikitext_passages
        texts = load_wikitext_passages(n=8, seed=seed) or FT_TRAIN_TEXTS
        grad_dict = compute_gradient_importance(model, tokenizer, texts, n_layers, n_heads)
        return sorted(grad_dict.keys(), key=lambda x: grad_dict[x])

    if method == "Magnitude":
        from benchmark import compute_magnitude_scores
        scores_dict = compute_magnitude_scores(model, n_layers, n_heads)
        return sorted(scores_dict.keys(), key=lambda x: scores_dict[x])

    if method == "Random":
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        rng = np.random.default_rng(seed)
        rng.shuffle(all_heads)
        return all_heads

    raise ValueError(f"Unknown method: {method}")

def main():
    print("=== Pairwise Statistical Significance Testing ===")
    model_name = "gpt2"
    seed = 42
    budget = 0.30
    num_resamples = 1000
    
    # 1. Load model & tokenizer
    model, tokenizer = load_base_model(model_name)
    layout = get_model_layout(model_name)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    n_prune = int(n_layers * n_heads * budget)
    
    # Load evaluation texts
    from multiseed_validation import load_wikitext_passages
    eval_texts = load_wikitext_passages(n=15, seed=seed) or FT_TRAIN_TEXTS
    
    # 2. Generate pruned models and collect passage-level perplexities
    methods = ["FLOOD", "FLOOD-Simplified", "Gradient-Taylor", "Magnitude", "Random"]
    passage_ppls = {}
    
    print("\nEvaluating passage-level perplexities for all methods...")
    for method in methods:
        ordered = get_ordered_heads(model, tokenizer, model_name, method, seed, n_layers, n_heads)
        heads_to_prune = ordered[:n_prune]
        m_pruned = prune_attention_heads(model, heads_to_prune)
        
        ppls = []
        with torch.no_grad():
            for text in eval_texts:
                inp = tokenizer(text, return_tensors="pt")
                if torch.cuda.is_available():
                    inp = {k: v.cuda() for k, v in inp.items()}
                labels = inp["input_ids"].clone()
                out = m_pruned(**inp, labels=labels)
                ppls.append(np.exp(out.loss.item()))
        
        passage_ppls[method] = np.array(ppls)
        print(f"  {method:<18} -> Mean PPL: {np.mean(ppls):.2f}")
        del m_pruned
        
    del model
    
    # 3. Perform all 10 pairwise comparisons
    comparisons = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            mA, mB = methods[i], methods[j]
            comparisons.append((mA, mB))
            
    results = []
    np.random.seed(seed)
    
    for mA, mB in comparisons:
        pplA = passage_ppls[mA]
        pplB = passage_ppls[mB]
        
        # We test: H0: mean(PPL(A)) - mean(PPL(B)) = 0
        diffs = pplA - pplB
        mean_diff = np.mean(diffs)
        
        # Paired bootstrap
        boot_diffs = []
        for _ in range(num_resamples):
            sample = np.random.choice(diffs, size=len(diffs), replace=True)
            boot_diffs.append(np.mean(sample))
            
        boot_diffs = np.array(boot_diffs)
        
        # 95% Confidence Interval of the difference
        ci_lo = np.percentile(boot_diffs, 2.5)
        # We force CI center at mean_diff for standard display or use percentile
        ci_hi = np.percentile(boot_diffs, 97.5)
        
        # Empirical p-value
        # If mean_diff > 0, we test H1: mean(A) > mean(B) (A is worse than B)
        # If mean_diff < 0, we test H1: mean(A) < mean(B) (A is better than B)
        if mean_diff > 0:
            p_val = np.mean(boot_diffs <= 0)
        else:
            p_val = np.mean(boot_diffs >= 0)
            
        results.append({
            "comparison": f"{mA} vs {mB}",
            "diff": mean_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_raw": float(p_val)
        })
        
    # 4. Holm-Bonferroni correction
    # Sort comparisons by raw p-value ascending
    results.sort(key=lambda x: x["p_raw"])
    m_tests = len(results)
    
    for rank, res in enumerate(results):
        # Holm-Bonferroni adjusted threshold is alpha / (m_tests - rank)
        alpha = 0.05
        threshold = alpha / (m_tests - rank)
        # Adjusted p-value: p_adj = p_raw * (m_tests - rank)
        # Bound adjusted p-value at 1.0 and ensure monotonicity
        p_adj = res["p_raw"] * (m_tests - rank)
        p_adj = min(1.0, max(0.0, p_adj))
        res["p_adj"] = p_adj
        res["significant"] = "Yes" if p_adj < 0.05 else "No"
        
    # Ensure adjusted p-values are non-decreasing
    for rank in range(1, m_tests):
        if results[rank]["p_adj"] < results[rank-1]["p_adj"]:
            results[rank]["p_adj"] = results[rank-1]["p_adj"]
            results[rank]["significant"] = results[rank-1]["significant"]

    # Print Markdown table
    print("\n=== Pairwise Comparison Results (Holm-Bonferroni Corrected) ===")
    print("| Comparison | Mean d_PPL | 95% CI of d | Raw p-value | Holm-Adj p | Significant (alpha=0.05) |")
    print("|---|---|---|---|---|---|")
    for res in results:
        sig_str = "**Yes**" if res["significant"] == "Yes" else "No"
        print(f"| {res['comparison']:<25} | {res['diff']:>9.2f} | [{res['ci_lo']:.2f}, {res['ci_hi']:.2f}] | {res['p_raw']:.6f} | {res['p_adj']:.6f} | {sig_str} |")

if __name__ == "__main__":
    main()
