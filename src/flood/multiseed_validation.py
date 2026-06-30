"""
src/flood/multiseed_validation.py
==================================
Multi-seed reproducibility experiment for FLOOD head pruning.

Generates:
  1. Per-seed PPL table  (mean +/- std across seeds, for each method × budget)
  2. Jaccard head-set stability across seeds (for each method × budget)
  3. Numerical weight-robustness summary  (within +/-20% perturbation)
  4. Cross-model routing-role comparison table  (bridge / broadcaster / receiver %)
  5. Recovery vs. unpruned baseline summary table

Usage:
  python -u -s src/flood/multiseed_validation.py --model gpt2
  python -u -s src/flood/multiseed_validation.py --model EleutherAI/pythia-70m
  python -u -s src/flood/multiseed_validation.py --model gpt2 EleutherAI/pythia-70m
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

# -- path setup ----------------------------------------------------------------
_FLOOD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _FLOOD_DIR)

from score import compute_routing_importance_scores
from prune import prune_attention_heads
from layout import get_model_layout
from benchmark import (
    load_base_model,
    FT_TRAIN_TEXTS,
    calculate_perplexity,
    fine_tune_and_track_recovery,
)

# -- constants -----------------------------------------------------------------
SEEDS       = [1, 7, 13, 21, 42]
BUDGETS     = [0.10, 0.20, 0.30]
METHODS     = ["FLOOD", "FLOOD-Simplified", "Gradient-Taylor", "Magnitude", "Random"]
FT_STEPS    = [0, 5, 10, 20]
WIKITEXT_N  = 15          # passages per evaluation

# -- helpers -------------------------------------------------------------------

def load_wikitext_passages(n=WIKITEXT_N, seed=42):
    """Return n WikiText-2 test passages, sampled with a given seed."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        candidates = [t for t in ds["text"] if len(t.strip()) > 100]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        return [candidates[i] for i in sorted(idx)]
    except Exception as e:
        print(f"  [Warning] WikiText-2 unavailable ({e}). Falling back to domain probes.")
        return None   # caller will use EVAL_TEXTS from benchmark.py


def get_ordered_heads(model, tokenizer, model_name, method, seed, n_layers, n_heads):
    """Return heads sorted worst-first (to remove) for the given method."""
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()

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
        # Use seed-specific eval texts so gradient estimates vary across seeds
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


def jaccard(set_a, set_b):
    a, b = set(map(tuple, set_a)), set(map(tuple, set_b))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# -- Experiment 1: multi-seed PPL ----------------------------------------------

def run_multiseed_ppl(model_name, n_runs=5):
    """
    For each seed, sample eval texts, prune with each method×budget,
    measure PPL. Return nested dict: results[method][budget] = [ppl_seed1, ...].
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 1: Multi-Seed PPL  |  model={model_name}")
    print(f"{'='*70}")

    layout    = get_model_layout(model_name)
    n_layers  = layout["layers"]
    n_heads   = layout["heads"]
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()

    results   = {m: {b: [] for b in BUDGETS} for m in METHODS}
    head_sets = {m: {b: [] for b in BUDGETS} for m in METHODS}  # for Jaccard

    for seed in SEEDS[:n_runs]:
        print(f"\n  Seed {seed} ---------------------------------------------")
        np.random.seed(seed)
        torch.manual_seed(seed)

        model, tokenizer = load_base_model(model_name)
        eval_texts = load_wikitext_passages(n=WIKITEXT_N, seed=seed) or FT_TRAIN_TEXTS

        for method in METHODS:
            ordered = get_ordered_heads(model, tokenizer, model_name,
                                        method, seed, n_layers, n_heads)
            for budget in BUDGETS:
                n_prune = int(n_layers * n_heads * budget)
                heads_to_prune = ordered[:n_prune]
                m_pruned = prune_attention_heads(model, heads_to_prune)
                ppl = calculate_perplexity(m_pruned, tokenizer, eval_texts)
                results[method][budget].append(ppl)
                head_sets[method][budget].append(heads_to_prune)
                del m_pruned
                print(f"    {method:20s}  {int(budget*100):2d}%  PPL={ppl:.2f}")

        del model, tokenizer

    return results, head_sets


def summarise_ppl(results, model_name):
    """Print and return a summary table dict."""
    print(f"\n-- PPL Summary  ({model_name}) ------------------------------")
    print(f"{'Method':<22} {'Budget':>6}  {'Mean PPL':>9}  {'Std':>7}  {'95% CI':>18}")
    print("-" * 70)
    summary = {}
    for method in METHODS:
        for budget in BUDGETS:
            vals = np.array(results[method][budget])
            mean, std = vals.mean(), vals.std()
            ci_lo = np.percentile(vals, 2.5) if len(vals) > 1 else mean
            ci_hi = np.percentile(vals, 97.5) if len(vals) > 1 else mean
            key = f"{method}@{int(budget*100)}%"
            summary[key] = {"mean": float(mean), "std": float(std),
                            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
                            "vals": vals.tolist()}
            print(f"  {method:<20}  {int(budget*100):>4}%  {mean:>9.2f}  {std:>7.2f}  "
                  f"[{ci_lo:.2f}, {ci_hi:.2f}]")
    return summary


# -- Experiment 2: Jaccard head-set stability ----------------------------------

def run_jaccard_stability(head_sets, model_name):
    """Compute pairwise Jaccard across seeds for each method×budget."""
    print(f"\n-- Jaccard Stability  ({model_name}) -------------------------")
    print(f"{'Method':<22} {'Budget':>6}  {'Mean Jaccard':>13}  {'Min':>6}")
    print("-" * 55)
    jaccard_summary = {}
    for method in METHODS:
        for budget in BUDGETS:
            sets = head_sets[method][budget]
            pairs = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    pairs.append(jaccard(sets[i], sets[j]))
            mean_j = np.mean(pairs) if pairs else 1.0
            min_j  = np.min(pairs)  if pairs else 1.0
            key = f"{method}@{int(budget*100)}%"
            jaccard_summary[key] = {"mean": float(mean_j), "min": float(min_j)}
            print(f"  {method:<20}  {int(budget*100):>4}%  {mean_j:>13.4f}  {min_j:>6.4f}")
    return jaccard_summary


# -- Experiment 3: Weight robustness (numerical summary) -----------------------

def run_weight_robustness(model_name, perturbation_range=0.20, n_steps=9):
    """
    Vary FLOOD weight vector within +/-perturbation_range.
    Report PPL change relative to nominal weights.
    """
    print(f"\n-- Weight Robustness Summary  ({model_name}) -----------------")
    layout   = get_model_layout(model_name)
    n_layers = layout["layers"]
    n_heads  = layout["heads"]
    n_prune  = int(n_layers * n_heads * 0.30)

    model, tokenizer = load_base_model(model_name)
    eval_texts = load_wikitext_passages(n=WIKITEXT_N, seed=42) or FT_TRAIN_TEXTS

    nominal_weights = {"w_bridge": 0.50, "w_broadcast": 0.30,
                       "w_injector": 0.10, "w_backbone": 0.10, "w_receiver": 0.00}

    nominal_scores = compute_routing_importance_scores(
        model_name,
        w_bridge=nominal_weights["w_bridge"],
        w_broadcast=nominal_weights["w_broadcast"],
        w_injector=nominal_weights["w_injector"],
        w_backbone=nominal_weights["w_backbone"],
        w_receiver=nominal_weights["w_receiver"]
    )
    nominal_heads  = sorted(nominal_scores.keys(), key=lambda x: nominal_scores[x])[:n_prune]
    m_nominal = prune_attention_heads(model, nominal_heads)
    ppl_nominal = calculate_perplexity(m_nominal, tokenizer, eval_texts)
    del m_nominal
    print(f"  Nominal PPL (w_bridge=0.50, w_bcast=0.30): {ppl_nominal:.2f}")

    ppl_changes = []
    jaccards    = []
    alphas = np.linspace(-perturbation_range, perturbation_range, n_steps)

    for alpha in alphas:
        w_bridge   = max(0.0, nominal_weights["w_bridge"]    * (1 + alpha))
        w_broadcast= max(0.0, nominal_weights["w_broadcast"] * (1 + alpha))
        # renormalize so weights sum to 1
        total = w_bridge + w_broadcast + nominal_weights["w_injector"] + nominal_weights["w_backbone"]
        if total < 1e-9:
            continue
        w_bridge /= total; w_broadcast /= total
        w_inj = nominal_weights["w_injector"] / total
        w_bb  = nominal_weights["w_backbone"]  / total

        scores  = compute_routing_importance_scores(
            model_name,
            w_bridge=w_bridge, w_broadcast=w_broadcast,
            w_injector=w_inj, w_backbone=w_bb, w_receiver=0.0
        )

        heads   = sorted(scores.keys(), key=lambda x: scores[x])[:n_prune]
        m_pert  = prune_attention_heads(model, heads)
        ppl_pert = calculate_perplexity(m_pert, tokenizer, eval_texts)
        del m_pert

        ppl_change = abs(ppl_pert - ppl_nominal) / (ppl_nominal + 1e-9) * 100
        jac        = jaccard(nominal_heads, heads)
        ppl_changes.append(ppl_change)
        jaccards.append(jac)
        print(f"  alpha={alpha:+.2f}  PPL={ppl_pert:.2f}  Change={ppl_change:.1f}%  Jaccard={jac:.3f}")

    del model, tokenizer
    summary = {
        "nominal_ppl": float(ppl_nominal),
        "mean_ppl_change_pct": float(np.mean(ppl_changes)) if ppl_changes else 0.0,
        "max_ppl_change_pct":  float(np.max(ppl_changes))  if ppl_changes else 0.0,
        "mean_head_jaccard":   float(np.mean(jaccards))     if jaccards    else 1.0,
    }
    print(f"\n  Within +/-{int(perturbation_range*100)}% weight perturbations:")
    print(f"    Mean PPL change: {summary['mean_ppl_change_pct']:.1f}%")
    print(f"    Max  PPL change: {summary['max_ppl_change_pct']:.1f}%")
    print(f"    Mean head Jaccard: {summary['mean_head_jaccard']:.3f}")
    return summary


# -- Experiment 4: Cross-model routing-role comparison ------------------------

def routing_role_table(model_names):
    """
    For each model: what fraction of heads are primarily
    Bridge / Broadcaster / Receiver (by top-30% threshold)?
    """
    print(f"\n-- Cross-Model Routing Role Table ---------------------------")
    print(f"{'Model':<30} {'Bridge%':>8} {'Bcast%':>8} {'Recv%':>8} {'Hub Overlap':>12}")
    print("-" * 70)
    table = {}
    for model_name in model_names:
        layout   = get_model_layout(model_name)
        n_layers = layout["layers"]
        n_heads  = layout["heads"]
        n_total  = n_layers * n_heads
        k        = max(1, int(n_total * 0.30))

        try:
            bridge_dict = compute_routing_importance_scores(model_name, w_bridge=1.0, w_broadcast=0.0, w_injector=0.0, w_receiver=0.0, w_backbone=0.0)
            bcast_dict  = compute_routing_importance_scores(model_name, w_bridge=0.0, w_broadcast=1.0, w_injector=0.0, w_receiver=0.0, w_backbone=0.0)

            bridge_heads = sorted(bridge_dict.keys(), key=lambda x: bridge_dict[x])
            bcast_heads  = sorted(bcast_dict.keys(), key=lambda x: bcast_dict[x])

            # "Receiver" proxy: top forward-PageRank from the dependency graph
            safe = model_name.replace("/", "_").replace("-", "_").lower()
            pr_path = f"{safe}_pagerank.npy"
            if os.path.exists(pr_path):
                recv_dict = np.load(pr_path, allow_pickle=True).item()
                parsed_recv_dict = {}
                for name, val in recv_dict.items():
                    l_idx = int(name[1:3])
                    h_idx = int(name[4:6])
                    parsed_recv_dict[(l_idx, h_idx)] = val
                recv_heads = sorted(parsed_recv_dict.keys(), key=lambda x: parsed_recv_dict[x])
            else:
                recv_heads = bridge_heads   # fallback

            top_bridge = set(bridge_heads[-k:])
            top_bcast  = set(bcast_heads[-k:])
            top_recv   = set(recv_heads[-k:])

            hub_overlap = len(top_bridge & top_bcast & top_recv) / max(1, k) * 100
            bridge_bcast_overlap = len(top_bridge & top_bcast) / max(1, k) * 100
            bridge_recv_overlap = len(top_bridge & top_recv) / max(1, k) * 100
            bcast_recv_overlap = len(top_bcast & top_recv) / max(1, k) * 100

            row = {
                "bridge_pct":    len(top_bridge) / n_total * 100,
                "bcast_pct":     len(top_bcast)  / n_total * 100,
                "recv_pct":      len(top_recv)   / n_total * 100,
                "hub_overlap":   hub_overlap,
                "bridge_bcast_overlap": bridge_bcast_overlap,
                "bridge_recv_overlap":  bridge_recv_overlap,
                "bcast_recv_overlap":   bcast_recv_overlap,
            }
            table[model_name] = row
            print(f"  {model_name:<28}  {row['bridge_pct']:>7.1f}%  "
                  f"{row['bcast_pct']:>7.1f}%  {row['recv_pct']:>7.1f}%")
            print(f"    Overlaps -> Bridge-Bcast: {bridge_bcast_overlap:.1f}% | Bridge-Recv: {bridge_recv_overlap:.1f}% | Bcast-Recv: {bcast_recv_overlap:.1f}% | Triple: {hub_overlap:.1f}%")
        except Exception as e:
            print(f"  {model_name:<28}  ERROR: {e}")
            table[model_name] = {"error": str(e)}
    return table


# -- Experiment 5: Recovery vs. baseline control -------------------------------

def run_recovery_vs_control(model_name, seed=42):
    """
    Fine-tune the 30%-FLOOD-pruned model AND the unpruned model on the same
    training texts, track PPL at each step, and report gap/convergence.
    """
    print(f"\n-- Recovery vs. Baseline Control  ({model_name}) -----------")
    np.random.seed(seed); torch.manual_seed(seed)

    layout   = get_model_layout(model_name)
    n_layers = layout["layers"]
    n_heads  = layout["heads"]
    n_prune  = int(n_layers * n_heads * 0.30)

    model, tokenizer = load_base_model(model_name)
    eval_texts = load_wikitext_passages(n=WIKITEXT_N, seed=seed) or FT_TRAIN_TEXTS

    flood_scores  = compute_routing_importance_scores(model_name)
    flood_heads   = sorted(flood_scores.keys(), key=lambda x: flood_scores[x])[:n_prune]
    m_flood_30    = prune_attention_heads(model, flood_heads)

    ppl_before_pruning  = calculate_perplexity(model,      tokenizer, eval_texts)
    ppl_after_pruning   = calculate_perplexity(m_flood_30, tokenizer, eval_texts)

    steps = [0, 5, 10, 20]
    flood_rec    = fine_tune_and_track_recovery(m_flood_30, tokenizer,
                                                FT_TRAIN_TEXTS, eval_texts, steps_list=steps)
    baseline_rec = fine_tune_and_track_recovery(model,     tokenizer,
                                                FT_TRAIN_TEXTS, eval_texts, steps_list=steps)

    print(f"  Unpruned PPL (before FT): {ppl_before_pruning:.2f}")
    print(f"  FLOOD-30%  PPL (before FT): {ppl_after_pruning:.2f}")
    print(f"\n  {'Step':>4}  {'FLOOD':>9}  {'Baseline':>10}  {'Gap':>8}")
    print("  " + "-" * 38)
    result = {"steps": steps, "flood": flood_rec, "baseline": baseline_rec,
              "ppl_before": float(ppl_before_pruning), "ppl_after_prune": float(ppl_after_pruning)}
    for i, st in enumerate(steps):
        gap = flood_rec[i] - baseline_rec[i]
        print(f"  {st:>4}  {flood_rec[i]:>9.2f}  {baseline_rec[i]:>10.2f}  {gap:>+8.2f}")

    del model, tokenizer, m_flood_30
    return result


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FLOOD Multi-Seed Validation Suite")
    parser.add_argument("--model",    type=str, nargs="+", default=["gpt2"],
                        help="One or more model names")
    parser.add_argument("--seeds",    type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budgets",  type=float, nargs="+", default=BUDGETS)
    parser.add_argument("--skip_ppl",           action="store_true")
    parser.add_argument("--skip_robustness",     action="store_true")
    parser.add_argument("--skip_recovery",       action="store_true")
    args = parser.parse_args()

    SEEDS.clear()
    SEEDS.extend(args.seeds)
    BUDGETS.clear()
    BUDGETS.extend(args.budgets)

    all_results = {}

    for model_name in args.model:
        safe = model_name.replace("/", "_").replace("-", "_").lower()
        model_results = {}

        # - Exp 1 + 2: multi-seed PPL + Jaccard -
        if not args.skip_ppl:
            ppl_raw, head_sets = run_multiseed_ppl(model_name, n_runs=len(SEEDS))
            ppl_summary        = summarise_ppl(ppl_raw, model_name)
            jaccard_summary    = run_jaccard_stability(head_sets, model_name)
            model_results["ppl_summary"]    = ppl_summary
            model_results["jaccard_summary"] = jaccard_summary

        # - Exp 3: weight robustness -
        if not args.skip_robustness:
            rob = run_weight_robustness(model_name)
            model_results["weight_robustness"] = rob

        # - Exp 5: recovery vs control -
        if not args.skip_recovery:
            rec = run_recovery_vs_control(model_name)
            model_results["recovery_vs_control"] = rec

        all_results[model_name] = model_results

    # - Exp 4: cross-model routing role table (all models together) -
    role_table = routing_role_table(args.model)
    all_results["routing_role_table"] = role_table

    # - Save -
    out_dir  = "paper/figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multiseed_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Saved all results -> {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
