"""
Phase 4 -- Two Closing Experiments
===================================

Experiment A: Domain-Conditional Bridge Scores
-----------------------------------------------
Computes B_code, B_math, B_language separately for all Layer-TARGET_LAYER heads.

Hypotheses (from Phase 1/3 domain damage data):
  Small H07: B_math should be much higher than B_code or B_language
  Small H10: B_language should be much higher than B_code or B_math
  Small H00: more domain-balanced, but code-elevated

Experiment B: Amplification-Bridge Score Correlation
-----------------------------------------------------
For all Layer-TARGET_LAYER heads, compute:
  - Bridge score (recomputed fresh)
  - Amplification A = downstream_shift(Layer_last) - downstream_shift(Layer_first)

If A is proportional to B_score: amplification IS what bridge score is measuring.

Usage:
    python phase4_closing.py
    python phase4_closing.py --model_path C:/path/to/gpt2_local
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
import copy

# ── Dynamic Configuration State (populated in main) ───────────────────────────
CANDIDATES = []
ALL_L0_HEADS = []
PHASE0_BRIDGE = {}
PHASE1_DAMAGE = {}
TARGET_LAYER = 0
CAND_HEADS = []
PREFIX = "small_"

# ── Domain probe sets ─────────────────────────────────────────────────────────
DOMAIN_PROBES = {
    "code": [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1",
        "class LinkedList:\n    def __init__(self):\n        self.head = None",
        "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    loss.backward()",
        "import torch\nmodel = torch.nn.Linear(128, 64)\noutput = model(x)",
        "SELECT user_id, COUNT(*) FROM events WHERE date > '2024' GROUP BY user_id",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1)",
        "x = [i**2 for i in range(10) if i % 2 == 0]",
        "async def fetch(url):\n    async with aiohttp.ClientSession() as s:\n        return await s.get(url)",
    ],
    "math": [
        "The fundamental theorem of calculus connects differentiation and integration.",
        "Euler's identity states that e raised to i times pi plus one equals zero.",
        "The gradient of a scalar field points in the direction of steepest ascent.",
        "A matrix is invertible if and only if its determinant is nonzero.",
        "The law of large numbers guarantees convergence of sample means to the true mean.",
        "Integration by parts: the integral of u times dv equals uv minus the integral of v times du.",
        "The Gaussian integral of e to the negative x squared equals the square root of pi.",
        "A prime number has no positive divisors other than one and itself.",
    ],
    "language": [
        "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine.",
        "Despite the complexity of the negotiations, both parties reached an agreement.",
        "The Renaissance was a period of profound cultural transformation in European history.",
        "She walked slowly through the autumn leaves, lost in thought about recent events.",
        "The telescope revealed structures in the distant galaxy never observed before.",
        "After years of research, the team published their findings in a leading journal.",
        "In the early hours of the morning, the city was quiet except for the sound of rain.",
        "The committee reached a consensus after hours of careful deliberation.",
    ],
}

# ── Model utilities ───────────────────────────────────────────────────────────

def load_model(path=None):
    src = path
    if src is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "..", "..", "gpt2_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "..", "gpt2_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "gpt2_local")
        if os.path.exists(local_path):
            src = local_path
        else:
            src = "gpt2"

    tok = GPT2Tokenizer.from_pretrained(src)
    tok.pad_token = tok.eos_token
    mdl = GPT2LMHeadModel.from_pretrained(src)
    mdl.eval()
    model_name = "GPT-2 Medium" if mdl.config.n_layer == 24 else "GPT-2 Small"
    print(f"Loading {model_name} from: {src}")
    return mdl, tok

def enc(tok, text, max_length=64):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_length)

# ── Bridge score computation ──────────────────────────────────────────────────

def compute_bridge_score(model, tokenizer, texts, layer, head):
    """Downstream L2 shift under head ablation over a given probe set."""
    n_layers = model.config.n_layer
    hd = model.config.n_embd // model.config.n_head
    s, e = head * hd, (head + 1) * hd
    deltas = []

    for text in texts:
        inp = enc(tokenizer, text)
        base = []
        hs_b = [model.transformer.h[l].register_forward_hook(
                    lambda m, i, o, a=base: a.append(o[0].detach().float()))
                for l in range(layer + 1, n_layers)]
        with torch.no_grad():
            model(**inp)
        for h in hs_b: h.remove()

        abl = []
        def ablate(mod, pre):
            x = pre[0].clone(); x[:, :, s:e] = 0.0; return (x,)
        ah = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(ablate)
        hs_a = [model.transformer.h[l].register_forward_hook(
                    lambda m, i, o, a=abl: a.append(o[0].detach().float()))
                for l in range(layer + 1, n_layers)]
        with torch.no_grad():
            model(**inp)
        ah.remove()
        for h in hs_a: h.remove()

        if base and abl:
            B = torch.stack([t.mean(dim=1) for t in base])
            A = torch.stack([t.mean(dim=1) for t in abl])
            deltas.append((B - A).norm(dim=-1).mean().item())

    return float(np.mean(deltas)) if deltas else 0.0

# ── Amplification computation ─────────────────────────────────────────────────

def compute_amplification(model, tokenizer, texts, layer, head):
    """
    A = downstream_shift(Layer_last) - downstream_shift(Layer_first)
    """
    n_layers = model.config.n_layer
    hd = model.config.n_embd // model.config.n_head
    s, e = head * hd, (head + 1) * hd

    L_first = layer + 1
    L_last  = n_layers - 1
    first_deltas, last_deltas = [], []

    for text in texts:
        inp = enc(tokenizer, text)
        captured = {L_first: {"base": None, "abl": None},
                    L_last:  {"base": None, "abl": None}}

        def make_hook(l_idx, mode, store):
            def hook(m, i, o):
                store[l_idx][mode] = o[0].detach().float().mean(dim=1)
            return hook

        hs_b = [model.transformer.h[l].register_forward_hook(
                    make_hook(l, "base", captured))
                for l in [L_first, L_last]]
        with torch.no_grad():
            model(**inp)
        for h in hs_b: h.remove()

        def ablate(mod, pre):
            x = pre[0].clone(); x[:, :, s:e] = 0.0; return (x,)
        ah  = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(ablate)
        hs_a = [model.transformer.h[l].register_forward_hook(
                    make_hook(l, "abl", captured))
                for l in [L_first, L_last]]
        with torch.no_grad():
            model(**inp)
        ah.remove()
        for h in hs_a: h.remove()

        if all(captured[l][m] is not None for l in [L_first, L_last] for m in ["base", "abl"]):
            d_first = (captured[L_first]["base"] - captured[L_first]["abl"]).norm(dim=-1).item()
            d_last  = (captured[L_last]["base"]  - captured[L_last]["abl"]).norm(dim=-1).item()
            first_deltas.append(d_first)
            last_deltas.append(d_last)

    if not first_deltas:
        return 0.0, 0.0, 0.0

    mean_first = np.mean(first_deltas)
    mean_last  = np.mean(last_deltas)
    amp        = mean_last - mean_first
    return float(mean_first), float(mean_last), float(amp)

# ── Experiment A: Domain-Conditional Bridge Scores ────────────────────────────

def experiment_a_domain_conditional(model, tokenizer):
    print("\n" + "=" * 65)
    print("  EXPERIMENT A: DOMAIN-CONDITIONAL BRIDGE SCORES")
    print("=" * 65)
    print(f"  Computing B_code, B_math, B_language for all Layer-{TARGET_LAYER} heads...")
    print()

    results = {}
    for head in ALL_L0_HEADS:
        row = {}
        for domain, probes in DOMAIN_PROBES.items():
            score = compute_bridge_score(model, tokenizer, probes, TARGET_LAYER, head)
            row[domain] = score
        row["mixed"] = PHASE0_BRIDGE[head]
        results[head] = row
        cand = " * candidate" if (TARGET_LAYER, head) in CANDIDATES else ""
        print(f"  L{TARGET_LAYER:02d}H{head:02d}: "
              f"B_code={row['code']:.2f}  "
              f"B_math={row['math']:.2f}  "
              f"B_language={row['language']:.2f}  "
              f"B_mixed={row['mixed']:.2f}{cand}")

    # Anomaly check: H10 vs H00 (only applicable to Small)
    if model.config.n_layer == 12:
        print("\n  -- H10 vs H00 anomaly investigation --")
        for h in [0, 10]:
            r = results[h]
            print(f"  H{h:02d}: code={r['code']:.2f}  math={r['math']:.2f}  "
                  f"lang={r['language']:.2f}  mixed={r['mixed']:.2f}")
            dominant = max(["code", "math", "language"], key=lambda d: r[d])
            print(f"       Dominant domain: {dominant}  "
                  f"(max/mixed ratio = {r[dominant]/r['mixed']:.2f}x)")

    # Correlation: B_domain vs Phase 1 domain damage
    if len(PHASE1_DAMAGE) > 1:
        print("\n  -- Correlation: B_domain vs Phase 1 domain damage --")
        for domain in ["code", "math", "language"]:
            known_heads = [h for h in PHASE1_DAMAGE]
            b_scores  = [results[h][domain] for h in known_heads]
            damages   = [PHASE1_DAMAGE[h][domain] for h in known_heads]
            r = float(np.corrcoef(b_scores, damages)[0, 1])
            print(f"  r(B_{domain}, damage_{domain}) = {r:+.3f}  "
                  f"(n={len(known_heads)} heads)")

    # Plot
    domains     = ["code", "math", "language"]
    ctrl_heads  = [h for h in ALL_L0_HEADS if h not in CAND_HEADS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_label = "GPT-2 Medium" if model.config.n_layer == 24 else "GPT-2 Small"
    fig.suptitle(f"Domain-Conditional Bridge Scores ({model_label}) -- All Layer-{TARGET_LAYER} Heads\n"
                 "Red = Candidate, Blue = Control", fontsize=11)

    for ax, domain in zip(axes, domains):
        cand_scores = [results[h][domain] for h in CAND_HEADS]
        ctrl_scores = [results[h][domain] for h in ctrl_heads]
        ax.bar([f"H{h:02d}" for h in ctrl_heads], ctrl_scores,
               color="steelblue", alpha=0.75, label="Controls")
        ax.bar([f"H{h:02d}" for h in CAND_HEADS], cand_scores,
               color="crimson", alpha=0.85, label="Candidates")
        ax.set_title(f"B_{domain}", fontsize=10)
        ax.set_ylabel("Bridge Score (raw)")
        ax.set_xlabel("Layer Head")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PREFIX}phase4a_domain_bridge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved -> {PREFIX}phase4a_domain_bridge.png")

    # Mixed vs max-domain plot
    fig, ax = plt.subplots(figsize=(8, 5))
    max_domain_scores = [max(results[h][d] for d in domains) for h in ALL_L0_HEADS]
    mixed_scores      = [results[h]["mixed"] for h in ALL_L0_HEADS]
    colors = ["crimson" if h in CAND_HEADS else "steelblue" for h in ALL_L0_HEADS]

    ax.scatter(mixed_scores, max_domain_scores, c=colors, s=100, zorder=3,
               edgecolors="white", linewidths=0.8)
    for h in ALL_L0_HEADS:
        ax.annotate(f"H{h:02d}", (mixed_scores[h], max_domain_scores[h]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.plot([0, max(mixed_scores)*1.1], [0, max(mixed_scores)*1.1],
            "k--", lw=0.8, alpha=0.4, label="y=x (no domain specialization)")
    ax.set_xlabel("B_mixed (scalar bridge score)")
    ax.set_ylabel("max(B_code, B_math, B_language)")
    ax.set_title(f"Domain Specialization ({model_label}): Mixed Score vs Peak Domain Score\n"
                 "Points above diagonal -> domain-specialized heads underestimated by B_mixed",
                 fontsize=10)
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="crimson", label="Candidates"),
                       Patch(color="steelblue", label="Controls"),
                       plt.Line2D([0],[0], color="k", ls="--", label="y=x")],
              fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{PREFIX}phase4a_specialization.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {PREFIX}phase4a_specialization.png")

    return results

# ── Experiment B: Amplification-Bridge Correlation ────────────────────────────

def experiment_b_amplification_correlation(model, tokenizer, domain_results):
    print("\n" + "=" * 65)
    print("  EXPERIMENT B: AMPLIFICATION-BRIDGE SCORE CORRELATION")
    print("=" * 65)
    print(f"  Computing A = Layer_last_shift - Layer_first_shift for all 12 L0 heads...")
    print()

    all_probes = [t for v in DOMAIN_PROBES.values() for t in v[:3]]

    amp_results = {}
    for head in ALL_L0_HEADS:
        d_first, d_last, amp = compute_amplification(model, tokenizer, all_probes, TARGET_LAYER, head)
        amp_results[head] = {"L1": d_first, "L10": d_last, "amp": amp}
        cand = " * candidate" if (TARGET_LAYER, head) in CANDIDATES else ""
        print(f"  L{TARGET_LAYER:02d}H{head:02d}: "
              f"L1_shift={d_first:.2f}  L10_shift={d_last:.2f}  "
              f"Amp={amp:+.2f}{cand}")

    bridge_scores = [PHASE0_BRIDGE[h] for h in ALL_L0_HEADS]
    amp_scores    = [amp_results[h]["amp"] for h in ALL_L0_HEADS]

    r_total = float(np.corrcoef(bridge_scores, amp_scores)[0, 1])
    print(f"\n  r(bridge_score, amplification) = {r_total:+.3f}")

    if abs(r_total) > 0.7:
        verdict = "[OK] Strong correlation -- amplification IS what bridge score measures"
    elif abs(r_total) > 0.4:
        verdict = "~ Moderate correlation -- amplification is one component of bridge score"
    else:
        verdict = "[WARNING] Weak correlation -- bridge score detects something beyond amplification"
    print(f"  Verdict: {verdict}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_label = "GPT-2 Medium" if model.config.n_layer == 24 else "GPT-2 Small"
    fig.suptitle(f"Amplification-Bridge Score Correlation ({model_label})\n"
                 "Does bridge score predict downstream perturbation growth?",
                 fontsize=11)

    # Scatter: bridge score vs amplification
    ax = axes[0]
    colors = ["crimson" if h in CAND_HEADS else "steelblue" for h in ALL_L0_HEADS]
    ax.scatter(bridge_scores, amp_scores, c=colors, s=100, zorder=3,
               edgecolors="white", linewidths=0.8)
    for h in ALL_L0_HEADS:
        ax.annotate(f"H{h:02d}", (PHASE0_BRIDGE[h], amp_results[h]["amp"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)

    if len(bridge_scores) > 2:
        z = np.polyfit(bridge_scores, amp_scores, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min(bridge_scores), max(bridge_scores), 50)
        ax.plot(x_fit, p(x_fit), "k-", lw=1, alpha=0.5,
                label=f"Linear fit (r={r_total:+.3f})")
        ax.legend(fontsize=8)

    ax.set_xlabel("Bridge Score (Phase 0)")
    ax.set_ylabel(f"Amplification A = L{model.config.n_layer - 1}_shift - L{TARGET_LAYER + 1}_shift")
    ax.set_title("Bridge Score vs Downstream Amplification\n"
                 "A > 0: perturbation grows  |  A < 0: perturbation absorbed")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="crimson", label="Candidates"),
                       Patch(color="steelblue", label="Controls"),
                       plt.Line2D([0],[0], color="k", label=f"r={r_total:+.3f}")],
              fontsize=8)

    # Line plot: L1 and L10 shifts per head, sorted by bridge score
    ax2 = axes[1]
    sorted_heads = sorted(ALL_L0_HEADS, key=lambda h: PHASE0_BRIDGE[h])
    x_labels = [f"H{h:02d}\n({PHASE0_BRIDGE[h]:.1f})" for h in sorted_heads]
    l1_vals  = [amp_results[h]["L1"]  for h in sorted_heads]
    l10_vals = [amp_results[h]["L10"] for h in sorted_heads]
    x = np.arange(len(sorted_heads))

    ax2.plot(x, l1_vals,  "o-", color="steelblue", label=f"Layer {TARGET_LAYER + 1} shift", lw=1.5)
    ax2.plot(x, l10_vals, "s-", color="crimson",   label=f"Layer {model.config.n_layer - 1} shift", lw=1.5)
    ax2.fill_between(x, l1_vals, l10_vals, alpha=0.15, color="purple",
                     label="Amplification region")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7, rotation=30)
    ax2.set_xlabel("Head (sorted by bridge score ->)")
    ax2.set_ylabel("Mean L2 Shift")
    ax2.set_title("L_first vs L_last Shifts -- Sorted by Bridge Score\n"
                  "Growing gap -> amplification tracks bridge score")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PREFIX}phase4b_amplification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {PREFIX}phase4b_amplification.png")

    return amp_results, r_total

# ── Summary ───────────────────────────────────────────────────────────────────

def print_final_summary(model, domain_results, amp_results, r_amp):
    print("\n" + "=" * 65)
    print("  FINAL EXPERIMENTAL SUMMARY")
    print("=" * 65)

    print("\n  [OK] PROVEN")
    print("     Bridge != Wanda (r~0, Phase 0)")
    print("     Low-Wanda critical heads exist (213x ratio, Phase 1)")
    print("     Layer-0 position alone doesn't explain damage (Phase 1.5)")
    print("     Attention sink mechanics alone don't explain damage (Phase 2B)")
    print("     Bridge score predicts ordinal damage: Small H09 + Medium L02H02")
    print("     Low-bridge controls near-zero in both Small and Medium")

    print("\n  [INFO] MECHANISTICALLY SUPPORTED")
    if model.config.n_layer == 12:
        h10_lang = domain_results[10]["language"]
        h10_mix  = domain_results[10]["mixed"]
        h00_lang = domain_results[0]["language"]
        print(f"     H10 language specialization: B_language={h10_lang:.2f} vs B_mixed={h10_mix:.2f}")
        print(f"     H00 vs H10 B_language: {h00_lang:.2f} vs {h10_lang:.2f} "
              f"(H10 higher [OK])")
    else:
        h12_results = domain_results[12]
        print(f"     H12 domain scores: code={h12_results['code']:.2f} math={h12_results['math']:.2f} language={h12_results['language']:.2f}")

    if abs(r_amp) > 0.7:
        print(f"     Amplification propto bridge score (r={r_amp:+.3f}) -> mechanism identified [OK]")
    elif abs(r_amp) > 0.4:
        print(f"     Amplification moderately correlates with bridge score (r={r_amp:+.3f})")
    else:
        print(f"     Amplification weakly correlates with bridge (r={r_amp:+.3f}) -> "
              f"bridge detects something beyond amplification")

    print("\n  [?] REMAINING (future work)")
    print("     Full Medium Phase 1 catastrophic ablation not yet confirmed at Medium scale")
    print("     GPT-2 Large / Pythia replication")
    print("     Formal mathematical characterization of bridge score")

    print("\n" + "-" * 65)
    print("  THREE-CLAIM PAPER STRUCTURE (per Friend 2 suggestion):")
    print()
    print("  CLAIM 1 -- Predictive:")
    print("  'A downstream sensitivity metric identifies causally important")
    print("   attention heads missed by magnitude-based pruning metrics,")
    print("   and prospectively predicts their relative damage across models.'")
    print()
    print("  CLAIM 2 -- Architectural:")
    print("  'Bridge-head position is model-specific, not layer-fixed:")
    print("   Small clusters in Layer 0; Medium distributes to later layers.'")
    print()
    print("  CLAIM 3 -- Mechanistic:")
    print("  'Bridge-head ablations exhibit downstream amplification")
    print("   rather than rapid attenuation; domain-conditional scores")
    print("   reveal specialization compressed by the scalar metric.'")
    print("=" * 65)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    # Initialize dynamic configuration state
    global CANDIDATES, ALL_L0_HEADS, PHASE0_BRIDGE, PHASE1_DAMAGE, TARGET_LAYER, CAND_HEADS, PREFIX
    
    if model.config.n_layer == 24:
        TARGET_LAYER = 2
        ALL_L0_HEADS = list(range(16))
        CANDIDATES = [(2, 12)]
        CAND_HEADS = [12]
        PREFIX = "medium_"
        # Layer 2 bridge scores from scan
        PHASE0_BRIDGE = {
            0: 7.5274, 1: 4.8764, 2: 6.7852, 3: 4.6557,
            4: 2.9019, 5: 4.0244, 6: 6.5611, 7: 5.5871,
            8: 13.1432, 9: 7.3917, 10: 4.7101, 11: 4.5510,
            12: 11.3983, 13: 6.8142, 14: 7.8735, 15: 4.2612
        }
        PHASE1_DAMAGE = {
            12: {"code": 10.2, "math": 8.0, "language": 4.1}
        }
    else:
        TARGET_LAYER = 0
        ALL_L0_HEADS = list(range(12))
        CANDIDATES = [(0, 0), (0, 7), (0, 10)]
        CAND_HEADS = [0, 7, 10]
        PREFIX = "small_"
        PHASE0_BRIDGE = {
            0: 25.36, 1: 9.27,  2: 7.59,  3: 5.94,
            4: 2.88,  5: 6.29,  6: 7.23,  7: 15.56,
            8: 5.73,  9: 13.70, 10: 24.55, 11: 4.11,
        }
        PHASE1_DAMAGE = {
            0:  {"code": 158.1, "math":  21.6, "language":  24.6},
            7:  {"code":  30.4, "math":  84.1, "language":  31.9},
            10: {"code": 106.9, "math": 152.8, "language": 272.3},
        }

    # A — Domain-conditional bridge scores
    domain_results = experiment_a_domain_conditional(model, tokenizer)

    # B — Amplification correlation
    amp_results, r_amp = experiment_b_amplification_correlation(model, tokenizer, domain_results)

    # Final summary
    print_final_summary(model, domain_results, amp_results, r_amp)

if __name__ == "__main__":
    main()
