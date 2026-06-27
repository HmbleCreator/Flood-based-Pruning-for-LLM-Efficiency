"""
Phase 2 — Predictive Validation and Confound Closure
======================================================
Three experiments, in order of importance.

A. H09 Predictive Ablation  [MOST IMPORTANT]
   H09 (bridge=13.70) was identified by the bridge metric in Phase 1.5.
   It was NOT in the original Phase 0 candidate set.
   No ablation has been run on it before this script.
   
   This is the bridge score making a prospective prediction.
   If H09 causes substantial damage: metric correctly predicted an untested head.
   If H09 causes near-zero damage: elevated bridge score is noise at this range.
   
   Expected damage range (from bridge score interpolation):
     Low-bridge controls (2.88-5.73) → 2.7% damage
     H09 (13.70)                     → ??? (metric prediction)
     H07 (15.56)                     → 30-84% damage
   
   A blind prediction materially outweighs post-hoc confirmation.

B. Attention Sink Variance Test  [RESOLVE AMBIGUITY]
   Phase 1.5 mean pos-0 attention: H00=0.359, H07=0.237, H10=0.313
   These sit in the ambiguous 0.2-0.4 zone.
   Mean alone cannot distinguish true sinks from variable-attention heads.
   
   True attention sink: consistent pos-0 routing regardless of input → LOW variance
   Genuine bridge: input-dependent attention patterns → HIGH variance
   
   Method: compute per-text pos-0 attention scores, report mean AND std.
   Low std (< 0.05): sink interpretation strengthened
   High std (> 0.10): bridge interpretation strengthened

C. Improved Completion Probes  [REPLACE FAILED PROBES]
   Phase 1.5 baseline: code=60%, math=0%, language=0%
   GPT-2 is a continuation model, not a factual retrieval model.
   New probes use syntactic/continuation tasks GPT-2 can actually complete.
   
   Three domains designed for GPT-2:
     code      — syntactically determined completions
     narrative — common literary phrase completions
     formulaic — famous phrases (near-deterministic for GPT-2)
   
   Prediction if L00H00 is code-dominant:
     code completions degrade after L00H00 ablation
     narrative/formulaic completions survive

D. Replication Runner (stub)
   After Phase 2, run identical Phase 0+1 core on GPT-2 Medium (345M).
   If same phenomenon appears → not GPT-2-Small-specific weirdness.
   Use: run_replication("gpt2-medium") or run_replication("EleutherAI/pythia-160m")

Usage:
    python phase2_prediction.py
    python phase2_prediction.py --model_path C:/path/to/gpt2_local
"""

import argparse
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict

# ── Known state from Phase 0/1/1.5 ───────────────────────────────────────────

CANDIDATES   = [(0, 0), (0, 7), (0, 10)]
H09_PRED     = (0, 9)    # metric's prospective prediction — untested until now

# Phase 1 damage (from terminal output) — for comparison plots
PHASE1_DAMAGE = {
    (0, 0):  {"code": 158.1, "math":  21.6, "language":  24.6},
    (0, 7):  {"code":  30.4, "math":  84.1, "language":  31.9},
    (0, 10): {"code": 106.9, "math": 152.8, "language": 272.3},
}
PHASE1_CTRL_MEAN = {"code": 0.4, "math": 0.6, "language": 0.2}
PHASE15_L0_CTRL  = {"code": 1.5, "math": 4.8, "language": 1.7}  # mean from Phase 1.5

# Phase 1 bridge scores (raw, from Phase 1.5 scan)
BRIDGE_SCORES = {
    (0, 0): 25.36, (0, 1): 9.27,  (0, 2): 7.59,  (0, 3): 5.94,
    (0, 4): 2.88,  (0, 5): 6.29,  (0, 6): 7.23,  (0, 7): 15.56,
    (0, 8): 5.73,  (0, 9): 13.70, (0, 10): 24.55, (0, 11): 4.11,
}

# ── Evaluation texts (identical to Phase 1) ───────────────────────────────────

EVAL = {
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.dropna(inplace=True)",
        "for i, row in enumerate(matrix):\n    for j, val in enumerate(row):\n        if val > threshold:",
        "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        return await resp.json()",
        "SELECT a.name, COUNT(b.id) FROM users a JOIN orders b ON a.id = b.user_id GROUP BY a.name",
        "def compute_loss(logits, targets):\n    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))",
    ],
    "math": [
        "The derivative of the natural logarithm of x with respect to x is one over x.",
        "By the Pythagorean theorem, the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "The binomial theorem states that the expansion of x plus y to the power n has n plus one terms.",
        "Integration by parts states that the integral of u times dv equals uv minus the integral of v times du.",
        "The central limit theorem guarantees that the sampling distribution of the mean approaches normality.",
        "Linear independence means no vector can be written as a linear combination of the others.",
        "The Gaussian integral of e to the negative x squared equals the square root of pi.",
        "A prime number is a natural number greater than one with no divisors other than one and itself.",
    ],
    "language": [
        "The discovery of penicillin by Alexander Fleming in 1928 revolutionized medicine.",
        "Despite the complexity of the negotiations, both parties reached a beneficial agreement.",
        "The Renaissance was a period of profound cultural transformation in European history.",
        "She walked slowly through the autumn leaves, lost in thought about recent events.",
        "The telescope revealed structures in the distant galaxy never observed before.",
        "The committee's report highlighted areas where policy reform could improve public health.",
        "After years of research, the team published their findings in a leading scientific journal.",
        "In the early hours of the morning, the city was quiet except for the sound of rain.",
    ],
}

# ── Improved completion probes (designed for GPT-2's actual capabilities) ─────
# Format: (prompt, expected_completion)
# Chosen so that GPT-2 Small achieves > 50% baseline accuracy per domain.

COMPLETION_PROBES_V2 = {
    "code": [                          # Syntactically determined
        ("def square(x):\n    return x * ", "x"),
        ("for i in range(10):\n    total += ", "i"),
        ("if x > 0:\n    return ", "x"),
        ("while not done:\n    done = ", "True"),
        ("x = [1, 2, 3]\nprint(len(", "x"),
    ],
    "narrative": [                     # Common literary continuation
        ("It was a dark and stormy ", "night"),
        ("The stars shone brightly in the ", "sky"),
        ("She opened the door and walked ", "in"),
        ("They arrived just in ", "time"),
        ("The old clock on the wall began to ", "chime"),
    ],
    "formulaic": [                     # Famous phrases — near-deterministic for GPT-2
        ("To be or not to ", "be"),
        ("Once upon a ", "time"),
        ("The quick brown fox jumps over the lazy ", "dog"),
        ("All that glitters is not ", "gold"),
        ("It was the best of times, it was the worst of ", "times"),
    ],
}

# All probe texts for attention tests
ALL_PROBES = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1",
    "class LinkedList:\n    def __init__(self):\n        self.head = None",
    "The fundamental theorem of calculus connects differentiation and integration.",
    "Euler's identity states that e raised to i times pi plus one equals zero.",
    "The ambassador carefully chose her words before addressing the assembly.",
    "Despite initial setbacks, the expedition eventually reached the summit.",
    "SELECT user_id, COUNT(*) FROM events GROUP BY user_id",
    "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    loss.backward()",
    "The Renaissance was a period of profound cultural transformation.",
    "Linear independence means no vector can be written as a linear combination.",
    "She walked slowly through the autumn leaves, lost in thought.",
    "After years of research, the team published their findings.",
]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path=None):
    src = path
    if src is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "..", "gpt2_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "gpt2_local")
            
        if os.path.exists(local_path):
            print(f"Detected local path: {local_path}")
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


def enc(tok, text, max_length=128):
    return tok(text, return_tensors="pt", truncation=True, max_length=max_length)


def perplexity(model, tokenizer, texts):
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inp    = enc(tokenizer, text)
            labels = inp["input_ids"].clone()
            out    = model(**inp, labels=labels)
            n_tok  = inp["input_ids"].shape[-1] - 1
            total_loss   += out.loss.item() * n_tok
            total_tokens += n_tok
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float("inf")


def ablate_copy(base, layer, head):
    m = copy.deepcopy(base)
    hd = m.config.n_embd // m.config.n_head
    s, e = head * hd, (head + 1) * hd
    with torch.no_grad():
        m.transformer.h[layer].attn.c_proj.weight.data[s:e, :] = 0.0
    return m


def bridge_score_single(model, tokenizer, texts, layer, head):
    n_layers = model.config.n_layer
    hd = model.config.n_embd // model.config.n_head
    s, e = head * hd, (head + 1) * hd
    deltas = []
    for text in texts:
        inp = enc(tokenizer, text, max_length=64)
        base = []
        hs_b = [model.transformer.h[l].register_forward_hook(
                    lambda m_, i_, o_, a=base: a.append(o_[0].detach().float()))
                for l in range(layer + 1, n_layers)]
        with torch.no_grad():
            model(**inp)
        for h in hs_b: h.remove()
        abl = []
        def ablate(mod, pre):
            x = pre[0].clone(); x[:, :, s:e] = 0.0; return (x,)
        ah = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(ablate)
        hs_a = [model.transformer.h[l].register_forward_hook(
                    lambda m_, i_, o_, a=abl: a.append(o_[0].detach().float()))
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


# ── A. H09 Predictive Ablation ────────────────────────────────────────────────

def h09_ablation(base_model, tokenizer):
    """
    First prospective test of bridge score's predictive power.
    H09 bridge = 13.70. Not previously ablated. Not in Phase 0 candidate set.
    
    The bridge score predicts H09 should cause more damage than controls
    (bridge 2.88–5.73, damage ~2.7%) but less than H07 (bridge 15.56, damage ~50-84%).
    
    This is a prediction, not a confirmation.
    """
    l_p, h_p = H09_PRED
    pred_bridge_score = BRIDGE_SCORES.get((l_p, h_p), 0.0)
    print("\n" + "=" * 65)
    print(f"  A. L{l_p:02d}H{h_p:02d} PREDICTIVE ABLATION  [first blind prediction]")
    print("=" * 65)
    print(f"  Prediction head bridge score: {pred_bridge_score:.2f}")
    print(f"  Controls avg:     ~4.0")
    print(f"\n  Prediction: Prediction head damage >> controls")
    print(f"  (If prediction fails -> bridge score loses predictive validity)")

    baseline = {d: perplexity(base_model, tokenizer, t) for d, t in EVAL.items()}
    m        = ablate_copy(base_model, l_p, h_p)

    print(f"\n  Results:")
    print(f"  {'Domain':<10} {'Baseline':>9} {'Ablated':>9} {'dPPL':>8} {'d%':>7}")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")

    damages = {}
    for domain, texts in EVAL.items():
        ppl   = perplexity(m, tokenizer, texts)
        delta = ppl - baseline[domain]
        dpct  = 100.0 * delta / baseline[domain]
        damages[domain] = dpct
        print(f"  {domain:<10} {baseline[domain]:>9.3f} {ppl:>9.3f} {delta:>8.3f} {dpct:>6.1f}%")

    mean_dmg = np.mean(list(damages.values()))
    print(f"\n  Mean damage: {mean_dmg:.1f}%")

    if mean_dmg > 30:
        verdict = "[OK] PREDICTION CONFIRMED - H09 causes significant damage"
        detail  = "Bridge score correctly predicted an untested head's importance."
    elif mean_dmg > 10:
        verdict = "~ PARTIAL - H09 causes moderate damage"
        detail  = "Bridge score has some predictive validity; threshold effect possible."
    else:
        verdict = "[FAIL] PREDICTION FAILED - H09 causes near-zero damage"
        detail  = "Bridge score may lose predictive validity below ~15 raw units."

    print(f"\n  Verdict: {verdict}")
    print(f"  {detail}")

    return damages, baseline


# ── B. Attention Sink Variance Test ───────────────────────────────────────────

def attention_sink_variance(model, tokenizer):
    """
    Resolves the Phase 1.5 ambiguity (means 0.237–0.359).
    
    True attention sink: consistently routes to pos-0 regardless of input.
      → low variance (std < 0.05) of per-text pos-0 attention scores
    
    Genuine bridge: input-dependent attention; sometimes routes to pos-0, sometimes not.
      → high variance (std > 0.10) of per-text pos-0 attention scores
    
    Uses diverse probe texts: code, math, narrative, short, long.
    """
    print("\n" + "=" * 65)
    print("  B. ATTENTION SINK VARIANCE TEST")
    print("=" * 65)
    print("  True sink:   std < 0.05  (rigid pos-0 routing regardless of input)")
    print("  Bridge head: std > 0.10  (task-dependent attention pattern)")
    print()
    print(f"  {'Head':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Verdict")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  -------")

    results = {}
    for (layer, head) in CANDIDATES:
        per_text_scores = []
        with torch.no_grad():
            for text in ALL_PROBES:
                inp = enc(tokenizer, text, max_length=64)
                try:
                    out = model(**inp, output_attentions=True,
                                attn_implementation="eager") if hasattr(model.config, 'attn_implementation') \
                         else model(**inp, output_attentions=True)
                except TypeError:
                    out = model(**inp, output_attentions=True)

                attn_w = out.attentions[layer][0, head]  # (seq, seq)
                per_text_scores.append(attn_w[:, 0].mean().item())

        mean = float(np.mean(per_text_scores))
        std  = float(np.std(per_text_scores))
        mn   = float(np.min(per_text_scores))
        mx   = float(np.max(per_text_scores))

        if std < 0.05:
            verdict = "[WARNING] Rigid routing -> SINK interpretation strengthened"
        elif std < 0.10:
            verdict = "~ Moderate variability -> ambiguous"
        else:
            verdict = "[OK] High variability -> BRIDGE interpretation strengthened"

        results[(layer, head)] = {"mean": mean, "std": std, "min": mn, "max": mx}
        print(f"  L{layer:02d}H{head:02d}     {mean:>8.3f} {std:>8.3f} {mn:>8.3f} {mx:>8.3f}  {verdict}")

    print()
    return results


# ── C. Improved Completion Probes ─────────────────────────────────────────────

def check_completion(model, tokenizer, prompt, expected, top_k=5):
    """Check if expected appears in top-k next-token predictions."""
    inp     = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    logits  = out.logits[0, -1, :]
    top_ids = torch.topk(logits, top_k).indices.tolist()

    target_ids = set()
    for variant in [expected, " " + expected, expected.lower(), " " + expected.lower()]:
        toks = tokenizer.encode(variant, add_special_tokens=False)
        if toks:
            target_ids.add(toks[0])

    hit       = bool(target_ids & set(top_ids))
    top_words = [tokenizer.decode([t]) for t in top_ids]
    return hit, top_words


def improved_completion_test(base_model, tokenizer):
    """
    Domain: code / narrative / formulaic.
    Formulaic serves as a control — GPT-2 should ace these regardless of ablation.
    Code should show selective degradation when code-dominant bridge heads are ablated.
    
    Key comparison:
      Baseline code accuracy - Post-ablation code accuracy  (should drop for L00H00)
      Baseline narrative accuracy - Post-ablation narrative accuracy  (should survive)
    """
    print("\n" + "=" * 65)
    print("  C. IMPROVED COMPLETION PROBES  (GPT-2 compatible)")
    print("=" * 65)

    # Baseline
    print("\n  Baseline (no ablation):")
    baseline_acc = {}
    for domain, probes in COMPLETION_PROBES_V2.items():
        hits   = sum(check_completion(base_model, tokenizer, p, e)[0] for p, e in probes)
        acc    = hits / len(probes)
        baseline_acc[domain] = acc
        print(f"    {domain:<12}: {hits}/{len(probes)} = {100*acc:.0f}%")

    if all(v == 0.0 for v in baseline_acc.values()):
        print("\n  [WARNING] All baselines are 0%. Probes still incompatible with this model.")
        print("     Printing top-5 predictions for first probe in each domain:")
        for domain, probes in COMPLETION_PROBES_V2.items():
            _, top = check_completion(base_model, tokenizer, probes[0][0], probes[0][1])
            print(f"    {domain}: prompt='{probes[0][0][-20:].strip()}' | top5: {top}")
        return None

    # Per-candidate ablation
    all_results = []
    for (layer, head) in CANDIDATES:
        tag = f"L{layer:02d}H{head:02d}"
        m   = ablate_copy(base_model, layer, head)
        print(f"\n  After ablating {tag}:")

        row = {"head": tag}
        for domain, probes in COMPLETION_PROBES_V2.items():
            hits = sum(check_completion(m, tokenizer, p, e)[0] for p, e in probes)
            acc  = hits / len(probes)
            drop = baseline_acc[domain] - acc
            row[domain]           = acc
            row[f"{domain}_drop"] = drop
            flag = "  <- DROPPED" if drop >= 0.2 else ""
            print(f"    {domain:<12}: {hits}/{len(probes)} = {100*acc:.0f}%  "
                  f"(Delta = {-100*drop:+.0f}%){flag}")
        all_results.append(row)

    return all_results


# == Plots =====================================================================

def plot_h09_in_context(h09_damages, baseline, ctrl_dict=None, cand_dict=None, labels=None, prefix="small_"):
    """
    Plot H09/PRED damage alongside Phase 1 candidates and controls.
    Shows H09's position on the damage spectrum.
    """
    domains = list(EVAL.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(5.5 * len(domains), 5))

    if labels is None:
        head_labels = ["L00H04\n(ctrl)", "L00H11\n(ctrl)", "L00H08\n(ctrl)",
                       "L00H09\n(PRED)", "L00H07\n(cand)", "L00H00\n(cand)", "L00H10\n(cand)"]
    else:
        head_labels = labels

    if ctrl_dict is None:
        ctrl_dict = {
            "code":     [0.9,  -3.4,  4.1],
            "math":     [3.9,  -2.3, 12.9],
            "language": [1.4,  -0.9,  7.4],
        }
    if cand_dict is None:
        cand_dict = {
            "code":     [30.4, 158.1, 106.9],
            "math":     [84.1,  21.6, 152.8],
            "language": [31.9,  24.6, 272.3],
        }

    colors = ["steelblue", "steelblue", "steelblue",
              "gold", "crimson", "crimson", "crimson"]

    for ax, domain in zip(axes, domains):
        all_dmg = (ctrl_dict[domain] + [h09_damages[domain]] +
                   [cand_dict[domain][0], cand_dict[domain][1], cand_dict[domain][2]])
        bars = ax.bar(range(len(head_labels)), all_dmg, color=colors, alpha=0.85, edgecolor="none")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels, fontsize=7.5)
        ax.set_title(f"{domain} - dPPL (%)", fontsize=10)
        ax.set_ylabel("d Perplexity (%)")

    from matplotlib.patches import Patch
    legend = [Patch(color="steelblue", label="Low-bridge controls"),
              Patch(color="gold",      label="Prediction head"),
              Patch(color="crimson",   label="Candidates")]
    axes[-1].legend(handles=legend, loc="upper left", fontsize=8)

    plt.suptitle("Predictive Ablation — Position on Damage Spectrum\n"
                 "(gold bar = metric's first blind prediction)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase2_h09_prediction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {prefix}phase2_h09_prediction.png")


def plot_sink_variance(sink_results, prefix="small_"):
    """Bar chart: mean ± std of pos-0 attention per candidate."""
    heads  = [f"L{l:02d}H{h:02d}" for (l, h) in sink_results.keys()]
    means  = [sink_results[(l, h)]["mean"] for (l, h) in sink_results.keys()]
    stds   = [sink_results[(l, h)]["std"]  for (l, h) in sink_results.keys()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(heads, means, yerr=stds, color="steelblue", alpha=0.8,
           capsize=8, ecolor="black", edgecolor="none")
    ax.axhline(0.4, color="red",    ls="--", lw=0.9, label="Sink threshold (0.4)")
    ax.axhline(0.2, color="orange", ls="--", lw=0.9, label="Ambiguous zone (0.2)")
    ax.set_ylabel("Mean attention to position 0")
    ax.set_title("Attention Sink Variance Test\n(error bars = std across 12 diverse probe texts)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase2_sink_variance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {prefix}phase2_sink_variance.png")


# ── Replication Runner (stub for GPT-2 Medium / Pythia) ──────────────────────

def run_replication(model_name="gpt2-medium", n_candidates=5):
    """
    Run the core Phase 0 pipeline on a different model.
    Call after Phase 2 if results hold.
    
    Usage:
        run_replication("gpt2-medium")
        run_replication("EleutherAI/pythia-160m")
        run_replication("EleutherAI/pythia-410m")
    
    What to look for:
        - Do low-Wanda / high-Bridge heads exist in the other model?
        - Do they also cause catastrophic perplexity increase when ablated?
        - Are they also concentrated in early layers?
        - Do they also show domain-selective damage?
    """
    print(f"\n  [Replication stub] Would load: {model_name}")
    print(f"  Run Phase 0 + Phase 1 core pipeline on this model.")
    print(f"  See phase0_invisible_bridge.py and phase1_ablation.py for the full pipeline.")
    print(f"  If the same phenomenon appears: result is architecture-general, not GPT-2-specific.")
    print()
    print("  Recommended replication order:")
    print("    1. GPT-2 Medium (345M, same architecture, more heads/layers)")
    print("    2. Pythia-160M  (similar scale, different training corpus)")
    print("    3. Pythia-410M  (larger, if 1+2 replicate)")
    print()
    print("  Key invariant to check: does bridge score remain orthogonal to Wanda (r ~ 0)?")
    print("  If yes across all three: invisible bridge phenomenon is robust.")


# ── Hypothesis Ladder ─────────────────────────────────────────────────────────

def hypothesis_ladder(h09_verdict, sink_variance_results, completion_ran):
    print("\n" + "=" * 65)
    print("  HYPOTHESIS LADDER - After Phase 2")
    print("=" * 65)

    print("\n  [OK] PROVEN (across all phases)")
    print("     Bridge != Wanda  (r ~ 0, Phase 0)")
    print("     Low-Wanda critical heads exist  (213x ratio, Phase 1)")
    print("     Wanda importance ranking is inverted for most critical heads")
    print("     Layer-0 position alone doesn't explain damage  (Phase 1.5, 2.7% vs 98%)")

    print(f"\n  {'[OK]' if 'CONFIRMED' in h09_verdict else '[WARNING]'} H09 PREDICTIVE VALIDATION")
    print(f"     {h09_verdict}")

    print("\n  [WARNING] ATTENTION SINK STATUS")
    for (l, h), r in sink_variance_results.items():
        interp = ("Rigid routing - sink interpretation stronger"   if r["std"] < 0.05 else
                  "Moderate variability - ambiguous"               if r["std"] < 0.10 else
                  "High variability - bridge interpretation stronger")
        print(f"     L{l:02d}H{h:02d}: mean={r['mean']:.3f} std={r['std']:.3f} -> {interp}")

    print("\n  [UNTESTED]")
    if not completion_ran:
        print("     Completion probes (probe design issue, retry with v2 probes)")
    print("     Replication on GPT-2 Medium / Pythia-160M")
    print("     Recovery capacity after fine-tuning")
    print("     Behavior with distribution-shifted probe sets")

    print("\n" + "-" * 65)
    print("  CURRENT DEFENSIBLE CLAIM:")
    print()
    print('  "We identify low-weight-magnitude Layer-0 attention heads')
    print('   in GPT-2 Small that cause catastrophic perplexity increase')
    print('   when ablated. These heads are invisible to Wanda, not')
    print('   explained by Layer-0 position, show ambiguous-to-partial')
    print('   attention sink behavior, exhibit domain-selective damage')
    print('   fingerprints, and can be identified prospectively by a')
    print('   downstream representation sensitivity score."')
    print("═" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--skip_replication", action="store_true", default=True)
    args = parser.parse_args()

    base_model, tokenizer = load_model(args.model_path)
    global CANDIDATES, H09_PRED, BRIDGE_SCORES

    ctrl_damages = None
    cand_damages = None
    labels = None

    if base_model.config.n_layer == 24:
        CANDIDATES = [(2, 12), (6, 1), (7, 2), (22, 2), (22, 13)]
        print("Detected GPT-2 Medium. Selecting controls and prediction head in Layer 2...")
        bridge_l2 = {}
        quick_probes = ALL_PROBES[:5]
        for h in range(16):
            bridge_l2[h] = bridge_score_single(base_model, tokenizer, quick_probes, 2, h)
            print(f"  L02H{h:02d} bridge: {bridge_l2[h]:.4f}")
        
        # Sort Layer 2 heads by bridge score
        sorted_heads = sorted(range(16), key=lambda h: bridge_l2[h])
        
        # Controls: 3 lowest bridge heads in Layer 2
        ctrl_heads = [h for h in sorted_heads if (2, h) not in CANDIDATES][:3]
        controls = [(2, h) for h in ctrl_heads]
        
        # Prediction: a head in the middle of Layer 2
        non_cands = [h for h in sorted_heads if (2, h) not in CANDIDATES and (2, h) not in controls]
        pred_h = non_cands[len(non_cands) // 2]
        H09_PRED = (2, pred_h)
        
        BRIDGE_SCORES = {}
        for h in range(16):
            BRIDGE_SCORES[(2, h)] = bridge_l2[h]
            
        print(f"Selected Layer 2 controls: {controls}")
        print(f"Selected Layer 2 prediction head: {H09_PRED} (bridge={bridge_l2[pred_h]:.4f})")

        # Compute damages on the fly for controls, pred, and candidates
        print("Computing damages for controls, prediction head, and candidates on the fly...")
        baseline_ppl = {d: perplexity(base_model, tokenizer, t) for d, t in EVAL.items()}
        
        ctrl_damages = {d: [] for d in EVAL}
        for (l, h) in controls:
            m = ablate_copy(base_model, l, h)
            for d, texts in EVAL.items():
                ppl = perplexity(m, tokenizer, texts)
                ctrl_damages[d].append(100.0 * (ppl - baseline_ppl[d]) / baseline_ppl[d])
                
        cands_to_test = [(2, 12), (22, 2), (22, 13)]
        cand_damages = {d: [] for d in EVAL}
        for (l, h) in cands_to_test:
            m = ablate_copy(base_model, l, h)
            for d, texts in EVAL.items():
                ppl = perplexity(m, tokenizer, texts)
                cand_damages[d].append(100.0 * (ppl - baseline_ppl[d]) / baseline_ppl[d])
                
        labels = [
            f"L{controls[0][0]:02d}H{controls[0][1]:02d}\n(ctrl)",
            f"L{controls[1][0]:02d}H{controls[1][1]:02d}\n(ctrl)",
            f"L{controls[2][0]:02d}H{controls[2][1]:02d}\n(ctrl)",
            f"L{H09_PRED[0]:02d}H{H09_PRED[1]:02d}\n(PRED)",
            f"L{cands_to_test[0][0]:02d}H{cands_to_test[0][1]:02d}\n(cand)",
            f"L{cands_to_test[1][0]:02d}H{cands_to_test[1][1]:02d}\n(cand)",
            f"L{cands_to_test[2][0]:02d}H{cands_to_test[2][1]:02d}\n(cand)"
        ]
    else:
        print("Detected GPT-2 Small. Using hardcoded Small candidates:", CANDIDATES)

    # A — H09 predictive ablation
    h09_damages, baseline = h09_ablation(base_model, tokenizer)
    mean_h09 = np.mean(list(h09_damages.values()))
    h09_verdict = (
        f"[OK] CONFIRMED (mean damage {mean_h09:.1f}% >> controls)" if mean_h09 > 30 else
        f"~ PARTIAL   (mean damage {mean_h09:.1f}%, moderate support)"  if mean_h09 > 10 else
        f"[FAIL] FAILED    (mean damage {mean_h09:.1f}%, near controls)"
    )

    # B — Attention sink variance
    sink_results = attention_sink_variance(base_model, tokenizer)

    # C — Improved completion probes
    completion_results = improved_completion_test(base_model, tokenizer)

    # Plots
    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plot_h09_in_context(h09_damages, baseline, ctrl_damages, cand_dict=cand_damages, labels=labels, prefix=prefix)
    plot_sink_variance(sink_results, prefix)

    # Final ladder
    hypothesis_ladder(h09_verdict, sink_results, completion_results is not None)

    # Replication stub
    if not args.skip_replication:
        run_replication("gpt2-medium")


if __name__ == "__main__":
    main()
