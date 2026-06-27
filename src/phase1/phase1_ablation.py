"""
Phase 1 — Targeted Ablation Study
====================================
Model: GPT-2 Small  (GPT2LMHeadModel for perplexity)

Experimental groups
-------------------
  A  Invisible Bridges    low Wanda + high Bridge  (Phase 0 candidates)
  B  Wanda-matched ctrl   similar Wanda,  low Bridge  (Layer 1+ heads)
  C  High-Wanda heads     top Wanda scores  (sanity check, expected high damage)
  D  All bridges combined  cumulative effect

Core prediction
---------------
  damage(A) >> damage(B)   [same Wanda magnitude, very different damage]
  This validates bridge score as containing information Wanda misses.

Layer-bias check
----------------
  All Phase 0 candidates were in Layer 0.
  Two hypotheses:
    H1: Layer 0 heads are genuinely foundational (low weight, high structural role)
    H2: Bridge score is biased toward early layers (more downstream layers to disrupt)
  This script normalizes bridge score by downstream layer count to distinguish them.

Usage
-----
  pip install torch transformers matplotlib numpy
  python phase1_ablation.py
  # or with local weights:
  # python phase1_ablation.py --model_path C:/path/to/gpt2_local
"""

import argparse
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Phase 0 results (hardcoded from our run) ──────────────────────────────────

ALL_CANDIDATE_HEADS = [(0, 0), (0, 7), (0, 10)]   # layer, head

# Domain-specific bridge candidates for selective damage prediction
DOMAIN_CANDIDATES = {
    "code":     [(0, 0), (0, 10)],
    "math":     [(0, 0), (0, 7), (0, 10)],
    "language": [(0, 7), (0, 10)],
}

# Phase 0 normalized Wanda values (for control matching)
CANDIDATE_WANDA_NORM = {
    (0, 0):  0.026,
    (0, 7):  0.047,
    (0, 10): 0.103,
}


# ── Evaluation sets ───────────────────────────────────────────────────────────

EVAL = {
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.dropna(inplace=True)\nprint(df.describe())",
        "for i, row in enumerate(matrix):\n    for j, val in enumerate(row):\n        if val > threshold:\n            result.append((i, j))",
        "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.json()",
        "SELECT a.name, COUNT(b.id) as total FROM users a\nJOIN orders b ON a.id = b.user_id\nGROUP BY a.name HAVING total > 5",
        "def compute_loss(logits, targets):\n    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))",
    ],
    "math": [
        "The derivative of the natural logarithm of x with respect to x is one over x.",
        "By the Pythagorean theorem, the square of the hypotenuse equals the sum of the squares of the other two sides.",
        "The binomial theorem states that the expansion of x plus y to the power n has n plus one terms.",
        "Integration by parts states that the integral of u times dv equals uv minus the integral of v times du.",
        "The central limit theorem guarantees that the sampling distribution of the mean approaches normality as sample size increases.",
        "Linear independence means no vector in the set can be written as a linear combination of the others.",
        "The Gaussian integral of e to the negative x squared from negative infinity to infinity equals the square root of pi.",
        "A prime number is a natural number greater than one with no positive divisors other than one and itself.",
    ],
    "language": [
        "The discovery of penicillin by Alexander Fleming in 1928 revolutionized the treatment of bacterial infections.",
        "Despite the complexity of the negotiations, the two parties eventually reached a mutually beneficial agreement.",
        "The Renaissance was a period of profound cultural and intellectual transformation in European history.",
        "She walked slowly through the autumn leaves, lost in thought about the events of the past several months.",
        "The telescope revealed structures in the distant galaxy that had never been observed by astronomers before.",
        "The committee's report highlighted several areas where policy reform could improve public health outcomes.",
        "After years of research, the team finally published their findings in a leading peer-reviewed scientific journal.",
        "In the early hours of the morning, the city was quiet except for the distant sound of rain on the pavement.",
    ],
}


# ── Model loading ─────────────────────────────────────────────────────────────

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
            print(f"Detected local path: {local_path}")
            src = local_path
        else:
            src = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(src)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(src)
    model.eval()
    model_name = "GPT-2 Medium" if model.config.n_layer == 24 else "GPT-2 Small"
    print(f"Loading {model_name} from: {src}")
    return model, tokenizer


def enc(tokenizer, text, max_length=128):
    return tokenizer(text, return_tensors="pt",
                     truncation=True, max_length=max_length)


# ── Wanda score (for control selection) ───────────────────────────────────────

def compute_wanda(model, tokenizer, texts):
    """Per-head Wanda score = input_norm × weight_magnitude at c_proj."""
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    head_dim = model.config.n_embd // n_heads
    captured = defaultdict(list)

    def make_hook(l):
        def hook(mod, inp, out):
            captured[l].append(inp[0].detach())
        return hook

    hooks = [model.transformer.h[l].attn.c_proj.register_forward_hook(make_hook(l))
             for l in range(n_layers)]
    with torch.no_grad():
        for t in texts:
            model(**enc(tokenizer, t, max_length=64))
    for h in hooks:
        h.remove()

    wanda = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        W = model.transformer.h[l].attn.c_proj.weight
        for head in range(n_heads):
            s, e = head * head_dim, (head + 1) * head_dim
            # Average the norms of the head slice over sequence positions across all texts
            norms = [x[:, :, s:e].norm(dim=-1).mean().item() for x in captured[l]]
            input_norm = float(np.mean(norms))
            weight_mag = W[s:e, :].abs().mean().item()
            wanda[l, head] = input_norm * weight_mag
    return wanda


# ── Bridge score (for layer-bias check) ───────────────────────────────────────

def bridge_score(model, tokenizer, texts, layer, head):
    """
    Downstream L2 shift when head (layer, head) is ablated via c_proj pre-hook.
    Raw score: total disruption across all downstream layers.
    Normalized score: divide by number of downstream layers (layer-bias correction).
    """
    n_layers = model.config.n_layer
    head_dim = model.config.n_embd // model.config.n_head
    s, e     = head * head_dim, (head + 1) * head_dim
    deltas   = []

    for text in texts:
        inp = enc(tokenizer, text, max_length=64)

        base = []
        hs_b = [model.transformer.h[l].register_forward_hook(
                    lambda m, i, o, a=base: a.append(o[0].detach().float()))
                for l in range(layer + 1, n_layers)]
        with torch.no_grad():
            model(**inp)
        for h in hs_b:
            h.remove()

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
        for h in hs_a:
            h.remove()

        if base and abl:
            B = torch.stack([t.mean(dim=1) for t in base])
            A = torch.stack([t.mean(dim=1) for t in abl])
            deltas.append((B - A).norm(dim=-1).mean().item())

    raw        = float(np.mean(deltas)) if deltas else 0.0
    downstream = max(n_layers - 1 - layer, 1)
    return raw, raw / downstream   # (raw, layer-normalized)


# ── Ablation ──────────────────────────────────────────────────────────────────

def ablate_heads(base_model, head_list):
    """Deep-copy model, permanently zero each head's c_proj weight slice."""
    m        = copy.deepcopy(base_model)
    head_dim = m.config.n_embd // m.config.n_head
    for (layer, head) in head_list:
        s, e = head * head_dim, (head + 1) * head_dim
        with torch.no_grad():
            m.transformer.h[layer].attn.c_proj.weight.data[s:e, :] = 0.0
    return m


# ── Perplexity ────────────────────────────────────────────────────────────────

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
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(base_model, tokenizer, controls, high_wanda_heads):
    all_texts = [t for v in EVAL.values() for t in v]

    # Baseline
    print("\nBaseline perplexity...")
    baseline = {d: perplexity(base_model, tokenizer, texts) for d, texts in EVAL.items()}
    for d, v in baseline.items():
        print(f"  {d}: {v:.3f}")

    rows = []

    def record(label, group, head_list):
        m = ablate_heads(base_model, head_list)
        for domain, texts in EVAL.items():
            ppl   = perplexity(m, tokenizer, texts)
            delta = ppl - baseline[domain]
            rows.append({
                'label':     label,
                'group':     group,
                'domain':    domain,
                'baseline':  baseline[domain],
                'ablated':   ppl,
                'delta':     delta,
                'delta_pct': 100.0 * delta / baseline[domain],
            })

    # Group A — invisible bridges (one at a time)
    print("\n[A] Invisible bridge ablations...")
    for (l, h) in ALL_CANDIDATE_HEADS:
        tag = f"L{l:02d}H{h:02d}"
        print(f"  Ablating InvBridge {tag}...")
        record(f"InvBridge {tag}", "A", [(l, h)])

    # Group B — wanda-matched controls (one at a time)
    print("\n[B] Wanda-matched control ablations...")
    for (l, h) in controls:
        tag = f"L{l:02d}H{h:02d}"
        print(f"  Ablating WandaCtrl {tag}...")
        record(f"WandaCtrl {tag}", "B", [(l, h)])

    # Group C — high-Wanda heads (one at a time)
    print("\n[C] High-Wanda ablations...")
    for (l, h) in high_wanda_heads:
        tag = f"L{l:02d}H{h:02d}"
        print(f"  Ablating HighWanda {tag}...")
        record(f"HighWanda {tag}", "C", [(l, h)])

    # Group D — all invisible bridges together
    print("\n[D] All invisible bridges simultaneously...")
    record("AllBridges", "D", ALL_CANDIDATE_HEADS)

    return rows, baseline


# ── Layer-bias check ──────────────────────────────────────────────────────────

def layer_bias_check(base_model, tokenizer):
    """
    Recompute bridge score for candidate heads only.
    Compare raw vs layer-normalized scores.
    If normalized scores remain high → H1 (genuine bridge, not artifact).
    If normalized scores are unremarkable → H2 (layer-bias artifact).
    """
    probe_texts = [t for v in list(EVAL.values()) for t in v[:3]]   # quick subset
    print("\n-- Layer-bias check ------------------------------------------")
    print(f"  {'Head':<10} {'Layer':>6} {'DownstreamL':>12} {'Raw':>10} {'Normalized':>12}  Verdict")
    print(f"  {'-'*10} {'-'*6} {'-'*12} {'-'*10} {'-'*12}  -------")

    n_layers = base_model.config.n_layer
    results  = []
    for (l, h) in ALL_CANDIDATE_HEADS:
        raw, normd = bridge_score(base_model, tokenizer, probe_texts, l, h)
        n_ds = n_layers - 1 - l
        verdict = "H1 likely (genuine)" if normd > raw * 0.15 else "H2 possible (check)"
        print(f"  L{l:02d}H{h:02d}     {l:>6} {n_ds:>12} {raw:>10.4f} {normd:>12.4f}  {verdict}")
        results.append((l, h, raw, normd, n_ds))

    print()
    return results


# ── Results table ─────────────────────────────────────────────────────────────

def print_table(rows, baseline):
    print("\n" + "=" * 75)
    print("  PHASE 1  -  Delta Perplexity (dPPL vs baseline)")
    print("  Prediction: damage(A) >> damage(B)  despite matched Wanda")
    print("=" * 75)

    group_cfg = [
        ("A", "Invisible Bridges  (low Wanda, high Bridge)"),
        ("B", "Wanda-matched ctrl (low Wanda, low Bridge)"),
        ("C", "High-Wanda heads   (high Wanda - sanity check)"),
        ("D", "All Bridges combined"),
    ]

    for g_id, g_name in group_cfg:
        g_rows = [r for r in rows if r["group"] == g_id]
        if not g_rows:
            continue
        print(f"\n  Group {g_id}: {g_name}")
        print(f"  {'Head':<22} {'Domain':<10} {'Baseline':>9} {'Ablated':>9} {'dPPL':>8} {'d%':>7}")
        print(f"  {'-'*22} {'-'*10} {'-'*9} {'-'*9} {'-'*8} {'-'*7}")
        for r in sorted(g_rows, key=lambda x: (x["label"], x["domain"])):
            print(f"  {r['label']:<22} {r['domain']:<10} "
                  f"{r['baseline']:>9.2f} {r['ablated']:>9.2f} "
                  f"{r['delta']:>8.2f} {r['delta_pct']:>6.1f}%")

    print("\n" + "=" * 75)

    a_deltas = [r["delta"] for r in rows if r["group"] == "A"]
    b_deltas = [r["delta"] for r in rows if r["group"] == "B"]

    if a_deltas and b_deltas:
        mean_a = np.mean(a_deltas)
        mean_b = np.mean(b_deltas)
        ratio  = mean_a / (mean_b + 1e-6)
        verdict = (
            "[OK] SUPPORTED    (bridges cause disproportionate damage)"   if ratio > 2.0 else
            "~  MARGINAL     (some signal but weak)"                     if ratio > 1.2 else
            "[X] NOT SUPPORTED (bridge score doesn't predict damage)"
        )
        print(f"\n  KEY RESULT:")
        print(f"    Mean dPPL - Invisible Bridges:  {mean_a:+.3f}")
        print(f"    Mean dPPL - Wanda-matched ctrl: {mean_b:+.3f}")
        print(f"    Damage ratio  (A / B):          {ratio:.2f}x")
        print(f"    Verdict:  {verdict}")

    print("=" * 75)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(rows, prefix="small_"):
    domains     = list(EVAL.keys())
    group_cfg   = [("A", "Invisible Bridges", "crimson"),
                   ("B", "Wanda Controls",    "steelblue"),
                   ("C", "High-Wanda",        "darkorange"),
                   ("D", "All Bridges",       "purple")]

    fig, axes = plt.subplots(1, len(domains), figsize=(5.5 * len(domains), 5))
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        x_pos  = []
        x_tick = []
        offset = 0

        for g_id, g_label, color in group_cfg:
            g_rows = [r for r in rows if r["group"] == g_id and r["domain"] == domain]
            if not g_rows:
                continue
            deltas = [r["delta_pct"] for r in g_rows]
            mean   = np.mean(deltas)
            std    = np.std(deltas)
            ax.bar(offset, mean, yerr=std if std > 0 else None,
                   color=color, alpha=0.82, capsize=5, width=0.7, label=g_label)
            # Individual points
            for d in deltas:
                ax.scatter(offset, d, color=color, s=25, zorder=3,
                           edgecolors='white', linewidths=0.5, alpha=0.8)
            x_pos.append(offset)
            x_tick.append(g_label)
            offset += 1

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_tick, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Perplexity increase — {domain}", fontsize=10)
        ax.set_ylabel("ΔPPL (%)")

    plt.suptitle("Phase 1: Ablation Damage by Group and Domain\n"
                 "Prediction: Group A (red) >> Group B (blue) despite equal Wanda",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase1_ablation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {prefix}phase1_ablation.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local GPT-2 weights (default: download from HF)")
    args = parser.parse_args()

    base_model, tokenizer = load_model(args.model_path)
    global ALL_CANDIDATE_HEADS, DOMAIN_CANDIDATES
    if base_model.config.n_layer == 24:
        ALL_CANDIDATE_HEADS = [(2, 12), (6, 1), (7, 2), (22, 2), (22, 13)]
        DOMAIN_CANDIDATES = {
            "code":     [(2, 12)],
            "math":     [(22, 2), (22, 13), (7, 2)],
            "language": [(22, 2), (6, 1)],
        }
        print("Detected GPT-2 Medium. Using Medium candidates:", ALL_CANDIDATE_HEADS)
    else:
        print("Detected GPT-2 Small. Using Small candidates:", ALL_CANDIDATE_HEADS)

    all_texts = [t for v in EVAL.values() for t in v]

    # ── Layer-bias check ──────────────────────────────────────────────────────
    # Must run before claiming Layer 0 concentration is a genuine finding.
    layer_bias_check(base_model, tokenizer)

    # ── Select controls via Wanda ──────────────────────────────────────────────
    print("Computing Wanda scores for control selection...")
    wanda  = compute_wanda(base_model, tokenizer, all_texts)
    w_norm = (wanda - wanda.min()) / (wanda.max() - wanda.min() + 1e-9)

    # Controls: heads with Wanda similar to each candidate
    controls = []
    skip_layer_0 = (base_model.config.n_layer == 12)
    start_layer = 1 if skip_layer_0 else 0
    for (l_c, h_c) in ALL_CANDIDATE_HEADS:
        target = w_norm[l_c, h_c]
        best, best_dist = None, 999.0
        for l in range(start_layer, wanda.shape[0]):
            for h in range(wanda.shape[1]):
                if (l, h) in ALL_CANDIDATE_HEADS or (l, h) in controls:
                    continue
                dist = abs(w_norm[l, h] - target)
                if dist < best_dist:
                    best_dist = dist
                    best = (l, h)
        if best:
            controls.append(best)
            print(f"  Control for L{l_c:02d}H{h_c:02d} (Wanda={target:.3f}): "
                  f"L{best[0]:02d}H{best[1]:02d} (Wanda={w_norm[best[0], best[1]]:.3f})")

    # High-Wanda heads (top 3)
    flat = np.argsort(wanda.flatten())[::-1]
    high_wanda = [(int(i // wanda.shape[1]), int(i % wanda.shape[1])) for i in flat[:3]]
    print(f"  High-Wanda heads: {high_wanda}")

    # ── Run ablations ─────────────────────────────────────────────────────────
    rows, baseline = run_experiment(base_model, tokenizer, controls, high_wanda)

    # ── Report ────────────────────────────────────────────────────────────────
    print_table(rows, baseline)
    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plot(rows, prefix)


if __name__ == "__main__":
    main()
    # With local weights:
    # python phase1_ablation.py --model_path "C:/Users/amiku/Downloads/ExpModel/gpt2_local"
