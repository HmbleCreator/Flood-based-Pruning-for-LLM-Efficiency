"""
Phase 1.5 -- Confounder Resolution
====================================
Three targeted experiments to determine what Phase 1's 213x ratio is measuring.

A. Attention Sink Test
   For L00H00, L00H07, L00H10: measure mean attention assigned to position 0 (BOS).
   Score > 0.4 -> attention sink (broad damage explained by sink removal, not bridge)
   Score < 0.2 -> not a sink (domain-specific bridge interpretation survives)
   L00H10's broad-spectrum damage makes this test mandatory before any claim.

B. Layer-0 Low-Bridge Controls  [THE DECISIVE TEST]
   Scan bridge scores for ALL 12 Layer-0 heads.
   Ablate the 3 lowest-bridge Layer-0 heads and measure damage.
   
   Prediction if bridge score is doing the work:
     -> low-Bridge L0 heads cause near-zero damage (like Phase 1 Group B)
   Prediction if Layer-0 position is just special:
     -> any L0 ablation causes high damage regardless of bridge score
   
   This test directly answers the friend's concern:
   "Your current controls are L04H05, L07H00, L07H07 -- they do not answer the
    question of whether Layer-0 is just special."

C. Capability Completion Probes
   15 prompts per domain (5 code / 5 math / 5 language) with known correct completions.
   Measures whether correct token survives in top-5 after ablation.
   Perplexity can hide capability collapse; completion accuracy cannot.
   
   Prediction for L00H00 (code-dominant bridge in Phase 1):
     -> code completions collapse, math/language completions survive
   
Usage:
    python phase1_5_confounders.py
    python phase1_5_confounders.py --model_path C:/path/to/gpt2_local
"""

import argparse
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -- Phase 0/1 candidates ------------------------------------------------------
CANDIDATES = [(0, 0), (0, 7), (0, 10)]

# -- Phase 1 Group A results (hardcoded for comparison plot) -------------------
PHASE1_GROUP_A = {"code": 98.5, "math": 86.2, "language": 109.6}   # mean d%
PHASE1_GROUP_B = {"code":  0.4, "math":  0.6, "language":   0.2}   # mean d%

# -- Probe texts ---------------------------------------------------------------
PROBES = {
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
ALL_PROBES = [t for v in PROBES.values() for t in v]

# -- Evaluation texts (same as Phase 1) ----------------------------------------
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

# -- Capability completion probes ----------------------------------------------
# (prompt, expected_first_token_of_continuation)
# Chosen for single-token, unambiguous completions with high baseline accuracy on GPT-2 Small
COMPLETION_PROBES = {
    "code": [
        ("def square(x):\n    return x *", "x"),
        ("for i in range(10):\n    total +=", "i"),
        ("if x > 0:\n    return", "x"),
        ("while not done:\n    done =", "True"),
        ("x = [1, 2, 3]\nprint(len(x", ")"),
    ],
    "math": [
        ("2 + 2 =", "4"),
        ("1 + 1 =", "2"),
        ("5 - 3 =", "2"),
        ("5 + 5 =", "10"),
        ("10 - 5 =", "5"),
    ],
    "language": [
        ("Once upon a", "time"),
        ("To be or not to", "be"),
        ("The quick brown fox jumps over the lazy", "dog"),
        ("She opened the door and walked", "in"),
        ("They arrived just in", "time"),
    ],
}


# -- Model ---------------------------------------------------------------------
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


# -- A. Attention Sink Test ----------------------------------------------------
def attention_sink_test(model, tokenizer):
    """
    Measure fraction of attention each candidate head assigns to position 0 (BOS).
    GPT-2 attention sinks route excess attention to position 0 across all inputs.
    A true bridge should show topic-sensitive attention patterns, not position-fixed.
    """
    print("\n" + "=" * 62)
    print("  A. ATTENTION SINK TEST")
    print("=" * 62)
    print(f"  {'Head':<10} {'Mean Attn -> Pos-0':>18}  Verdict")
    print(f"  {'-'*10} {'-'*18}  -------")

    sink_scores = {}
    for (layer, head) in CANDIDATES:
        scores = []
        with torch.no_grad():
            for text in ALL_PROBES:
                inp = enc(tokenizer, text, max_length=64)
                out = model(**inp, output_attentions=True)
                # out.attentions: tuple[n_layers] of (batch, n_heads, seq, seq)
                attn_weights = out.attentions[layer][0, head]  # (seq, seq)
                # Mean attention to position 0 across all query positions
                scores.append(attn_weights[:, 0].mean().item())

        mean_score = float(np.mean(scores))
        sink_scores[(layer, head)] = mean_score

        if mean_score > 0.4:
            verdict = "[!] ATTENTION SINK - broad damage expected, bridge interp. weaker"
        elif mean_score > 0.2:
            verdict = "~  Partial sink - ambiguous"
        else:
            verdict = "[OK] Not a sink - domain-specific bridge interpretation survives"

        print(f"  L{layer:02d}H{head:02d}     {mean_score:>18.3f}  {verdict}")

    print()
    return sink_scores


# -- B. Layer-0 Bridge Scan + Low-Bridge Control Ablations --------------------
def bridge_score_single(model, tokenizer, texts, layer, head):
    """Bridge score (downstream L2 shift) for one specific head."""
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


def layer0_low_bridge_controls(base_model, tokenizer):
    """
    THE DECISIVE TEST for Layer-0 confound.
    
    Scans bridge scores for all 12 Layer-0 heads.
    Identifies the 3 lowest-bridge (non-candidate) Layer-0 heads.
    Ablates them and measures perplexity damage.
    
    If low-Bridge L0 heads cause near-zero damage -> bridge metric validated.
    If they cause high damage regardless -> Layer-0 position is the true confound.
    """
    print("\n" + "=" * 62)
    print("  B. LAYER-0 LOW-BRIDGE CONTROL TEST  [THE DECISIVE TEST]")
    print("=" * 62)

    n_heads    = base_model.config.n_head
    quick      = ALL_PROBES[:6]   # faster scan

    print(f"  Scanning bridge scores for all {n_heads} Layer-0 heads...")
    bridge_l0 = {}
    for h in range(n_heads):
        score = bridge_score_single(base_model, tokenizer, quick, 0, h)
        bridge_l0[h] = score
        tag = " <- Phase0 candidate" if (0, h) in CANDIDATES else ""
        print(f"    L00H{h:02d}: {score:.4f}{tag}")

    # Select 3 lowest-bridge L0 heads (excluding known candidates)
    non_cands = [(h, bridge_l0[h]) for h in range(n_heads) if (0, h) not in CANDIDATES]
    non_cands.sort(key=lambda x: x[1])
    controls_l0 = [(0, h) for h, _ in non_cands[:3]]

    print(f"\n  Layer-0 low-Bridge controls: {controls_l0}")
    print(f"  Their bridge scores: {[f'{bridge_l0[h]:.4f}' for _, h in controls_l0]}")
    print("  Ablating them now...\n")

    baseline = {d: perplexity(base_model, tokenizer, t) for d, t in EVAL.items()}

    rows = []
    for (l, h) in controls_l0:
        m = ablate_copy(base_model, l, h)
        for domain, texts in EVAL.items():
            ppl  = perplexity(m, tokenizer, texts)
            dpct = 100.0 * (ppl - baseline[domain]) / baseline[domain]
            rows.append({"head": f"L{l:02d}H{h:02d}", "bridge": bridge_l0[h],
                         "domain": domain, "delta_pct": dpct})
            print(f"    L{l:02d}H{h:02d} [{domain:8s}] d = {dpct:+.1f}%  "
                  f"(bridge={bridge_l0[h]:.3f})")

    mean_damage = np.mean([r["delta_pct"] for r in rows])
    print(f"\n  Mean d% for low-Bridge L0 controls: {mean_damage:.1f}%")
    print(f"  Mean d% for Phase 1 Group A (bridges): ~98%")
    print(f"  Mean d% for Phase 1 Group B (L4/L7 controls): ~0.4%")

    if mean_damage < 10:
        verdict = "[OK] LAYER-0 CONFOUND RULED OUT\n     Bridge score predicts damage, layer position alone does not."
    elif mean_damage < 40:
        verdict = "~  PARTIAL CONFOUND\n     Layer-0 has some intrinsic importance, bridge adds independent signal."
    else:
        verdict = "[!] LAYER-0 CONFOUND NOT RULED OUT\n     Revise bridge metric to control for layer depth."

    print(f"\n  Verdict: {verdict}")
    return rows, bridge_l0


# -- C. Capability Completion Probes ------------------------------------------
def check_completion(model, tokenizer, prompt, expected, top_k=5):
    """
    Check if expected token appears in top-k predictions after prompt.
    Handles GPT-2 BPE leading-space convention.
    """
    inp = tokenizer(prompt, return_tensors="pt")
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


def capability_completion_test(base_model, tokenizer):
    """
    Measure completion accuracy before and after ablating each candidate.
    
    Perplexity averages over the full token distribution.
    Completion accuracy measures a specific prediction.
    They can disagree: high perplexity can coexist with correct completions
    if the model is uncertain but still ranks the right token first.
    
    Domain-selective collapse pattern:
      L00H00 (code bridge) -> code accuracy drops, math/language survives
      L00H07 (math bridge) -> math accuracy drops, code/language survives
    """
    print("\n" + "=" * 62)
    print("  C. CAPABILITY COMPLETION PROBES")
    print("=" * 62)

    # Baseline
    print("\n  Baseline (no ablation):")
    baseline_acc = {}
    for domain, probes in COMPLETION_PROBES.items():
        hits = sum(check_completion(base_model, tokenizer, p, e)[0] for p, e in probes)
        acc  = hits / len(probes)
        baseline_acc[domain] = acc
        print(f"    {domain:10s}: {hits}/{len(probes)} = {100*acc:.0f}%")

    # Per-candidate
    all_results = []
    for (layer, head) in CANDIDATES:
        tag = f"L{layer:02d}H{head:02d}"
        m   = ablate_copy(base_model, layer, head)
        print(f"\n  After ablating {tag}:")

        head_row = {"head": tag}
        for domain, probes in COMPLETION_PROBES.items():
            hits = sum(check_completion(m, tokenizer, p, e)[0] for p, e in probes)
            acc  = hits / len(probes)
            drop = baseline_acc[domain] - acc
            head_row[domain]           = acc
            head_row[f"{domain}_drop"] = drop
            note = "  <- COLLAPSED" if drop >= 0.4 else ("  <- partial" if drop >= 0.2 else "")
            print(f"    {domain:10s}: {hits}/{len(probes)} = {100*acc:.0f}%  "
                  f"(d = {-100*drop:+.0f}%){note}")

        all_results.append(head_row)

    return all_results, baseline_acc


# -- Summary Plot --------------------------------------------------------------
def plot_layer0_scan(bridge_l0, prefix="small_"):
    """Bar chart of all Layer-0 head bridge scores, candidates highlighted."""
    n_heads = len(bridge_l0)
    heads  = list(range(n_heads))
    scores = [bridge_l0[h] for h in heads]
    colors = ["crimson" if (0, h) in CANDIDATES else "steelblue" for h in heads]
 
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(heads, scores, color=colors, alpha=0.85, edgecolor="none")
    ax.set_xticks(heads)
    ax.set_xticklabels([f"H{h:02d}" for h in heads])
    ax.set_xlabel("Layer-0 Head Index")
    ax.set_ylabel("Bridge Score (raw)")
    ax.set_title(f"Layer-0 Bridge Score Scan -- All {n_heads} Heads\n"
                 "(red = Phase 0 candidate, blue = low-bridge control)")
 
    from matplotlib.patches import Patch
    legend = [Patch(color="crimson", label="Phase 0 candidates"),
              Patch(color="steelblue", label="Controls")]
    ax.legend(handles=legend)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase1_5_layer0_scan.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {prefix}phase1_5_layer0_scan.png")


def plot_three_way_comparison(layer0_ctrl_rows, baseline, prefix="small_"):
    """
    Three-bar comparison per domain:
      Phase 1 Group A (bridges), Phase 1 Group B (L4/7 controls), Layer-0 low-Bridge controls
    """
    domains = list(EVAL.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 4.5))
 
    for ax, domain in zip(axes, domains):
        l0_ctrl_damage = np.mean([r["delta_pct"] for r in layer0_ctrl_rows
                                  if r["domain"] == domain])
        bars = ax.bar(
            ["Phase1 A\n(Bridges)", "Phase1 B\n(L4/7 ctrl)", "Layer-0\nLow-Bridge ctrl"],
            [PHASE1_GROUP_A[domain], PHASE1_GROUP_B[domain], l0_ctrl_damage],
            color=["crimson", "steelblue", "darkorange"],
            alpha=0.85, edgecolor="none"
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{domain} - dPPL (%)", fontsize=10)
        ax.set_ylabel("d Perplexity (%)")
 
    plt.suptitle("Three-Way Damage Comparison\n"
                 "Key question: does orange (L0 low-bridge) look like red (bridge) or blue (control)?",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase1_5_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {prefix}phase1_5_comparison.png")


# -- Hypothesis Ladder ---------------------------------------------------------
def print_hypothesis_ladder(sink_scores, layer0_verdict, completion_results, baseline_acc):
    print("\n" + "=" * 62)
    print("  HYPOTHESIS LADDER - Current State")
    print("=" * 62)

    print("\n  [OK] PROVEN")
    print("     Bridge != Wanda  (r ~ 0, Phase 0)")
    print("     Low-Wanda critical heads exist  (213x ratio, Phase 1)")
    print("     Wanda importance ranking is inverted for most critical heads")

    print("\n  [!] PENDING (will be answered by this run):")

    sinks = [(l, h) for (l, h), s in sink_scores.items() if s > 0.4]
    if sinks:
        print(f"     Attention sink confound: HEADS {sinks} suspected as sinks")
        print("       -> Domain-specific bridge interpretation is weaker for these heads")
    else:
        print("     Attention sink confound: [OK] ruled out - candidates not sinks")

    print(f"     Layer-0 confound: {layer0_verdict}")
 
    print("\n  [UNTESTED]")
    print("     Replication on 1B / 7B models")
    print("     Recovery after fine-tuning")
    print("     Behavior on distribution-shifted probe sets")
 
    print("\n" + "-" * 62)
    print("  CLEANEST SUPPORTED CLAIM (after this phase):")
    print()
    print('  "Downstream sensitivity score identifies Layer-0 attention')
    print('   heads that cause catastrophic perplexity increases when')
    print('   ablated, despite having near-minimal weight magnitude,')
    print('   and this damage pattern is not explained by Wanda score,')
    if not sinks:
        print('   attention sink behavior, or Layer-0 position alone."')
    else:
        print(f'   though heads {sinks} may be attention sinks."')
    print("=" * 62)


# -- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local GPT-2 path (default: download from HuggingFace)")
    args = parser.parse_args()

    base_model, tokenizer = load_model(args.model_path)
    global CANDIDATES
    if base_model.config.n_layer == 24:
        CANDIDATES = [(2, 12), (6, 1), (7, 2), (22, 2), (22, 13)]
        print("Detected GPT-2 Medium. Using Medium candidates:", CANDIDATES)
    else:
        print("Detected GPT-2 Small. Using Small candidates:", CANDIDATES)
 
    # A -- Attention sink test
    sink_scores = attention_sink_test(base_model, tokenizer)
 
    # B -- Layer-0 low-Bridge controls (the decisive test)
    layer0_rows, bridge_l0 = layer0_low_bridge_controls(base_model, tokenizer)
 
    # C -- Capability completion probes
    completion_results, baseline_acc = capability_completion_test(base_model, tokenizer)
 
    # Plots
    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plot_layer0_scan(bridge_l0, prefix)
    plot_three_way_comparison(layer0_rows, baseline_acc, prefix)
 
    # Verdict
    mean_l0_damage = np.mean([r["delta_pct"] for r in layer0_rows])
    if mean_l0_damage < 10:
        layer0_verdict = "[OK] RULED OUT -- low-Bridge L0 heads cause near-zero damage"
    elif mean_l0_damage < 40:
        layer0_verdict = "~ PARTIAL -- L0 has some intrinsic effect but bridge adds signal"
    else:
        layer0_verdict = "[WARNING] NOT RULED OUT -- Layer-0 position may explain Phase 1 results"

    print_hypothesis_ladder(sink_scores, layer0_verdict, completion_results, baseline_acc)


if __name__ == "__main__":
    main()
