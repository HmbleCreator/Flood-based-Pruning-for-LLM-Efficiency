"""
Phase 3b -- H07 vs H09 Mechanistic Comparison
===============================================
Why this pair, not another:

    H00  bridge=25.36  damage=~68%   ┐
    H07  bridge=15.56  damage=~48%   │ -- steep jump: +1.86 bridge -> +4x damage
    H09  bridge=13.70  damage=~12.8% ┘
    H04  bridge=2.88   damage=~2.7%

H07 and H09 sit 1.86 bridge-score units apart (13% relative difference).
Yet their damage differs by 3.8x (48.8% vs 12.8%).

H00 vs H04 shows large bridge-score difference -> large damage difference.
H07 vs H09 shows SMALL bridge-score difference -> LARGE damage difference.

This is the nonlinearity. Understanding it is more informative than
confirming that very different bridge scores produce different outcomes.

Three candidate explanations:
  E1 -- Circuit membership: H07 participates in a damage-amplifying circuit;
       H09 does not. Ablating H07 breaks the circuit; ablating H09 leaves it intact.
       Signature: H07 and H09 attend to different token types; H07's loss damage
       concentrates on tokens that require multi-head coordination.

  E2 -- Domain concentration: H07 is math-dominant (84% math damage in Phase 1);
       H09 is diffuse (12-20% across domains). Concentrated domain damage inflates
       H07's mean. Domain-conditional bridge scores would distinguish these.
       Signature: H07 entropy lower on math inputs than on language/code;
       H09 entropy similar across domains.

  E3 -- Downstream amplification: H07's output flows into a later high-gain layer;
       H09's does not. Same local bridge score, different downstream multiplier.
       Signature: H07 residual stream delta concentrates in specific downstream layers.

These are distinguishable from the same data. Run this script to find which.

Usage:
    python phase3b_h07_vs_h09.py
    python phase3b_h07_vs_h09.py --model_path C:/path/to/gpt2_local
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

# -- Subjects ------------------------------------------------------------------
H07 = (0, 7)   # bridge=15.56, mean damage ~48.8% -- HIGH in pair
H09 = (0, 9)   # bridge=13.70, mean damage ~12.8% -- LOW in pair
# Bridge difference: 1.86 units (13%). Damage difference: 3.8x. THIS IS THE MYSTERY.

# -- Analysis texts (same as Phase 3) -----------------------------------------
ANALYSIS_TEXTS = {
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "for i in range(len(arr)):\n    total += arr[i]",
        "class Node:\n    def __init__(self, val):\n        self.val = val",
    ],
    "math": [
        "The integral of x squared from zero to one equals one third.",
        "A matrix is invertible if and only if its determinant is nonzero.",
        "The gradient points in the direction of steepest ascent.",
    ],
    "language": [
        "She walked slowly down the corridor, lost in thought.",
        "The committee reached a consensus after hours of deliberation.",
        "Despite the heavy rain, the expedition continued northward.",
    ],
}

# Standardized eval (identical to Phase 1 -- do not modify)
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


def ablate_copy(base, layer, head):
    m = copy.deepcopy(base)
    hd = m.config.n_embd // m.config.n_head
    s, e = head * hd, (head + 1) * hd
    with torch.no_grad():
        m.transformer.h[layer].attn.c_proj.weight.data[s:e, :] = 0.0
    return m


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


# -- A. Attention Patterns: H07 vs H09 ----------------------------------------

def attention_patterns(model, tokenizer):
    """
    Same analysis as Phase 3 Section A, applied to H07 vs H09.
    Bridge scores are similar (15.56 vs 13.70).
    If attention patterns differ qualitatively -> circuit membership hypothesis (E1).
    If attention patterns look similar -> domain concentration hypothesis (E2).
    """
    b07 = 0.816 if model.config.n_layer == 24 else 15.56
    b09 = 0.719 if model.config.n_layer == 24 else 13.70
    print("\n" + "=" * 65)
    print("  A. ATTENTION PATTERNS: H07 vs H09")
    print(f"     Bridge scores: H07={b07:.2f}  H09={b09:.2f}  (difference = {abs(b07-b09):.2f})")
    print("=" * 65)

    viz_texts = {
        "code":     "def fibonacci(n):\n    if n <= 1:\n        return n",
        "math":     "A matrix is invertible if its determinant is nonzero.",
        "language": "She walked slowly down the corridor, lost in thought.",
    }

    fig, axes = plt.subplots(2, len(viz_texts), figsize=(6 * len(viz_texts), 10))
    fig.suptitle(
        "Attention Patterns: H07 (bridge=15.56, damage~49%) vs H09 (bridge=13.70, damage~13%)\n"
        "Similar bridge scores, 3.8x damage difference -- looking for qualitative difference",
        fontsize=10
    )

    for col, (domain, text) in enumerate(viz_texts.items()):
        for row, ((layer, head), label) in enumerate([(H07, "H07"), (H09, "H09")]):
            ax  = axes[row, col]
            inp = enc(tokenizer, text, max_length=64)
            with torch.no_grad():
                out = model(**inp, output_attentions=True)
            attn    = out.attentions[layer][0, head].numpy()
            tokens  = [tokenizer.decode([t]) for t in inp["input_ids"][0]]
            seq_len = min(len(tokens), 18)
            attn    = attn[:seq_len, :seq_len]
            tokens  = [t.replace('\n', '\n').replace(' ', '_') for t in tokens[:seq_len]]

            im = ax.imshow(attn, cmap="Reds" if (head == 7 or head == 2) else "Blues",
                           aspect="auto", vmin=0, vmax=min(attn.max(), 1.0))
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            bridge = 0.816 if model.config.n_layer == 24 else 15.56 if head == 7 else 13.70 if model.config.n_layer != 24 else 0.719
            dmg    = "high" if (head == 7 or head == 2) else "low/medium"
            ax.set_title(f"{label} (bridge={bridge:.2f}, damage={dmg})\n{domain}", fontsize=9)

    prefix = "medium_" if (model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3b_patterns.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {prefix}phase3b_patterns.png")
    print("  Key question: do H07 and H09 attend to DIFFERENT token types?")
    print("  If H07 focuses on mathematical/structural tokens and H09 does not -> E2 (domain)")
    print("  If both look similar but H07 is more concentrated -> E3 (downstream amplification)")


# -- B. Per-Token Loss Map: H07 vs H09 ----------------------------------------

def per_token_loss_map(base_model, tokenizer):
    """
    Which token positions suffer more from H07 ablation vs H09 ablation?
    
    If H07 damage concentrates on math-relevant tokens (numbers, equations, 'if', 'equals')
    -> domain concentration hypothesis (E2) supported.
    
    If H07 damage is distributed but uniformly larger than H09
    -> downstream amplification hypothesis (E3) more likely.
    
    Run on a math text (H07 is math-dominant) and a language text (for comparison).
    """
    print("\n" + "=" * 65)
    print("  B. PER-TOKEN LOSS MAP: H07 vs H09")
    print("=" * 65)

    m_h07 = ablate_copy(base_model, *H07)
    m_h09 = ablate_copy(base_model, *H09)

    viz_texts = {
        "math":     "A matrix is invertible if and only if its determinant is nonzero.",
        "language": "She walked slowly down the corridor, lost in thought about everything.",
    }

    fig, axes = plt.subplots(2, len(viz_texts), figsize=(8 * len(viz_texts), 8))
    fig.suptitle(
        "Per-Token Loss Increase: H07 ablation (red) vs H09 ablation (blue)\n"
        "Math text: does H07 damage concentrate on mathematical terms?",
        fontsize=10
    )

    for col, (domain, text) in enumerate(viz_texts.items()):
        inp    = enc(tokenizer, text)
        ids    = inp["input_ids"][0]
        tokens = [tokenizer.decode([t]) for t in ids]
        tokens = [t.replace('\n', '\n').replace(' ', '·') for t in tokens]
        n_tok  = len(tokens)

        def token_losses(model):
            losses = []
            with torch.no_grad():
                out    = model(**inp)
                logits = out.logits[0]
                for i in range(1, n_tok):
                    lp = torch.log_softmax(logits[i - 1], dim=-1)
                    losses.append(-lp[ids[i]].item())
            return np.array(losses)

        base_loss = token_losses(base_model)
        h07_loss  = token_losses(m_h07)
        h09_loss  = token_losses(m_h09)

        delta_h07 = h07_loss - base_loss
        delta_h09 = h09_loss - base_loss

        b07 = 0.816 if base_model.config.n_layer == 24 else 15.56
        b09 = 0.719 if base_model.config.n_layer == 24 else 13.70
        pos = list(range(len(base_loss)))
        for row, (delta, label, color) in enumerate([
            (delta_h07, f"H07 ablation (bridge={b07:.2f})", "crimson"),
            (delta_h09, f"H09 ablation (bridge={b09:.2f})", "steelblue"),
        ]):
            ax = axes[row, col]
            ax.bar(pos, delta, color=color, alpha=0.75, edgecolor="none")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(pos)
            ax.set_xticklabels(tokens[1:], rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("d Log-Loss", fontsize=8)
            ax.set_title(f"{label}\n{domain}", fontsize=9)

        print(f"\n  [{domain}]")
        print(f"    H07: mean dloss={delta_h07.mean():.3f}  "
              f"max at '{tokens[1:][delta_h07.argmax()]}' (+{delta_h07.max():.3f})")
        print(f"    H09: mean dloss={delta_h09.mean():.3f}  "
              f"max at '{tokens[1:][delta_h09.argmax()]}' (+{delta_h09.max():.3f})")

        top3_h07 = np.argsort(delta_h07)[::-1][:3]
        top3_h09 = np.argsort(delta_h09)[::-1][:3]
        print(f"    H07 top-3: " + ", ".join(f"'{tokens[1:][i]}'(+{delta_h07[i]:.2f})" for i in top3_h07))
        print(f"    H09 top-3: " + ", ".join(f"'{tokens[1:][i]}'(+{delta_h09[i]:.2f})" for i in top3_h09))

    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3b_token_loss.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved -> {prefix}phase3b_token_loss.png")


# -- C. Domain-Conditional Damage ---------------------------------------------

def domain_conditional_damage(base_model, tokenizer):
    """
    Recompute per-domain perplexity for H07 and H09.
    H07 is expected to be math-dominant (Phase 1: code+30%, math+84%, lang+32%).
    H09 is expected to be diffuse (Phase 2: code+19.5%, math+12.8%, lang+6.0%).
    
    This directly tests hypothesis E2: H07's elevated MEAN damage is driven by
    high damage in ONE domain, not uniform elevation across all three.
    
    If H07 math damage >> H09 math damage, but H07 code/lang ~ H09 code/lang
    -> H07 is a domain-specific bridge; mixed-probe bridge score underestimates it.
    -> Domain-conditional bridge scores B_math, B_code, B_lang would rank H07 higher.
    """
    print("\n" + "=" * 65)
    print("  C. DOMAIN-CONDITIONAL DAMAGE: H07 vs H09")
    print("=" * 65)

    baseline = {d: perplexity(base_model, tokenizer, EVAL[d]) for d in EVAL}
    m_h07    = ablate_copy(base_model, *H07)
    m_h09    = ablate_copy(base_model, *H09)

    print(f"\n  {'Domain':<10} {'Baseline':>9} {'H07 d%':>9} {'H09 d%':>9}  Ratio (H07/H09)")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*9}  ---------------")

    domain_ratios = {}
    for domain, texts in EVAL.items():
        ppl_h07  = perplexity(m_h07, tokenizer, texts)
        ppl_h09  = perplexity(m_h09, tokenizer, texts)
        dpct_h07 = 100.0 * (ppl_h07 - baseline[domain]) / baseline[domain]
        dpct_h09 = 100.0 * (ppl_h09 - baseline[domain]) / baseline[domain]
        ratio    = dpct_h07 / (abs(dpct_h09) + 1e-6)
        domain_ratios[domain] = ratio
        print(f"  {domain:<10} {baseline[domain]:>9.3f} {dpct_h07:>8.1f}% {dpct_h09:>8.1f}%  {ratio:.1f}x")

    print()
    max_ratio_domain = max(domain_ratios, key=domain_ratios.get)
    if domain_ratios[max_ratio_domain] > 3:
        print(f"  -> H07 damage concentrated in [{max_ratio_domain}] domain")
        print(f"    E2 (domain concentration) is likely contributing to the H07/H09 gap.")
        print(f"    Mixed bridge score averages this away; domain-conditional B_{max_ratio_domain} would rank H07 higher.")
    else:
        print(f"  -> H07 damage is roughly uniform across domains (ratio similar everywhere)")
        print(f"    E2 (domain concentration) is not the explanation.")
        print(f"    E1 (circuit membership) or E3 (downstream amplification) more likely.")

    # Plot
    domains  = list(EVAL.keys())
    m_h07_d  = []
    m_h09_d  = []
    for domain, texts in EVAL.items():
        ppl_h07 = perplexity(m_h07, tokenizer, texts)
        ppl_h09 = perplexity(m_h09, tokenizer, texts)
        m_h07_d.append(100.0 * (ppl_h07 - baseline[domain]) / baseline[domain])
        m_h09_d.append(100.0 * (ppl_h09 - baseline[domain]) / baseline[domain])

    b07 = 0.816 if base_model.config.n_layer == 24 else 15.56
    b09 = 0.719 if base_model.config.n_layer == 24 else 13.70
    x   = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, m_h07_d, 0.35, label=f"H07 (bridge={b07:.2f})", color="crimson", alpha=0.8)
    ax.bar(x + 0.2, m_h09_d, 0.35, label=f"H09 (bridge={b09:.2f})", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("d Perplexity (%)")
    ax.set_title(
        "Domain-Conditional Damage: H07 vs H09\n"
        "If red bars differ across domains but blue bars are flat -> domain concentration (E2)",
        fontsize=10
    )
    ax.legend()
    ax.axhline(0, color="black", lw=0.8)
    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3b_domain_damage.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {prefix}phase3b_domain_damage.png")


# -- D. Downstream Layer Sensitivity ------------------------------------------

def downstream_layer_sensitivity(model, tokenizer):
    """
    How does the bridge score's downstream L2 shift distribute across layers?
    
    For H07 vs H09 ablation, measure the L2 shift at each downstream layer separately.
    
    If H07's shift concentrates in specific later layers -> E3 (amplification)
    If H07's shift is uniformly larger across all layers -> scaled version of H09
    
    This distinguishes whether H07 has privileged connections to high-gain layers.
    """
    print("\n" + "=" * 65)
    print("  D. DOWNSTREAM LAYER SENSITIVITY: H07 vs H09")
    print("=" * 65)

    n_layers = model.config.n_layer
    head_dim = model.config.n_embd // model.config.n_head

    probe_texts = [
        "A matrix is invertible if and only if its determinant is nonzero.",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "She walked slowly down the corridor, lost in thought.",
    ]

    layer_deltas = {H07: np.zeros(n_layers - 1), H09: np.zeros(n_layers - 1)}

    for (layer, head), ldeltas in layer_deltas.items():
        s, e = head * head_dim, (head + 1) * head_dim
        counts = np.zeros(n_layers - 1)

        for text in probe_texts:
            inp = enc(tokenizer, text, max_length=64)

            base = defaultdict(list)
            hs_b = [model.transformer.h[dl].register_forward_hook(
                        lambda m, i, o, dl_=dl, store=base: store[dl_].append(o[0].detach().float()))
                    for dl in range(layer + 1, n_layers)]
            with torch.no_grad():
                model(**inp)
            for h in hs_b: h.remove()

            abl = defaultdict(list)
            def ablate(mod, pre):
                x = pre[0].clone(); x[:, :, s:e] = 0.0; return (x,)
            ah = model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(ablate)
            hs_a = [model.transformer.h[dl].register_forward_hook(
                        lambda m, i, o, dl_=dl, store=abl: store[dl_].append(o[0].detach().float()))
                    for dl in range(layer + 1, n_layers)]
            with torch.no_grad():
                model(**inp)
            ah.remove()
            for h in hs_a: h.remove()

            for dl in range(layer + 1, n_layers):
                if base[dl] and abl[dl]:
                    B = base[dl][0].mean(dim=1)
                    A = abl[dl][0].mean(dim=1)
                    ldeltas[dl - layer - 1] += (B - A).norm(dim=-1).mean().item()
                    counts[dl - layer - 1]  += 1

        mask = counts > 0
        ldeltas[mask] /= counts[mask]

    # Print
    print(f"\n  Layer  {'H07 d':>10}  {'H09 d':>10}  Ratio")
    print(f"  -----  {'-'*10}  {'-'*10}  -----")
    for dl in range(min(n_layers - 1, 10)):
        d07 = layer_deltas[H07][dl]
        d09 = layer_deltas[H09][dl]
        ratio = d07 / (d09 + 1e-9)
        print(f"  {dl+1:>5}  {d07:>10.4f}  {d09:>10.4f}  {ratio:.2f}x")

    # Plot
    b07 = 0.816 if model.config.n_layer == 24 else 15.56
    b09 = 0.719 if model.config.n_layer == 24 else 13.70
    x     = np.arange(min(n_layers - 1, 10))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x + 1, layer_deltas[H07][:10], "o-", color="crimson",   label=f"H07 ablation (bridge={b07:.2f})")
    ax.plot(x + 1, layer_deltas[H09][:10], "s-", color="steelblue", label=f"H09 ablation (bridge={b09:.2f})")
    ax.set_xlabel("Downstream Layer")
    ax.set_ylabel("Mean L2 Shift in Hidden State")
    ax.set_title(
        "Per-Layer Downstream Sensitivity: H07 vs H09\n"
        "If H07 shift spikes at specific layers -> downstream amplification (E3)",
        fontsize=10
    )
    ax.legend()
    prefix = "medium_" if (model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3b_downstream_layers.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {prefix}phase3b_downstream_layers.png")


# -- Summary -------------------------------------------------------------------

def print_summary():
    print("\n" + "=" * 65)
    print("  READING THE OUTPUTS")
    print("=" * 65)
    print()
    print("  phase3b_patterns.png")
    print("    H07 and H09 attention heatmaps look DIFFERENT -> E1 (circuit)")
    print("    H07 and H09 attention heatmaps look SIMILAR   -> E2 or E3")
    print()
    print("  phase3b_token_loss.png")
    print("    H07 damage peaks on math tokens (numbers, 'matrix', 'if') -> E2")
    print("    H07 damage distributed like H09 but uniformly larger -> E3")
    print()
    print("  phase3b_domain_damage.png")
    print("    H07 math bar much taller than H09 math bar,")
    print("    but H07 code/lang ~ H09 code/lang -> E2 confirmed")
    print("    All H07 bars taller than H09 equally -> not E2, look at E3")
    print()
    print("  phase3b_downstream_layers.png")
    print("    H07 shift spikes at a specific layer (e.g. layer 5) -> E3 (amplification)")
    print("    H07 shift uniformly larger at all layers -> scaled version of H09")
    print()
    print("  The answer to 'why is H07 3.8x more damaging than H09")
    print("  despite similar bridge scores' is in these four plots.")
    print("=" * 65)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    base_model, tokenizer = load_model(args.model_path)
    global H07, H09
    if base_model.config.n_layer == 24:
        H07 = (22, 2)
        H09 = (22, 13)
        print(f"Detected GPT-2 Medium. Setting subjects to: H07={H07}, H09={H09}")
    else:
        print(f"Detected GPT-2 Small. Setting subjects to: H07={H07}, H09={H09}")

    attention_patterns(base_model, tokenizer)
    per_token_loss_map(base_model, tokenizer)
    domain_conditional_damage(base_model, tokenizer)
    downstream_layer_sensitivity(base_model, tokenizer)
    print_summary()


if __name__ == "__main__":
    main()
