"""
Phase 3 -- Mechanistic Analysis
================================
Central question: What is H00 doing that H04 isn't?

    H00: bridge=25.36, mean damage ~68%
    H04: bridge=2.88,  mean damage  ~2.7%

Same layer (Layer 0). Same architecture. 9x bridge score difference. 25x damage difference.
This controlled comparison should reveal what bridge score is actually measuring.

Four analyses:

A. Attention Pattern Visualization
   Heatmap of H00 vs H04 attention weights on the same inputs.
   If H00 attends to specific structural tokens (function names, operators, delimiters)
   while H04 attends diffusely or positionally -> structural routing hypothesis.

B. Per-Token Loss Mapping
   After ablating H00 vs H04, which token positions suffer increased loss?
   This reveals WHAT H00 is computationally responsible for.
   Code-token positions hurt more -> code-routing hypothesis.
   Function/structure tokens hurt more -> syntactic bridge hypothesis.

C. Attention Entropy Comparison
   Shannon entropy of attention distributions: H(attn) = -sum(a_i * log(a_i))
   Low entropy = focused attention (attends to few tokens strongly)
   High entropy = diffuse attention (spreads weight broadly)
   H00 being lower entropy than H04 -> more selective routing.

D. Token-Type Sensitivity
   For code inputs: measure loss increase at function tokens, operator tokens,
   identifier tokens, whitespace/indent tokens separately.
   This identifies what syntactic role H00 is supporting.

After running:
   Compare results to known mechanistic findings in GPT-2 literature.
   (Induction heads, duplicate token heads, previous-token heads, etc.)

REPLICATION NOTE:
   run_replication() at the bottom runs Phase 0 + Phase 1 core on any model.
   Call with "gpt2-medium" or "EleutherAI/pythia-160m" after Phase 3.
   Replication is the single highest-value remaining experiment.

Usage:
    python phase3_mechanistic.py
    python phase3_mechanistic.py --model_path C:/path/to/gpt2_local
"""

import argparse
import copy
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict

# -- Subjects ------------------------------------------------------------------
H00 = (0, 0)   # bridge=25.36, damage ~67%  -- HIGH bridge
H04 = (0, 4)   # bridge=2.88,  damage ~2.7% -- LOW bridge (control)
H09 = (0, 9)   # bridge=13.70, damage ~12.8% -- MEDIUM bridge (for spectrum plot)

# -- Analysis texts ------------------------------------------------------------
# Chosen for interpretability: short enough to read attention heatmaps,
# rich enough to have structure that a routing head could exploit.

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

# Full eval set for perplexity (standardized, matches Phase 1)
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


def get_attention(model, tokenizer, text, layer, head, max_length=64):
    """Get attention weights for a specific head on a specific input."""
    inp = enc(tokenizer, text, max_length=max_length)
    with torch.no_grad():
        out = model(**inp, output_attentions=True)
    attn = out.attentions[layer][0, head]   # (seq, seq)
    tokens = [tokenizer.decode([t]) for t in inp["input_ids"][0]]
    return attn.numpy(), tokens


# -- A. Attention Pattern Visualization ---------------------------------------

def attention_pattern_analysis(model, tokenizer):
    """
    Side-by-side attention heatmaps: H00 vs H04 on the same inputs.
    H00 (high bridge): expected to show structured, content-sensitive attention.
    H04 (low bridge): expected to show diffuse or positional attention.
    """
    print("\n" + "=" * 65)
    print("  A. ATTENTION PATTERN VISUALIZATION")
    print("=" * 65)

    # Use one text per domain for visualization
    viz_texts = {
        "code":     "def fibonacci(n):\n    if n <= 1:\n        return n",
        "math":     "A matrix is invertible if its determinant is nonzero.",
        "language": "She walked slowly down the corridor, lost in thought.",
    }

    fig, axes = plt.subplots(2, len(viz_texts), figsize=(6 * len(viz_texts), 10))
    fig.suptitle("Attention Patterns: H00 (high-bridge) vs H04 (low-bridge control)\n"
                 "Structural routing -> H00 should show focused, content-selective attention",
                 fontsize=11)

    for col, (domain, text) in enumerate(viz_texts.items()):
        for row, (layer, head) in enumerate([H00, H04]):
            ax    = axes[row, col]
            attn, tokens = get_attention(model, tokenizer, text, layer, head)
            seq_len = min(len(tokens), 20)   # cap for readability
            attn    = attn[:seq_len, :seq_len]
            tokens  = tokens[:seq_len]
            tokens  = [t.replace('\n', '\n').replace(' ', '_') for t in tokens]

            im = ax.imshow(attn, cmap="Blues", aspect="auto",
                           vmin=0, vmax=attn.max())
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            head_label = "H00 (bridge=25.36)" if head == 0 else "H04 (bridge=2.88)"
            ax.set_title(f"{head_label}\n{domain}", fontsize=9)
            ax.set_xlabel("Keys (attended to)", fontsize=7)
            ax.set_ylabel("Queries (attending from)", fontsize=7)

    prefix = "medium_" if (model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3_attention_patterns.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {prefix}phase3_attention_patterns.png")
    print("  What to look for:")
    print("    H00: concentrated attention to specific tokens (colons, keywords, punctuation)?")
    print("    H04: more uniform / diagonal / position-based attention?")
    print("    Domain sensitivity: does H00 pattern change across code/math/language?")


# -- B. Per-Token Loss Mapping -------------------------------------------------

def per_token_loss_map(base_model, tokenizer):
    """
    For each position in a text, compute:
        loss_increase(pos) = CE(ablated, pos) - CE(baseline, pos)
    
    High values -> ablating H00 strongly hurts prediction at this position.
    This reveals WHAT tokens/positions H00 is computationally responsible for.
    
    Comparison: H00 ablation vs H04 ablation.
    If H00's damage concentrates on structural tokens (def, :, indent, function names)
    -> H00 is doing syntactic/structural routing.
    If damage is uniform -> H00 is doing something distributed.
    """
    print("\n" + "=" * 65)
    print("  B. PER-TOKEN LOSS MAPPING")
    print("=" * 65)

    m_h00 = ablate_copy(base_model, *H00)
    m_h04 = ablate_copy(base_model, *H04)

    viz_texts = {
        "code":     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1)",
        "language": "She walked slowly down the corridor, lost in thought about the past.",
    }

    fig, axes = plt.subplots(2, len(viz_texts), figsize=(7 * len(viz_texts), 8))
    fig.suptitle("Per-Token Loss Increase After Ablation: H00 vs H04\n"
                 "High bars -> ablating this head strongly hurts prediction at this position",
                 fontsize=11)

    for col, (domain, text) in enumerate(viz_texts.items()):
        inp    = enc(tokenizer, text)
        ids    = inp["input_ids"][0]
        tokens = [tokenizer.decode([t]) for t in ids]
        tokens = [t.replace('\n', '\n').replace(' ', '·') for t in tokens]
        n_tok  = len(tokens)
        pos    = list(range(1, n_tok))   # predict positions 1..n

        def per_token_losses(model):
            losses = []
            with torch.no_grad():
                out    = model(**inp)
                logits = out.logits[0]   # (seq, vocab)
                for i in range(1, n_tok):
                    target = ids[i]
                    lp     = torch.log_softmax(logits[i - 1], dim=-1)
                    losses.append(-lp[target].item())
            return np.array(losses)

        base_losses = per_token_losses(base_model)
        h00_losses  = per_token_losses(m_h00)
        h04_losses  = per_token_losses(m_h04)

        delta_h00 = h00_losses - base_losses
        delta_h04 = h04_losses - base_losses

        for row, (delta, label, color) in enumerate([
            (delta_h00, "H00 ablation (bridge=25.36)", "crimson"),
            (delta_h04, "H04 ablation (bridge=2.88)",  "steelblue"),
        ]):
            ax = axes[row, col]
            ax.bar(range(len(pos)), delta, color=color, alpha=0.75, edgecolor="none")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(pos)))
            ax.set_xticklabels(tokens[1:], rotation=60, ha="right", fontsize=6)
            ax.set_ylabel("d Log-Loss", fontsize=8)
            ax.set_title(f"{label}\n{domain}", fontsize=9)

        print(f"\n  [{domain}]")
        print(f"    H00 ablation: mean dloss = {delta_h00.mean():.3f}, "
              f"max at '{tokens[1:][delta_h00.argmax()]}' (+{delta_h00.max():.3f})")
        print(f"    H04 ablation: mean dloss = {delta_h04.mean():.3f}, "
              f"max at '{tokens[1:][delta_h04.argmax()]}' (+{delta_h04.max():.3f})")

        # Identify top-3 most affected positions under H00 ablation
        top3 = np.argsort(delta_h00)[::-1][:3]
        print(f"    H00 top-3 affected positions: "
              + ", ".join(f"'{tokens[1:][i]}' (+{delta_h00[i]:.2f})" for i in top3))

    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3_token_loss_map.png", dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved -> {prefix}phase3_token_loss_map.png")
    print("  Key question: does H00 damage concentrate on specific token types?")


# -- C. Attention Entropy ------------------------------------------------------

def attention_entropy_analysis(model, tokenizer):
    """
    Shannon entropy of attention distributions.
    H(attn_row) = -sum_j attn[i,j] * log(attn[i,j] + eps)
    
    Low entropy  = focused attention on few tokens (selective routing)
    High entropy = diffuse, spread-out attention (weak routing signal)
    
    Hypothesis: H00 has systematically lower entropy than H04,
    meaning it is doing more selective, structured routing.
    """
    print("\n" + "=" * 65)
    print("  C. ATTENTION ENTROPY COMPARISON")
    print("=" * 65)
    print(f"  {'Domain':<12} {'H00 entropy':>14} {'H04 entropy':>14}  Ratio")
    print(f"  {'-'*12} {'-'*14} {'-'*14}  -----")

    all_texts = [t for v in ANALYSIS_TEXTS.values() for t in v]
    results   = {}

    for (layer, head), label in [(H00, "H00"), (H04, "H04")]:
        entropies = defaultdict(list)
        for domain, texts in ANALYSIS_TEXTS.items():
            for text in texts:
                inp  = enc(tokenizer, text, max_length=64)
                with torch.no_grad():
                    out  = model(**inp, output_attentions=True)
                attn = out.attentions[layer][0, head].numpy()  # (seq, seq)
                eps  = 1e-9
                ent  = -np.sum(attn * np.log(attn + eps), axis=-1).mean()
                entropies[domain].append(ent)
        results[label] = {d: float(np.mean(v)) for d, v in entropies.items()}

    for domain in ANALYSIS_TEXTS:
        h00_ent = results["H00"][domain]
        h04_ent = results["H04"][domain]
        ratio   = h00_ent / (h04_ent + 1e-9)
        marker  = " <- H00 more focused" if h00_ent < h04_ent else " <- H04 more focused"
        print(f"  {domain:<12} {h00_ent:>14.4f} {h04_ent:>14.4f}  {ratio:.3f}{marker}")

    # Plot
    domains  = list(ANALYSIS_TEXTS.keys())
    h00_vals = [results["H00"][d] for d in domains]
    h04_vals = [results["H04"][d] for d in domains]
    x = np.arange(len(domains))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, h00_vals, 0.35, label="H00 (bridge=25.36)", color="crimson", alpha=0.8)
    ax.bar(x + 0.2, h04_vals, 0.35, label="H04 (bridge=2.88)",  color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("Mean attention entropy")
    ax.set_title("Attention Entropy: H00 vs H04\n"
                 "Lower entropy = more selective routing")
    ax.legend()
    prefix = "medium_" if (model.config.n_layer == 24) else "small_"
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3_entropy.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved -> {prefix}phase3_entropy.png")
    return results


# -- D. Bridge Score Damage Spectrum ------------------------------------------

def damage_spectrum_plot(prefix="small_"):
    """
    Plot bridge score vs mean damage for ALL tested heads.
    Shows whether bridge score tracks damage continuously (gradient)
    or as a threshold (binary).
    """
    print("\n" + "=" * 65)
    print("  D. BRIDGE SCORE -- DAMAGE SPECTRUM")
    print("=" * 65)

    # All data points from Phase 0/1/1.5/2
    data = [
        # (head_label, bridge_score, mean_damage_pct, color, marker)
        ("H04 ctrl",  2.88,  2.7,   "steelblue", "o"),
        ("H11 ctrl",  4.11,  2.1,   "steelblue", "o"),
        ("H08 ctrl",  5.73,  8.1,   "steelblue", "o"),
        ("H09 pred",  13.70, 12.8,  "gold",      "D"),
        ("H07 cand",  15.56, 48.8,  "crimson",   "s"),
        ("H00 cand",  25.36, 68.3,  "crimson",   "s"),
        ("H10 cand",  24.55, 177.3, "crimson",   "s"),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, bridge, damage, color, marker in data:
        ax.scatter(bridge, damage, c=color, s=100, marker=marker,
                   zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(label, (bridge, damage),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    ax.set_xlabel("Bridge Score (raw)", fontsize=10)
    ax.set_ylabel("Mean d Perplexity (%)", fontsize=10)
    ax.set_title("Bridge Score vs Ablation Damage -- All Tested Heads\n"
                 "Ordinal gradient suggests bridge score tracks a continuous property",
                 fontsize=10)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", ms=10, label="Controls"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="gold",      ms=10, label="H09 (prediction)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="crimson",   ms=10, label="Candidates"),
    ]
    ax.legend(handles=legend)
    ax.set_yscale("log")   # log scale to see the full range
    ax.set_ylim(0.5, 500)
    plt.tight_layout()
    plt.savefig(f"{prefix}phase3_damage_spectrum.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved -> {prefix}phase3_damage_spectrum.png")
    print()
    print("  Note: H10 (bridge=24.55) causes higher mean damage than H00 (bridge=25.36)")
    print("  despite slightly lower bridge score. Likely because bridge score uses")
    print("  mixed-domain probes -- H10 is language-heavy, language underweighted in probes.")
    print("  Domain-conditional bridge scores (B_code, B_math, B_lang) would fix this.")


# -- Replication Runner --------------------------------------------------------

def run_replication(model_name="gpt2-medium", model_path=None, n_probe_texts=5):
    """
    Run Phase 0 + Phase 1 core pipeline on a different model.
    
    The key invariant to check:
    1. Does bridge score remain orthogonal to Wanda? (r ~ 0)
    2. Do low-Wanda / high-Bridge heads exist?
    3. Do they cause catastrophic damage when ablated?
    4. Are they concentrated in early layers?
    
    If all four hold on a second model -> result is architecture-general.
    
    Usage:
        run_replication("gpt2-medium")
        run_replication("EleutherAI/pythia-160m")
        run_replication("EleutherAI/pythia-410m")
    """
    print(f"\n{'='*65}")
    print(f"  REPLICATION: {model_name}")
    print(f"{'='*65}")
    print("  Loading model...")

    src = model_path
    if src is None:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, "..", "..", "gpt2_medium_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "..", "gpt2_medium_local")
        if not os.path.exists(local_path):
            local_path = os.path.join(script_dir, "gpt2_medium_local")
        if os.path.exists(local_path) and model_name == "gpt2-medium":
            print(f"Detected local path: {local_path}")
            src = local_path
        else:
            src = model_name

    tok = GPT2Tokenizer.from_pretrained(src)
    tok.pad_token = tok.eos_token
    mdl = GPT2LMHeadModel.from_pretrained(src)
    mdl.eval()

    n_layers = mdl.config.n_layer
    n_heads  = mdl.config.n_head
    head_dim = mdl.config.n_embd // n_heads
    print(f"  Architecture: {n_layers} layers x {n_heads} heads = {n_layers*n_heads} total heads")

    probe_texts = [
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1",
        "The fundamental theorem of calculus connects differentiation and integration.",
        "She walked slowly down the corridor, lost in thought.",
        "import torch\nmodel = torch.nn.Linear(128, 64)\noutput = model(x)",
        "A matrix is invertible if and only if its determinant is nonzero.",
    ][:n_probe_texts]

    # Phase 0 core: Wanda + Bridge for all heads
    print("  Computing Wanda scores...")
    captured = defaultdict(list)
    def make_hook(l):
        def h(mod, inp, out): captured[l].append(inp[0].detach())
        return h
    hooks = [mdl.transformer.h[l].attn.c_proj.register_forward_hook(make_hook(l))
             for l in range(n_layers)]
    with torch.no_grad():
        for t in probe_texts:
            mdl(**enc(tok, t, max_length=64))
    for h in hooks: h.remove()

    wanda = np.zeros((n_layers, n_heads))
    for l in range(n_layers):
        W = mdl.transformer.h[l].attn.c_proj.weight
        for head in range(n_heads):
            s, e = head * head_dim, (head + 1) * head_dim
            norms = [x[:, :, s:e].norm(dim=-1).mean().item() for x in captured[l]]
            input_norm = float(np.mean(norms))
            weight_mag = W[s:e, :].abs().mean().item()
            wanda[l, head] = input_norm * weight_mag

    # Bridge scores for first 2 layers only (speed)
    print("  Computing bridge scores (layers 0-1 only for speed)...")
    bridge = np.zeros((min(2, n_layers), n_heads))
    for l in range(min(2, n_layers)):
        for head in range(n_heads):
            s, e = head * head_dim, (head + 1) * head_dim
            deltas = []
            for text in probe_texts[:3]:
                inp_ = enc(tok, text, max_length=64)
                base = []
                hs_b = [mdl.transformer.h[dl].register_forward_hook(
                            lambda m,i,o,a=base: a.append(o[0].detach().float()))
                        for dl in range(l + 1, n_layers)]
                with torch.no_grad(): mdl(**inp_)
                for h_ in hs_b: h_.remove()
                abl = []
                def abl_fn(mod, pre):
                    x = pre[0].clone(); x[:, :, s:e] = 0.0; return (x,)
                ah = mdl.transformer.h[l].attn.c_proj.register_forward_pre_hook(abl_fn)
                hs_a = [mdl.transformer.h[dl].register_forward_hook(
                            lambda m,i,o,a=abl: a.append(o[0].detach().float()))
                        for dl in range(l + 1, n_layers)]
                with torch.no_grad(): mdl(**inp_)
                ah.remove()
                for h_ in hs_a: h_.remove()
                if base and abl:
                    B = torch.stack([t_.mean(dim=1) for t_ in base])
                    A = torch.stack([t_.mean(dim=1) for t_ in abl])
                    deltas.append((B - A).norm(dim=-1).mean().item())
            bridge[l, head] = float(np.mean(deltas)) if deltas else 0.0

    # Correlation check
    w_flat = wanda[:2].flatten()
    b_flat = bridge.flatten()
    r = float(np.corrcoef(w_flat, b_flat)[0, 1])
    print(f"\n  Wanda-Bridge correlation (layers 0-1): r = {r:+.3f}")
    if abs(r) < 0.3:
        print("  [OK] Independent signals -- bridge != Wanda on this model")
    else:
        print("  [WARNING] Correlated -- bridge and Wanda overlap on this model")

    # Find top bridge heads in Layer 0
    top_bridge_l0 = sorted(range(n_heads), key=lambda h: -bridge[0, h])[:3]
    w_norm = (wanda - wanda.min()) / (wanda.max() - wanda.min() + 1e-9)
    print(f"\n  Top-3 bridge heads in Layer 0:")
    for h in top_bridge_l0:
        print(f"    L00H{h:02d}: bridge={bridge[0,h]:.3f}  wanda_norm={w_norm[0,h]:.3f}")

    invisible = [(0, h) for h in top_bridge_l0 if w_norm[0, h] < 0.35]
    print(f"  Low-Wanda / high-Bridge candidates: {invisible}")

    if invisible:
        print("\n  Ablating top candidates (quick damage check)...")
        baseline = {d: perplexity(mdl, tok, EVAL[d]) for d in EVAL}
        for (l, h) in invisible[:2]:
            m = ablate_copy(mdl, l, h)
            for d in EVAL:
                ppl   = perplexity(m, tok, EVAL[d])
                dpct  = 100 * (ppl - baseline[d]) / baseline[d]
                print(f"    L{l:02d}H{h:02d} [{d:8s}]: d = {dpct:+.1f}%")
    else:
        print("  No low-Wanda / high-Bridge candidates found in layers 0-1.")
        print("  Check deeper layers or try a longer bridge scan.")

    print(f"\n  Replication complete: {model_name}")
    return r, invisible


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",  type=str, default=None,
                        help="Local GPT-2 path")
    parser.add_argument("--replicate_on", type=str, default=None,
                        help="Run replication on this model (e.g. gpt2-medium)")
    args = parser.parse_args()

    base_model, tokenizer = load_model(args.model_path)
    global H00, H04, H09
    if base_model.config.n_layer == 24:
        H00 = (2, 12)
        H04 = (2, 1)
        H09 = (2, 6)
        print(f"Detected GPT-2 Medium. Setting subjects to: H00={H00}, H04={H04}, H09={H09}")
    else:
        print(f"Detected GPT-2 Small. Setting subjects to: H00={H00}, H04={H04}, H09={H09}")

    # A -- Attention patterns
    attention_pattern_analysis(base_model, tokenizer)

    # B -- Per-token loss map
    per_token_loss_map(base_model, tokenizer)

    # C -- Attention entropy
    entropy_results = attention_entropy_analysis(base_model, tokenizer)

    # D -- Full damage spectrum
    prefix = "medium_" if (base_model.config.n_layer == 24) else "small_"
    damage_spectrum_plot(prefix)

    print("\n" + "=" * 65)
    print("  WHAT TO LOOK FOR IN THE OUTPUTS")
    print("=" * 65)
    print()
    print("  phase3_attention_patterns.png")
    print("    H00 pattern focused on specific tokens -> structural routing")
    print("    H04 pattern diffuse/diagonal -> positional or weak routing")
    print()
    print("  phase3_token_loss_map.png")
    print("    Loss increase concentrated on structural tokens (colons, keywords)?")
    print("    -> H00 is a syntactic bridge")
    print("    Loss increase uniform across all positions?")
    print("    -> H00 is doing something more distributed")
    print()
    print("  phase3_entropy.png")
    print("    H00 lower entropy than H04 -> more selective routing signal")
    print()
    print("  phase3_damage_spectrum.png")
    print("    Monotone relationship (bridge score -> damage)?")
    print("    -> Bridge score tracks a continuous structural property")
    print("    -> Paper claim: 'downstream sensitivity correlates with ablation damage'")
    print()

    # Replication
    if args.replicate_on:
        run_replication(args.replicate_on)
    else:
        print("  To replicate on GPT-2 Medium:")
        print("    python phase3_mechanistic.py --replicate_on gpt2-medium")
        print()
        print("  To replicate on Pythia-160M:")
        print("    python phase3_mechanistic.py --replicate_on EleutherAI/pythia-160m")


if __name__ == "__main__":
    main()
