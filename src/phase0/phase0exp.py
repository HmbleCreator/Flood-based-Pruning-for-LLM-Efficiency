"""
Phase 0 — Invisible Bridge Hypothesis
======================================
GPT-2 Small: 12 layers × 12 heads = 144 attention heads

Computes per attention head:
  · Wanda score  — weight magnitude × input activation norm  (existing baseline)
  · Bridge score — downstream representation shift under head ablation  (new signal)

Central question: does a Low-Wanda / High-Bridge population exist?

  World A → No such population.
            Transformers are already distributed. Relational importance is rare.
            Revisit bridge metric definition.

  World B → Population exists.
            Structural bridges exist that individual metrics miss.
            Proceed to Phase 1 targeted ablation.

Both outcomes are scientifically informative.

Usage:
  pip install torch transformers matplotlib numpy
  python phase0_invisible_bridge.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

# ── Probe texts (5 per domain) ────────────────────────────────────────────────

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
            print(f"Loading from local path: {local_path}")
            src = local_path
        else:
            src = "gpt2"

    print(f"Loading model from: {src}")
    tokenizer = GPT2Tokenizer.from_pretrained(src)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(src)
    model.eval()
    return model, tokenizer


def encode(tokenizer, text, max_length=64):
    return tokenizer(text, return_tensors="pt",
                     truncation=True, max_length=max_length)


# ── Wanda Score ───────────────────────────────────────────────────────────────

def compute_wanda(model, tokenizer, texts):
    """
    Per-head Wanda score = mean_input_norm(head_slice) × mean_abs_weight(head_slice)

    Scoring weight: output projection c_proj at each layer.
    GPT-2 uses Conv1D with weight shape (n_embd, n_embd) — i.e. (in, out).
    Head h occupies input slice  [h * head_dim : (h+1) * head_dim].

    This is the standard Wanda criterion (Sun et al., 2023) applied at head
    granularity rather than individual weight level.
    """
    n_layers = model.config.n_layer          # 12
    n_heads  = model.config.n_head           # 12
    head_dim = model.config.n_embd // n_heads  # 64

    # Register hooks to capture c_proj inputs
    captured = {l: [] for l in range(n_layers)}

    def make_capture_hook(l):
        def hook(module, inp, out):
            captured[l].append(inp[0].detach())   # (batch, seq, n_embd)
        return hook

    hooks = [
        model.h[l].attn.c_proj.register_forward_hook(make_capture_hook(l))
        for l in range(n_layers)
    ]

    with torch.no_grad():
        for text in texts:
            model(**encode(tokenizer, text))

    for h in hooks:
        h.remove()

    wanda = np.zeros((n_layers, n_heads))

    for l in range(n_layers):
        W = model.h[l].attn.c_proj.weight  # (n_embd, n_embd)

        for head in range(n_heads):
            s, e = head * head_dim, (head + 1) * head_dim
            # Average the norms of the head slice over sequence positions across all texts
            norms = [x[:, :, s:e].norm(dim=-1).mean().item() for x in captured[l]]
            input_norm = float(np.mean(norms))
            weight_mag = W[s:e, :].abs().mean().item()
            wanda[l, head] = input_norm * weight_mag

    return wanda


# ── Bridge Score ──────────────────────────────────────────────────────────────

def _ds_hook(store):
    """Captures hidden state output of a transformer block."""
    def hook(module, inp, out):
        store.append(out[0].detach().float())  # (batch, seq, hidden)
    return hook


def compute_bridge(model, tokenizer, texts, domain=""):
    """
    Per-head bridge score = mean downstream L2 shift under head ablation.

    Ablation mechanism:
      Zero head h's slice in the INPUT to c_proj at layer l.
      c_proj input shape: (batch, seq, n_embd).
      Head h occupies positions [h*head_dim : (h+1)*head_dim].
      This removes head h's contribution before the output projection mixes heads.

    Bridge score for head (l, h) =
        mean over probe texts of
            mean over downstream layers dl in [l+1, n_layers) of
                L2( mean_seq(hidden_baseline[dl]) − mean_seq(hidden_ablated[dl]) )

    High bridge score = ablating this head strongly disrupts downstream representations.
    The novel claim: some heads have low Wanda but high Bridge — "invisible bridges."
    """
    n_layers = model.config.n_layer
    n_heads  = model.config.n_head
    head_dim = model.config.n_embd // n_heads
    bridge   = np.zeros((n_layers, n_heads))

    total = (n_layers - 1) * n_heads  # last layer has no downstream
    done  = 0

    # Pre-compute baseline activations for all layers across all texts
    print(f"    Pre-computing baseline activations for {len(texts)} texts...")
    base_acts_all = []
    for text in texts:
        inp = encode(tokenizer, text)
        base_acts = []
        base_hooks = [
            model.h[dl].register_forward_hook(_ds_hook(base_acts))
            for dl in range(n_layers)
        ]
        with torch.no_grad():
            model(**inp)
        for bh in base_hooks:
            bh.remove()
        base_acts_all.append(base_acts)

    for l in range(n_layers - 1):
        for h in range(n_heads):
            s, e = h * head_dim, (h + 1) * head_dim
            text_deltas = []

            for text_idx, text in enumerate(texts):
                inp = encode(tokenizer, text)
                base_acts = base_acts_all[text_idx][l + 1:]

                # ── Ablated forward pass ──────────────────────────────────────
                def ablate(module, pre_inp):
                    """Zero head h's slice in c_proj input."""
                    x = pre_inp[0].clone()
                    x[:, :, s:e] = 0.0
                    return (x,)

                abl_acts    = []
                abl_hook    = model.h[l].attn.c_proj \
                                   .register_forward_pre_hook(ablate)
                abl_ds_hooks = [
                    model.h[dl].register_forward_hook(_ds_hook(abl_acts))
                    for dl in range(l + 1, n_layers)
                ]
                with torch.no_grad():
                    model(**inp)
                abl_hook.remove()
                for ah in abl_ds_hooks:
                    ah.remove()

                # ── Compute delta ─────────────────────────────────────────────
                if base_acts and abl_acts:
                    # Average over sequence, stack over downstream layers
                    B = torch.stack([t.mean(dim=1) for t in base_acts])  # (ds_l, batch, hidden)
                    A = torch.stack([t.mean(dim=1) for t in abl_acts])
                    delta = (B - A).norm(dim=-1).mean().item()
                    text_deltas.append(delta)

            bridge[l, h] = float(np.mean(text_deltas)) if text_deltas else 0.0

            done += 1
            if done % 24 == 0 or done == total:
                print(f"  [{domain:8s}] {done:3d}/{total} heads processed")

    return bridge


# -- Utilities -----------------------------------------------------------------

def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


# -- Scatter Plot --------------------------------------------------------------

def plot(wanda, bridge_dict, tw=0.35, tb=0.65, model=None):
    """
    Four-quadrant scatter per domain.
    tw = Wanda threshold (normalized), tb = Bridge threshold (normalized).

    Quadrants:
      Top-left    (low W, high B) -> ! Invisible Bridges   <- target of Phase 1
      Top-right   (high W, high B) -> Load-bearing & utilized
      Bottom-right (high W, low B) -> Redundant workhorse
      Bottom-left (low W, low B)  -> Safe to prune
    """
    domains  = list(bridge_dict.keys())
    n_layers = wanda.shape[0]
    w_norm   = norm01(wanda)
    cmap     = plt.cm.plasma

    fig, axes = plt.subplots(1, len(domains), figsize=(6.5 * len(domains), 5.5))
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        b_norm = norm01(bridge_dict[domain])
        layer_ids = np.repeat(np.arange(n_layers), wanda.shape[1])
        colors = cmap(layer_ids / (n_layers - 1))

        ax.scatter(w_norm.flatten(), b_norm.flatten(),
                   c=colors, alpha=0.75, s=60, edgecolors='none')

        ax.axvline(tw, color='crimson', ls='--', lw=0.9, alpha=0.5)
        ax.axhline(tb, color='crimson', ls='--', lw=0.9, alpha=0.5)

        quadrant_labels = [
            (0.02, 0.90, 'top',    'crimson',  'bold',   '!  Invisible\n    Bridges'),
            (0.64, 0.90, 'top',    'darkgreen', 'normal', 'Load-bearing\n& utilized'),
            (0.64, 0.03, 'bottom', 'slategray', 'normal', 'Redundant\nworkhorse'),
            (0.02, 0.03, 'bottom', 'steelblue', 'normal', 'Safe to\nprune'),
        ]
        for x_, y_, va_, col_, wt_, txt_ in quadrant_labels:
            ax.text(x_, y_, txt_, transform=ax.transAxes, fontsize=7.5,
                    color=col_, va=va_, fontweight=wt_)

        n_inv = int(((w_norm < tw) & (b_norm > tb)).sum())
        model_name = "GPT-2 Medium" if (model and model.config.n_layer == 24) else "GPT-2 Small"
        ax.set_title(f"{model_name} - {domain} probes (Candidates: {n_inv})", fontsize=10, pad=15)
        ax.set_xlabel("Wanda Score (normalized)", fontsize=9)
        ax.set_ylabel("Bridge Score (normalized)", fontsize=9)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[-1], label='Layer (early -> late)')
    model_name = "GPT-2 Medium" if (model and model.config.n_layer == 24) else "GPT-2 Small"
    n_layers = model.config.n_layer if model else 12
    n_heads = model.config.n_head if model else 12
    prefix = "medium_" if (model and model.config.n_layer == 24) else "small_"
    plt.suptitle(f"Phase 0: Wanda vs Bridge Score - {model_name} ({n_layers*n_heads} heads)",
                 fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{prefix}phase0_scatter.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved -> {prefix}phase0_scatter.png")


# -- Console report ------------------------------------------------------------

def report(wanda, bridge_dict, tw=0.35, tb=0.65):
    w_norm = norm01(wanda)
    print("\n" + "=" * 58)
    print("  INVISIBLE BRIDGE CANDIDATES  (Low Wanda  +  High Bridge)")
    print("=" * 58)

    any_found = False
    for domain, bridge in bridge_dict.items():
        b_norm = norm01(bridge)
        rows   = [
            (l, h)
            for l in range(wanda.shape[0])
            for h in range(wanda.shape[1])
            if w_norm[l, h] < tw and b_norm[l, h] > tb
        ]

        print(f"\n  [{domain.upper()}]  -  {len(rows)} candidate(s)")
        if not rows:
            print("    (none - World A may hold for this domain)")
        else:
            any_found = True
            top = sorted(rows, key=lambda x: -bridge[x[0], x[1]])[:8]
            for (l, h) in top:
                print(f"    Layer {l:02d}  Head {h:02d}  |"
                      f"  Wanda={w_norm[l,h]:.3f}  |  Bridge={b_norm[l,h]:.3f}")

    print("\n" + "=" * 58)
    if any_found:
        print("  [OK] World B signal detected.")
        print("  Next -> Phase 1: ablate candidates, measure domain-specific")
        print("          capability degradation on perplexity + task benchmarks.")
    else:
        print("  World A holds - no invisible bridges found.")
        print("  Next -> Revisit bridge metric (try PID-based uniqueness,")
        print("          domain-conditional probing, or finer granularity).")
    print("=" * 58)


# -- Correlation check ---------------------------------------------------------

def correlation_check(wanda, bridge_dict):
    """
    Quick Pearson correlation between Wanda and Bridge per domain.
    Strong correlation (|r| > 0.7) -> Bridge ~ Wanda; metric adds little.
    Weak correlation (|r| < 0.3)   -> genuinely different signal.
    """
    w_flat = norm01(wanda).flatten()
    print("\n  Pearson r(Wanda, Bridge) per domain:")
    for domain, bridge in bridge_dict.items():
        b_flat = norm01(bridge).flatten()
        r = float(np.corrcoef(w_flat, b_flat)[0, 1])
        verdict = ("~ independent signal [OK]" if abs(r) < 0.3
                   else "moderate overlap" if abs(r) < 0.7
                   else "high overlap - Bridge ~ Wanda")
        print(f"    {domain:10s}: r = {r:+.3f}  ->  {verdict}")


# -- Main ----------------------------------------------------------------------

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)
    all_texts = [t for v in PROBES.values() for t in v]

    # -- Step 1: Wanda (once, on all texts) -----------------------------------
    print("\n[1/2] Computing Wanda scores...")
    wanda = compute_wanda(model, tokenizer, all_texts)
    print(f"  shape: {wanda.shape}   range: [{wanda.min():.4f}, {wanda.max():.4f}]")

    # -- Step 2: Bridge (per domain) -------------------------------------------
    print("\n[2/2] Computing Bridge scores (per domain)...")
    bridge_dict = {}
    for domain, texts in PROBES.items():
        print(f"\n  Domain: {domain}")
        bridge_dict[domain] = compute_bridge(model, tokenizer, texts, domain)

    # -- Analysis --------------------------------------------------------------
    print("\n-- Signal independence check --")
    correlation_check(wanda, bridge_dict)

    print("\n-- Generating scatter plots --")
    plot(wanda, bridge_dict, model=model)

    report(wanda, bridge_dict)


if __name__ == "__main__":
    main()