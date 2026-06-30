"""
src/flood/benchmark.py
======================
Runs the main pruning perplexity sweep across various budgets
comparing FLOOD against Random, Magnitude, Attention Entropy, and Bridge-Only baselines,
as well as component ablations.
Includes a topology-first evaluation and fine-tuning recovery experiment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import numpy as np
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from score import compute_routing_importance_scores
from prune import prune_attention_heads
from evaluate import run_topological_evaluation, copy_model_weights

# Evaluation prompts: combined 15 domain probes + 24 phase 1 EVAL prompts
EVAL_TEXTS = [
    # domain probes
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1",
    "class LinkedList:\n    def __init__(self):\n        self.head = None",
    "for epoch in range(num_epochs):\n    optimizer.zero_grad()\n    loss.backward()",
    "import torch\nmodel = torch.nn.Linear(128, 64)\noutput = model(x)",
    "SELECT user_id, COUNT(*) FROM events WHERE date > '2024' GROUP BY user_id",
    "The fundamental theorem of calculus connects differentiation and integration.",
    "Euler's identity states that e raised to i times pi plus one equals zero.",
    "The gradient of a scalar field points in the direction of steepest ascent.",
    "A matrix is invertible if and only if its determinant is nonzero.",
    "The law of large numbers guarantees convergence of sample means to the true mean.",
    "The ambassador carefully chose her words before addressing the assembly.",
    "Despite initial setbacks, the expedition eventually reached the summit.",
    "The novel explores themes of identity, memory, and the passage of time.",
    "She noticed the subtle shift in his expression when the name was mentioned.",
    "The committee reached a consensus after hours of careful deliberation.",
    # phase 1 EVAL
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, item):\n        self.items.append(item)",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.dropna(inplace=True)\nprint(df.describe())",
    "for i, row in enumerate(matrix):\n    for j, val in enumerate(row):\n        if val > threshold:\n            result.append((i, j))",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as resp:\n            return await resp.json()",
    "SELECT a.name, COUNT(b.id) as total FROM users a\nJOIN orders b ON a.id = b.user_id\nGROUP BY a.name HAVING total > 5",
    "def compute_loss(logits, targets):\n    return F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))",
    "The derivative of the natural logarithm of x with respect to x is one over x.",
    "By the Pythagorean theorem, the square of the hypotenuse equals the sum of the squares of the other two sides.",
    "The binomial theorem states that the expansion of x plus y to the power n has n plus one terms.",
    "Integration by parts states that the integral of u times dv equals uv minus the integral of v times du.",
    "The central limit theorem guarantees that the sampling distribution of the mean approaches normality as sample size increases.",
    "Linear independence means no vector in the set can be written as a linear combination of the others.",
    "The Gaussian integral of e to the negative x squared from negative infinity to infinity equals the square root of pi.",
    "A prime number is a natural number greater than one with no positive divisors other than one and itself.",
    "The discovery of penicillin by Alexander Fleming in 1928 revolutionized the treatment of bacterial infections.",
    "Despite the complexity of the negotiations, the two parties eventually reached a mutually beneficial agreement.",
    "The Renaissance was a period of profound cultural and intellectual transformation in European history.",
    "She walked slowly through the autumn leaves, lost in thought about the events of the past several months.",
    "The telescope revealed structures in the distant galaxy that had never been observed by astronomers before.",
    "The committee's report highlighted several areas where policy reform could improve public health outcomes.",
    "After years of research, the team finally published their findings in a leading peer-reviewed scientific journal.",
    "In the early hours of the morning, the city was quiet except for the distant sound of rain on the pavement."
]

FT_TRAIN_TEXTS = [
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
    "Let V be a vector space over the field F. A subset S of V is called a subspace if V contains the zero vector, is closed under vector addition, and is closed under scalar multiplication.",
    "The rain in Spain stays mainly in the plain, which has historically been a fertile agricultural region suited for growing olives and wheat.",
    "In deep learning, batch normalization is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs.",
    "INSERT INTO customers (customer_name, contact_name, country) VALUES ('Cardinal', 'Tom B. Erichsen', 'Norway');"
]

def load_base_model(model_name="gpt2", path=None):
    src = path
    if src is None:
        if "gpt2" in model_name.lower():
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
        if src is None:
            src = model_name
            
    is_local = "gpt2" in src.lower()
    tokenizer = AutoTokenizer.from_pretrained(src, local_files_only=is_local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, local_files_only=is_local
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts):
    """Compute document-level mean perplexity for consistent comparisons."""
    doc_ppls = []
    device = model.device
    with torch.no_grad():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inp = {k: v.to(device) for k, v in inp.items()}
            labels = inp["input_ids"].clone()
            out = model(**inp, labels=labels)
            doc_ppls.append(np.exp(out.loss.item()))
    if len(doc_ppls) == 0:
        return float('inf')
    return float(np.mean(doc_ppls))

def calculate_perplexity_bootstrap(model, tokenizer, texts, num_resamples=100, seed=42):
    """Compute perplexity over texts and bootstrap document-level perplexities to estimate 95% confidence interval."""
    doc_ppls = []
    device = model.device
    with torch.no_grad():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inp = {k: v.to(device) for k, v in inp.items()}
            labels = inp["input_ids"].clone()
            out = model(**inp, labels=labels)
            
            # Perplexity of document i: exp(loss)
            doc_ppls.append(np.exp(out.loss.item()))
            
    np.random.seed(seed)
    boot_means = []
    for _ in range(num_resamples):
        sample = np.random.choice(doc_ppls, size=len(doc_ppls), replace=True)
        boot_means.append(np.mean(sample))
        
    base_ppl = float(np.mean(doc_ppls))
    if len(boot_means) > 0:
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
    else:
        ci_lower, ci_upper = base_ppl, base_ppl
    return base_ppl, ci_lower, ci_upper

def run_paired_bootstrap_test(model_flood, model_baseline, tokenizer, texts, num_resamples=1000, seed=42):
    """
    Runs a paired bootstrap hypothesis test:
      H0: PPL(Baseline) - PPL(FLOOD) <= 0
      H1: PPL(Baseline) - PPL(FLOOD) > 0
    Returns the bootstrap p-value.
    """
    device = model_flood.device
    ppl_flood = []
    ppl_base = []
    with torch.no_grad():
        for text in texts:
            inp = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inp = {k: v.to(device) for k, v in inp.items()}
            labels = inp["input_ids"].clone()
            
            out_f = model_flood(**inp, labels=labels)
            out_b = model_baseline(**inp, labels=labels)
            
            ppl_flood.append(np.exp(out_f.loss.item()))
            ppl_base.append(np.exp(out_b.loss.item()))
            
    diffs = np.array(ppl_base) - np.array(ppl_flood)
    
    np.random.seed(seed)
    boot_means = []
    for _ in range(num_resamples):
        sample = np.random.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
        
    # Proportion of bootstrap means where baseline does NOT have higher perplexity than FLOOD
    p_value = float(np.mean(np.array(boot_means) <= 0.0))
    return p_value, float(np.mean(ppl_flood)), float(np.mean(ppl_base))

def compute_gradient_importance(model, tokenizer, texts, n_layers, n_heads):
    """
    Computes First-Order Taylor Expansion Importance Score:
      I(h) = sum_{tokens} | activation * gradient |
    """
    device = next(model.parameters()).device
    importance = {(l, h): 0.0 for l in range(n_layers) for h in range(n_heads)}
    
    # Save requires_grad state and enable gradients temporarily
    grad_states = {}
    for name, param in model.named_parameters():
        grad_states[name] = param.requires_grad
        param.requires_grad = True
        
    d_model = model.config.hidden_size
    d_head = d_model // n_heads
    
    activations = {}
    gradients = {}
    
    def get_forward_hook(l_idx):
        def hook(module, input, output):
            # For Conv1D or Linear, input[0] has shape (batch, seq, d_model)
            activations[l_idx] = input[0].detach().clone()
        return hook
        
    def get_backward_hook(l_idx):
        def hook(module, grad_input, grad_output):
            gradients[l_idx] = grad_input[0].detach().clone()
        return hook
        
    from evaluate import find_attn_out_modules
    layer_mapping = find_attn_out_modules(model)
    hooks = []
    for l in range(n_layers):
        if l not in layer_mapping:
            continue
        name, c_proj = layer_mapping[l]
        hooks.append(c_proj.register_forward_hook(get_forward_hook(l)))
        hooks.append(c_proj.register_full_backward_hook(get_backward_hook(l)))
        
    # Accumulate gradients over evaluation texts
    model.eval()
    for text in texts[:5]:
        inp = tokenizer(text, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        labels = inp["input_ids"].clone()
        
        model.zero_grad()
        out = model(**inp, labels=labels)
        loss = out.loss
        loss.backward()
        
        for l in range(n_layers):
            if l not in activations or l not in gradients:
                continue
            act = activations[l].squeeze(0)  # (seq, d_model)
            grad = gradients[l].squeeze(0)  # (seq, d_model)
            
            for h in range(n_heads):
                act_h = act[:, h * d_head : (h + 1) * d_head]
                grad_h = grad[:, h * d_head : (h + 1) * d_head]
                score = torch.sum(torch.abs(act_h * grad_h)).item()
                importance[(l, h)] += score
                
    # Cleanup hooks and restore requires_grad states
    for hk in hooks:
        hk.remove()
    for name, param in model.named_parameters():
        param.requires_grad = grad_states[name]
        
    return importance

def compute_attention_entropy(model, tokenizer, texts, n_layers, n_heads):
    """Compute average attention entropy for each head."""
    entropy_accum = np.zeros((n_layers, n_heads))
    counts = 0
    device = model.device
    
    with torch.no_grad():
        for text in texts[:10]:  # Use subset of 10 texts for faster evaluation
            inp = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inp = {k: v.to(device) for k, v in inp.items()}
            
            outputs = model(**inp, output_attentions=True)
            attentions = outputs.attentions  # Tuple of shape (batch, heads, seq, seq)
            
            for l in range(n_layers):
                att = attentions[l].squeeze(0)  # Shape: (heads, seq, seq)
                eps = 1e-12
                entropy = - (att * torch.log(att + eps)).sum(dim=-1).mean(dim=-1) # Shape: (heads,)
                entropy_accum[l] += entropy.cpu().numpy()
            counts += 1
            
    return entropy_accum / max(1, counts)

def compute_magnitude_scores(model, n_layers, n_heads):
    """Compute L2 norm of attention output projection weights for each head."""
    from evaluate import find_attn_out_modules
    layer_mapping = find_attn_out_modules(model)
    scores = {}
    d_model = getattr(model.config, "hidden_size", getattr(model.config, "n_embd", None))
    d_head = d_model // n_heads
    
    for l in range(n_layers):
        if l not in layer_mapping:
            continue
        name, c_proj = layer_mapping[l]
        weight = c_proj.weight.data
        is_linear = c_proj.__class__.__name__ == 'Linear'
        
        for h in range(n_heads):
            if is_linear:
                w_slice = weight[:, h * d_head : (h + 1) * d_head]
            else:
                w_slice = weight[h * d_head : (h + 1) * d_head, :]
            scores[(l, h)] = w_slice.norm().item()
            
    return scores

def fine_tune_and_track_recovery(model, tokenizer, train_texts, eval_texts, steps_list=[0, 2, 5, 10, 20], lr=1e-5):
    """
    Fine-tunes the model copy and evaluates perplexity at intermediate steps
    to build a recovery curve.
    """
    ft_model = copy_model_weights(model)
    ppl_history = []
    max_steps = max(steps_list)
    
    if 0 in steps_list:
        ppl_history.append(calculate_perplexity(ft_model, tokenizer, eval_texts))
        
    if max_steps == 0:
        return ppl_history
        
    ft_model.train()
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr)
    device = ft_model.device
    
    for step in range(1, max_steps + 1):
        text = train_texts[(step - 1) % len(train_texts)]
        inp = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inp = {k: v.to(device) for k, v in inp.items()}
        labels = inp["input_ids"].clone()
        
        optimizer.zero_grad()
        out = ft_model(**inp, labels=labels)
        loss = out.loss
        loss.backward()
        optimizer.step()
        
        if step in steps_list:
            ft_model.eval()
            ppl_history.append(calculate_perplexity(ft_model, tokenizer, eval_texts))
            ft_model.train()
            
    return ppl_history

def main():
    parser = argparse.ArgumentParser(description="FLOOD: Routing-Aware Head Pruning Benchmark")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local weights")
    parser.add_argument("--w_bridge", type=float, default=0.50, help="Causal damage weight")
    parser.add_argument("--w_broadcast", type=float, default=0.30, help="Broadcast centrality weight")
    parser.add_argument("--w_injector", type=float, default=0.10, help="Injector alignment weight")
    parser.add_argument("--w_receiver", type=float, default=0.00, help="Receiver centrality weight")
    parser.add_argument("--w_backbone", type=float, default=0.10, help="Continuous backbone weight")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["domain_probes", "wikitext"],
                        help="Evaluation dataset: 'wikitext' (WikiText-2) or 'domain_probes' (built-in texts)")
    args = parser.parse_args()
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading base model & tokenizer...")
    safe_name = args.model.replace("/", "_").replace("-", "_").lower()
    model, tokenizer = load_base_model(args.model, args.model_path)
    
    from layout import get_model_layout
    layout = get_model_layout(args.model)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    n_total_heads = n_layers * n_heads
    print(f"Model has {n_layers} layers and {n_heads} heads ({n_total_heads} total heads).")
    
    # Calculate baseline perplexity
    if args.dataset == "wikitext":
        print("Loading WikiText-2 evaluation set...")
        try:
            from datasets import load_dataset as hf_load
            wikitext = hf_load("wikitext", "wikitext-2-raw-v1", split="test")
            EVAL_TEXTS_RUN = [t for t in wikitext["text"] if len(t.strip()) > 100][:15]
            print(f"  Loaded {len(EVAL_TEXTS_RUN)} WikiText-2 passages for evaluation.")
        except Exception as e:
            print(f"  Warning: Could not load WikiText-2 ({e}). Falling back to domain probes.")
            EVAL_TEXTS_RUN = EVAL_TEXTS
    else:
        EVAL_TEXTS_RUN = EVAL_TEXTS
        print("Using built-in domain probe texts for evaluation.")

    print("\nEvaluating baseline perplexity (0% pruned)...")
    baseline_ppl = calculate_perplexity(model, tokenizer, EVAL_TEXTS_RUN)
    print(f"  Baseline Perplexity: {baseline_ppl:.4f}")
    
    # ── Calculate Scoring Lists ──
    # 1. FLOOD Routing Importance Scores (Full)
    print("Computing FLOOD Routing Importance Scores (Full)...")
    flood_scores = compute_routing_importance_scores(
        args.model, args.w_bridge, args.w_broadcast, args.w_injector, args.w_receiver, args.w_backbone
    )
    flood_ordered = sorted(flood_scores.keys(), key=lambda x: flood_scores[x])
    
    # 2. FLOOD-Simplified (Bridge + Broadcast only: w_bridge=0.67, w_broadcast=0.33)
    print("Computing FLOOD-Simplified (Bridge + Broadcast only) scores...")
    flood_simple_scores = compute_routing_importance_scores(
        args.model, w_bridge=0.67, w_broadcast=0.33, w_injector=0.0, w_receiver=0.0, w_backbone=0.0
    )
    flood_simple_ordered = sorted(flood_simple_scores.keys(), key=lambda x: flood_simple_scores[x])
    
    # 3. Bridge-Only scores (Paper 1 only)
    print("Computing Bridge-Only scores...")
    bridge_only_scores = compute_routing_importance_scores(
        args.model, w_bridge=1.0, w_broadcast=0.0, w_injector=0.0, w_receiver=0.0, w_backbone=0.0
    )
    bridge_ordered = sorted(bridge_only_scores.keys(), key=lambda x: bridge_only_scores[x])
    
    # 4. Broadcast-Only scores (Paper 3 only)
    print("Computing Broadcast-Only scores...")
    broadcast_only_scores = compute_routing_importance_scores(
        args.model, w_bridge=0.0, w_broadcast=1.0, w_injector=0.0, w_receiver=0.0, w_backbone=0.0
    )
    broadcast_ordered = sorted(broadcast_only_scores.keys(), key=lambda x: broadcast_only_scores[x])
    
    # 5. First-Order Taylor Gradient Importance (Michel et al. 2019 baseline)
    print("Computing Gradient Importance (Taylor) scores...")
    grad_scores = compute_gradient_importance(model, tokenizer, EVAL_TEXTS_RUN, n_layers, n_heads)
    grad_ordered = sorted(grad_scores.keys(), key=lambda x: grad_scores[x])
    
    # 6. Magnitude scores
    print("Computing Magnitude Pruning scores...")
    mag_scores = compute_magnitude_scores(model, n_layers, n_heads)
    mag_ordered = sorted(mag_scores.keys(), key=lambda x: mag_scores[x])
    
    # 6. Attention Entropy
    print("Computing Attention Entropy scores...")
    entropy_matrix = compute_attention_entropy(model, tokenizer, EVAL_TEXTS_RUN, n_layers, n_heads)
    entropy_scores = {(l, h): entropy_matrix[l, h] for l in range(n_layers) for h in range(n_heads)}
    entropy_ordered = sorted(entropy_scores.keys(), key=lambda x: entropy_scores[x], reverse=True)
    
    # 7. Ablation studies
    print("Computing FLOOD Ablation Importance Scores...")
    sc_no_bridge = compute_routing_importance_scores(args.model, w_bridge=0.0, w_broadcast=0.3, w_injector=0.2, w_receiver=0.1, w_backbone=0.1)
    ordered_no_bridge = sorted(sc_no_bridge.keys(), key=lambda x: sc_no_bridge[x])
    
    sc_no_broadcast = compute_routing_importance_scores(args.model, w_bridge=0.3, w_broadcast=0.0, w_injector=0.2, w_receiver=0.1, w_backbone=0.1)
    ordered_no_broadcast = sorted(sc_no_broadcast.keys(), key=lambda x: sc_no_broadcast[x])
    
    sc_no_injector = compute_routing_importance_scores(args.model, w_bridge=0.3, w_broadcast=0.3, w_injector=0.0, w_receiver=0.1, w_backbone=0.1)
    ordered_no_injector = sorted(sc_no_injector.keys(), key=lambda x: sc_no_injector[x])
    
    sc_no_receiver = compute_routing_importance_scores(args.model, w_bridge=0.3, w_broadcast=0.3, w_injector=0.2, w_receiver=0.0, w_backbone=0.1)
    ordered_no_receiver = sorted(sc_no_receiver.keys(), key=lambda x: sc_no_receiver[x])
    
    # 8. Random Pruning
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    
    # ── Phase 1: Topology-First Evaluation (at 30% budget) ──
    n_prune_30 = int(n_total_heads * 0.30)
    if n_total_heads <= 144 or torch.cuda.is_available():
        try:
            run_topological_evaluation(
                model, tokenizer,
                flood_ordered[:n_prune_30],
                mag_ordered[:n_prune_30],
                None,
                EVAL_TEXTS_RUN
            )
        except Exception as e:
            print(f"[Warning] Topological evaluation failed: {e}")
    else:
        print("[Info] Skipping slow CPU topological evaluation for larger model.")
        
    # ── Phase 2: Perplexity Budget Sweep ──
    budgets = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    results = {
        "budgets": [0.0] + budgets,
        # Baselines
        "FLOOD": [baseline_ppl],
        "FLOOD-Simplified": [baseline_ppl],
        "Bridge-Only": [baseline_ppl],
        "Broadcast-Only": [baseline_ppl],
        "Gradient-Taylor": [baseline_ppl],
        "Magnitude": [baseline_ppl],
        "Entropy": [baseline_ppl],
        "Random": [baseline_ppl],
        # Confidence Intervals for key methods
        "FLOOD_CIs": [(baseline_ppl, baseline_ppl)],
        "FLOOD-Simplified_CIs": [(baseline_ppl, baseline_ppl)],
        "Magnitude_CIs": [(baseline_ppl, baseline_ppl)],
        # Ablations
        "Ablation-No-Bridge": [baseline_ppl],
        "Ablation-No-Broadcast": [baseline_ppl],
        "Ablation-No-Injector": [baseline_ppl],
        "Ablation-No-Receiver": [baseline_ppl],
    }
    
    for pct in budgets:
        n_to_prune = int(n_total_heads * pct)
        print(f"\nEvaluating Pruning Budget: {pct * 100:.0f}% ({n_to_prune}/{n_total_heads} heads)")
        
        # A. FLOOD (with Bootstrap CIs)
        m_flood = prune_attention_heads(model, flood_ordered[:n_to_prune])
        ppl_val, ci_l, ci_u = calculate_perplexity_bootstrap(m_flood, tokenizer, EVAL_TEXTS_RUN)
        results["FLOOD"].append(ppl_val)
        results["FLOOD_CIs"].append((ci_l, ci_u))
        print(f"  - FLOOD Perplexity: {ppl_val:.4f} [95% CI: {ci_l:.4f}, {ci_u:.4f}]")
        del m_flood
        
        # B. FLOOD-Simplified (with Bootstrap CIs)
        m_flood_simple = prune_attention_heads(model, flood_simple_ordered[:n_to_prune])
        ppl_val_s, ci_l_s, ci_u_s = calculate_perplexity_bootstrap(m_flood_simple, tokenizer, EVAL_TEXTS_RUN)
        results["FLOOD-Simplified"].append(ppl_val_s)
        results["FLOOD-Simplified_CIs"].append((ci_l_s, ci_u_s))
        print(f"  - FLOOD-Simplified Perplexity: {ppl_val_s:.4f} [95% CI: {ci_l_s:.4f}, {ci_u_s:.4f}]")
        del m_flood_simple
        
        # C. Bridge-Only
        m_bridge = prune_attention_heads(model, bridge_ordered[:n_to_prune])
        results["Bridge-Only"].append(calculate_perplexity(m_bridge, tokenizer, EVAL_TEXTS_RUN))
        print(f"  - Bridge-Only Perplexity: {results['Bridge-Only'][-1]:.4f}")
        del m_bridge
        
        # C2. Broadcast-Only
        m_broadcast = prune_attention_heads(model, broadcast_ordered[:n_to_prune])
        results["Broadcast-Only"].append(calculate_perplexity(m_broadcast, tokenizer, EVAL_TEXTS_RUN))
        print(f"  - Broadcast-Only Perplexity: {results['Broadcast-Only'][-1]:.4f}")
        del m_broadcast
        
        # D. Gradient-Taylor
        m_grad = prune_attention_heads(model, grad_ordered[:n_to_prune])
        results["Gradient-Taylor"].append(calculate_perplexity(m_grad, tokenizer, EVAL_TEXTS_RUN))
        print(f"  - Gradient-Taylor Perplexity: {results['Gradient-Taylor'][-1]:.4f}")
        del m_grad
        
        # E. Magnitude (with Bootstrap CIs)
        m_mag = prune_attention_heads(model, mag_ordered[:n_to_prune])
        ppl_val_m, ci_l_m, ci_u_m = calculate_perplexity_bootstrap(m_mag, tokenizer, EVAL_TEXTS_RUN)
        results["Magnitude"].append(ppl_val_m)
        results["Magnitude_CIs"].append((ci_l_m, ci_u_m))
        print(f"  - Magnitude Perplexity: {ppl_val_m:.4f} [95% CI: {ci_l_m:.4f}, {ci_u_m:.4f}]")
        del m_mag
        
        # F. Attention Entropy
        m_ent = prune_attention_heads(model, entropy_ordered[:n_to_prune])
        results["Entropy"].append(calculate_perplexity(m_ent, tokenizer, EVAL_TEXTS_RUN))
        print(f"  - Attention Entropy Perplexity: {results['Entropy'][-1]:.4f}")
        del m_ent
        
        # G. Random
        rand_runs = []
        for seed in [42, 100, 2026]:
            np.random.seed(seed)
            shuffled = list(all_heads)
            np.random.shuffle(shuffled)
            m_rand = prune_attention_heads(model, shuffled[:n_to_prune])
            rand_runs.append(calculate_perplexity(m_rand, tokenizer, EVAL_TEXTS_RUN))
            del m_rand
        ppl_rand_mean = float(np.mean(rand_runs))
        results["Random"].append(ppl_rand_mean)
        print(f"  - Random Perplexity (mean): {ppl_rand_mean:.4f}")
        
        # H. Ablations
        # No-Bridge
        m_nb = prune_attention_heads(model, ordered_no_bridge[:n_to_prune])
        results["Ablation-No-Bridge"].append(calculate_perplexity(m_nb, tokenizer, EVAL_TEXTS_RUN))
        del m_nb
        # No-Broadcast
        m_nbc = prune_attention_heads(model, ordered_no_broadcast[:n_to_prune])
        results["Ablation-No-Broadcast"].append(calculate_perplexity(m_nbc, tokenizer, EVAL_TEXTS_RUN))
        del m_nbc
        # No-Injector
        m_ni = prune_attention_heads(model, ordered_no_injector[:n_to_prune])
        results["Ablation-No-Injector"].append(calculate_perplexity(m_ni, tokenizer, EVAL_TEXTS_RUN))
        del m_ni
        # No-Receiver
        m_nr = prune_attention_heads(model, ordered_no_receiver[:n_to_prune])
        results["Ablation-No-Receiver"].append(calculate_perplexity(m_nr, tokenizer, EVAL_TEXTS_RUN))
        del m_nr

    # ── Paired Bootstrap Significance Testing (at 30% budget) ──
    print("\n" + "="*80)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING (At 30% Pruning Budget)")
    print("="*80)
    try:
        m_flood_simple_30 = prune_attention_heads(model, flood_simple_ordered[:n_prune_30])
        m_grad_30 = prune_attention_heads(model, grad_ordered[:n_prune_30])
        m_mag_30 = prune_attention_heads(model, mag_ordered[:n_prune_30])
        
        p_val_grad, f_mean, b_mean = run_paired_bootstrap_test(m_flood_simple_30, m_grad_30, tokenizer, EVAL_TEXTS_RUN)
        print(f"FLOOD-Simplified vs. Gradient-Taylor:")
        print(f"  - FLOOD-Simplified Mean PPL: {f_mean:.4f}")
        print(f"  - Gradient-Taylor Mean PPL: {b_mean:.4f}")
        print(f"  - Paired Bootstrap P-Value: {p_val_grad:.6f} " + ("(Statistically Significant, p < 0.05)" if p_val_grad < 0.05 else "(Not Statistically Significant)"))
        
        p_val_mag, f_mean, m_mean = run_paired_bootstrap_test(m_flood_simple_30, m_mag_30, tokenizer, EVAL_TEXTS_RUN)
        print(f"FLOOD-Simplified vs. Magnitude:")
        print(f"  - FLOOD-Simplified Mean PPL: {f_mean:.4f}")
        print(f"  - Magnitude Mean PPL: {m_mean:.4f}")
        print(f"  - Paired Bootstrap P-Value: {p_val_mag:.6f} " + ("(Statistically Significant, p < 0.05)" if p_val_mag < 0.05 else "(Not Statistically Significant)"))
        
        results["Significance_P_Value_vs_Gradient"] = p_val_grad
        results["Significance_P_Value_vs_Magnitude"] = p_val_mag
        
        del m_flood_simple_30
        del m_grad_30
        del m_mag_30
    except Exception as e:
        print(f"[Warning] Paired bootstrap test failed: {e}")

    # ── Phase 3: Fine-Tuning Recovery Sweep (at 30% budget) ──
    print("\n" + "="*80)
    print("RUNNING FINE-TUNING RECOVERY EVALUATION (At 30% Pruning Budget)")
    print("="*80)
    
    m_flood_30 = prune_attention_heads(model, flood_ordered[:n_prune_30])
    m_flood_simple_30 = prune_attention_heads(model, flood_simple_ordered[:n_prune_30])
    m_mag_30 = prune_attention_heads(model, mag_ordered[:n_prune_30])
    m_grad_30 = prune_attention_heads(model, grad_ordered[:n_prune_30])
    
    np.random.seed(42)
    shuffled = list(all_heads)
    np.random.shuffle(shuffled)
    m_rand_30 = prune_attention_heads(model, shuffled[:n_prune_30])
    
    steps_eval = [0, 2, 5, 10, 20]
    print(f"Fine-tuning models and tracking perplexity at steps: {steps_eval}...")
    
    print("Fine-tuning FLOOD model...")
    flood_rec = fine_tune_and_track_recovery(m_flood_30, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    print("Fine-tuning FLOOD-Simplified model...")
    flood_simple_rec = fine_tune_and_track_recovery(m_flood_simple_30, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    print("Fine-tuning Magnitude model...")
    mag_rec = fine_tune_and_track_recovery(m_mag_30, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    print("Fine-tuning Gradient-Taylor model...")
    grad_rec = fine_tune_and_track_recovery(m_grad_30, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    print("Fine-tuning Random model...")
    rand_rec = fine_tune_and_track_recovery(m_rand_30, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    print("Fine-tuning Baseline (unpruned control)...")
    baseline_rec = fine_tune_and_track_recovery(model, tokenizer, FT_TRAIN_TEXTS, EVAL_TEXTS_RUN, steps_list=steps_eval)
    
    results["Recovery_Steps"] = steps_eval
    results["Recovery_FLOOD"] = flood_rec
    results["Recovery_FLOOD-Simplified"] = flood_simple_rec
    results["Recovery_Magnitude"] = mag_rec
    results["Recovery_Gradient-Taylor"] = grad_rec
    results["Recovery_Random"] = rand_rec
    results["Recovery_Baseline"] = baseline_rec
    
    print("\nFine-tuning Recovery Results:")
    for idx, st in enumerate(steps_eval):
        print(f"  Step {st:02d} -> FLOOD PPL: {flood_rec[idx]:.4f} | FLOOD-Simplified PPL: {flood_simple_rec[idx]:.4f} | Mag PPL: {mag_rec[idx]:.4f} | Grad PPL: {grad_rec[idx]:.4f} | Rand PPL: {rand_rec[idx]:.4f} | Baseline PPL: {baseline_rec[idx]:.4f}")
        
    # Save perplexity benchmark results
    fig_dir = "paper/figures"
    os.makedirs(fig_dir, exist_ok=True)
    json_path = os.path.join(fig_dir, f"{safe_name}_flood_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark perplexity results -> {json_path}")
    
    # ── Call Visualizers ──
    try:
        from visualize import plot_perplexity_curves, plot_ablation_curves, plot_recovery_curves
        plot_perplexity_curves(results, safe_name)
        plot_ablation_curves(results, safe_name)
        plot_recovery_curves(results, safe_name)
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")

if __name__ == "__main__":
    main()
