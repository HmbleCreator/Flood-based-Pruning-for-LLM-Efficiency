"""
src/flood/test_role_significance.py
===================================
Permutation test to evaluate the statistical significance of routing role overlaps
(Bridge vs. Broadcaster vs. Receiver) against a random selection null hypothesis.
"""

import numpy as np

# Observed values from validation results
OBSERVED = {
    "gpt2": {
        "n_total": 144,
        "k": 43,
        "bridge_bcast": 67.4,
        "bridge_recv": 9.3,
        "bcast_recv": 0.0,
        "triple": 0.0
    },
    "pythia-70m": {
        "n_total": 48,
        "k": 14,
        "bridge_bcast": 64.3,
        "bridge_recv": 0.0,
        "bcast_recv": 0.0,
        "triple": 0.0
    }
}

NUM_PERMUTATIONS = 10000

def run_permutation_test(model_name, stats):
    n_total = stats["n_total"]
    k = stats["k"]
    
    print(f"\n=== Permutation Test for {model_name} (N={n_total}, k={k}, {NUM_PERMUTATIONS} trials) ===")
    
    # Generate null distributions
    null_bridge_bcast = []
    null_bridge_recv = []
    null_bcast_recv = []
    null_triple = []
    
    indices = np.arange(n_total)
    
    for _ in range(NUM_PERMUTATIONS):
        s_bridge = set(np.random.choice(indices, size=k, replace=False))
        s_bcast  = set(np.random.choice(indices, size=k, replace=False))
        s_recv   = set(np.random.choice(indices, size=k, replace=False))
        
        null_bridge_bcast.append(len(s_bridge & s_bcast) / k * 100)
        null_bridge_recv.append(len(s_bridge & s_recv) / k * 100)
        null_bcast_recv.append(len(s_bcast & s_recv) / k * 100)
        null_triple.append(len(s_bridge & s_bcast & s_recv) / k * 100)
        
    null_bridge_bcast = np.array(null_bridge_bcast)
    null_bridge_recv = np.array(null_bridge_recv)
    null_bcast_recv = np.array(null_bcast_recv)
    null_triple = np.array(null_triple)
    
    # Calculate p-values
    # For high overlap (Bridge-Bcast), p-value is P(Null >= Observed)
    p_bridge_bcast = np.mean(null_bridge_bcast >= stats["bridge_bcast"])
    
    # For low overlap (Bcast-Recv, Triple), p-value is P(Null <= Observed)
    p_bcast_recv = np.mean(null_bcast_recv <= stats["bcast_recv"])
    p_triple = np.mean(null_triple <= stats["triple"])
    
    # For Bridge-Recv
    mean_null_br = np.mean(null_bridge_recv)
    if stats["bridge_recv"] > mean_null_br:
        p_bridge_recv = np.mean(null_bridge_recv >= stats["bridge_recv"])
        direction = "higher"
    else:
        p_bridge_recv = np.mean(null_bridge_recv <= stats["bridge_recv"])
        direction = "lower"
        
    print(f"  Bridge-Broadcaster Overlap:")
    print(f"    Observed: {stats['bridge_bcast']:.1f}% | Null Mean: {np.mean(null_bridge_bcast):.1f}% ± {np.std(null_bridge_bcast):.1f}%")
    print(f"    p-value (Null >= Observed): {p_bridge_bcast:.6f} {'*** SIGNIFICANT' if p_bridge_bcast < 0.01 else ''}")
    
    print(f"  Bridge-Receiver Overlap:")
    print(f"    Observed: {stats['bridge_recv']:.1f}% | Null Mean: {np.mean(null_bridge_recv):.1f}% ± {np.std(null_bridge_recv):.1f}%")
    print(f"    p-value (Null {direction} than Observed): {p_bridge_recv:.6f} {'*** SIGNIFICANT' if p_bridge_recv < 0.05 else ''}")
    
    print(f"  Broadcaster-Receiver Overlap:")
    print(f"    Observed: {stats['bcast_recv']:.1f}% | Null Mean: {np.mean(null_bcast_recv):.1f}% ± {np.std(null_bcast_recv):.1f}%")
    print(f"    p-value (Null <= Observed): {p_bcast_recv:.6f} {'*** SIGNIFICANT (Conserved Segregation)' if p_bcast_recv < 0.05 else ''}")
    
    print(f"  Triple Hub Overlap:")
    print(f"    Observed: {stats['triple']:.1f}% | Null Mean: {np.mean(null_triple):.1f}% ± {np.std(null_triple):.1f}%")
    print(f"    p-value (Null <= Observed): {p_triple:.6f} {'*** SIGNIFICANT (Conserved Segregation)' if p_triple < 0.05 else ''}")

def main():
    for model, stats in OBSERVED.items():
        run_permutation_test(model, stats)

if __name__ == "__main__":
    main()
