"""
src/flood/run_damage_regression.py
==================================
Runs a linear regression predicting experimental Causal Damage from 
four structural centralities: Broadcast (rev PageRank), Receiver (fwd PageRank),
Injector (out-degree flow bottleneck), and Betweenness (routing bottlenecks).
Computes standardized coefficients, 95% CIs, R^2, and VIFs.
"""

import numpy as np
import networkx as nx
import os
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_regression_for_model(model_name):
    print(f"\n=======================================================")
    print(f"Regression Analysis for {model_name.upper()}")
    print(f"=======================================================")
    
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    prefix = f"{safe_name}_"
    
    # 1. Load data
    try:
        causal_damage = np.load(f"{prefix}causal_damage.npy")
        pagerank_rev_dict = np.load(f"{prefix}pagerank_rev.npy", allow_pickle=True).item()
        pagerank_dict = np.load(f"{prefix}pagerank.npy", allow_pickle=True).item()
        injector_scores = np.load(f"{prefix}injector_scores.npy")
        dep_matrix = np.load(f"{prefix}dependency_matrix.npy")
    except FileNotFoundError as e:
        print(f"Error loading files for {model_name}: {e}")
        return
        
    n_total = len(causal_damage)
    from layout import get_model_layout
    layout = get_model_layout(n_total)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    
    # 2. Extract centralities
    broadcast = np.array([pagerank_rev_dict[lbl] for lbl in head_labels])
    receiver = np.array([pagerank_dict[lbl] for lbl in head_labels])
    injector = injector_scores
    
    # Compute Node Betweenness Centrality on the dependency graph
    G = nx.DiGraph()
    for i, src in enumerate(head_labels):
        for j, dst in enumerate(head_labels):
            weight = dep_matrix[i, j]
            if weight > 1e-5:
                G.add_edge(src, dst, weight=weight)
                
    # Betweenness
    bet_dict = nx.betweenness_centrality(G, weight='weight')
    betweenness = np.array([bet_dict.get(lbl, 0.0) for lbl in head_labels])
    
    # 3. Standardize variables (mean=0, std=1) for standardized coefficients
    def standardize(x):
        std_val = np.std(x)
        if std_val < 1e-8:
            return x - np.mean(x)
        return (x - np.mean(x)) / std_val
        
    y = standardize(causal_damage)
    x_bcast = standardize(broadcast)
    x_recv = standardize(receiver)
    x_inj = standardize(injector)
    x_bet = standardize(betweenness)
    
    # Design matrix
    X = np.stack([x_bcast, x_recv, x_inj, x_bet], axis=1)
    X_df = sm.add_constant(X)
    
    # Fit OLS
    model = sm.OLS(y, X_df)
    results = model.fit()
    
    # Calculate VIFs
    vifs = []
    # statsmodels add_constant appends constant at column 0
    for idx in range(1, X_df.shape[1]):
        vifs.append(variance_inflation_factor(X_df, idx))
        
    features = ["Broadcast Centrality", "Receiver Centrality", "Injector Centrality", "Betweenness Centrality"]
    
    # Print results
    print(f"R-squared: {results.rsquared:.4f} | Adj R-squared: {results.rsquared_adj:.4f}")
    print(f"\nCoefficients table:")
    print(f"{'Feature':<25} | {'Beta (Std Coeff)':<16} | {'95% Conf. Interval':<20} | {'p-value':<10} | {'VIF':<6}")
    print("-" * 88)
    for i, feat in enumerate(features):
        coef = results.params[i+1]
        ci_lo, ci_hi = results.conf_int()[i+1]
        p_val = results.pvalues[i+1]
        vif = vifs[i]
        print(f"{feat:<25} | {coef:>16.4f} | [{ci_lo:.3f}, {ci_hi:.3f}] | {p_val:>10.6f} | {vif:>6.2f}")
        
    print(f"{'Intercept':<25} | {results.params[0]:>16.4f} | [{results.conf_int()[0][0]:.3f}, {results.conf_int()[0][1]:.3f}] | {results.pvalues[0]:>10.6f} | N/A")

def main():
    run_regression_for_model("gpt2")
    run_regression_for_model("facebook/opt-125m")

if __name__ == "__main__":
    main()
