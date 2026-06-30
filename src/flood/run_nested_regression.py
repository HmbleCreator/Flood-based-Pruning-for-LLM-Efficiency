"""
src/flood/run_nested_regression.py
==================================
1. Evaluates individual feature R^2 scores for Broadcast, Receiver, Injector, and Betweenness.
2. Performs nested model F-tests to prove that adding components increases explanatory power.
3. Evaluates cross-architecture transferability: Trains regression on GPT-2 + OPT and predicts
   Pythia-70m causal damage, reporting rank correlation.
"""

import numpy as np
import networkx as nx
import os
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

def load_variables(model_name):
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    prefix = f"{safe_name}_"
    causal_damage = np.load(f"{prefix}causal_damage.npy")
    pagerank_rev_dict = np.load(f"{prefix}pagerank_rev.npy", allow_pickle=True).item()
    pagerank_dict = np.load(f"{prefix}pagerank.npy", allow_pickle=True).item()
    injector_scores = np.load(f"{prefix}injector_scores.npy")
    dep_matrix = np.load(f"{prefix}dependency_matrix.npy")
    
    n_total = len(causal_damage)
    from layout import get_model_layout
    layout = get_model_layout(n_total)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    
    broadcast = np.array([pagerank_rev_dict[lbl] for lbl in head_labels])
    receiver = np.array([pagerank_dict[lbl] for lbl in head_labels])
    injector = injector_scores
    
    G = nx.DiGraph()
    for i, src in enumerate(head_labels):
        for j, dst in enumerate(head_labels):
            weight = dep_matrix[i, j]
            if weight > 1e-5:
                G.add_edge(src, dst, weight=weight)
    bet_dict = nx.betweenness_centrality(G, weight='weight')
    betweenness = np.array([bet_dict.get(lbl, 0.0) for lbl in head_labels])
    
    return causal_damage, broadcast, receiver, injector, betweenness

def standardize(x):
    std_val = np.std(x)
    if std_val < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std_val

def main():
    print("=== Nested Model Comparison & R-squared Decompositions ===")
    
    for model_name in ["gpt2", "facebook/opt-125m"]:
        print(f"\nModel: {model_name.upper()}")
        y, bcast, recv, inj, bet = load_variables(model_name)
        
        y_std = standardize(y)
        bcast_std = standardize(bcast)
        recv_std = standardize(recv)
        inj_std = standardize(inj)
        bet_std = standardize(bet)
        
        # 1. Single Feature R^2
        r2_bcast = sm.OLS(y_std, sm.add_constant(bcast_std)).fit().rsquared
        r2_recv = sm.OLS(y_std, sm.add_constant(recv_std)).fit().rsquared
        r2_inj = sm.OLS(y_std, sm.add_constant(inj_std)).fit().rsquared
        r2_bet = sm.OLS(y_std, sm.add_constant(bet_std)).fit().rsquared
        
        # 2. Nested Models
        # Model 1: Broadcast
        X1 = sm.add_constant(bcast_std)
        fit1 = sm.OLS(y_std, X1).fit()
        
        # Model 2: Broadcast + Receiver
        X2 = sm.add_constant(np.stack([bcast_std, recv_std], axis=1))
        fit2 = sm.OLS(y_std, X2).fit()
        
        # Model 3: Broadcast + Receiver + Injector
        X3 = sm.add_constant(np.stack([bcast_std, recv_std, inj_std], axis=1))
        fit3 = sm.OLS(y_std, X3).fit()
        
        # Model 4: Broadcast + Receiver + Injector + Betweenness (Full)
        X4 = sm.add_constant(np.stack([bcast_std, recv_std, inj_std, bet_std], axis=1))
        fit4 = sm.OLS(y_std, X4).fit()
        
        print("  Single Feature Explanatory Power:")
        print(f"    - Broadcast-only R^2: {r2_bcast:.4f}")
        print(f"    - Receiver-only  R^2: {r2_recv:.4f}")
        print(f"    - Injector-only  R^2: {r2_inj:.4f}")
        print(f"    - Betweenness-only R^2: {r2_bet:.4f}")
        
        print("  Nested Model Progression:")
        print(f"    - Model 1 (Bcast):          R^2 = {fit1.rsquared:.4f}")
        print(f"    - Model 2 (+Recv):          R^2 = {fit2.rsquared:.4f}")
        print(f"    - Model 3 (+Inj):           R^2 = {fit3.rsquared:.4f}")
        print(f"    - Model 4 (+Bet - Full):    R^2 = {fit4.rsquared:.4f}")
        
        # Calculate F-test p-value manually for nested models
        # Model 1 vs Model 2
        df_num = 1
        df_denom = len(y_std) - 3
        f_stat = ((fit2.ssr - fit1.ssr) / -df_num) / (fit2.ssr / df_denom)
        p_val_1_2 = stats.f.sf(f_stat, df_num, df_denom)
        print(f"    - F-test: Model 1 vs Model 2 p-value: {p_val_1_2:.6f} {'* Significant' if p_val_1_2 < 0.05 else ''}")
        
        # Model 2 vs Model 3
        f_stat = ((fit3.ssr - fit2.ssr) / -df_num) / (fit3.ssr / (len(y_std) - 4))
        p_val_2_3 = stats.f.sf(f_stat, df_num, len(y_std) - 4)
        print(f"    - F-test: Model 2 vs Model 3 p-value: {p_val_2_3:.6f} {'* Significant' if p_val_2_3 < 0.05 else ''}")
        
    # ── 3. Cross-Architecture Transferability ──
    print("\n=== Cross-Architecture Transferability (Generalization) ===")
    
    # Train regression on GPT-2 + OPT combined, predict Pythia-70m
    y_gpt, bcast_gpt, recv_gpt, inj_gpt, bet_gpt = load_variables("gpt2")
    y_opt, bcast_opt, recv_opt, inj_opt, bet_opt = load_variables("facebook/opt-125m")
    y_pyt, bcast_pyt, recv_pyt, inj_pyt, bet_pyt = load_variables("eleutherai/pythia-70m")
    
    # Standardize and stack train sets
    X_gpt = np.stack([standardize(bcast_gpt), standardize(recv_gpt), standardize(inj_gpt), standardize(bet_gpt)], axis=1)
    X_opt = np.stack([standardize(bcast_opt), standardize(recv_opt), standardize(inj_opt), standardize(bet_opt)], axis=1)
    
    X_train = np.vstack([X_gpt, X_opt])
    y_train = np.concatenate([standardize(y_gpt), standardize(y_opt)])
    
    # Train OLS model
    X_train_const = sm.add_constant(X_train)
    model_train = sm.OLS(y_train, X_train_const).fit()
    
    # Test set: Pythia-70m
    X_test = np.stack([standardize(bcast_pyt), standardize(recv_pyt), standardize(inj_pyt), standardize(bet_pyt)], axis=1)
    X_test_const = sm.add_constant(X_test)
    
    # Predict
    y_pred = model_train.predict(X_test_const)
    y_true_std = standardize(y_pyt)
    
    # Evaluate rank correlation
    spearman_rho, spearman_p = stats.spearmanr(y_pred, y_true_std)
    pearson_r, pearson_p = stats.pearsonr(y_pred, y_true_std)
    
    print(f"Regression Trained on GPT-2 + OPT-125m.")
    print(f"Evaluating zero-shot prediction on Pythia-70m (48 heads):")
    print(f"  - Spearman Rank Correlation: rho = {spearman_rho:.4f} | p-value = {spearman_p:.6f} {'* SIGNIFICANT' if spearman_p < 0.05 else ''}")
    print(f"  - Pearson Correlation:       r   = {pearson_r:.4f} | p-value = {pearson_p:.6f}")

if __name__ == "__main__":
    main()
