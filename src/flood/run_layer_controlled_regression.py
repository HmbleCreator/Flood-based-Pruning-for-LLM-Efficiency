"""
src/flood/run_layer_controlled_regression.py
============================================
Evaluates whether routing descriptors retain significant explanatory power
over causal damage after controlling for layer index.
"""

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from run_nested_regression import load_variables, standardize

def get_layer_indices(model_name, n_total):
    from layout import get_model_layout
    layout = get_model_layout(n_total)
    n_layers = layout["layers"]
    n_heads = layout["heads"]
    
    # Create layer index for each head
    layer_indices = np.array([l for l in range(n_layers) for h in range(n_heads)])
    return layer_indices

def main():
    print("=== Layer-Controlled Regression Analysis ===")
    
    models = ["gpt2", "facebook/opt-125m", "eleutherai/pythia-160m"]
    
    for model_name in models:
        print(f"\nModel: {model_name.upper()}")
        y, bcast, recv, inj, bet = load_variables(model_name)
        
        y_std = standardize(y)
        bcast_std = standardize(bcast)
        recv_std = standardize(recv)
        inj_std = standardize(inj)
        bet_std = standardize(bet)
        
        layer_indices = get_layer_indices(model_name, len(y))
        layer_std = standardize(layer_indices)
        
        # Fit Model A: Damage ~ Layer
        X_A = sm.add_constant(layer_std)
        fit_A = sm.OLS(y_std, X_A).fit()
        
        # Fit Model B: Damage ~ Layer + Broadcast + Receiver + Injector + Betweenness
        X_B = np.stack([layer_std, bcast_std, recv_std, inj_std, bet_std], axis=1)
        X_B_const = sm.add_constant(X_B)
        fit_B = sm.OLS(y_std, X_B_const).fit()
        
        # F-test comparing Model A vs Model B
        # Model A has 2 parameters (intercept + layer)
        # Model B has 6 parameters (intercept + layer + 4 centralities)
        df_num = 4
        df_denom = len(y_std) - 6
        f_stat = ((fit_B.ssr - fit_A.ssr) / -df_num) / (fit_B.ssr / df_denom)
        p_val = stats.f.sf(f_stat, df_num, df_denom)
        
        # Cohen's f^2 of adding centralities over layer index
        f2 = (fit_B.rsquared - fit_A.rsquared) / (1.0 - fit_B.rsquared)
        
        print(f"  - Model A (Layer only) R^2: {fit_A.rsquared:.4f}")
        print(f"  - Model B (Layer + RIS) R^2: {fit_B.rsquared:.4f}")
        print(f"  - F-test p-value:          {p_val:.6f} {'* Significant' if p_val < 0.05 else ''}")
        print(f"  - Cohen's f^2 (effect):    {f2:.4f}")
        
        print("  - Model B coefficients:")
        features = ["Intercept", "Layer Index", "Broadcast", "Receiver", "Injector", "Betweenness"]
        for i, feat in enumerate(features):
            coef = fit_B.params[i]
            p = fit_B.pvalues[i]
            print(f"    {feat:<12} : coef = {coef:>8.4f} | p = {p:.6f} {'* Significant' if p < 0.05 else ''}")

if __name__ == "__main__":
    main()
