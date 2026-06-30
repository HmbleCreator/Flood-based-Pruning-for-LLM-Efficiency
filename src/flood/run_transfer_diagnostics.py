"""
src/flood/run_transfer_diagnostics.py
=====================================
1. Analyzes the Pearson/Spearman discrepancy via predicted vs. observed values.
2. Computes 95% Bootstrap Confidence Intervals for OLS regression coefficients (1000 trials).
3. Performs a calibration prompt sensitivity analysis.
"""

import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from run_nested_regression import load_variables, standardize

def compute_bootstrap_cis(X, y, num_trials=1000):
    n = len(y)
    boot_coefs = []
    
    # Add constant
    X_const = sm.add_constant(X)
    
    np.random.seed(42)
    for _ in range(num_trials):
        idx = np.random.choice(n, size=n, replace=True)
        y_boot = y[idx]
        X_boot = X_const[idx]
        
        # Fit OLS
        try:
            fit = sm.OLS(y_boot, X_boot).fit()
            boot_coefs.append(fit.params)
        except:
            continue
            
    boot_coefs = np.array(boot_coefs)
    
    cis = {}
    features = ["Intercept", "Broadcast", "Receiver", "Injector", "Betweenness"]
    for i, feat in enumerate(features):
        ci_lo = np.percentile(boot_coefs[:, i], 2.5)
        ci_hi = np.percentile(boot_coefs[:, i], 97.5)
        cis[feat] = (ci_lo, ci_hi)
        
    return cis

def main():
    print("=== OLS Regression Bootstrap Confidence Intervals (1,000 trials) ===")
    
    for model_name in ["gpt2", "facebook/opt-125m"]:
        y, bcast, recv, inj, bet = load_variables(model_name)
        X = np.stack([standardize(bcast), standardize(recv), standardize(inj), standardize(bet)], axis=1)
        y_std = standardize(y)
        
        cis = compute_bootstrap_cis(X, y_std)
        print(f"\nModel: {model_name.upper()}")
        for feat, (lo, hi) in cis.items():
            print(f"  {feat:<12} -> Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
            
    # ── Pearson/Spearman Discrepancy Diagnostics ──
    print("\n=== Pearson vs. Spearman Discrepancy Analysis (Pythia-70m) ===")
    y_gpt, bcast_gpt, recv_gpt, inj_gpt, bet_gpt = load_variables("gpt2")
    y_opt, bcast_opt, recv_opt, inj_opt, bet_opt = load_variables("facebook/opt-125m")
    y_pyt, bcast_pyt, recv_pyt, inj_pyt, bet_pyt = load_variables("eleutherai/pythia-70m")
    
    # Train model
    X_gpt = np.stack([standardize(bcast_gpt), standardize(recv_gpt), standardize(inj_gpt), standardize(bet_gpt)], axis=1)
    X_opt = np.stack([standardize(bcast_opt), standardize(recv_opt), standardize(inj_opt), standardize(bet_opt)], axis=1)
    X_train = np.vstack([X_gpt, X_opt])
    y_train = np.concatenate([standardize(y_gpt), standardize(y_opt)])
    
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    
    # Predict Pythia
    X_test = np.stack([standardize(bcast_pyt), standardize(recv_pyt), standardize(inj_pyt), standardize(bet_pyt)], axis=1)
    y_pred = model.predict(sm.add_constant(X_test))
    y_true = standardize(y_pyt)
    
    # Look at top predicted vs actual heads
    from layout import get_model_layout
    layout = get_model_layout(len(y_pyt))
    n_heads = layout["heads"]
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(layout["layers"]) for h in range(n_heads)]
    
    sorted_idx = np.argsort(y_true)[::-1]
    print("\nTop 5 actual damaged heads in Pythia-70m:")
    for idx in sorted_idx[:5]:
        print(f"  Head {head_labels[idx]} -> Actual Std Damage: {y_true[idx]:.4f} | Predicted: {y_pred[idx]:.4f}")
        
    print("\nTop 5 predicted damaged heads in Pythia-70m:")
    sorted_pred_idx = np.argsort(y_pred)[::-1]
    for idx in sorted_pred_idx[:5]:
        print(f"  Head {head_labels[idx]} -> Predicted: {y_pred[idx]:.4f} | Actual Std Damage: {y_true[idx]:.4f}")
        
    # Standard deviation of actual vs predicted
    print(f"\nPythia-70m True Damage Std: {np.std(y_true):.4f} | Range: [{np.min(y_true):.4f}, {np.max(y_true):.4f}]")
    print(f"Pythia-70m Pred Damage Std: {np.std(y_pred):.4f} | Range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")
    
    # ── Calibration Sensitivity Analysis ──
    print("\n=== Calibration Prompt Sensitivity Analysis (GPT-2 Small) ===")
    # Load different prompt-level causal damages
    # Prompt 0, 1, 2 exist as files
    p0_dmg = np.load("gpt2_prompt_0_causal_damage.npy")
    p1_dmg = np.load("gpt2_prompt_1_causal_damage.npy")
    p2_dmg = np.load("gpt2_prompt_2_causal_damage.npy")
    
    # Compare correlation between single prompt damage vs 3-prompt mean damage
    mean_3 = (p0_dmg + p1_dmg + p2_dmg) / 3.0
    
    corr_p0_p1 = stats.spearmanr(p0_dmg, p1_dmg)[0]
    corr_p0_p2 = stats.spearmanr(p0_dmg, p2_dmg)[0]
    corr_p1_p2 = stats.spearmanr(p1_dmg, p2_dmg)[0]
    
    corr_p0_mean = stats.spearmanr(p0_dmg, mean_3)[0]
    corr_p1_mean = stats.spearmanr(p1_dmg, mean_3)[0]
    corr_p2_mean = stats.spearmanr(p2_dmg, mean_3)[0]
    
    print("Spearman Correlation between single prompts:")
    print(f"  - Prompt 0 vs Prompt 1: {corr_p0_p1:.4f}")
    print(f"  - Prompt 0 vs Prompt 2: {corr_p0_p2:.4f}")
    print(f"  - Prompt 1 vs Prompt 2: {corr_p1_p2:.4f}")
    print("\nSpearman Correlation between single prompt and 3-prompt aggregate:")
    print(f"  - Prompt 0 vs 3-Prompt Mean: {corr_p0_mean:.4f}")
    print(f"  - Prompt 1 vs 3-Prompt Mean: {corr_p1_mean:.4f}")
    print(f"  - Prompt 2 vs 3-Prompt Mean: {corr_p2_mean:.4f}")

if __name__ == "__main__":
    main()
