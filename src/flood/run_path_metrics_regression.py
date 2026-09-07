"""
src/flood/run_path_metrics_regression.py
========================================
Tests the REDUNDANCY HYPOTHESIS for the GPT-2 Medium R^2 anomaly (Paper 3,
anomalies registry 4.2): if larger models route information through parallel
multi-hop paths, single-node linear centrality regression fails (R^2 = 7.9%
on GPT-2 Medium), but PATH- and FLOW-based metrics that aggregate over all
routes should not.

Three pre-registered questions:
  Q1 (functional form): does a nonlinear model (random forest, 5-fold CV) on
     the SAME four node centralities beat linear OLS? If yes -> wrong
     functional form. If no -> the information is not in single-node topology.
  Q2 (features): do path/flow metrics (Katz broadcast/receiver over all
     walks, current-flow betweenness, communicability) raise R^2 on GPT-2
     Medium without degrading the small models?
  Q3 (pre-specified success): path-based OLS R^2 on GPT-2 Medium >= 3x the
     linear node-centrality R^2 (>= 0.237).

CPU-only; uses cached .npy artifacts in the repo root.

Usage:
    python -s src/flood/run_path_metrics_regression.py
"""

import os
import sys
import csv

import numpy as np
import networkx as nx
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout import get_model_layout

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

MODELS = [
    "gpt2",
    "gpt2-medium",
    "facebook/opt-125m",
    "eleutherai/pythia-70m",
    "eleutherai/pythia-160m",
]

EDGE_THRESHOLD = 1e-5  # matches run_damage_regression.py graph construction


def standardize(x):
    std_val = np.std(x)
    if std_val < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std_val


def load_model_arrays(model_name):
    safe_name = model_name.replace("/", "_").replace("-", "_").lower()
    prefix = f"{safe_name}_"
    causal_damage = np.load(f"{prefix}causal_damage.npy")
    pagerank_rev = np.load(f"{prefix}pagerank_rev.npy", allow_pickle=True).item()
    pagerank_fwd = np.load(f"{prefix}pagerank.npy", allow_pickle=True).item()
    injector = np.load(f"{prefix}injector_scores.npy")
    dep = np.load(f"{prefix}dependency_matrix.npy")
    return causal_damage, pagerank_rev, pagerank_fwd, injector, dep


def build_graph(dep, head_labels):
    G = nx.DiGraph()
    G.add_nodes_from(head_labels)
    n = len(head_labels)
    for i in range(n):
        for j in range(n):
            w = dep[i, j]
            if w > EDGE_THRESHOLD:
                G.add_edge(head_labels[i], head_labels[j], weight=w)
    return G


def katz_flow_scores(dep, attenuation=0.85):
    """Total multi-hop flow via the Katz resolvent (I - alpha*A)^-1.

    broadcast_k[i] = total attenuated flow LEAVING i over all walks
    receiver_k[j]  = total attenuated flow ENTERING j over all walks
    Unlike PageRank (single stationary distribution), this counts every
    parallel path explicitly -- redundant routes accumulate.
    """
    n = dep.shape[0]
    lam_max = max(np.abs(np.linalg.eigvals(dep)).max(), 1e-12)
    alpha = attenuation / lam_max
    R = np.linalg.inv(np.eye(n) - alpha * dep) - np.eye(n)
    return R.sum(axis=1), R.sum(axis=0)  # broadcast, receiver


def communicability_scores(dep):
    """Subgraph communicability via matrix exponential of the symmetrized,
    spectrally normalized graph: counts weighted closed walks of all lengths."""
    A = (dep + dep.T) / 2.0
    lam_max = max(np.abs(np.linalg.eigvalsh(A)).max(), 1e-12)
    A = A / lam_max
    from scipy.linalg import expm
    E = expm(A)
    return np.diag(E)


def current_flow_betweenness(G, head_labels):
    """Current-flow (random-walk) betweenness on the largest connected
    component of the undirected graph. Unlike shortest-path betweenness, it
    credits ALL paths -- the metric of interest under the redundancy
    hypothesis. Nodes outside the LCC get 0."""
    U = G.to_undirected()
    scores = {lbl: 0.0 for lbl in head_labels}
    components = list(nx.connected_components(U))
    if not components:
        return np.zeros(len(head_labels))
    lcc = max(components, key=len)
    sub = U.subgraph(lcc)
    # invert weights? No: current-flow treats weight as conductance, which is
    # the correct reading of dependency strength.
    cfb = nx.current_flow_betweenness_centrality(sub, weight="weight")
    scores.update(cfb)
    return np.array([scores[lbl] for lbl in head_labels])


def shortest_path_betweenness(G, head_labels):
    bet = nx.betweenness_centrality(G, weight="weight")
    return np.array([bet.get(lbl, 0.0) for lbl in head_labels])


def fit_ols(y, X, feature_names):
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()
    return res


def rf_cv_r2(y, X, seed=0):
    """Nonlinearity ceiling: 5-fold cross-validated R^2 of a random forest."""
    if not HAVE_SKLEARN:
        return float("nan")
    rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def analyze_model(model_name):
    print(f"\n{'=' * 70}")
    print(f"PATH-METRICS ANALYSIS: {model_name.upper()}")
    print(f"{'=' * 70}")

    damage, pr_rev, pr_fwd, injector, dep = load_model_arrays(model_name)
    n_total = len(damage)
    layout = get_model_layout(n_total)
    n_layers, n_heads = layout["layers"], layout["heads"]
    head_labels = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]

    G = build_graph(dep, head_labels)

    # --- node-centrality features (Paper 3 baseline) ---
    broadcast = np.array([pr_rev[lbl] for lbl in head_labels])
    receiver = np.array([pr_fwd[lbl] for lbl in head_labels])
    bet_sp = shortest_path_betweenness(G, head_labels)

    # --- path/flow features ---
    katz_bcast, katz_recv = katz_flow_scores(dep)
    comm = communicability_scores(dep)
    bet_cf = current_flow_betweenness(G, head_labels)

    y = standardize(damage)
    node_feats = {
        "Broadcast (revPR)": standardize(broadcast),
        "Receiver (fwdPR)": standardize(receiver),
        "Injector": standardize(injector),
        "Betweenness (SP)": standardize(bet_sp),
    }
    path_feats = {
        "Katz Broadcast": standardize(katz_bcast),
        "Katz Receiver": standardize(katz_recv),
        "Communicability": standardize(comm),
        "CurrentFlow Betweenness": standardize(bet_cf),
    }

    X_node = np.stack(list(node_feats.values()), axis=1)
    X_path = np.stack(list(path_feats.values()), axis=1)
    X_all = np.concatenate([X_node, X_path], axis=1)

    res_node = fit_ols(y, X_node, list(node_feats))
    res_path = fit_ols(y, X_path, list(path_feats))
    res_all = fit_ols(y, X_all, list(node_feats) + list(path_feats))

    rf_node = rf_cv_r2(y, X_node)
    rf_all = rf_cv_r2(y, X_all)

    print(f"N heads: {n_total} ({n_layers} layers x {n_heads} heads)")
    print(f"\n{'Model':<38} {'R2':>8} {'AdjR2':>8}")
    print("-" * 58)
    print(f"{'OLS node centralities (baseline)':<38} {res_node.rsquared:>8.4f} {res_node.rsquared_adj:>8.4f}")
    print(f"{'OLS path/flow metrics':<38} {res_path.rsquared:>8.4f} {res_path.rsquared_adj:>8.4f}")
    print(f"{'OLS node + path combined':<38} {res_all.rsquared:>8.4f} {res_all.rsquared_adj:>8.4f}")
    print(f"{'RF 5-fold CV, node feats (Q1)':<38} {rf_node:>8.4f}")
    print(f"{'RF 5-fold CV, all feats':<38} {rf_all:>8.4f}")

    print("\nPath-metric OLS coefficients:")
    for i, name in enumerate(path_feats):
        coef = res_path.params[i + 1]
        p = res_path.pvalues[i + 1]
        print(f"  {name:<26} beta={coef:>8.4f}  p={p:.6f}")

    return {
        "model": model_name,
        "n_heads": n_total,
        "ols_node_r2": res_node.rsquared,
        "ols_path_r2": res_path.rsquared,
        "ols_all_r2": res_all.rsquared,
        "ols_all_adj_r2": res_all.rsquared_adj,
        "rf_node_cv_r2": rf_node,
        "rf_all_cv_r2": rf_all,
        "beta_katz_bcast": res_path.params[1],
        "beta_katz_recv": res_path.params[2],
        "beta_comm": res_path.params[3],
        "beta_cf_bet": res_path.params[4],
    }


def main():
    rows = []
    for m in MODELS:
        try:
            rows.append(analyze_model(m))
        except FileNotFoundError as e:
            print(f"\n[skip] {m}: missing artifact ({e})")

    out = "path_metrics_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 70}")
    print("SUMMARY (pre-specified test: GPT-2 Medium path OLS R2 >= 0.237)")
    print(f"{'=' * 70}")
    print(f"{'Model':<24} {'nodeOLS':>8} {'pathOLS':>8} {'allOLS':>8} {'RFnode':>8} {'RFall':>8}")
    for r in rows:
        print(f"{r['model']:<24} {r['ols_node_r2']:>8.4f} {r['ols_path_r2']:>8.4f} "
              f"{r['ols_all_r2']:>8.4f} {r['rf_node_cv_r2']:>8.4f} {r['rf_all_cv_r2']:>8.4f}")
    med = next((r for r in rows if r["model"] == "gpt2-medium"), None)
    if med:
        verdict = "PASS" if max(med["ols_path_r2"], med["ols_all_r2"]) >= 0.237 else "FAIL"
        print(f"\nGPT-2 Medium pre-specified test: {verdict} "
              f"(path={med['ols_path_r2']:.4f}, all={med['ols_all_r2']:.4f}, threshold=0.237)")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
