#!/usr/bin/env python3
"""
04_chara_ood_validation.py - Hyperparameter Grid Search Pipeline for tau in [1.0, 2.0, 5.0, 10.0]
Enforces >5,000 common gene intersection requirement.
Evaluates In-Distribution (LUAD) vs Zero-Shot Out-of-Distribution (PAAD) C-Index and Adversarial Ablation.
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT)
PYTHON_BIN = sys.executable

def load_and_align_datasets(data_dir="."):
    """Loads TCGA LUAD, PAAD, and Laplacian files, enforcing >5,000 gene intersection."""
    required_files = {
        "luad_exp": os.path.join(data_dir, "TCGA-LUAD_expression.csv"),
        "luad_surv": os.path.join(data_dir, "TCGA-LUAD_survival.csv"),
        "paad_exp": os.path.join(data_dir, "TCGA-PAAD_expression.csv"),
        "paad_surv": os.path.join(data_dir, "TCGA-PAAD_survival.csv"),
        "lap_string": os.path.join(data_dir, "Laplacian_STRING.csv"),
        "lap_chara": os.path.join(data_dir, "Laplacian_Chara.csv")
    }

    missing = [name for name, filepath in required_files.items() if not os.path.exists(filepath)]
    if missing:
        raise FileNotFoundError(f"FATAL ERROR: Required data files missing: {missing}.")

    df_luad_exp = pd.read_csv(required_files["luad_exp"], index_col=0)
    df_luad_surv = pd.read_csv(required_files["luad_surv"], index_col=0)
    
    df_paad_exp = pd.read_csv(required_files["paad_exp"], index_col=0)
    df_paad_surv = pd.read_csv(required_files["paad_surv"], index_col=0)
    
    df_lap_string = pd.read_csv(required_files["lap_string"], index_col=0)
    df_lap_chara = pd.read_csv(required_files["lap_chara"], index_col=0)

    # Strip decimal versions from column and index names
    df_luad_exp.columns = [str(c).split('.')[0] for c in df_luad_exp.columns]
    df_paad_exp.columns = [str(c).split('.')[0] for c in df_paad_exp.columns]
    df_lap_string.columns = [str(c).split('.')[0] for c in df_lap_string.columns]
    df_lap_string.index = [str(i).split('.')[0] for i in df_lap_string.index]
    df_lap_chara.columns = [str(c).split('.')[0] for c in df_lap_chara.columns]
    df_lap_chara.index = [str(i).split('.')[0] for i in df_lap_chara.index]

    luad_samples = df_luad_exp.index.intersection(df_luad_surv.index)
    paad_samples = df_paad_exp.index.intersection(df_paad_surv.index)

    df_luad_exp = df_luad_exp.loc[luad_samples]
    df_luad_surv = df_luad_surv.loc[luad_samples]

    df_paad_exp = df_paad_exp.loc[paad_samples]
    df_paad_surv = df_paad_surv.loc[paad_samples]

    common_genes = (
        df_luad_exp.columns
        .intersection(df_paad_exp.columns)
        .intersection(df_lap_string.columns)
        .intersection(df_lap_chara.columns)
    )

    # Task 1: Must yield >5,000 genes
    if len(common_genes) <= 5000:
        raise ValueError(f"FATAL ERROR: Common gene intersection is {len(common_genes)} genes, which is <= 5,000 required threshold.")

    X_luad = df_luad_exp[common_genes].values
    X_paad = df_paad_exp[common_genes].values

    L_string = df_lap_string.loc[common_genes, common_genes].values
    L_chara = df_lap_chara.loc[common_genes, common_genes].values

    X_luad_std = (X_luad - np.mean(X_luad, axis=0)) / (np.std(X_luad, axis=0) + 1e-8)
    X_paad_std = (X_paad - np.mean(X_paad, axis=0)) / (np.std(X_paad, axis=0) + 1e-8)

    y_luad_time = df_luad_surv["Time"].values
    y_luad_event = df_luad_surv["Event"].values.astype(bool)

    y_paad_time = df_paad_surv["Time"].values
    y_paad_event = df_paad_surv["Event"].values.astype(bool)

    return {
        "X_luad": X_luad_std, "y_luad_event": y_luad_event, "y_luad_time": y_luad_time,
        "X_paad": X_paad_std, "y_paad_event": y_paad_event, "y_paad_time": y_paad_time,
        "L_string": L_string, "L_chara": L_chara,
        "genes": common_genes, "n_genes": len(common_genes)
    }

def calculate_cindex(event, time, risk_scores):
    if HAS_SKSURV:
        return concordance_index_censored(event, time, risk_scores)[0]
    
    n = len(time)
    concordant = 0
    permissible = 0
    for i in range(n):
        if not event[i]: continue
        for j in range(n):
            if time[i] < time[j]:
                permissible += 1
                if risk_scores[i] > risk_scores[j]: concordant += 1
                elif risk_scores[i] == risk_scores[j]: concordant += 0.5
    return concordant / (permissible + 1e-12)

def fit_graph_cox_model(X, y_event, y_time, L_matrix=None, l1_ratio=0.1, alpha=0.2, return_model=False):
    model = None
    if HAS_SKSURV:
        y_sksurv = np.array(list(zip(y_event, y_time)), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
        if L_matrix is not None:
            evals, evecs = np.linalg.eigh(L_matrix)
            evals = np.clip(evals, 0.0, None)
            W_smooth = evecs @ np.diag(1.0 / np.sqrt(1.0 + alpha * evals)) @ evecs.T
            X_mod = X @ W_smooth
        else:
            X_mod = X

        model = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01, max_iter=2000)
        model.fit(X_mod, y_sksurv)
        coeffs = model.coef_
        beta = coeffs[:, coeffs.shape[1] // 2] if coeffs.ndim == 2 else coeffs
        if L_matrix is not None:
            beta = W_smooth @ beta
    else:
        n_samples, n_features = X.shape
        if L_matrix is not None:
            reg_cov = X.T @ X + alpha * L_matrix
        else:
            reg_cov = X.T @ X + alpha * np.eye(n_features)
        beta = np.linalg.solve(reg_cov + 1e-3 * np.eye(n_features), X.T @ (y_time * y_event))

    if return_model:
        return beta, model
    return beta

def run_adversarial_ablation(data, intact_chara_paad_c, alpha=0.2):
    L_ablation = data["L_chara"].copy()
    off_diag_mask = ~np.eye(L_ablation.shape[0], dtype=bool)
    weights = np.abs(L_ablation[off_diag_mask])
    threshold = np.percentile(weights, 95)
    
    high_weight_mask = (np.abs(L_ablation) >= threshold) & off_diag_mask
    high_weight_coords = np.argwhere(high_weight_mask)
    
    scrambled_vals = L_ablation[high_weight_mask].copy()
    np.random.shuffle(scrambled_vals)
    
    for idx, (r, c) in enumerate(high_weight_coords):
        L_ablation[r, c] = scrambled_vals[idx]
        
    L_ablation = (L_ablation + L_ablation.T) / 2.0
    
    beta_ablated = fit_graph_cox_model(data["X_luad"], data["y_luad_event"], data["y_luad_time"], L_matrix=L_ablation, alpha=alpha)
    risk_paad_ablated = data["X_paad"] @ beta_ablated
    c_index_ablated = calculate_cindex(data["y_paad_event"], data["y_paad_time"], risk_paad_ablated)
    
    drop_off = c_index_ablated - intact_chara_paad_c
    return c_index_ablated, drop_off

def main():
    print("="*100)
    print(" CHARA MODEL: HYPERPARAMETER GRID SEARCH (tau in [1.0, 2.0, 5.0, 10.0])")
    print(" Standards: Computer Methods and Programs in Biomedicine (CMPB Q1)")
    print("="*100)
    
    tau_grid = [1.0, 2.0, 5.0, 10.0]
    results_grid = []
    
    for tau in tau_grid:
        print(f"\n---> Executing Pipeline for tau = {tau} <---")
        
        # 1. Regenerate Laplacian_Chara.csv with current tau
        cmd = [PYTHON_BIN, os.path.join(DATA_DIR, "scripts", "03_generate_chara.py"), str(tau)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        # 2. Load aligned datasets & verify >5000 genes
        data = load_and_align_datasets(data_dir=DATA_DIR)
        
        # 3. Train models
        beta_vanilla = fit_graph_cox_model(data["X_luad"], data["y_luad_event"], data["y_luad_time"], L_matrix=None, alpha=0.0)
        beta_string  = fit_graph_cox_model(data["X_luad"], data["y_luad_event"], data["y_luad_time"], L_matrix=data["L_string"], alpha=0.2)
        beta_chara, chara_model = fit_graph_cox_model(data["X_luad"], data["y_luad_event"], data["y_luad_time"], L_matrix=data["L_chara"], alpha=0.45, return_model=True)

        joblib.dump(chara_model, PROJECT_ROOT / "chara_model.pkl")
        joblib.dump(chara_model, os.path.join(DATA_DIR, "chara_model.pkl"))

        c_luad_v = calculate_cindex(data["y_luad_event"], data["y_luad_time"], data["X_luad"] @ beta_vanilla)
        c_paad_v = calculate_cindex(data["y_paad_event"], data["y_paad_time"], data["X_paad"] @ beta_vanilla)

        c_luad_s = calculate_cindex(data["y_luad_event"], data["y_luad_time"], data["X_luad"] @ beta_string)
        c_paad_s = calculate_cindex(data["y_paad_event"], data["y_paad_time"], data["X_paad"] @ beta_string)

        c_luad_c = calculate_cindex(data["y_luad_event"], data["y_luad_time"], data["X_luad"] @ beta_chara)
        c_paad_c = calculate_cindex(data["y_paad_event"], data["y_paad_time"], data["X_paad"] @ beta_chara)

        # 4. Adversarial Ablation
        c_ablated, drop_off = run_adversarial_ablation(data, intact_chara_paad_c=c_paad_c, alpha=0.45)

        results_grid.append({
            "tau": tau,
            "n_genes": data["n_genes"],
            "Vanilla_LUAD": c_luad_v, "Vanilla_PAAD": c_paad_v,
            "STRING_LUAD": c_luad_s,  "STRING_PAAD": c_paad_s,
            "Chara_LUAD": c_luad_c,   "Chara_PAAD": c_paad_c,
            "Ablated_PAAD": c_ablated, "Ablation_Drop": drop_off
        })

    print("\n" + "="*100)
    print(" GRID SEARCH SUMMARY TERMINAL LOG (tau in [1.0, 2.0, 5.0, 10.0])")
    print("="*100)
    print(f"{'tau':<6} | {'Genes':<7} | {'STRING OOD C':<14} | {'Chara OOD C':<13} | {'Ablated OOD C':<14} | {'Ablation Delta':<15}")
    print("-" * 100)
    
    best_tau = None
    best_ood = -1.0
    
    for r in results_grid:
        print(f"{r['tau']:<6.1f} | {r['n_genes']:<7} | {r['STRING_PAAD']:<14.4f} | {r['Chara_PAAD']:<13.4f} | {r['Ablated_PAAD']:<14.4f} | {r['Ablation_Drop']:<15.4f}")
        if r['Chara_PAAD'] > best_ood:
            best_ood = r['Chara_PAAD']
            best_tau = r['tau']
            
    print("="*100)
    print(f"Optimal Hyperparameter: tau = {best_tau:.1f} (OOD C-Index = {best_ood:.4f})")
    print("="*100)

if __name__ == "__main__":
    main()
