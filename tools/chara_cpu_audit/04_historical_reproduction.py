#!/usr/bin/env python3
"""
tools/chara_cpu_audit/04_historical_reproduction.py
Reproduces and deconstructs every historical result reported in Chara:
1. frontier_benchmark_results.csv (LUAD 75/25 split, seed 4337)
2. Zero-shot external validation on GSE31210
3. Out-of-distribution transfer on TCGA-PAAD and grid-search target leakage
4. Synthetic CAGPR simulation results (-17.92% MSE reduction, +28.14% Jaccard gain)
5. Analysis of test-time graph transformation discrepancy (X vs X @ W)
"""

import sys
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter
import torch
from torch import nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
HORIZONS = np.array([365.0, 1095.0, 1825.0])
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def deepsurv_risk(x_train, y_train, x_test):
    torch.manual_seed(4337)
    network = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-3)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    order = np.argsort(-y_train["time"])
    events = torch.tensor(y_train["event"][order].astype(np.float32))
    for _ in range(120):
        scores = network(x_tensor[order]).flatten()
        log_risk = torch.logcumsumexp(scores, dim=0)
        loss = -((scores - log_risk) * events).sum() / events.sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return network(torch.tensor(x_test, dtype=torch.float32)).flatten().numpy()

def evaluate_metrics(name, risk, train_y, test_y):
    c = float(concordance_index_censored(test_y["event"], test_y["time"], risk)[0])
    try:
        auc, _ = cumulative_dynamic_auc(train_y, test_y, risk, HORIZONS)
        auc_1y, auc_3y, auc_5y = float(auc[0]), float(auc[1]), float(auc[2])
    except Exception as e:
        auc_1y, auc_3y, auc_5y = np.nan, np.nan, np.nan
    return {"model": name, "c_index": c, "auc_1y": auc_1y, "auc_3y": auc_3y, "auc_5y": auc_5y}

def reproduce_frontier_benchmark():
    print("[INFO] Reproducing frontier_benchmark_results.csv...")
    x_df = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    y_df = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    ids = x_df.index.intersection(y_df.index)
    x_num = x_df.loc[ids].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    event = y_df.loc[ids, "Event"].astype(bool).to_numpy()
    time = y_df.loc[ids, "Time"].to_numpy(float) * 365.0
    
    y = np.array(list(zip(event, time)), dtype=[("event", "?"), ("time", "<f8")])
    train_idx, test_idx = train_test_split(np.arange(len(x_num)), test_size=0.25, random_state=4337, stratify=event)
    train_y, test_y = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    xz = scaler.fit_transform(x_num)
    x_train, x_test = xz[train_idx], xz[test_idx]

    # 1. Elastic Net Coxnet
    cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=3000).fit(x_train, train_y)
    cox_valid = np.flatnonzero(np.sum(cox.coef_ != 0, axis=0))
    cox_risk = x_test @ cox.coef_[:, cox_valid[-1]]
    res_cox = evaluate_metrics("Elastic Net Coxnet", cox_risk, train_y, test_y)

    # 2. Random Survival Forest
    forest = RandomSurvivalForest(n_estimators=300, min_samples_leaf=8, random_state=4337, n_jobs=4).fit(x_train, train_y)
    res_rsf = evaluate_metrics("Random Survival Forest", -forest.predict(x_test), train_y, test_y)

    # 3. DeepSurv
    res_deepsurv = evaluate_metrics("DeepSurv", deepsurv_risk(x_train, train_y, x_test), train_y, test_y)

    # 4. Clinical Cox-PH with hardcoded constants (The artifact)
    clinical = pd.DataFrame({"duration": train_y["time"], "event": train_y["event"].astype(int), "age": 65.0, "gender": 0.0, "stage": 2.0})
    clinical_test = clinical.iloc[:len(test_idx)].copy()
    try:
        clinical_model = CoxPHFitter(penalizer=0.1).fit(clinical, "duration", "event")
        clinical_risk = -clinical_model.predict_partial_hazard(clinical_test).to_numpy()
    except Exception:
        clinical_risk = np.zeros(len(test_idx), dtype=float)
    res_clinical_dummy = evaluate_metrics("Clinical Cox-PH (Constant Dummy)", clinical_risk, train_y, test_y)

    # 5. Chara (from bundle chara_model_4337.pkl)
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    chara_coef = bundle["model"].coef_[:, bundle["alpha_index"]]
    chara_features = bundle["features"]
    chara_x = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0).loc[:, chara_features].loc[ids]
    chara_xz = StandardScaler().fit_transform(chara_x.apply(pd.to_numeric, errors="coerce").fillna(0.0))
    
    # As evaluated in benchmark_frontiers.py: raw chara_xz[test_idx] @ chara_coef
    res_chara_raw = evaluate_metrics("Chara (As Evaluated: raw X)", chara_xz[test_idx] @ chara_coef, train_y, test_y)

    # Compare with test-time graph smoothing: chara_xz @ W @ chara_coef
    L_chara = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0).loc[chara_features, chara_features].to_numpy(float)
    evals, evecs = np.linalg.eigh((L_chara + L_chara.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-0.1 * evals)) @ evecs.T
    chara_smooth_risk = (chara_xz[test_idx] @ W) @ chara_coef
    res_chara_smooth = evaluate_metrics("Chara (With Test-Time W filter)", chara_smooth_risk, train_y, test_y)

    historical_csv = pd.read_csv(ROOT / "frontier_benchmark_results.csv").to_dict(orient="records")

    return {
        "historical_csv": historical_csv,
        "reproduced_benchmarks": [
            res_cox,
            res_rsf,
            res_deepsurv,
            res_clinical_dummy,
            res_chara_raw,
            res_chara_smooth
        ],
        "dummy_clinical_variance": float(np.var(clinical_risk)),
        "chara_nonzero_coefficients": int(np.count_nonzero(chara_coef)),
        "chara_alpha_index": int(bundle["alpha_index"]),
        "chara_alpha_value": float(bundle["model"].alphas_[bundle["alpha_index"]])
    }

def reproduce_gse31210_zeroshot():
    print("[INFO] Reproducing GSE31210 zero-shot validation...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
    v09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v09)

    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    model, genes = bundle["model"], list(bundle["features"])
    gse, survival = v09.parse_geo_gse31210()
    external = v09.load_gse31210_expression(gse).reindex(columns=genes)
    external = external.apply(lambda c: pd.to_numeric(c, errors="coerce")).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = min(len(external), len(survival))
    X = external.iloc[:n].to_numpy(dtype=np.float64)
    x_scaled = StandardScaler().fit_transform(X)
    
    coef_matrix = np.asarray(model.coef_, dtype=np.float64)
    opt_idx = int(bundle["alpha_index"])
    optimal_betas = coef_matrix[:, opt_idx]
    
    # 1. Without W filter (as in 10d_zeroshot_validation.py)
    risk_raw = np.dot(x_scaled, optimal_betas)
    event = survival.iloc[:n]["Event"].astype(bool).to_numpy()
    time = survival.iloc[:n]["TimeMonths"].astype(float).to_numpy()
    c_raw = float(concordance_index_censored(event, time, risk_raw)[0])

    # 2. With W filter
    L_chara = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0).loc[genes, genes].to_numpy(float)
    evals, evecs = np.linalg.eigh((L_chara + L_chara.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-0.1 * evals)) @ evecs.T
    risk_smoothed = np.dot(x_scaled @ W, optimal_betas)
    c_smoothed = float(concordance_index_censored(event, time, risk_smoothed)[0])

    return {
        "n_patients": n,
        "n_genes": len(genes),
        "alpha_index": opt_idx,
        "c_index_raw_expression": c_raw,
        "c_index_smoothed_expression": c_smoothed,
        "event_count": int(event.sum()),
        "censoring_rate": float(1.0 - event.mean())
    }

def reproduce_paad_ood():
    print("[INFO] Reproducing TCGA-PAAD OOD validation and auditing tau grid search...")
    genes_4337 = pd.read_csv(ROOT / "intersecting_genes_4337.txt", header=None)[0].astype(str).tolist()
    luad_x = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    luad_y = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    paad_x = pd.read_csv(ROOT / "TCGA-PAAD_expression.csv", index_col=0)
    paad_y = pd.read_csv(ROOT / "TCGA-PAAD_survival.csv", index_col=0)

    luad_ids = luad_x.index.intersection(luad_y.index)
    paad_ids = paad_x.index.intersection(paad_y.index)

    X_luad = luad_x.loc[luad_ids, genes_4337].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    X_paad = paad_x.loc[paad_ids, genes_4337].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)

    X_luad_std = StandardScaler().fit_transform(X_luad)
    X_paad_std = StandardScaler().fit_transform(X_paad)

    y_luad_e = luad_y.loc[luad_ids, "Event"].astype(bool).to_numpy()
    y_luad_t = luad_y.loc[luad_ids, "Time"].to_numpy(float) * 365.0
    y_paad_e = paad_y.loc[paad_ids, "Event"].astype(bool).to_numpy()
    y_paad_t = paad_y.loc[paad_ids, "Time"].to_numpy(float) * 365.0

    # Test chara_model_4337 directly on PAAD
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    chara_coef = bundle["model"].coef_[:, bundle["alpha_index"]]
    paad_risk = X_paad_std @ chara_coef
    c_paad_4337 = float(concordance_index_censored(y_paad_e, y_paad_t, paad_risk)[0])

    # Also test with W filter
    L_chara = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0).loc[genes_4337, genes_4337].to_numpy(float)
    evals, evecs = np.linalg.eigh((L_chara + L_chara.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-0.1 * evals)) @ evecs.T
    paad_risk_w = (X_paad_std @ W) @ chara_coef
    c_paad_4337_w = float(concordance_index_censored(y_paad_e, y_paad_t, paad_risk_w)[0])

    return {
        "n_paad_samples": len(paad_ids),
        "paad_events": int(y_paad_e.sum()),
        "c_index_paad_raw": c_paad_4337,
        "c_index_paad_smoothed": c_paad_4337_w,
        "target_leakage_note": "In scripts/04_chara_ood_validation.py line 230, hyperparameter tau was selected by maximizing Chara_PAAD (the target evaluation cohort)."
    }

def reproduce_synthetic_cagpr():
    print("[INFO] Reproducing synthetic CAGPR simulation (scripts/01_master_cagpr_validation.py)...")
    PROTEINS = ["KRAS_G12D", "Mut_p53", "PTPN11", "cMYC_MAX"]
    SEEDS = {"KRAS_G12D": 101, "Mut_p53": 202, "PTPN11": 303, "cMYC_MAX": 404}
    PARAMS = {
        "KRAS_G12D": {"base_C": 0.81, "base_S": 0.08, "seed": 101},
        "Mut_p53":   {"base_C": 0.72, "base_S": 0.12, "seed": 202},
        "PTPN11":    {"base_C": 0.74, "base_S": 0.10, "seed": 303},
        "cMYC_MAX":  {"base_C": 0.69, "base_S": 0.15, "seed": 404}
    }
    results = {}

    for prot in PROTEINS:
        seed = SEEDS[prot]
        K, S_len = 500, 40
        np.random.seed(seed)
        W_rand = np.random.uniform(0.1, 0.9, size=(K, K))
        mask = np.random.rand(K, K) < 0.02
        A_global = (W_rand * mask + (W_rand * mask).T) / 2.0
        np.fill_diagonal(A_global, 0.0)
        for i in range(K):
            n = (i + 1) % K
            A_global[i, n] = A_global[n, i] = max(A_global[i, n], 0.5)

        S_nodes = list(range(S_len))
        p = PARAMS[prot]
        np.random.seed(p["seed"])
        C_ij = np.random.normal(loc=p["base_C"], scale=0.1, size=(S_len, S_len))
        C_ij = np.abs((C_ij + C_ij.T) / 2.0)
        np.fill_diagonal(C_ij, 0.0)
        sigma2_ij = np.random.exponential(scale=p["base_S"], size=(S_len, S_len))
        sigma2_ij = (sigma2_ij + sigma2_ij.T) / 2.0
        np.fill_diagonal(sigma2_ij, 0.0)

        tau = 0.5
        A_local = C_ij / (1.0 + tau * sigma2_ij)
        np.fill_diagonal(A_local, 0.0)
        A_local_norm = (A_local - np.min(A_local)) / (np.max(A_local) - np.min(A_local) + 1e-12)
        np.fill_diagonal(A_local_norm, 0.0)

        A_hybrid = A_global.copy()
        A_hybrid[np.ix_(S_nodes, S_nodes)] = A_local_norm

        def get_L_sym(A):
            d = np.sum(A, axis=1)
            di = np.zeros_like(d)
            m = d > 1e-12
            di[m] = 1.0 / np.sqrt(d[m])
            D = np.diag(di)
            return np.eye(A.shape[0]) - D @ A @ D

        L_global = get_L_sym(A_global)
        L_hybrid = get_L_sym(A_hybrid)

        N_samples = 250
        X = np.random.normal(0, 1, size=(N_samples, K))
        beta_true = np.zeros(K)
        beta_true[S_nodes[:10]] = [2.5, -1.8, 3.0, 1.2, -2.2, 1.9, -2.7, 1.5, 2.1, -1.6]
        y = X @ beta_true + np.random.normal(0, 0.5, size=N_samples)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        alpha = 15.0
        b_global = la.solve(X_train.T @ X_train + alpha * L_global, X_train.T @ y_train, assume_a='pos')
        b_hybrid = la.solve(X_train.T @ X_train + alpha * L_hybrid, X_train.T @ y_train, assume_a='pos')

        mse_global = float(mean_squared_error(y_test, X_test @ b_global))
        mse_hybrid = float(mean_squared_error(y_test, X_test @ b_hybrid))
        mse_red_pct = float((mse_global - mse_hybrid) / mse_global * 100.0)

        results[prot] = {
            "mse_global": mse_global,
            "mse_hybrid": mse_hybrid,
            "mse_red_pct": mse_red_pct,
            "note": "Computed on synthetic N=250 Gaussian samples with hardcoded ground-truth beta on S_nodes, not patient survival data"
        }

    return results

def main():
    print("=" * 80)
    print(" CHARA HISTORICAL REPRODUCTION AUDIT")
    print("=" * 80)
    
    frontier_results = reproduce_frontier_benchmark()
    gse_results = reproduce_gse31210_zeroshot()
    paad_results = reproduce_paad_ood()
    synthetic_results = reproduce_synthetic_cagpr()

    consolidated = {
        "frontier_benchmark": frontier_results,
        "gse31210_zeroshot": gse_results,
        "paad_ood": paad_results,
        "synthetic_cagpr": synthetic_results
    }

    out_path = EVIDENCE_DIR / "historical_reproduction.json"
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"[OK] Historical reproduction complete. Saved to: {out_path}")

if __name__ == "__main__":
    main()
