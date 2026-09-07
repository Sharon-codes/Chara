#!/usr/bin/env python3
"""
tools/chara_cpu_audit/05_fair_graph_benchmark.py
Executes a strictly matched, fair benchmark comparing:
1. No Graph (Vanilla Coxnet)
2. Ordinary STRING Laplacian
3. Historical Chara Laplacian
4. Hash-Weighted STRING (re-synthesized with seed 42)
5. Shuffled STRING Control (permuted node graph)
6. Frozen Historical Chara Model (chara_model_4337.pkl)

Evaluates on:
- TCGA-LUAD (Internal 5-Fold CV & Held-Out Test Set)
- GSE31210 (External Zero-Shot LUAD)
- TCGA-PAAD (Out-of-Distribution Zero-Shot)

Computes Harrell's C-index, Time-Dependent Dynamic AUC (1y, 3y, 5y),
and 2,000 paired bootstrap difference distributions to determine whether
Chara provides any statistically significant improvement over Ordinary STRING or No Graph.
"""

import sys
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = np.array([365.0, 1095.0, 1825.0])

def load_data():
    print("[INFO] Loading datasets and gene intersection (4,337 genes)...")
    genes = pd.read_csv(ROOT / "intersecting_genes_4337.txt", header=None)[0].astype(str).tolist()

    # 1. TCGA-LUAD
    luad_exp = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    luad_surv = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    luad_ids = luad_exp.index.intersection(luad_surv.index)
    X_luad = luad_exp.loc[luad_ids, genes].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y_luad_e = luad_surv.loc[luad_ids, "Event"].astype(bool).to_numpy()
    y_luad_t = luad_surv.loc[luad_ids, "Time"].to_numpy(float) * 365.0
    y_luad_struct = np.array(list(zip(y_luad_e, y_luad_t)), dtype=[("event", "?"), ("time", "<f8")])

    # 2. TCGA-PAAD
    paad_exp = pd.read_csv(ROOT / "TCGA-PAAD_expression.csv", index_col=0)
    paad_surv = pd.read_csv(ROOT / "TCGA-PAAD_survival.csv", index_col=0)
    paad_ids = paad_exp.index.intersection(paad_surv.index)
    X_paad = paad_exp.loc[paad_ids, genes].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    y_paad_e = paad_surv.loc[paad_ids, "Event"].astype(bool).to_numpy()
    y_paad_t = paad_surv.loc[paad_ids, "Time"].to_numpy(float) * 365.0

    # 3. GSE31210
    import importlib.util
    spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
    v09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v09)

    gse, gse_clinical = v09.parse_geo_gse31210()
    gse_df = v09.load_gse31210_expression(gse).reindex(columns=genes)
    gse_df = gse_df.apply(lambda c: pd.to_numeric(c, errors="coerce")).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_gse = min(len(gse_df), len(gse_clinical))
    X_gse = gse_df.iloc[:n_gse].to_numpy(dtype=float)
    y_gse_e = gse_clinical.iloc[:n_gse]["Event"].astype(bool).to_numpy()
    y_gse_t = gse_clinical.iloc[:n_gse]["TimeMonths"].astype(float).to_numpy() * (365.25 / 12.0)

    # Standardize matrices independently
    X_luad_std = StandardScaler().fit_transform(X_luad)
    X_paad_std = StandardScaler().fit_transform(X_paad)
    X_gse_std = StandardScaler().fit_transform(X_gse)

    return {
        "genes": genes,
        "luad": {"X": X_luad_std, "event": y_luad_e, "time": y_luad_t, "struct": y_luad_struct, "ids": luad_ids},
        "paad": {"X": X_paad_std, "event": y_paad_e, "time": y_paad_t},
        "gse": {"X": X_gse_std, "event": y_gse_e, "time": y_gse_t}
    }

def compute_graph_filter(L_matrix, alpha_smooth=0.1):
    if L_matrix is None:
        return None
    evals, evecs = np.linalg.eigh((L_matrix + L_matrix.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-alpha_smooth * evals)) @ evecs.T
    return W

def construct_hash_laplacian(L_string, genes, seed=42):
    rng = np.random.default_rng(seed)
    A = -L_string.copy()
    np.fill_diagonal(A, 0.0)
    A = np.clip(A, 0.0, None)
    # Apply hash perturbation to edges
    rows, cols = np.nonzero(A)
    for r, c in zip(rows, cols):
        if r < c:
            # Deterministic hash per gene pair with seed
            h_val = float(rng.uniform(0.0, 1.0))
            A[r, c] *= np.exp(- (h_val - 0.5) / 1.0)
            A[c, r] = A[r, c]
    # Recompute normalized Laplacian
    deg = np.sum(A, axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    mask = deg > 1e-12
    d_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    L_hash = np.eye(len(genes)) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return L_hash

def construct_shuffled_laplacian(L_string, seed=4337):
    rng = np.random.default_rng(seed)
    n = L_string.shape[0]
    perm = rng.permutation(n)
    L_shuf = L_string[np.ix_(perm, perm)].copy()
    return L_shuf

def train_and_select_alpha(X_train_g, y_train, random_state=4337):
    model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=3000)
    model.fit(X_train_g, y_train)
    counts = np.sum(model.coef_ != 0, axis=0)
    valid = np.where(counts > 0)[0]
    if len(valid) == 0:
        valid = [model.coef_.shape[1] - 1]

    # Select alpha using 5-fold CV strictly within the training set
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_scores = [[] for _ in range(model.coef_.shape[1])]
    for f_tr, f_va in kf.split(X_train_g):
        f_model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=model.alphas_, max_iter=3000)
        try:
            f_model.fit(X_train_g[f_tr], y_train[f_tr])
            f_risk = X_train_g[f_va] @ f_model.coef_
            for j in range(f_risk.shape[1]):
                if np.count_nonzero(f_model.coef_[:, j]) > 0:
                    fold_scores[j].append(
                        concordance_index_censored(y_train["event"][f_va], y_train["time"][f_va], f_risk[:, j])[0]
                    )
        except Exception:
            continue

    cv_means = np.full(model.coef_.shape[1], np.nan)
    for j, sc in enumerate(fold_scores):
        if sc:
            cv_means[j] = np.mean(sc)

    valid_cv = np.where(np.isfinite(cv_means))[0]
    if len(valid_cv) > 0:
        opt_idx = int(valid_cv[np.argmax(cv_means[valid_cv])])
    else:
        opt_idx = int(valid[-1])

    best_coef = model.coef_[:, opt_idx]
    return model, opt_idx, best_coef, float(model.alphas_[opt_idx])

def evaluate_predictions(e_true, t_true, risk, train_y_struct=None):
    c = float(concordance_index_censored(e_true, t_true, risk)[0])
    auc_1y, auc_3y, auc_5y = np.nan, np.nan, np.nan
    if train_y_struct is not None:
        try:
            test_struct = np.array(list(zip(e_true, t_true)), dtype=[("event", "?"), ("time", "<f8")])
            auc, _ = cumulative_dynamic_auc(train_y_struct, test_struct, risk, HORIZONS)
            auc_1y, auc_3y, auc_5y = float(auc[0]), float(auc[1]), float(auc[2])
        except Exception:
            pass
    return {"c_index": c, "auc_1y": auc_1y, "auc_3y": auc_3y, "auc_5y": auc_5y}

def bootstrap_evaluation(e_true, t_true, risk_dict, train_y_struct=None, n_boot=2000, seed=4337):
    rng = np.random.default_rng(seed)
    n = len(e_true)
    model_names = list(risk_dict.keys())
    boot_c = {m: [] for m in model_names}

    for _ in range(n_boot):
        b_idx = rng.choice(n, size=n, replace=True)
        b_e = e_true[b_idx]
        b_t = t_true[b_idx]
        if b_e.sum() == 0 or (~b_e).sum() == 0:
            continue
        for m in model_names:
            try:
                c = concordance_index_censored(b_e, b_t, risk_dict[m][b_idx])[0]
                boot_c[m].append(c)
            except Exception:
                pass

    summary = {}
    for m in model_names:
        arr = np.array(boot_c[m])
        summary[m] = {
            "mean": float(np.mean(arr)),
            "ci_95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        }

    # Compute paired differences against Ordinary STRING and No Graph
    paired_diffs = {}
    if "Ordinary STRING" in boot_c and "Chara (Retrained)" in boot_c:
        diff_str = np.array(boot_c["Chara (Retrained)"]) - np.array(boot_c["Ordinary STRING"])
        paired_diffs["Chara_vs_STRING"] = {
            "mean_delta": float(np.mean(diff_str)),
            "ci_95": [float(np.percentile(diff_str, 2.5)), float(np.percentile(diff_str, 97.5))],
            "p_value_two_tailed": float(2.0 * min(np.mean(diff_str <= 0), np.mean(diff_str >= 0)))
        }

    if "No Graph (Vanilla Coxnet)" in boot_c and "Chara (Retrained)" in boot_c:
        diff_ng = np.array(boot_c["Chara (Retrained)"]) - np.array(boot_c["No Graph (Vanilla Coxnet)"])
        paired_diffs["Chara_vs_NoGraph"] = {
            "mean_delta": float(np.mean(diff_ng)),
            "ci_95": [float(np.percentile(diff_ng, 2.5)), float(np.percentile(diff_ng, 97.5))],
            "p_value_two_tailed": float(2.0 * min(np.mean(diff_ng <= 0), np.mean(diff_ng >= 0)))
        }

    if "No Graph (Vanilla Coxnet)" in boot_c and "Ordinary STRING" in boot_c:
        diff_s_ng = np.array(boot_c["Ordinary STRING"]) - np.array(boot_c["No Graph (Vanilla Coxnet)"])
        paired_diffs["STRING_vs_NoGraph"] = {
            "mean_delta": float(np.mean(diff_s_ng)),
            "ci_95": [float(np.percentile(diff_s_ng, 2.5)), float(np.percentile(diff_s_ng, 97.5))],
            "p_value_two_tailed": float(2.0 * min(np.mean(diff_s_ng <= 0), np.mean(diff_s_ng >= 0)))
        }

    return summary, paired_diffs

def main():
    print("=" * 80)
    print(" COMPREHENSIVE CONTROLLED GRAPH BENCHMARK")
    print("=" * 80)
    t0 = time.time()
    data = load_data()
    genes = data["genes"]
    luad = data["luad"]
    paad = data["paad"]
    gse = data["gse"]

    # 75/25 Train/Test split on TCGA-LUAD
    train_idx, test_idx = train_test_split(
        np.arange(len(luad["X"])), test_size=0.25, random_state=4337, stratify=luad["event"]
    )
    X_tr, X_te = luad["X"][train_idx], luad["X"][test_idx]
    y_tr, y_te = luad["struct"][train_idx], luad["struct"][test_idx]

    # Load Laplacians
    print("[INFO] Loading Laplacians and computing filter matrices...")
    L_string = pd.read_csv(ROOT / "Laplacian_STRING_4337.csv", index_col=0).loc[genes, genes].to_numpy(float)
    L_chara = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0).loc[genes, genes].to_numpy(float)
    L_hash = construct_hash_laplacian(L_string, genes, seed=42)
    L_shuf = construct_shuffled_laplacian(L_string, seed=4337)

    filters = {
        "No Graph (Vanilla Coxnet)": None,
        "Ordinary STRING": compute_graph_filter(L_string),
        "Chara (Retrained)": compute_graph_filter(L_chara),
        "Hash-Weighted STRING (Seed 42)": compute_graph_filter(L_hash),
        "Shuffled Graph Control": compute_graph_filter(L_shuf)
    }

    # Load Frozen Historical Chara Model
    frozen_bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    frozen_coef = frozen_bundle["model"].coef_[:, frozen_bundle["alpha_index"]]

    models = {}
    predictions_luad_test = {}
    predictions_luad_test_w = {}
    predictions_paad = {}
    predictions_paad_w = {}
    predictions_gse = {}
    predictions_gse_w = {}
    model_metadata = {}

    # Train each model condition
    for name, W in filters.items():
        print(f"\n[INFO] Fitting and cross-validating condition: {name}...")
        if W is not None:
            X_tr_g = X_tr @ W
            X_te_g = X_te @ W
            X_paad_g = paad["X"] @ W
            X_gse_g = gse["X"] @ W
        else:
            X_tr_g = X_tr
            X_te_g = X_te
            X_paad_g = paad["X"]
            X_gse_g = gse["X"]

        model, opt_idx, best_coef, alpha_val = train_and_select_alpha(X_tr_g, y_tr, random_state=4337)
        models[name] = {"model": model, "coef": best_coef, "alpha_index": opt_idx, "alpha": alpha_val}
        n_nz = int(np.count_nonzero(best_coef))
        model_metadata[name] = {"alpha_index": opt_idx, "alpha": alpha_val, "nonzero_genes": n_nz}
        print(f"       -> Optimal Alpha: {alpha_val:.6f} (index {opt_idx}), Non-zero genes: {n_nz}")

        # Predictions: With graph smoothing at test time
        predictions_luad_test[name] = X_te_g @ best_coef
        predictions_paad[name] = X_paad_g @ best_coef
        predictions_gse[name] = X_gse_g @ best_coef

        # Predictions: Raw expression at test time (historical repository behavior)
        predictions_luad_test_w[name] = X_te @ best_coef
        predictions_paad_w[name] = paad["X"] @ best_coef
        predictions_gse_w[name] = gse["X"] @ best_coef

    # Add Frozen Historical Chara
    frozen_name = "Frozen Historical Chara (chara_model_4337.pkl)"
    W_chara = filters["Chara (Retrained)"]
    predictions_luad_test[frozen_name] = (X_te @ W_chara) @ frozen_coef
    predictions_paad[frozen_name] = (paad["X"] @ W_chara) @ frozen_coef
    predictions_gse[frozen_name] = (gse["X"] @ W_chara) @ frozen_coef
    predictions_luad_test_w[frozen_name] = X_te @ frozen_coef
    predictions_paad_w[frozen_name] = paad["X"] @ frozen_coef
    predictions_gse_w[frozen_name] = gse["X"] @ frozen_coef
    model_metadata[frozen_name] = {
        "alpha_index": int(frozen_bundle["alpha_index"]),
        "alpha": float(frozen_bundle["model"].alphas_[frozen_bundle["alpha_index"]]),
        "nonzero_genes": int(np.count_nonzero(frozen_coef))
    }

    # Evaluate point metrics
    def eval_suite(risk_map, e_true, t_true, tr_struct):
        res = {}
        for m, r in risk_map.items():
            res[m] = evaluate_predictions(e_true, t_true, r, tr_struct)
        return res

    point_luad = eval_suite(predictions_luad_test, y_te["event"], y_te["time"], y_tr)
    point_paad = eval_suite(predictions_paad, paad["event"], paad["time"], y_tr)
    point_gse = eval_suite(predictions_gse, gse["event"], gse["time"], y_tr)

    point_luad_raw = eval_suite(predictions_luad_test_w, y_te["event"], y_te["time"], y_tr)
    point_paad_raw = eval_suite(predictions_paad_w, paad["event"], paad["time"], y_tr)
    point_gse_raw = eval_suite(predictions_gse_w, gse["event"], gse["time"], y_tr)

    # Run 2,000 bootstrap draws for C-indices and paired differences
    print("\n[INFO] Running 2,000 bootstrap resamples on TCGA-LUAD Test...")
    boot_luad, diff_luad = bootstrap_evaluation(y_te["event"], y_te["time"], predictions_luad_test, n_boot=2000)

    print("[INFO] Running 2,000 bootstrap resamples on TCGA-PAAD...")
    boot_paad, diff_paad = bootstrap_evaluation(paad["event"], paad["time"], predictions_paad, n_boot=2000)

    print("[INFO] Running 2,000 bootstrap resamples on GSE31210...")
    boot_gse, diff_gse = bootstrap_evaluation(gse["event"], gse["time"], predictions_gse, n_boot=2000)

    # Consolidate all evidence
    benchmark_evidence = {
        "execution_time_seconds": float(time.time() - t0),
        "cohort_sizes": {
            "luad_train": len(train_idx),
            "luad_train_events": int(y_tr["event"].sum()),
            "luad_test": len(test_idx),
            "luad_test_events": int(y_te["event"].sum()),
            "paad_samples": len(paad["event"]),
            "paad_events": int(paad["event"].sum()),
            "gse31210_samples": len(gse["event"]),
            "gse31210_events": int(gse["event"].sum())
        },
        "model_metadata": model_metadata,
        "point_metrics_smoothed_test": {
            "TCGA_LUAD_Test": point_luad,
            "TCGA_PAAD_OOD": point_paad,
            "GSE31210_External": point_gse
        },
        "point_metrics_raw_expression_test": {
            "TCGA_LUAD_Test": point_luad_raw,
            "TCGA_PAAD_OOD": point_paad_raw,
            "GSE31210_External": point_gse_raw
        },
        "bootstrap_ci_95": {
            "TCGA_LUAD_Test": boot_luad,
            "TCGA_PAAD_OOD": boot_paad,
            "GSE31210_External": boot_gse
        },
        "paired_differences": {
            "TCGA_LUAD_Test": diff_luad,
            "TCGA_PAAD_OOD": diff_paad,
            "GSE31210_External": diff_gse
        }
    }

    out_file = EVIDENCE_DIR / "fair_graph_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_evidence, f, indent=2)

    print(f"\n[OK] Fair Graph Benchmark complete. Saved to: {out_file}")
    print("\n" + "=" * 80)
    print(f"{'Condition':<35} | {'LUAD C':<8} | {'PAAD C':<8} | {'GSE C':<8}")
    print("-" * 80)
    for m in point_luad:
        print(f"{m:<35} | {point_luad[m]['c_index']:<8.4f} | {point_paad[m]['c_index']:<8.4f} | {point_gse[m]['c_index']:<8.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
