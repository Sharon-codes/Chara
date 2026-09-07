#!/usr/bin/env python3
"""
tools/chara_final_check/03_run_bounded_benchmark.py
Executes the final, bounded, leakage-free three-model evaluation:
  Model 1: Clinical-Only (Age, Sex, Pathological Stage)
  Model 2: Clinical + Ordinary Coxnet (No Graph)
  Model 3: Clinical + Chara (Spectral Heat-Kernel Filter)

Protocol:
- TCGA-LUAD (Development, N=502):
  5-fold stratified outer CV, with inner 3-fold cross-fitting within outer training
  partitions to train combination models without in-sample outcome leakage.
  Pooled within-fold concordance aggregation across folds.
- GSE31210 (External LUAD, N=226):
  Zero-shot evaluation with 2,000 paired bootstrap resamples.
- TCGA-PAAD (Different-Cancer Stress Test, N=177):
  Zero-shot evaluation with 2,000 paired bootstrap resamples.
- Pre-specified comparisons:
  Model 2 - Model 1 (Value of gene expression over clinical)
  Model 3 - Model 1 (Value of Chara over clinical)
  Model 3 - Model 2 (Value of Chara graph over ordinary Coxnet)
- Pragmatic screening margin: +0.02 C-index gain.
"""

import sys
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter
import GEOparse

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_final_check" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = np.array([365.0, 1095.0, 1825.0])

def parse_stage_str(s):
    s = str(s).strip().upper()
    if 'IV' in s: return 4.0
    if 'III' in s: return 3.0
    if 'II' in s: return 2.0
    if 'I' in s: return 1.0
    return np.nan

def load_tcga_luad(genes):
    print("[INFO] Loading TCGA-LUAD development cohort...")
    exp = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    surv = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    clin = pd.read_csv(ROOT / "data" / "TCGA-LUAD_clinical.csv", index_col=0)

    common = sorted(list(exp.index.intersection(surv.index).intersection(clin.index)))
    print(f"       -> TCGA-LUAD matching patients: {len(common)}")

    X = exp.loc[common, genes].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    event = surv.loc[common, "Event"].astype(bool).to_numpy()
    time_days = surv.loc[common, "Time"].to_numpy(float) * 365.0

    age = clin.loc[common, "AGE"].to_numpy(float)
    sex = clin.loc[common, "SEX"].to_numpy(float)
    stage = clin.loc[common, "STAGE"].to_numpy(float)

    return {
        "cohort": "TCGA-LUAD",
        "ids": common,
        "X": X,
        "event": event,
        "time": time_days,
        "age": age,
        "sex": sex,
        "stage": stage,
        "n_samples": len(common),
        "n_events": int(event.sum())
    }

def load_gse31210(genes):
    print("[INFO] Loading GSE31210 external cohort...")
    import importlib.util
    spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
    v09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v09)

    gse = GEOparse.get_GEO(geo="GSE31210", destdir=str(ROOT / "geo_cache"), include_data=True)
    expr_df = v09.load_gse31210_expression(gse)

    # Extract metadata for all GSMs
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        meta = gsm.metadata.get("characteristics_ch1", [])
        d = {"gsm": gsm_id}
        for item in meta:
            if ":" in item:
                k, v = item.split(":", 1)
                d[k.strip().lower()] = v.strip()
        rows.append(d)
    meta_df = pd.DataFrame(rows).set_index("gsm")

    # Common samples between expression and metadata
    common = sorted(list(expr_df.index.intersection(meta_df.index)))
    
    # Filter to patients with valid death/censor status
    valid_pats = []
    for g in common:
        d_val = str(meta_df.loc[g, "death"]).lower()
        if d_val in ["dead", "alive"]:
            valid_pats.append(g)

    expr_sub = expr_df.loc[valid_pats].reindex(columns=genes).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta_sub = meta_df.loc[valid_pats]

    X = expr_sub.to_numpy(float)
    event = (meta_sub["death"].str.lower() == "dead").to_numpy(bool)
    
    # Time in days
    time_days = pd.to_numeric(meta_sub["days before death/censor"], errors="coerce").to_numpy(float)
    # If missing days, use months * (365.25/12)
    if np.isnan(time_days).any():
        months = pd.to_numeric(meta_sub["months before relapse/censor"], errors="coerce").to_numpy(float)
        time_days = np.where(np.isnan(time_days), months * (365.25 / 12.0), time_days)

    age = pd.to_numeric(meta_sub["age (years)"], errors="coerce").to_numpy(float)
    sex = (meta_sub["gender"].str.lower() == "male").astype(float).to_numpy()
    stage = meta_sub["pathological stage"].apply(parse_stage_str).to_numpy(float)

    print(f"       -> GSE31210 valid patients: {len(valid_pats)} (Events: {int(event.sum())})")
    return {
        "cohort": "GSE31210",
        "ids": valid_pats,
        "X": X,
        "event": event,
        "time": time_days,
        "age": age,
        "sex": sex,
        "stage": stage,
        "n_samples": len(valid_pats),
        "n_events": int(event.sum())
    }

def load_tcga_paad(genes):
    print("[INFO] Loading TCGA-PAAD different-cancer stress test...")
    exp = pd.read_csv(ROOT / "TCGA-PAAD_expression.csv", index_col=0)
    surv = pd.read_csv(ROOT / "TCGA-PAAD_survival.csv", index_col=0)
    
    # Fetch clinical data for PAAD from cBioPortal or local cache
    paad_clin_path = ROOT / "data" / "TCGA-PAAD_clinical.csv"
    if paad_clin_path.exists():
        clin_df = pd.read_csv(paad_clin_path, index_col=0)
    else:
        import requests, io
        url = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/paad_tcga/data_clinical_patient.txt"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        lines = [l for l in r.text.split('\n') if not l.startswith('#')]
        raw = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', low_memory=False)
        pat_ids = raw['PATIENT_ID'].astype(str).str.strip().str.slice(0, 12)
        clin_df = pd.DataFrame(index=pat_ids)
        clin_df['AGE'] = pd.to_numeric(raw['AGE'], errors='coerce').values
        clin_df['SEX'] = raw['SEX'].str.lower().map({'male': 1.0, 'm': 1.0, 'female': 0.0, 'f': 0.0}).values
        clin_df['STAGE'] = raw['AJCC_PATHOLOGIC_TUMOR_STAGE'].apply(parse_stage_str).values
        clin_df = clin_df[~clin_df.index.duplicated(keep='first')]
        clin_df.to_csv(paad_clin_path)

    common = sorted(list(exp.index.intersection(surv.index).intersection(clin_df.index)))
    print(f"       -> TCGA-PAAD matching patients: {len(common)}")

    X = exp.loc[common, genes].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    event = surv.loc[common, "Event"].astype(bool).to_numpy()
    time_days = surv.loc[common, "Time"].to_numpy(float) * 365.0

    age = clin_df.loc[common, "AGE"].to_numpy(float)
    sex = clin_df.loc[common, "SEX"].to_numpy(float)
    stage = clin_df.loc[common, "STAGE"].to_numpy(float)

    return {
        "cohort": "TCGA-PAAD",
        "ids": common,
        "X": X,
        "event": event,
        "time": time_days,
        "age": age,
        "sex": sex,
        "stage": stage,
        "n_samples": len(common),
        "n_events": int(event.sum())
    }

def compute_chara_filter(genes):
    print("[INFO] Precomputing Chara spectral heat-kernel filter (W)...")
    L_df = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0)
    L = L_df.loc[genes, genes].to_numpy(float)
    evals, evecs = np.linalg.eigh((L + L.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-0.1 * evals)) @ evecs.T
    print("[OK] Computed W matrix shape:", W.shape)
    return W

def fit_gene_coxnet(X_tr, y_tr, random_state=4337):
    # Fit Coxnet and select alpha via internal CV on training partition
    model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, max_iter=2000)
    model.fit(X_tr, y_tr)
    
    # Internal 3-fold CV for alpha selection
    counts = np.sum(model.coef_ != 0, axis=0)
    valid = np.where(counts > 0)[0]
    if len(valid) == 0:
        valid = [model.coef_.shape[1] - 1]

    kf = KFold(n_splits=3, shuffle=True, random_state=random_state)
    fold_scores = [[] for _ in range(model.coef_.shape[1])]
    for f_tr, f_va in kf.split(X_tr):
        f_m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=model.alphas_, max_iter=2000)
        try:
            f_m.fit(X_tr[f_tr], y_tr[f_tr])
            f_risk = X_tr[f_va] @ f_m.coef_
            for j in range(f_risk.shape[1]):
                if np.count_nonzero(f_m.coef_[:, j]) > 0:
                    fold_scores[j].append(
                        concordance_index_censored(y_tr["event"][f_va], y_tr["time"][f_va], f_risk[:, j])[0]
                    )
        except Exception:
            continue

    cv_means = np.full(model.coef_.shape[1], np.nan)
    for j, sc in enumerate(fold_scores):
        if sc:
            cv_means[j] = np.mean(sc)

    valid_cv = np.where(np.isfinite(cv_means))[0]
    opt_idx = int(valid_cv[np.argmax(cv_means[valid_cv])]) if len(valid_cv) > 0 else int(valid[-1])
    best_coef = model.coef_[:, opt_idx]
    return model, best_coef, opt_idx

def compute_pairs(e, t, risk):
    # Computes concordant, tied, and total permissible pairs for exact Harrell's C aggregation
    n = len(e)
    concordant = 0.0
    tied = 0.0
    permissible = 0.0
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if t[i] < t[j]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    tied += 1.0
    return concordant, tied, permissible

def run_nested_cv(luad_data, W_chara):
    print("\n" + "=" * 80)
    print(" EXECUTING NESTED DOUBLE CROSS-VALIDATION ON TCGA-LUAD")
    print("=" * 80)

    X_all = luad_data["X"]
    e_all = luad_data["event"]
    t_all = luad_data["time"]
    age_all = luad_data["age"]
    sex_all = luad_data["sex"]
    stage_all = luad_data["stage"]
    n = len(e_all)

    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=4337)
    
    fold_results = []
    all_test_preds = {
        "m1_clinical": np.zeros(n),
        "m2_clin_coxnet": np.zeros(n),
        "m3_clin_chara": np.zeros(n),
        "diag_coxnet_only": np.zeros(n),
        "diag_chara_only": np.zeros(n)
    }

    pairs_m1 = {"conc": 0.0, "tied": 0.0, "perm": 0.0}
    pairs_m2 = {"conc": 0.0, "tied": 0.0, "perm": 0.0}
    pairs_m3 = {"conc": 0.0, "tied": 0.0, "perm": 0.0}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(X_all, e_all)):
        print(f"\n---> Outer Fold {fold_idx + 1} / 5 (Train: {len(tr_idx)}, Test: {len(te_idx)}) <---")

        # 1. Isolate partitions
        X_tr, X_te = X_all[tr_idx], X_all[te_idx]
        e_tr, e_te = e_all[tr_idx], e_all[te_idx]
        t_tr, t_te = t_all[tr_idx], t_all[te_idx]
        y_tr_struct = np.array(list(zip(e_tr, t_tr)), dtype=[("event", "?"), ("time", "<f8")])

        age_tr, age_te = age_all[tr_idx], age_all[te_idx]
        sex_tr, sex_te = sex_all[tr_idx], sex_all[te_idx]
        stage_tr, stage_te = stage_all[tr_idx], stage_all[te_idx]

        # Standardize features using Outer Train ONLY
        scaler_exp = StandardScaler().fit(X_tr)
        X_tr_std = scaler_exp.transform(X_tr)
        X_te_std = scaler_exp.transform(X_te)

        # Impute clinical variables using Outer Train stats
        age_med = float(np.nanmedian(age_tr))
        age_std_val = float(np.nanstd(age_tr)) if np.nanstd(age_tr) > 0 else 1.0
        age_tr_clean = np.nan_to_num(age_tr, nan=age_med)
        age_te_clean = np.nan_to_num(age_te, nan=age_med)
        age_tr_z = (age_tr_clean - age_med) / age_std_val
        age_te_z = (age_te_clean - age_med) / age_std_val

        sex_mode = float(pd.Series(sex_tr).mode()[0]) if len(pd.Series(sex_tr).dropna()) > 0 else 0.0
        sex_tr_clean = np.nan_to_num(sex_tr, nan=sex_mode)
        sex_te_clean = np.nan_to_num(sex_te, nan=sex_mode)

        stage_med = float(np.nanmedian(stage_tr))
        stage_tr_clean = np.nan_to_num(stage_tr, nan=stage_med)
        stage_te_clean = np.nan_to_num(stage_te, nan=stage_med)

        # 2. Fit Model 1: Clinical-Only
        df_tr_m1 = pd.DataFrame({
            "time": t_tr, "event": e_tr.astype(int),
            "age": age_tr_z, "sex": sex_tr_clean, "stage": stage_tr_clean
        })
        df_te_m1 = pd.DataFrame({
            "time": t_te, "event": e_te.astype(int),
            "age": age_te_z, "sex": sex_te_clean, "stage": stage_te_clean
        })
        cph_m1 = CoxPHFitter(penalizer=0.01).fit(df_tr_m1, "time", "event")
        pred_m1_te = cph_m1.predict_partial_hazard(df_te_m1).to_numpy()

        # 3. Inner 3-Fold Cross-Fitting to produce out-of-fold gene scores for Outer Train
        skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_idx * 100 + 42)
        oof_coxnet_tr = np.zeros(len(tr_idx))
        oof_chara_tr = np.zeros(len(tr_idx))

        for in_tr_idx, in_val_idx in skf_inner.split(X_tr_std, e_tr):
            # Scale inner train
            in_scaler = StandardScaler().fit(X_tr[in_tr_idx])
            X_in_tr = in_scaler.transform(X_tr[in_tr_idx])
            X_in_val = in_scaler.transform(X_tr[in_val_idx])
            y_in_tr = np.array(list(zip(e_tr[in_tr_idx], t_tr[in_tr_idx])), dtype=[("event", "?"), ("time", "<f8")])

            # Inner Ordinary Coxnet
            _, coef_cox, _ = fit_gene_coxnet(X_in_tr, y_in_tr, random_state=fold_idx * 10 + 1)
            oof_coxnet_tr[in_val_idx] = X_in_val @ coef_cox

            # Inner Chara
            _, coef_chara, _ = fit_gene_coxnet(X_in_tr @ W_chara, y_in_tr, random_state=fold_idx * 10 + 2)
            oof_chara_tr[in_val_idx] = (X_in_val @ W_chara) @ coef_chara

        # Standardize cross-fitted gene scores using training stats
        mu_cox_tr, std_cox_tr = float(np.mean(oof_coxnet_tr)), float(np.std(oof_coxnet_tr)) + 1e-8
        mu_cha_tr, std_cha_tr = float(np.mean(oof_chara_tr)), float(np.std(oof_chara_tr)) + 1e-8

        z_coxnet_tr = (oof_coxnet_tr - mu_cox_tr) / std_cox_tr
        z_chara_tr = (oof_chara_tr - mu_cha_tr) / std_cha_tr

        # 4. Fit Combination Models 2 & 3 on Outer Train
        df_tr_m2 = df_tr_m1.copy()
        df_tr_m2["gene_risk"] = z_coxnet_tr
        cph_m2 = CoxPHFitter(penalizer=0.01).fit(df_tr_m2, "time", "event")

        df_tr_m3 = df_tr_m1.copy()
        df_tr_m3["gene_risk"] = z_chara_tr
        cph_m3 = CoxPHFitter(penalizer=0.01).fit(df_tr_m3, "time", "event")

        # 5. Refit Base Gene Models on Full Outer Train
        _, base_coef_cox, opt_cox = fit_gene_coxnet(X_tr_std, y_tr_struct, random_state=4337)
        _, base_coef_cha, opt_cha = fit_gene_coxnet(X_tr_std @ W_chara, y_tr_struct, random_state=4337)

        # Generate Test Gene Risk Scores
        raw_gene_te_cox = X_te_std @ base_coef_cox
        raw_gene_te_cha = (X_te_std @ W_chara) @ base_coef_cha

        z_gene_te_cox = (raw_gene_te_cox - mu_cox_tr) / std_cox_tr
        z_gene_te_cha = (raw_gene_te_cha - mu_cha_tr) / std_cha_tr

        # Apply Combination Models to Outer Test
        df_te_m2 = df_te_m1.copy()
        df_te_m2["gene_risk"] = z_gene_te_cox
        pred_m2_te = cph_m2.predict_partial_hazard(df_te_m2).to_numpy()

        df_te_m3 = df_te_m1.copy()
        df_te_m3["gene_risk"] = z_gene_te_cha
        pred_m3_te = cph_m3.predict_partial_hazard(df_te_m3).to_numpy()

        # Save predictions
        all_test_preds["m1_clinical"][te_idx] = pred_m1_te
        all_test_preds["m2_clin_coxnet"][te_idx] = pred_m2_te
        all_test_preds["m3_clin_chara"][te_idx] = pred_m3_te
        all_test_preds["diag_coxnet_only"][te_idx] = raw_gene_te_cox
        all_test_preds["diag_chara_only"][te_idx] = raw_gene_te_cha

        # Compute Fold C-indices
        c1 = float(concordance_index_censored(e_te, t_te, pred_m1_te)[0])
        c2 = float(concordance_index_censored(e_te, t_te, pred_m2_te)[0])
        c3 = float(concordance_index_censored(e_te, t_te, pred_m3_te)[0])

        c_diag_cox = float(concordance_index_censored(e_te, t_te, raw_gene_te_cox)[0])
        c_diag_cha = float(concordance_index_censored(e_te, t_te, raw_gene_te_cha)[0])

        fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "events_test": int(e_te.sum()),
            "m1_clinical_c": c1,
            "m2_clin_coxnet_c": c2,
            "m3_clin_chara_c": c3,
            "delta_m2_minus_m1": c2 - c1,
            "delta_m3_minus_m1": c3 - c1,
            "delta_m3_minus_m2": c3 - c2,
            "diag_coxnet_only_c": c_diag_cox,
            "diag_chara_only_c": c_diag_cha
        })
        print(f"       Fold {fold_idx+1} C-indices | M1 (Clin): {c1:.4f} | M2 (Clin+Coxnet): {c2:.4f} | M3 (Clin+Chara): {c3:.4f}")
        print(f"       Deltas: M2-M1 = {c2-c1:+.4f} | M3-M1 = {c3-c1:+.4f} | M3-M2 = {c3-c2:+.4f}")

        # Accumulate within-fold pairs
        c_p1, t_p1, perm1 = compute_pairs(e_te, t_te, pred_m1_te)
        c_p2, t_p2, perm2 = compute_pairs(e_te, t_te, pred_m2_te)
        c_p3, t_p3, perm3 = compute_pairs(e_te, t_te, pred_m3_te)

        pairs_m1["conc"] += c_p1; pairs_m1["tied"] += t_p1; pairs_m1["perm"] += perm1
        pairs_m2["conc"] += c_p2; pairs_m2["tied"] += t_p2; pairs_m2["perm"] += perm2
        pairs_m3["conc"] += c_p3; pairs_m3["tied"] += t_p3; pairs_m3["perm"] += perm3

    # Calculate pooled within-fold aggregate concordance
    agg_c1 = (pairs_m1["conc"] + 0.5 * pairs_m1["tied"]) / pairs_m1["perm"]
    agg_c2 = (pairs_m2["conc"] + 0.5 * pairs_m2["tied"]) / pairs_m2["perm"]
    agg_c3 = (pairs_m3["conc"] + 0.5 * pairs_m3["tied"]) / pairs_m3["perm"]

    aggregate_summary = {
        "m1_clinical_pooled_c": float(agg_c1),
        "m2_clin_coxnet_pooled_c": float(agg_c2),
        "m3_clin_chara_pooled_c": float(agg_c3),
        "delta_m2_minus_m1": float(agg_c2 - agg_c1),
        "delta_m3_minus_m1": float(agg_c3 - agg_c1),
        "delta_m3_minus_m2": float(agg_c3 - agg_c2),
        "total_within_fold_pairs": int(pairs_m1["perm"])
    }

    print("\n" + "=" * 80)
    print(" POOLED WITHIN-FOLD AGGREGATE RESULTS (TCGA-LUAD 5-FOLD NESTED CV)")
    print("=" * 80)
    print(f"  Model 1: Clinical-Only             : {agg_c1:.4f}")
    print(f"  Model 2: Clinical + Ordinary Coxnet : {agg_c2:.4f} (Delta vs M1: {agg_c2 - agg_c1:+.4f})")
    print(f"  Model 3: Clinical + Chara           : {agg_c3:.4f} (Delta vs M1: {agg_c3 - agg_c1:+.4f}, Delta vs M2: {agg_c3 - agg_c2:+.4f})")
    print("=" * 80)

    return fold_results, aggregate_summary, all_test_preds

def train_full_development_pipeline(luad_data, W_chara):
    print("\n[INFO] Training final three-model pipelines on full TCGA-LUAD for external evaluation...")
    X = luad_data["X"]
    e = luad_data["event"]
    t = luad_data["time"]
    age = luad_data["age"]
    sex = luad_data["sex"]
    stage = luad_data["stage"]
    y_struct = np.array(list(zip(e, t)), dtype=[("event", "?"), ("time", "<f8")])

    # 1. Standardize full dataset
    scaler_exp = StandardScaler().fit(X)
    X_std = scaler_exp.transform(X)

    age_med = float(np.nanmedian(age))
    age_std_val = float(np.nanstd(age)) if np.nanstd(age) > 0 else 1.0
    age_z = (np.nan_to_num(age, nan=age_med) - age_med) / age_std_val
    sex_clean = np.nan_to_num(sex, nan=float(pd.Series(sex).mode()[0]))
    stage_clean = np.nan_to_num(stage, nan=float(np.nanmedian(stage)))

    # Model 1
    df_m1 = pd.DataFrame({"time": t, "event": e.astype(int), "age": age_z, "sex": sex_clean, "stage": stage_clean})
    cph_m1 = CoxPHFitter(penalizer=0.01).fit(df_m1, "time", "event")

    # 2. 5-fold cross-fitting to produce out-of-fold training gene scores
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4337)
    oof_cox = np.zeros(len(e))
    oof_cha = np.zeros(len(e))

    for in_tr, in_val in skf.split(X_std, e):
        # Scale
        s = StandardScaler().fit(X[in_tr])
        x_tr = s.transform(X[in_tr])
        x_va = s.transform(X[in_val])
        y_tr_in = np.array(list(zip(e[in_tr], t[in_tr])), dtype=[("event", "?"), ("time", "<f8")])

        _, c_cox, _ = fit_gene_coxnet(x_tr, y_tr_in, random_state=1)
        oof_cox[in_val] = x_va @ c_cox

        _, c_cha, _ = fit_gene_coxnet(x_tr @ W_chara, y_tr_in, random_state=2)
        oof_cha[in_val] = (x_va @ W_chara) @ c_cha

    mu_cox, std_cox = float(np.mean(oof_cox)), float(np.std(oof_cox)) + 1e-8
    mu_cha, std_cha = float(np.mean(oof_cha)), float(np.std(oof_cha)) + 1e-8

    # Combination Models
    df_m2 = df_m1.copy(); df_m2["gene_risk"] = (oof_cox - mu_cox) / std_cox
    cph_m2 = CoxPHFitter(penalizer=0.01).fit(df_m2, "time", "event")

    df_m3 = df_m1.copy(); df_m3["gene_risk"] = (oof_cha - mu_cha) / std_cha
    cph_m3 = CoxPHFitter(penalizer=0.01).fit(df_m3, "time", "event")

    # Refit base gene models on full cohort
    _, base_coef_cox, _ = fit_gene_coxnet(X_std, y_struct, random_state=4337)
    _, base_coef_cha, _ = fit_gene_coxnet(X_std @ W_chara, y_struct, random_state=4337)

    return {
        "scaler_exp": scaler_exp,
        "age_med": age_med,
        "age_std": age_std_val,
        "sex_mode": float(pd.Series(sex).mode()[0]),
        "stage_med": float(np.nanmedian(stage)),
        "cph_m1": cph_m1,
        "cph_m2": cph_m2,
        "cph_m3": cph_m3,
        "base_coef_cox": base_coef_cox,
        "base_coef_cha": base_coef_cha,
        "mu_cox": mu_cox,
        "std_cox": std_cox,
        "mu_cha": mu_cha,
        "std_cha": std_cha
    }

def evaluate_external_cohort(ext_data, pipeline, W_chara, n_boot=2000, seed=4337):
    name = ext_data["cohort"]
    print(f"\n[INFO] Evaluating external cohort: {name} (N={ext_data['n_samples']}, Events={ext_data['n_events']})...")

    X = ext_data["X"]
    e = ext_data["event"]
    t = ext_data["time"]
    age = ext_data["age"]
    sex = ext_data["sex"]
    stage = ext_data["stage"]

    # Transform using source-fitted parameters
    X_std = pipeline["scaler_exp"].transform(X)

    age_z = (np.nan_to_num(age, nan=pipeline["age_med"]) - pipeline["age_med"]) / pipeline["age_std"]
    sex_clean = np.nan_to_num(sex, nan=pipeline["sex_mode"])
    stage_clean = np.nan_to_num(stage, nan=pipeline["stage_med"])

    df_clin = pd.DataFrame({"time": t, "event": e.astype(int), "age": age_z, "sex": sex_clean, "stage": stage_clean})

    # Model 1
    pred_m1 = pipeline["cph_m1"].predict_partial_hazard(df_clin).to_numpy()

    # Model 2
    raw_gene_cox = X_std @ pipeline["base_coef_cox"]
    z_gene_cox = (raw_gene_cox - pipeline["mu_cox"]) / pipeline["std_cox"]
    df_m2 = df_clin.copy(); df_m2["gene_risk"] = z_gene_cox
    pred_m2 = pipeline["cph_m2"].predict_partial_hazard(df_m2).to_numpy()

    # Model 3
    raw_gene_cha = (X_std @ W_chara) @ pipeline["base_coef_cha"]
    z_gene_cha = (raw_gene_cha - pipeline["mu_cha"]) / pipeline["std_cha"]
    df_m3 = df_clin.copy(); df_m3["gene_risk"] = z_gene_cha
    pred_m3 = pipeline["cph_m3"].predict_partial_hazard(df_m3).to_numpy()

    # Point C-indices
    c1 = float(concordance_index_censored(e, t, pred_m1)[0])
    c2 = float(concordance_index_censored(e, t, pred_m2)[0])
    c3 = float(concordance_index_censored(e, t, pred_m3)[0])

    # Bootstrap 2,000 paired draws
    rng = np.random.default_rng(seed)
    n = len(e)
    b_c1, b_c2, b_c3 = [], [], []
    d_21, d_31, d_32 = [], [], []
    valid_draws = 0

    for _ in range(n_boot):
        b_idx = rng.choice(n, size=n, replace=True)
        be, bt = e[b_idx], t[b_idx]
        if be.sum() == 0 or (~be).sum() == 0:
            continue
        try:
            bc1 = concordance_index_censored(be, bt, pred_m1[b_idx])[0]
            bc2 = concordance_index_censored(be, bt, pred_m2[b_idx])[0]
            bc3 = concordance_index_censored(be, bt, pred_m3[b_idx])[0]
            b_c1.append(bc1); b_c2.append(bc2); b_c3.append(bc3)
            d_21.append(bc2 - bc1)
            d_31.append(bc3 - bc1)
            d_32.append(bc3 - bc2)
            valid_draws += 1
        except Exception:
            continue

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    def pval(diff_arr):
        # Two-sided empirical bootstrap p-value
        return float(2.0 * min(np.mean(np.array(diff_arr) <= 0), np.mean(np.array(diff_arr) >= 0)))

    results = {
        "cohort": name,
        "n_samples": n,
        "n_events": int(e.sum()),
        "valid_bootstrap_draws": valid_draws,
        "point_estimates": {
            "m1_clinical": c1,
            "m2_clin_coxnet": c2,
            "m3_clin_chara": c3,
            "delta_m2_minus_m1": c2 - c1,
            "delta_m3_minus_m1": c3 - c1,
            "delta_m3_minus_m2": c3 - c2
        },
        "bootstrap_summary": {
            "m1_clinical": {"mean": float(np.mean(b_c1)), "ci_95": ci(b_c1)},
            "m2_clin_coxnet": {"mean": float(np.mean(b_c2)), "ci_95": ci(b_c2)},
            "m3_clin_chara": {"mean": float(np.mean(b_c3)), "ci_95": ci(b_c3)}
        },
        "paired_differences": {
            "m2_minus_m1": {"mean_delta": float(np.mean(d_21)), "ci_95": ci(d_21), "p_value_two_sided": pval(d_21)},
            "m3_minus_m1": {"mean_delta": float(np.mean(d_31)), "ci_95": ci(d_31), "p_value_two_sided": pval(d_31)},
            "m3_minus_m2": {"mean_delta": float(np.mean(d_32)), "ci_95": ci(d_32), "p_value_two_sided": pval(d_32)}
        }
    }

    print(f"       C-indices: M1 = {c1:.4f} | M2 = {c2:.4f} | M3 = {c3:.4f}")
    print(f"       Paired Deltas: M2-M1 = {c2-c1:+.4f} (p={results['paired_differences']['m2_minus_m1']['p_value_two_sided']:.4f}) | M3-M1 = {c3-c1:+.4f} (p={results['paired_differences']['m3_minus_m1']['p_value_two_sided']:.4f}) | M3-M2 = {c3-c2:+.4f} (p={results['paired_differences']['m3_minus_m2']['p_value_two_sided']:.4f})")
    return results

def main():
    print("=" * 80)
    print(" CHARA FINAL BOUNDED CPU-ONLY BENCHMARK")
    print("=" * 80)
    t0 = time.time()

    genes = pd.read_csv(ROOT / "intersecting_genes_4337.txt", header=None)[0].astype(str).tolist()
    luad_data = load_tcga_luad(genes)
    gse_data = load_gse31210(genes)
    paad_data = load_tcga_paad(genes)
    W_chara = compute_chara_filter(genes)

    # 1. Nested Cross-Validation on TCGA-LUAD
    fold_res, agg_res, test_preds = run_nested_cv(luad_data, W_chara)

    # 2. Train final pipeline on full development population
    pipeline = train_full_development_pipeline(luad_data, W_chara)

    # 3. External evaluation
    gse_res = evaluate_external_cohort(gse_data, pipeline, W_chara, n_boot=2000)
    paad_res = evaluate_external_cohort(paad_data, pipeline, W_chara, n_boot=2000)

    # Consolidate benchmark evidence
    final_evidence = {
        "execution_time_seconds": float(time.time() - t0),
        "screening_margin_threshold": 0.02,
        "tcga_luad_development": {
            "individual_folds": fold_res,
            "pooled_within_fold_aggregate": agg_res
        },
        "gse31210_external_luad": gse_res,
        "tcga_paad_stress_test": paad_res
    }

    out_file = EVIDENCE_DIR / "bounded_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_evidence, f, indent=2)

    # Save patient-level predictions
    preds_df = pd.DataFrame({
        "patient_id": luad_data["ids"],
        "event": luad_data["event"].astype(int),
        "time": luad_data["time"],
        "pred_m1_clinical": test_preds["m1_clinical"],
        "pred_m2_clin_coxnet": test_preds["m2_clin_coxnet"],
        "pred_m3_clin_chara": test_preds["m3_clin_chara"],
        "pred_diag_coxnet_only": test_preds["diag_coxnet_only"],
        "pred_diag_chara_only": test_preds["diag_chara_only"]
    })
    preds_df.to_csv(EVIDENCE_DIR / "patient_predictions_tcga_luad.csv", index=False)

    print("\n[OK] Final Bounded Benchmark execution complete.")
    print(f"     Saved evidence: {out_file}")
    print(f"     Saved predictions: {EVIDENCE_DIR / 'patient_predictions_tcga_luad.csv'}")

if __name__ == "__main__":
    main()
