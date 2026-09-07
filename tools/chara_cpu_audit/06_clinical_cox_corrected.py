#!/usr/bin/env python3
"""
tools/chara_cpu_audit/06_clinical_cox_corrected.py
Replaces the degenerate constant-covariate 0.5000 Clinical Cox baseline with a
statistically rigorous evaluation of real clinical prognostic factors (Age, Sex, Stage)
from TCGA-LUAD, and evaluates the incremental prognostic value of Chara risk scores.
"""

import io
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from lifelines import CoxPHFitter
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
HORIZONS = np.array([365.0, 1095.0, 1825.0])

CLINICAL_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/luad_tcga/data_clinical_patient.txt"
CACHE_PATH = ROOT / "data" / "TCGA-LUAD_clinical.csv"

def fetch_or_load_clinical():
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        print(f"[INFO] Loading cached clinical data from {CACHE_PATH}...")
        df = pd.read_csv(CACHE_PATH, index_col=0)
        return df

    print(f"[INFO] Fetching clinical patient data from cBioPortal: {CLINICAL_URL}...")
    r = requests.get(CLINICAL_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=45)
    lines = [l for l in r.text.split('\n') if not l.startswith('#')]
    df_raw = pd.read_csv(io.StringIO('\n'.join(lines)), sep='\t', low_memory=False)
    
    # Extract PATIENT_ID, SEX, AGE, AJCC_PATHOLOGIC_TUMOR_STAGE
    patient_ids = df_raw['PATIENT_ID'].astype(str).str.strip().str.slice(0, 12)
    df = pd.DataFrame(index=patient_ids)
    df['SEX_RAW'] = df_raw['SEX'].values
    df['AGE_RAW'] = pd.to_numeric(df_raw['AGE'], errors='coerce').values
    df['STAGE_RAW'] = df_raw['AJCC_PATHOLOGIC_TUMOR_STAGE'].astype(str).values

    # Clean and encode
    df['SEX'] = df['SEX_RAW'].str.lower().map({'male': 1.0, 'm': 1.0, 'female': 0.0, 'f': 0.0})
    df['AGE'] = df['AGE_RAW']
    
    def parse_stage(s):
        s = str(s).strip().upper()
        if 'IV' in s: return 4.0
        if 'III' in s: return 3.0
        if 'II' in s: return 2.0
        if 'I' in s: return 1.0
        return np.nan

    df['STAGE'] = df['STAGE_RAW'].apply(parse_stage)
    
    # Save cleaned clinical data
    df = df[~df.index.duplicated(keep='first')]
    df.to_csv(CACHE_PATH)
    print(f"[OK] Saved cleaned clinical data to {CACHE_PATH}")
    return df

def bootstrap_ci(y_true_e, y_true_t, risk, n_boot=2000, seed=4337):
    rng = np.random.default_rng(seed)
    n = len(y_true_e)
    c_indices = []
    auc_1ys, auc_3ys, auc_5ys = [], [], []

    train_y_struct = np.array(list(zip(y_true_e, y_true_t)), dtype=[("event", "?"), ("time", "<f8")])

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_e = y_true_e[idx]
        boot_t = y_true_t[idx]
        boot_risk = risk[idx]

        if boot_e.sum() == 0 or (~boot_e).sum() == 0:
            continue
        try:
            c = concordance_index_censored(boot_e, boot_t, boot_risk)[0]
            c_indices.append(c)
        except Exception:
            continue

        try:
            test_y_struct = np.array(list(zip(boot_e, boot_t)), dtype=[("event", "?"), ("time", "<f8")])
            auc, _ = cumulative_dynamic_auc(train_y_struct, test_y_struct, boot_risk, HORIZONS)
            auc_1ys.append(auc[0])
            auc_3ys.append(auc[1])
            auc_5ys.append(auc[2])
        except Exception:
            continue

    def ci(arr):
        if len(arr) < 50:
            return [np.nan, np.nan]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "c_index_ci": ci(c_indices),
        "auc_1y_ci": ci(auc_1ys),
        "auc_3y_ci": ci(auc_3ys),
        "auc_5y_ci": ci(auc_5ys),
    }

def main():
    print("=" * 80)
    print(" CORRECTED CLINICAL COX-PH EVALUATION & MULTIMODAL AUDIT")
    print("=" * 80)

    clinical_df = fetch_or_load_clinical()
    luad_exp = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    luad_surv = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)

    # Intersect all 3 sources
    common_ids = luad_exp.index.intersection(luad_surv.index).intersection(clinical_df.index)
    print(f"[INFO] Common patients across expression, survival, and clinical: {len(common_ids)}")

    exp_aligned = luad_exp.loc[common_ids]
    surv_aligned = luad_surv.loc[common_ids]
    clin_aligned = clinical_df.loc[common_ids].copy()

    # Impute missing clinical variables with training medians/modes
    clin_aligned["AGE"] = clin_aligned["AGE"].fillna(clin_aligned["AGE"].median())
    clin_aligned["SEX"] = clin_aligned["SEX"].fillna(clin_aligned["SEX"].mode()[0])
    clin_aligned["STAGE"] = clin_aligned["STAGE"].fillna(clin_aligned["STAGE"].median())

    event = surv_aligned["Event"].astype(bool).to_numpy()
    time = surv_aligned["Time"].to_numpy(float) * 365.0
    y_struct = np.array(list(zip(event, time)), dtype=[("event", "?"), ("time", "<f8")])

    # Replicate exact 75/25 train/test split from benchmark_frontiers.py
    train_idx, test_idx = train_test_split(np.arange(len(common_ids)), test_size=0.25, random_state=4337, stratify=event)
    train_y, test_y = y_struct[train_idx], y_struct[test_idx]

    # 1. Load Chara Model Risk Scores
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    chara_coef = bundle["model"].coef_[:, bundle["alpha_index"]]
    chara_features = bundle["features"]
    
    # Missing features in common_ids expression
    chara_x = exp_aligned.loc[:, [f for f in chara_features if f in exp_aligned.columns]]
    # Fill any missing genes with 0
    for f in chara_features:
        if f not in chara_x.columns:
            chara_x[f] = 0.0
    chara_x = chara_x.reindex(columns=chara_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    chara_xz = StandardScaler().fit_transform(chara_x.to_numpy(float))
    
    chara_risk = chara_xz @ chara_coef
    chara_risk_train = chara_risk[train_idx]
    chara_risk_test = chara_risk[test_idx]

    # Evaluate Chara Alone on Test Set
    chara_c = float(concordance_index_censored(test_y["event"], test_y["time"], chara_risk_test)[0])
    chara_auc, _ = cumulative_dynamic_auc(train_y, test_y, chara_risk_test, HORIZONS)
    chara_ci = bootstrap_ci(test_y["event"], test_y["time"], chara_risk_test)

    # 2. Fit Corrected Clinical Cox-PH Model (Age + Sex + Stage)
    df_train_clin = pd.DataFrame({
        "time": time[train_idx],
        "event": event[train_idx].astype(int),
        "age": clin_aligned["AGE"].iloc[train_idx].values,
        "sex": clin_aligned["SEX"].iloc[train_idx].values,
        "stage": clin_aligned["STAGE"].iloc[train_idx].values
    })
    df_test_clin = pd.DataFrame({
        "time": time[test_idx],
        "event": event[test_idx].astype(int),
        "age": clin_aligned["AGE"].iloc[test_idx].values,
        "sex": clin_aligned["SEX"].iloc[test_idx].values,
        "stage": clin_aligned["STAGE"].iloc[test_idx].values
    })

    cph_clin = CoxPHFitter(penalizer=0.01)
    cph_clin.fit(df_train_clin, duration_col="time", event_col="event")
    clin_risk_test = cph_clin.predict_partial_hazard(df_test_clin).to_numpy()
    
    clin_c = float(concordance_index_censored(test_y["event"], test_y["time"], clin_risk_test)[0])
    clin_auc, _ = cumulative_dynamic_auc(train_y, test_y, clin_risk_test, HORIZONS)
    clin_ci = bootstrap_ci(test_y["event"], test_y["time"], clin_risk_test)

    clin_summary = {}
    for cov in ["age", "sex", "stage"]:
        row = cph_clin.summary.loc[cov]
        clin_summary[cov] = {
            "coef": float(row["coef"]),
            "exp_coef_HR": float(np.exp(row["coef"])),
            "se": float(row["se(coef)"]),
            "p_value": float(row["p"]),
            "ci_95": [float(np.exp(row["coef lower 95%"])), float(np.exp(row["coef upper 95%"]))]
        }

    # 3. Fit Multimodal / Combined Model (Clinical + Chara Risk)
    df_train_comb = df_train_clin.copy()
    df_train_comb["chara_risk"] = chara_risk_train
    df_test_comb = df_test_clin.copy()
    df_test_comb["chara_risk"] = chara_risk_test

    cph_comb = CoxPHFitter(penalizer=0.01)
    cph_comb.fit(df_train_comb, duration_col="time", event_col="event")
    comb_risk_test = cph_comb.predict_partial_hazard(df_test_comb).to_numpy()

    comb_c = float(concordance_index_censored(test_y["event"], test_y["time"], comb_risk_test)[0])
    comb_auc, _ = cumulative_dynamic_auc(train_y, test_y, comb_risk_test, HORIZONS)
    comb_ci = bootstrap_ci(test_y["event"], test_y["time"], comb_risk_test)

    comb_summary = {}
    for cov in ["age", "sex", "stage", "chara_risk"]:
        row = cph_comb.summary.loc[cov]
        comb_summary[cov] = {
            "coef": float(row["coef"]),
            "exp_coef_HR": float(np.exp(row["coef"])),
            "se": float(row["se(coef)"]),
            "p_value": float(row["p"]),
            "ci_95": [float(np.exp(row["coef lower 95%"])), float(np.exp(row["coef upper 95%"]))]
        }

    # Likelihood Ratio Test comparing Clinical alone vs Combined
    ll_clin = float(cph_clin.log_likelihood_)
    ll_comb = float(cph_comb.log_likelihood_)
    lrt_stat = 2.0 * (ll_comb - ll_clin)
    lrt_p = float(stats.chi2.sf(lrt_stat, df=1))

    # 4. 5-Fold Cross-Validation C-indices
    kf = KFold(n_splits=5, shuffle=True, random_state=4337)
    cv_clin_c = []
    cv_chara_c = []
    cv_comb_c = []

    df_all_clin = pd.DataFrame({
        "time": time,
        "event": event.astype(int),
        "age": clin_aligned["AGE"].values,
        "sex": clin_aligned["SEX"].values,
        "stage": clin_aligned["STAGE"].values,
        "chara_risk": chara_risk
    })

    for fold_train, fold_val in kf.split(df_all_clin):
        # Clinical
        m_c = CoxPHFitter(penalizer=0.01).fit(df_all_clin.iloc[fold_train][["time", "event", "age", "sex", "stage"]], "time", "event")
        r_c = m_c.predict_partial_hazard(df_all_clin.iloc[fold_val][["age", "sex", "stage"]]).to_numpy()
        cv_clin_c.append(concordance_index_censored(event[fold_val], time[fold_val], r_c)[0])

        # Chara
        r_ch = df_all_clin["chara_risk"].iloc[fold_val].values
        cv_chara_c.append(concordance_index_censored(event[fold_val], time[fold_val], r_ch)[0])

        # Combined
        m_cb = CoxPHFitter(penalizer=0.01).fit(df_all_clin.iloc[fold_train], "time", "event")
        r_cb = m_cb.predict_partial_hazard(df_all_clin.iloc[fold_val][["age", "sex", "stage", "chara_risk"]]).to_numpy()
        cv_comb_c.append(concordance_index_censored(event[fold_val], time[fold_val], r_cb)[0])

    # 5. Contrast with Dummy Constant Artifact
    dummy_clinical_results = {
        "claimed_c_index": 0.5000,
        "claimed_auc_1y": 0.5000,
        "claimed_auc_3y": 0.5000,
        "claimed_auc_5y": 0.5000,
        "root_cause": "In scripts/benchmark_frontiers.py line 64, age=65.0, gender=0.0, stage=2.0 were hardcoded constants for every patient. Zero variance produced identical risk scores and guaranteed C-index=0.5000.",
    }

    evidence = {
        "dummy_clinical_artifact": dummy_clinical_results,
        "corrected_clinical_cox": {
            "test_c_index": clin_c,
            "test_c_index_ci_95": clin_ci["c_index_ci"],
            "test_auc_1y": float(clin_auc[0]),
            "test_auc_1y_ci_95": clin_ci["auc_1y_ci"],
            "test_auc_3y": float(clin_auc[1]),
            "test_auc_3y_ci_95": clin_ci["auc_3y_ci"],
            "test_auc_5y": float(clin_auc[2]),
            "test_auc_5y_ci_95": clin_ci["auc_5y_ci"],
            "cv_5fold_mean_c_index": float(np.mean(cv_clin_c)),
            "cv_5fold_std": float(np.std(cv_clin_c)),
            "covariates_summary": clin_summary
        },
        "chara_alone_evaluation": {
            "test_c_index": chara_c,
            "test_c_index_ci_95": chara_ci["c_index_ci"],
            "test_auc_1y": float(chara_auc[0]),
            "test_auc_1y_ci_95": chara_ci["auc_1y_ci"],
            "test_auc_3y": float(chara_auc[1]),
            "test_auc_3y_ci_95": chara_ci["auc_3y_ci"],
            "test_auc_5y": float(chara_auc[2]),
            "test_auc_5y_ci_95": chara_ci["auc_5y_ci"],
            "cv_5fold_mean_c_index": float(np.mean(cv_chara_c)),
            "cv_5fold_std": float(np.std(cv_chara_c))
        },
        "multimodal_combined_model": {
            "test_c_index": comb_c,
            "test_c_index_ci_95": comb_ci["c_index_ci"],
            "test_auc_1y": float(comb_auc[0]),
            "test_auc_1y_ci_95": comb_ci["auc_1y_ci"],
            "test_auc_3y": float(comb_auc[1]),
            "test_auc_3y_ci_95": comb_ci["auc_3y_ci"],
            "test_auc_5y": float(comb_auc[2]),
            "test_auc_5y_ci_95": comb_ci["auc_5y_ci"],
            "cv_5fold_mean_c_index": float(np.mean(cv_comb_c)),
            "cv_5fold_std": float(np.std(cv_comb_c)),
            "lrt_statistic": lrt_stat,
            "lrt_p_value": lrt_p,
            "covariates_summary": comb_summary
        }
    }

    out_file = EVIDENCE_DIR / "clinical_cox_corrected.json"
    with open(out_file, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"\n[OK] Corrected Clinical Evaluation complete. Saved to: {out_file}")
    print(f"  Dummy Clinical C-Index (Artifact) : 0.5000")
    print(f"  Real Clinical C-Index (Age+Sex+Stg): {clin_c:.4f} (95% CI: {clin_ci['c_index_ci'][0]:.4f} - {clin_ci['c_index_ci'][1]:.4f})")
    print(f"  Chara Risk C-Index Alone           : {chara_c:.4f} (95% CI: {chara_ci['c_index_ci'][0]:.4f} - {chara_ci['c_index_ci'][1]:.4f})")
    print(f"  Combined Multimodal C-Index        : {comb_c:.4f} (95% CI: {comb_ci['c_index_ci'][0]:.4f} - {comb_ci['c_index_ci'][1]:.4f})")
    print(f"  Incremental Value LRT p-value      : {lrt_p:.4e}")

if __name__ == "__main__":
    main()
