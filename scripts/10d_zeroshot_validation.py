#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
v09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v09)

def main():
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    model, genes = bundle["model"], list(bundle["features"])
    gse, survival = v09.parse_geo_gse31210()
    external = v09.load_gse31210_expression(gse).reindex(columns=genes)
    external = external.apply(lambda c: v09.pd.to_numeric(c, errors="coerce"))
    external = external.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = min(len(external), len(survival))
    X = external.iloc[:n].to_numpy(dtype=np.float64)
    x_scaled = StandardScaler().fit_transform(X)
    coef_matrix = np.asarray(model.coef_, dtype=np.float64)
    non_zero_counts = np.sum(coef_matrix != 0, axis=0)
    valid_indices = np.where(non_zero_counts > 0)[0]
    if len(valid_indices) == 0:
        raise ValueError("Catastrophic model failure: all coefficient paths are zero.")
    opt_idx = int(bundle.get("alpha_index", valid_indices[-1]))
    if opt_idx not in valid_indices:
        opt_idx = int(valid_indices[-1])
    optimal_betas = coef_matrix[:, opt_idx]
    risk_scores = np.dot(x_scaled, optimal_betas)
    print(f"Debug X_scaled Variance: {np.var(x_scaled):.6f}")
    print(f"Debug Risk Score Unique Values: {len(np.unique(risk_scores))}")
    event = survival.iloc[:n]["Event"].astype(bool).to_numpy()
    time = survival.iloc[:n]["TimeMonths"].astype(float).to_numpy()
    c = concordance_index_censored(event, time, risk_scores)[0]
    print("======================================================================")
    print(" STRICT-INTERSECTION ZERO-SHOT VALIDATION: GSE31210 (LUAD)")
    print("======================================================================")
    print(f" External Patients Processed   : {n}")
    print(f" Intersection Features         : {len(genes)}")
    print(f" Selected Alpha Index          : {opt_idx}")
    print(f" Zero-Shot C-Index             : {c:.4f}")
    print("======================================================================")
    if c <= 0.60:
        raise SystemExit(f"FAIL: Zero-Shot C-Index {c:.4f} does not exceed 0.60.")

if __name__ == "__main__":
    main()
