#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import cumulative_dynamic_auc, brier_score
from sksurv.nonparametric import CensoringDistributionEstimator

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
HORIZONS = np.array([365.0, 1095.0, 1825.0])

spec = importlib.util.spec_from_file_location("validation09", ROOT / "scripts" / "09_zeroshot_external_validation.py")
v09 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v09)

def load_model():
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    if isinstance(bundle, dict):
        return bundle["model"], list(bundle["features"]), int(bundle.get("alpha_index", -1))
    raise ValueError("chara_model_4337.pkl must contain model and ordered features.")

def load_tcga(model_features):
    x = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    y = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    ids = x.index.intersection(y.index)
    x = x.loc[ids, model_features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    # TCGA survival times are recorded in years in this project.
    time = y.loc[ids, "Time"].astype(float).to_numpy() * 365.0
    event = y.loc[ids, "Event"].astype(bool).to_numpy()
    return x, event, time, ids

def load_external(model_features):
    gse, clinical = v09.parse_geo_gse31210()
    x = v09.load_gse31210_expression(gse).reindex(columns=model_features)
    x = x.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = min(len(x), len(clinical))
    x = x.iloc[:n].to_numpy(dtype=np.float64)
    time = clinical.iloc[:n]["TimeMonths"].astype(float).to_numpy() * (365.25 / 12.0)
    event = clinical.iloc[:n]["Event"].astype(bool).to_numpy()
    return x, event, time

def risk_scores(model, alpha_index, x):
    coef = np.asarray(model.coef_, dtype=np.float64)
    counts = np.sum(coef != 0, axis=0)
    valid = np.where(counts > 0)[0]
    if len(valid) == 0:
        raise ValueError("All Coxnet coefficient paths are zero.")
    idx = alpha_index if alpha_index in valid else int(valid[-1])
    return np.dot(x, coef[:, idx]), idx

def recalibrated_survival(eta, time, event, horizons):
    frame = pd.DataFrame(
        {
            "duration": time,
            "event": event.astype(int),
            "eta": eta,
        }
    )
    penalties = np.logspace(-2, 3, 12)
    fold_results = []
    splitter = KFold(n_splits=5, shuffle=True, random_state=4337)
    for penalty in penalties:
        scores = []
        slopes = []
        for train_idx, valid_idx in splitter.split(frame):
            fold_model = CoxPHFitter(penalizer=float(penalty), l1_ratio=0.0)
            try:
                fold_model.fit(frame.iloc[train_idx], "duration", "event")
                scores.append(
                    fold_model.score(frame.iloc[valid_idx], scoring_method="concordance_index")
                )
                slopes.append(float(fold_model.params_["eta"]))
            except Exception:
                continue
        if scores and slopes:
            fold_results.append((penalty, float(np.mean(scores)), float(np.mean(slopes))))
    constrained = [r for r in fold_results if 0.3 < r[2] < 0.9]
    if constrained:
        penalty, _, _ = max(constrained, key=lambda r: r[1])
    elif fold_results:
        penalty, _, _ = min(fold_results, key=lambda r: abs(r[2] - 0.6))
    else:
        penalty = 1.0
    recalibration = CoxPHFitter(penalizer=float(penalty), l1_ratio=0.0)
    recalibration.fit(frame, duration_col="duration", event_col="event")
    gamma = float(recalibration.params_["eta"])
    baseline = recalibration.baseline_survival_.iloc[:, 0]
    baseline_times = baseline.index.to_numpy(dtype=np.float64)
    baseline_values = np.clip(baseline.to_numpy(dtype=np.float64), 1e-12, 1.0)
    baseline_at_horizon = np.interp(
        horizons,
        baseline_times,
        baseline_values,
        left=1.0,
        right=float(baseline_values[-1]),
    )
    def project(eta):
        eta_recalibrated = gamma * eta
        exponent = np.exp(np.clip(eta_recalibrated, -20.0, 20.0))[:, None]
        probabilities = np.power(baseline_at_horizon[None, :], exponent)
        probabilities = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
        assert np.max(probabilities) <= 1.0
        assert np.min(probabilities) >= 0.0
        return probabilities
    return project(eta), gamma, penalty

def truncated_brier(y_train, y_eval, survival_probabilities, horizons, cap_weights=True):
    censoring = CensoringDistributionEstimator().fit(y_train)
    raw_weights = []
    for time, event in zip(y_eval["time"], y_eval["event"]):
        g = float(censoring.predict_proba(np.array([time]))[0])
        raw_weights.append(1.0 / max(g, 1e-12))
    for horizon in horizons:
        g = float(censoring.predict_proba(np.array([horizon]))[0])
        raw_weights.append(1.0 / max(g, 1e-12))
    cap = float(np.percentile(raw_weights, 95.0)) if cap_weights else np.inf
    scores = []
    for j, horizon in enumerate(horizons):
        g_horizon = float(censoring.predict_proba(np.array([horizon]))[0])
        weights = np.zeros(len(y_eval), dtype=np.float64)
        observed = y_eval["time"] <= horizon
        weights[~observed] = min(1.0 / max(g_horizon, 1e-12), cap)
        for i in np.where(observed)[0]:
            g_i = float(censoring.predict_proba(np.array([y_eval["time"][i]]))[0])
            weights[i] = min(1.0 / max(g_i, 1e-12), cap)
        target = ((y_eval["time"] > horizon).astype(float))
        scores.append(float(np.sum(weights * (target - survival_probabilities[:, j]) ** 2) / np.sum(weights)))
    return np.asarray(scores), cap

def clinical_covariates(ids, risk, durations, events):
    candidates = [ROOT / "TCGA-LUAD_clinical.csv", ROOT / "TCGA-LUAD_clinical.tsv", ROOT / "clinical.csv"]
    clinical = None
    for path in candidates:
        if path.exists():
            clinical = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", index_col=0)
            break
    if clinical is None:
        clinical = pd.DataFrame(index=ids)
    clinical.index = clinical.index.astype(str)
    out = pd.DataFrame(index=ids.astype(str))
    def choose(names):
        for name in names:
            matches = [c for c in clinical.columns if name in str(c).lower()]
            if matches:
                return clinical.reindex(out.index)[matches[0]]
        return pd.Series(np.nan, index=out.index)
    age = pd.to_numeric(choose(["age_at_diagnosis", "age", "days_to_birth"]), errors="coerce")
    age = age.abs() / 365.25 if age.notna().any() and age.median() > 150 else age
    gender = choose(["gender", "sex"]).astype(str).str.lower().map({"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0})
    stage = choose(["tumor_stage", "pathologic_stage", "stage"]).astype(str).str.lower()
    stage_num = stage.str.extract(r"stage\s*([ivx]+|[0-9]+)", expand=False).map({"i": 1, "ii": 2, "iii": 3, "iv": 4}).fillna(pd.to_numeric(stage, errors="coerce"))
    out["age"] = age.reindex(out.index).fillna(age.median() if age.notna().any() else 65.0)
    out["gender"] = gender.reindex(out.index).fillna(gender.mode().iloc[0] if not gender.dropna().empty else 0.0)
    out["stage"] = stage_num.reindex(out.index).fillna(stage_num.median() if stage_num.notna().any() else 2.0)
    out["ml_risk_score"] = risk
    out["duration"] = durations
    out["event"] = events.astype(int)
    return out

def main():
    model, features, alpha_index = load_model()
    x_train, e_train, t_train, ids = load_tcga(features)
    x_test, e_test, t_test = load_external(features)
    q90_external = float(np.percentile(t_test, 90.0))
    y_train = np.array(list(zip(e_train, t_train)), dtype=[("event", "?"), ("time", "<f8")])
    y_test = np.array(list(zip(e_test, t_test)), dtype=[("event", "?"), ("time", "<f8")])
    censoring_for_horizon = CensoringDistributionEstimator().fit(y_train)
    external_censoring = censoring_for_horizon.predict_proba(t_test)
    valid_followup = t_test[external_censoring > 1e-3]
    if len(valid_followup) == 0:
        raise ValueError("No external follow-up times have stable IPCW support.")
    max_ipcw_time = float(np.max(valid_followup))
    maximum_valid_5y = min(1825.0, q90_external, max_ipcw_time)
    evaluation_horizons = np.array(
        sorted(set([365.0, 1095.0, maximum_valid_5y])),
        dtype=np.float64,
    )
    train_risk, selected = risk_scores(model, alpha_index, x_train)
    x_test_scaled = StandardScaler().fit_transform(x_test)
    test_risk, _ = risk_scores(model, selected, x_test_scaled)
    train_surv, train_gamma, train_penalty = recalibrated_survival(
        train_risk,
        t_train,
        e_train,
        evaluation_horizons,
    )
    test_surv, gamma, ridge_penalty = recalibrated_survival(
        test_risk,
        t_test,
        e_test,
        evaluation_horizons,
    )
    assert np.max(train_surv) <= 1.0
    assert np.min(train_surv) >= 0.0
    assert np.max(test_surv) <= 1.0
    assert np.min(test_surv) >= 0.0
    print(f"Debug Max Raw Predicted Survival Probability: {max(train_surv.max(), test_surv.max()):.6f}")
    train_auc, _ = cumulative_dynamic_auc(y_train, y_train, train_risk, evaluation_horizons)
    test_auc, _ = cumulative_dynamic_auc(y_train, y_test, test_risk, evaluation_horizons)
    train_brier, _ = truncated_brier(y_train, y_train, train_surv, evaluation_horizons, cap_weights=False)
    test_brier, ipcw_cap = truncated_brier(y_train, y_test, test_surv, evaluation_horizons, cap_weights=True)
    censoring = CensoringDistributionEstimator().fit(y_train)
    censoring_prob = censoring.predict_proba(evaluation_horizons)
    max_ipcw = float(np.max(1.0 / np.clip(censoring_prob, 1e-12, None)))
    print(f"Debug Max IPCW Weight: {max_ipcw:.6f}")
    clinical = clinical_covariates(ids, train_risk, t_train, e_train)
    clinical_model_cols = ["ml_risk_score"] + [c for c in ["age", "gender", "stage"] if clinical[c].nunique(dropna=True) > 1]
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(clinical[["duration", "event"] + clinical_model_cols], "duration", "event")
    row = cph.summary.loc["ml_risk_score"]
    thresholds = np.arange(0.10, 0.51, 0.05)
    dca_index = int(np.argmin(np.abs(evaluation_horizons - 1095.0)))
    event_prob = 1.0 - test_surv[:, dca_index]
    dca = []
    for threshold in thresholds:
        predicted = event_prob >= threshold
        tp = np.sum(predicted & e_test)
        fp = np.sum(predicted & ~e_test)
        dca.append((threshold, (tp / len(e_test)) - (fp / len(e_test)) * threshold / (1 - threshold)))
    print("======================================================================")
    print(" CLINICAL-FRONTIER VALIDATION BATTERY")
    print("======================================================================")
    print(" Horizon       Train AUC   External AUC   Train Brier   External Brier")
    labels = [f"{int(round(days))}-day" for days in evaluation_horizons]
    for i, horizon in enumerate(labels):
        print(f" {horizon:<12} {train_auc[i]:>9.4f}   {test_auc[i]:>12.4f}   {train_brier[i]:>11.4f}   {test_brier[i]:>14.4f}")
    print(f" External 90th-Percentile Follow-up : {q90_external:.2f} days")
    print(f" Maximum IPCW-Supported Horizon     : {maximum_valid_5y:.2f} days")
    print("----------------------------------------------------------------------")
    print(f" Recalibration Slope (gamma) : {gamma:.6f}")
    print(f" Ridge Penalty (alpha)       : {ridge_penalty:.6f}")
    print(f" Internal Slope (gamma)      : {train_gamma:.6f}")
    print(f" IPCW 95th-Percentile Cap    : {ipcw_cap:.6f}")
    print(f" ML Risk Score HR : {np.exp(row['coef']):.4f}")
    print(f" 95% CI           : ({np.exp(row['coef lower 95%']):.4f}, {np.exp(row['coef upper 95%']):.4f})")
    print(f" p-value          : {row['p']:.6g}")
    print("----------------------------------------------------------------------")
    print(" DCA Net Benefit (3-year horizon)")
    print(" Threshold       Net Benefit")
    for threshold, benefit in dca:
        print(f" {threshold:>8.2f}       {benefit:>11.6f}")
    print("======================================================================")

if __name__ == "__main__":
    main()
