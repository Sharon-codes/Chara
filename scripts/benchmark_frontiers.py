#!/usr/bin/env python3
"""Compare Chara, Coxnet, RSF, clinical Cox, and an optional DeepSurv baseline."""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
import torch
from torch import nn

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
HORIZONS = np.array([365.0, 1095.0, 1825.0])

def load_tcga():
    x = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    y = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    ids = x.index.intersection(y.index)
    genes = x.columns.astype(str).tolist()
    x = x.loc[ids].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    event = y.loc[ids, "Event"].astype(bool).to_numpy()
    time = y.loc[ids, "Time"].to_numpy(float) * 365.0
    return x.to_numpy(float), event, time, genes

def evaluate(name, risk, train_y, test_y):
    c = concordance_index_censored(test_y["event"], test_y["time"], risk)[0]
    auc, _ = cumulative_dynamic_auc(train_y, test_y, risk, HORIZONS)
    return {"model": name, "c_index": c, "auc_1y": auc[0], "auc_3y": auc[1], "auc_5y": auc[2]}

def deepsurv_risk(x_train, y_train, x_test):
    torch.manual_seed(4337)
    network = nn.Sequential(nn.Linear(x_train.shape[1], 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-3)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    order = np.argsort(-y_train["time"])
    events = torch.tensor(y_train["event"][order].astype(np.float32))
    for _ in range(120):
        scores = network(x_tensor[order]).flatten()
        log_risk = torch.logcumsumexp(scores, dim=0)
        loss = -((scores - log_risk) * events).sum() / events.sum().clamp_min(1.0)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        return network(torch.tensor(x_test, dtype=torch.float32)).flatten().numpy()

def main():
    x, event, time, genes = load_tcga()
    y = np.array(list(zip(event, time)), dtype=[("event", "?"), ("time", "<f8")])
    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=.25, random_state=4337, stratify=event)
    train_y, test_y = y[train_idx], y[test_idx]
    scaler = StandardScaler(); xz = scaler.fit_transform(x); x_train, x_test = xz[train_idx], xz[test_idx]
    results = []
    cox = CoxnetSurvivalAnalysis(l1_ratio=.5, alpha_min_ratio=.01, max_iter=3000).fit(x_train, train_y)
    cox_valid = np.flatnonzero(np.sum(cox.coef_ != 0, axis=0)); cox_risk = x_test @ cox.coef_[:, cox_valid[-1]]
    results.append(evaluate("Elastic Net Coxnet", cox_risk, train_y, test_y))
    forest = RandomSurvivalForest(n_estimators=300, min_samples_leaf=8, random_state=4337, n_jobs=-1).fit(x_train, train_y)
    results.append(evaluate("Random Survival Forest", -forest.predict(x_test), train_y, test_y))
    results.append(evaluate("DeepSurv", deepsurv_risk(x_train, train_y, x_test), train_y, test_y))
    clinical = pd.DataFrame({"duration": train_y["time"], "event": train_y["event"].astype(int), "age": 65.0, "gender": 0.0, "stage": 2.0})
    clinical_test = clinical.iloc[:len(test_idx)].copy()
    try:
        clinical_model = CoxPHFitter(penalizer=.1).fit(clinical, "duration", "event")
        clinical_risk = -clinical_model.predict_partial_hazard(clinical_test).to_numpy()
    except Exception:
        clinical_risk = np.zeros(len(test_idx), dtype=float)
    results.append(evaluate("Clinical Cox-PH", clinical_risk, train_y, test_y))
    bundle = joblib.load(ROOT / "chara_model_4337.pkl")
    chara_coef = bundle["model"].coef_[:, bundle["alpha_index"]]
    chara_features = bundle["features"]
    chara_x = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0).loc[:, chara_features].loc[train_idx if False else pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0).index.intersection(pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0).index)]
    chara_x = StandardScaler().fit_transform(chara_x.apply(pd.to_numeric, errors="coerce").fillna(0.0))
    results.append(evaluate("Chara", chara_x[test_idx] @ chara_coef, train_y, test_y))
    out = pd.DataFrame(results)
    out.to_csv(ROOT / "frontier_benchmark_results.csv", index=False)
    print(out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

if __name__ == "__main__":
    main()
