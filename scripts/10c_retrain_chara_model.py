#!/usr/bin/env python3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent

def main():
    genes = pd.read_csv(ROOT / "intersecting_genes_4337.txt", header=None)[0].astype(str).tolist()
    expression = pd.read_csv(ROOT / "TCGA-LUAD_expression.csv", index_col=0)
    survival = pd.read_csv(ROOT / "TCGA-LUAD_survival.csv", index_col=0)
    samples = expression.index.intersection(survival.index)
    X = expression.loc[samples, genes].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0).to_numpy(dtype=np.float64)
    L = pd.read_csv(ROOT / "Laplacian_Chara_4337.csv", index_col=0).loc[genes, genes].to_numpy(dtype=np.float64)
    if L.shape != (len(genes), len(genes)):
        raise ValueError("Chara Laplacian and feature order are inconsistent.")
    # Topological heat-kernel smoothing in the valid feature space.
    evals, evecs = np.linalg.eigh((L + L.T) / 2.0)
    evals = np.clip(evals, 0.0, None)
    W = (evecs * np.exp(-0.1 * evals)) @ evecs.T
    X_graph = X @ W
    y = np.array(list(zip(survival.loc[samples, "Event"].astype(bool), survival.loc[samples, "Time"].astype(float))), dtype=[("event", "?"), ("time", "<f8")])
    model = CoxnetSurvivalAnalysis(
        l1_ratio=0.5,
        alpha_min_ratio=0.01,
        max_iter=5000,
        fit_baseline_model=True,
    )
    model.fit(X_graph, y)
    counts = np.sum(model.coef_ != 0, axis=0)
    valid = np.where(counts > 0)[0]
    if len(valid) == 0:
        raise ValueError("All retrained Coxnet coefficient paths are zero.")
    # Select alpha using training-only cross-validation, never the external cohort.
    cv_scores = np.full(model.coef_.shape[1], np.nan, dtype=np.float64)
    fold_scores = [[] for _ in range(model.coef_.shape[1])]
    for train_idx, test_idx in KFold(n_splits=3, shuffle=True, random_state=4337).split(X_graph):
        fold_model = CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alphas=model.alphas_, max_iter=5000
        )
        fold_model.fit(X_graph[train_idx], y[train_idx])
        fold_risk = X_graph[test_idx] @ fold_model.coef_
        for j in range(fold_risk.shape[1]):
            if np.count_nonzero(fold_model.coef_[:, j]) > 0:
                fold_scores[j].append(
                    concordance_index_censored(
                        y[test_idx]["event"], y[test_idx]["time"], fold_risk[:, j]
                    )[0]
                )
    for j, scores in enumerate(fold_scores):
        if scores:
            cv_scores[j] = float(np.mean(scores))
    opt_idx = int(np.nanargmax(cv_scores))
    if not np.isfinite(cv_scores[opt_idx]):
        opt_idx = int(valid[-1])
    joblib.dump({"model": model, "features": genes, "alpha_index": opt_idx, "non_zero_count": int(counts[opt_idx])}, ROOT / "chara_model_4337.pkl")
    print(f"Saved chara_model_4337.pkl with {len(genes)} features; alpha {opt_idx}, non-zero {counts[opt_idx]}, CV C-index {cv_scores[opt_idx]:.4f}.")

if __name__ == "__main__":
    main()
