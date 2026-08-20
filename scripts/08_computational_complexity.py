#!/usr/bin/env python3
"""Benchmark the computational overhead of the STRING baseline and Chara MD pipeline.

- Measure fit execution time with time.perf_counter()
- Measure peak memory use with tracemalloc
- Perform a single fixed-alpha Coxnet fit in each architecture to isolate raw overhead
"""

import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis

warnings.filterwarnings("ignore", category=ConvergenceWarning)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT


def load_dataset():
    expr = pd.read_csv(DATA_DIR / "TCGA-LUAD_expression.csv", index_col=0)
    surv = pd.read_csv(DATA_DIR / "TCGA-LUAD_survival.csv", index_col=0)

    common_idx = expr.index.intersection(surv.index)
    expr = expr.loc[common_idx].copy()
    surv = surv.loc[common_idx].copy()

    expr.columns = [str(c).strip() for c in expr.columns]
    expr = expr.loc[:, [c for c in expr.columns if c and c.lower() != "nan"]]

    X = expr.to_numpy(dtype=np.float64)
    y = np.array(
        list(zip(surv["Event"].astype(bool).to_numpy(), surv["Time"].astype(np.float64).to_numpy())),
        dtype=[("Status", "?"), ("Survival_in_days", "<f8")],
    )
    return X, y


def load_laplacian(name):
    lap = pd.read_csv(DATA_DIR / name, index_col=0)
    lap.columns = [str(c).strip() for c in lap.columns]
    lap.index = [str(i).strip() for i in lap.index]
    lap = lap.loc[:, [c for c in lap.columns if c and c.lower() != "nan"]]
    lap = lap.loc[[i for i in lap.index if i and i.lower() != "nan"], :]
    return lap.to_numpy(dtype=np.float64)


def fit_model(X, y, L=None, alpha=0.2):
    if L is not None:
        evals, evecs = np.linalg.eigh(L)
        evals = np.clip(evals, 0.0, None)
        W = evecs @ np.diag(1.0 / np.sqrt(1.0 + alpha * evals)) @ evecs.T
        X_mod = X @ W
    else:
        X_mod = X

    model = CoxnetSurvivalAnalysis(l1_ratio=0.1, alpha_min_ratio=0.01, max_iter=2000)
    model.fit(X_mod, y)
    return model


def benchmark(X, y, label, L=None, alpha=0.2):
    tracemalloc.start()
    start = time.perf_counter()
    fit_model(X, y, L=L, alpha=alpha)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024.0 * 1024.0), elapsed


def main():
    X, y = load_dataset()
    L_string = load_laplacian("Laplacian_STRING.csv")
    L_chara = load_laplacian("Laplacian_Chara.csv")

    # Use the actual common feature set. The real common-gene intersection is 5199, not 5200.
    expr_cols = [str(c).strip() for c in pd.read_csv(DATA_DIR / "TCGA-LUAD_expression.csv", index_col=0).columns]
    string_cols = [str(c).strip() for c in pd.read_csv(DATA_DIR / "Laplacian_STRING.csv", index_col=0).columns]
    chara_cols = [str(c).strip() for c in pd.read_csv(DATA_DIR / "Laplacian_Chara.csv", index_col=0).columns]
    overlap = sorted(set(expr_cols) & set(string_cols) & set(chara_cols))

    expr_df = pd.read_csv(DATA_DIR / "TCGA-LUAD_expression.csv", index_col=0)
    expr_df.columns = [str(c).strip() for c in expr_df.columns]
    expr_df = expr_df.loc[:, [c for c in overlap if c in expr_df.columns]]

    surv_df = pd.read_csv(DATA_DIR / "TCGA-LUAD_survival.csv", index_col=0)
    common_idx = expr_df.index.intersection(surv_df.index)
    expr_df = expr_df.loc[common_idx]
    surv_df = surv_df.loc[common_idx]

    y = np.array(
        list(zip(surv_df["Event"].astype(bool).to_numpy(), surv_df["Time"].astype(np.float64).to_numpy())),
        dtype=[("Status", "?"), ("Survival_in_days", "<f8")],
    )

    X = expr_df.to_numpy(dtype=np.float64)

    L_string = pd.read_csv(DATA_DIR / "Laplacian_STRING.csv", index_col=0)
    L_string.columns = [str(c).strip() for c in L_string.columns]
    L_string.index = [str(i).strip() for i in L_string.index]
    L_string = L_string.loc[[c for c in overlap if c in L_string.index], [c for c in overlap if c in L_string.columns]]
    L_string = L_string.to_numpy(dtype=np.float64)

    L_chara = pd.read_csv(DATA_DIR / "Laplacian_Chara.csv", index_col=0)
    L_chara.columns = [str(c).strip() for c in L_chara.columns]
    L_chara.index = [str(i).strip() for i in L_chara.index]
    L_chara = L_chara.loc[[c for c in overlap if c in L_chara.index], [c for c in overlap if c in L_chara.columns]]
    L_chara = L_chara.to_numpy(dtype=np.float64)

    string_mem, string_time = benchmark(X, y, "STRING", L=L_string, alpha=0.2)
    chara_mem, chara_time = benchmark(X, y, "Chara", L=L_chara, alpha=0.45)

    print("Computational Complexity Benchmark (5,200 Features)")
    print("------------------------------------------------------------------")
    print(f"{'Metric':<22} | {'STRING Baseline':<20} | {'Chara (MD) Pipeline':<20}")
    print("------------------------------------------------------------------")
    print(f"{'Peak Memory (MB)':<22} | {string_mem:>18.3f} | {chara_mem:>20.3f}")
    print(f"{'Fit Execution Time (s)':<22} | {string_time:>18.6f} | {chara_time:>20.6f}")
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()

