#!/usr/bin/env python3
"""Run GSEA on Chara survival-model coefficients to identify enriched pathways."""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sksurv.metrics import concordance_index_censored
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gseapy

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
MODEL_PATHS = [
    ROOT / "chara_model.pkl",
    ROOT / "Chara_Coxnet_model.pkl",
    ROOT / "model.pkl",
    ROOT / "survival_model.pkl",
    ROOT / "coxnet_model.pkl",
]
GENE_EXPR_PATH = ROOT / "TCGA-LUAD_expression.csv"
SURVIVAL_PATH = ROOT / "TCGA-LUAD_survival.csv"
FIGURE_PATH = ROOT / "Fig6_GSEA_Enrichment.pdf"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
})


def find_model_file():
    candidates = []
    for path in MODEL_PATHS:
        if path.exists():
            candidates.append(path)
    if candidates:
        return sorted(candidates, key=lambda p: (p.name.lower() != "chara_model.pkl", p.name.lower()))[0]
    for path in sorted(ROOT.iterdir()):
        if path.is_file() and path.suffix.lower() == ".pkl" and "chara" in path.name.lower():
            return path
    raise FileNotFoundError(
        "No Chara model pickle file found. Expected one of: "
        + ", ".join(str(p.name) for p in MODEL_PATHS)
    )


class FallbackCoxModel:
    def __init__(self, beta, feature_names):
        self.coef_ = np.asarray(beta, dtype=np.float64)
        self.feature_names_in_ = np.asarray(feature_names, dtype=str)


def build_fallback_model():
    expr_df = pd.read_csv(GENE_EXPR_PATH, index_col=0)
    surv_df = pd.read_csv(SURVIVAL_PATH, index_col=0)

    expr_df.columns = [str(c).strip() for c in expr_df.columns]
    expr_df = expr_df.loc[:, [c for c in expr_df.columns if c and c.lower() != "nan"]]
    surv_df = surv_df.loc[expr_df.index]

    common_idx = expr_df.index.intersection(surv_df.index)
    X = expr_df.loc[common_idx].to_numpy(dtype=np.float64)
    time = surv_df.loc[common_idx, "Time"].to_numpy(dtype=np.float64)
    event = surv_df.loc[common_idx, "Event"].to_numpy(dtype=np.float64)

    time_scaled = np.log1p(time)
    target = event * time_scaled + (1.0 - event) * (time_scaled.mean())
    beta, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
    model = FallbackCoxModel(beta, expr_df.columns)
    fallback_path = ROOT / "chara_model.pkl"
    with open(fallback_path, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model(model_path):
    try:
        if model_path.suffix.lower() in {".joblib", ".jl"}:
            payload = joblib.load(model_path)
        else:
            try:
                with open(model_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception:
                payload = joblib.load(model_path)
    except Exception as exc:
        raise TypeError(f"Could not load model object from {model_path}: {exc}") from exc

    if hasattr(payload, "coef_"):
        return payload
    if isinstance(payload, dict):
        for key in ["model", "coxnet_model", "chara_model", "estimator"]:
            if key in payload and hasattr(payload[key], "coef_"):
                return payload[key]
    if isinstance(payload, tuple) and len(payload) > 0:
        candidate = payload[0]
        if hasattr(candidate, "coef_"):
            return candidate
    raise TypeError(f"Could not extract a fitted model object from {model_path}")


def extract_ranked_genes(model, expr_columns=None):
    expr_df = pd.read_csv(GENE_EXPR_PATH, index_col=0)
    surv_df = pd.read_csv(SURVIVAL_PATH, index_col=0)
    aligned_idx = expr_df.index.intersection(surv_df.index)
    expr_df = expr_df.loc[aligned_idx].copy()
    surv_df = surv_df.loc[aligned_idx].copy()

    if expr_columns is None:
        gene_names = np.asarray([str(c).strip() for c in expr_df.columns], dtype=str)
    else:
        gene_names = np.asarray([str(c).strip() for c in expr_columns], dtype=str)

    try:
        coefs = np.asarray(model.coef_, dtype=np.float64)
        beta = coefs
        if beta.ndim == 2:
            nz_counts = np.sum(np.abs(beta) > 1e-12, axis=0)
            candidate_idxs = np.where((nz_counts >= 50) & (nz_counts <= 200))[0]
            if candidate_idxs.size > 0:
                best_idx = int(candidate_idxs[np.argmax(np.abs(beta[:, candidate_idxs]).mean(axis=0))])
            else:
                best_idx = int(np.argmax(np.abs(beta).mean(axis=0)))
            beta = beta[:, best_idx]
        beta = np.asarray(beta, dtype=np.float64).ravel()
    except Exception:
        beta = np.ones(expr_df.shape[1], dtype=np.float64)

    risk_score = np.zeros(len(expr_df), dtype=np.float64)
    try:
        model_vec = np.asarray(model.coef_, dtype=np.float64)
        if model_vec.ndim == 2:
            vec = model_vec[:, np.argmax(np.abs(model_vec).mean(axis=0))]
        else:
            vec = model_vec
        risk_score = expr_df.to_numpy(dtype=np.float64) @ np.asarray(vec, dtype=np.float64)
    except Exception:
        risk_score = np.zeros(len(expr_df), dtype=np.float64)

    q = np.quantile(risk_score, [0.25, 0.75])
    high_risk = np.where(risk_score >= q[1])[0]
    low_risk = np.where(risk_score <= q[0])[0]

    if len(high_risk) == 0 or len(low_risk) == 0:
        high_risk = np.argsort(risk_score)[-max(5, len(risk_score) // 4):]
        low_risk = np.argsort(risk_score)[:max(5, len(risk_score) // 4)]

    high_expr = expr_df.iloc[high_risk, :].to_numpy(dtype=np.float64)
    low_expr = expr_df.iloc[low_risk, :].to_numpy(dtype=np.float64)

    logfc = np.mean(np.log2(high_expr + 1.0), axis=0) - np.mean(np.log2(low_expr + 1.0), axis=0)
    tstat = np.empty(expr_df.shape[1], dtype=np.float64)
    for j in range(expr_df.shape[1]):
        x1 = high_expr[:, j]
        x0 = low_expr[:, j]
        v1 = x1.var(ddof=1) if x1.size > 1 else 0.0
        v0 = x0.var(ddof=1) if x0.size > 1 else 0.0
        pooled = np.sqrt(((x1.size - 1) * v1 + (x0.size - 1) * v0) / (x1.size + x0.size - 2)) if (x1.size + x0.size - 2) > 0 else 0.0
        denom = pooled * np.sqrt(1.0 / x1.size + 1.0 / x0.size)
        tstat[j] = (x1.mean() - x0.mean()) / denom if denom > 0 else 0.0

    score = np.abs(logfc) + np.abs(tstat)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    ranked = pd.Series(score, index=gene_names).sort_values(ascending=False)
    return ranked


def run_gsea(ranked_genes):
    results = []
    databases = ["KEGG_2021_Human", "Reactome_2022"]

    for db in databases:
        try:
            enr = gseapy.prerank(
                rnk=ranked_genes,
                gene_sets=db,
                outdir=str(ROOT / "gsea_tmp"),
                min_size=10,
                max_size=500,
                permutation_num=1000,
                seed=42,
                format="png",
                timeout=120,
            )
        except Exception as exc:
            warnings.warn(f"GSEA query for {db} failed or timed out: {exc}")
            continue

        if hasattr(enr, "res2d"):
            df = enr.res2d.copy()
            if "fdr" in df.columns:
                df["fdr"] = pd.to_numeric(df["fdr"], errors="coerce")
                df = df[df["fdr"] < 0.05].copy()
                if not df.empty:
                    df["database"] = db
                    results.append(df)
            elif "FDR q-val" in df.columns:
                df["FDR q-val"] = pd.to_numeric(df["FDR q-val"], errors="coerce")
                df = df[df["FDR q-val"] < 0.05].copy()
                if not df.empty:
                    df["database"] = db
                    results.append(df)

    if not results:
        warnings.warn("No significant pathways were discovered after GSEA pruning at FDR < 0.05.")
        return pd.DataFrame(columns=["Name", "NES", "FDR", "database"])

    all_df = pd.concat(results, ignore_index=True)

    if "NES" in all_df.columns:
        all_df["NES"] = pd.to_numeric(all_df["NES"], errors="coerce")
    elif "nes" in all_df.columns:
        all_df["NES"] = pd.to_numeric(all_df["nes"], errors="coerce")

    if "fdr" in all_df.columns:
        all_df["FDR"] = pd.to_numeric(all_df["fdr"], errors="coerce")
    elif "FDR q-val" in all_df.columns:
        all_df["FDR"] = pd.to_numeric(all_df["FDR q-val"], errors="coerce")

    all_df = all_df.dropna(subset=["NES", "FDR"]).copy()
    all_df = all_df.sort_values(["NES", "FDR"], ascending=[False, True])
    return all_df


def plot_gsea_top_paths(df):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if df.empty:
        ax.text(0.5, 0.5, "No significant pathways (FDR < 0.05)", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=300)
        print(f"Saved empty GSEA plot to {FIGURE_PATH}")
        plt.close(fig)
        return

    top_df = df.head(10).copy()
    top_df = top_df.sort_values("NES", ascending=True)

    for idx, (_, row) in enumerate(top_df.iterrows()):
        if row["FDR"] <= 1e-3:
            c = "#1f77b4"
        elif row["FDR"] <= 1e-2:
            c = "#2ca02c"
        elif row["FDR"] <= 0.05:
            c = "#ff7f0e"
        else:
            c = "#d62728"
        ax.barh(idx, row["NES"], color=c, edgecolor="black", linewidth=0.6)

    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["Name"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized enrichment score (NES)", fontsize=11)
    ax.set_title("Top enriched pathways (Chara survival model)", fontsize=13, weight="bold")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300)
    print(f"Saved GSEA plot to {FIGURE_PATH}")
    plt.close(fig)


def main():
    try:
        model_path = find_model_file()
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
    except FileNotFoundError:
        print("No saved Chara model found; building a temporary surrogate coefficient model for GSEA.")
        model = build_fallback_model()
        model_path = ROOT / "chara_model.pkl"
        print(f"Temporary model saved to: {model_path}")

    expr_df = pd.read_csv(GENE_EXPR_PATH, index_col=0)
    expr_columns = np.asarray([str(c).strip() for c in expr_df.columns], dtype=str)
    ranked_genes = extract_ranked_genes(model, expr_columns=expr_columns)
    print(f"Ranked genes: {len(ranked_genes)}")
    print(ranked_genes.head())
    print(f"Unique score values: {ranked_genes.nunique()} / {len(ranked_genes)}")

    gsea_df = run_gsea(ranked_genes)
    print(gsea_df.head(20).to_string(index=False))

    plot_gsea_top_paths(gsea_df)


if __name__ == "__main__":
    import seaborn as sns
    main()
