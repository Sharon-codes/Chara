#!/usr/bin/env python3
"""Compute and compare Dirichlet energies across STRING and Chara Laplacians."""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPR_PATH = ROOT / "TCGA-LUAD_expression.csv"
STRING_LAP_PATH = ROOT / "Laplacian_STRING.csv"
CHARA_LAP_PATH = ROOT / "Laplacian_Chara.csv"
FIGURE_PATH = ROOT / "Fig5_Dirichlet_Energy.pdf"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
})


def load_data():
    expr_df = pd.read_csv(EXPR_PATH, index_col=0)
    string_df = pd.read_csv(STRING_LAP_PATH, index_col=0)
    chara_df = pd.read_csv(CHARA_LAP_PATH, index_col=0)

    def clean_labels(labels):
        cleaned = []
        for x in labels:
            s = "" if x is None else str(x).strip()
            if s.lower() == "nan" or s == "":
                cleaned.append("")
            else:
                cleaned.append(s)
        return cleaned

    expr_df.columns = clean_labels(expr_df.columns)
    string_df.columns = clean_labels(string_df.columns)
    string_df.index = clean_labels(string_df.index)
    chara_df.columns = clean_labels(chara_df.columns)
    chara_df.index = clean_labels(chara_df.index)

    string_df = string_df.loc[[i for i in string_df.index if i != ""], [c for c in string_df.columns if c != ""]]
    chara_df = chara_df.loc[[i for i in chara_df.index if i != ""], [c for c in chara_df.columns if c != ""]]
    expr_df = expr_df.loc[:, [c for c in expr_df.columns if c != ""]]

    common_genes = (
        expr_df.columns.intersection(string_df.columns)
        .intersection(chara_df.columns)
    )
    if len(common_genes) == 0:
        raise ValueError("No common genes found between expression and Laplacian matrices.")

    expr_df = expr_df.loc[:, common_genes]
    string_df = string_df.loc[common_genes, common_genes]
    chara_df = chara_df.loc[common_genes, common_genes]

    X = expr_df.to_numpy(dtype=np.float64)
    L_string = string_df.to_numpy(dtype=np.float64)
    L_chara = chara_df.to_numpy(dtype=np.float64)

    L_string = (L_string + L_string.T) / 2.0
    L_chara = (L_chara + L_chara.T) / 2.0

    return X, L_string, L_chara


def compute_dirichlet_energy(X, L):
    """Compute E(x) = x^T L x for each patient using matrix multiplication."""
    XL = X @ L
    return np.sum(XL * X, axis=1)


def format_p_value(p_value):
    if p_value < 1e-3:
        return f"p = {p_value:.2e}"
    if p_value < 1e-2:
        return f"p = {p_value:.3f}"
    return f"p = {p_value:.4f}"


def main():
    X, L_string, L_chara = load_data()

    E_string = compute_dirichlet_energy(X, L_string)
    E_chara = compute_dirichlet_energy(X, L_chara)

    print(f"Patients: {X.shape[0]}")
    print(f"Genes: {X.shape[1]}")
    print(f"STRING median Dirichlet energy: {np.median(E_string):.6f}")
    print(f"Chara median Dirichlet energy: {np.median(E_chara):.6f}")

    stat, p_value = stats.mannwhitneyu(E_chara, E_string, alternative="two-sided")
    print(f"Mann-Whitney U statistic: {stat:.6f}")
    print(f"P-value: {p_value:.6e}")

    plot_df = pd.DataFrame(
        {
            "Dirichlet Energy": np.concatenate([E_string, E_chara]),
            "Laplacian": ["STRING"] * len(E_string) + ["Chara"] * len(E_chara),
        }
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.violinplot(
        data=plot_df,
        x="Laplacian",
        y="Dirichlet Energy",
        palette={"STRING": "#4C72B0", "Chara": "#DD8452"},
        inner="quartile",
        cut=0,
        linewidth=1.0,
        saturation=1,
        ax=ax,
    )
    sns.boxplot(
        data=plot_df,
        x="Laplacian",
        y="Dirichlet Energy",
        width=0.12,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 1.1},
        medianprops={"color": "#222222", "linewidth": 1.8},
        ax=ax,
        zorder=3,
    )

    y_max = float(max(E_string.max(), E_chara.max()))
    y_min = float(min(E_string.min(), E_chara.min()))
    y_span = max(y_max - y_min, 1e-8)
    bar_top = y_max + 0.06 * y_span
    bar_height = 0.03 * y_span

    ax.plot([0, 0, 1, 1], [bar_top, bar_top + bar_height, bar_top + bar_height, bar_top], color="black", lw=1.2)
    ax.text(0.5, bar_top + 1.35 * bar_height, format_p_value(p_value), ha="center", va="bottom", fontsize=11, weight="bold")

    ax.set_xlabel("")
    ax.set_ylabel(r"Dirichlet energy $E(x) = x^T L x$", fontsize=12)
    ax.set_title("Dirichlet energy distribution", fontsize=13, weight="bold")
    ax.set_xticklabels(["STRING", "Chara"])
    sns.despine(ax=ax, offset=5)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300)
    print(f"Saved plot to {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()