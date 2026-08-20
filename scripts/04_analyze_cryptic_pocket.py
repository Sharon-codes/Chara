#!/usr/bin/env python3
"""
Cryptic Pocket Dynamics & Conformational Landscape Analysis Pipeline
Automates GROMACS extraction (gmx make_ndx, distance, sasa, covar, anaeig)
and generates a Q1 publication-ready 3-panel visual analysis (pocket_analysis.png).
"""

import os
import sys
import subprocess
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

CONDA_BIN_DIR = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN_DIR, "gmx") if os.path.exists(os.path.join(CONDA_BIN_DIR, "gmx")) else "gmx"


def run_cmd(cmd, cwd, input_str=None):
    """Helper to execute subprocess commands safely."""
    logging.info(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:" + env.get("PATH", "")
    res = subprocess.run(cmd, cwd=cwd, input=input_str, text=True, capture_output=True, env=env)
    if res.returncode != 0:
        logging.error(f"Command failed in {cwd}:\nStderr: {res.stderr}\nStdout: {res.stdout}")
        raise RuntimeError(f"GROMACS command failed: {' '.join(cmd)}")
    return res


def extract_gromacs_data(rep_dir: str):
    """Phase 1: Automate GROMACS index generation, distance, sasa, and PCA calculations."""
    logging.info("=== Phase 1: GROMACS Data Extraction ===")

    tpr = os.path.join(rep_dir, "production.tpr")
    xtc = os.path.join(rep_dir, "production.xtc")
    ndx = os.path.join(rep_dir, "pocket.ndx")

    if not os.path.exists(tpr) or not os.path.exists(xtc):
        raise FileNotFoundError(f"Missing required trajectory files in {rep_dir}")

    # 1. Index Generation (Switch I: 30-38, Switch II: 60-76, Combined: 30-38 | 60-76)
    if not os.path.exists(ndx):
        logging.info("Generating custom index groups for Switch I (30-38) and Switch II (60-76)...")
        ndx_input = "r 30-38\nr 60-76\n17 | 18\nq\n"
        run_cmd([GMX_BIN, "make_ndx", "-f", tpr, "-o", "pocket.ndx"], cwd=rep_dir, input_str=ndx_input)

    # 2. Distance Analysis (COM distance between Switch I and Switch II)
    dist_xvg = os.path.join(rep_dir, "sw1_sw2_dist.xvg")
    if not os.path.exists(dist_xvg):
        logging.info("Calculating Switch I - Switch II COM distance over time...")
        run_cmd([
            GMX_BIN, "distance",
            "-s", tpr,
            "-f", xtc,
            "-n", "pocket.ndx",
            "-oav", "sw1_sw2_dist.xvg",
            "-select", 'com of group "r_30-38" plus com of group "r_60-76"',
            "-tu", "ns"
        ], cwd=rep_dir)

    # 3. SASA Analysis (Combined Switch I + Switch II pocket region)
    sasa_xvg = os.path.join(rep_dir, "pocket_sasa.xvg")
    if not os.path.exists(sasa_xvg):
        logging.info("Calculating Solvent Accessible Surface Area (SASA) of pocket region...")
        run_cmd([
            GMX_BIN, "sasa",
            "-s", tpr,
            "-f", xtc,
            "-n", "pocket.ndx",
            "-o", "pocket_sasa.xvg",
            "-tu", "ns"
        ], cwd=rep_dir, input_str="19\n")

    # 4. PCA Analysis (Covariance matrix & 2D Projection PC1 vs PC2)
    pca_xvg = os.path.join(rep_dir, "pca_2d.xvg")
    trr_file = os.path.join(rep_dir, "eigenvec.trr")
    if not os.path.exists(pca_xvg):
        logging.info("Performing Principal Component Analysis (gmx covar & anaeig)...")
        if not os.path.exists(trr_file):
            run_cmd([
                GMX_BIN, "covar",
                "-s", tpr,
                "-f", xtc,
                "-o", "eigenval.xvg",
                "-v", "eigenvec.trr",
                "-tu", "ns"
            ], cwd=rep_dir, input_str="1\n1\n")

        run_cmd([
            GMX_BIN, "anaeig",
            "-s", tpr,
            "-f", xtc,
            "-v", "eigenvec.trr",
            "-2d", "pca_2d.xvg",
            "-first", "1",
            "-last", "2",
            "-tu", "ns"
        ], cwd=rep_dir, input_str="1\n1\n")

    logging.info("GROMACS data extraction complete.")


def parse_xvg(filepath: str) -> pd.DataFrame:
    """Parse GROMACS .xvg files into a clean pandas DataFrame, ignoring headers."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("@", "#")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                data.append([float(x) for x in parts[:2]])
    return pd.DataFrame(data, columns=["Col1", "Col2"])


def plot_pocket_analysis(rep_dir: str, output_image: str):
    """Phase 2: Create a Q1 publication-grade 3-panel figure (pocket_analysis.png)."""
    logging.info("=== Phase 2: Generating Q1 Publication-Quality Visualization ===")

    dist_df = parse_xvg(os.path.join(rep_dir, "sw1_sw2_dist.xvg"))
    dist_df.columns = ["Time_ns", "Distance_nm"]

    sasa_df = parse_xvg(os.path.join(rep_dir, "pocket_sasa.xvg"))
    sasa_df.columns = ["Time_ns", "SASA_nm2"]

    pca_df = parse_xvg(os.path.join(rep_dir, "pca_2d.xvg"))
    pca_df.columns = ["PC1_nm", "PC2_nm"]

    # Q1 Journal Styling Configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.0
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel A: Switch I - Switch II Distance vs. Time
    ax1 = axes[0]
    ax1.set_facecolor("white")
    ax1.plot(dist_df["Time_ns"], dist_df["Distance_nm"] * 10, color="#1f77b4", linewidth=1.5, label="SW1-SW2 Distance")
    mean_dist = dist_df["Distance_nm"].mean() * 10
    ax1.axhline(mean_dist, color="#d62728", linestyle="--", linewidth=1.5, label=f"Mean: {mean_dist:.2f} Å")
    ax1.set_title("A  Switch I - Switch II COM Distance", fontweight="bold", loc="left", pad=12)
    ax1.set_xlabel("Time (ns)", fontweight="bold")
    ax1.set_ylabel("Distance (Å)", fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="upper right")

    # Panel B: SASA of Pocket vs. Time
    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.plot(sasa_df["Time_ns"], sasa_df["SASA_nm2"], color="#d62728", linewidth=1.5, label="Pocket SASA")
    mean_sasa = sasa_df["SASA_nm2"].mean()
    ax2.axhline(mean_sasa, color="#1f77b4", linestyle="--", linewidth=1.5, label=f"Mean: {mean_sasa:.2f} nm²")
    ax2.set_title("B  Cryptic Pocket Solvent Exposure (SASA)", fontweight="bold", loc="left", pad=12)
    ax2.set_xlabel("Time (ns)", fontweight="bold")
    ax2.set_ylabel("SASA (nm²)", fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")

    # Panel C: 2D PCA Projection (PC1 vs PC2)
    ax3 = axes[2]
    ax3.set_facecolor("white")
    # Kernel Density Estimation Energy Landscape Contour
    sns.kdeplot(
        x=pca_df["PC1_nm"],
        y=pca_df["PC2_nm"],
        ax=ax3,
        cmap="Blues",
        fill=True,
        thresh=0.05,
        levels=10,
        alpha=0.6
    )
    sc = ax3.scatter(
        pca_df["PC1_nm"],
        pca_df["PC2_nm"],
        c=np.linspace(0, 500, len(pca_df)),
        cmap="viridis",
        s=12,
        alpha=0.8,
        edgecolor="none"
    )
    cbar = fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Time (ns)", fontweight="bold")
    ax3.set_title("C  Conformational Landscape (PC1 vs PC2)", fontweight="bold", loc="left", pad=12)
    ax3.set_xlabel("PC1 Eigenvector (nm)", fontweight="bold")
    ax3.set_ylabel("PC2 Eigenvector (nm)", fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Publication-ready 3-panel figure saved to: {output_image}")


def main():
    rep_dir = os.path.expanduser("~/Desktop/Sharon/data/md_runs/KRAS_G12D/rep1")
    output_image = os.path.join(rep_dir, "pocket_analysis.png")

    logging.info(f"Target trajectory folder: {rep_dir}")

    # Phase 1: Extract Data
    extract_gromacs_data(rep_dir)

    # Phase 2: Plot Data
    plot_pocket_analysis(rep_dir, output_image)

    # Output Biophysical Summary
    dist_df = parse_xvg(os.path.join(rep_dir, "sw1_sw2_dist.xvg"))
    sasa_df = parse_xvg(os.path.join(rep_dir, "pocket_sasa.xvg"))

    mean_d = dist_df["Col2"].mean() * 10
    max_d = dist_df["Col2"].max() * 10
    mean_sasa = sasa_df["Col2"].mean()
    max_sasa = sasa_df["Col2"].max()

    print("\n" + "=" * 65)
    print("      CRYPTIC POCKET DYNAMICS ANALYSIS SUMMARY (KRAS_G12D)")
    print("=" * 65)
    print(f"1. Switch I - Switch II COM Distance:")
    print(f"   - Mean Pocket Distance: {mean_d:.2f} Å ({mean_d/10:.4f} nm)")
    print(f"   - Maximum Opening Distance: {max_d:.2f} Å ({max_d/10:.4f} nm)")
    print(f"2. Cryptic Pocket SASA Exposure:")
    print(f"   - Mean Pocket SASA: {mean_sasa:.2f} nm²")
    print(f"   - Maximum Hydrophobic Exposure: {max_sasa:.2f} nm²")
    print(f"3. Publication Figure Generated:")
    print(f"   - {output_image} (300 DPI, 3-panel Q1 format)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
