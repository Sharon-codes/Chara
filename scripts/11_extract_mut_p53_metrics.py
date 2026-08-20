#!/usr/bin/env python3
"""
Extract Empirical GROMACS Trajectory Metrics for Mut_p53 (rep1, rep2, rep3)
Runs gmx rms, gmx gyrate, gmx sasa, gmx rmsf, gmx distance directly on production.xtc / production.tpr
Using MARTINI 3 coarse-grained Group 1 (Protein).
"""

import os
import subprocess
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT_P53_DIR = os.path.join(BASE_DIR, "data", "md_runs", "Mut_p53")
CONDA_BIN = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN, "gmx")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_gmx_command(cmd_list, stdin_str, cwd_dir):
    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN}:" + env.get("PATH", "")
    try:
        res = subprocess.run(cmd_list, input=stdin_str, text=True, capture_output=True, cwd=cwd_dir)
        return res.returncode == 0
    except Exception as e:
        logging.error(f"GROMACS command error in {cwd_dir}: {e}")
        return False


def extract_metrics_for_rep(rep_name: str):
    rep_dir = os.path.join(MUT_P53_DIR, rep_name)
    tpr_file = os.path.join(rep_dir, "production.tpr")
    xtc_file = os.path.join(rep_dir, "production.xtc")

    if not os.path.exists(tpr_file) or not os.path.exists(xtc_file):
        logging.warning(f"Files missing in {rep_dir}, skipping...")
        return

    logging.info(f"=== Extracting empirical metrics for Mut_p53 / {rep_name} ===")

    # 1. RMSD (Protein = group 1)
    run_gmx_command([GMX_BIN, "rms", "-s", "production.tpr", "-f", "production.xtc", "-o", "rmsd.xvg", "-tu", "ns"], "1\n1\n", rep_dir)

    # 2. Radius of Gyration (Protein = group 1)
    run_gmx_command([GMX_BIN, "gyrate", "-s", "production.tpr", "-f", "production.xtc", "-o", "gyrate.xvg"], "1\n", rep_dir)

    # 3. RMSF per residue (Protein = group 1)
    run_gmx_command([GMX_BIN, "rmsf", "-s", "production.tpr", "-f", "production.xtc", "-o", "rmsf.xvg", "-res"], "1\n", rep_dir)

    # 4. SASA (Protein = group 1)
    run_gmx_command([GMX_BIN, "sasa", "-s", "production.tpr", "-f", "production.xtc", "-o", "pocket_sasa.xvg"], "1\n", rep_dir)

    # 5. Distance (Inter-domain distance)
    run_gmx_command([GMX_BIN, "distance", "-s", "production.tpr", "-f", "production.xtc", "-o", "p53_dimer_dist.xvg", "-select", "com of group 1 plus com of group 1"], "1\n1\n", rep_dir)

    logging.info(f"[✓] Completed extraction for Mut_p53 / {rep_name}")


def main():
    for rep in ["rep1", "rep2", "rep3"]:
        extract_metrics_for_rep(rep)


if __name__ == "__main__":
    main()
