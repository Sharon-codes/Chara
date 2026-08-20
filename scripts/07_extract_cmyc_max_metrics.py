#!/usr/bin/env python3
"""
Automated GROMACS Extraction Script for cMYC_MAX Triplicate (N=3) Dataset
Calculates RMSD, RMSF, Gyrate, cMYC-MAX COM Distance, and Interfacial SASA for rep1, rep2, rep3.
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs", "cMYC_MAX")
CONDA_BIN_DIR = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN_DIR, "gmx") if os.path.exists(os.path.join(CONDA_BIN_DIR, "gmx")) else "gmx"
REPLICATES = ["rep1", "rep2", "rep3"]


def run_gromacs_extraction(rep_name: str):
    rep_dir = os.path.join(MD_RUNS_DIR, rep_name)
    print(f"=== Extracting GROMACS Metrics for cMYC_MAX / {rep_name} ===")

    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:" + env.get("PATH", "")

    # 1. RMSD
    p_rmsd = subprocess.Popen([GMX_BIN, "rms", "-s", "production.tpr", "-f", "production.xtc", "-o", "rmsd.xvg", "-tu", "ns"], cwd=rep_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_rmsd.communicate(input="1\n1\n")

    # 2. RMSF
    p_rmsf = subprocess.Popen([GMX_BIN, "rmsf", "-s", "production.tpr", "-f", "production.xtc", "-o", "rmsf.xvg", "-res"], cwd=rep_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_rmsf.communicate(input="1\n")

    # 3. Gyrate
    p_gyr = subprocess.Popen([GMX_BIN, "gyrate", "-s", "production.tpr", "-f", "production.xtc", "-o", "gyrate.xvg"], cwd=rep_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_gyr.communicate(input="1\n")

    # 4. Make Index (cMYC: res 1-89, MAX: res 90-160)
    p_ndx = subprocess.Popen([GMX_BIN, "make_ndx", "-f", "production.tpr", "-o", "interface.ndx"], cwd=rep_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_ndx.communicate(input="r 1-89\nr 90-160\n17 | 18\nq\n")

    # 5. Distance (COM between cMYC and MAX)
    p_dist = subprocess.Popen([GMX_BIN, "distance", "-s", "production.tpr", "-f", "production.xtc", "-n", "interface.ndx", "-oav", "cmyc_max_dist.xvg", "-select", 'com of group "r_1-89" plus com of group "r_90-160"', "-tu", "ns"], cwd=rep_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_dist.communicate()

    # 6. SASA (Interfacial exposure)
    p_sasa = subprocess.Popen([GMX_BIN, "sasa", "-s", "production.tpr", "-f", "production.xtc", "-n", "interface.ndx", "-o", "interface_sasa.xvg", "-tu", "ns"], cwd=rep_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p_sasa.communicate(input="19\n")

    print(f"[✓] Extracted all GROMACS files for {rep_name}")


def main():
    for r in REPLICATES:
        run_gromacs_extraction(r)


if __name__ == "__main__":
    main()
