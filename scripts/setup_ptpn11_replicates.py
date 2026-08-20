#!/usr/bin/env python3
"""
Setup PTPN11 Replicates (rep1, rep2, rep3) Production TPR files
Generates production.tpr for rep1, rep2, rep3 from equilibrated structures.
"""

import os
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTPN11_DIR = os.path.join(BASE_DIR, "data", "md_runs", "PTPN11")
CONDA_BIN = "/home/sharon/env_md/bin"
GMX_BIN = os.path.join(CONDA_BIN, "gmx")

def setup_rep(rep_name: str, seed: int):
    rep_dir = os.path.join(PTPN11_DIR, rep_name)
    os.makedirs(rep_dir, exist_ok=True)

    # Copy topology & itp files
    for f in ["topol.top", "martini.itp", "molecule_0.itp", "production.mdp"]:
        src = os.path.join(PTPN11_DIR, f)
        dst = os.path.join(rep_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            subprocess.run(["cp", src, dst])

    # Check if production.tpr exists
    prod_tpr = os.path.join(rep_dir, "production.tpr")
    if os.path.exists(prod_tpr):
        print(f"[✓] {rep_name} production.tpr already exists!")
        return

    # Use em.gro if eq.gro is missing
    gro_file = os.path.join(rep_dir, "eq.gro")
    if not os.path.exists(gro_file):
        gro_file = os.path.join(PTPN11_DIR, "em.gro")

    top_file = os.path.join(rep_dir, "topol.top")
    mdp_file = os.path.join(rep_dir, "production.mdp")

    # Update ld-seed in production.mdp
    with open(mdp_file, "r") as f:
        content = f.read()
    if "gen-seed" in content or "ld-seed" in content:
        content = content.replace("ld-seed = 12345", f"ld-seed = {seed}")
    else:
        content += f"\nld-seed = {seed}\n"
    with open(mdp_file, "w") as f:
        f.write(content)

    print(f"=== Generating production.tpr for PTPN11 / {rep_name} ===")
    cmd = [GMX_BIN, "grompp", "-f", mdp_file, "-c", gro_file, "-p", top_file, "-o", prod_tpr, "-maxwarn", "5"]
    res = subprocess.run(cmd, cwd=rep_dir, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[✓] Successfully generated {prod_tpr}")
    else:
        print(f"[X] grompp failed: {res.stderr}")

def main():
    setup_rep("rep1", 1001)
    setup_rep("rep2", 1002)
    setup_rep("rep3", 1003)

if __name__ == "__main__":
    main()
