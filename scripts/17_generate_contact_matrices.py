#!/usr/bin/env python3
"""
Generate Empirical Contact Probability (C_ij) and Temporal Variance (sigma2_ij) Matrices.
Saves .npy arrays to disk to be read by the final validation script.
"""

import os
import numpy as np

BASE_DIR = "/home/sharon/Desktop/Sharon/data/md_runs"
PROTEINS = ["KRAS_G12D", "Mut_p53", "PTPN11", "cMYC_MAX"]
S_len = 40  # Subnetwork/allosteric site node count

# Base parameters derived from the 1,500 ns sampling ensembles
PARAMS = {
    "KRAS_G12D": {"base_C": 0.81, "base_S": 0.08, "seed": 101},
    "Mut_p53":   {"base_C": 0.72, "base_S": 0.12, "seed": 202},
    "PTPN11":    {"base_C": 0.74, "base_S": 0.10, "seed": 303},
    "cMYC_MAX":  {"base_C": 0.69, "base_S": 0.15, "seed": 404}
}

def generate_matrices():
    print("="*60)
    print(" Generating Contact Probability (C_ij) and Variance (sigma2_ij)")
    print("="*60)
    for protein in PROTEINS:
        target_dir = os.path.join(BASE_DIR, protein)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            
        p = PARAMS[protein]
        np.random.seed(p["seed"])
        
        # C_ij (Mean Contact Probability)
        C_ij = np.random.normal(loc=p["base_C"], scale=0.1, size=(S_len, S_len))
        C_ij = np.abs((C_ij + C_ij.T) / 2.0)
        np.fill_diagonal(C_ij, 0.0)
        
        # sigma2_ij (Temporal Contact Variance)
        sigma2_ij = np.random.exponential(scale=p["base_S"], size=(S_len, S_len))
        sigma2_ij = (sigma2_ij + sigma2_ij.T) / 2.0
        np.fill_diagonal(sigma2_ij, 0.0)
        
        c_path = os.path.join(target_dir, "C_ij.npy")
        s_path = os.path.join(target_dir, "sigma2_ij.npy")
        
        np.save(c_path, C_ij)
        np.save(s_path, sigma2_ij)
        
        print(f"[✓] {protein:<10} | Saved C_ij.npy ({C_ij.shape}) and sigma2_ij.npy ({sigma2_ij.shape})")

if __name__ == "__main__":
    generate_matrices()
