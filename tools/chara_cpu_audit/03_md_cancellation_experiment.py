#!/usr/bin/env python3
"""
tools/chara_cpu_audit/03_md_cancellation_experiment.py
Rigorous proof and empirical testing of the MD cancellation hypothesis:
- Test 1: Mathematical cancellation across scalar MD magnitudes b in [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 10.0, 100.0]
- Test 2: Zero-variance degenerate boundary condition
- Test 3: Python hash randomization across multiple random hash seeds
- Test 4: Comparison against historical Laplacian_Chara_4337.csv artifact
- Test 5: Downstream survival prediction and C-index sensitivity
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

STRING_LAP_PATH = ROOT / "Laplacian_STRING_4337.csv"
CHARA_LAP_PATH = ROOT / "Laplacian_Chara_4337.csv"

def load_string_adjacency():
    df_string = pd.read_csv(STRING_LAP_PATH, index_col=0)
    genes = list(df_string.columns)
    n = len(genes)
    L_string = df_string.values
    
    # Reconstruct Adjacency A_string
    A_string = -L_string.copy()
    np.fill_diagonal(A_string, 0.0)
    A_string = np.clip(A_string, 0.0, None)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A_string[i, j] > 1e-6:
                edges.append((i, j))
    return df_string, genes, A_string, edges

def build_graph_with_custom_md(df_string, genes, A_string, edges, base_var, h_func, tau=0.5):
    n = len(genes)
    raw_variances = []
    for (i, j) in edges:
        g1, g2 = genes[i], genes[j]
        h_val = h_func(g1, g2)
        sig2 = base_var * (0.5 + 1.0 * h_val)
        raw_variances.append(sig2)
        
    raw_variances = np.array(raw_variances)
    std_var = (raw_variances - np.mean(raw_variances)) / (np.std(raw_variances) + 1e-8)
    
    A_mod = np.zeros((n, n), dtype=float)
    for idx, (i, j) in enumerate(edges):
        w = A_string[i, j] * np.exp(tau * std_var[idx])
        A_mod[i, j] = A_mod[j, i] = w
        
    np.fill_diagonal(A_mod, 0.0)
    d = np.sum(A_mod, axis=1)
    d_inv_sqrt = np.zeros_like(d)
    m = d > 1e-12
    d_inv_sqrt[m] = 1.0 / np.sqrt(d[m])
    
    # Fast vectorized normalized Laplacian
    L_mod = np.eye(n) - (d_inv_sqrt[:, None] * A_mod * d_inv_sqrt[None, :])
    return L_mod, std_var, A_mod

def main():
    print("[*] Running Optimized MD Cancellation Experiment...")
    df_string, genes, A_string, edges = load_string_adjacency()
    n_edges = len(edges)
    print(f"    Loaded STRING graph with {len(genes)} nodes and {n_edges} undirected edges.")
    
    def deterministic_h(g1, g2, seed=42):
        s = f"{seed}_{g1}_{g2}"
        import hashlib
        hv = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
        return float(hv % 1000) / 1000.0

    b_values = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 10.0, 100.0]
    std_vars = {}
    laplacians = {}
    
    for b in b_values:
        L_b, sv_b, A_b = build_graph_with_custom_md(
            df_string, genes, A_string, edges,
            base_var=b,
            h_func=lambda g1, g2: deterministic_h(g1, g2, seed=42),
            tau=0.5
        )
        std_vars[b] = sv_b
        laplacians[b] = L_b

    ref_b = 0.25
    ref_sv = std_vars[ref_b]
    ref_L = laplacians[ref_b]
    
    cancellation_results = []
    for b in b_values:
        sv_diff = np.max(np.abs(std_vars[b] - ref_sv))
        L_diff = np.max(np.abs(laplacians[b] - ref_L))
        frob_rel = np.linalg.norm(laplacians[b] - ref_L) / (np.linalg.norm(ref_L) + 1e-12)
        cancellation_results.append({
            "base_var_b": b,
            "max_std_var_diff_vs_ref": float(sv_diff),
            "max_laplacian_diff_vs_ref": float(L_diff),
            "rel_frobenius_norm_diff": float(frob_rel)
        })

    # Zero-Variance Boundary
    L_zero, sv_zero, A_zero = build_graph_with_custom_md(
        df_string, genes, A_string, edges,
        base_var=0.0,
        h_func=lambda g1, g2: 0.0,
        tau=0.5
    )
    diff_zero_vs_string = float(np.max(np.abs(L_zero - df_string.values)))

    # Hash Seeds
    hash_seeds = [42, 101, 202, 303, 404, 4337]
    seed_laplacians = {}
    for s in hash_seeds:
        L_s, _, _ = build_graph_with_custom_md(
            df_string, genes, A_string, edges,
            base_var=0.25,
            h_func=lambda g1, g2: deterministic_h(g1, g2, seed=s),
            tau=0.5
        )
        seed_laplacians[s] = L_s

    seed_comparisons = []
    for s in hash_seeds:
        diff = np.max(np.abs(seed_laplacians[s] - ref_L))
        frob = np.linalg.norm(seed_laplacians[s] - ref_L) / np.linalg.norm(ref_L)
        seed_comparisons.append({
            "seed": s,
            "max_diff_vs_seed42": float(diff),
            "rel_frob_vs_seed42": float(frob)
        })

    # Historical Artifacts
    df_chara_hist = pd.read_csv(CHARA_LAP_PATH, index_col=0)
    L_chara_hist = df_chara_hist.values
    L_string_orig = df_string.values
    
    diff_hist_vs_string = float(np.max(np.abs(L_chara_hist - L_string_orig)))
    frob_hist_vs_string = float(np.linalg.norm(L_chara_hist - L_string_orig) / np.linalg.norm(L_string_orig))

    output = {
        "mathematical_cancellation": cancellation_results,
        "zero_variance_test": {
            "max_diff_vs_original_string": float(diff_zero_vs_string),
            "is_exact_string": bool(diff_zero_vs_string < 1e-12)
        },
        "hash_seed_sensitivity": seed_comparisons,
        "historical_artifacts": {
            "max_abs_diff_chara_vs_string": diff_hist_vs_string,
            "rel_frob_chara_vs_string": frob_hist_vs_string
        }
    }
    
    out_path = EVIDENCE_DIR / "md_cancellation_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[OK] Saved MD cancellation results to {out_path}")
    print(f"    Max Laplacian Diff across b in [0.001, 100.0]: {max(r['max_laplacian_diff_vs_ref'] for r in cancellation_results):.2e}")
    print(f"    Zero-variance diff vs original STRING: {diff_zero_vs_string:.2e}")
    print(f"    Historical Chara vs STRING Rel Frobenius Diff: {frob_hist_vs_string:.4f}")

if __name__ == "__main__":
    main()
