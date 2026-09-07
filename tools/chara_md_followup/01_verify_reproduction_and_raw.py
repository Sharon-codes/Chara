#!/usr/bin/env python3
"""
tools/chara_md_followup/01_verify_reproduction_and_raw.py

Verifies historical pilot artifacts, recomputes numerical metrics from saved predictions,
performs independent raw-distance re-extraction directly from XTC coordinates,
and freshly executes KRAS Fold 1 at k=10.
Outputs:
  reports/chara_md_followup/20260907_v1/evidence/reproduction_checks.csv
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader
from scipy import linalg
from scipy.spatial.distance import cdist

BASE_DIR = Path("E:/Sharon")
PILOT_EVID = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"
FOLLOWUP_EVID = BASE_DIR / "reports" / "chara_md_followup" / "20260907_v1" / "evidence"
FOLLOWUP_EVID.mkdir(parents=True, exist_ok=True)

TOLERANCE_EXACT = 1e-12
TOLERANCE_FLOAT32 = 1e-5
TOLERANCE_ROUNDED = 5e-4

def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "N/A"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_bb_indices(itp_path: Path):
    bb_idx = []
    idx = 0
    with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
        in_at = False
        for l in f:
            if "[ atoms ]" in l:
                in_at = True
                continue
            if in_at and l.startswith("["):
                break
            if in_at and l.strip() and not l.strip().startswith(";"):
                parts = l.split()
                if len(parts) >= 5 and parts[4] == "BB":
                    bb_idx.append(idx)
                idx += 1
    return np.array(bb_idx, dtype=int)

def main():
    print("[INFO] Running Stage 1: Artifact & Numerical Reproduction Verification...", flush=True)
    
    reproduction_rows = []

    # 1. Saved Prediction Recomputation
    with open(PILOT_EVID / "benchmark_results.json", "r", encoding="utf-8") as f:
        pilot_bench = json.load(f)

    for fid in [1, 2, 3]:
        fold_key = f"fold_{fid}"
        pred_npz_path = PILOT_EVID / f"test_predictions_{fold_key}.npz"
        npz = np.load(pred_npz_path)
        y_true = npz["Y_true"]
        y_mean = npz["m1_mean"]
        sse_mean = float(np.sum((y_true - y_mean) ** 2))
        rmse_mean = float(np.sqrt(sse_mean / y_true.size))

        methods_map = [
            ("m1_mean", "m1_mean", "constant"),
            ("m3_variance", "m3_variance_k10", "10"),
            ("m4_pca_qr", "m4_pca_qr_k10", "10"),
            ("m5_info_imbalance", "m5_info_imbalance_k10", "10"),
            ("m6_pooled_greedy", "m6_pooled_greedy_k10", "10"),
            ("m7_mean_transfer", "m7_mean_transfer_k10", "10"),
            ("m8_robust_transfer", "m8_robust_transfer_k10", "10"),
            ("m9_graph_assisted", "m9_graph_assisted_k10", "10"),
        ]

        for m_id, npz_key, k_str in methods_map:
            pred = npz[npz_key]
            recomp_sse = float(np.sum((y_true - pred) ** 2))
            recomp_rmse = float(np.sqrt(recomp_sse / y_true.size))
            recomp_skill = float(1.0 - (recomp_sse / sse_mean))

            # Saved result from benchmark_results.json
            m_saved = pilot_bench["folds"][fold_key]["methods"][m_id]["k_eval"][k_str]
            saved_rmse = m_saved["rmse_nm"]
            saved_skill = m_saved["skill"]

            diff_rmse = abs(recomp_rmse - saved_rmse)
            status = "PASSED" if diff_rmse <= TOLERANCE_FLOAT32 else "FAILED"

            reproduction_rows.append({
                "system": "KRAS_G12D",
                "fold": fold_key,
                "method": m_id,
                "k": 10 if k_str == "10" else "constant",
                "metric": "RMSE_nm",
                "original_report_value": f"{saved_rmse:.4f}",
                "saved_result_value": f"{saved_rmse:.8f}",
                "independent_recalculation": f"{recomp_rmse:.8f}",
                "fresh_rerun_value": "N/A",
                "absolute_difference": f"{diff_rmse:.2e}",
                "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
                "technical_check_status": f"RECOMPUTED_FROM_SAVED_PREDICTIONS_{status}",
                "source_artifact": f"evidence/test_predictions_{fold_key}.npz"
            })

    # 2. Raw Distance Re-extraction from XTC Coordinates
    with open(PILOT_EVID / "pair_definitions.json", "r", encoding="utf-8") as f:
        pdefs = json.load(f)

    bb_idx = get_bb_indices(BASE_DIR / "data" / "cg_topologies" / "KRAS_G12D" / "molecule_0.itp")
    sample_frames = [150, 300, 500, 700, 900]
    sample_candidates = [0, 10, 50, 100, 500]
    sample_targets = [0, 10, 50, 100, 500]

    for rep in ["rep1", "rep2", "rep3"]:
        xtc_path = BASE_DIR / "data" / "md_runs" / "KRAS_G12D" / rep / "production_centered.xtc"
        r = XTCReader(str(xtc_path))
        npz = np.load(PILOT_EVID / f"distances_{rep}.npz")
        c_cached = npz["candidate_dists"]
        t_cached = npz["target_dists"]

        for f_idx in sample_frames:
            ts = r[f_idx]
            pos_nm = ts.positions[bb_idx] / 10.0  # Convert A to nm
            
            # Check candidates
            for c_p in sample_candidates:
                p_info = pdefs["candidates"][c_p]
                i, j = p_info["i_idx"], p_info["j_idx"]
                raw_d = float(np.linalg.norm(pos_nm[i] - pos_nm[j]))
                cached_d = float(c_cached[f_idx, c_p])
                diff = abs(raw_d - cached_d)
                reproduction_rows.append({
                    "system": "KRAS_G12D",
                    "fold": rep,
                    "method": "RAW_DISTANCE_CANDIDATE",
                    "k": f"pair_{c_p}_frame_{f_idx}",
                    "metric": "distance_nm",
                    "original_report_value": f"{cached_d:.4f}",
                    "saved_result_value": f"{cached_d:.8f}",
                    "independent_recalculation": f"{raw_d:.8f}",
                    "fresh_rerun_value": f"{raw_d:.8f}",
                    "absolute_difference": f"{diff:.2e}",
                    "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
                    "technical_check_status": "REEXTRACTED_FROM_RAW_TRAJECTORY_PASSED" if diff <= TOLERANCE_FLOAT32 else "FAILED",
                    "source_artifact": f"data/md_runs/KRAS_G12D/{rep}/production_centered.xtc"
                })

            # Check targets
            for t_p in sample_targets:
                p_info = pdefs["targets"][t_p]
                i, j = p_info["i_idx"], p_info["j_idx"]
                raw_d = float(np.linalg.norm(pos_nm[i] - pos_nm[j]))
                cached_d = float(t_cached[f_idx, t_p])
                diff = abs(raw_d - cached_d)
                reproduction_rows.append({
                    "system": "KRAS_G12D",
                    "fold": rep,
                    "method": "RAW_DISTANCE_TARGET",
                    "k": f"pair_{t_p}_frame_{f_idx}",
                    "metric": "distance_nm",
                    "original_report_value": f"{cached_d:.4f}",
                    "saved_result_value": f"{cached_d:.8f}",
                    "independent_recalculation": f"{raw_d:.8f}",
                    "fresh_rerun_value": f"{raw_d:.8f}",
                    "absolute_difference": f"{diff:.2e}",
                    "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
                    "technical_check_status": "REEXTRACTED_FROM_RAW_TRAJECTORY_PASSED" if diff <= TOLERANCE_FLOAT32 else "FAILED",
                    "source_artifact": f"data/md_runs/KRAS_G12D/{rep}/production_centered.xtc"
                })

    # 3. Fresh Execution of KRAS Fold 1 at k=10
    print("[INFO] Running fresh execution of KRAS Fold 1 at k=10...", flush=True)
    # Load raw data from distances_rep*.npz
    npz1 = np.load(PILOT_EVID / "distances_rep1.npz")
    npz2 = np.load(PILOT_EVID / "distances_rep2.npz")
    npz3 = np.load(PILOT_EVID / "distances_rep3.npz")
    
    p_mask = npz1["primary_mask"]
    X1, Y1 = npz1["candidate_dists"][p_mask], npz1["target_dists"][p_mask]
    X2, Y2 = npz2["candidate_dists"][p_mask], npz2["target_dists"][p_mask]
    X3, Y3 = npz3["candidate_dists"][p_mask], npz3["target_dists"][p_mask]
    
    Y_pool = np.vstack([Y1, Y2])
    Y_train_mean = np.mean(Y_pool, axis=0, keepdims=True)
    
    # Fresh M1 Training Mean
    fresh_pred_mean = np.repeat(Y_train_mean, Y3.shape[0], axis=0)
    fresh_rmse_m1 = float(np.sqrt(np.mean((Y3 - fresh_pred_mean)**2)))
    orig_rmse_m1 = pilot_bench["folds"]["fold_1"]["methods"]["m1_mean"]["k_eval"]["constant"]["rmse_nm"]
    diff_m1 = abs(fresh_rmse_m1 - orig_rmse_m1)
    reproduction_rows.append({
        "system": "KRAS_G12D",
        "fold": "fold_1",
        "method": "m1_mean",
        "k": "constant",
        "metric": "RMSE_nm",
        "original_report_value": f"{orig_rmse_m1:.4f}",
        "saved_result_value": f"{orig_rmse_m1:.8f}",
        "independent_recalculation": f"{fresh_rmse_m1:.8f}",
        "fresh_rerun_value": f"{fresh_rmse_m1:.8f}",
        "absolute_difference": f"{diff_m1:.2e}",
        "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
        "technical_check_status": "FRESH_MODEL_RERUN_PASSED" if diff_m1 <= TOLERANCE_FLOAT32 else "FAILED",
        "source_artifact": "tools/chara_md_followup/01_verify_reproduction_and_raw.py"
    })

    # Fresh Ridge Decoder Function
    def fit_ridge(X_tr, Y_tr, alpha=1.0):
        xm = np.mean(X_tr, axis=0, keepdims=True)
        xs = np.std(X_tr, axis=0, keepdims=True)
        xs[xs < 1e-8] = 1.0
        ym = np.mean(Y_tr, axis=0, keepdims=True)
        Xn = (X_tr - xm) / xs
        Yc = Y_tr - ym
        k = X_tr.shape[1]
        XtX = Xn.T @ Xn
        reg = alpha * np.eye(k, dtype=X_tr.dtype)
        try:
            W = linalg.solve(XtX + reg, Xn.T @ Yc, assume_a='pos')
        except linalg.LinAlgError:
            W = linalg.pinv(XtX + reg) @ (Xn.T @ Yc)
        return xm, xs, ym, W

    def pred_ridge(X_te, xm, xs, ym, W):
        Xn = (X_te - xm) / xs
        return Xn @ W + ym

    # Fresh M8 Robust Selection on Fold 1
    # Correlation screening top 40
    y_guide1 = Y1[:, :50].mean(axis=1)
    y_guide2 = Y2[:, :50].mean(axis=1)
    corrs = []
    for c in range(1000):
        r1 = np.corrcoef(X1[:, c], y_guide1)[0, 1]
        r2 = np.corrcoef(X2[:, c], y_guide2)[0, 1]
        sc = 0.5 * (abs(r1) + abs(r2)) if (not np.isnan(r1) and not np.isnan(r2)) else 0.0
        corrs.append((sc, c))
    corrs.sort(reverse=True)
    screened_pool = [c for _, c in corrs[:40]]

    var_y = np.var(np.vstack([Y1, Y2]), axis=0)
    top_t_idx = np.argsort(-var_y)[:50]
    Y1_sub, Y2_sub = Y1[:, top_t_idx], Y2[:, top_t_idx]
    y1_var, y2_var = np.var(Y1_sub), np.var(Y2_sub)

    sel_robust = []
    for step in range(10):
        best_sc = float('inf')
        best_c = None
        for c in screened_pool:
            if c in sel_robust: continue
            tr = sel_robust + [c]
            xm1, xs1, ym1, W1 = fit_ridge(X1[:, tr], Y1_sub, alpha=1.0)
            e12 = np.mean((Y2_sub - pred_ridge(X2[:, tr], xm1, xs1, ym1, W1))**2) / y1_var
            xm2, xs2, ym2, W2 = fit_ridge(X2[:, tr], Y2_sub, alpha=1.0)
            e21 = np.mean((Y1_sub - pred_ridge(X1[:, tr], xm2, xs2, ym2, W2))**2) / y2_var
            sc = max(e12, e21)
            if sc < best_sc:
                best_sc = sc
                best_c = c
        sel_robust.append(best_c)

    # Fit ridge with alpha=1.0 (untuned test)
    X_pool = np.vstack([X1[:, sel_robust], X2[:, sel_robust]])
    xm_f1, xs_f1, ym_f1, W_f1 = fit_ridge(X_pool, Y_pool, alpha=1.0)
    fresh_pred_m8_a1 = pred_ridge(X3[:, sel_robust], xm_f1, xs_f1, ym_f1, W_f1)
    fresh_rmse_m8_a1 = float(np.sqrt(np.mean((Y3 - fresh_pred_m8_a1)**2)))
    orig_rmse_m8 = pilot_bench["folds"]["fold_1"]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["rmse_nm"]
    diff_m8_a1 = abs(fresh_rmse_m8_a1 - orig_rmse_m8)

    reproduction_rows.append({
        "system": "KRAS_G12D",
        "fold": "fold_1",
        "method": "m8_robust_transfer_alpha_1.0_untuned",
        "k": 10,
        "metric": "RMSE_nm",
        "original_report_value": f"{orig_rmse_m8:.4f}",
        "saved_result_value": f"{orig_rmse_m8:.8f}",
        "independent_recalculation": f"{fresh_rmse_m8_a1:.8f}",
        "fresh_rerun_value": f"{fresh_rmse_m8_a1:.8f}",
        "absolute_difference": f"{diff_m8_a1:.2e}",
        "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
        "technical_check_status": "UNTUNED_ALPHA_DEVIATION_RECORDED",
        "source_artifact": f"tools/chara_md_followup/01_verify_reproduction_and_raw.py (Features: {sel_robust})"
    })

    # Fit ridge with cross-validated tuned alpha (matching protocol)
    best_a = 1.0
    best_loss = float('inf')
    for a in [1e-4, 1e-2, 1.0, 10.0, 100.0]:
        xm1, xs1, ym1, W1 = fit_ridge(X1[:, sel_robust], Y1, alpha=a)
        loss12 = np.mean((Y2 - pred_ridge(X2[:, sel_robust], xm1, xs1, ym1, W1))**2)
        xm2, xs2, ym2, W2 = fit_ridge(X2[:, sel_robust], Y2, alpha=a)
        loss21 = np.mean((Y1 - pred_ridge(X1[:, sel_robust], xm2, xs2, ym2, W2))**2)
        avg_loss = 0.5 * (loss12 + loss21)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_a = a

    xm_fa, xs_fa, ym_fa, W_fa = fit_ridge(X_pool, Y_pool, alpha=best_a)
    fresh_pred_m8_tuned = pred_ridge(X3[:, sel_robust], xm_fa, xs_fa, ym_fa, W_fa)
    fresh_rmse_m8_tuned = float(np.sqrt(np.mean((Y3 - fresh_pred_m8_tuned)**2)))
    diff_m8_tuned = abs(fresh_rmse_m8_tuned - orig_rmse_m8)
    feature_match = (sel_robust == pilot_bench["folds"]["fold_1"]["methods"]["m8_robust_transfer"]["k_eval"]["10"]["selected_features"])

    reproduction_rows.append({
        "system": "KRAS_G12D",
        "fold": "fold_1",
        "method": "m8_robust_transfer_tuned_alpha",
        "k": 10,
        "metric": "RMSE_nm",
        "original_report_value": f"{orig_rmse_m8:.4f}",
        "saved_result_value": f"{orig_rmse_m8:.8f}",
        "independent_recalculation": f"{fresh_rmse_m8_tuned:.8f}",
        "fresh_rerun_value": f"{fresh_rmse_m8_tuned:.8f}",
        "absolute_difference": f"{diff_m8_tuned:.2e}",
        "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
        "technical_check_status": "FRESH_MODEL_RERUN_PASSED" if (diff_m8_tuned <= TOLERANCE_FLOAT32 and feature_match) else "FAILED",
        "source_artifact": f"tools/chara_md_followup/01_verify_reproduction_and_raw.py (Tuned alpha={best_a}, Features: {sel_robust})"
    })

    # Save to CSV
    csv_path = FOLLOWUP_EVID / "reproduction_checks.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reproduction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reproduction_rows)

    print(f"[OK] Completed {len(reproduction_rows)} reproduction and raw distance checks.")
    print(f"     Saved: {csv_path}")

if __name__ == "__main__":
    main()
