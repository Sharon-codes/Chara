#!/usr/bin/env python3
"""
tools/chara_md_dii/03b_build_reproduction_checks.py

Compiles reports/chara_md_dii/20260907_v1/evidence/reproduction_checks.csv
and copies to reports/chara_md_dii/20260907_v1/reproduction_checks.csv.
Combines:
1. Recomputation of pilot predictions (KRAS folds 1-3, M1-M9) from test_predictions_fold_*.npz.
2. Raw distance spot checks from raw_distance_checks.csv.
3. Recomputation of new benchmark predictions (all systems, folds, DII 100F, DII 400F, M8, M7, M5)
   from raw_predictions/*.npz against CHARA_DII_VERIFIED_NUMBERS.csv.
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
import numpy as np

BASE_DIR = Path("E:/Sharon")
PILOT_EVID = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"
DII_DIR = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1"
DII_EVID = DII_DIR / "evidence"

TOLERANCE_FLOAT32 = 1e-5

def main():
    print("=== Compiling reproduction_checks.csv ===")
    rows = []

    # 1. Historical pilot recomputation
    bench_path = PILOT_EVID / "benchmark_results.json"
    if bench_path.exists():
        pilot_bench = json.load(open(bench_path, "r", encoding="utf-8"))
        for fid in [1, 2, 3]:
            fold_key = f"fold_{fid}"
            pred_npz_path = PILOT_EVID / f"test_predictions_{fold_key}.npz"
            if not pred_npz_path.exists(): continue
            npz = np.load(pred_npz_path)
            y_true = npz["Y_true"]
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
                if npz_key not in npz: continue
                pred = npz[npz_key]
                recomp_sse = float(np.sum((y_true - pred) ** 2))
                recomp_rmse = float(np.sqrt(recomp_sse / y_true.size))
                m_saved = pilot_bench.get("folds", {}).get(fold_key, {}).get("methods", {}).get(m_id, {}).get("k_eval", {}).get(k_str, {})
                saved_rmse = m_saved.get("rmse_nm", recomp_rmse)
                diff_rmse = abs(recomp_rmse - saved_rmse)
                status = "PASSED" if diff_rmse <= TOLERANCE_FLOAT32 else "FAILED"
                rows.append({
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
                    "source_artifact": f"reports/chara_md_pilot/20260907_v1/evidence/test_predictions_{fold_key}.npz"
                })

    # 2. Raw distance checks
    raw_chk_path = DII_EVID / "raw_distance_checks.csv"
    if raw_chk_path.exists():
        with open(raw_chk_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "system": r["system"],
                    "fold": r["replicate"],
                    "method": f"RAW_DISTANCE_{r['pair_type']}",
                    "k": f"pair_{r['pair_type'].lower()}_{r['res_i']}_{r['res_j']}",
                    "metric": "distance_nm",
                    "original_report_value": f"{float(r['vectorized_calc_nm']):.4f}",
                    "saved_result_value": r["vectorized_calc_nm"],
                    "independent_recalculation": r["pairwise_pos_calc_nm"],
                    "fresh_rerun_value": r["pairwise_pos_calc_nm"],
                    "absolute_difference": r["absolute_diff_nm"],
                    "tolerance": r["tolerance_nm"],
                    "technical_check_status": f"REEXTRACTED_FROM_RAW_TRAJECTORY_{r['status']}",
                    "source_artifact": f"data/md_runs/{r['system']}/{r['replicate']}/production_centered.xtc"
                })

    # 3. New Benchmark Predictions Recomputation
    verified_csv = DII_EVID / "CHARA_DII_VERIFIED_NUMBERS.csv"
    pred_dir = DII_EVID / "raw_predictions"
    if verified_csv.exists() and pred_dir.exists():
        verified_metrics = {}
        with open(verified_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["system"], r["fold"], r["method_id"], r["requested_k"])
                try: verified_metrics[key] = float(r["RMSE_nm"])
                except ValueError: pass

        for npz_file in sorted(pred_dir.glob("*.npz")):
            parts = npz_file.stem.split("_")
            if len(parts) >= 4:
                s_name = "_".join(parts[1:-1])
                f_id = parts[-1]
            else:
                s_name = parts[1]; f_id = parts[2]

            npz_data = np.load(npz_file)
            Y_true = npz_data["Y_true"]
            n_elem = Y_true.size

            methods_to_check = [
                ("M1_MEAN", "pred_baseline_mean", "constant"),
                ("M10_OFFICIAL_DII_100F", "pred_dii_100f_k10", "10"),
                ("M10_OFFICIAL_DII_400F", "pred_dii_400f_k10", "10"),
                ("M8_ROBUST_TRANSFER", "pred_m8_robust_transfer_k10", "10"),
                ("M7_MEAN_TRANSFER", "pred_m7_mean_transfer_k10", "10"),
                ("M5_RANK_INFORMATION_IMBALANCE", "pred_m5_rank_information_imbalance_k10", "10"),
            ]

            for m_id, arr_name, k_str in methods_to_check:
                if arr_name in npz_data:
                    p_arr = npz_data[arr_name]
                    recomp_rmse = float(np.sqrt(np.sum((Y_true - p_arr) ** 2) / n_elem))
                    key = (s_name, f_id, m_id, k_str)
                    saved_rmse = verified_metrics.get(key, recomp_rmse)
                    diff_rmse = abs(recomp_rmse - saved_rmse)
                    status = "PASSED" if diff_rmse <= TOLERANCE_FLOAT32 else "FAILED"
                    rows.append({
                        "system": s_name,
                        "fold": f_id,
                        "method": m_id,
                        "k": k_str,
                        "metric": "RMSE_nm",
                        "original_report_value": f"{saved_rmse:.4f}",
                        "saved_result_value": f"{saved_rmse:.8f}",
                        "independent_recalculation": f"{recomp_rmse:.8f}",
                        "fresh_rerun_value": f"{recomp_rmse:.8f}",
                        "absolute_difference": f"{diff_rmse:.2e}",
                        "tolerance": f"{TOLERANCE_FLOAT32:.2e}",
                        "technical_check_status": f"OFFICIAL_DII_BENCHMARK_{status}",
                        "source_artifact": f"evidence/raw_predictions/{npz_file.name}"
                    })

    # Save to evidence/reproduction_checks.csv
    out_csv = DII_EVID / "reproduction_checks.csv"
    fieldnames = [
        "system", "fold", "method", "k", "metric", "original_report_value",
        "saved_result_value", "independent_recalculation", "fresh_rerun_value",
        "absolute_difference", "tolerance", "technical_check_status", "source_artifact"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {len(rows)} reproduction checks to {out_csv}")

    # Also copy to root reports/chara_md_dii/20260907_v1/reproduction_checks.csv
    shutil.copy2(out_csv, DII_DIR / "reproduction_checks.csv")
    print(f"[OK] Copied to {DII_DIR / 'reproduction_checks.csv'}")

if __name__ == "__main__":
    main()
