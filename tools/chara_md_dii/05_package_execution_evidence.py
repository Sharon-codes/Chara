#!/usr/bin/env python3
"""
tools/chara_md_dii/05_package_execution_evidence.py

Packages CHARA_DII_EXECUTION_EVIDENCE.zip containing:
1. All generated CSV and JSON evidence files
2. Candidate pair and target pair indices for each system
3. Raw predictions (.npz) containing Y_true and model predictions
4. Standalone evaluation script (standalone_evaluation.py)
5. Environment and package manifest with SHA256 checksums (run_manifest.json)
"""

import os
import sys
import json
import csv
import shutil
import hashlib
import zipfile
import platform
from pathlib import Path

BASE_DIR = Path("E:/Sharon")
FOLLOWUP_DIR = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1"
EVID_DIR = FOLLOWUP_DIR / "evidence"
ARCHIVE_ZIP = FOLLOWUP_DIR / "CHARA_DII_EXECUTION_EVIDENCE.zip"

STAGE_DIR = FOLLOWUP_DIR / "evidence_staging"
STAGE_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Packaging Execution Evidence Archive ===")

    # Step 1: Copy evidence CSVs and JSONs
    csv_files = [
        "CHARA_DII_VERIFIED_NUMBERS.csv", "paired_differences.csv",
        "dii_optimization_summary.csv", "dii_software_check.json",
        "dii_weights_and_history.json", "artifact_inventory.csv",
        "raw_distance_checks.csv", "pair_disjointness_checks.csv",
        "screening_pool_audit.csv", "reproduction_checks.csv"
    ]
    for fn in csv_files:
        src = EVID_DIR / fn
        if src.exists():
            shutil.copy2(src, STAGE_DIR / fn)

    # Step 2: Copy raw predictions directory
    src_preds = EVID_DIR / "raw_predictions"
    dst_preds = STAGE_DIR / "raw_predictions"
    if src_preds.exists():
        if dst_preds.exists(): shutil.rmtree(dst_preds)
        shutil.copytree(src_preds, dst_preds)

    # Step 3: Copy candidate and target pair indices
    cand_dir_src = BASE_DIR / "reports" / "chara_md_followup" / "20260907_v1" / "archive_staging" / "candidate_pair_indices"
    dst_cand = STAGE_DIR / "candidate_pair_indices"
    if cand_dir_src.exists():
        if dst_cand.exists(): shutil.rmtree(dst_cand)
        shutil.copytree(cand_dir_src, dst_cand)

    # Step 4: Write standalone evaluation script
    evaluator_code = '''#!/usr/bin/env python3
"""
standalone_evaluation.py
Self-contained independent evaluator to recompute metrics directly from raw prediction arrays.
Evaluates KRAS fold 1 baseline, robust, and official DII predictions.
"""

import sys
import csv
from pathlib import Path
import numpy as np

def verify_predictions():
    base_dir = Path(__file__).resolve().parent
    pred_dir = base_dir / "raw_predictions"
    if not pred_dir.exists():
        print("ERROR: raw_predictions directory not found.")
        sys.exit(1)

    npz_files = sorted(pred_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} prediction files in raw_predictions/")
    
    # Load verified numbers for cross-checking
    verified_metrics = {}
    csv_path = base_dir / "CHARA_DII_VERIFIED_NUMBERS.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["system"], r["fold"], r["method_id"], r["requested_k"])
                verified_metrics[key] = float(r["RMSE_nm"])

    mismatches = 0
    checks_passed = 0

    print("\\nRecomputing RMSE from raw prediction arrays:")
    print(f"{'System':<12} {'Fold':<8} {'Method':<25} {'k':<5} {'Recomputed RMSE':<18} {'CSV RMSE':<12} {'Diff':<12} {'Status'}")
    print("-" * 95)

    for fpath in npz_files:
        data = np.load(fpath)
        stem = fpath.stem
        if stem.startswith("predictions_"):
            stem = stem[len("predictions_"):]
        s_name, f_num = stem.rsplit("_fold_", 1)
        f_id = f"fold_{f_num}"

        Y_true = data["Y_true"]
        n_elem = Y_true.size

        # Check baseline
        p_base = data["pred_baseline_mean"]
        rmse_base = float(np.sqrt(np.sum((Y_true - p_base)**2) / n_elem))
        key_base = (s_name, f_id, "M1_MEAN", "constant")
        csv_base = verified_metrics.get(key_base, None)
        diff_base = abs(rmse_base - csv_base) if csv_base is not None else 0.0
        status_base = "MATCH" if (csv_base is not None and diff_base < 1e-5) else "MISMATCH"
        if status_base == "MATCH": checks_passed += 1
        else: mismatches += 1
        csv_base_str = f"{csv_base:.6f} nm" if csv_base is not None else "N/A"
        print(f"{s_name:<12} {f_id:<8} {'M1_MEAN':<25} {'const':<5} {rmse_base:.6f} nm        {csv_base_str:<12} {diff_base:.2e}     {status_base}")

        # Check key methods at k=10
        check_methods = [
            ("M10_OFFICIAL_DII_100F", "pred_dii_100f_k10", "10"),
            ("M10_OFFICIAL_DII_400F", "pred_dii_400f_k10", "10"),
            ("M8_ROBUST_TRANSFER", "pred_m8_robust_transfer_k10", "10"),
            ("M7_MEAN_TRANSFER", "pred_m7_mean_transfer_k10", "10"),
            ("M5_RANK_INFORMATION_IMBALANCE", "pred_m5_rank_information_imbalance_k10", "10"),
        ]

        for m_id, arr_name, k_str in check_methods:
            if arr_name in data:
                p_arr = data[arr_name]
                rmse_val = float(np.sqrt(np.sum((Y_true - p_arr)**2) / n_elem))
                key = (s_name, f_id, m_id, k_str)
                csv_val = verified_metrics.get(key, None)
                diff = abs(rmse_val - csv_val) if csv_val is not None else 0.0
                status = "MATCH" if (csv_val is not None and diff < 1e-5) else "MISMATCH"
                if status == "MATCH": checks_passed += 1
                else: mismatches += 1
                csv_val_str = f"{csv_val:.6f} nm" if csv_val is not None else "N/A"
                print(f"{s_name:<12} {f_id:<8} {m_id:<25} {k_str:<5} {rmse_val:.6f} nm        {csv_val_str:<12} {diff:.2e}     {status}")

    print("-" * 95)
    print(f"Summary: {checks_passed} checks passed, {mismatches} mismatches.")
    if mismatches == 0:
        print("[OK] All raw prediction arrays successfully verified against metrics table.")
        return 0
    else:
        print("[FAIL] Mismatches detected.")
        return 1

if __name__ == "__main__":
    sys.exit(verify_predictions())
'''
    with open(STAGE_DIR / "standalone_evaluation.py", "w", encoding="utf-8") as f:
        f.write(evaluator_code)

    # Step 5: Write run manifest
    manifest = {
        "benchmark": "Chara Official DADApy DII Benchmark & Multi-System Recomputation",
        "timestamp": "2026-09-07 20:52:00",
        "platform": platform.platform(),
        "python_version": sys.version,
        "files_sha256": {}
    }

    for root, _, files in os.walk(STAGE_DIR):
        for file in files:
            p = Path(root) / file
            rel_p = p.relative_to(STAGE_DIR).as_posix()
            manifest["files_sha256"][rel_p] = sha256_file(p)

    with open(STAGE_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Step 6: Create ZIP archive
    print(f"Creating ZIP archive {ARCHIVE_ZIP}...")
    with zipfile.ZipFile(ARCHIVE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(STAGE_DIR):
            for file in files:
                p = Path(root) / file
                rel_p = p.relative_to(STAGE_DIR).as_posix()
                zf.write(p, arcname=rel_p)

    sz_mb = os.path.getsize(ARCHIVE_ZIP) / (1024 * 1024)
    print(f"[OK] Successfully packaged: {ARCHIVE_ZIP} ({sz_mb:.2f} MB)")

if __name__ == "__main__":
    main()
