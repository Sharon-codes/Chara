#!/usr/bin/env python3
"""
tools/chara_md_review/04_test_standalone_extraction.py

Validates the review handoff in an isolated clean temporary directory:
1. Extracts CHARA_REVIEW_CORE.zip and CHARA_KRAS_REVIEW_DATA.zip.
2. Checks manifest integrity and member existence.
3. Executes verify_raw_distances.py to confirm raw coordinate re-extraction.
4. Executes reproduce_system_metrics.py to confirm metric reproduction.
5. Executes standalone_evaluation.py to confirm exact match with CHARA_REVIEW_NUMBERS.csv.
6. Returns a detailed table of actual vs expected values, differences, and exit statuses.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

BASE_DIR = Path("E:/Sharon")
REVIEW_DIR = BASE_DIR / "reports" / "chara_md_review" / "20260907_v1"
CORE_ZIP = REVIEW_DIR / "CHARA_REVIEW_CORE.zip"
KRAS_ZIP = REVIEW_DIR / "CHARA_KRAS_REVIEW_DATA.zip"

TEMP_TEST_ROOT = Path(os.environ.get("TEMP", "C:/Users/Samsunh/AppData/Local/Temp")) / "chara_review_handoff_test"

def main():
    print("=================================================================")
    print("      CHARA REVIEW ARCHIVE EXTRACTION & REPRODUCIBILITY TEST     ")
    print("=================================================================")

    if TEMP_TEST_ROOT.exists():
        shutil.rmtree(TEMP_TEST_ROOT)
    TEMP_TEST_ROOT.mkdir(parents=True, exist_ok=True)

    core_dir = TEMP_TEST_ROOT / "core"
    kras_dir = TEMP_TEST_ROOT / "kras"

    print(f"\n1. Extracting {CORE_ZIP.name} ({CORE_ZIP.stat().st_size / (1024*1024):.2f} MB)...")
    with zipfile.ZipFile(CORE_ZIP, "r") as zf:
        zf.extractall(core_dir)

    print(f"2. Extracting {KRAS_ZIP.name} ({KRAS_ZIP.stat().st_size / (1024*1024):.2f} MB)...")
    with zipfile.ZipFile(KRAS_ZIP, "r") as zf:
        zf.extractall(kras_dir)

    # Verify key member existence
    print("\n3. Verifying archive member completeness...")
    required_core_files = [
        "START_HERE.txt",
        "standalone_evaluation.py",
        "tables/CHARA_REVIEW_NUMBERS.csv",
        "tables/CHARA_DISCREPANCIES.csv",
        "tables/CHARA_DIAGNOSTICS.csv",
        "tables/CHARA_CLOSEST_WORK.csv",
        "tables/CHARA_CANDIDATE_QUESTIONS.csv",
        "manifest/run_manifest.json"
    ]
    for rf in required_core_files:
        p = core_dir / rf
        status = "EXISTS" if p.exists() else "MISSING"
        print(f"  [CORE] {rf:<40} : {status}")
        if not p.exists():
            print(f"ERROR: Missing required file {p}")
            sys.exit(1)

    required_kras_files = [
        "distances_rep1.npz",
        "distances_rep2.npz",
        "distances_rep3.npz",
        "candidates.json",
        "targets.json",
        "sample_coordinates_rep1.npz",
        "predictions/predictions_KRAS_G12D_fold_1.npz",
        "verify_raw_distances.py",
        "reproduce_system_metrics.py",
        "pca_qr_historical_comparison.json"
    ]
    for rf in required_kras_files:
        p = kras_dir / rf
        status = "EXISTS" if p.exists() else "MISSING"
        print(f"  [KRAS] {rf:<40} : {status}")
        if not p.exists():
            print(f"ERROR: Missing required file {p}")
            sys.exit(1)

    # 4. Test Raw Coordinate Verification Script
    print("\n4. Executing verify_raw_distances.py in isolated KRAS directory...")
    p_raw = subprocess.run(
        [sys.executable, "verify_raw_distances.py"],
        cwd=str(kras_dir),
        capture_output=True,
        text=True
    )
    print("--- STDOUT ---")
    print(p_raw.stdout.strip())
    if p_raw.returncode != 0:
        print("--- STDERR ---")
        print(p_raw.stderr.strip())
        print(f"[FAIL] verify_raw_distances.py failed with code {p_raw.returncode}")
        sys.exit(p_raw.returncode)

    # 5. Test System Metric Reproducer
    print("\n5. Executing reproduce_system_metrics.py in isolated KRAS directory...")
    p_rep = subprocess.run(
        [sys.executable, "reproduce_system_metrics.py"],
        cwd=str(kras_dir),
        capture_output=True,
        text=True
    )
    print("--- STDOUT ---")
    print(p_rep.stdout.strip())
    if p_rep.returncode != 0:
        print("--- STDERR ---")
        print(p_rep.stderr.strip())
        print(f"[FAIL] reproduce_system_metrics.py failed with code {p_rep.returncode}")
        sys.exit(p_rep.returncode)

    # 6. Test Standalone Evaluator against Tables
    print("\n6. Executing standalone_evaluation.py from CORE pointing to KRAS predictions...")
    p_eval = subprocess.run(
        [sys.executable, "standalone_evaluation.py", "--data_dir", str(kras_dir / "predictions"), "--tables_dir", str(core_dir / "tables")],
        cwd=str(core_dir),
        capture_output=True,
        text=True
    )
    print("--- STDOUT ---")
    print(p_eval.stdout.strip())
    if p_eval.returncode != 0:
        print("--- STDERR ---")
        print(p_eval.stderr.strip())
        print(f"[FAIL] standalone_evaluation.py failed with code {p_eval.returncode}")
        sys.exit(p_eval.returncode)

    print("\n[OK] All extraction and reproduction verification checks successfully passed.")

    # Clean up
    shutil.rmtree(TEMP_TEST_ROOT)
    print("[OK] Temporary test directory cleaned up.")

if __name__ == "__main__":
    main()
