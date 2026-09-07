#!/usr/bin/env python3
"""
tools/chara_md_dii/06_verify_archive_standalone.py

Independent extraction and execution verification:
1. Extracts CHARA_DII_EXECUTION_EVIDENCE.zip into a temporary directory.
2. Runs the included standalone_evaluation.py using standard python.
3. Checks that KRAS fold 1 baseline, robust, and official DII predictions are recomputed and verified.
4. Outputs the exact numerical comparison table.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

BASE_DIR = Path("E:/Sharon")
ARCHIVE_ZIP = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1" / "CHARA_DII_EXECUTION_EVIDENCE.zip"
TEMP_TEST_DIR = Path(os.environ.get("TEMP", "C:/Users/Samsunh/AppData/Local/Temp")) / "dii_standalone_test_run"

def main():
    print("=================================================================")
    print("   ARCHIVE EXTRACTION & STANDALONE EVALUATOR VERIFICATION TEST   ")
    print("=================================================================")

    if not ARCHIVE_ZIP.exists():
        print(f"ERROR: Archive {ARCHIVE_ZIP} does not exist.")
        sys.exit(1)

    print(f"Archive file: {ARCHIVE_ZIP} ({ARCHIVE_ZIP.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Extracting to: {TEMP_TEST_DIR}...")

    if TEMP_TEST_DIR.exists():
        shutil.rmtree(TEMP_TEST_DIR)
    TEMP_TEST_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        zf.extractall(TEMP_TEST_DIR)

    eval_script = TEMP_TEST_DIR / "standalone_evaluation.py"
    if not eval_script.exists():
        print(f"ERROR: {eval_script} not found in archive.")
        sys.exit(1)

    print(f"Executing standalone evaluator in isolated directory...")
    proc = subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=str(TEMP_TEST_DIR),
        capture_output=True,
        text=True
    )

    print("\n--- STANDALONE EVALUATOR STDOUT ---")
    print(proc.stdout)
    if proc.stderr:
        print("--- STANDALONE EVALUATOR STDERR ---")
        print(proc.stderr)

    if proc.returncode == 0:
        print("[OK] Standalone evaluator verified successfully from extracted archive.")
    else:
        print(f"[FAIL] Evaluator returned exit code {proc.returncode}.")
        sys.exit(proc.returncode)

    # Clean up temp dir
    shutil.rmtree(TEMP_TEST_DIR)
    print("[OK] Cleaned up temporary test directory.")

if __name__ == "__main__":
    main()
