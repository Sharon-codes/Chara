#!/usr/bin/env python3
"""
tools/chara_closure/execute_closure.py
Master runner for CHARA Bounded Closure.
Executes all audit, diagnostic, and reporting modules,
packages CHARA_CLOSURE_CORE.zip, and tests standalone verification
in an isolated temporary directory.
"""

import os
import sys
import zipfile
import hashlib
import tempfile
import subprocess
import pandas as pd

BASE_DIR = r"E:\Sharon"
TOOLS_DIR = os.path.join(BASE_DIR, "tools", "chara_closure")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "chara_closure")

CORE_FILES = [
    "CHARA_CLOSURE_REPORT.html",
    "RUN_PARAMETER_STATUS.csv",
    "EFFECTIVE_PAIR_COMPARISON.csv",
    "DIAGNOSTICS_FIXED.csv",
    "SUBGROUP_MEMBERSHIP_COMPLETE.json",
    "SUBGROUP_CORRECTION_DIFF.csv",
    "DATASET_ACCESS_VERIFIED.csv",
    "CLOSEST_WORK_CORRECTED.csv",
    "CLAIM_CORRECTIONS_FINAL.csv",
    "ORIGINAL_INPUTS_MANIFEST.sha256",
    "EXECUTION_LOG.txt",
    "REPRODUCE_CLOSURE.md",
    "verify_chara_closure.py"
]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=================================================================")
    print("=== EXECUTING CHARA BOUNDED CLOSURE PIPELINE ===")
    print("=================================================================")

    # 1. GROMACS inspection
    cmd = [sys.executable, os.path.join(TOOLS_DIR, "01_gromacs_inspection.py")]
    print(f"[*] Running {cmd[1]}...")
    subprocess.run(cmd, check=True)

    # 2. Correct diagnostics
    cmd = [sys.executable, os.path.join(TOOLS_DIR, "02_correct_diagnostics.py")]
    print(f"[*] Running {cmd[1]}...")
    subprocess.run(cmd, check=True)

    # 3. Datasets and literature
    cmd = [sys.executable, os.path.join(TOOLS_DIR, "03_dataset_and_literature.py")]
    print(f"[*] Running {cmd[1]}...")
    subprocess.run(cmd, check=True)

    # 4. Programmatic HTML generation
    cmd = [sys.executable, os.path.join(TOOLS_DIR, "04_closure_report_generator.py")]
    print(f"[*] Running {cmd[1]}...")
    subprocess.run(cmd, check=True)

    # Copy verify script to reports dir
    v_src = os.path.join(TOOLS_DIR, "verify_chara_closure.py")
    v_dst = os.path.join(REPORTS_DIR, "verify_chara_closure.py")
    with open(v_src, "r", encoding="utf-8") as f_in, open(v_dst, "w", encoding="utf-8") as f_out:
        f_out.write(f_in.read())

    # Build manifest_checksums.sha256
    print("[*] Generating manifest_checksums.sha256...")
    manifest_lines = []
    for fname in CORE_FILES:
        fpath = os.path.join(REPORTS_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file missing for manifest: {fpath}")
        manifest_lines.append(f"{sha256_file(fpath)}  {fname}")

    manifest_p = os.path.join(REPORTS_DIR, "manifest_checksums.sha256")
    with open(manifest_p, "w", encoding="utf-8") as mf:
        mf.write("\n".join(manifest_lines) + "\n")
    print(f"  [OK] Saved {manifest_p}")

    # Package CHARA_CLOSURE_CORE.zip
    zip_path = os.path.join(REPORTS_DIR, "CHARA_CLOSURE_CORE.zip")
    print(f"[*] Packaging {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in CORE_FILES:
            zf.write(os.path.join(REPORTS_DIR, fname), fname)
        zf.write(manifest_p, "manifest_checksums.sha256")
        
        # Also include reproduction tool scripts in code/ directory of zip
        for tool_name in ["01_gromacs_inspection.py", "02_correct_diagnostics.py", "03_dataset_and_literature.py", "04_closure_report_generator.py", "verify_chara_closure.py"]:
            t_path = os.path.join(TOOLS_DIR, tool_name)
            if os.path.exists(t_path):
                zf.write(t_path, f"code/{tool_name}")

    zip_sz = os.path.getsize(zip_path)
    print(f"  [OK] Successfully created {zip_path} ({zip_sz:,} bytes, {zip_sz / (1024*1024):.2f} MiB)")

    # Run standalone verification test in an isolated temporary directory
    print("\n[*] Testing standalone verification in isolated temporary directory...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        v_cmd = [sys.executable, "verify_chara_closure.py"]
        res = subprocess.run(v_cmd, cwd=tmpdir, capture_output=True, text=True)
        print("  Isolated verification output:\n" + res.stdout.strip())
        if res.returncode != 0:
            print("  [ERROR] Isolated verification failed:\n" + res.stderr)
            raise RuntimeError("Verification in isolated directory failed!")
        print("  [OK] Standalone verification passed in isolated temporary directory!")

    print("\n=================================================================")
    print("=== CHARA BOUNDED CLOSURE COMPLETE AND VERIFIED ===")
    print("=================================================================")

if __name__ == "__main__":
    main()
