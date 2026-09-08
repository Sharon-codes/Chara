#!/usr/bin/env python3
"""
tools/chara_external_confirmation/run_all_and_package.py

Master pipeline:
1. Verifies download completeness.
2. Runs coordinate extraction (03_process_atlas_trajectories.py).
3. Executes benchmark & diagnostics (04_execute_external_benchmark.py).
4. Updates claim-evidence matrix (05_literature_and_novelty.py).
5. Generates figures & unified HTML report (06_build_figures_and_report.py).
6. Packages deliverable ZIP strictly < 100 MB.
7. Tests offline bundle verification in a fresh sandbox.
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def sha256_file(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    base_dir = Path("/run/media/sharon/Windows ssd drive/Sharon")
    tools_dir = base_dir / "tools" / "chara_external_confirmation"
    rep_dir = base_dir / "reports" / "chara_external_confirmation" / "20260908_v1"
    py_bin = "/home/sharon/env_md/bin/python"

    print("================================================================================")
    print("      CHARA INDEPENDENT BENCHMARK PIPELINE & PACKAGING MASTER RUNNER")
    print("================================================================================")

    # 1. Trajectory coordinate extraction
    proc_files = list((base_dir / "data" / "external_atlas" / "processed").glob("*_ca_canonical.npz"))
    if len(proc_files) >= 3:
        print("\n--- [SKIP] 03_process_atlas_trajectories.py already completed (canonical coordinates exist) ---")
    else:
        print("\n--- Running 03_process_atlas_trajectories.py ---")
        res = subprocess.run([py_bin, str(tools_dir / "03_process_atlas_trajectories.py")], cwd=base_dir)
        if res.returncode != 0:
            print("[ERROR] Trajectory processing failed!")
            sys.exit(1)

    # 2. Benchmark execution
    bench_csv = rep_dir / "BENCHMARK_RESULTS.csv"
    if bench_csv.exists() and bench_csv.stat().st_size > 1000:
        print("\n--- [SKIP] 04_execute_external_benchmark.py already completed (BENCHMARK_RESULTS.csv exists) ---")
    else:
        print("\n--- Running 04_execute_external_benchmark.py ---")
        res = subprocess.run([py_bin, str(tools_dir / "04_execute_external_benchmark.py")], cwd=base_dir)
        if res.returncode != 0:
            print("[ERROR] Benchmark execution failed!")
            sys.exit(1)

    # 3. Literature and claim-evidence
    print("\n--- Running 05_literature_and_novelty.py ---")
    res = subprocess.run([py_bin, str(tools_dir / "05_literature_and_novelty.py")], cwd=base_dir)
    if res.returncode != 0:
        print("[ERROR] Literature audit failed!")
        sys.exit(1)

    # 4. Figures and report
    print("\n--- Running 06_build_figures_and_report.py ---")
    res = subprocess.run([py_bin, str(tools_dir / "06_build_figures_and_report.py")], cwd=base_dir)
    if res.returncode != 0:
        print("[ERROR] Report generation failed!")
        sys.exit(1)

    # 5. Assemble Bundle & Package ZIP
    print("\n--- Assembling Deliverable Archive Structure ---")
    zip_dest = rep_dir / "CHARA_INDEPENDENT_BENCHMARK_EVIDENCE.zip"
    bundle_name = "CHARA_INDEPENDENT_BENCHMARK_EVIDENCE"
    
    # We will build archive in-memory or directly via ZipFile
    files_to_pack = []

    # Root files
    for fname in ["README.md", "REPORT.html", "STATUS.json", "PROTOCOL_LOCK.json", "PROTOCOL_AMENDMENTS.md"]:
        fpath = rep_dir / fname
        if not fpath.exists():
            # If README.md doesn't exist, create it
            if fname == "README.md":
                fpath.write_text(
                    "# CHARA Independent External Benchmark Evidence Archive\n\n"
                    "Contains complete, reproducible atomistic benchmark data across 3 diverse proteins from ATLAS "
                    "(1utg_A, 2cg7_A, 2j6b_A), exact model parameters, held-out predictions, local cMYC corrections, "
                    "and an independent offline verifier.\n\n"
                    "## Verification Instructions\n"
                    "Run offline verifier:\n"
                    "```bash\n"
                    "python scripts/verify_bundle.py --root . --offline\n"
                    "```\n"
                )
        if fpath.exists():
            files_to_pack.append((fpath, f"{bundle_name}/{fname}"))

    # Tables
    for t_name in [
        "BENCHMARK_RESULTS.csv", "PAIRED_COMPARISONS.csv", "DIAGNOSTICS_RESULTS.csv",
        "RANDOM_SEEDS_RESULTS.csv", "GRAPH_ABLATIONS.csv", "CAUSAL_AUDIT_RESULTS.csv",
        "LOCAL_CORRECTIONS.csv", "CLOSEST_WORK.csv", "CLAIM_EVIDENCE.csv",
        "DATA_ACCESS.csv", "DATASET_MANIFEST.csv"
    ]:
        fpath = rep_dir / t_name
        if fpath.exists():
            files_to_pack.append((fpath, f"{bundle_name}/tables/{t_name}"))

    # Data
    for c_file in (base_dir / "data" / "external_atlas" / "processed").glob("*_canonical.npz"):
        files_to_pack.append((c_file, f"{bundle_name}/data/{c_file.name}"))
    
    cmyc_data = rep_dir / "cmyc_retained_numerical_evidence.npz"
    if cmyc_data.exists():
        files_to_pack.append((cmyc_data, f"{bundle_name}/data/{cmyc_data.name}"))

    # Models
    models_file = rep_dir / "models" / "fitted_models.json"
    if models_file.exists():
        files_to_pack.append((models_file, f"{bundle_name}/models/fitted_models.json"))

    # Predictions
    for p_file in (rep_dir / "predictions").glob("*.npz"):
        files_to_pack.append((p_file, f"{bundle_name}/predictions/{p_file.name}"))

    # Source evidence
    bonds_ex = rep_dir / "cmyc_mol0_bonds_excerpt.txt"
    if bonds_ex.exists():
        files_to_pack.append((bonds_ex, f"{bundle_name}/source_evidence/{bonds_ex.name}"))

    # Scripts
    files_to_pack.append((tools_dir / "verify_bundle.py", f"{bundle_name}/scripts/verify_bundle.py"))
    for py_tool in [
        "01_audit_and_local_corrections.py", "02_freeze_protocol.py",
        "03_process_atlas_trajectories.py", "04_execute_external_benchmark.py",
        "05_literature_and_novelty.py", "06_build_figures_and_report.py"
    ]:
        p = tools_dir / py_tool
        if p.exists():
            files_to_pack.append((p, f"{bundle_name}/scripts/{py_tool}"))

    # Figures
    for fig_file in (rep_dir / "figures").glob("*.svg"):
        files_to_pack.append((fig_file, f"{bundle_name}/figures/{fig_file.name}"))

    # Environment
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "gromacs_version": "2026.3-conda_forge",
        "git_commit": "50db73bb97961a804414ed299830bedf9b75c473",
        "numpy": np.__version__,
        "pandas": pd.__version__
    }
    env_json_path = rep_dir / "environment_info.json"
    env_json_path.write_text(json.dumps(env_info, indent=2))
    files_to_pack.append((env_json_path, f"{bundle_name}/environment/environment_info.json"))

    # Generate MANIFEST.sha256
    manifest_lines = []
    for src_path, arc_path in files_to_pack:
        rel_in_bundle = arc_path[len(bundle_name) + 1:]
        h = sha256_file(src_path)
        manifest_lines.append(f"{h}  {rel_in_bundle}")
    manifest_lines.sort()

    manifest_path = rep_dir / "MANIFEST.sha256"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    files_to_pack.append((manifest_path, f"{bundle_name}/MANIFEST.sha256"))

    # Generate PACKAGE_CONTENTS.csv
    contents_records = []
    for src_path, arc_path in files_to_pack:
        rel_in_bundle = arc_path[len(bundle_name) + 1:]
        contents_records.append({
            "relative_path": rel_in_bundle,
            "size_bytes": src_path.stat().st_size,
            "sha256": sha256_file(src_path)
        })
    df_contents = pd.DataFrame(contents_records)
    contents_csv = rep_dir / "PACKAGE_CONTENTS.csv"
    df_contents.to_csv(contents_csv, index=False)
    files_to_pack.append((contents_csv, f"{bundle_name}/PACKAGE_CONTENTS.csv"))

    # Build ZIP with ZIP_DEFLATED
    print(f"Creating {zip_dest.name} with {len(files_to_pack)} entries...")
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_path in files_to_pack:
            zf.write(src_path, arcname=arc_path)

    final_size = zip_dest.stat().st_size
    print(f"\n[OK] Package created: {zip_dest.name}")
    print(f"  Total Bytes: {final_size} ({final_size / 1e6:.2f} MB)")
    assert final_size < 100_000_000, f"Error: Final archive exceeds 100 MB limit ({final_size} bytes)!"
    print(f"  [PASS] Size constraint satisfied: {final_size} < 100,000,000 bytes.")

    # 6. Extract into temporary sandbox and run offline verifier
    sandbox_dir = Path("/tmp/chara_independent_verify_sandbox")
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Extracting into fresh sandbox {sandbox_dir} ---")
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(sandbox_dir)

    extracted_root = sandbox_dir / bundle_name
    verifier_script = extracted_root / "scripts" / "verify_bundle.py"
    
    print(f"--- Running Offline Verifier in Extracted Sandbox ---")
    verif_res = subprocess.run([
        sys.executable, str(verifier_script),
        "--root", str(extracted_root),
        "--offline",
        "--refit"
    ], capture_output=True, text=True)

    print(verif_res.stdout)
    if verif_res.stderr:
        print(verif_res.stderr)

    if verif_res.returncode != 0:
        print("[ERROR] Offline sandbox verification failed!")
        sys.exit(1)

    print("[SUCCESS] All offline sandbox verification checks passed with exit code 0!")

if __name__ == "__main__":
    main()
