#!/usr/bin/env python3
"""
tools/chara_md_review/03_package_all_review_bundles.py

Packages the review archives:
1. CHARA_REVIEW_CORE.zip (Code, configs, lineage, tables, metadata, START_HERE.txt, standalone_evaluation.py)
2. CHARA_KRAS_REVIEW_DATA.zip (Small complete KRAS data bundle with distance arrays, pair maps, sample coords, predictions, evaluator)
3. CHARA_PTPN11_REVIEW_DATA.zip (PTPN11 data bundle)
4. CHARA_MUTP53_REVIEW_DATA.zip (Mut_p53 data bundle)
5. CHARA_CMYC_REVIEW_DATA.zip (cMYC_MAX data bundle)
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
import numpy as np

BASE_DIR = Path("E:/Sharon")
REVIEW_DIR = BASE_DIR / "reports" / "chara_md_review" / "20260907_v1"
EXTRACTED_DIR = REVIEW_DIR / "extracted_data"
EVID_DII = BASE_DIR / "reports" / "chara_md_dii" / "20260907_v1" / "evidence"
STAGE_DIR = REVIEW_DIR / "staging"
STAGE_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def make_zip(source_dir: Path, zip_dest: Path):
    if zip_dest.exists():
        zip_dest.unlink()
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for f in sorted(files):
                p = Path(root) / f
                rel_p = p.relative_to(source_dir).as_posix()
                zf.write(p, arcname=rel_p)
    sz_mb = zip_dest.stat().st_size / (1024 * 1024)
    print(f"[OK] Packaged {zip_dest.name} ({sz_mb:.2f} MB)")

def build_core_archive():
    print("\n--- Staging CHARA_REVIEW_CORE.zip ---")
    core_stage = STAGE_DIR / "core"
    if core_stage.exists(): shutil.rmtree(core_stage)
    core_stage.mkdir(parents=True, exist_ok=True)

    # 1. Tables
    tbl_dir = core_stage / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        "CHARA_REVIEW_NUMBERS.csv", "CHARA_DISCREPANCIES.csv",
        "CHARA_DIAGNOSTICS.csv", "CHARA_CLOSEST_WORK.csv",
        "CHARA_CANDIDATE_QUESTIONS.csv"
    ]
    for t in tables:
        src = REVIEW_DIR / t
        if src.exists(): shutil.copy2(src, tbl_dir / t)
    # Also copy reproduction checks and artifact inventory from DII evidence
    if (EVID_DII / "reproduction_checks.csv").exists():
        shutil.copy2(EVID_DII / "reproduction_checks.csv", tbl_dir / "reproduction_checks.csv")
    if (EVID_DII / "artifact_inventory.csv").exists():
        shutil.copy2(EVID_DII / "artifact_inventory.csv", tbl_dir / "artifact_inventory.csv")

    # 2. Code
    code_dir = core_stage / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    # Copy tools
    for td in ["tools/chara_md_review", "tools/chara_md_dii", "tools/chara_md_followup", "tools/chara_md_pilot"]:
        src_td = BASE_DIR / td
        if src_td.exists():
            shutil.copytree(src_td, code_dir / td, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # Copy chara/graph.py
    chara_src = BASE_DIR / "chara"
    if chara_src.exists():
        shutil.copytree(chara_src, code_dir / "chara", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 3. Metadata & Simulation Identities
    meta_dir = core_stage / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    top_dir = BASE_DIR / "data" / "cg_topologies"
    if top_dir.exists():
        for s in ["KRAS_G12D", "PTPN11", "Mut_p53", "cMYC_MAX"]:
            s_top = top_dir / s
            dst_top = meta_dir / "cg_topologies" / s
            dst_top.mkdir(parents=True, exist_ok=True)
            for f in ["molecule_0.itp", "martini.itp"]:
                f_p = s_top / f
                if f_p.exists(): shutil.copy2(f_p, dst_top / f)
                # also check md_runs for martini.itp
                f_md = BASE_DIR / "data" / "md_runs" / s / f
                if f_md.exists() and not (dst_top / f).exists():
                    shutil.copy2(f_md, dst_top / f)

    # 4. Features & Histories
    feat_dir = core_stage / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    if (EVID_DII / "dii_weights_and_history.json").exists():
        shutil.copy2(EVID_DII / "dii_weights_and_history.json", feat_dir / "dii_weights_and_history.json")
    if (EVID_DII / "dii_optimization_summary.csv").exists():
        shutil.copy2(EVID_DII / "dii_optimization_summary.csv", feat_dir / "dii_optimization_summary.csv")

    # 5. START_HERE.txt
    start_here_content = """================================================================================
CHARA REVIEW CORE EVIDENCE BUNDLE: START_HERE.txt
================================================================================

DOCUMENT PURPOSE:
This bundle provides the complete source code, configurations, corrected metadata,
lineage traces, numerical tables, and standalone evaluator for the Chara MD
sparse distance sensing and Differentiable Information Imbalance (DII) benchmarks.

DIRECTORY LAYOUT:
  START_HERE.txt             : This instruction file.
  standalone_evaluation.py   : Independent evaluator to recompute metrics from predictions.
  tables/                    : Consolidated numerical and diagnostic tables:
    - CHARA_REVIEW_NUMBERS.csv       : 348 evaluated model rows with exact lineage traces.
    - CHARA_DISCREPANCIES.csv        : Factual resolution of 7 known comparison discrepancies.
    - CHARA_DIAGNOSTICS.csv          : Per-target RMSE, bias^2+var, Pearson r, restraint stratification.
    - CHARA_CLOSEST_WORK.csv         : 8 primary literature comparisons with DOIs and specific gaps.
    - CHARA_CANDIDATE_QUESTIONS.csv  : 3 grounded, falsifiable research questions.
    - reproduction_checks.csv        : 128 independent recalculation and trajectory spot checks.
    - artifact_inventory.csv         : Full SHA256 artifact manifest.
  code/                      : Standalone source code:
    - tools/chara_md_review/         : Review export and diagnostic scripts.
    - tools/chara_md_dii/            : Official DADApy DII benchmark execution suite.
    - tools/chara_md_followup/       : Multi-system cross-validation pipeline.
    - tools/chara_md_pilot/          : Initial bounded pilot scripts.
    - chara/graph.py                 : Heat-kernel and Laplacian utilities.
  metadata/                  : Primary GROMACS topologies (Martini 3.0.01) and parameters.
  features/                  : DII optimization histories, weights, and loss trajectories.
  manifest/                  : SHA256 cryptographic manifest for all bundle members.

REPRODUCIBILITY & INDEPENDENT VERIFICATION:
To verify predictions against the verified metrics table:
  1. Ensure numpy is available (Python 3.9+).
  2. Download and extract CHARA_KRAS_REVIEW_DATA.zip (or other system bundle) alongside this core bundle.
  3. Run the standalone evaluator:
       python standalone_evaluation.py --data_dir <path_to_extracted_system_bundle>
  4. Expected output: All recomputed RMSE metrics match CSV entries to within 1e-05 nm tolerance.

SIMULATION FORCE FIELD & PARAMETERS:
  - Engine: GROMACS 2026.3
  - Force field: Martini 3.0.01 (martinize2 -ff martini3001 -elastic -ef 500 -el 0.5 -eu 0.9)
  - Ensemble: NPT at 310 K, 1.0 bar
  - Integrator: md (leap-frog), dt = 20 fs
  - Thermostat: V-rescale (tau_t = 1.0 ps)
  - Barostat: C-rescale (tau_p = 12 ps, compressibility = 3.0e-4 bar^-1)
  - Analyzed Constructs:
      * KRAS_G12D: Chain A (170 residues, PDB 4OBE)
      * PTPN11: Chain A (536 residues, PDB 4DGP)
      * Mut_p53: Chain A (219 residues, PDB 2J1X)
      * cMYC_MAX: Chain E monomer (88 residues, PDB 1NKP; MAX chain absent)
================================================================================
"""
    with open(core_stage / "START_HERE.txt", "w", encoding="utf-8") as f:
        f.write(start_here_content)

    # 6. standalone_evaluation.py
    evaluator_code = """#!/usr/bin/env python3
\"\"\"
standalone_evaluation.py
Self-contained independent evaluator to verify model prediction arrays against CHARA_REVIEW_NUMBERS.csv.
\"\"\"
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Standalone Chara Review Prediction Evaluator")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing raw_predictions or system npz files")
    parser.add_argument("--tables_dir", type=str, default="tables", help="Directory containing CHARA_REVIEW_NUMBERS.csv")
    args = parser.parse_args()

    base_dir = Path(args.data_dir).resolve()
    tbl_dir = Path(args.tables_dir).resolve()
    if not tbl_dir.exists():
        tbl_dir = Path(__file__).resolve().parent / "tables"

    csv_path = tbl_dir / "CHARA_REVIEW_NUMBERS.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        sys.exit(1)

    verified = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["system"], r["fold"], r["method_id"], r["requested_k"])
            try: verified[key] = float(r["RMSE_nm"])
            except ValueError: pass

    # Find npz files
    npz_files = list(base_dir.glob("**/*predictions*.npz"))
    if not npz_files:
        npz_files = list(base_dir.glob("*.npz"))

    print(f"Found {len(npz_files)} prediction files to evaluate.")
    print(f"{'System':<12} {'Fold':<8} {'Method':<25} {'k':<5} {'Recomputed RMSE':<18} {'CSV RMSE':<14} {'Diff':<12} {'Status'}")
    print("-" * 100)

    checks_passed = 0
    mismatches = 0

    for fpath in sorted(npz_files):
        stem = fpath.stem
        if stem.startswith("predictions_"): stem = stem[len("predictions_"):]
        if "_fold_" not in stem: continue
        s_name, f_num = stem.rsplit("_fold_", 1)
        f_id = f"fold_{f_num}"

        data = np.load(fpath)
        if "Y_true" not in data: continue
        Y_true = data["Y_true"]
        n_elem = Y_true.size

        # Check baseline
        if "pred_baseline_mean" in data:
            p_base = data["pred_baseline_mean"]
            rmse_b = float(np.sqrt(np.sum((Y_true - p_base)**2) / n_elem))
            key_b = (s_name, f_id, "M1_MEAN", "constant")
            csv_b = verified.get(key_b, None)
            diff_b = abs(rmse_b - csv_b) if csv_b is not None else 0.0
            st_b = "MATCH" if (csv_b is not None and diff_b < 1e-5) else "MISMATCH"
            if st_b == "MATCH": checks_passed += 1
            else: mismatches += 1
            csv_str = f"{csv_b:.6f} nm" if csv_b is not None else "N/A"
            print(f"{s_name:<12} {f_id:<8} {'M1_MEAN':<25} {'const':<5} {rmse_b:.6f} nm        {csv_str:<14} {diff_b:.2e}     {st_b}")

        # Check other methods at k=10
        methods = [
            ("M10_OFFICIAL_DII_100F", "pred_dii_100f_k10", "10"),
            ("M10_OFFICIAL_DII_400F", "pred_dii_400f_k10", "10"),
            ("M8_ROBUST_TRANSFER", "pred_m8_robust_transfer_k10", "10"),
            ("M7_MEAN_TRANSFER", "pred_m7_mean_transfer_k10", "10"),
            ("M5_RANK_INFORMATION_IMBALANCE", "pred_m5_rank_information_imbalance_k10", "10"),
        ]
        for m_id, arr_name, k_str in methods:
            if arr_name in data:
                p_arr = data[arr_name]
                rmse_v = float(np.sqrt(np.sum((Y_true - p_arr)**2) / n_elem))
                key = (s_name, f_id, m_id, k_str)
                csv_v = verified.get(key, None)
                diff = abs(rmse_v - csv_v) if csv_v is not None else 0.0
                st = "MATCH" if (csv_v is not None and diff < 1e-5) else "MISMATCH"
                if st == "MATCH": checks_passed += 1
                else: mismatches += 1
                csv_str = f"{csv_v:.6f} nm" if csv_v is not None else "N/A"
                print(f"{s_name:<12} {f_id:<8} {m_id:<25} {k_str:<5} {rmse_v:.6f} nm        {csv_str:<14} {diff:.2e}     {st}")

    print("-" * 100)
    print(f"Summary: {checks_passed} checks passed, {mismatches} mismatches.")
    if mismatches == 0 and checks_passed > 0:
        print("[OK] All recomputed model predictions match verified numbers.")
        sys.exit(0)
    elif checks_passed == 0:
        print("[WARN] No matching prediction files found in directory.")
        sys.exit(2)
    else:
        print("[FAIL] Mismatches detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    with open(core_stage / "standalone_evaluation.py", "w", encoding="utf-8") as f:
        f.write(evaluator_code)

    # 7. Cryptographic Manifest
    manifest = {
        "bundle": "CHARA_REVIEW_CORE",
        "timestamp": "2026-09-07 21:40:00",
        "platform": platform.platform(),
        "python_version": sys.version,
        "files_sha256": {}
    }
    for root, _, files in os.walk(core_stage):
        for file in sorted(files):
            p = Path(root) / file
            rel_p = p.relative_to(core_stage).as_posix()
            manifest["files_sha256"][rel_p] = sha256_file(p)

    man_dir = core_stage / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    with open(man_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create ZIP
    core_zip = REVIEW_DIR / "CHARA_REVIEW_CORE.zip"
    make_zip(core_stage, core_zip)


def build_system_data_bundle(system_name: str, zip_name: str):
    print(f"\n--- Staging {zip_name} for {system_name} ---")
    sys_stage = STAGE_DIR / system_name
    if sys_stage.exists(): shutil.rmtree(sys_stage)
    sys_stage.mkdir(parents=True, exist_ok=True)

    src_sys_data = EXTRACTED_DIR / system_name
    # 1. Distances and Mappings
    for f in ["distances_rep1.npz", "distances_rep2.npz", "distances_rep3.npz",
              "candidates.json", "targets.json", "bb_residue_map.json",
              "sample_coordinates_rep1.npz"]:
        f_p = src_sys_data / f
        if f_p.exists(): shutil.copy2(f_p, sys_stage / f)

    # 2. Primary k=10 Predictions & Baselines
    pred_dir = sys_stage / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    k10_keys = [
        "Y_true", "pred_baseline_mean",
        "pred_dii_100f_k10", "indices_dii_100f_k10",
        "pred_dii_400f_k10", "indices_dii_400f_k10",
        "pred_m8_robust_transfer_k10", "indices_m8_robust_transfer_k10",
        "pred_m7_mean_transfer_k10", "indices_m7_mean_transfer_k10",
        "pred_m9_graph_assisted_k10", "indices_m9_graph_assisted_k10",
        "pred_m6_pooled_greedy_k10", "indices_m6_pooled_greedy_k10",
        "pred_m5_rank_information_imbalance_k10", "indices_m5_rank_information_imbalance_k10",
        "pred_m4_pca_qr_k10", "indices_m4_pca_qr_k10",
        "pred_m3_variance_k10", "indices_m3_variance_k10",
        "pred_m2_random_k10", "indices_m2_random_k10"
    ]
    for fid in ["fold_1", "fold_2", "fold_3"]:
        npz_p = EVID_DII / "raw_predictions" / f"predictions_{system_name}_{fid}.npz"
        if npz_p.exists():
            full_data = np.load(npz_p)
            k10_data = {k: full_data[k] for k in k10_keys if k in full_data}
            np.savez_compressed(pred_dir / npz_p.name, **k10_data)

    if system_name == "KRAS_G12D":
        pca_comp = {
            "explanation": "Comparison of historical pilot vs current benchmark PCA/QR feature selection",
            "pilot_method": "SVD on pooled train: Vt[:20, :] (20 modes) pivoted via QR, sliced [:10]",
            "pilot_kras_fold_1_k10_rmse_nm": 0.06908166,
            "pilot_kras_3fold_mean_k10_rmse_nm": 0.07390177,
            "current_method": "SVD on pooled train: Vt[:10, :] (10 modes) pivoted via QR for k=10",
            "current_kras_fold_1_k10_rmse_nm": 0.083115,
            "current_kras_3fold_mean_k10_rmse_nm": 0.078486,
            "difference_fold_1_nm": 0.014033,
            "discrepancy_record": "DISC_02_PCA_QR_BASELINE_CHANGE"
        }
        with open(sys_stage / "pca_qr_historical_comparison.json", "w", encoding="utf-8") as f:
            json.dump(pca_comp, f, indent=2)

    # 3. Verification Script for Raw Coordinate Spot Checks
    verify_raw_script = """#!/usr/bin/env python3
\"\"\"
verify_raw_distances.py
Independently recomputes distances from sample_coordinates_rep1.npz and verifies exact match with distances_rep1.npz.
\"\"\"
import json
from pathlib import Path
import numpy as np

def main():
    base = Path(__file__).resolve().parent
    coords_npz = base / "sample_coordinates_rep1.npz"
    dist_npz = base / "distances_rep1.npz"
    cand_json = base / "candidates.json"
    targ_json = base / "targets.json"

    if not coords_npz.exists() or not dist_npz.exists():
        print("ERROR: sample coordinates or distance file missing.")
        return 1

    c_data = np.load(coords_npz)
    d_data = np.load(dist_npz)
    candidates = json.load(open(cand_json))
    targets = json.load(open(targ_json))

    coords = c_data["coords_nm"] # (10, n_bb, 3)
    frame_indices = c_data["frame_indices"]
    dist_frame_indices = d_data["frame_indices"]

    print(f"Testing raw distance recalculation across {len(frame_indices)} sample coordinate frames...")
    checks = 0
    max_diff = 0.0

    c_i = np.array([p["i_idx"] for p in candidates])
    c_j = np.array([p["j_idx"] for p in candidates])
    t_i = np.array([p["i_idx"] for p in targets])
    t_j = np.array([p["j_idx"] for p in targets])

    for local_idx, f_idx in enumerate(frame_indices):
        # find matching row in distances_rep1
        match_rows = np.where(dist_frame_indices == f_idx)[0]
        if len(match_rows) == 0: continue
        d_row = match_rows[0]

        frame_crd = coords[local_idx] # (n_bb, 3)
        calc_cand = np.linalg.norm(frame_crd[c_i] - frame_crd[c_j], axis=-1)
        calc_targ = np.linalg.norm(frame_crd[t_i] - frame_crd[t_j], axis=-1)

        diff_c = np.max(np.abs(calc_cand - d_data["X"][d_row]))
        diff_t = np.max(np.abs(calc_targ - d_data["Y"][d_row]))
        max_diff = max(max_diff, diff_c, diff_t)
        checks += 1

    print(f"[OK] {checks} coordinate frames verified against distance array. Max absolute difference: {max_diff:.2e} nm.")
    if max_diff < 1e-5:
        print("[PASSED] Exact raw coordinate distance reproduction confirmed.")
        return 0
    else:
        print("[FAILED] Difference exceeds tolerance.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
"""
    with open(sys_stage / "verify_raw_distances.py", "w", encoding="utf-8") as f:
        f.write(verify_raw_script)

    # 4. Standalone System Evaluator
    sys_eval_script = f"""#!/usr/bin/env python3
\"\"\"
reproduce_system_metrics.py
Standalone metric reproduction script for {system_name}.
\"\"\"
from pathlib import Path
import numpy as np

def main():
    base = Path(__file__).resolve().parent
    pred_dir = base / "predictions"
    print(f"=== Reproducing {system_name} Held-Out Metrics at k=10 ===")
    print(f"{{'Fold':<8}} {{'Method':<30}} {{'RMSE (nm)':<12}} {{'Skill':<10}}")
    print("-" * 65)

    for f_id in ["fold_1", "fold_2", "fold_3"]:
        npz_p = pred_dir / f"predictions_{system_name}_{{f_id}}.npz"
        if not npz_p.exists(): continue
        data = np.load(npz_p)
        Y_true = data["Y_true"]
        n_elem = Y_true.size
        p_base = data["pred_baseline_mean"]
        base_sse = float(np.sum((Y_true - p_base)**2))
        base_rmse = float(np.sqrt(base_sse / n_elem))

        print(f"{{f_id:<8}} {{'M1_MEAN (Baseline)':<30}} {{base_rmse:.6f}} nm  0.000000")

        for m_id, arr in [
            ("M10_OFFICIAL_DII_100F", "pred_dii_100f_k10"),
            ("M10_OFFICIAL_DII_400F", "pred_dii_400f_k10"),
            ("M8_ROBUST_TRANSFER", "pred_m8_robust_transfer_k10"),
            ("M7_MEAN_TRANSFER", "pred_m7_mean_transfer_k10"),
            ("M5_RANK_INFO_IMBALANCE", "pred_m5_rank_information_imbalance_k10"),
        ]:
            if arr in data:
                p_arr = data[arr]
                sse = float(np.sum((Y_true - p_arr)**2))
                rmse = float(np.sqrt(sse / n_elem))
                skill = float(1.0 - sse / base_sse)
                print(f"{{f_id:<8}} {{m_id:<30}} {{rmse:.6f}} nm  {{skill:+.6f}}")
        print("-" * 65)

if __name__ == "__main__":
    main()
"""
    with open(sys_stage / "reproduce_system_metrics.py", "w", encoding="utf-8") as f:
        f.write(sys_eval_script)

    # 5. Manifest
    manifest = {
        "system": system_name,
        "bundle": zip_name,
        "files_sha256": {}
    }
    for root, _, files in os.walk(sys_stage):
        for file in sorted(files):
            p = Path(root) / file
            rel_p = p.relative_to(sys_stage).as_posix()
            manifest["files_sha256"][rel_p] = sha256_file(p)
    with open(sys_stage / "data_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Create ZIP
    out_zip = REVIEW_DIR / zip_name
    make_zip(sys_stage, out_zip)


def main():
    print("=== Step 3: Packaging All Review Evidence Bundles ===")
    build_core_archive()
    build_system_data_bundle("KRAS_G12D", "CHARA_KRAS_REVIEW_DATA.zip")
    build_system_data_bundle("PTPN11", "CHARA_PTPN11_REVIEW_DATA.zip")
    build_system_data_bundle("Mut_p53", "CHARA_MUTP53_REVIEW_DATA.zip")
    build_system_data_bundle("cMYC_MAX", "CHARA_CMYC_REVIEW_DATA.zip")
    print("\n[OK] All review archives successfully created.")

if __name__ == "__main__":
    main()
