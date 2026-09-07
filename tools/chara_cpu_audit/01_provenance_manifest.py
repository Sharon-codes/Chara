#!/usr/bin/env python3
"""
tools/chara_cpu_audit/01_provenance_manifest.py
Audits repository status, hardware/CPU specs, Python environment, dependencies,
and computes SHA-256 hashes of all relevant codebase and data artifacts.
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def get_sha256(filepath):
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_git_info():
    info = {}
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True)
        info["commit_hash"] = res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
    except Exception as e:
        info["commit_hash"] = f"ERROR: {e}"

    try:
        res = subprocess.run(["git", "branch", "--show-current"], cwd=str(ROOT), capture_output=True, text=True)
        info["branch"] = res.stdout.strip() if res.returncode == 0 else "UNKNOWN"
    except Exception as e:
        info["branch"] = f"ERROR: {e}"

    try:
        res = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True)
        info["is_dirty"] = bool(res.stdout.strip())
        info["dirty_files"] = [line.strip() for line in res.stdout.splitlines()] if res.stdout.strip() else []
    except Exception as e:
        info["is_dirty"] = "UNKNOWN"
        info["dirty_files"] = [str(e)]
    return info

def get_system_info():
    sys_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }
    
    sys_info["thread_env"] = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "Not set (default)"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "Not set (default)"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "Not set (default)"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "Not set (default)"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "Not set (default)"),
    }
    
    packages = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "sksurv", "lifelines", "joblib", "torch", "gseapy"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = "NOT INSTALLED"
    sys_info["packages"] = packages
    return sys_info

def audit_file_inventory():
    tracked_files = [
        "TCGA-LUAD_expression.csv",
        "TCGA-LUAD_survival.csv",
        "TCGA-PAAD_expression.csv",
        "TCGA-PAAD_survival.csv",
        "Laplacian_STRING.csv",
        "Laplacian_STRING_4337.csv",
        "Laplacian_Chara.csv",
        "Laplacian_Chara_4337.csv",
        "intersecting_genes_4337.txt",
        "frontier_benchmark_results.csv",
        "chara_model_4337.pkl",
        "chara_model.pkl",
        "data/raw/4OBE.pdb",
        "data/raw/1NKP.pdb",
        "data/raw/2J1X.pdb",
        "data/raw/4DGP.pdb",
        "data/processed/KRAS_4OBE_clean.pdb",
        "data/processed/c-MYC_1NKP_clean.pdb",
        "data/processed/p53_2J1X_clean.pdb",
        "data/processed/PTPN11_4DGP_clean.pdb",
        "setup.py",
        "pyproject.toml",
        "README.md",
        "chara/__init__.py",
        "chara/model.py",
        "chara/graph.py",
        "chara/metrics.py",
        "chara/preprocessing.py",
        "scripts/01_phase1_prep.py",
        "scripts/02_phase2_cg.py",
        "scripts/03_phase3_gromacs.py",
        "scripts/03_generate_chara.py",
        "scripts/04_chara_ood_validation.py",
        "scripts/05_adversarial_poisoning.py",
        "scripts/06_dirichlet_energy.py",
        "scripts/07_biological_gsea.py",
        "scripts/09_zeroshot_external_validation.py",
        "scripts/17_generate_contact_matrices.py",
        "scripts/benchmark_frontiers.py",
        "scripts/01_master_cagpr_validation.py",
    ]
    
    inventory = {}
    for rel_path in tracked_files:
        full_p = ROOT / rel_path
        if full_p.exists():
            inventory[rel_path] = {
                "exists": True,
                "size_bytes": full_p.stat().st_size,
                "sha256": get_sha256(full_p),
                "modified_iso": str(full_p.stat().st_mtime),
            }
        else:
            inventory[rel_path] = {
                "exists": False,
                "size_bytes": None,
                "sha256": None,
                "modified_iso": None,
            }
    return inventory

def main():
    print("[*] Generating Provenance Manifest...")
    manifest = {
        "root": str(ROOT),
        "git": get_git_info(),
        "system": get_system_info(),
        "inventory": audit_file_inventory()
    }
    
    out_path = EVIDENCE_DIR / "manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Saved provenance manifest to {out_path}")
    print(f"    Git Commit : {manifest['git']['commit_hash']} (dirty: {manifest['git']['is_dirty']})")
    print(f"    OS / Python: {manifest['system']['os']} {manifest['system']['os_release']} / Python {manifest['system']['python_version']}")
    print(f"    Inventory  : {sum(1 for v in manifest['inventory'].values() if v['exists'])} / {len(manifest['inventory'])} files present")

if __name__ == "__main__":
    main()
