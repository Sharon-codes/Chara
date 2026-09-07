#!/usr/bin/env python3
"""
tools/chara_cpu_audit/02_molecular_audit.py
Forensic molecular identity audit:
- PDB sequence and construct verification
- Target mutations (G12D, R273H)
- Target PTMs (SEP 181, TPO 58, SEP 392, PTR 542)
- Heterogen, ion, cofactor and DNA stripping verification
- Trajectory completion, duration, timesteps, and log audit
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = ROOT / "reports" / "chara_cpu_audit" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

def parse_pdb_residues(filepath):
    p = Path(filepath)
    if not p.exists():
        return None
    residues = []
    seen = set()
    heterogens = set()
    with open(p, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                res_num = line[22:26].strip()
                res_name = line[17:20].strip()
                key = (chain, res_num, res_name)
                if key not in seen:
                    seen.add(key)
                    residues.append({"chain": chain, "res_num": res_num, "res_name": res_name})
            elif line.startswith("HETATM"):
                res_name = line[17:20].strip()
                heterogens.add(res_name)
    return {"residues": residues, "heterogens": sorted(list(heterogens))}

def audit_kras():
    raw = parse_pdb_residues(ROOT / "data" / "raw" / "4OBE.pdb")
    clean = parse_pdb_residues(ROOT / "data" / "processed" / "KRAS_4OBE_clean.pdb")
    
    # Check residue 12 in clean
    res12_clean = [r for r in clean["residues"] if r["res_num"] == "12"]
    has_g12d = any(r["res_name"] == "ASP" for r in res12_clean)
    has_sep181 = any(r["res_num"] == "181" and r["res_name"] == "SEP" for r in clean["residues"])
    has_res181 = any(r["res_num"] == "181" for r in clean["residues"])
    
    # Max residue number
    nums = [int(r["res_num"]) for r in clean["residues"] if r["res_num"].isdigit()]
    max_res = max(nums) if nums else None
    
    return {
        "system": "KRAS",
        "pdb_id": "4OBE",
        "raw_total_residues": len(raw["residues"]) if raw else 0,
        "clean_total_residues": len(clean["residues"]) if clean else 0,
        "chains": sorted(list(set(r["chain"] for r in clean["residues"]))),
        "residue_range": f"1 to {max_res}",
        "residue_12_identity": [r["res_name"] for r in res12_clean],
        "claimed_G12D_present": has_g12d,
        "actual_mutation_introduced": False,
        "claimed_SEP181_present": has_sep181,
        "residue_181_exists_in_construct": has_res181,
        "raw_heterogens": raw["heterogens"] if raw else [],
        "clean_heterogens": clean["heterogens"] if clean else [],
        "notes": "4OBE crystal structure is KRAS catalytic domain (residues 1-169, with residue 12 as ALA). Residue 181 is in disordered hypervariable tail, entirely absent from construct."
    }

def audit_cmyc():
    raw = parse_pdb_residues(ROOT / "data" / "raw" / "1NKP.pdb")
    clean = parse_pdb_residues(ROOT / "data" / "processed" / "c-MYC_1NKP_clean.pdb")
    
    has_tpo58 = any(r["res_num"] == "58" and r["res_name"] == "TPO" for r in clean["residues"])
    has_res58 = any(r["res_num"] == "58" for r in clean["residues"])
    res58_types = [r["res_name"] for r in clean["residues"] if r["res_num"] == "58"]
    
    return {
        "system": "c-MYC/MAX",
        "pdb_id": "1NKP",
        "raw_total_residues": len(raw["residues"]) if raw else 0,
        "clean_total_residues": len(clean["residues"]) if clean else 0,
        "chains": sorted(list(set(r["chain"] for r in clean["residues"]))),
        "claimed_TPO58_present": has_tpo58,
        "res58_present_in_construct": has_res58,
        "res58_identity": res58_types,
        "raw_heterogens": raw["heterogens"] if raw else [],
        "clean_heterogens": clean["heterogens"] if clean else [],
        "notes": "1NKP is the bHLH-LZ domain (c-Myc ~355-437, Max ~22-102) bound to DNA. The N-terminal TAD containing Thr58 is absent from the construct. DNA duplex was stripped."
    }

def audit_p53():
    raw = parse_pdb_residues(ROOT / "data" / "raw" / "2J1X.pdb")
    clean = parse_pdb_residues(ROOT / "data" / "processed" / "p53_2J1X_clean.pdb")
    
    has_sep392 = any(r["res_num"] == "392" and r["res_name"] == "SEP" for r in clean["residues"])
    has_res392 = any(r["res_num"] == "392" for r in clean["residues"])
    nums = [int(r["res_num"]) for r in clean["residues"] if r["res_num"].isdigit()]
    max_res = max(nums) if nums else None
    
    return {
        "system": "Mutant p53",
        "pdb_id": "2J1X",
        "raw_total_residues": len(raw["residues"]) if raw else 0,
        "clean_total_residues": len(clean["residues"]) if clean else 0,
        "chains": sorted(list(set(r["chain"] for r in clean["residues"]))),
        "residue_range": f"1 to {max_res} (corresponding to core domain 94-312)",
        "claimed_R273H_present": False,
        "claimed_SEP392_present": has_sep392,
        "residue_392_exists_in_construct": has_res392,
        "raw_heterogens": raw["heterogens"] if raw else [],
        "clean_heterogens": clean["heterogens"] if clean else [],
        "notes": "2J1X is a core domain construct (residues 94-312) containing stabilizing mutations M133L, V203A, N239Y, N268D. R273H was not introduced. Ser392 is in the C-terminal regulatory domain, absent from construct. Zinc cofactor was stripped."
    }

def audit_ptpn11():
    raw = parse_pdb_residues(ROOT / "data" / "raw" / "4DGP.pdb")
    clean = parse_pdb_residues(ROOT / "data" / "processed" / "PTPN11_4DGP_clean.pdb")
    
    has_ptr542 = any(r["res_num"] == "542" and r["res_name"] == "PTR" for r in clean["residues"])
    has_res542 = any(r["res_num"] == "542" for r in clean["residues"])
    nums = [int(r["res_num"]) for r in clean["residues"] if r["res_num"].isdigit()]
    max_res = max(nums) if nums else None
    
    return {
        "system": "PTPN11 (SHP2)",
        "pdb_id": "4DGP",
        "raw_total_residues": len(raw["residues"]) if raw else 0,
        "clean_total_residues": len(clean["residues"]) if clean else 0,
        "chains": sorted(list(set(r["chain"] for r in clean["residues"]))),
        "residue_range": f"1 to {max_res}",
        "claimed_PTR542_present": has_ptr542,
        "residue_542_exists_in_construct": has_res542,
        "raw_heterogens": raw["heterogens"] if raw else [],
        "clean_heterogens": clean["heterogens"] if clean else [],
        "notes": "4DGP represents SHP2 autoinhibited form (1-528 plus C-term His-tag up to 536). Tyr542 resides in the disordered C-terminal tail (529-593), not present in this crystal structure."
    }

def audit_trajectories():
    md_dir = ROOT / "data" / "md_runs"
    targets = ["KRAS_G12D", "cMYC_MAX", "Mut_p53", "PTPN11"]
    runs = {}
    total_ns = 0.0
    
    for t in targets:
        runs[t] = {}
        t_dir = md_dir / t
        if not t_dir.exists():
            continue
        for rep in ["rep1", "rep2", "rep3"]:
            rep_dir = t_dir / rep
            log_p = rep_dir / "production.log"
            xtc_p = rep_dir / "production.xtc"
            
            info = {
                "completed": False,
                "nsteps": None,
                "timestep_ps": None,
                "total_time_ns": 0.0,
                "xtc_bytes": xtc_p.stat().st_size if xtc_p.exists() else 0,
                "wall_time_s": None,
                "gmx_version": None,
            }
            
            if log_p.exists():
                with open(log_p, "r", errors="ignore") as f:
                    lines = f.readlines()
                for l in lines[:30]:
                    if "GROMACS - gmx mdrun," in l:
                        info["gmx_version"] = l.strip()
                for l in lines[:300]:
                    if "dt" in l and "=" in l:
                        parts = l.split("=")
                        if len(parts) == 2 and parts[0].strip() == "dt":
                            try: info["timestep_ps"] = float(parts[1].split()[0])
                            except: pass
                    if "nsteps" in l and "=" in l:
                        parts = l.split("=")
                        if len(parts) == 2 and parts[0].strip() == "nsteps":
                            try: info["nsteps"] = int(parts[1].split()[0])
                            except: pass
                for l in reversed(lines[-200:]):
                    if "Finished mdrun" in l:
                        info["completed"] = True
                    if "Total" in l and len(l.split()) >= 3:
                        try:
                            info["wall_time_s"] = float(l.split()[1])
                        except: pass
                if info["nsteps"] and info["timestep_ps"]:
                    info["total_time_ns"] = (info["nsteps"] * info["timestep_ps"]) / 1000.0
                    if info["completed"]:
                        total_ns += info["total_time_ns"]
            runs[t][rep] = info
    return {"runs": runs, "total_completed_sampling_ns": total_ns}

def main():
    print("[*] Running Molecular Identity & Trajectory Audit...")
    audit = {
        "systems": {
            "KRAS": audit_kras(),
            "cMYC_MAX": audit_cmyc(),
            "Mut_p53": audit_p53(),
            "PTPN11": audit_ptpn11()
        },
        "trajectories": audit_trajectories()
    }
    
    out_path = EVIDENCE_DIR / "molecular_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"[OK] Saved molecular audit to {out_path}")
    print(f"    Total Completed MD Sampling: {audit['trajectories']['total_completed_sampling_ns']} ns")
    for s_name, s_data in audit["systems"].items():
        print(f"    {s_name:<12} | Claimed PTM Present: {s_data.get('claimed_SEP181_present') or s_data.get('claimed_TPO58_present') or s_data.get('claimed_SEP392_present') or s_data.get('claimed_PTR542_present')}")

if __name__ == "__main__":
    main()
