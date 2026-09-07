#!/usr/bin/env python3
"""
tools/chara_md_pilot/01_discover_and_manifest.py

Comprehensive discovery, provenance verification, and manifest generation
for the four GROMACS/Martini molecular dynamics simulation systems in the repository:
KRAS_G12D, PTPN11, Mut_p53, and cMYC_MAX.
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from MDAnalysis.coordinates.XTC import XTCReader

BASE_DIR = Path("E:/Sharon")
DATA_DIR = BASE_DIR / "data"
MD_RUNS_DIR = DATA_DIR / "md_runs"
CG_TOP_DIR = DATA_DIR / "cg_topologies"
OUT_DIR = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["KRAS_G12D", "PTPN11", "Mut_p53", "cMYC_MAX"]
REPLICATES = ["rep1", "rep2", "rep3"]

def get_file_hash(filepath: Path, max_bytes: int = 10 * 1024 * 1024):
    """Compute sha256. If file > max_bytes, computes sha256 on first max_bytes and labels as partial."""
    file_size = filepath.stat().st_size
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    is_partial = file_size > max_bytes
    return {
        "sha256": h.hexdigest(),
        "is_partial": is_partial,
        "bytes_hashed": min(file_size, max_bytes),
        "total_bytes": file_size
    }

def parse_itp_file(itp_path: Path):
    """Parses Martini .itp file to extract atoms, residues, beads, bonds, and elastic restraints."""
    atoms = []
    elastic_bonds = []
    current_section = None
    
    with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                continue
            parts = line.split()
            if current_section == "atoms":
                if len(parts) >= 5:
                    atoms.append({
                        "nr": int(parts[0]),
                        "type": parts[1],
                        "resnr": int(parts[2]),
                        "resname": parts[3],
                        "atomname": parts[4]
                    })
            elif current_section in ("bonds", "constraints"):
                if len(parts) >= 2:
                    try:
                        i, j = int(parts[0]), int(parts[1])
                        elastic_bonds.append((i, j, current_section))
                    except ValueError:
                        pass
    
    res_dict = {}
    for a in atoms:
        rnr = a["resnr"]
        if rnr not in res_dict:
            res_dict[rnr] = {"resname": a["resname"], "beads": []}
        res_dict[rnr]["beads"].append(a["atomname"])
        
    return {
        "atoms": atoms,
        "residues": res_dict,
        "elastic_bonds_count": len(elastic_bonds),
        "elastic_bonds": elastic_bonds
    }

def parse_gmx_log(log_path: Path):
    """Extracts integrator parameters, time steps, temperature, pressure, seeds from GROMACS log."""
    info = {}
    if not log_path.exists():
        return info
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read(100000)
        
    def find_val(pattern, text, default=None):
        m = re.search(pattern, text)
        return m.group(1) if m else default

    info["gmx_version"] = find_val(r"GROMACS version:\s*([^\n\r]+)", text)
    info["dt_ps"] = float(find_val(r"dt\s*=\s*([0-9\.]+)", text, "0.02"))
    info["nsteps"] = int(find_val(r"nsteps\s*=\s*([0-9]+)", text, "0"))
    info["tcoupl"] = find_val(r"tcoupl\s*=\s*(\w+)", text)
    info["pcoupl"] = find_val(r"pcoupl\s*=\s*(\w+)", text)
    info["ref_t_k"] = find_val(r"ref_t\s*=\s*([0-9\.]+)", text)
    info["ref_p_bar"] = find_val(r"ref_p\s*=\s*([0-9\.]+)", text)
    info["ld_seed"] = find_val(r"ld[-_]seed\s*=\s*(-?[0-9]+)", text)
    info["total_sim_time_ns"] = (info["dt_ps"] * info["nsteps"]) / 1000.0 if info["nsteps"] else None
    return info

def inspect_system(sysname: str):
    print(f"[INFO] Inspecting system: {sysname}...")
    sys_dir = MD_RUNS_DIR / sysname
    top_dir = CG_TOP_DIR / sysname
    
    itp_files = sorted(top_dir.glob("*.itp")) if top_dir.exists() else []
    itp_info = {}
    total_beads = 0
    total_residues = 0
    for itp in itp_files:
        pdata = parse_itp_file(itp)
        rkeys = list(pdata["residues"].keys())
        itp_info[itp.name] = {
            "num_beads": len(pdata["atoms"]),
            "num_residues": len(pdata["residues"]),
            "first_residue": f"{rkeys[0]} {pdata['residues'][rkeys[0]]['resname']}" if rkeys else None,
            "last_residue": f"{rkeys[-1]} {pdata['residues'][rkeys[-1]]['resname']}" if rkeys else None,
            "elastic_bonds_count": pdata["elastic_bonds_count"]
        }
        total_beads += len(pdata["atoms"])
        total_residues += len(pdata["residues"])
        
    reps_data = {}
    for rep in REPLICATES:
        rep_dir = sys_dir / rep
        prod_xtc = rep_dir / "production.xtc"
        centered_xtc = rep_dir / "production_centered.xtc"
        prod_tpr = rep_dir / "production.tpr"
        prod_log = rep_dir / "production.log"
        
        rep_entry = {
            "exists": rep_dir.exists(),
            "paths": {
                "production_xtc": str(prod_xtc),
                "production_centered_xtc": str(centered_xtc),
                "production_tpr": str(prod_tpr),
                "production_log": str(prod_log)
            },
            "hashes": {},
            "log_params": parse_gmx_log(prod_log),
            "trajectory_info": {}
        }
        
        if centered_xtc.exists():
            rep_entry["hashes"]["production_centered_xtc"] = get_file_hash(centered_xtc)
            try:
                r = XTCReader(str(centered_xtc))
                rep_entry["trajectory_info"] = {
                    "n_atoms": r.n_atoms,
                    "n_frames": len(r),
                    "start_time_ps": float(r[0].time),
                    "end_time_ps": float(r[-1].time),
                    "dt_ps": float(r.trajectory.dt),
                    "dimensions_frame0": [float(x) for x in r[0].dimensions[:3]]
                }
            except Exception as e:
                rep_entry["trajectory_info"]["error"] = str(e)
                
        if prod_xtc.exists():
            rep_entry["hashes"]["production_xtc"] = get_file_hash(prod_xtc)
            
        reps_data[rep] = rep_entry
        
    return {
        "system_name": sysname,
        "topologies": itp_info,
        "total_topology_beads": total_beads,
        "total_topology_residues": total_residues,
        "replicates": reps_data
    }

def main():
    manifest = {
        "environment": {
            "python_version": sys.version,
            "mdanalysis_version": None,
            "os": "Windows 11"
        },
        "systems": {}
    }
    try:
        import MDAnalysis
        manifest["environment"]["mdanalysis_version"] = MDAnalysis.__version__
    except ImportError:
        pass
        
    for sysname in SYSTEMS:
        manifest["systems"][sysname] = inspect_system(sysname)
        
    out_file = OUT_DIR / "simulation_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"[OK] Simulation manifest successfully generated at: {out_file}")
    print(f"     File size: {out_file.stat().st_size / 1024.0:.1f} KB")

if __name__ == "__main__":
    main()
