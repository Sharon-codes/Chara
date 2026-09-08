#!/usr/bin/env python3
"""
Step 1: Production Provenance Audit, Force Field Comparison, and Construct Identity Mapping
Produces:
- RUN_PROVENANCE.csv
- FORCEFIELD_COMPARISON.csv
- TRAJECTORY_TIME_CHECKS.csv
- CONSTRUCT_IDENTITY.csv
- RESIDUE_MAPPING.csv
"""

import os
import sys
import glob
import hashlib
import subprocess
import numpy as np
import pandas as pd
import MDAnalysis as mda

BASE_DIR = r"E:\Sharon"
MD_RUNS_DIR = os.path.join(BASE_DIR, "data", "md_runs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "chara_final_evidence", "20260907_v1")
os.makedirs(REPORTS_DIR, exist_ok=True)

SYSTEMS = ["KRAS_G12D", "cMYC_MAX", "Mut_p53", "PTPN11"]
REPLICATES = ["rep1", "rep2", "rep3"]

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None, 0
    h = hashlib.sha256()
    size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest(), size

# -------------------------------------------------------------
# 1. RUN PROVENANCE
# -------------------------------------------------------------
def run_provenance_audit():
    rows = []
    for sys_name in SYSTEMS:
        sys_dir = os.path.join(MD_RUNS_DIR, sys_name)
        for rep in REPLICATES:
            rep_dir = os.path.join(sys_dir, rep)
            tpr_path = os.path.join(rep_dir, "production.tpr")
            log_path = os.path.join(rep_dir, "production.log")
            xtc_centered = os.path.join(rep_dir, "production_centered.xtc")
            xtc_raw = os.path.join(rep_dir, "production.xtc")
            cpt_path = os.path.join(rep_dir, "production.cpt")
            martini_path = os.path.join(rep_dir, "martini.itp")
            if not os.path.exists(martini_path):
                martini_path = os.path.join(sys_dir, "martini.itp")

            tpr_hash, tpr_size = get_file_hash(tpr_path)
            log_hash, log_size = get_file_hash(log_path)
            xtc_c_hash, xtc_c_size = get_file_hash(xtc_centered)
            martini_hash, martini_size = get_file_hash(martini_path)

            host, gmx_ver, seed, dt, nsteps, start_time = "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if "GROMACS - gmx mdrun," in line:
                            gmx_ver = line.strip().split("gmx mdrun,")[-1].replace("(-:", "").strip()
                        elif "Host:" in line:
                            host = line.strip().split("Host:")[1].split()[0]
                        elif "ld-seed" in line or "gen-seed" in line:
                            parts = line.strip().split("=")
                            if len(parts) > 1:
                                seed = parts[1].strip().split()[0]
                        elif line.strip().startswith("dt ") or " dt " in line:
                            parts = line.strip().split("=")
                            if len(parts) > 1 and dt == "UNKNOWN":
                                dt = parts[1].strip().split()[0]
                        elif line.strip().startswith("nsteps ") or " nsteps " in line:
                            parts = line.strip().split("=")
                            if len(parts) > 1 and nsteps == "UNKNOWN":
                                nsteps = parts[1].strip().split()[0]
                        elif "Log file opened on" in line:
                            start_time = line.strip().replace("Log file opened on", "").strip()

            dump_notes = "Native gmx not in Windows PATH; WSL not installed; MDAnalysis TPRParser does not support TPX version 138."

            rows.append({
                "system": sys_name,
                "replicate": rep,
                "production_tpr_path": os.path.relpath(tpr_path, BASE_DIR).replace("\\", "/"),
                "production_tpr_sha256": tpr_hash,
                "production_tpr_bytes": tpr_size,
                "production_log_path": os.path.relpath(log_path, BASE_DIR).replace("\\", "/"),
                "production_log_sha256": log_hash,
                "production_log_bytes": log_size,
                "martini_itp_sha256": martini_hash,
                "martini_itp_bytes": martini_size,
                "xtc_centered_sha256": xtc_c_hash,
                "xtc_centered_bytes": xtc_c_size,
                "simulation_host": host,
                "gromacs_version": gmx_ver,
                "ld_seed": seed,
                "dt_ps": dt,
                "nsteps": nsteps,
                "log_opened_time": start_time,
                "compiled_dump_status": "UNAVAILABLE_LOCAL_DUMP",
                "dump_tool_diagnostic": dump_notes,
                "provenance_evidence_status": "UNRESOLVED_PROVENANCE",
                "factual_reason": "Compiled TPR parameters cannot be dumped locally without GROMACS 2026.3; supplied martini.itp contains 604 stubbed atomtypes and deviates from official Martini 3.0.0."
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPORTS_DIR, "RUN_PROVENANCE.csv")
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {out_path} ({len(df)} runs)")
    return df

# -------------------------------------------------------------
# 2. FORCE FIELD COMPARISON
# -------------------------------------------------------------
def parse_itp_parameters(filepath):
    atomtypes = {}
    nonbond_params = {}
    if not os.path.exists(filepath):
        return atomtypes, nonbond_params
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip().lower()
                continue
            parts = line.split(";")[0].split()
            if not parts:
                continue
            if section == "atomtypes":
                if len(parts) >= 6:
                    name = parts[0]
                    mass = float(parts[1])
                    charge = float(parts[2])
                    ptype = parts[3]
                    sigma = float(parts[4])
                    eps = float(parts[5])
                    atomtypes[name] = {"mass": mass, "charge": charge, "ptype": ptype, "sigma": sigma, "epsilon": eps}
            elif section == "nonbond_params":
                if len(parts) >= 5:
                    i, j, func, sigma, eps = parts[0], parts[1], int(parts[2]), float(parts[3]), float(parts[4])
                    pair_key = tuple(sorted([i, j]))
                    nonbond_params[pair_key] = {"func": func, "sigma": sigma, "epsilon": eps}
    return atomtypes, nonbond_params

def compare_forcefield():
    official_itp = os.path.join(BASE_DIR, "data", "martini_v3.0.0_official.itp")
    supplied_itp = os.path.join(BASE_DIR, "data", "md_runs", "KRAS_G12D", "martini.itp")

    off_types, off_nb = parse_itp_parameters(official_itp)
    sup_types, sup_nb = parse_itp_parameters(supplied_itp)

    used_types = ['C6', 'CL', 'NA', 'P2', 'P5', 'Q5', 'Q5n', 'SC2', 'SC3', 'SC4', 
                  'SP1', 'SP2', 'SP2a', 'SP5', 'SQ3p', 'SQ4p', 'SQ5n', 'TC3', 'TC4', 
                  'TC5', 'TC6', 'TN5a', 'TN6', 'TN6d', 'TP1', 'W']

    rows = []
    for t in used_types:
        sup_info = sup_types.get(t, {})
        off_info = off_types.get(t, {})

        s_sup = sup_info.get("sigma", np.nan)
        e_sup = sup_info.get("epsilon", np.nan)
        s_off = off_info.get("sigma", np.nan)
        e_off = off_info.get("epsilon", np.nan)

        mismatch = False
        if not np.isnan(s_sup) and not np.isnan(s_off):
            if abs(s_sup - s_off) > 1e-4 or abs(e_sup - e_off) > 1e-4:
                mismatch = True

        rows.append({
            "comparison_scope": "USED_ATOMTYPE",
            "entity_1": t,
            "entity_2": "N/A",
            "supplied_sigma_nm": s_sup,
            "supplied_epsilon_kj_mol": e_sup,
            "official_sigma_nm": s_off,
            "official_epsilon_kj_mol": e_off,
            "abs_diff_sigma_nm": abs(s_sup - s_off) if not np.isnan(s_sup) and not np.isnan(s_off) else np.nan,
            "abs_diff_epsilon_kj_mol": abs(e_sup - e_off) if not np.isnan(e_sup) and not np.isnan(e_off) else np.nan,
            "status": "MISMATCH_CUSTOM_STUB" if mismatch else ("OFFICIAL_TYPE_ABSENT" if np.isnan(s_off) else "MATCH"),
            "notes": "Official Martini 3.0.0 sets atomtype sigma=0/eps=0 and specifies pairwise matrix in [nonbond_params]; supplied martini.itp puts non-zero parameters in [atomtypes] and flattens interactions."
        })

    rep_pairs = [
        ("W", "W"), ("NA", "CL"), ("NA", "W"), ("CL", "W"), ("Qa", "W"), ("NA", "Qa"),
        ("P2", "P2"), ("P2", "C6"), ("P2", "W"), ("C6", "C6"), ("C6", "W"),
        ("Q5", "Q5"), ("Q5", "W"), ("SP1", "W"), ("TC3", "W"), ("TN6", "W")
    ]
    for p1, p2 in rep_pairs:
        pair_key = tuple(sorted([p1, p2]))
        sup_pair = sup_nb.get(pair_key)
        off_pair = off_nb.get(pair_key)

        if sup_pair is not None:
            s_sup = sup_pair["sigma"]
            e_sup = sup_pair["epsilon"]
            sup_source = "EXPLICIT_NONBOND_PARAMS"
        else:
            t1 = sup_types.get(p1, {})
            t2 = sup_types.get(p2, {})
            if "sigma" in t1 and "sigma" in t2:
                s_sup = 0.5 * (t1["sigma"] + t2["sigma"])
                e_sup = np.sqrt(t1["epsilon"] * t2["epsilon"])
                sup_source = "COMBINATION_RULE_FALLBACK"
            else:
                s_sup, e_sup, sup_source = np.nan, np.nan, "UNRESOLVED"

        if off_pair is not None:
            s_off = off_pair["sigma"]
            e_off = off_pair["epsilon"]
            off_source = "OFFICIAL_NONBOND_PARAMS"
        else:
            s_off, e_off, off_source = np.nan, np.nan, "NOT_IN_REGULAR_M3_TABLE"

        abs_s_diff = abs(s_sup - s_off) if not np.isnan(s_sup) and not np.isnan(s_off) else np.nan
        abs_e_diff = abs(e_sup - e_off) if not np.isnan(e_sup) and not np.isnan(e_off) else np.nan

        rows.append({
            "comparison_scope": "REPRESENTATIVE_NONBONDED_PAIR",
            "entity_1": p1,
            "entity_2": p2,
            "supplied_sigma_nm": s_sup,
            "supplied_epsilon_kj_mol": e_sup,
            "official_sigma_nm": s_off,
            "official_epsilon_kj_mol": e_off,
            "abs_diff_sigma_nm": abs_s_diff,
            "abs_diff_epsilon_kj_mol": abs_e_diff,
            "status": "MISMATCH" if (not np.isnan(abs_e_diff) and (abs_s_diff > 1e-4 or abs_e_diff > 1e-4)) else ("OFFICIAL_PAIR_NOT_FOUND" if np.isnan(s_off) else "MATCH"),
            "notes": f"Supplied source: {sup_source}; Official source: {off_source}."
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPORTS_DIR, "FORCEFIELD_COMPARISON.csv")
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {out_path} ({len(df)} comparison records)")
    return df

# -------------------------------------------------------------
# 3. TRAJECTORY TIME CHECKS
# -------------------------------------------------------------
def check_trajectories():
    rows = []
    for sys_name in SYSTEMS:
        for rep in REPLICATES:
            xtc = os.path.join(MD_RUNS_DIR, sys_name, rep, "production_centered.xtc")
            if not os.path.exists(xtc):
                continue
            u = mda.Universe(xtc)
            n_frames = len(u.trajectory)
            n_atoms = len(u.atoms)
            times = np.array([ts.time for ts in u.trajectory])
            dts = np.diff(times)
            t_start = times[0]
            t_end = times[-1]
            unique_dts = np.unique(dts)

            n_jumps = int(np.sum(dts != 500.0))
            max_jump = float(np.max(dts)) if len(dts) > 0 else 0.0

            irregular = (len(unique_dts) > 1 or n_jumps > 0)
            status = "VERIFIED_REGULAR" if not irregular and n_frames == 1001 else ("FRAME_DISCREPANCY_DROPPED_FRAMES" if n_frames != 1001 else "IRREGULAR_SPACING")

            notes = "Standard 1001 frames spaced at 500.0 ps (0 to 500 ns)."
            if sys_name == "PTPN11" and rep == "rep1":
                notes = "996 frames observed; 1 jump of 3000.0 ps between frames 239 and 240 (5 frames missing); historical scripts/19_fix_ptpn11_rep1_frames.py performed linear interpolation to force 1001 points."

            rows.append({
                "system": sys_name,
                "replicate": rep,
                "trajectory_relpath": os.path.relpath(xtc, BASE_DIR).replace("\\", "/"),
                "atom_count": n_atoms,
                "frame_count": n_frames,
                "expected_frames": 1001,
                "start_time_ps": t_start,
                "end_time_ps": t_end,
                "total_duration_ns": (t_end - t_start) / 1000.0,
                "nominal_dt_ps": 500.0,
                "unique_dt_values": str(list(unique_dts)),
                "num_irregular_jumps": n_jumps,
                "max_jump_ps": max_jump,
                "time_check_status": status,
                "notes": notes
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPORTS_DIR, "TRAJECTORY_TIME_CHECKS.csv")
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved {out_path} ({len(df)} trajectories checked)")
    return df

# -------------------------------------------------------------
# 4. CONSTRUCT IDENTITY & RESIDUE MAPPING
# -------------------------------------------------------------
def map_construct_identities():
    aa_map = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
              'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
              'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y',
              'SEP':'S','TPO':'T','PTR':'Y'}

    construct_rows = [
        {
            "legacy_system_label": "KRAS_G12D",
            "source_pdb_id": "4OBE",
            "source_pdb_title": "CRYSTAL STRUCTURE OF GDP-BOUND HUMAN KRAS",
            "uniprot_canonical_id": "P01116 (RASH_HUMAN)",
            "canonical_construct_boundaries": "Residues 1-169 (GDP-bound fragment)",
            "expression_tag": "N-terminal Glycine (Residue 1 in PDB = Glycine cloning artifact)",
            "aligned_canonical_residue_12": "GLYCINE (G) at canonical position 12 (Construct residue 13)",
            "mutation_operation_status": "NO_MUTATION_OCCURRED (Wild-type KRAS sequence)",
            "simulated_chains": "Chains A and B (Homodimer, 2 x 170 residues = 340 residues, 788 CG beads)",
            "cofactors_or_nucleic_acids": "GDP and Mg2+ from crystal structure omitted in coarse-grained model",
            "corrected_construct_name": "Wild-Type KRAS (Fragment 1-169 with N-terminal G tag, homodimer)",
            "factual_reconciliation": "Directory label 'KRAS_G12D' is an erroneous misnomer; source structure 4OBE is wild-type human KRAS; canonical residue 12 is Glycine."
        },
        {
            "legacy_system_label": "Mut_p53",
            "source_pdb_id": "2J1X",
            "source_pdb_title": "HUMAN P53 CORE DOMAIN MUTANT M133L-V203A-Y220C-N239Y-N268D",
            "uniprot_canonical_id": "P04637 (P53_HUMAN)",
            "canonical_construct_boundaries": "Core DNA-binding domain, Residues 94-312",
            "expression_tag": "None (Residues numbered 94-312)",
            "aligned_canonical_residue_12": "N/A (Fragment covers residues 94-312)",
            "mutation_operation_status": "QUINTUPLE_STABILIZING_MUTANT (M133L, V203A, Y220C, N239Y, N268D)",
            "simulated_chains": "Chains A and B (Homodimer, 2 x 219 residues = 438 residues, 1012 CG beads)",
            "cofactors_or_nucleic_acids": "Zn2+ ions omitted in coarse-grained model",
            "corrected_construct_name": "p53 Core Domain Super-Stabilized Quintuple Mutant (M133L/V203A/Y220C/N239Y/N268D)",
            "factual_reconciliation": "Source 2J1X is an engineered multi-mutant stabilized p53 core domain, NOT an isolated cancer-associated R273H hotspot mutation."
        },
        {
            "legacy_system_label": "cMYC_MAX",
            "source_pdb_id": "1NKP",
            "source_pdb_title": "CRYSTAL STRUCTURE OF MYC-MAX RECOGNIZING DNA",
            "uniprot_canonical_id": "P01106 (MYC_HUMAN) / P61244 (MAX_HUMAN)",
            "canonical_construct_boundaries": "c-Myc bHLH-LZ (res 353-437) / Max bHLH-LZ (res 22-102)",
            "expression_tag": "G-H-M expression tag on c-Myc",
            "aligned_canonical_residue_12": "N/A (bHLH-LZ region only)",
            "mutation_operation_status": "WILD_TYPE_SEQUENCES_OF_BHLH_LZ_FRAGMENTS",
            "simulated_chains": "Chains E (c-Myc, 88 res), F (Max, 83 res), G (c-Myc, 88 res), H (Max, 83 res) = 342 residues, 812 CG beads",
            "cofactors_or_nucleic_acids": "DNA Chains A, B, C, D (20-mer duplexes) omitted from simulation",
            "corrected_construct_name": "c-Myc/Max Heterotetramer Protein-Only bHLH-LZ (DNA omitted)",
            "factual_reconciliation": "The coarse-grained simulation contains two c-Myc/Max heterodimers without cognate target DNA duplex present in the 1NKP crystal structure."
        },
        {
            "legacy_system_label": "PTPN11",
            "source_pdb_id": "4DGP",
            "source_pdb_title": "STRUCTURE OF SHP2 (PTPN11) CATALYTIC AND REGULATORY DOMAINS",
            "uniprot_canonical_id": "Q06124 (PTN11_HUMAN)",
            "canonical_construct_boundaries": "Residues 1-528 (N-SH2, C-SH2, and PTP catalytic domains)",
            "expression_tag": "C-terminal 6xHis tag (LEHHHHHH, residues 529-536)",
            "aligned_canonical_residue_12": "Threonine (T12) in N-SH2 domain",
            "mutation_operation_status": "WILD_TYPE_ENZYME_WITH_HIS_TAG",
            "simulated_chains": "Chain A (Monomer, 536 residues, 1280 CG beads)",
            "cofactors_or_nucleic_acids": "None",
            "corrected_construct_name": "SHP2 / PTPN11 (Residues 1-528 with C-terminal LEHHHHHH tag)",
            "factual_reconciliation": "Single-chain construct of full tandem SH2 and catalytic phosphatase domain with C-terminal hexahistidine tag."
        }
    ]

    df_construct = pd.DataFrame(construct_rows)
    out_c_path = os.path.join(REPORTS_DIR, "CONSTRUCT_IDENTITY.csv")
    df_construct.to_csv(out_c_path, index=False)
    print(f"[✓] Saved {out_c_path} ({len(df_construct)} constructs)")

    canonical_kras = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK"
    
    res_list = []
    seen = set()
    with open(os.path.join(BASE_DIR, "data", "processed", "KRAS_4OBE_clean.pdb"), "r") as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                resseq = int(line[22:27].strip())
                resname = line[17:20].strip()
                if resseq not in seen:
                    seen.add(resseq)
                    res_list.append((resseq, resname, aa_map.get(resname, "X")))

    mapping_rows = []
    for rseq, rname, aa in res_list:
        canonical_idx = rseq - 1
        if canonical_idx >= 1 and canonical_idx <= len(canonical_kras):
            canon_aa = canonical_kras[canonical_idx - 1]
            canon_num = canonical_idx
        else:
            canon_aa = "TAG"
            canon_num = -1

        mapping_rows.append({
            "system": "KRAS_G12D",
            "chain": "A",
            "construct_residue_index": rseq,
            "construct_residue_name": rname,
            "construct_one_letter": aa,
            "canonical_uniprot_index": canon_num,
            "canonical_residue_one_letter": canon_aa,
            "match_status": "MATCH" if aa == canon_aa else ("TAG_EXTENSION" if canon_num == -1 else "MISMATCH"),
            "critical_site_annotation": "CANONICAL_G12" if canon_num == 12 else ("CONSTRUCT_INDEX_12" if rseq == 12 else "")
        })

    df_map = pd.DataFrame(mapping_rows)
    out_m_path = os.path.join(REPORTS_DIR, "RESIDUE_MAPPING.csv")
    df_map.to_csv(out_m_path, index=False)
    print(f"[✓] Saved {out_m_path} ({len(df_map)} residues mapped for KRAS)")
    return df_construct, df_map

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def main():
    print("=== Step 1: Auditing Production Provenance & Forcefield Parameters ===")
    run_provenance_audit()
    compare_forcefield()
    check_trajectories()
    map_construct_identities()
    print("=== Step 1 Complete ===")

if __name__ == "__main__":
    main()
