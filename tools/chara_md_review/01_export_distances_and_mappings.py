#!/usr/bin/env python3
"""
tools/chara_md_review/01_export_distances_and_mappings.py

Extracts and exports:
1. Candidate and target distance arrays (X, Y) for all 4 systems and 3 replicates.
2. Exact pair definitions (candidates.json, targets.json) with residue numbers and restraint status.
3. Residue and backbone bead mappings.
4. Sample raw coordinates (10 frames) for independent recalculation check.
"""

import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

BASE_DIR = Path("E:/Sharon")
DATA_DIR = BASE_DIR / "data"
MD_RUNS_DIR = DATA_DIR / "md_runs"
CG_TOP_DIR = DATA_DIR / "cg_topologies"
OUT_DIR = BASE_DIR / "reports" / "chara_md_review" / "20260907_v1" / "extracted_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS_CONFIG = [
    {"name": "KRAS_G12D", "pdb": "4OBE", "chain_label": "A", "itp": "molecule_0.itp", "n_cand": 1000, "n_targ": 1000, "max_ref_dist": 2.0},
    {"name": "PTPN11", "pdb": "4DGP", "chain_label": "A", "itp": "molecule_0.itp", "n_cand": 1000, "n_targ": 1000, "max_ref_dist": 2.0},
    {"name": "Mut_p53", "pdb": "2J1X", "chain_label": "A", "itp": "molecule_0.itp", "n_cand": 1000, "n_targ": 1000, "max_ref_dist": 2.0},
    {"name": "cMYC_MAX", "pdb": "1NKP", "chain_label": "E", "itp": "molecule_0.itp", "n_cand": 500, "n_targ": 500, "max_ref_dist": 2.0}
]

def parse_itp(itp_path: Path):
    atoms = []; bonds = []; constraints = []
    curr = None
    with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith(";"): continue
            if l.startswith("[") and l.endswith("]"):
                curr = l[1:-1].strip().lower(); continue
            parts = l.split()
            if curr == "atoms" and len(parts) >= 5:
                atoms.append({"nr": int(parts[0]), "type": parts[1], "resnr": int(parts[2]), "resname": parts[3], "atomname": parts[4]})
            elif curr == "bonds" and len(parts) >= 2:
                try: bonds.append((int(parts[0]), int(parts[1])))
                except ValueError: pass
            elif curr == "constraints" and len(parts) >= 2:
                try: constraints.append((int(parts[0]), int(parts[1])))
                except ValueError: pass

    atom_to_res = {a["nr"]: a["resnr"] for a in atoms}
    bb_beads = []
    for idx, a in enumerate(atoms):
        if a["atomname"] == "BB":
            bb_beads.append({"atom_idx": idx, "atom_nr": a["nr"], "resnr": a["resnr"], "resname": a["resname"]})

    restrained_pairs = set()
    for b1, b2 in bonds + constraints:
        if b1 in atom_to_res and b2 in atom_to_res:
            r1, r2 = atom_to_res[b1], atom_to_res[b2]
            if r1 != r2:
                restrained_pairs.add((min(r1, r2), max(r1, r2)))
    return bb_beads, restrained_pairs

def main():
    print("=== Step 1: Exporting Distances, Mappings, and Sample Coordinates ===")
    
    for sys_cfg in SYSTEMS_CONFIG:
        s_name = sys_cfg["name"]
        print(f"\nProcessing system {s_name}...")
        sys_out = OUT_DIR / s_name
        sys_out.mkdir(parents=True, exist_ok=True)

        itp_path = CG_TOP_DIR / s_name / sys_cfg["itp"]
        bb_beads, restrained_pairs = parse_itp(itp_path)
        n_res = len(bb_beads)
        bb_indices = [b["atom_idx"] for b in bb_beads]

        # Read reference frame from rep1
        xtc_rep1 = MD_RUNS_DIR / s_name / "rep1" / "production_centered.xtc"
        r0 = XTCReader(str(xtc_rep1))
        c0 = r0[0].positions[bb_indices] / 10.0
        dist0 = np.linalg.norm(c0[:, None, :] - c0[None, :, :], axis=-1)

        eligible = []
        for i in range(n_res):
            for j in range(i + 4, n_res):
                d = dist0[i, j]
                if d <= sys_cfg["max_ref_dist"]:
                    r_i = bb_beads[i]["resnr"]
                    r_j = bb_beads[j]["resnr"]
                    eligible.append({
                        "i_idx": i, "j_idx": j, "res_i": r_i, "res_j": r_j,
                        "name_i": f"{bb_beads[i]['resname']}{r_i}",
                        "name_j": f"{bb_beads[j]['resname']}{r_j}",
                        "ref_dist_nm": float(d), "is_restrained": bool((r_i, r_j) in restrained_pairs)
                    })

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(eligible))
        n_c = min(sys_cfg["n_cand"], len(eligible) // 2)
        n_t = min(sys_cfg["n_targ"], len(eligible) - n_c)

        candidates = [eligible[k] for k in perm[:n_c]]
        targets = [eligible[k] for k in perm[n_c:n_c + n_t]]

        c_i = np.array([p["i_idx"] for p in candidates])
        c_j = np.array([p["j_idx"] for p in candidates])
        t_i = np.array([p["i_idx"] for p in targets])
        t_j = np.array([p["j_idx"] for p in targets])

        # Save pair definitions JSON
        with open(sys_out / "candidates.json", "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)
        with open(sys_out / "targets.json", "w", encoding="utf-8") as f:
            json.dump(targets, f, indent=2)
        with open(sys_out / "bb_residue_map.json", "w", encoding="utf-8") as f:
            json.dump(bb_beads, f, indent=2)

        print(f"  Saved candidates ({len(candidates)} pairs) and targets ({len(targets)} pairs)")

        # Export distances for rep1, rep2, rep3
        sample_frames_idx = [100, 150, 200, 300, 400, 500, 600, 700, 800, 900]
        sample_coords_dict = {}

        for rep in ["rep1", "rep2", "rep3"]:
            xtc_p = MD_RUNS_DIR / s_name / rep / "production_centered.xtc"
            r = XTCReader(str(xtc_p))
            n_tot = len(r)
            p_mask = np.arange(101, n_tot)
            crds = np.array([ts.positions[bb_indices] for ts in r], dtype=np.float32) / 10.0
            c_d = np.linalg.norm(crds[:, c_i, :] - crds[:, c_j, :], axis=-1)
            t_d = np.linalg.norm(crds[:, t_i, :] - crds[:, t_j, :], axis=-1)

            # Save full extracted arrays
            npz_rep = sys_out / f"distances_{rep}.npz"
            np.savez_compressed(
                npz_rep,
                X=c_d[p_mask].astype(np.float32),
                Y=t_d[p_mask].astype(np.float32),
                frame_indices=p_mask.astype(np.int32),
                timestamps_ps=(p_mask * 500.0).astype(np.float32),
                c_i=c_i.astype(np.int32), c_j=c_j.astype(np.int32),
                t_i=t_i.astype(np.int32), t_j=t_j.astype(np.int32)
            )
            print(f"  Saved {npz_rep.name}: X={c_d[p_mask].shape}, Y={t_d[p_mask].shape}")

            if rep == "rep1":
                # Sample coordinate frames for verification
                valid_sample_idx = [idx for idx in sample_frames_idx if idx < n_tot]
                sampled_crds = crds[valid_sample_idx]
                sample_coords_dict = {
                    "coords_nm": sampled_crds.astype(np.float32),
                    "frame_indices": np.array(valid_sample_idx, dtype=np.int32),
                    "bb_atom_indices": np.array(bb_indices, dtype=np.int32),
                    "resnrs": np.array([b["resnr"] for b in bb_beads], dtype=np.int32),
                    "resnames": np.array([b["resname"] for b in bb_beads]),
                }

        # Save sample coordinates
        npz_samples = sys_out / "sample_coordinates_rep1.npz"
        np.savez_compressed(npz_samples, **sample_coords_dict)
        print(f"  Saved {npz_samples.name} ({len(sample_coords_dict['frame_indices'])} frames)")

    print("\n[OK] All distance arrays, pair mappings, and coordinate samples successfully exported.")

if __name__ == "__main__":
    main()
