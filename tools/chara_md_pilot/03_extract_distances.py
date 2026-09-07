#!/usr/bin/env python3
"""
tools/chara_md_pilot/03_extract_distances.py

Extracts, validates, and caches residue-pair distances for KRAS_G12D (Chain A, 170 residues)
across three independent replicates (rep1, rep2, rep3).
Generates disjoint candidate (C) and target (T) sets of 1,000 pairs each.
Saves:
  - reports/chara_md_pilot/20260907_v1/evidence/pair_definitions.json
  - reports/chara_md_pilot/20260907_v1/evidence/distances_rep1.npz
  - reports/chara_md_pilot/20260907_v1/evidence/distances_rep2.npz
  - reports/chara_md_pilot/20260907_v1/evidence/distances_rep3.npz
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

BASE_DIR = Path("E:/Sharon")
DATA_DIR = BASE_DIR / "data"
EVID_DIR = BASE_DIR / "reports" / "chara_md_pilot" / "20260907_v1" / "evidence"
EVID_DIR.mkdir(parents=True, exist_ok=True)

ITP_PATH = DATA_DIR / "cg_topologies" / "KRAS_G12D" / "molecule_0.itp"
REPLICATES = ["rep1", "rep2", "rep3"]
SPLIT_SEED = 42
N_CANDIDATES = 1000
N_TARGETS = 1000
MAX_REF_DIST_NM = 2.0  # 20.0 Angstroms
MIN_SEQ_SEP = 4

def parse_itp_atoms_and_bonds(itp_path: Path):
    atoms = []
    bonds = []
    with open(itp_path, "r", encoding="utf-8", errors="replace") as f:
        curr = None
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                curr = line[1:-1].strip().lower()
                continue
            parts = line.split()
            if curr == "atoms" and len(parts) >= 5:
                # nr type resnr residue atom cgnr charge mass
                atoms.append({
                    "nr": int(parts[0]),
                    "resnr": int(parts[2]),
                    "resname": parts[3],
                    "atomname": parts[4]
                })
            elif curr in ("bonds", "constraints") and len(parts) >= 2:
                try:
                    bonds.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass

    # Map atom index to residue index and identify BB atom index
    atom_to_res = {a["nr"]: a["resnr"] for a in atoms}
    bb_beads = []
    for idx, a in enumerate(atoms):
        if a["atomname"] == "BB":
            bb_beads.append({
                "atom_idx": idx,
                "atom_nr": a["nr"],
                "resnr": a["resnr"],
                "resname": a["resname"]
            })
            
    # Find all residue pairs directly connected by bonds
    restrained_pairs = set()
    for b1, b2 in bonds:
        if b1 in atom_to_res and b2 in atom_to_res:
            r1, r2 = atom_to_res[b1], atom_to_res[b2]
            if r1 != r2:
                pair = (min(r1, r2), max(r1, r2))
                restrained_pairs.add(pair)
                
    return bb_beads, restrained_pairs

def main():
    print("[INFO] Parsing topology and backbone beads from molecule_0.itp...")
    bb_beads, restrained_pairs = parse_itp_atoms_and_bonds(ITP_PATH)
    n_res = len(bb_beads)
    print(f"       Found {n_res} residues with BB beads (Residues {bb_beads[0]['resnr']} to {bb_beads[-1]['resnr']}).")
    print(f"       Found {len(restrained_pairs)} unique residue pairs with elastic network restraints.")

    # Load reference frame (rep1 frame 0)
    ref_xtc = DATA_DIR / "md_runs" / "KRAS_G12D" / "rep1" / "production_centered.xtc"
    r_ref = XTCReader(str(ref_xtc))
    bb_indices = [b["atom_idx"] for b in bb_beads]
    pos0 = r_ref[0].positions[bb_indices] / 10.0  # Convert Angstroms to nm
    
    # Calculate all pair distances at frame 0
    diff0 = pos0[:, None, :] - pos0[None, :, :]
    dist0 = np.linalg.norm(diff0, axis=-1)
    
    # Filter eligible pairs: non-local (|i - j| >= MIN_SEQ_SEP) and dist0 <= MAX_REF_DIST_NM
    eligible_pairs = []
    for i in range(n_res):
        for j in range(i + MIN_SEQ_SEP, n_res):
            d = dist0[i, j]
            if d <= MAX_REF_DIST_NM:
                r_i = bb_beads[i]["resnr"]
                r_j = bb_beads[j]["resnr"]
                is_rest = (r_i, r_j) in restrained_pairs
                eligible_pairs.append({
                    "i_idx": i,
                    "j_idx": j,
                    "res_i": r_i,
                    "res_j": r_j,
                    "name_i": f"{bb_beads[i]['resname']}{r_i}",
                    "name_j": f"{bb_beads[j]['resname']}{r_j}",
                    "ref_dist_nm": float(d),
                    "is_restrained": bool(is_rest)
                })
                
    print(f"[INFO] Eligible non-local pairs (sep >= {MIN_SEQ_SEP}, ref_dist <= {MAX_REF_DIST_NM} nm): {len(eligible_pairs)}")
    
    # Deterministic split into Candidates (C) and Targets (T)
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(eligible_pairs))
    
    c_indices = perm[:N_CANDIDATES]
    t_indices = perm[N_CANDIDATES:N_CANDIDATES + N_TARGETS]
    
    candidates = [eligible_pairs[k] for k in c_indices]
    targets = [eligible_pairs[k] for k in t_indices]
    
    c_set = set((p["res_i"], p["res_j"]) for p in candidates)
    t_set = set((p["res_i"], p["res_j"]) for p in targets)
    overlap = c_set.intersection(t_set)
    assert len(overlap) == 0, f"Disjoint split failed: {len(overlap)} overlapping pairs!"
    
    c_restrained = sum(1 for p in candidates if p["is_restrained"])
    t_restrained = sum(1 for p in targets if p["is_restrained"])
    print(f"[OK] Split verified: Candidates C={len(candidates)} ({c_restrained} restrained), Targets T={len(targets)} ({t_restrained} restrained).")
    print(f"     Disjointness check passed: |C cap T| = {len(overlap)}.")
    
    # Save pair definitions
    pair_defs = {
        "metadata": {
            "system": "KRAS_G12D_ChainA",
            "residues": n_res,
            "min_seq_sep": MIN_SEQ_SEP,
            "max_ref_dist_nm": MAX_REF_DIST_NM,
            "split_seed": SPLIT_SEED,
            "candidates_count": len(candidates),
            "targets_count": len(targets)
        },
        "candidates": candidates,
        "targets": targets
    }
    with open(EVID_DIR / "pair_definitions.json", "w", encoding="utf-8") as f:
        json.dump(pair_defs, f, indent=2)
        
    # Index arrays for fast vectorized distance computation
    c_i = np.array([p["i_idx"] for p in candidates], dtype=int)
    c_j = np.array([p["j_idx"] for p in candidates], dtype=int)
    t_i = np.array([p["i_idx"] for p in targets], dtype=int)
    t_j = np.array([p["j_idx"] for p in targets], dtype=int)
    
    # Extract distances across replicates
    for rep in REPLICATES:
        t0 = time.time()
        xtc_path = DATA_DIR / "md_runs" / "KRAS_G12D" / rep / "production_centered.xtc"
        r = XTCReader(str(xtc_path))
        n_tot_frames = len(r)
        
        # Load all coordinates for Chain A BB beads: shape (N_frames, 170, 3) in nm
        frames_nm = np.array([ts.positions[bb_indices] for ts in r], dtype=np.float32) / 10.0
        times_ps = np.array([ts.time for ts in r], dtype=np.float32)
        
        # Primary analysis: discard first 10% (frames 0..100), keep 101..1000 (901 frames)
        primary_mask = np.arange(101, n_tot_frames)
        # Sensitivity analysis: discard first 20% (frames 0..200), keep 201..1000 (801 frames)
        sens_mask = np.arange(201, n_tot_frames)
        
        # Compute candidate distances: shape (N_frames, 1000)
        c_diff = frames_nm[:, c_i, :] - frames_nm[:, c_j, :]
        c_dists = np.linalg.norm(c_diff, axis=-1).astype(np.float32)
        
        # Compute target distances: shape (N_frames, 1000)
        t_diff = frames_nm[:, t_i, :] - frames_nm[:, t_j, :]
        t_dists = np.linalg.norm(t_diff, axis=-1).astype(np.float32)
        
        # Save NPZ
        out_npz = EVID_DIR / f"distances_{rep}.npz"
        np.savez_compressed(
            out_npz,
            times_ps=times_ps,
            primary_mask=primary_mask,
            sens_mask=sens_mask,
            candidate_dists=c_dists,
            target_dists=t_dists
        )
        
        c_var = np.var(c_dists[primary_mask], axis=0)
        t_var = np.var(t_dists[primary_mask], axis=0)
        print(f"[OK] Processed {rep} in {time.time() - t0:.2f}s:")
        print(f"     Saved {out_npz.name} ({out_npz.stat().st_size / 1024 / 1024:.2f} MB)")
        print(f"     Candidate dists: min={c_dists[primary_mask].min():.3f} nm, max={c_dists[primary_mask].max():.3f} nm, mean variance={c_var.mean():.6f}")
        print(f"     Target dists: min={t_dists[primary_mask].min():.3f} nm, max={t_dists[primary_mask].max():.3f} nm, mean variance={t_var.mean():.6f}")
        print(f"     Near-constant candidates (var < 1e-6): {(c_var < 1e-6).sum()}, near-constant targets: {(t_var < 1e-6).sum()}")

if __name__ == "__main__":
    main()
