#!/usr/bin/env python3
"""
tools/chara_external_confirmation/02_freeze_protocol.py

Freezes the external benchmark evaluation protocol before downloading or evaluating
held-out test outcomes. Outputs:
- reports/chara_external_confirmation/20260908_v1/PROTOCOL_LOCK.json
- reports/chara_external_confirmation/20260908_v1/PROTOCOL_AMENDMENTS.md
Computes and locks SHA-256 hashes.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
    out_dir = base_dir / "reports" / "chara_external_confirmation" / "20260908_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    utc_now = datetime.now(timezone.utc).isoformat()

    print("=== Step 2: Freeze Evaluation Protocol Before External Evaluation ===")

    # 1. PROTOCOL_AMENDMENTS.md
    amendments_md = f"""# PROTOCOL AMENDMENTS: INDEPENDENT EXTERNAL CONFIRMATION BENCHMARK

**Execution Date:** {utc_now}  
**Status:** FROZEN & PRE-REGISTERED  
**Superseded Specification:** `reports/chara_gromacs_resolution/20260908_v1/CONFIRMATION_SPECIFICATION.md` (SHA-256: `e3214acd645205e92f173f41b2ea866599f168054bdbbb657ab7c638caa390b7`)

---

## 1. Rationale for Protocol Amendments

1. **ATLAS Candidate Eligibility Correction**:
   - The prior exploratory document hypothetically listed PDB IDs `1UBQ`, `1AKI`, and `1PGB`. Live API queries against the official ATLAS database (`https://www.dsimb.inserm.fr/ATLAS/api/parsable` and `/ATLAS/metadata/{{pdb_chain}}`) confirm that none of these entries exist in ATLAS (HTTP 404).
   - In accordance with the pre-registered amendment rules, replacement candidates were identified using a deterministic, metadata-only filtering rule applied to the official ATLAS catalog (`2023_03_09_ATLAS_info.tsv`):
     - Monomeric constructs without contacting chains, ligands, ions, or nucleotides (`no_contact == True`).
     - High-resolution crystal structures ($PDB\_resolution \le 1.8$ Å).
     - Standard length range between 60 and 120 residues.
     - Classified as non-redundant in the ATLAS database (`non_redundant_protein == True`).
     - Distinct structural architectures across major secondary structure classes:
       * **Protein 1 (`1utg_A`)**: Uteroglobin (70 residues, resolution 1.34 Å, all-alpha: 70% alpha, 0% beta).
       * **Protein 2 (`2cg7_A`)**: Fibronectin type III domain (90 residues, resolution 1.20 Å, all-beta: 0% alpha, 56% beta).
       * **Protein 3 (`2j6b_A`)**: Viral protein (109 residues, resolution 1.30 Å, mixed alpha+beta: 25% alpha, 26% beta).
       * *(Qualified backup: `1bxy_A`, 50S ribosomal protein L30, 60 residues, 1.90 Å, alpha+beta)*.

2. **Dataset Acquisition Route & Size Management**:
   - Rather than attempting to download full-solvent trajectories (>15 GB), the protocol utilizes the official ATLAS protein-only trajectory route (`/ATLAS/protein/{{pdb_chain}}`), which packages the full 100 ns standardized CHARMM36m atomistic simulations across all 3 independent replicates with solvent removed.
   - Total download size across all 3 proteins is ~601.9 MB, well within system and transfer budgets.

3. **Coordinate Representation & PBC Treatment**:
   - The benchmark extracts atomistic C-alpha coordinates in nanometers (nm) as the canonical backbone representation.
   - All trajectories in ATLAS have periodic boundary condition jumps removed and coordinates fitted to the reference crystal structure. Intramolecular Euclidean pairwise distances are computed directly between C-alpha beads.

4. **Temporal Sampling & Equilibration Discard**:
   - Native ATLAS trajectories contain 10,000 frames per run ($100$ ns at $10$ ps sampling).
   - A fixed 5% initial-time discard (first $5$ ns, frames 0 to 499) is applied for equilibration.
   - The remaining 9,500 frames are sampled with an extraction stride of 10, producing exactly 950 evaluated frames per run (2,850 frames per protein across 3 replicates).
"""

    amendments_path = out_dir / "PROTOCOL_AMENDMENTS.md"
    amendments_path.write_text(amendments_md)
    amendments_hash = sha256_file(amendments_path)
    print(f"Saved {amendments_path.name} (SHA-256: {amendments_hash})")

    # 2. PROTOCOL_LOCK.json
    protocol_lock = {
        "protocol_name": "CHARA_INDEPENDENT_EXTERNAL_BENCHMARK_PROTOCOL",
        "freeze_timestamp_utc": utc_now,
        "protocol_status": "FROZEN_PRE_EVALUATION",
        "amendments_file": str(amendments_path.name),
        "amendments_sha256": amendments_hash,
        "source_code_hash": sha256_file(Path(__file__).resolve()),
        "target_proteins": [
            {
                "protein_id": "1utg_A",
                "protein_name": "Uteroglobin",
                "organism": "Oryctolagus cuniculus",
                "length": 70,
                "structural_class": "all-alpha (70% alpha, 0% beta)",
                "pdb_resolution_angstrom": 1.34,
                "simulation_engine": "GROMACS",
                "force_field": "CHARMM36m",
                "temperature_kelvin": 300,
                "pressure_bar": 1.0,
                "replicates": ["R1", "R2", "R3"],
                "total_run_duration_ns": 100,
                "frames_per_run_native": 10000,
                "stride": 10,
                "equilibration_discard_pct": 5.0,
                "evaluated_frames_per_run": 950
            },
            {
                "protein_id": "2cg7_A",
                "protein_name": "Fibronectin type III domain",
                "organism": "Homo sapiens",
                "length": 90,
                "structural_class": "all-beta (0% alpha, 56% beta)",
                "pdb_resolution_angstrom": 1.20,
                "simulation_engine": "GROMACS",
                "force_field": "CHARMM36m",
                "temperature_kelvin": 300,
                "pressure_bar": 1.0,
                "replicates": ["R1", "R2", "R3"],
                "total_run_duration_ns": 100,
                "frames_per_run_native": 10000,
                "stride": 10,
                "equilibration_discard_pct": 5.0,
                "evaluated_frames_per_run": 950
            },
            {
                "protein_id": "2j6b_A",
                "protein_name": "Viral protein",
                "organism": "Acidianus filamentous virus 1",
                "length": 109,
                "structural_class": "alpha+beta (25% alpha, 26% beta)",
                "pdb_resolution_angstrom": 1.30,
                "simulation_engine": "GROMACS",
                "force_field": "CHARMM36m",
                "temperature_kelvin": 300,
                "pressure_bar": 1.0,
                "replicates": ["R1", "R2", "R3"],
                "total_run_duration_ns": 100,
                "frames_per_run_native": 10000,
                "stride": 10,
                "equilibration_discard_pct": 5.0,
                "evaluated_frames_per_run": 950
            }
        ],
        "backup_protein": {
            "protein_id": "1bxy_A",
            "protein_name": "50S ribosomal protein L30",
            "length": 60,
            "structural_class": "alpha+beta (32% alpha, 27% beta)"
        },
        "evaluation_design": {
            "folds": [
                {"fold": 1, "test_replicate": "R1", "dev_replicates": ["R2", "R3"]},
                {"fold": 2, "test_replicate": "R2", "dev_replicates": ["R1", "R3"]},
                {"fold": 3, "test_replicate": "R3", "dev_replicates": ["R1", "R2"]}
            ],
            "coordinate_representation": "C-alpha atomistic backbone coordinates in nanometers (nm)",
            "pair_filtering": "Sequence separation |i - j| >= 3",
            "max_candidate_budget": 500,
            "max_target_budget": 500,
            "candidate_selection_rule": "Deterministic proximity ranking on development reference structure (|i-j| >= 3)",
            "target_selection_rule": "Deterministic development distance band [0.5 nm, 2.0 nm], disjoint from candidates",
            "disjoint_guarantee": "Candidate pairs intersect Target pairs == empty set",
            "primary_sensor_budget_k": 10,
            "secondary_sensor_budgets_k": [5, 20],
            "inner_tuning_division": {
                "train_pct": 60.0,
                "guard_pct": 10.0,
                "val_pct": 30.0,
                "ridge_lambda_grid": [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]
            }
        },
        "required_methods": [
            "M1_DEVELOPMENT_CONSTANT_MEAN",
            "M2_UNIFORM_RANDOM_SEEDS_1_TO_20",
            "M3_DEVELOPMENT_WITHIN_RUN_VARIANCE",
            "M4_PCA_PIVOTED_QR",
            "M7_MEAN_TRANSFER_SELECTION",
            "M8_ROBUST_TRANSFER_SELECTION"
        ],
        "diagnostics_and_controls": {
            "moment_decomposition": "MSE = Bias^2 + Var_res (population variance)",
            "test_centered_skill": "1 - sum[(Y - Ybar_test) - (pred - predbar_test)]^2 / sum[(Y - Ybar_test)^2]",
            "non_circular_temporal_lags": [0, 10, 25, 50, 100],
            "paired_block_bootstrap": {
                "n_bootstraps": 1000,
                "block_length_frames": 20,
                "seed": 42
            },
            "primary_hypothesis_test": "M8_ROBUST_TRANSFER minus M4_PCA_PIVOTED_QR at k=10",
            "practical_superiority_threshold_delta_skill": 0.05
        }
    }

    protocol_lock_path = out_dir / "PROTOCOL_LOCK.json"
    protocol_lock_path.write_text(json.dumps(protocol_lock, indent=2))
    protocol_lock_hash = sha256_file(protocol_lock_path)
    print(f"Saved {protocol_lock_path.name} (SHA-256: {protocol_lock_hash})")
    print(f"[OK] Protocol frozen before external data evaluation.")

if __name__ == "__main__":
    main()
