#!/usr/bin/env python3
"""
Phase 1: Protein Structure Curation & PTM Preprocessing
Automated molecular dynamics pipeline preprocessing script.
Target Oncogenic Proteins:
1. KRAS G12D (PDB: 4OBE) - Target PTM: SEP at residue 181
2. c-MYC/MAX (PDB: 1NKP) - Target PTM: TPO at residue 58
3. Mutant p53 (PDB: 2J1X) - Target PTM: SEP at residue 392
4. PTPN11/SHP2 (PDB: 4DGP) - Target PTM: PTR at residue 542
"""

import os
import sys
import logging
import requests
import pdbfixer
from openmm.app import PDBFile

# Define project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for directory in [RAW_DIR, PROCESSED_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure Logging
log_file = os.path.join(LOG_DIR, "phase1_prep.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

TARGET_PTMS = {"SEP", "TPO", "PTR"}

TARGETS = [
    {
        "name": "KRAS",
        "pdb_id": "4OBE",
        "ptm_res": "SEP",
        "ptm_num": 181,
    },
    {
        "name": "c-MYC",
        "pdb_id": "1NKP",
        "ptm_res": "TPO",
        "ptm_num": 58,
    },
    {
        "name": "p53",
        "pdb_id": "2J1X",
        "ptm_res": "SEP",
        "ptm_num": 392,
    },
    {
        "name": "PTPN11",
        "pdb_id": "4DGP",
        "ptm_res": "PTR",
        "ptm_num": 542,
    },
]


def download_pdb(pdb_id: str, dest_path: str) -> str:
    """Download raw PDB structure from RCSB REST API."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logging.info(f"Downloading PDB {pdb_id} from {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download PDB {pdb_id}: HTTP {response.status_code}")
    
    with open(dest_path, "w") as f:
        f.write(response.text)
    logging.info(f"Successfully saved raw PDB to {dest_path}")
    return dest_path


def process_target(target: dict):
    name = target["name"]
    pdb_id = target["pdb_id"]
    ptm_res = target["ptm_res"]
    ptm_num = target["ptm_num"]

    logging.info("=" * 60)
    logging.info(f"Starting Phase 1 preprocessing for {name} (PDB: {pdb_id})")
    logging.info(f"Target PTM requirement: {ptm_res} at residue {ptm_num}")

    # 1. Download PDB
    raw_path = os.path.join(RAW_DIR, f"{pdb_id}.pdb")
    download_pdb(pdb_id, raw_path)

    # 2. Load into PDBFixer
    logging.info(f"Loading structure into PDBFixer: {raw_path}")
    fixer = pdbfixer.PDBFixer(filename=raw_path)

    # 3. Handle non-standard residues with PTM preservation filter
    fixer.findNonstandardResidues()
    logging.info(f"Initial non-standard residues detected: {len(fixer.nonstandardResidues)}")
    
    preserved_ptms = []
    filtered_nonstandard = []
    for item in fixer.nonstandardResidues:
        res = item[0]
        replacement = item[1]
        if res.name in TARGET_PTMS:
            preserved_ptms.append((res, replacement))
            logging.info(f"[PTM PRESERVATION] Preserving non-standard residue {res.name} (Residue ID: {res.id}, Chain: {res.chain.id}); preventing mutation to {replacement}.")
        else:
            filtered_nonstandard.append(item)
            logging.info(f"Will mutate non-standard residue {res.name} (Residue ID: {res.id}) -> {replacement}")

    fixer.nonstandardResidues = filtered_nonstandard
    fixer.replaceNonstandardResidues()
    logging.info("Completed non-standard residue replacement step.")

    # 4. Remove heterogens (water molecules, crystallization buffers, ions)
    logging.info("Removing heterogens (water molecules, ions, and ligands)...")
    fixer.removeHeterogens(keepWater=False)

    # 5. Add missing heavy atoms and missing loops
    logging.info("Finding missing residues...")
    fixer.findMissingResidues()
    
    logging.info("Finding missing heavy atoms...")
    fixer.findMissingAtoms()
    
    logging.info("Adding missing heavy atoms...")
    fixer.addMissingAtoms()

    # 6. Add missing hydrogens at physiological pH (7.4)
    logging.info("Adding missing hydrogens at physiological pH 7.4...")
    fixer.addMissingHydrogens(pH=7.4)

    # 7. Save cleaned structure
    output_filename = f"{name}_{pdb_id}_clean.pdb"
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    logging.info(f"Saving cleaned structure to {output_path}...")
    with open(output_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    logging.info(f"Successfully finished Phase 1 curation for {name} ({pdb_id}) -> {output_filename}")


def main():
    logging.info("Starting Phase 1 Protein Structure Curation & PTM Preprocessing Pipeline")
    for target in TARGETS:
        try:
            process_target(target)
        except Exception as e:
            logging.error(f"Error processing target {target['name']} ({target['pdb_id']}): {e}", exc_info=True)
            sys.exit(1)

    logging.info("=" * 60)
    logging.info("All 4 oncogenic targets processed successfully in Phase 1.")


if __name__ == "__main__":
    main()
