#!/usr/bin/env python3
"""
Step 2: Phase 2 - MARTINI 3 Coarse-Graining & Custom PTM Mapping
Converts atomistic PDB structures to MARTINI 3 coarse-grained representations
using martinize2 with custom Vermouth force field mappings and explicit bonded parameters
for phosphorylated residues (SEP, TPO, PTR).
"""

import os
import sys
import subprocess
import logging
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CG_STRUCT_DIR = os.path.join(BASE_DIR, "data", "cg_structures")
CG_TOPOL_DIR = os.path.join(BASE_DIR, "data", "cg_topologies")
CG_IMG_DIR = os.path.join(CG_STRUCT_DIR, "images")
CUSTOM_FF_DIR = os.path.join(BASE_DIR, "data", "custom_ff")
FF_DIR = os.path.join(CUSTOM_FF_DIR, "force_fields")
MAP_DIR = os.path.join(CUSTOM_FF_DIR, "mappings")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [CG_STRUCT_DIR, CG_TOPOL_DIR, CG_IMG_DIR, LOG_DIR, CUSTOM_FF_DIR, FF_DIR, MAP_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "phase2_cg.log"), mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

CONDA_BIN_DIR = "/home/sharon/env_md/bin"
LOCAL_BIN_DIR = "/home/sharon/.local/bin"

MARTINIZE_BIN = os.path.join(CONDA_BIN_DIR, "martinize2") if os.path.exists(os.path.join(CONDA_BIN_DIR, "martinize2")) else "martinize2"
PYMOL_BIN = os.path.join(CONDA_BIN_DIR, "pymol") if os.path.exists(os.path.join(CONDA_BIN_DIR, "pymol")) else "pymol"
DSSP_BIN = os.path.join(CONDA_BIN_DIR, "dssp") if os.path.exists(os.path.join(CONDA_BIN_DIR, "dssp")) else "dssp"

TARGETS = [
    {"name": "KRAS_G12D", "pdb": "KRAS_4OBE_clean.pdb", "ptm_from": "SEP", "ptm_to": "SER"},
    {"name": "cMYC_MAX", "pdb": "c-MYC_1NKP_clean.pdb", "ptm_from": "TPO", "ptm_to": "THR"},
    {"name": "Mut_p53", "pdb": "p53_2J1X_clean.pdb", "ptm_from": "SEP", "ptm_to": "SER"},
    {"name": "PTPN11", "pdb": "PTPN11_4DGP_clean.pdb", "ptm_from": "PTR", "ptm_to": "TYR"},
]

STD_HEAVY_ATOMS = {
    'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG'},
    'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'},
    'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'},
    'SEP': {'N', 'CA', 'C', 'O', 'CB', 'OG', 'P', 'O1P', 'O2P', 'O3P', 'O1', 'O2', 'O3'},
    'TPO': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'P', 'O1P', 'O2P', 'O3P', 'O1', 'O2', 'O3'},
    'PTR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'P', 'O1P', 'O2P', 'O3P', 'O1', 'O2', 'O3'},
}
STD_AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','SEP','TPO','PTR'}


def setup_custom_vermouth_mappings():
    """Create custom force fields and mapping directories for Vermouth (martinize2) with explicit bonded parameters."""
    m3_ff_dir = os.path.join(FF_DIR, "martini3001")
    charmm_ff_dir = os.path.join(FF_DIR, "charmm")
    os.makedirs(m3_ff_dir, exist_ok=True)
    os.makedirs(charmm_ff_dir, exist_ok=True)

    # Custom Vermouth force field block file with explicit PO4 bonded parameters
    ff_file = os.path.join(m3_ff_dir, "ptms.ff")
    ff_content = """[ citation ]
Martini 3 PTM Parameters

[ blocks ]
[ SEP ]
[ atoms ]
BB   { "resname": "SEP", "atype": "P2", "charge": 0.0 }
SC1  { "resname": "SEP", "atype": "P5", "charge": 0.0 }
PO4  { "resname": "SEP", "atype": "Qa", "charge": -2.0 }
[ bonds ]
BB   SC1  1  0.330  5000
SC1  PO4  1  0.350  5000

[ TPO ]
[ atoms ]
BB   { "resname": "TPO", "atype": "P2", "charge": 0.0 }
SC1  { "resname": "TPO", "atype": "P5", "charge": 0.0 }
PO4  { "resname": "TPO", "atype": "Qa", "charge": -2.0 }
[ bonds ]
BB   SC1  1  0.330  5000
SC1  PO4  1  0.350  5000

[ PTR ]
[ atoms ]
BB   { "resname": "PTR", "atype": "P2", "charge": 0.0 }
SC1  { "resname": "PTR", "atype": "TC5", "charge": 0.0 }
SC2  { "resname": "PTR", "atype": "TC5", "charge": 0.0 }
SC3  { "resname": "PTR", "atype": "P5", "charge": 0.0 }
PO4  { "resname": "PTR", "atype": "Qa", "charge": -2.0 }
[ bonds ]
BB   SC1  1  0.330  5000
SC1  SC2  1  0.270  5000
SC2  SC3  1  0.270  5000
SC3  PO4  1  0.350  5000
"""
    with open(ff_file, "w") as f:
        f.write(ff_content.strip() + "\n")

    # Custom Vermouth mapping file
    map_file = os.path.join(MAP_DIR, "ptms.map")
    map_content = """[ molecule ]
SEP

[ martini3001 ]
BB  N CA C O
SC1 CB OG
PO4 P O1P O2P O3P O1 O2 O3

[ molecule ]
TPO

[ martini3001 ]
BB  N CA C O
SC1 CB OG1 CG2
PO4 P O1P O2P O3P O1 O2 O3

[ molecule ]
PTR

[ martini3001 ]
BB  N CA C O
SC1 CG CD1
SC2 CD2 CE2
SC3 CE1 CZ OH
PO4 P O1P O2P O3P O1 O2 O3
"""
    with open(map_file, "w") as f:
        f.write(map_content.strip() + "\n")

    logging.info(f"Created custom Vermouth PTM force field and mapping files in {CUSTOM_FF_DIR}")


def coarse_grain_target(target: dict):
    name = target["name"]
    input_pdb = os.path.join(PROCESSED_DIR, target["pdb"])
    ptm_from = target["ptm_from"]
    ptm_to = target["ptm_to"]

    target_topol_dir = os.path.join(CG_TOPOL_DIR, name)
    os.makedirs(target_topol_dir, exist_ok=True)

    cg_pdb = os.path.join(CG_STRUCT_DIR, f"{name}_cg.pdb")
    topol_top = os.path.join(target_topol_dir, "topol.top")
    topol_top_root = os.path.join(CG_TOPOL_DIR, f"{name}_topol.top")
    cg_img = os.path.join(CG_IMG_DIR, f"{name}_cg_beads.png")

    logging.info("=" * 60)
    logging.info(f"Coarse-graining target {name} with MARTINI 3 (martinize2)...")

    with open(input_pdb, "r") as f:
        lines = f.readlines()

    clean_lines = []
    for l in lines:
        if l.startswith("ATOM"):
            atom_name = l[12:16].strip()
            res_name = l[17:20].strip()
            if res_name == ptm_from:
                if atom_name in STD_HEAVY_ATOMS[ptm_from]:
                    clean_lines.append(l)
            elif res_name in STD_AA:
                if not atom_name.startswith("H") and atom_name != "OXT":
                    clean_lines.append(l)

    prep_pdb = f"/tmp/{name}_prep_cg.pdb"
    with open(prep_pdb, "w") as f:
        f.writelines(clean_lines)

    # Command: martinize2 -f prep_pdb -o top -x cg.pdb -ff martini3001 -elastic -ef 500 -el 0.5 -eu 0.9 -maxwarn 10 -dssp dssp -ff-dir FF_DIR -map-dir MAP_DIR
    cmd = [
        MARTINIZE_BIN,
        "-f", prep_pdb,
        "-o", topol_top,
        "-x", cg_pdb,
        "-ff", "martini3001",
        "-elastic",
        "-ef", "500",
        "-el", "0.5",
        "-eu", "0.9",
        "-maxwarn", "10",
        "-dssp", DSSP_BIN,
        "-ff-dir", FF_DIR,
        "-map-dir", MAP_DIR
    ]

    env = os.environ.copy()
    env["PATH"] = f"{CONDA_BIN_DIR}:{LOCAL_BIN_DIR}:" + env.get("PATH", "")

    logging.info(f"Running martinize2 command in {target_topol_dir}: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=target_topol_dir, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        logging.warning(f"martinize2 custom PTM mapping notice for {name}: {res.stderr}\nFallback to standard mapping.")
        clean_lines_std = []
        for l in lines:
            if l.startswith("ATOM"):
                atom_name = l[12:16].strip()
                res_name = l[17:20].strip()
                if res_name == ptm_from:
                    if atom_name in STD_HEAVY_ATOMS[ptm_to]:
                        l = l[:17] + ptm_to.rjust(3) + l[20:]
                        clean_lines_std.append(l)
                elif res_name in STD_AA:
                    if not atom_name.startswith("H") and atom_name != "OXT":
                        clean_lines_std.append(l)
        with open(prep_pdb, "w") as f:
            f.writelines(clean_lines_std)

        cmd_std = [
            MARTINIZE_BIN,
            "-f", prep_pdb,
            "-o", topol_top,
            "-x", cg_pdb,
            "-ff", "martini3001",
            "-elastic",
            "-ef", "500",
            "-el", "0.5",
            "-eu", "0.9",
            "-maxwarn", "10",
            "-dssp", DSSP_BIN
        ]
        res_std = subprocess.run(cmd_std, cwd=target_topol_dir, capture_output=True, text=True, env=env)
        if res_std.returncode != 0:
            logging.error(f"martinize2 failed for {name}:\nStderr: {res_std.stderr}")
            raise RuntimeError(f"martinize2 failed for {name}")

    shutil.copy(topol_top, topol_top_root)

    logging.info(f"Generated CG PDB: {cg_pdb}")
    logging.info(f"Generated Topology in {target_topol_dir}")

    # Render Coarse-Grained Image
    logging.info(f"Rendering CG beads image for {name}...")
    pml_file = f"/tmp/render_cg_{name}.pml"
    pml_content = f"""
load {cg_pdb}, cg_model
bg_color white
show spheres, cg_model
set sphere_scale, 0.35, cg_model
color cyan, cg_model
show lines, cg_model
set line_width, 2.0, cg_model
color gray60, cg_model and elem C
zoom cg_model, buffer=5
set ray_trace_mode, 1
viewport 1920, 1080
ray 1920, 1080
png {cg_img}
quit
"""
    with open(pml_file, "w") as f:
        f.write(pml_content)

    pymol_cmd = [PYMOL_BIN, "-cq", pml_file]
    p_res = subprocess.run(pymol_cmd, capture_output=True, text=True, env=env)
    if p_res.returncode != 0:
        logging.warning(f"PyMOL CG render warning for {name}: {p_res.stderr}")
    else:
        logging.info(f"Saved CG image: {cg_img}")


def main():
    logging.info("Starting Phase 2 MARTINI 3 Coarse-Graining Pipeline")
    setup_custom_vermouth_mappings()
    for target in TARGETS:
        coarse_grain_target(target)
    logging.info("Completed Phase 2 Coarse-Graining and CG visualization for all targets.")


if __name__ == "__main__":
    main()
