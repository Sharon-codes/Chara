#!/usr/bin/env python3
"""
Step 1: Automated High-Resolution Image Rendering
Renders 1920x1080 ray-traced PNG images for each cleaned atomistic PDB structure,
highlighting the critical PTM residue.
"""

import os
import sys
import subprocess
import logging

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
IMG_DIR = os.path.join(PROCESSED_DIR, "images")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "render_images.log"), mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Detect PyMOL executable
CONDA_PYMOL = "/home/sharon/env_md/bin/pymol"
PYMOL_BIN = CONDA_PYMOL if os.path.exists(CONDA_PYMOL) else "pymol"

TARGETS = [
    {
        "name": "KRAS_G12D",
        "pdb": os.path.join(PROCESSED_DIR, "KRAS_4OBE_clean.pdb"),
        "ptm_res": "181",
        "ptm_name": "SEP",
        "color": "slateblue",
        "out": os.path.join(IMG_DIR, "KRAS_G12D_structure_PTM.png"),
    },
    {
        "name": "cMYC_MAX",
        "pdb": os.path.join(PROCESSED_DIR, "c-MYC_1NKP_clean.pdb"),
        "ptm_res": "58",
        "ptm_name": "TPO",
        "color": "teal",
        "out": os.path.join(IMG_DIR, "cMYC_MAX_structure_PTM.png"),
    },
    {
        "name": "Mut_p53",
        "pdb": os.path.join(PROCESSED_DIR, "p53_2J1X_clean.pdb"),
        "ptm_res": "392",
        "ptm_name": "SEP",
        "color": "orange",
        "out": os.path.join(IMG_DIR, "Mut_p53_structure_PTM.png"),
    },
    {
        "name": "PTPN11",
        "pdb": os.path.join(PROCESSED_DIR, "PTPN11_4DGP_clean.pdb"),
        "ptm_res": "542",
        "ptm_name": "PTR",
        "color": "magenta",
        "out": os.path.join(IMG_DIR, "PTPN11_structure_PTM.png"),
    },
]


def render_target(target: dict):
    logging.info(f"Rendering high-resolution image for {target['name']}...")
    pml_file = f"/tmp/render_{target['name']}.pml"
    
    pml_content = f"""
load {target['pdb']}, prot
bg_color white
show cartoon, prot
color {target['color']}, prot

select ptm, resn {target['ptm_name']} or resi {target['ptm_res']}
show sticks, ptm
show spheres, ptm
set sphere_scale, 0.35, ptm
color atom, ptm

zoom prot, buffer=5
set ray_trace_mode, 1
set ray_trace_quality, 2
set antialias, 2
viewport 1920, 1080
ray 1920, 1080
png {target['out']}
quit
"""
    with open(pml_file, "w") as f:
        f.write(pml_content)

    cmd = [PYMOL_BIN, "-cq", pml_file]
    logging.info(f"Executing command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logging.error(f"PyMOL error for {target['name']}:\n{res.stderr}")
        raise RuntimeError(f"PyMOL rendering failed for {target['name']}")

    if os.path.exists(target['out']):
        logging.info(f"Saved publication-grade PNG: {target['out']}")
    else:
        raise FileNotFoundError(f"Output PNG not found: {target['out']}")


def main():
    logging.info("Starting High-Resolution Atomistic Image Rendering Step")
    for target in TARGETS:
        render_target(target)
    logging.info("All atomistic structure images rendered successfully.")


if __name__ == "__main__":
    main()
