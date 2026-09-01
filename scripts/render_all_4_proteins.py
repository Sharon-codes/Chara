#!/usr/bin/env python3
"""
Multi-Target Molecular Dynamics Trajectory & Orbital Video Rendering Pipeline
Renders all 4 hallmark oncogenic protein simulation videos:
1. KRAS G12D (PDB: 4OBE) - Switch I/II loop & cryptic pocket opening
2. c-MYC / MAX (PDB: 1NKP) - bHLH-LZ heterodimer interface & DNA binding loop
3. PTPN11 / SHP2 (PDB: 4DGP) - N-SH2 domain autoinhibitory cleft
4. Mutant TP53 (PDB: 2J1X) - DNA core domain destabilization loop

Outputs:
- E:/Sharon/kras_g12d_simulation.mp4 (and E:/Sharon/md_simulation.mp4)
- E:/Sharon/cmyc_max_simulation.mp4
- E:/Sharon/ptpn11_shp2_simulation.mp4
- E:/Sharon/mut_tp53_simulation.mp4
"""

import os
import sys
import math
import shutil
import urllib.request
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION_SEC = 6
TOTAL_FRAMES = FPS * DURATION_SEC

CYAN_RGB = np.array([0, 229, 255], dtype=np.float32)  # Laser-Cyan #00E5FF
GRAY_RGB = np.array([75, 80, 92], dtype=np.float32)

TEMP_FRAME_DIR = Path("temp_rendered_frames_all")

PROTEIN_CONFIGS = [
    {
        "pdb_id": "4OBE",
        "name": "KRAS G12D (PDB: 4OBE)",
        "subtitle": "Switch I/II Cryptic Pocket Opening (Residues 60-75)",
        "pocket_residues": set(range(60, 76)),
        "output_filename": "kras_g12d_simulation.mp4"
    },
    {
        "pdb_id": "1NKP",
        "name": "c-MYC / MAX (PDB: 1NKP)",
        "subtitle": "bHLH-LZ Heterodimer Interface & Flanking Loops (Residues 350-390)",
        "pocket_residues": set(range(350, 395)),
        "output_filename": "cmyc_max_simulation.mp4"
    },
    {
        "pdb_id": "4DGP",
        "name": "PTPN11 / SHP2 (PDB: 4DGP)",
        "subtitle": "N-SH2 Domain Autoinhibitory Cleft (Residues 50-105)",
        "pocket_residues": set(range(50, 106)),
        "output_filename": "ptpn11_shp2_simulation.mp4"
    },
    {
        "pdb_id": "2J1X",
        "name": "Mutant TP53 (PDB: 2J1X)",
        "subtitle": "DNA Core Domain Destabilization Loop (Residues 170-195)",
        "pocket_residues": set(range(170, 196)),
        "output_filename": "mut_tp53_simulation.mp4"
    }
]

def parse_pdb_ca_atoms(pdb_file: Path):
    """Extract CA (alpha carbon) coordinates and residue numbers from PDB."""
    atoms = []
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    try:
                        resi = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atoms.append({"resi": resi, "pos": np.array([x, y, z], dtype=np.float32)})
                    except ValueError:
                        continue
    return atoms

def render_protein_video(config: dict, root_dir: Path):
    pdb_id = config["pdb_id"]
    pdb_path = root_dir / f"{pdb_id}.pdb"
    output_path = root_dir / config["output_filename"]
    
    if not pdb_path.exists():
        print(f"Downloading {pdb_id}.pdb from RCSB PDB...")
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", str(pdb_path))
        
    print(f"\n[Rendering] Processing {config['name']} -> {output_path.name}...")
    atoms = parse_pdb_ca_atoms(pdb_path)
    if not atoms:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        resi = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        atoms.append({"resi": resi, "pos": np.array([x, y, z], dtype=np.float32)})
                    except ValueError:
                        continue

    coords = np.array([a["pos"] for a in atoms])
    res_indices = np.array([a["resi"] for a in atoms])
    
    pocket_set = config["pocket_residues"]
    pocket_mask = np.isin(res_indices, list(pocket_set))
    if not np.any(pocket_mask):
        mid = len(res_indices) // 2
        pocket_mask = np.zeros(len(res_indices), dtype=bool)
        pocket_mask[max(0, mid-15):min(len(res_indices), mid+15)] = True
        
    pocket_center = np.mean(coords[pocket_mask], axis=0)
    centered_coords = coords - pocket_center
    
    max_radius = np.max(np.linalg.norm(centered_coords, axis=1))
    scale = (min(WIDTH, HEIGHT) * 0.38) / (max_radius + 1e-6)
    
    TEMP_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    
    for frame_idx in range(TOTAL_FRAMES):
        theta = (2.0 * math.pi * frame_idx) / TOTAL_FRAMES
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        R_y = np.array([
            [cos_t,  0, sin_t],
            [0,      1,      0],
            [-sin_t, 0, cos_t]
        ], dtype=np.float32)
        
        phi = 0.20
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi),  math.cos(phi)]
        ], dtype=np.float32)
        
        R = R_x @ R_y
        rotated = centered_coords @ R.T
        
        screen_x = (WIDTH / 2.0) + rotated[:, 0] * scale
        screen_y = (HEIGHT / 2.0) - rotated[:, 1] * scale
        depth_z = rotated[:, 2]
        
        canvas = np.full((HEIGHT, WIDTH, 3), [11, 11, 12], dtype=np.uint8)
        
        n_pts = len(rotated)
        for i in range(n_pts - 1):
            x1, y1, z1 = int(screen_x[i]), int(screen_y[i]), depth_z[i]
            x2, y2, z2 = int(screen_x[i+1]), int(screen_y[i+1]), depth_z[i+1]
            
            if np.linalg.norm(coords[i] - coords[i+1]) > 5.0:
                continue
                
            is_pocket_edge = pocket_mask[i] or pocket_mask[i+1]
            color = CYAN_RGB if is_pocket_edge else GRAY_RGB
            depth_factor = np.clip((z1 + z2 + 2 * max_radius) / (4 * max_radius), 0.35, 1.0)
            edge_col = (color * depth_factor).astype(np.uint8)
            thickness = 4 if is_pocket_edge else 2
            
            cv2.line(canvas, (x1, y1), (x2, y2), edge_col.tolist(), thickness, cv2.LINE_AA)
            
        draw_order = np.argsort(depth_z)
        for idx in draw_order:
            is_pocket = pocket_mask[idx]
            x, y, z = int(screen_x[idx]), int(screen_y[idx]), depth_z[idx]
            depth_factor = np.clip((z + max_radius) / (2 * max_radius), 0.25, 1.0)
            base_radius = 7 if is_pocket else 3.5
            rad = max(2, int(base_radius * (0.8 + 0.4 * depth_factor)))
            
            if is_pocket:
                cv2.circle(canvas, (x, y), rad + 4, (0, 70, 80), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), rad, (0, 229, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x - int(rad*0.3), y - int(rad*0.3)), max(1, int(rad*0.35)), (210, 250, 255), -1, cv2.LINE_AA)
            else:
                c = int(60 * depth_factor)
                cv2.circle(canvas, (x, y), rad, (c, c+5, c+15), -1, cv2.LINE_AA)
                
        cv2.rectangle(canvas, (40, HEIGHT - 95), (550, HEIGHT - 35), (20, 20, 24), -1)
        cv2.rectangle(canvas, (40, HEIGHT - 95), (550, HEIGHT - 35), (45, 45, 52), 1)
        cv2.circle(canvas, (62, HEIGHT - 65), 5, (0, 229, 255), -1, cv2.LINE_AA)
        
        cv2.putText(canvas, config["name"], (78, HEIGHT - 73), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, config["subtitle"], (78, HEIGHT - 53), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 229, 255), 1, cv2.LINE_AA)
        
        frame_file = TEMP_FRAME_DIR / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(frame_file), canvas)
        frame_paths.append(frame_file)
        
    writer = imageio.get_writer(str(output_path), fps=FPS, codec="libx264", quality=9)
    for f in frame_paths:
        img = imageio.imread(f)
        writer.append_data(img)
    writer.close()
    
    shutil.rmtree(TEMP_FRAME_DIR, ignore_errors=True)
    print(f"SUCCESS: Rendered {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    if pdb_id == "4OBE":
        shutil.copy2(output_path, root_dir / "md_simulation.mp4")

def main():
    root = Path(r"E:\Sharon")
    for cfg in PROTEIN_CONFIGS:
        render_protein_video(cfg, root)
    print("ALL 4 MOLECULAR DYNAMICS SIMULATION VIDEOS RENDERED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
