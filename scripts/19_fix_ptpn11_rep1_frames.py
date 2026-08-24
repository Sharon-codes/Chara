#!/usr/bin/env python3
"""
Fix PTPN11 rep1 XVG Frame Discrepancy (996 -> 1001 frames)
Linear interpolates the 5 missing frames in XVG files across the 0.0 to 500.0 ns timeline.
"""

import os
import numpy as np

REP1_DIR = "/home/sharon/Desktop/Sharon/data/md_runs/PTPN11/rep1"
XVG_FILES = ["rmsd.xvg", "gyrate.xvg", "sasa.xvg", "rmsf.xvg"]

def fix_xvg(filename):
    path = os.path.join(REP1_DIR, filename)
    if not os.path.exists(path):
        return
    
    header = []
    times, vals = [], []
    
    with open(path, "r") as f:
        for line in f:
            if line.startswith(("@", "#")):
                header.append(line)
            else:
                parts = line.strip().split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    vals.append(float(parts[1]))
                    
    times = np.array(times)
    vals = np.array(vals)
    
    # Target grid: exact 1001 points from min time to max time (or 0 to 500 ns)
    if filename == "rmsf.xvg":
        # RMSF is per-residue, length is fixed by residue count
        return
    
    target_times = np.linspace(0.0, 500.0, 1001)
    interp_vals = np.interp(target_times, times, vals)
    
    with open(path, "w") as f:
        for h in header:
            f.write(h)
        for t, v in zip(target_times, interp_vals):
            f.write(f"{t:10.4f}  {v:10.4f}\n")
            
    print(f"[✓] Fixed {filename}: Resampled from {len(vals)} to {len(interp_vals)} frames.")

def main():
    for f in XVG_FILES:
        fix_xvg(f)

if __name__ == "__main__":
    main()
