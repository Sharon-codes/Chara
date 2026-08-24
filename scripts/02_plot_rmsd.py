#!/usr/bin/env python3
"""
02_plot_rmsd.py - Full 6,000 ns Ensemble RMSD Publication Visualization
Reads all 12 replicate trajectories (4 target proteins x 3 replicates x 500 ns = 6,000 ns)
Outputs publication-ready 300 DPI vector PDF (RMSD_plot_300dpi.pdf).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Q1 Journal Typography & Styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

BASE_DIR = "/home/sharon/Desktop/Sharon/data/md_runs"
PROTEINS = ["KRAS_G12D", "Mut_p53", "PTPN11", "cMYC_MAX"]
REPLICATES = ["rep1", "rep2", "rep3"]

# Distinct palette for Q1 journal visualization
COLORS = {
    "KRAS_G12D": "#1f77b4",  # Deep Blue
    "Mut_p53": "#d62728",    # Crimson Red
    "PTPN11": "#2ca02c",     # Forest Green
    "cMYC_MAX": "#9467bd"    # Royal Purple
}

def parse_xvg(filepath):
    """Parse GROMACS XVG file ignoring header and comment lines."""
    time_list = []
    rmsd_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                time_list.append(float(parts[0]))
                rmsd_list.append(float(parts[1]))
                
    return np.array(time_list), np.array(rmsd_list)

def main():
    cumulative_time = []
    cumulative_rmsd = []
    target_segments = {}
    
    current_time_offset = 0.0
    
    print("Loading all 12 replicate trajectories across 6,000 ns dataset...")
    
    for protein in PROTEINS:
        protein_times = []
        protein_rmsds = []
        
        for rep in REPLICATES:
            xvg_path = os.path.join(BASE_DIR, protein, rep, "rmsd.xvg")
            if not os.path.exists(xvg_path):
                print(f"Warning: {xvg_path} not found. Skipping.")
                continue
            
            t, r = parse_xvg(xvg_path)
            if len(t) == 0:
                continue
                
            # Shift time to form continuous 0 to 6000 ns timeline
            dt = t[1] - t[0] if len(t) > 1 else 0.5
            t_shifted = t - t[0] + current_time_offset
            
            # Convert nm to Angstroms
            r_angstrom = r * 10.0
            
            cumulative_time.extend(t_shifted)
            cumulative_rmsd.extend(r_angstrom)
            
            protein_times.extend(t_shifted)
            protein_rmsds.extend(r_angstrom)
            
            current_time_offset = t_shifted[-1] + dt
            
        target_segments[protein] = (np.array(protein_times), np.array(protein_rmsds))
        print(f"Loaded {protein}: {len(protein_times)} frames up to {current_time_offset:.1f} ns")

    cum_t = np.array(cumulative_time)
    cum_r = np.array(cumulative_rmsd)

    print(f"Total dataset span: {cum_t[0]:.1f} ns to {cum_t[-1]:.1f} ns ({len(cum_t)} frames)")

    # --------------------------------------------------------------------------
    # Render Publication Figure
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Plot continuous trajectory colored by protein target
    for protein in PROTEINS:
        t_seg, r_seg = target_segments[protein]
        ax.plot(t_seg, r_seg, color=COLORS[protein], linewidth=1.2, label=f"{protein} (N=3, 1.5 μs)")

    # Add vertical target boundary markers
    boundary_times = [1500.0, 3000.0, 4500.0]
    for b_time in boundary_times:
        ax.axvline(x=b_time, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

    # Global continuous baseline line (black, thin behind)
    ax.plot(cum_t, cum_r, color='black', linewidth=0.5, alpha=0.3, zorder=1)

    # Styling and Labels
    ax.set_xlabel("Cumulative Simulation Time (ns)", fontweight='bold', fontsize=11)
    ax.set_ylabel("Backbone RMSD (Å)", fontweight='bold', fontsize=11)
    ax.set_title("MARTINI 3 Coarse-Grained MD Trajectory Ensemble (Full 6,000 ns Dataset)", fontweight='bold', pad=14)

    ax.set_xlim(0, 6000.0)
    ax.set_ylim(0, max(np.max(cum_r) * 1.1, 10.0))

    # Despine top and right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.grid(True, linestyle=':', alpha=0.4, color='gray')
    ax.legend(frameon=True, loc='upper right', fontsize=9, edgecolor='none', facecolor='#f8f9fa')

    plt.tight_layout()
    output_pdf = "RMSD_plot_300dpi.pdf"
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    
    # Also save PNG for instant preview
    output_png = "RMSD_plot_300dpi.png"
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"\nSUCCESS: Full 6,000 ns dataset plot rendered to '{output_pdf}' and '{output_png}' at 300 DPI.")

if __name__ == "__main__":
    main()
