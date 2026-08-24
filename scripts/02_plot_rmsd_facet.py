#!/usr/bin/env python3
"""
02_plot_rmsd_facet.py - Q1 Journal Multi-Panel 2x2 RMSD Facet Plotter
Parses real 6,000 ns dataset across 4 target proteins and 3 independent replicas (500 ns each).
Splits concatenated timelines or loads individual replicates, resets time axes to 0-500 ns,
and renders a standardized 2x2 publication figure formatted for CMPB Q1 standards.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Q1 Journal Typography & Styling (Arial / Times New Roman, Size 12/14)
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Times New Roman', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

BASE_DIR = "/home/sharon/Desktop/Sharon/data/md_runs"
PROTEINS = ["KRAS_G12D", "Mut_p53", "PTPN11", "cMYC_MAX"]
REPLICATES = ["rep1", "rep2", "rep3"]

# Publication palette: base color per target with 3 distinct shades/opacities for Replicas 1, 2, 3
COLOR_SHADES = {
    "KRAS_G12D": ["#08519c", "#3182bd", "#6baed6"],  # Deep, Medium, Light Blue
    "Mut_p53":   ["#a50f15", "#de2d26", "#fb6a4a"],  # Dark, Medium, Light Red
    "PTPN11":    ["#006d2c", "#2ca02c", "#74c476"],  # Dark, Medium, Light Green
    "cMYC_MAX":  ["#54278f", "#756bb1", "#9e9ac8"]   # Dark, Medium, Light Purple
}

def parse_xvg(filepath):
    """Parse GROMACS XVG file ignoring metadata lines."""
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

def split_and_reset_timeline(time_raw, rmsd_raw, chunk_duration=500.0):
    """
    Mathematical Logic for Chunking & Time Resetting:
    If a trajectory is continuous/concatenated, split it into 500 ns segments
    and reset each segment's time axis back to 0-500 ns.
    """
    chunks = []
    if len(time_raw) == 0:
        return chunks
        
    start_idx = 0
    num_frames = len(time_raw)
    
    for i in range(num_frames):
        if (i > 0 and (time_raw[i] - time_raw[start_idx] >= chunk_duration)) or i == num_frames - 1:
            end_idx = i + 1 if i == num_frames - 1 else i
            t_chunk = time_raw[start_idx:end_idx]
            r_chunk = rmsd_raw[start_idx:end_idx]
            
            # Reset time axis to strictly start from 0.0 ns
            t_reset = t_chunk - t_chunk[0]
            chunks.append((t_reset, r_chunk))
            start_idx = i
            
    return chunks

def load_dataset():
    """
    Load data from raw .xvg files for all 4 proteins and 3 replicates.
    Returns nested dictionary: data[protein][rep_index] = (time_0_500, rmsd_angstrom)
    """
    data = {}
    
    for protein in PROTEINS:
        data[protein] = {}
        for rep_idx, rep in enumerate(REPLICATES):
            xvg_path = os.path.join(BASE_DIR, protein, rep, "rmsd.xvg")
            
            if os.path.exists(xvg_path):
                t, r = parse_xvg(xvg_path)
                
                # Check if file contains concatenated data (> 500 ns)
                if len(t) > 0 and t[-1] - t[0] > 550.0:
                    chunks = split_and_reset_timeline(t, r, chunk_duration=500.0)
                    if rep_idx < len(chunks):
                        t_proc, r_proc = chunks[rep_idx]
                    else:
                        t_proc, r_proc = t - t[0], r
                else:
                    # Reset time axis strictly to 0-500 ns grid
                    t_proc = t - t[0] if len(t) > 0 else t
                    r_proc = r
                    
                # Convert nm to Angstroms
                r_angstrom = r_proc * 10.0
                data[protein][rep] = (t_proc, r_angstrom)
            else:
                print(f"Warning: {xvg_path} missing!")
                data[protein][rep] = (np.array([]), np.array([]))
                
    return data

def render_2x2_facet_plot(data, output_pdf="RMSD_Replicas_300dpi.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Determine standardized global Y-axis limit across all 4 subplots
    max_rmsd_global = 0.0
    for protein in PROTEINS:
        for rep in REPLICATES:
            _, r = data[protein][rep]
            if len(r) > 0:
                max_rmsd_global = max(max_rmsd_global, np.max(r))

    # Standardize Y-limit cleanly (e.g. 0 to 8 Å or 0 to 10 Å)
    y_limit = max(8.0, np.ceil(max_rmsd_global * 1.15))

    for idx, protein in enumerate(PROTEINS):
        ax = axes_flat[idx]
        shades = COLOR_SHADES[protein]

        for rep_idx, rep in enumerate(REPLICATES):
            t, r = data[protein][rep]
            if len(t) == 0:
                continue

            label_name = f"Replica {rep_idx + 1}"
            color = shades[rep_idx]

            # Plot line with distinct shade and alpha for high scientific readability
            ax.plot(t, r, color=color, linewidth=1.4, alpha=0.85, label=label_name)

        # Title & Subplot Labels
        ax.set_title(f"{protein} (N=3 Replicates)", fontweight='bold', fontsize=14, pad=10)

        # Axis boundaries: X strictly 0 to 500 ns, Y standardized
        ax.set_xlim(0, 500.0)
        ax.set_ylim(0, y_limit)

        # Despine top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # Light, dashed background grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.legend(frameon=True, loc='upper left', fontsize=9, edgecolor='none', facecolor='#f8f9fa')

    # Common Outer X and Y Labels
    fig.supxlabel("Simulation Time (ns)", fontweight='bold', fontsize=12, y=0.02)
    fig.supylabel("Backbone RMSD (Å)", fontweight='bold', fontsize=12, x=0.02)

    plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.96])

    # Save 300 DPI PDF and PNG preview
    output_png = output_pdf.replace(".pdf", ".png")
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Publication 2x2 facet plot saved strictly to '{output_pdf}' and '{output_png}' at 300 DPI.")

def main():
    print("Parsing raw RMSD XVG data across 4 target proteins & 3 independent 500 ns replicas...")
    dataset = load_dataset()
    render_2x2_facet_plot(dataset, output_pdf="RMSD_Replicas_300dpi.pdf")

if __name__ == "__main__":
    main()
