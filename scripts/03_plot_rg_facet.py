#!/usr/bin/env python3
"""
03_plot_rg_facet.py - Q1 Journal Multi-Panel 2x2 Radius of Gyration (Rg) Facet Plotter
Parses raw GROMACS gyrate.xvg files across 4 target proteins and 3 independent replicas (500 ns each).
Extracts Column 0 (Time) and Column 1 (Total Rg), ignoring Rx, Ry, Rz axes.
Maintains exact stylistic consistency with RMSD plot (palette, typography, despinning).
Outputs publication-grade figure strictly as 'Rg_Replicas_300dpi.pdf' at 300 DPI.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Q1 Journal Typography & Styling (Arial / Times New Roman, Size 12/14)
# Exact stylistic match with RMSD facet plot
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

# Palette matching the RMSD plot: base color per target with 3 distinct shades for Replicas 1, 2, 3
COLOR_SHADES = {
    "KRAS_G12D": ["#08519c", "#3182bd", "#6baed6"],  # Deep, Medium, Light Blue
    "Mut_p53":   ["#a50f15", "#de2d26", "#fb6a4a"],  # Dark, Medium, Light Red
    "PTPN11":    ["#006d2c", "#2ca02c", "#74c476"],  # Dark, Medium, Light Green
    "cMYC_MAX":  ["#54278f", "#756bb1", "#9e9ac8"]   # Dark, Medium, Light Purple
}

def parse_gyrate_xvg(filepath):
    """
    CRITICAL: Parses gmx gyrate output file.
    Extracts Column 0 (Time) and Column 1 (Total Rg in nm).
    Ignores metadata, comments, and Rx, Ry, Rz axes columns (Columns 2, 3, 4).
    Converts Time to nanoseconds (ns) if reported in picoseconds (ps).
    """
    time_list = []
    rg_total_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '@')):
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Column 0: Time | Column 1: Total Rg (nm)
                t_val = float(parts[0])
                rg_total = float(parts[1])
                time_list.append(t_val)
                rg_total_list.append(rg_total)
                
    t_arr = np.array(time_list)
    r_arr = np.array(rg_total_list)
    
    # Auto-convert time to nanoseconds if in picoseconds (ps > 1000)
    if len(t_arr) > 0 and np.max(t_arr) > 1000.0:
        t_arr = t_arr / 1000.0
        
    return t_arr, r_arr

def split_and_reset_timeline(time_raw, rg_raw, chunk_duration=500.0):
    """
    Mathematical Logic for Chunking & Time Resetting:
    If a trajectory is concatenated, split it into 500 ns segments
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
            r_chunk = rg_raw[start_idx:end_idx]
            
            # Reset time axis strictly to 0.0 ns
            t_reset = t_chunk - t_chunk[0]
            chunks.append((t_reset, r_chunk))
            start_idx = i
            
    return chunks

def load_rg_dataset():
    """
    Load total Rg data from raw gyrate.xvg files for all 4 proteins and 3 replicates.
    Returns nested dictionary: data[protein][rep_index] = (time_0_500, total_rg_nm)
    """
    data = {}
    
    for protein in PROTEINS:
        data[protein] = {}
        for rep_idx, rep in enumerate(REPLICATES):
            xvg_path = os.path.join(BASE_DIR, protein, rep, "gyrate.xvg")
            
            if os.path.exists(xvg_path):
                t, r = parse_gyrate_xvg(xvg_path)
                
                # Check if file contains concatenated data (> 550 ns)
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
                    
                data[protein][rep] = (t_proc, r_proc)
            else:
                print(f"Warning: {xvg_path} missing!")
                data[protein][rep] = (np.array([]), np.array([]))
                
    return data

def render_2x2_rg_facet_plot(data, output_pdf="Rg_Replicas_300dpi.pdf"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=300, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Dynamically calculate global min and max Total Rg across all 4 systems for Y-axis standardization
    global_min_rg = 1e9
    global_max_rg = -1e9
    
    for protein in PROTEINS:
        for rep in REPLICATES:
            _, r = data[protein][rep]
            if len(r) > 0:
                global_min_rg = min(global_min_rg, np.min(r))
                global_max_rg = max(global_max_rg, np.max(r))

    # Add a 10% buffer to standardized Y-limits for clean presentation
    rg_range = global_max_rg - global_min_rg
    y_min = max(0.0, global_min_rg - 0.10 * rg_range)
    y_max = global_max_rg + 0.10 * rg_range

    for idx, protein in enumerate(PROTEINS):
        ax = axes_flat[idx]
        shades = COLOR_SHADES[protein]

        for rep_idx, rep in enumerate(REPLICATES):
            t, r = data[protein][rep]
            if len(t) == 0:
                continue

            label_name = f"Replica {rep_idx + 1}"
            color = shades[rep_idx]

            # Plot line with distinct shade and high visibility
            ax.plot(t, r, color=color, linewidth=1.4, alpha=0.85, label=label_name)

        # Title & Subplot Header
        ax.set_title(f"{protein} (N=3 Replicates)", fontweight='bold', fontsize=14, pad=10)

        # Axis boundaries: X strictly 0 to 500 ns, Y dynamically standardized across all systems
        ax.set_xlim(0, 500.0)
        ax.set_ylim(y_min, y_max)

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
    fig.supylabel("Radius of Gyration, $R_g$ (nm)", fontweight='bold', fontsize=12, x=0.02)

    plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.96])

    # Save 300 DPI PDF and PNG preview
    output_png = output_pdf.replace(".pdf", ".png")
    plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Publication 2x2 Rg facet plot saved strictly to '{output_pdf}' and '{output_png}' at 300 DPI.")

def main():
    print("Parsing raw Total Rg (Column 1) gyrate.xvg data across 4 target proteins & 3 independent 500 ns replicas...")
    dataset = load_rg_dataset()
    render_2x2_rg_facet_plot(dataset, output_pdf="Rg_Replicas_300dpi.pdf")

if __name__ == "__main__":
    main()
