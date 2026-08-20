#!/bin/bash
# ==============================================================================
# 01_extract_rmsd.sh - Automated GROMACS RMSD Extraction Script
# Extracts Backbone RMSD in nanoseconds from production trajectory
# ==============================================================================

set -e

export PATH="/home/sharon/env_md/bin:$PATH"

# File verification
TPR_FILE="production.tpr"
XTC_FILE="production.xtc"
OUTPUT_XVG="rmsd.xvg"

if [ ! -f "$TPR_FILE" ] || [ ! -f "$XTC_FILE" ]; then
    echo "Error: Required GROMACS trajectory files ($TPR_FILE, $XTC_FILE) not found in current directory."
    exit 1
fi

echo "Extracting Backbone RMSD using gmx rms..."

# In MARTINI CG models, Group 1 'Protein' contains all coarse-grained backbone and sidechain beads.
echo -e "1\n1\n" | gmx rms \
    -s "$TPR_FILE" \
    -f "$XTC_FILE" \
    -o "$OUTPUT_XVG" \
    -tu ns

echo "RMSD extraction complete. Output saved to $OUTPUT_XVG."
