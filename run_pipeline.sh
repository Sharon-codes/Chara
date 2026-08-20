#!/bin/bash
set -e

# Ensure PATH includes conda environment and local binaries
export PATH="/home/sharon/env_md/bin:/home/sharon/.local/bin:$PATH"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

mkdir -p logs data/processed/images data/cg_structures/images data/cg_topologies data/md_runs

echo "================================================================="
echo " Starting Molecular Dynamics Pipeline Inside ~/Desktop/Sharon/"
echo "================================================================="

# Step 1: Render High-Resolution Atomistic Images
echo "[Step 1/3] Rendering publication-grade atomistic PTM structure images..."
python3 scripts/01_render_images.py

# Step 2: MARTINI 3 Coarse-Graining and CG Image Rendering
echo "[Step 2/3] Performing MARTINI 3 coarse-graining and CG visualization..."
python3 scripts/02_phase2_cg.py

# Step 3: Launch GROMACS MD Setup & GPU Execution in Background
echo "[Step 3/3] Launching GROMACS MD setup & GPU execution in background..."
nohup python3 scripts/03_phase3_gromacs.py > logs/phase3_gromacs.log 2>&1 &
PID=$!

echo "================================================================="
echo " GROMACS GPU MD Pipeline Launched Successfully (PID: $PID)"
echo " Logs: logs/phase3_gromacs.log"
echo " To view real-time progress monitor, run:"
echo "   python3 scripts/monitor.py"
echo "================================================================="
