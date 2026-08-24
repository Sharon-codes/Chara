#!/bin/bash
# Concurrent GPU Runner for Mut_p53 (rep2) and cMYC_MAX (rep1)
# Assigns 16 OpenMP threads each and dynamically time-slices the GPU.

GMX_BIN="/home/sharon/env_md/bin/gmx"

echo "=================================================="
echo " Preparing Mut_p53 / rep2 Clean Production Run"
echo "=================================================="
cd /home/sharon/Desktop/Sharon/data/md_runs/Mut_p53/rep2
mkdir -p broken_trj_archive
# Move any broken/concatenated files to archive to prevent appending errors
mv production.xtc production.cpt production.log production.edr production.part*.xtc production.part*.log production.part*.edr broken_trj_archive/ 2>/dev/null || true

# Start Mut_p53 rep2 in the background using exactly 16 CPU cores
echo "-> Launching Mut_p53/rep2 on GPU (16 threads)..."
$GMX_BIN mdrun -deffnm production -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 16 &
PID_MUT=$!

echo "=================================================="
echo " Preparing cMYC_MAX / rep1 Clean Production Run"
echo "=================================================="
cd /home/sharon/Desktop/Sharon/data/md_runs/cMYC_MAX/rep1
mkdir -p broken_trj_archive2
# Move any broken/canceled runs out of the way
mv production.xtc production.cpt production.log production.edr broken_trj_archive2/ 2>/dev/null || true

# Start cMYC_MAX rep1 in the background using exactly 16 CPU cores
echo "-> Launching cMYC_MAX/rep1 on GPU (16 threads)..."
$GMX_BIN mdrun -deffnm production -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 16 &
PID_CMYC=$!

echo "=================================================="
echo " Both processes launched! PIDs: Mut_p53=$PID_MUT | cMYC=$PID_CMYC"
echo " Waiting for both concurrent processes to hit 500 ns..."
echo "=================================================="

wait $PID_MUT
echo "Mut_p53 / rep2 Finished!"

wait $PID_CMYC
echo "cMYC_MAX / rep1 Finished!"

echo "All 500 ns concurrent simulations have successfully terminated."
