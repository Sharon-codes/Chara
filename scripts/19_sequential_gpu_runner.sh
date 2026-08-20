#!/bin/bash
# Sequential Full-Speed GPU Runner
# Executes Mut_p53 (rep2) first at max 32-thread speed, followed immediately by cMYC_MAX (rep1).

GMX_BIN="/home/sharon/env_md/bin/gmx"

echo "=================================================="
echo " Starting Mut_p53 / rep2 Sequential Run (32 threads)"
echo "=================================================="
cd /home/sharon/Desktop/Sharon/data/md_runs/Mut_p53/rep2

# Continue from last valid checkpoint if available, else start fresh production
if [ -f "production.cpt" ]; then
    echo "-> Continuing Mut_p53/rep2 from production.cpt..."
    $GMX_BIN mdrun -deffnm production -cpi production.cpt -append -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32 || \
    $GMX_BIN mdrun -deffnm production -cpi production.cpt -noappend -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32
else
    echo "-> Launching Mut_p53/rep2 from start..."
    $GMX_BIN mdrun -deffnm production -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32
fi

echo "=================================================="
echo " Mut_p53 / rep2 Complete! Starting cMYC_MAX / rep1"
echo "=================================================="
cd /home/sharon/Desktop/Sharon/data/md_runs/cMYC_MAX/rep1

if [ -f "production.cpt" ]; then
    echo "-> Continuing cMYC_MAX/rep1 from production.cpt..."
    $GMX_BIN mdrun -deffnm production -cpi production.cpt -append -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32 || \
    $GMX_BIN mdrun -deffnm production -cpi production.cpt -noappend -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32
else
    echo "-> Launching cMYC_MAX/rep1 from start..."
    $GMX_BIN mdrun -deffnm production -nb gpu -bonded gpu -pin on -ntmpi 1 -ntomp 32
fi

echo "=================================================="
echo " BOTH SIMULATIONS COMPLETE!"
echo "=================================================="
