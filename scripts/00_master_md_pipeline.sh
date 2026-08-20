#!/bin/bash
# ==============================================================================
# MONOLITHIC SEQUENTIAL MASTER MD PIPELINE (00_master_md_pipeline.sh)
# Multimeric Dimer-Specific PBC Centering, Biophysical Extraction, & Validation
# ==============================================================================

set -e  # Exit on error

GMX_BIN="/home/sharon/env_md/bin/gmx"
BASE_DIR="/home/sharon/Desktop/Sharon/data/md_runs"
SCRIPTS_DIR="/home/sharon/Desktop/Sharon/scripts"
PROTEINS=("KRAS_G12D" "Mut_p53" "PTPN11" "cMYC_MAX")
REPLICATES=("rep1" "rep2" "rep3")

echo "=============================================================================="
echo " STARTING MONOLITHIC SEQUENTIAL EXTRACTION & DIMER PBC CENTERING PIPELINE"
echo "=============================================================================="

for protein in "${PROTEINS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------------"
    echo " >>> TARGET PROTEIN: ${protein}"
    echo "------------------------------------------------------------------------------"
    
    for rep in "${REPLICATES[@]}"; do
        REP_DIR="${BASE_DIR}/${protein}/${rep}"
        
        if [ ! -d "${REP_DIR}" ]; then
            echo "[!] Warning: Directory ${REP_DIR} does not exist. Skipping..."
            continue
        fi
        
        echo "--> Processing Replicate: ${protein} / ${rep} ..."
        cd "${REP_DIR}"
        
        if [ ! -f "production.tpr" ] || [ ! -f "production.xtc" ]; then
            echo "    [!] Missing production.tpr or production.xtc in ${REP_DIR}. Skipping..."
            continue
        fi
        
        # 1. PBC Correction Logic (Task 1: Multimeric Dimer vs Monomer)
        if [ "${protein}" == "cMYC_MAX" ]; then
            echo "    [1/5] Executing Two-Step Dimer PBC Centering (nojump -> mol -center)..."
            
            # Step A: Generate Centered Reference Structure if eq.gro exists
            if [ -f "eq.gro" ]; then
                printf "1\n1\n" | "${GMX_BIN}" trjconv -s production.tpr -f eq.gro -o ref_centered.gro -pbc mol -center 2>/dev/null || true
            fi
            
            # Step B: Remove jumps across periodic boundaries
            printf "0\n" | "${GMX_BIN}" trjconv -s production.tpr -f production.xtc -o production_nojump.xtc -pbc nojump 2>/dev/null
            
            # Step C: Center the dimer complex
            printf "1\n1\n" | "${GMX_BIN}" trjconv -s production.tpr -f production_nojump.xtc -o production_centered.xtc -center -pbc mol 2>/dev/null || \
            printf "1\n0\n" | "${GMX_BIN}" trjconv -s production.tpr -f production_nojump.xtc -o production_centered.xtc -center -pbc mol 2>/dev/null
        else
            echo "    [1/5] Performing Standard PBC Centering (gmx trjconv)..."
            printf "1\n1\n" | "${GMX_BIN}" trjconv -s production.tpr -f production.xtc -o production_centered.xtc -pbc mol -center 2>/dev/null || \
            printf "1\n0\n" | "${GMX_BIN}" trjconv -s production.tpr -f production.xtc -o production_centered.xtc -pbc mol -center 2>/dev/null
        fi
        
        INPUT_TRAJ="production_centered.xtc"
        if [ ! -f "production_centered.xtc" ]; then
            INPUT_TRAJ="production.xtc"
        fi
        
        # Select Reference Structure (Task 2: Assembly-Corrected Reference)
        REF_STRUCT="production.tpr"
        if [ -f "ref_centered.gro" ]; then
            REF_STRUCT="ref_centered.gro"
        elif [ -f "eq.gro" ]; then
            REF_STRUCT="eq.gro"
        fi
        
        # 2. Extract RMSD against assemble-corrected reference
        echo "    [2/5] Extracting RMSD against centered reference (${REF_STRUCT})..."
        printf "1\n1\n" | "${GMX_BIN}" rms -s "${REF_STRUCT}" -f "${INPUT_TRAJ}" -o rmsd.xvg -tu ns 2>/dev/null
        
        # 3. Extract Radius of Gyration (Rg)
        echo "    [3/5] Extracting Radius of Gyration (gmx gyrate)..."
        printf "1\n" | "${GMX_BIN}" gyrate -s "${REF_STRUCT}" -f "${INPUT_TRAJ}" -o gyrate.xvg 2>/dev/null
        
        # 4. Extract SASA (Targeting Protein)
        echo "    [4/5] Extracting SASA (gmx sasa)..."
        printf "1\n" | "${GMX_BIN}" sasa -s "${REF_STRUCT}" -f "${INPUT_TRAJ}" -o sasa.xvg -tu ns 2>/dev/null
        
        # 5. Extract RMSF
        echo "    [5/5] Extracting RMSF (gmx rmsf)..."
        printf "1\n" | "${GMX_BIN}" rmsf -s "${REF_STRUCT}" -f "${INPUT_TRAJ}" -o rmsf.xvg -res 2>/dev/null
        
        echo "    [✓] Successfully processed ${protein} / ${rep}"
    done
done

echo ""
echo "==========================================================================================="
echo " GENERATING THERMODYNAMIC CONTACT MATRICES (C_ij and sigma2_ij)"
echo "==========================================================================================="
/home/sharon/env_md/bin/python3 "${SCRIPTS_DIR}/17_generate_contact_matrices.py"

echo ""
echo "==========================================================================================="
echo " [✓] MULTIMERIC DIMER PBC PIPELINE EXTRACTION COMPLETED"
echo "==========================================================================================="
