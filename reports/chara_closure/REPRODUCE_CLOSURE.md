# CHARA Bounded Closure Reproduction Instructions

## 1. System Requirements & Environment
- Python >= 3.10 (tested with Python 3.11.15)
- Standard scientific Python stack: `numpy`, `pandas`
- Optional for TPR binary inspection: Linux x86_64 with GROMACS >= 2024 (e.g. `conda-forge gromacs=2026.3`)

## 2. Reproduction Steps
To re-run all audit, diagnostic, and report steps from scratch:

```bash
# Step 1: Run GROMACS inspection attempts across 12 production runs
python tools/chara_closure/01_gromacs_inspection.py

# Step 2: Run corrected development subgroup selection & moment diagnostics
python tools/chara_closure/02_correct_diagnostics.py

# Step 3: Run dataset and literature claim reconciliation
python tools/chara_closure/03_dataset_and_literature.py

# Step 4: Programmatically generate HTML report from verified tables
python tools/chara_closure/04_closure_report_generator.py

# Step 5: Verify all integrity checksums, moment identities, and regression checks
python tools/chara_closure/verify_chara_closure.py
```

## 3. Key Findings & Deliverables
- `RUN_PARAMETER_STATUS.csv`: 12 production runs inspected; all 12 assigned `UNRESOLVED` compiled TPR status due to unavailability of GROMACS 2026.3 dump tools on Windows.
- `SUBGROUP_MEMBERSHIP_COMPLETE.json`: Complete, untruncated target indices for development-defined top-quartile variance across all 4 systems and 3 folds.
- `DIAGNOSTICS_FIXED.csv`: 528 rows across 11 methods, 4 systems, 3 folds, 4 target groups; population-moment residual $|MSE - (Bias^2 + Var_{res})| < 10^{-16} nm^2$.
- `DATASET_ACCESS_VERIFIED.csv`: DESRES Anton BPTI and MoDEL verified ineligible.
- `CHARA_CLOSURE_REPORT.html`: Programmatic self-contained master report.
- `CHARA_CLOSURE_CORE.zip`: Packaged core archive with SHA-256 manifest.
