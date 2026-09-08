# CHANGELOG - CHARA Web Platform Update (v0.3.5)

## Summary of Changes (2026-09-09)

### 1. Scientific Status & Benchmark Presentation
- **Status Updated:** Set to `"12-protein benchmark completed · Monitoring research exploratory"`.
- **Headline Counts:** Updated to 12 benchmark proteins, 36 held-out-run evaluations, 10 measured distances, and 500 target distances per fold. Clarified that each of the 12 proteins is evaluated across 3 run rotations (not 36 independent proteins).
- **Benchmark Metrics:** Displayed verified values across 36 held-out runs:
  - **M6 Pooled-Error Greedy:** Mean reconstruction skill = `0.449692` (~0.450), Mean within-protein worst-run skill = `0.329931` (~0.330).
  - **M7 Mean-Transfer Selection:** Mean reconstruction skill = `0.436346` (~0.436), Mean within-protein worst-run skill = `0.304091` (~0.304).
  - **M8 Worst-Direction Transfer:** Mean reconstruction skill = `0.432742` (~0.433), Mean within-protein worst-run skill = `0.302127` (~0.302).
- **Concise Benchmark Conclusion Added:**
  > *"Sparse distance reconstruction transferred across held-out simulation runs. Pooled-error selection achieved the highest average score among these three methods; worst-direction transfer selection did not demonstrate an advantage over mean-transfer selection."*
- **Bootstrap Uncertainty & Paired Comparison:**
  Documented M8-minus-M7 worst-run skill difference of `-0.001964` [95% paired protein-bootstrap CI: `-0.026101`, `+0.028768`]. Noted that this interval spans zero and does not prove equivalence. Emphasized that variation among frames cannot be used as uncertainty among proteins.
- **Retrospective Centered Diagnostic Correction:**
  Documented that correct diagnostic centering centers each target series and each predicted series by its own temporal mean. Published reviewed retrospective corrected values (M6: `0.375201`, M7: `0.356069`, M8: `0.354314`) explicitly labeled as post-hoc analytical diagnostics, not inference inputs.
- **Verification Disclosure:** Disclosed that saved primary predictions across all 36 runs were verified, but a complete refit hit a near-tied regularization-choice mismatch between numerical solvers.
- **Historical Pilot Retained:** 3-protein pilot (9 evaluations) preserved in a separate, clearly labeled table and dataset mode. Explicitly warned not to combine pilot and 12-protein panel averages.

### 2. Evidence Link Repairs (Zero 404s)
- Resolved previous 404 errors by adjusting `.vercelignore` and `.gitignore`:
  - **12-Protein Report:** `/reports/chara_final_novelty/REPORT.html` (HTTP 200).
  - **12-Protein Evidence Archive:** `/reports/chara_final_novelty/CHARA_FINAL_NOVELTY_EVIDENCE.zip` (42.4 MB, HTTP 200, valid ZIP).
  - **12-Protein Benchmark CSV:** `/data/BENCHMARK_RESULTS_12PROTEIN.csv` (HTTP 200).
  - **3-Protein Pilot Report:** `/reports/chara_external_confirmation/20260908_v1/REPORT.html` (HTTP 200).
  - **3-Protein Pilot Evidence Archive:** `/reports/chara_external_confirmation/20260908_v1/CHARA_INDEPENDENT_BENCHMARK_EVIDENCE.zip` (58.9 MB, HTTP 200, valid ZIP).
  - **3-Protein Pilot Benchmark CSV:** `/data/BENCHMARK_RESULTS.csv` (HTTP 200).

### 3. Interactive Saved-Results Viewer Corrections
- **Dataset Version Selector:** Added toggle between `"12-Protein Benchmark Panel (Latest Executed — 36 Folds)"` and `"3-Protein Pilot (Historical Milestone — 9 Folds)"`.
- **M6 Support:** Added M6 Pooled-Error Greedy method pill with real predictions.
- **Real M7 Predictions:** Reconstructed true M7 predictions across all 12 proteins and 36 folds, replacing previous baseline fallback curves.
- **Independent Pearson Correlation:** Pearson $r$ is now calculated independently from each method's own prediction series vs. $y_{true}$.
- **Constant Predictor Handling:** Constant predictions (e.g. M1 baseline where variance is 0) now explicitly render `"Undefined (constant prediction)"` instead of erroneous `0.0000`.
- **Trajectory CSV Export:** Added `"Export Trajectory CSV"` button dynamically downloading `#`-annotated CSV files containing true time series, true distances, and all active method predictions for the active selection.
- **Clear Aggregation Labels:** Separated target-level KPI labels (Target Skill, Target Recon RMSE, Target Base RMSE, Target Pearson r) from fold-level aggregate metrics and protein-mean metrics.

### 4. Scientific Wording Corrections
- **cMYC Complex:** Corrected wording to:
  > *"Analysis of chain E extracted from the simulated cMYC-containing complex."*
  Documented MARTINI coarse-grained simulation with 349 elastic network restraints plus 10 backbone bonds, noting that the analyzed chain was a subset extracted from the simulated assembly, and the monomer was not simulated alone.
- **Sensor Selection Discipline:** Clarified out-of-sample protocol:
  > *"Select measurements using transfer between development runs; evaluate once on the held-out run."*
  Noted that positive reconstruction skill indicates improvement over the development-mean baseline, rather than proof of an important biological mechanism.

### 5. Current-Research Item
- Added compact card in Case Studies:
  - **Title:** *"Can additional measurements reveal missed structural changes?"*
  - **Status:** `Exploratory investigation · Novelty not established`.
  - **Text:** *"We are testing whether a few additional residue-distance measurements can flag reconstruction failures that ordinary uncertainty scores miss."*
  - Explained that the planned comparison evaluates an equal total measurement budget (e.g. 10 reconstruction + 3 monitoring distances vs. 13 full reconstruction distances).

### 6. Structural Figure Addition
- Added 3D backbone Cα trace with aggregated residue RMSE and 10 sensor chords for `1bxy_A` (Fold 0, R1 held-out):
  - File: `data/structure_1bxy_A_residue_errors.svg` (23.4 KB).
  - Target pair RMSE aggregated onto incident residues (0.031 to 0.111 nm).
  - 10 green dashed chords indicate selected sensor residue pairs.

### 7. Software History & Scope Boundary
- Retained explicit labels distinguishing the earlier survival-analysis prototype (`chara-survival` on PyPI and Hugging Face) from the present MD distance reconstruction research.
- Retained strict scope boundary: models reconstruct selected intramolecular distances in another run of the SAME protein under identical conditions; they do not establish drug binding, cell entry, clinical benefit, future-time prediction, or MD integrator acceleration.
