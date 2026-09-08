# CHARA Website Focused Edits & Change Notes

## Overview
Focused in-place edits to the existing CHARA website repository ([https://chara-frontend.vercel.app/](https://chara-frontend.vercel.app/)). All modifications preserve the existing page layout, section order, colors (olive and parchment palette), fonts (Cinzel, Inter, JetBrains Mono), spacing system, navigation structure, responsive behavior, framework, and Vercel hosting arrangement. No new design system or scaffolding was introduced.

- **Baseline Commit:** `50db73bb97961a804414ed299830bedf9b75c473`
- **Target Files Modified:** `index.html`
- **Source-Controlled Data Files Added:** `data/chara_pilot_evidence.json`, `data/chara_pilot_evidence.js`, `data/BENCHMARK_RESULTS.csv`
- **Build & Verification Status:** Clean static HTML5/Tailwind/Chart.js build; verified on desktop (1280×900) and mobile (390×844) viewports.

---

## 1. Summary of Changed Components

| Component / Section | Previous Content | Updated Pilot Content |
|---|---|---|
| **Header & Meta** | *Chara \| Thermodynamic Graph Survival Inference* | *CHARA \| Sparse Distance Reconstruction from Molecular Dynamics*; accurate meta description; preserved official IIT Mandi logo (`assets/iit-mandi-logo.png`). |
| **Nav Links & Badges** | *Live Demo & Comparison*, unlabelled Hugging Face link | *Explore Saved Results*; relabeled Hugging Face link as *Earlier Survival Prototype*; preserved GitHub repository links. |
| **Hero Section** | *Thermodynamic Graph Laplacian Manifold Alignment for Survival Inference* | *Sparse Distance Reconstruction from Molecular Dynamics*; description: *“CHARA studies whether a small set of residue-pair distances can reconstruct other distances in an independent simulation run of the same protein.”*; added status tag: `External pilot · Validation ongoing`. |
| **Hero Metric Cards** | Clinical survival metrics (4,337 Genes, +0.268 C-Index, Zero Leakage, 58 Biomarkers) | Completed pilot counts: **3 External Proteins**, **9 Held-Out Runs**, **10 Selected Distances**, **500 Targets / Fold**. |
| **Overview Context** | Cross-platform sequencing discrepancies (Illumina vs Affymetrix microarrays) | *The Research Question: Sparse Distance Reconstruction*; biophysical challenge of trajectory sampling & multi-run transfer vs in-sample overfitting. |
| **Four-Stage Pipeline** | Cohort CSV, Graph Laplacian, Heat Diffusion, Survival Projection | 1. Extract residue-pair distances from MD trajectories.<br>2. Select informative measurements using two development runs.<br>3. Fit a distance-reconstruction model on development data.<br>4. Evaluate reconstruction in a third, held-out run. |
| **Package Release Card** | Promotional survival model hub | Relabeled `chara-survival` as earlier prototype (v0.2.9) and linked current MD benchmark analysis code. |
| **Benchmarks Section** | Clinical C-Index and 5-year Brier score drift across TCGA/ICGC | Verified MD reconstruction metrics across 3 ATLAS benchmark proteins. Bar chart comparing M8 vs M7 vs M8 Centered Diagnostic. Observed fold ranges chart without invented error bars. |
| **Benchmark Table** | Cross-platform clinical concordance table | Verified pilot consistency table (1utg_A: M8=0.385691, M7=0.333865, Centered=0.273502; 2cg7_A: M8=0.319264, M7=0.322845, Centered=0.248618; 2j6b_A: M8=0.405957, M7=0.388613, Centered=0.285084). Added limitation note. |
| **Case Studies** | Clinical cancer success stories (PDAC survival, Affymetrix microarrays, precision oncology) | 4 Research-Progress Cards:<br>1. *The Original MD Question* (drug binding vs distance reconstruction).<br>2. *Local cMYC Reconstruction Lead* (centered diagnostic +0.207 & simulation caveats).<br>3. *Three-Protein External Pilot* (ATLAS CHARMM36m triplicates).<br>4. *Next Validation Step: Multi-Protein Scaling* (planned study status). |
| **Demo / Inference Section** | Cohort CSV upload, Grok AI API call, jsPDF clinical report export, risk stratification charts | *Explore Saved MD Results*: Interactive selectors for Protein (`1utg_A`, `2cg7_A`, `2j6b_A`), Held-Out Run (`Fold 1`, `Fold 2`, `Fold 3`), Method (`M8`, `M4`, `M1`, `M7`), and Target Residue Pair. Real trajectory line chart (\(Y_{\text{true}}\) vs \(\hat{Y}\) vs Baseline in nm/ns), target KPIs, and CSV download button. |
| **Archived Deep Links** | Direct route to retired clinical file uploader | Added prominent archived notice for `#inference`, `#clinical`, `#survival` deep links explaining transition to verified MD trajectory exploration. |
| **Team & Footer** | Preserved layout, photos, acknowledgments | Sharon Melhi (Research Intern, CPG Lab, IIT Mandi), Supervisor: Dr. Kharerin Hungyo (PI, Assistant Professor, IIT Mandi), Khushi Mhamane acknowledgment, official IIT Mandi logo preserved. |

---

## 2. Removed & Updated Claims

1. **Removed Clinical Utility & Cancer Survival Claims:**
   - Removed claims regarding patient survival stratification, Kaplan-Meier curves, hazard ratios, early cancer detection, and adjuvant chemotherapy prioritization.
   - Clarified that the original inquiry regarding small-molecule binding to cancer-associated proteins is unanswered by the current distance-reconstruction framework.

2. **Removed Zero Leakage Clinical Marketing:**
   - Replaced clinical "zero leakage" marketing language with rigorous cross-validation protocol details: two development trajectories for greedy transfer selection/tuning and an independent third held-out trajectory for validation.

3. **Removed Proven Graph Superiority / Novelty Claims:**
   - Omitted the uncalibrated +0.233 PCA comparison.
   - Featured direct comparison between M8 (worst-direction transfer) and M7 (average-direction transfer), honestly displaying the case where M7 is slightly higher (2cg7_A: M7 = 0.322845 vs M8 = 0.319264).
   - Added explicit limitation note: *“This three-protein pilot supports further evaluation. Baseline configuration, nested selection/tuning and aggregate uncertainty require correction before claiming method superiority.”*

4. **Retired Frontend Network Requests:**
   - Removed hardcoded Grok AI API key and third-party AI chat completions endpoint.
   - Removed Gradio client network imports.
   - Removed jsPDF report generator dependencies.

---

## 3. Verified Benchmark Consistency Table

All values match source files (`BENCHMARK_RESULTS.csv`, `DIAGNOSTICS_RESULTS.csv`):

| Protein ID | Structural Class | M8 Skill (Worst-Dir) | M7 Skill (Average-Dir) | M8 Centered Diagnostic | Observed Fold Range (M8) | Status |
|---|---|---|---|---|---|---|
| **1utg_A** | all-alpha (70 res) | `0.385691` | `0.333865` | `0.273502` | `[0.279248, 0.570158]` | Completed Pilot |
| **2cg7_A** | all-beta (90 res) | `0.319264` | `0.322845` | `0.248618` | `[0.126464, 0.448078]` | Completed Pilot (M7 +0.0036) |
| **2j6b_A** | alpha+beta (109 res) | `0.405957` | `0.388613` | `0.285084` | `[0.320718, 0.476995]` | Completed Pilot |
| **Macro Mean** | 3 Diverse Classes | `0.370304` | `0.348441` | `0.269068` | `[0.126464, 0.570158]` | Verified Aggregate |

- **Skill Metric Definition:** Zero matches development-mean baseline; higher is better; negative values are worse.
- **Centered Diagnostic:** Retrospective diagnostic centering predictions with test-set means to verify dynamic fluctuation tracking.

---

## 4. Verification & Testing

1. **Selector Operations:** Tested interactive selection across all 3 proteins (`1utg_A`, `2cg7_A`, `2j6b_A`), all 3 held-out folds (`Fold 1`, `Fold 2`, `Fold 3`), all methods (`M8`, `M4`, `M1`, `M7`, `Overlay`), and representative target residue pairs.
2. **Chart Rendering:** Chart.js line and bar charts render cleanly without console errors or layout shifts.
3. **Data Download:** Direct download link for `data/BENCHMARK_RESULTS.csv` verified.
4. **Responsive Views:** Desktop (1280×900) and Mobile (390×844) viewport layouts verified via headless browser screenshots.
5. **No Broken Routes:** Deep links (`#inference`, `#clinical`, `#survival`) smoothly route to the interactive sandbox with the archived explanation banner.

---

## 5. Deployment Instructions (Vercel)

The repository uses Vercel's standard static site hosting serving `index.html` directly from the repository root:

1. **Commit and Push:**
   ```bash
   git add index.html data/chara_pilot_evidence.json data/chara_pilot_evidence.js data/BENCHMARK_RESULTS.csv screenshots/ CHANGE_NOTES.md patch.diff
   git commit -m "feat(website): update CHARA frontend for sparse MD distance reconstruction pilot"
   git push origin main
   ```
2. **Automatic Vercel Deployment:**
   - Because the Vercel project `chara-frontend` is linked to GitHub repository `Sharon-codes/Chara`, pushing to `main` automatically triggers a zero-configuration static deployment.
   - No build command or framework preset change is required.
3. **Manual Deploy (Optional via Vercel CLI):**
   ```bash
   vercel --prod
   ```
